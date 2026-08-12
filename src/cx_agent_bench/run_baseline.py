"""Run the baseline agent (or the dummy) over the public task set.

    python -m cx_agent_bench.run_baseline --agent qwen                  # all 50 tasks
    python -m cx_agent_bench.run_baseline --agent dummy-null
    python -m cx_agent_bench.run_baseline --agent qwen --tasks T1-E1 T4-H2
    python -m cx_agent_bench.run_baseline --agent qwen --limit 3

Execution only: traces are written to --out; score them afterwards with
    python -m cx_agent_bench.scoring <trace_dir> --out scores.csv
"""

import argparse
import csv
import json
import time
from pathlib import Path

from .agents import DummyAgent, OllamaAgent
from .harness import STEP_CAP, TIMEOUT_S, build_manifest, run_task
from .tasks_public import REPO_ROOT, load_public_tasks


def write_logs_csv(out_dir):
    """Flatten every trace in out_dir into logs.csv: one row per step, in run order.

    A complete tabular rendering of the JSONL records, nothing more — no metrics
    and no gold fields, so the execution/scoring split (R9) stays intact. args,
    output and raw_response are compact JSON, untruncated: this file carries the
    full log of every step.
    """
    rows = []
    for trace in sorted(Path(out_dir).glob("*.jsonl")):
        records = [json.loads(line) for line in
                   trace.read_text(encoding="utf-8").splitlines()]
        start = next(r for r in records if r["type"] == "run_start")
        end = next((r for r in records if r["type"] == "run_end"), {})
        final = end.get("final_answer") or {}
        for s in (r for r in records if r["type"] == "step"):
            rows.append({
                "task_id": start["task_id"],
                "terminal_state": end.get("terminal_state", ""),
                "idx": s["idx"],
                "tool": s["tool"] or "",
                "args": (json.dumps(s["args"], ensure_ascii=False)
                         if s["args"] is not None else ""),
                "handle": s["handle"] or "",
                "status": s["status"],
                "error_kind": s.get("error_kind") or "",
                "protocol_violation": s["protocol_violation"],
                "tokens_in": s["tokens_in"],
                "tokens_out": s["tokens_out"],
                "latency_s": s["latency_s"],
                "output": json.dumps(s["output"], ensure_ascii=False, default=str),
                "answer": final.get("text", "") if s["tool"] == "answer" else "",
                "raw_response": json.dumps(s["raw_response"], ensure_ascii=False,
                                           default=str),
            })
    path = Path(out_dir) / "logs.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


# Gold-path steps across the full 50-task set, from benchmark_outputs/tasks_gold.csv
# (sum of gold_num_steps; answer steps are never part of a gold path). Hard-coded
# like STEP_CAP so the execution side never reads the gold file (R3, R9).
GOLD_STEPS_TOTAL = 184


def write_log_report(out_dir, endpoint, model, n_tasks, total_s):
    """Aggregate the run's traces into a human-readable log_report.txt.

    Counts only — no gold fields beyond the offline GOLD_STEPS_TOTAL constant, so
    the execution/scoring split (R9) stays intact. Percentages are of the steps
    that carry analysis, i.e. answer steps are excluded, matching how gold paths
    are counted.
    """
    steps, ends = [], []
    for trace in sorted(Path(out_dir).glob("*.jsonl")):
        for line in trace.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record["type"] == "step":
                steps.append(record)
            elif record["type"] == "run_end":
                ends.append(record)

    work = [s for s in steps if s["tool"] != "answer"]
    errors = [s for s in work if s["status"] == "error"]
    no_call = [s for s in work if s["tool"] is None]

    def pct(n):
        return f"{100 * n / len(work):.1f}%" if work else "n/a"

    def tasks_in(state):
        ids = sorted(e["run_id"].split("__")[1] for e in ends
                     if e["terminal_state"] == state)
        return ", ".join(ids) if ids else "none"

    by_kind = {}
    for s in errors:
        kind = s.get("error_kind") or ("no_tool_call" if s["tool"] is None
                                       else "tool_error_result")
        n, v = by_kind.get(kind, (0, 0))
        by_kind[kind] = (n + 1, v + bool(s.get("protocol_violation")))

    hours, rest = divmod(int(total_s), 3600)
    lines = [
        f"Endpoint: {endpoint}",
        f"Model: {model}",
        f"Number of tasks: {n_tasks}",
        f"Total time: {hours}h {rest // 60}m {rest % 60}s",
        "",
        f"Total steps (excluding answer steps): {len(work)}",
        f"Error steps: {len(errors)} ({pct(len(errors))} of total steps)",
        f"No tool call steps: {len(no_call)} ({pct(len(no_call))} of total steps)",
        f"Gold steps: {GOLD_STEPS_TOTAL} (fixed total over the 50-task set)",
        "",
        "Terminated task_ids:",
        f"At step cap: {tasks_in('step_cap')}",
        f"At timeout: {tasks_in('timeout')}",
        "At loop (after 5 consecutive identical replies): "
        f"{tasks_in('stalled')}",
        "",
        "Error analysis:",
        f"{'error_kind':<20} {'steps':>5}  protocol_violation",
    ]
    for kind, (n, v) in sorted(by_kind.items(), key=lambda kv: -kv[1][0]):
        flag = "yes" if v == n else ("no" if v == 0 else f"mixed ({v}/{n})")
        lines.append(f"{kind:<20} {n:>5}  {flag}")
    if not by_kind:
        lines.append("(no error steps)")

    path = Path(out_dir) / "log_report.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def make_agent(name, seed):
    if name == "qwen":
        return OllamaAgent(model="qwen2.5:7b-instruct", temperature=0.0, seed=seed)
    if name == "dummy-null":
        return DummyAgent(mode="null", seed=seed)
    if name == "dummy-random":
        return DummyAgent(mode="random", seed=seed)
    raise SystemExit(f"unknown agent '{name}'")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", default="qwen",
                        choices=["qwen", "dummy-null", "dummy-random"])
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="task ids to run (default: all)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--step-cap", type=int, default=STEP_CAP)
    parser.add_argument("--timeout", type=int, default=TIMEOUT_S)
    parser.add_argument("--out", default=None,
                        help="trace directory (default: traces/baseline/<agent>_<ts>)")
    args = parser.parse_args()

    agent = make_agent(args.agent, args.seed)
    tasks = load_public_tasks()
    if args.tasks:
        wanted = set(args.tasks)
        tasks = [t for t in tasks if t["task_id"] in wanted]
        missing = wanted - {t["task_id"] for t in tasks}
        if missing:
            raise SystemExit(f"unknown task ids: {sorted(missing)}")
    if args.limit:
        tasks = tasks[:args.limit]

    out_dir = Path(args.out) if args.out else (
        REPO_ROOT / "traces" / "baseline"
        / f"{agent.agent_id}_{time.strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(agent)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                           encoding="utf-8")

    print(f"agent={agent.agent_id}  tasks={len(tasks)}  step_cap={args.step_cap}  "
          f"timeout={args.timeout}s\ntraces -> {out_dir}")

    print(f"loading model ({agent.model}) ...", flush=True)
    t_load = time.perf_counter()
    loaded = agent.warmup()
    print(f"{'model ready' if loaded else 'no model to load'} "
          f"({time.perf_counter() - t_load:.0f}s) - starting {tasks[0]['task_id']}",
          flush=True)

    durations = []
    for i, task in enumerate(tasks, 1):
        t0 = time.perf_counter()
        trace = run_task(agent, task, out_dir, manifest=manifest,
                         step_cap=args.step_cap, timeout_s=args.timeout)
        durations.append(time.perf_counter() - t0)
        last = json.loads(Path(trace).read_text(encoding="utf-8").splitlines()[-1])

        # ETA from the mean of completed tasks; drop the first one once there is a
        # better sample, since it carries the one-off model-load cost.
        sample = durations[1:] if len(durations) > 3 else durations
        remaining = (len(tasks) - i) * (sum(sample) / len(sample))
        eta = (f"eta ~{remaining / 60:.0f}m" if remaining >= 90 else
               f"eta ~{remaining:.0f}s") if i < len(tasks) else "done"
        print(f"[{i:>2}/{len(tasks)}] {task['task_id']:<7} "
              f"{last.get('terminal_state', '?'):<9} "
              f"calls={last.get('n_tool_calls', '?'):<3} "
              f"{durations[-1]:6.1f}s   {eta}")

    if tasks:
        print(f"step log -> {write_logs_csv(out_dir)}")
        report = write_log_report(out_dir, agent.endpoint, agent.model,
                                  len(tasks), sum(durations))
        print(f"report   -> {report}")


if __name__ == "__main__":
    main()
