"""Run the baseline agent (or the dummy) over the public task set.

    python -m cx_agent_bench.run_baseline --agent qwen                  # all 50 tasks
    python -m cx_agent_bench.run_baseline --agent dummy-null
    python -m cx_agent_bench.run_baseline --agent qwen --tasks T1-E1 T4-H2
    python -m cx_agent_bench.run_baseline --agent qwen --limit 3

Traces are written to --out; when the run finishes, scoring runs automatically:
the run dir gets its scores.csv and the repo-wide benchmark_outputs/results.csv
is refreshed from every scored run.
"""

import argparse
import csv
import json
import time
from pathlib import Path

from .agents import DummyAgent, OllamaAgent, OpenAICompatAgent
from .harness import (STEP_CAP, TIMEOUT_S, build_manifest, list_price_cost,
                      run_task)
from .tool_schemas import TOOL_SCHEMAS
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
                "cost_usd": s.get("cost_usd"),
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


def _money(value):
    return f"${value:.4f}" if value is not None else "n/a"


def write_log_report(out_dir, endpoint, model, n_tasks, total_s):
    """Aggregate the run's traces into a human-readable log_report.txt.

    Counts only — no gold fields beyond the offline GOLD_STEPS_TOTAL constant, so
    the execution/scoring split (R9) stays intact. Percentages are of the steps
    that carry analysis, i.e. answer steps are excluded, matching how gold paths
    are counted.
    """
    steps, ends = [], []
    for trace in sorted(Path(out_dir).glob("*.jsonl")):
        task_id = None
        for line in trace.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record["type"] == "run_start":
                task_id = record["task_id"]
            elif record["type"] == "step":
                record["_task_id"] = task_id
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

    gateway_cost = (sum(s.get("cost_usd") or 0 for s in steps)
                    if any(s.get("cost_usd") is not None for s in steps) else None)
    manifest_file = Path(out_dir) / "manifest.json"
    prices = (json.loads(manifest_file.read_text(encoding="utf-8"))
              .get("list_prices_usd_per_1m") if manifest_file.exists() else None)
    list_cost = list_price_cost(steps, prices)

    n_submitted = sum(e["terminal_state"] == "submitted" for e in ends)
    no_call_ids = sorted({s["_task_id"] for s in no_call})
    unknown_ids = sorted({s["_task_id"] for s in work
                          if s.get("error_kind") == "unknown_tool"})

    hours, rest = divmod(int(total_s), 3600)
    lines = [
        f"model: {model}",
        f"endpoint: {endpoint}",
        f"total time: {hours}h {rest // 60}m {rest % 60}s",
        "total cost (USD): "
        f"{_money(gateway_cost)} (gateway) / {_money(list_cost)} (list prices)",
        "",
        f"total tasks: {n_tasks}",
        f"submitted tasks: {n_submitted}",
        f"failed tasks (not submitted): {n_tasks - n_submitted}",
        "",
        f'tasks with "no tool call" errors: {len(no_call_ids)}',
        "task ids: " + (", ".join(no_call_ids) or "none"),
        f'tasks with "unknown tool" errors: {len(unknown_ids)}',
        "task ids: " + (", ".join(unknown_ids) or "none"),
        "",
        f"total steps: {len(work)} (excluding answer steps)",
        f"error steps: {len(errors)} ({pct(len(errors))} of total steps)",
        f"no tool call steps: {len(no_call)} ({pct(len(no_call))} of total steps)",
        f"gold steps: {GOLD_STEPS_TOTAL} (fixed)",
        "",
        "terminated task_ids:",
        f"reached step cap (38): {tasks_in('step_cap')}",
        f"reached timeout (600s): {tasks_in('timeout')}",
        f"reached loop cap (5 consecutive identical replies): {tasks_in('stalled')}",
        "",
        "error analysis:",
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


# qwen2.5-7b is not served by Vertex AI (404 in the gateway's project/region);
# the qwen the gateway can serve is the Qwen3-Next 80B MaaS model.
CHATTERMILL_DEFAULT_MODEL = "vertex_ai/qwen/qwen3-next-80b-a3b-instruct-maas"


def make_agent(name, seed, model=None):
    if name == "qwen":
        return OllamaAgent(model="qwen2.5:7b-instruct", temperature=0.0, seed=seed)
    if name == "qwen-openrouter":
        return OpenAICompatAgent(model="qwen/qwen-2.5-7b-instruct",
                                 temperature=0.0, seed=seed)
    if name == "chattermill":
        return OpenAICompatAgent(model=model or CHATTERMILL_DEFAULT_MODEL,
                                 temperature=0.0, seed=seed,
                                 provider="chattermill",
                                 api_key_env="OPENAI_API_KEY",
                                 api_base_env="OPENAI_BASE_URL")
    if name == "dummy-null":
        return DummyAgent(mode="null", seed=seed)
    if name == "dummy-random":
        return DummyAgent(mode="random", seed=seed)
    raise SystemExit(f"unknown agent '{name}'")


def select_tasks(task_ids=None, limit=None):
    """The public tasks to run, filtered by explicit ids and/or a count limit."""
    tasks = load_public_tasks()
    if task_ids:
        wanted = set(task_ids)
        tasks = [t for t in tasks if t["task_id"] in wanted]
        missing = wanted - {t["task_id"] for t in tasks}
        if missing:
            raise SystemExit(f"unknown task ids: {sorted(missing)}")
    if limit:
        tasks = tasks[:limit]
    return tasks


def run_suite(agent, tasks, out_dir, step_cap=STEP_CAP, timeout_s=TIMEOUT_S,
              tool_schemas=TOOL_SCHEMAS):
    """Manifest, warmup, the run loop with per-task progress, step log and
    report — the whole execution pipeline, shared by every runner script.

    tool_schemas is the tool set offered to the agent (recorded in the
    manifest); the no-tool ablation passes the answer schema alone."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(agent)
    manifest["offered_tools"] = [s["function"]["name"] for s in tool_schemas]
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                           encoding="utf-8")

    print(f"agent={agent.agent_id}  tasks={len(tasks)}  step_cap={step_cap}  "
          f"timeout={timeout_s}s\ntraces -> {out_dir}")

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
                         step_cap=step_cap, timeout_s=timeout_s,
                         tool_schemas=tool_schemas)
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

    print(f"step log -> {write_logs_csv(out_dir)}")
    report = write_log_report(out_dir, agent.endpoint, agent.model,
                              len(tasks), sum(durations))
    print(f"report   -> {report}")

    # Scoring stays downstream of execution (R9): the run is finished and its
    # traces are on disk before any gold field is read. Imported here so the
    # execution modules never load gold data at import time.
    from .scoring import score_dir
    from .summary import update_results
    score_dir(out_dir)
    print(f"scores   -> {out_dir / 'scores.csv'}")
    print(f"results  -> {update_results()}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", default="qwen",
                        choices=["qwen", "qwen-openrouter", "chattermill",
                                 "dummy-null", "dummy-random"])
    parser.add_argument("--model", default=None,
                        help="gateway model id for --agent chattermill "
                             f"(default: {CHATTERMILL_DEFAULT_MODEL})")
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="task ids to run (default: all)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--step-cap", type=int, default=STEP_CAP)
    parser.add_argument("--timeout", type=int, default=TIMEOUT_S)
    parser.add_argument("--out", default=None,
                        help="trace directory (default: traces/baseline/<agent>_<ts>)")
    args = parser.parse_args()

    agent = make_agent(args.agent, args.seed, model=args.model)
    tasks = select_tasks(args.tasks, args.limit)
    out_dir = Path(args.out) if args.out else (
        REPO_ROOT / "traces" / "baseline"
        / f"{agent.agent_id}_{time.strftime('%Y%m%d_%H%M%S')}")
    if tasks:
        run_suite(agent, tasks, out_dir, args.step_cap, args.timeout)


if __name__ == "__main__":
    main()
