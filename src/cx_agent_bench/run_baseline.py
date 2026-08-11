"""Run the baseline agent (or the dummy) over the public task set.

    python -m cx_agent_bench.run_baseline --agent qwen                  # all 50 tasks
    python -m cx_agent_bench.run_baseline --agent dummy-null
    python -m cx_agent_bench.run_baseline --agent qwen --tasks T1-E1 T4-H2
    python -m cx_agent_bench.run_baseline --agent qwen --limit 3

Execution only: traces are written to --out; score them afterwards with
    python -m cx_agent_bench.scoring <trace_dir> --out scores.csv
"""

import argparse
import json
import time
from pathlib import Path

from .agents import DummyAgent, OllamaAgent
from .harness import STEP_CAP, TIMEOUT_S, build_manifest, run_task
from .tasks_public import REPO_ROOT, load_public_tasks


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
    for i, task in enumerate(tasks, 1):
        t0 = time.perf_counter()
        trace = run_task(agent, task, out_dir, manifest=manifest,
                         step_cap=args.step_cap, timeout_s=args.timeout)
        last = json.loads(Path(trace).read_text(encoding="utf-8").splitlines()[-1])
        print(f"[{i:>2}/{len(tasks)}] {task['task_id']:<7} "
              f"{last.get('terminal_state', '?'):<9} "
              f"calls={last.get('n_tool_calls', '?'):<3} "
              f"{time.perf_counter() - t0:6.1f}s")


if __name__ == "__main__":
    main()
