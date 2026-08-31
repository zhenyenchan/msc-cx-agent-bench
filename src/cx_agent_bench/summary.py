"""Benchmark summary: aggregates per-run scores.csv files into results.csv.

Layout assumption: one run directory = one agent x one seed, holding the traces
and the scores.csv that scoring.py wrote (one row per task, the six metrics).
This script reads every scores.csv under a root and, per agent:

  1. averages each metric across the N tasks of a run  -> seed-level score
  2. mean and std of the seed-level scores across the k seeds
  3. pass^k for insight: fraction of tasks where ALL k runs scored >= 4

results.csv: one row per agent, 13 metric columns (mean+std of 6 metrics, pass^k).
The repo has exactly one results.csv, benchmark_outputs/results.csv; run_suite
refreshes it after every run, and the CLI defaults to the same file:

    python -m cx_agent_bench.summary [root_dir] [--out results.csv]
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from .scoring import METRIC_COLS, load_valid_task_ids
from .tasks_public import REPO_ROOT

PASS_THRESHOLD = 4  # insight score counted as a pass
RESULTS_CSV = REPO_ROOT / "benchmark_outputs" / "results.csv"


def run_agent_id(run_dir):
    """Agent identity from the first trace's run_start record. The model field is
    the agent; agent_id is the run label and carries the seed suffix, so grouping
    by it would split one agent's k seed runs into k agents."""
    for path in sorted(run_dir.glob("*.jsonl")):
        with open(path, encoding="utf-8") as fh:
            record = json.loads(fh.readline())
        if record.get("type") == "run_start":
            return record.get("model") or record.get("agent_id") or run_dir.name
    return run_dir.name


def summarise_agent(runs):
    """runs: list of per-task score frames, one per seed -> one results row."""
    seed_means = pd.DataFrame([df[METRIC_COLS].mean() for df in runs])
    row = {"k_seeds": len(runs), "n_tasks": int(runs[0]["task_id"].nunique())}
    for m in METRIC_COLS:
        row[f"{m}_mean"] = round(float(seed_means[m].mean()), 4)
        row[f"{m}_std"] = round(float(seed_means[m].std(ddof=1)), 4)

    # pass^k: tasks where every seed's insight score clears the threshold
    insight = pd.concat(
        [df.set_index("task_id")["insight_score"] for df in runs], axis=1)
    row["pass^k"] = (round(float((insight >= PASS_THRESHOLD).all(axis=1).mean()), 4)
                     if insight.notna().any().any() else None)
    return row


def summarise(root):
    valid = load_valid_task_ids()
    by_agent = {}
    for csv in sorted(Path(root).rglob("scores.csv")):
        df = pd.read_csv(csv)
        if not set(METRIC_COLS) <= set(df.columns):
            print(f"skipping old-format scores.csv: {csv} (re-score the dir)")
            continue
        if valid is not None:  # a stale 50-task file must not dilute the average
            df = df[df["task_id"].isin(valid)]
        by_agent.setdefault(run_agent_id(csv.parent), []).append(df)
    results = pd.DataFrame(
        {agent: summarise_agent(runs) for agent, runs in by_agent.items()}).T
    results.index.name = "agent_id"
    return results


def update_results(root=None, out=None):
    """Rebuild the single repo results.csv from every scores.csv under root.
    Returns the path written, or None if there was nothing to summarise."""
    results = summarise(root or REPO_ROOT / "traces")
    if not len(results):
        return None
    out = Path(out or RESULTS_CSV)
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Summarise all scores.csv under a root into results.csv.")
    parser.add_argument("root_dir", nargs="?", default=None,
                        help="default: <repo>/traces")
    parser.add_argument("--out", default=None,
                        help=f"results CSV path (default: {RESULTS_CSV})")
    args = parser.parse_args()
    out = update_results(args.root_dir, args.out)
    if out is None:
        print("no scores.csv found")
        return
    print(pd.read_csv(out).to_string(index=False))
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
