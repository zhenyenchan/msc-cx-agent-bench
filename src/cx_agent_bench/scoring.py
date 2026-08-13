"""Scoring: a downstream process over stored trace files (R9).

Nothing here runs during a benchmark run. It reads the JSONL traces the harness
wrote, joins gold fields on task_id, and computes per-run rows plus a small
aggregate summary. Gold tool paths never contain the answer step, so the agent's
answer call is stripped before path comparison.
"""

import json
from pathlib import Path

import pandas as pd

from .harness import list_price_cost
from .tasks_gold import load_gold_tasks


def read_trace(path):
    """One trace file -> (run_start, [steps], run_end|None)."""
    start, steps, end = None, [], None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            if record["type"] == "run_start":
                start = record
            elif record["type"] == "step":
                steps.append(record)
            elif record["type"] == "run_end":
                end = record
    return start, steps, end


def agent_canonical_json(steps):
    """The agent's analysis path in gold's normal form: the canonical signatures of
    the successfully validated calls, minus the terminal answer call."""
    path = [s["canonical_signature"] for s in steps
            if s.get("canonical_signature") is not None
            and s.get("status") == "ok" and s.get("tool") != "answer"]
    return json.dumps(path, sort_keys=True, separators=(",", ":"))


def error_kinds(steps):
    """Error steps of a run bucketed by kind, as 'no_tool_call:5;bad_ref:2'.

    Kinds stamped by the harness (unknown_tool, bad_args, bad_ref, tool_exception)
    are used as-is; a no-call turn has no tool, and a validated call whose tool
    returned an error dict is 'tool_error_result'."""
    counts = {}
    for s in steps:
        if s["status"] != "error":
            continue
        kind = s.get("error_kind") or ("no_tool_call" if s["tool"] is None
                                       else "tool_error_result")
        counts[kind] = counts.get(kind, 0) + 1
    return ";".join(f"{k}:{v}" for k, v in
                    sorted(counts.items(), key=lambda kv: -kv[1]))


def score_run(trace_path, gold):
    start, steps, end = read_trace(trace_path)
    if start is None:
        return None
    task_gold = gold.get(start["task_id"], {})
    final = (end or {}).get("final_answer") or {}
    agent_path = agent_canonical_json(steps)
    gold_path = task_gold.get("gold_tool_path_json")
    return {
        "run_id": start["run_id"],
        "task_id": start["task_id"],
        "agent_id": start["agent_id"],
        "model": start["model"],
        "terminal_state": (end or {}).get("terminal_state", "missing_run_end"),
        "agent_n_steps": len(steps),
        "n_errors": sum(s["status"] == "error" for s in steps),
        "error_kinds": error_kinds(steps),
        "n_protocol_violations": sum(bool(s.get("protocol_violation")) for s in steps),
        "path_match": (agent_path == gold_path) if gold_path is not None else None,
        "agent_tool_path": agent_path,
        "gold_tool_path": gold_path,
        "gold_n_steps": task_gold.get("gold_num_steps"),
        "abstained": final.get("abstain"),
        "question": task_gold.get("question"),
        "agent_answer": final.get("text"),
        "gold_answer": task_gold.get("gold_answer"),
        "tokens_in": sum(s["tokens_in"] or 0 for s in steps),
        "tokens_out": sum(s["tokens_out"] or 0 for s in steps),
        "cost_usd_gateway": (
            round(sum(s.get("cost_usd") or 0 for s in steps), 6)
            if any(s.get("cost_usd") is not None for s in steps) else None),
        "cost_usd_list": list_price_cost(
            steps, (start.get("manifest") or {}).get("list_prices_usd_per_1m")),
        "latency_s": (end or {}).get("wall_s"),
    }


def score_dir(trace_dir, out_csv=None):
    """Score every trace in a directory; optionally write a per-run CSV."""
    gold = load_gold_tasks()
    rows = [row for path in sorted(Path(trace_dir).glob("*.jsonl"))
            if (row := score_run(path, gold)) is not None]
    scores = pd.DataFrame(rows)
    if out_csv and len(scores):
        scores.to_csv(out_csv, index=False)
    return scores


def summarise_scores(scores):
    """Aggregate per agent: terminal-state mix, path-match rate, cost."""
    out = {}
    for agent_id, group in scores.groupby("agent_id"):
        out[agent_id] = {
            "runs": len(group),
            "submitted": int((group.terminal_state == "submitted").sum()),
            "stalled": int((group.terminal_state == "stalled").sum()),
            "step_cap": int((group.terminal_state == "step_cap").sum()),
            "timeout": int((group.terminal_state == "timeout").sum()),
            "error": int((group.terminal_state == "error").sum()),
            "path_match_rate": round(float(group.path_match.fillna(False).mean()), 3),
            "mean_steps": round(float(group.agent_n_steps.mean()), 2),
            "mean_errors": round(float(group.n_errors.mean()), 2),
            "protocol_violations": int(group.n_protocol_violations.sum()),
            "mean_tokens_out": round(float(group.tokens_out.mean()), 1),
            "total_cost_usd_gateway": (
                round(float(group.cost_usd_gateway.sum()), 4)
                if group.cost_usd_gateway.notna().any() else None),
            "total_cost_usd_list": (
                round(float(group.cost_usd_list.sum()), 4)
                if group.cost_usd_list.notna().any() else None),
        }
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Score stored traces (run separately "
                                                 "from execution).")
    parser.add_argument("trace_dir")
    parser.add_argument("--out", default=None, help="per-run scores CSV")
    args = parser.parse_args()
    scores = score_dir(args.trace_dir, args.out)
    if not len(scores):
        print("no traces found")
        return
    print(json.dumps(summarise_scores(scores), indent=2))
    if args.out:
        print(f"per-run scores -> {args.out}")


if __name__ == "__main__":
    main()
