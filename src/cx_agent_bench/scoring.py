"""Scoring: a downstream process over stored trace files (R9).

Nothing here runs during a benchmark run. It reads the JSONL traces the harness
wrote, joins gold fields on task_id, and computes per-run rows plus a small
aggregate summary. Gold tool paths never contain the answer step, so the agent's
answer call is stripped before path comparison. insight_score is graded by the
one LLM judge of the pipeline (INSIGHT_JUDGE_MODEL, Kimi K2 Thinking) over the
gateway, so scoring a run dir needs the gateway credentials in .env.
"""

import json
from pathlib import Path

import pandas as pd

from .harness import list_price_cost
from .tasks_gold import load_gold_tasks
from .tasks_public import REPO_ROOT

# The tasks that survived annotation pruning (wide human disagreement dropped).
# Agents still run all 50 tasks and their traces are kept, but scoring and the
# summary only count these, so every average over tasks divides by 44.
VALID_TASKS_CSV = REPO_ROOT / "data" / "valid_tasks_public.csv"


def load_valid_task_ids():
    """Set of valid task_ids, or None if the file is absent (score everything)."""
    if VALID_TASKS_CSV.exists():
        return set(pd.read_csv(VALID_TASKS_CSV)["task_id"])
    return None


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


def evaluation_metrics(steps, final, task_gold, end, cost_gateway, cost_list):
    """The six per-task benchmark metrics. Each is summed (never averaged) across
    the 50 tasks, so per-task values are bounded 0-1 except insight (0-5)."""
    # tool selection accuracy: recall of the gold tool set, order and args ignored
    gold_path = task_gold.get("gold_tool_path_json")
    gold_tools = ({call["tool"] for call in json.loads(gold_path)}
                  if gold_path else None)
    agent_tools = {s["tool"] for s in steps
                   if s.get("status") == "ok" and s.get("tool") != "answer"}

    # path efficiency: gold vs every non-answer turn spent (errors included)
    gold_n = task_gold.get("gold_num_steps")
    agent_n = sum(1 for s in steps if s.get("tool") != "answer")

    return {
        "task_completion": int(bool((final.get("text") or "").strip())),
        "insight_score": None,  # filled by add_insight_scores (LLM judge)
        "tool_selection_accuracy": (
            round(len(gold_tools & agent_tools) / len(gold_tools), 3)
            if gold_tools else None),
        "path_efficiency": (
            round(min(1.0, gold_n / agent_n), 3) if gold_n and agent_n else 0.0),
        "cost_usd": cost_gateway if cost_gateway is not None else cost_list,
        "latency_s": (end or {}).get("wall_s"),
    }


def score_run(trace_path, gold):
    start, steps, end = read_trace(trace_path)
    if start is None:
        return None
    task_gold = gold.get(start["task_id"], {})
    final = (end or {}).get("final_answer") or {}
    agent_path = agent_canonical_json(steps)
    gold_path = task_gold.get("gold_tool_path_json")
    cost_gateway = (round(sum(s.get("cost_usd") or 0 for s in steps), 6)
                    if any(s.get("cost_usd") is not None for s in steps) else None)
    cost_list = list_price_cost(
        steps, (start.get("manifest") or {}).get("list_prices_usd_per_1m"))
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
        "cost_usd_gateway": cost_gateway,
        "cost_usd_list": cost_list,
        # the six benchmark metrics, kept as the last columns of scores.csv
        **evaluation_metrics(steps, final, task_gold, end, cost_gateway, cost_list),
    }


METRIC_COLS = ["task_completion", "insight_score", "tool_selection_accuracy",
               "path_efficiency", "cost_usd", "latency_s"]

# The one LLM judge of the scoring pipeline: Kimi K2 Thinking, validated against
# the 3-rater human panel (kept-44 pairwise exact agreement at the human-human
# ceiling, see tagging/human_judge_results.ipynb). Do not swap judges per run;
# insight scores are only comparable when every run is graded by the same judge.
INSIGHT_JUDGE_MODEL = "vertex_ai/moonshotai/kimi-k2-thinking-maas"


def add_insight_scores(scores, model=INSIGHT_JUDGE_MODEL):
    """Fill the insight_score column in place via the LLM judge (~6s and ~$0.005
    per task through the gateway). A blank answer is scored 0 without an API
    call (the rubric's floor for a blank). If the gateway is not configured the
    column is left empty with a warning, so offline re-scoring still works."""
    if not len(scores):
        return scores
    from .run_judge import judge_one, make_client
    try:
        client = make_client()
    except SystemExit as exc:
        print(f"insight_score left empty (judge unavailable): {exc}")
        return scores
    for idx, row in scores.iterrows():
        answer = row.get("agent_answer")
        if not (isinstance(answer, str) and answer.strip()):
            scores.at[idx, "insight_score"] = 0
            continue
        if not (isinstance(row.get("gold_answer"), str) and row.get("question")):
            continue  # no gold reference to grade against
        try:
            result = judge_one(client, model, row)
        except Exception as exc:
            print(f"insight_score failed for {row['task_id']}: {exc}")
            continue
        scores.at[idx, "insight_score"] = result["judge_score"]
    return scores


def score_dir(trace_dir, out_csv=None):
    """Score every trace in one agent run's directory. The full per-run frame is
    returned, and the run gets its scores.csv: one row per task, the six metrics
    per row (aggregation is a separate later step). out_csv overrides the
    default location <trace_dir>/scores.csv."""
    gold = load_gold_tasks()
    rows = [row for path in sorted(Path(trace_dir).glob("*.jsonl"))
            if (row := score_run(path, gold)) is not None]
    scores = pd.DataFrame(rows)
    valid = load_valid_task_ids()
    if valid is not None and len(scores):
        scores = scores[scores["task_id"].isin(valid)]
    scores = add_insight_scores(scores)
    if len(scores):
        out_csv = out_csv or Path(trace_dir) / "scores.csv"
        scores[["task_id", *METRIC_COLS]].sort_values("task_id").to_csv(
            out_csv, index=False)
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
    parser.add_argument("--out", default=None,
                        help="scores CSV path (default: <trace_dir>/scores.csv)")
    args = parser.parse_args()
    scores = score_dir(args.trace_dir, args.out)
    if not len(scores):
        print("no traces found")
        return
    print(json.dumps(summarise_scores(scores), indent=2))
    print(f"scores -> {args.out or Path(args.trace_dir) / 'scores.csv'}")


if __name__ == "__main__":
    main()
