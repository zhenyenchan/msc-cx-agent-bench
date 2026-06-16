"""Evaluation harness that computes benchmark metrics over trace records"""

import pandas as pd

from cx_agent.tracing import run_with_trace


# Metrics

def num_tool_calls(record: dict) -> int:
    """How many tool calls did the agent make?"""
    return sum(1 for s in record["trace"] if s["type"] == "tool_call")


def num_loops_detected(record: dict) -> int:
    """How many duplicate-call loops were caught?"""
    return sum(1 for s in record["trace"] if s["type"] == "loop_detected")


def tools_used(record: dict) -> list[str]:
    """Which tools did the agent invoke, in order?"""
    return [s["tool"] for s in record["trace"] if s["type"] == "tool_call"]


def skill_selection_correct(record: dict) -> bool | None:
    """Did the agent pick the expected tool? (None if no ground truth)"""
    expected = record.get("expected_tool")
    if expected is None:
        # Out-of-scope: success = no tool call
        if record.get("question_type") == "out_of_scope":
            return num_tool_calls(record) == 0
        return None
    return expected in tools_used(record)


def path_efficient(record: dict) -> bool:
    """Did the agent finish in the minimum number of steps?

    For now: 1 tool call (in-scope) or 0 (out-of-scope) = efficient.
    """
    if record.get("question_type") == "out_of_scope":
        return num_tool_calls(record) == 0
    return num_tool_calls(record) == 1


def answer_has_markup(record: dict) -> bool:
    """Did the final answer leak JSON/tool-call markup?"""
    answer = record.get("answer", "")
    bad_signals = ["Tool Calls:", "### User:", "### Assistant:", '{"', "```json"]
    return any(sig in answer for sig in bad_signals)


# Benchmarks

def evaluate_record(record: dict) -> dict:
    """Compute all metrics for a single trace record."""
    return {
        "num_tool_calls": num_tool_calls(record),
        "num_loops_detected": num_loops_detected(record),
        "tools_used": tools_used(record),
        "skill_correct": skill_selection_correct(record),
        "path_efficient": path_efficient(record),
        "answer_has_markup": answer_has_markup(record),
        "latency_seconds": record["latency_seconds"],
    }


def run_benchmark(
    test_cases: list[dict],
    df_reviews,
    df_exploded,
) -> pd.DataFrame:
    """Run a list of test cases and return a metrics DataFrame.

    Each test case is a dict with: question, expected_tool, question_type.
    """
    rows = []
    for i, case in enumerate(test_cases):
        print(f"[{i + 1}/{len(test_cases)}] {case['question'][:60]}...")
        record = run_with_trace(
            question=case["question"],
            df_reviews=df_reviews,
            df_exploded=df_exploded,
            expected_tool=case.get("expected_tool"),
            question_type=case.get("question_type"),
        )
        metrics = evaluate_record(record)
        rows.append({
            "question": case["question"],
            "type": case.get("question_type"),
            "expected_tool": case.get("expected_tool"),
            **metrics,
            "answer": record["answer"][:150],
        })
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame) -> None:
    """Print headline benchmark numbers."""
    print("=== Benchmark Summary ===\n")
    print(f"Total questions:        {len(results)}")
    print(f"Skill selection acc:    {results['skill_correct'].mean():.0%}")
    print(f"Path efficiency:        {results['path_efficient'].mean():.0%}")
    print(f"Markup leakage rate:    {results['answer_has_markup'].mean():.0%}")
    print(f"Avg tool calls:         {results['num_tool_calls'].mean():.1f}")
    print(f"Total loops detected:   {results['num_loops_detected'].sum()}")
    print(f"Avg latency (s):        {results['latency_seconds'].mean():.1f}")

    print("\n=== By question type ===")
    print(results.groupby("type").agg({
        "skill_correct": "mean",
        "path_efficient": "mean",
        "num_tool_calls": "mean",
        "latency_seconds": "mean",
    }).round(2))