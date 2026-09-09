"""Scorer-only task loader (R3).

Gold answers and gold tool paths. Imported by scoring only — the harness and the
agents must never import this module; tests/test_isolation.py enforces that no
gold field can appear in any serialised prompt.
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS_GOLD_CSV = REPO_ROOT / "benchmark_outputs" / "tasks_gold.csv"

GOLD_FIELDS = ["gold_answer", "gold_tool_path_display", "gold_tool_path_json",
               "gold_num_steps"]


def load_gold_tasks():
    """Gold fields keyed by task_id: {task_id: {gold_answer, gold_tool_path_json, ...}}.

    The question text (a public field) rides along for scoring reports."""
    gold = pd.read_csv(TASKS_GOLD_CSV, index_col=0)
    return {row["task_id"]: {field: row[field] for field in ["question"] + GOLD_FIELDS}
            for _, row in gold.iterrows()}
