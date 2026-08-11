"""Agent-facing task loader (R3).

Loads only what the agent is allowed to see: task_id, dataset, question.
Gold answers and gold tool paths live in tasks_gold.py, which the harness
never imports; the two sides join on task_id downstream, in scoring.
"""

import hashlib
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

TASKS_PUBLIC_CSV = DATA_DIR / "tasks_public.csv"
DATASET_CSVS = {"easy": DATA_DIR / "easy_data.csv", "hard": DATA_DIR / "hard_data.csv"}
TOOLS_MODULE_PATH = DATA_DIR / "tools.py"

PUBLIC_FIELDS = ["task_id", "dataset", "question"]


def load_public_tasks():
    """The 50 public tasks as a list of {task_id, dataset, question} dicts."""
    tasks = pd.read_csv(TASKS_PUBLIC_CSV)
    assert list(tasks.columns) == PUBLIC_FIELDS, (
        f"tasks_public.csv must contain exactly {PUBLIC_FIELDS}, "
        f"got {list(tasks.columns)}")
    return tasks.to_dict("records")


def load_dataset(name):
    """The review corpus a task is asked of: 'easy' or 'hard'."""
    return pd.read_csv(DATASET_CSVS[name])


def file_sha256(path):
    """Provenance hash for the manifest (R10)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
