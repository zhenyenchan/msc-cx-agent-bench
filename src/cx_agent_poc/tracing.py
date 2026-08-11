"""Trace harness that records agent runs"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path

from cx_agent.agent import run

TRACES_DIR = Path("traces")
TRACES_DIR.mkdir(exist_ok=True)


def run_with_trace(
    question: str,
    df_reviews,
    df_exploded,
    expected_tool: str | None = None,
    question_type: str | None = None,
    save: bool = True,
    model: str = "ollama/qwen2.5:7b-instruct",
) -> dict:
    """Run a single question and capture a full trace record.

    Args:
        question: the user question
        df_reviews, df_exploded: pre-loaded data
        expected_tool: ground-truth tool for skill selection metric (optional)
        question_type: 'descriptive' | 'inferential' | 'reporting' | 'out_of_scope'
        save: write the trace to disk
        model: model string

    Returns:
        A trace record (dict) with run metadata, the agent's answer, and the full step trace.
    """
    run_id = str(uuid.uuid4())[:8]
    start = time.time()

    result = run(question, df_reviews=df_reviews, df_exploded=df_exploded, model=model, verbose=False)

    latency = round(time.time() - start, 2)

    record = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "question_type": question_type,
        "expected_tool": expected_tool,
        "answer": result["answer"],
        "latency_seconds": latency,
        "trace": result["trace"],
        "model": model,
    }

    if save:
        path = TRACES_DIR / f"{run_id}.json"
        with path.open("w") as f:
            json.dump(record, f, indent=2)

    return record


def load_trace(run_id: str) -> dict:
    """Load a previously saved trace by run_id."""
    path = TRACES_DIR / f"{run_id}.json"
    with path.open() as f:
        return json.load(f)