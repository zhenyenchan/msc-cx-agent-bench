"""R3: the agent process receives only the question and the tool schemas.

The serialised prompt for every task — everything that could ever cross into the
model — must contain no gold field, and the harness-side modules must never import
the gold loader.
"""

import ast
import inspect
import json

from cx_agent_bench import agents, harness, tool_schemas, tasks_public
from cx_agent_bench.harness import build_messages
from cx_agent_bench.tasks_gold import load_gold_tasks
from cx_agent_bench.tasks_public import load_public_tasks
from cx_agent_bench.tool_schemas import TOOL_SCHEMAS


def _serialised_prompt(task):
    """Everything the agent is given for a task, as one string."""
    return json.dumps(build_messages(task)) + json.dumps(TOOL_SCHEMAS)


def test_no_gold_field_in_any_serialised_prompt():
    gold = load_gold_tasks()
    for task in load_public_tasks():
        prompt = _serialised_prompt(task)
        fields = gold[task["task_id"]]
        assert fields["gold_answer"] not in prompt
        assert str(fields["gold_tool_path_json"]) not in prompt
        assert str(fields["gold_tool_path_display"]) not in prompt
        assert "gold" not in prompt.lower()


def test_public_loader_exposes_only_public_fields():
    for task in load_public_tasks():
        assert set(task) == {"task_id", "dataset", "question"}


def test_harness_side_modules_never_import_gold_loader():
    for module in (harness, agents, tool_schemas, tasks_public):
        tree = ast.parse(inspect.getsource(module))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
                imported |= {alias.name for alias in node.names}
        assert not any("tasks_gold" in name for name in imported), module.__name__
