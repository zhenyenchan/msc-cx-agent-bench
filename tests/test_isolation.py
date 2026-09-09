"""R3: the agent process receives only the question and the tool schemas.

The serialised prompt for every task — everything that could ever cross into the
model — must contain no gold field, and the harness-side modules must never import
the gold loader.
"""

import ast
import inspect
import json

from cx_agent_bench import agents, harness, tool_schemas, tasks_public, tools
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


def test_tools_docstrings_never_reach_the_agent_surface():
    """tools.py docstrings are implementation notes (they cite the gold notebook
    and its conventions) and must stay implementation-side: not in the serialised
    prompt surface, and never echoed by an observation."""
    surface = harness.SYSTEM_PROMPT + json.dumps(TOOL_SCHEMAS)
    for task in load_public_tasks():
        surface += task["question"]
    for marker in ("gold", "notebook", "ipynb"):
        assert marker not in surface.lower()

    doclines = set()
    for obj in (tools, tools.Tools,
                *(getattr(tools.Tools, n)
                  for n in ("filter", "summarise", "rank", "ztest", "answer"))):
        doc = inspect.getdoc(obj) or ""
        doclines |= {ln.strip() for ln in doc.splitlines() if len(ln.strip()) >= 30}
    assert doclines

    # drive every tool through success, note and error paths; no observation
    # may contain a docstring line
    t = tools.Tools(tasks_public.load_dataset("easy"))
    t.filter(industry="Banking")                     # ok
    t.filter(org="No Such Org")                      # unrecognised-name note
    t.filter(aspect="")                              # malformed-value error
    t.summarise("s1", group_by="aspect", min_n=30)   # ok, with below_min_n
    t.summarise("s2")                                # empty-selection error
    t.summarise("s99")                               # bad ref
    t.rank("t1", by="neg_rate", top_k=3, exclude="non_actionable")
    t.rank("t1", by="nope")                          # bad column
    t.filter(industry="Fashion")
    t.ztest("s1", "s3")                              # ok
    t.ztest("s1", "s2")                              # one side empty
    t.answer("No answer.", abstain=True)
    observations = json.dumps([c["result"] for c in t.calls], default=str)
    for line in doclines:
        assert line not in observations, line


def test_harness_side_modules_never_import_gold_loader():
    for module in (harness, agents, tool_schemas, tasks_public, tools):
        tree = ast.parse(inspect.getsource(module))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
                imported |= {alias.name for alias in node.names}
        assert not any("tasks_gold" in name for name in imported), module.__name__
