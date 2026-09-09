"""R4, R6–R8, R11: protocol enforcement and trace format, exercised offline.

A scripted agent (same interface as OllamaAgent/DummyAgent) drives the harness
without a model, so handle validation, structured error observations, the single
terminal action, the step cap and the JSONL trace shape are all testable in CI.
"""

import json

from cx_agent_bench.agents import AgentStep, DummyAgent, ToolCallReq
from cx_agent_bench.harness import run_task
from cx_agent_bench.scoring import agent_canonical_json, read_trace
from cx_agent_bench.tasks_public import load_public_tasks

TASK = next(t for t in load_public_tasks() if t["task_id"] == "T1-E1")

STEP_FIELDS = {"idx", "tool", "args", "handle", "canonical_signature", "output",
               "status", "tokens_in", "tokens_out", "latency_s", "raw_response"}


class ScriptedAgent:
    """Replays a fixed list of AgentSteps through the standard agent interface."""

    def __init__(self, steps):
        self.steps = iter(steps)
        self.agent_id, self.model = "scripted", "scripted"
        self.temperature, self.seed = 0.0, 0

    def model_digest(self):
        return None

    def step(self, messages, tool_schemas):
        return next(self.steps)


def _run(agent, tmp_path, **kw):
    trace = run_task(agent, TASK, tmp_path, **kw)
    return read_trace(trace)


def test_dummy_agent_runs_through_identical_harness(tmp_path):
    start, steps, end = _run(DummyAgent(mode="null"), tmp_path)
    assert start["task_id"] == "T1-E1" and start["agent_id"] == "dummy-null"
    assert [s["tool"] for s in steps] == ["answer"]
    assert STEP_FIELDS <= set(steps[0])
    assert end["terminal_state"] == "submitted"
    assert end["final_answer"] == {"text": "", "abstain": False}


def test_invented_handle_is_an_agent_mistake_not_a_violation(tmp_path):
    # only no_tool_call and unknown_tool are protocol violations; a handle the
    # agent invents is a genuine mistake, fed back as a corrective observation
    agent = ScriptedAgent([
        AgentStep(tool_calls=[ToolCallReq("summarise", {"ref": "s99"})]),
        AgentStep(tool_calls=[ToolCallReq("answer", {"text": "done"})]),
    ])
    _, steps, end = _run(agent, tmp_path)
    assert steps[0]["status"] == "error"
    assert steps[0]["error_kind"] == "bad_ref"
    assert steps[0]["protocol_violation"] is False
    assert steps[0]["output"]["status"] == "error"          # structured, not raised
    assert end["terminal_state"] == "submitted"             # the loop continued


def test_wrong_kind_handle_is_rejected_with_the_producing_tool_named(tmp_path):
    # a real handle in the wrong slot (ztest on a table) must come back as a
    # corrective bad_ref, not crash inside the tool as KeyError: 'sentiment'
    agent = ScriptedAgent([
        AgentStep(tool_calls=[ToolCallReq("filter", {"industry": "Banking"})]),
        AgentStep(tool_calls=[ToolCallReq("summarise", {"ref": "s1"})]),
        AgentStep(tool_calls=[ToolCallReq("ztest", {"ref_a": "t1", "ref_b": "s1"})]),
        AgentStep(tool_calls=[ToolCallReq("answer", {"text": "done"})]),
    ])
    _, steps, end = _run(agent, tmp_path)
    assert steps[2]["status"] == "error"
    assert steps[2]["error_kind"] == "bad_ref"
    assert steps[2]["protocol_violation"] is False       # real handle, wrong slot
    assert "produced by filter" in steps[2]["output"]["error"]
    assert end["terminal_state"] == "submitted"


def test_mistyped_args_are_screened_before_dispatch(tmp_path):
    # each of these previously crashed inside pandas/statsmodels with an opaque
    # tool_exception; the schema declares the types, so screening happens upfront
    agent = ScriptedAgent([
        AgentStep(tool_calls=[ToolCallReq("filter", {"industry": "Banking"})]),
        AgentStep(tool_calls=[ToolCallReq("summarise",
                                          {"ref": "s1", "group_by": ["aspect"]})]),
        AgentStep(tool_calls=[ToolCallReq("summarise",
                                          {"ref": "s1", "group_by": "aspect"})]),
        AgentStep(tool_calls=[ToolCallReq("rank", {"ref": "t1", "top_k": "all"})]),
        AgentStep(tool_calls=[ToolCallReq("rank", {"ref": "t1",
                                                   "ascending": "yes"})]),
        AgentStep(tool_calls=[ToolCallReq("ztest", {"ref_a": "s1", "ref_b": "s1",
                                                    "alternative": "one-sided"})]),
        AgentStep(tool_calls=[ToolCallReq("answer", {"text": "done"})]),
    ])
    _, steps, end = _run(agent, tmp_path)
    assert [s["status"] for s in steps] == ["ok", "error", "ok", "error", "error",
                                            "error", "ok"]
    errors = [s for s in steps if s["status"] == "error"]
    assert all(s["error_kind"] == "bad_args" for s in errors)
    assert all(not s["protocol_violation"] for s in errors)
    assert "one of" in steps[5]["output"]["error"]          # enum named
    assert end["terminal_state"] == "submitted"


def test_unoffered_tools_are_rejected_in_the_no_tool_condition(tmp_path):
    from cx_agent_bench.tool_schemas import TOOL_SCHEMAS
    answer_only = [s for s in TOOL_SCHEMAS if s["function"]["name"] == "answer"]
    agent = ScriptedAgent([
        AgentStep(tool_calls=[ToolCallReq("filter", {"industry": "Banking"})]),
        AgentStep(tool_calls=[ToolCallReq("answer", {"text": "No answer.",
                                                     "abstain": True})]),
    ])
    _, steps, end = _run(agent, tmp_path, tool_schemas=answer_only)
    # a real tool outside the offered set must not execute or issue a handle
    assert steps[0]["status"] == "error"
    assert steps[0]["error_kind"] == "unknown_tool"
    assert steps[0]["handle"] is None
    assert "available tools: answer" in steps[0]["output"]["error"]
    assert end["terminal_state"] == "submitted"


def test_unknown_tool_and_bad_args_come_back_as_observations(tmp_path):
    agent = ScriptedAgent([
        AgentStep(tool_calls=[ToolCallReq("plot", {})]),
        AgentStep(tool_calls=[ToolCallReq("filter", {"nonsense": 1})]),
        AgentStep(tool_calls=[ToolCallReq("answer", {"text": "done"})]),
    ])
    _, steps, end = _run(agent, tmp_path)
    assert [s["status"] for s in steps] == ["error", "error", "ok"]
    assert steps[0]["protocol_violation"] is True           # unknown tool
    assert steps[1]["protocol_violation"] is False          # malformed args only
    assert end["terminal_state"] == "submitted"


def test_unusable_filter_values_error_instead_of_matching_nothing(tmp_path):
    # '' / {} / [] can never match a row; reporting ok with n_rows=0 would tell
    # the agent the slice does not exist when its query was merely malformed.
    agent = ScriptedAgent([
        AgentStep(tool_calls=[ToolCallReq("filter", {"industry": "Banking",
                                                     "aspect": ""})]),
        AgentStep(tool_calls=[ToolCallReq("filter", {"aspect": {}})]),
        AgentStep(tool_calls=[ToolCallReq("filter", {"org": []})]),
        AgentStep(tool_calls=[ToolCallReq("answer", {"text": "done"})]),
    ])
    _, steps, end = _run(agent, tmp_path)
    assert [s["status"] for s in steps] == ["error", "error", "error", "ok"]
    assert all(s["protocol_violation"] is False for s in steps[:3])
    assert "empty string" in steps[0]["output"]["error"]
    assert end["terminal_state"] == "submitted"


def test_five_identical_no_call_replies_stall_the_run(tmp_path):
    agent = ScriptedAgent([AgentStep(tool_calls=[], content="The rate is 50.0%.")
                           for _ in range(10)])
    _, steps, end = _run(agent, tmp_path)
    assert len(steps) == 5
    assert all(s["protocol_violation"] for s in steps)
    assert end["terminal_state"] == "stalled"
    # feedback escalates after the first repeat and names the answer tool
    assert steps[0]["output"]["error"].startswith("no tool call in your reply;")
    assert "pass it to the answer tool" in steps[1]["output"]["error"]
    assert "5 times" in steps[4]["output"]["error"]


def test_varied_no_call_replies_reset_the_stall_counter(tmp_path):
    replies = [AgentStep(tool_calls=[], content=f"thinking, take {i}")
               for i in range(8)]
    replies.append(AgentStep(tool_calls=[ToolCallReq("answer", {"text": "done"})]))
    _, steps, end = _run(agent := ScriptedAgent(replies), tmp_path)
    assert len(steps) == 9
    assert end["terminal_state"] == "submitted"


def test_structured_call_resets_the_stall_counter(tmp_path):
    prose = [AgentStep(tool_calls=[], content="same text") for _ in range(4)]
    replies = (prose
               + [AgentStep(tool_calls=[ToolCallReq("filter", {"industry": "Banking"})])]
               + prose
               + [AgentStep(tool_calls=[ToolCallReq("answer", {"text": "done"})])])
    _, steps, end = _run(ScriptedAgent(replies), tmp_path)
    assert end["terminal_state"] == "submitted"


def test_step_cap_terminates_run(tmp_path):
    agent = ScriptedAgent([AgentStep(tool_calls=[ToolCallReq("filter",
                                                             {"industry": "Banking"})])
                           for _ in range(10)])
    _, steps, end = _run(agent, tmp_path, step_cap=3)
    assert len(steps) == 3
    assert end["terminal_state"] == "step_cap"


def test_log_report_aggregates_traces(tmp_path):
    from cx_agent_bench.run_baseline import write_log_report
    run_task(ScriptedAgent([
        AgentStep(tool_calls=[ToolCallReq("filter", {"industry": "Banking"})]),
        AgentStep(tool_calls=[]),                                # no tool call
        AgentStep(tool_calls=[ToolCallReq("summarise", {"ref": "s9"})]),  # bad ref
        AgentStep(tool_calls=[ToolCallReq("answer", {"text": "done"})]),
    ]), TASK, tmp_path)
    run_task(ScriptedAgent([AgentStep(tool_calls=[], content="stuck")
                            for _ in range(6)]), TASK, tmp_path)
    report = write_log_report(tmp_path, "in-process", "scripted", 2, 61.0)
    text = report.read_text(encoding="utf-8")
    assert "model: scripted" in text and "total tasks: 2" in text
    assert "total time: 0h 1m 1s" in text
    assert "total cost (USD): n/a (gateway) / n/a (list prices)" in text
    assert "submitted tasks: 1" in text
    assert "failed tasks (not submitted): 1" in text
    # both runs are T1-E1, so one distinct task carries the no-call errors
    assert 'tasks with "no tool call" errors: 1' in text
    assert "task ids: T1-E1" in text
    assert 'tasks with "unknown tool" errors: 0' in text
    # 3 + 5 non-answer steps; 1+1+5 errors; 1+5 no-call steps
    assert "total steps: 8 (excluding answer steps)" in text
    assert "error steps: 7 (87.5% of total steps)" in text
    assert "no tool call steps: 6 (75.0% of total steps)" in text
    assert "reached loop cap (5 consecutive identical replies): T1-E1" in text
    assert "no_tool_call             6  yes" in text
    assert "bad_ref                  1  no" in text


def test_list_price_cost_prices_cached_tokens_separately():
    from cx_agent_bench.harness import list_price_cost
    prices = {"input": 0.30, "cached_input": 0.03, "output": 2.50}
    steps = [{"tokens_in": 1000, "tokens_out": 100,
              "raw_response": {"usage": {"prompt_tokens_details":
                                         {"cached_tokens": 600}}}},
             {"tokens_in": 500, "tokens_out": 50, "raw_response": {}}]
    # (400*0.30 + 600*0.03 + 100*2.50 + 500*0.30 + 50*2.50) / 1e6
    assert list_price_cost(steps, prices) == 0.000663
    assert list_price_cost(steps, None) is None


def test_trace_is_valid_jsonl_and_canonicalises(tmp_path):
    agent = ScriptedAgent([
        AgentStep(tool_calls=[ToolCallReq("filter", {"industry": "Banking",
                                                     "aspect": "app-website"})]),
        AgentStep(tool_calls=[ToolCallReq("summarise", {"ref": "s1"})]),
        AgentStep(tool_calls=[ToolCallReq("answer", {"text": "18.7%"})]),
    ])
    trace = run_task(agent, TASK, tmp_path)
    lines = [json.loads(l) for l in trace.read_text(encoding="utf-8").splitlines()]
    assert [l["type"] for l in lines] == ["run_start", "step", "step", "step",
                                          "run_end"]
    _, steps, _ = read_trace(trace)
    # ref expanded into the producing step's signature, per the shared normal form
    assert steps[1]["canonical_signature"]["args"]["ref"]["tool"] == "filter"
    assert json.loads(agent_canonical_json(steps))[0]["tool"] == "filter"
