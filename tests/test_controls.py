"""R10: fixed agent controls and their logging.

Every agent sends the same inference parameters (temperature 0, seed, a per-step
completion cap) and nothing else; every run records those parameters, the
stopping rules and the served model version string on disk.
"""

import json

from cx_agent_bench.agents import (MAX_OUTPUT_TOKENS, TEMPERATURE, AgentStep,
                                   DummyAgent, OpenAICompatAgent, ToolCallReq)
from cx_agent_bench.harness import (STALL_LIMIT, STEP_CAP, TIMEOUT_S,
                                    TOKEN_BUDGET, build_manifest, run_task)
from cx_agent_bench.run_baseline import load_seeds
from cx_agent_bench.scoring import read_trace
from cx_agent_bench.tasks_public import load_public_tasks

TASK = next(t for t in load_public_tasks() if t["task_id"] == "T1-E1")


class VersionedAgent:
    """Scripted agent that reports a served model string and token usage."""

    def __init__(self, steps, tokens_in=1000, served="model-v1.2.3"):
        self.steps = iter(steps)
        self.agent_id, self.model = "versioned", "model"
        self.temperature, self.seed = 0.0, 1
        self.tokens_in, self.served = tokens_in, served

    def request_options(self):
        return {"temperature": 0.0, "seed": 1, "max_tokens": 4096}

    def model_digest(self):
        return {"provider": "test"}

    def step(self, messages, tool_schemas):
        step = next(self.steps)
        step.tokens_in, step.tokens_out = self.tokens_in, 10
        step.served_model, step.finish_reason = self.served, "tool_calls"
        return step


def _filter_then_answer(n_filters):
    return ([AgentStep(tool_calls=[ToolCallReq("filter", {"industry": "Banking"})])
             for _ in range(n_filters)]
            + [AgentStep(tool_calls=[ToolCallReq("answer", {"text": "done"})])])


def test_frozen_seeds_are_five_distinct_ints():
    seeds = load_seeds()
    assert len(seeds) == 5 and len(set(seeds)) == 5
    assert all(isinstance(s, int) for s in seeds)
    assert seeds == sorted(seeds) == [2555, 3829, 4429, 7633, 7902]


def test_run_start_records_inference_and_stopping_rules(tmp_path):
    agent = VersionedAgent(_filter_then_answer(1))
    start, steps, end = read_trace(run_task(agent, TASK, tmp_path))
    assert start["inference"]["request_options"] == agent.request_options()
    assert "provider" in start["inference"]["unlisted_params"]
    assert start["stopping_rules"] == {
        "step_cap": STEP_CAP, "timeout_s": TIMEOUT_S,
        "token_budget": TOKEN_BUDGET, "stall_limit": STALL_LIMIT}
    assert all(s["served_model"] == "model-v1.2.3" for s in steps)
    assert all(s["finish_reason"] == "tool_calls" for s in steps)
    assert end["served_models"] == ["model-v1.2.3"]
    assert end["tokens_in_total"] == 2000 and end["tokens_out_total"] == 20


def test_token_budget_terminates_run(tmp_path):
    # 1000 in + 10 out per step; budget 2500 is exceeded after step 3, so the
    # run stops before the 4th step and never reaches the answer call
    agent = VersionedAgent(_filter_then_answer(10))
    _, steps, end = read_trace(run_task(agent, TASK, tmp_path, token_budget=2500))
    assert end["terminal_state"] == "token_budget"
    assert len(steps) == 3 and end["final_answer"] is None
    assert "budget 2500" in end["error"]


def test_default_budget_does_not_interfere(tmp_path):
    agent = VersionedAgent(_filter_then_answer(2))
    _, steps, end = read_trace(run_task(agent, TASK, tmp_path))
    assert end["terminal_state"] == "submitted" and len(steps) == 3


def test_manifest_carries_controls_and_software(tmp_path):
    agent = VersionedAgent(_filter_then_answer(0))
    manifest = build_manifest(agent, step_cap=7, timeout_s=9, token_budget=11)
    assert manifest["inference"]["request_options"] == agent.request_options()
    assert manifest["stopping_rules"] == {"step_cap": 7, "timeout_s": 9,
                                          "token_budget": 11,
                                          "stall_limit": STALL_LIMIT}
    assert manifest["software"]["python"] and manifest["software"]["platform"]
    assert "pandas" in manifest["software"]["packages"]
    assert manifest["model_info"] == {"provider": "test"}
    json.dumps(manifest)  # serialisable as written to manifest.json


def test_openai_compat_agent_sends_only_the_fixed_controls(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-test")
    monkeypatch.setenv("TEST_BASE", "http://localhost:1/v1")
    agent = OpenAICompatAgent("vertex_ai/gemini-2.5-flash", seed=7,
                              provider="chattermill", api_key_env="TEST_KEY",
                              api_base_env="TEST_BASE")
    assert agent.request_options() == {"temperature": TEMPERATURE, "seed": 7,
                                       "max_tokens": MAX_OUTPUT_TOKENS}
    assert TEMPERATURE == 0.0
    digest = agent.model_digest()
    assert digest["requested_model"] == "vertex_ai/gemini-2.5-flash"
    assert digest["openai_python"]


def test_dummy_agent_reports_its_controls(tmp_path):
    agent = DummyAgent(mode="null", seed=3)
    assert agent.request_options() == {} and agent.temperature == 0.0
    start, steps, end = read_trace(run_task(agent, TASK, tmp_path))
    assert start["inference"]["request_options"] == {}
    assert steps[0]["served_model"] == "dummy-null"
    assert end["served_models"] == ["dummy-null"]
