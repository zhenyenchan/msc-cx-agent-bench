"""The five-model comparison roster: one architecture, per-model API controls."""

import pytest

from cx_agent_bench.harness import LIST_PRICES, inference_config
from cx_agent_bench.run_agent import PROVIDERS, ROSTER, make_hosted_agent


@pytest.fixture(autouse=True)
def fake_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1/v1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")


def test_roster_has_the_five_models_on_both_routes():
    assert set(ROSTER) == {"gpt-5.6-terra", "gpt-5.6-sol", "claude-fable-5",
                           "claude-sonnet-5", "deepseek-v3.2"}
    for entry in ROSTER.values():
        assert set(PROVIDERS) <= set(entry)
        assert isinstance(entry["temperature"], bool)
        assert isinstance(entry["seed"], bool)


def test_every_roster_model_has_a_list_price_on_every_route():
    for entry in ROSTER.values():
        for provider in PROVIDERS:
            assert entry[provider] in LIST_PRICES, entry[provider]


def test_gateway_is_the_default_route():
    agent = make_hosted_agent("gpt-5.6-terra", seed=2555)
    assert agent.provider == "chattermill"
    assert agent.model == "gpt-5.6-terra"
    assert agent.endpoint == "http://localhost:1/v1"
    assert agent.request_options() == {"temperature": 0.0, "seed": 2555,
                                       "max_tokens": 4096,
                                       "reasoning_effort": "none"}
    assert any("reasoning_effort" in n for n in agent.control_notes)


def test_non_gpt_models_send_no_reasoning_effort():
    for name in ("claude-sonnet-5", "claude-fable-5", "deepseek-v3.2"):
        agent = make_hosted_agent(name, seed=1)
        assert "reasoning_effort" not in agent.request_options()


def test_claude_models_do_not_send_temperature_or_seed():
    agent = make_hosted_agent("claude-sonnet-5", seed=2555)
    assert agent.request_options() == {"max_tokens": 4096}
    assert agent.seed == 2555  # still labels the replicate
    notes = inference_config(agent)["control_notes"]
    assert any("temperature not sent" in n for n in notes)
    assert any("seed not sent" in n for n in notes)


def test_openrouter_fallback_route_uses_openrouter_ids():
    agent = make_hosted_agent("deepseek-v3.2", provider="openrouter", seed=1)
    assert agent.model == "deepseek/deepseek-v3.2"
    assert "openrouter.ai" in agent.endpoint


def test_raw_model_id_keeps_full_controls():
    agent = make_hosted_agent(None, model="vertex_ai/gemini-2.5-flash", seed=3)
    assert agent.model == "vertex_ai/gemini-2.5-flash"
    assert agent.request_options() == {"temperature": 0.0, "seed": 3,
                                       "max_tokens": 4096}
