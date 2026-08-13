"""The raw-response fallback parser: packaging repair, never call invention.

Shapes come from the qwen baseline traces: Ollama 0.32.0 dropping a well-formed
<tool_call> block, and the model writing a call as text ('-answer {...}').
"""

from cx_agent_bench.agents import _to_openai_messages, _tool_calls_from_text

NAMES = {"filter", "summarise", "rank", "ztest", "answer"}


def test_harness_transcript_translates_to_openai_format():
    harness_style = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"function": {"name": "filter",
                                      "arguments": {"industry": "Banking"}}}]},
        {"role": "tool", "tool_name": "filter", "content": '{"status": "ok"}'},
        {"role": "assistant", "content": "no call here"},
        {"role": "user", "content": '{"status": "error"}'},
        {"role": "assistant", "content": "",
         "tool_calls": [{"function": {"name": "summarise",
                                      "arguments": {"ref": "s1"}}}]},
        {"role": "tool", "tool_name": "summarise", "content": '{"status": "ok"}'},
    ]
    out = _to_openai_messages(harness_style)
    assert [m["role"] for m in out] == [m["role"] for m in harness_style]
    first_call = out[2]["tool_calls"][0]
    assert first_call["type"] == "function"
    assert first_call["function"]["arguments"] == '{"industry": "Banking"}'
    assert out[3]["tool_call_id"] == first_call["id"]          # linked reply
    assert out[7]["tool_call_id"] == out[6]["tool_calls"][0]["id"]
    assert first_call["id"] != out[6]["tool_calls"][0]["id"]   # unique per turn
    assert "tool_name" not in out[3] and "tool_calls" not in out[4]


def test_tool_call_block_is_recovered():
    text = ('<tool_call>\n{"name": "filter", "arguments": '
            '{"industry": "Banking", "aspect": ["app-website"]}}\n</tool_call>')
    [call] = _tool_calls_from_text(text, NAMES)
    assert call.name == "filter"
    assert call.args == {"industry": "Banking", "aspect": ["app-website"]}


def test_unterminated_tool_call_block_is_recovered():
    text = '<tool_call>{"name": "summarise", "arguments": {"ref": "s1"}}'
    [call] = _tool_calls_from_text(text, NAMES)
    assert call.name == "summarise" and call.args == {"ref": "s1"}


def test_bare_name_arguments_object_is_recovered():
    text = ('Sure, here is the call:\n```json\n'
            '{"name": "rank", "arguments": {"ref": "t1", "top_k": 3}}\n```')
    [call] = _tool_calls_from_text(text, NAMES)
    assert call.name == "rank" and call.args == {"ref": "t1", "top_k": 3}


def test_tool_name_before_args_object_is_recovered():
    text = '-answer {"text": "No answer. The slice is below the volume floor."}'
    [call] = _tool_calls_from_text(text, NAMES)
    assert call.name == "answer"
    assert call.args["text"].startswith("No answer.")


def test_random_dummy_shapes_answers_to_the_template():
    from cx_agent_bench.agents import DummyAgent

    def msgs(q):
        return [{"role": "system", "content": ""},
                {"role": "user", "content": q}]

    agent = DummyAgent(mode="random", seed=7)
    q3 = "What are the top 3 topics with the most number of complaints in Fashion?"
    answers = [agent.step(msgs(q3), []).tool_calls[0].args for _ in range(40)]
    abstained = [a for a in answers if a["abstain"]]
    guessed = [a for a in answers if not a["abstain"]]
    assert abstained and guessed
    assert all(a["text"] == "No answer. Insufficient data." for a in abstained)
    assert all(a["text"].startswith("The top topics are ") for a in guessed)

    q4 = ("Looking only at reviews about ease of use, is the negative "
          "sentiment rate higher in Groceries or Trading?")
    while True:
        args = agent.step(msgs(q4), []).tool_calls[0].args
        if not args["abstain"]:
            break
    assert args["text"] in ("The negative sentiment rate is higher in Groceries.",
                            "The negative sentiment rate is higher in Trading.")


def test_prose_and_empty_content_yield_nothing():
    assert _tool_calls_from_text("", NAMES) == []
    assert _tool_calls_from_text(
        "The negative rate is 21.3% (117 of 549 mentions).", NAMES) == []
    assert _tool_calls_from_text(
        "the set {negative, positive} covers most rows", NAMES) == []
