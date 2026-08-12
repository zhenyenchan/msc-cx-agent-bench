"""The agents (R1, R11).

Every agent exposes the same interface: `step(messages, tool_schemas)` returns an
AgentStep with zero or more structured tool calls. The harness owns the loop, the
tools and the trace; an agent only maps a message history to the next action.

    OllamaAgent  — the baseline under evaluation: one local LLM using Ollama's
                   native function-calling interface (no free-form code execution).
    DummyAgent   — a trivial floor: answers immediately with a null or random
                   canned answer, through the identical interface (R11).
"""

import json
import os
import random
import re
import time
from dataclasses import dataclass, field


@dataclass
class ToolCallReq:
    """A structured tool call as requested by the agent: machine-readable name + args."""
    name: str
    args: dict


@dataclass
class AgentStep:
    """One agent turn: what came back from the model, plus its cost."""
    tool_calls: list          # list[ToolCallReq]; empty if the model produced no call
    content: str = ""         # any free text alongside (or instead of) the call
    tokens_in: int | None = None
    tokens_out: int | None = None
    latency_s: float | None = None
    raw_response: dict = field(default_factory=dict)


def _tool_calls_from_text(content, tool_names):
    """Salvage a structured call the server dropped or the model wrote as text.

    Ollama 0.32.0 sometimes fails to parse a well-formed <tool_call> block out of
    the model's output, and a small model sometimes writes the call as text
    (`answer {"text": ...}`). Recovering these is argument-format plumbing in the
    spirit of harness._coerce_args: the call itself is entirely the model's; only
    its packaging is repaired. Returns [] when no call can be read.
    """
    if not content:
        return []
    decoder = json.JSONDecoder()

    def first_object(text):
        for i, ch in enumerate(text):
            if ch != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(text[i:])
            except ValueError:
                continue
            if isinstance(obj, dict):
                return obj
        return None

    def as_call(obj):
        if (isinstance(obj, dict) and isinstance(obj.get("name"), str)
                and isinstance(obj.get("arguments"), dict)):
            return [ToolCallReq(obj["name"], obj["arguments"])]
        return None

    # 1. an explicit <tool_call> block; the harness validates the name it claims
    block = re.search(r"<tool_call>(.*?)(?:</tool_call>|$)", content, re.S)
    if block:
        call = as_call(first_object(block.group(1)))
        if call:
            return call

    # 2. a bare {"name": ..., "arguments": {...}} object anywhere in the text
    call = as_call(first_object(content))
    if call:
        return call

    # 3. a known tool name written directly before its argument object,
    #    e.g. `-answer {"text": "No answer. ..."}`
    for m in re.finditer(r"([A-Za-z_]\w*)\s*\{", content):
        if m.group(1) in tool_names:
            obj = first_object(content[m.end(1):])
            if isinstance(obj, dict) and "name" not in obj:
                return [ToolCallReq(m.group(1), obj)]
    return []


class OllamaAgent:
    """Single-model ReAct agent over Ollama's structured tool-calling API."""

    def __init__(self, model="qwen2.5:7b-instruct", temperature=0.0, seed=42,
                 num_ctx=16384):
        import ollama  # imported here so DummyAgent runs without the dependency
        self._ollama = ollama
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.num_ctx = num_ctx
        self.endpoint = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.agent_id = f"ollama-{model.replace(':', '-').replace('/', '-')}"

    def warmup(self):
        """Load the model into Ollama's memory before the first task, so model-load
        time is not billed to the first run. Returns True: a model was loaded."""
        self._ollama.generate(model=self.model, prompt="")  # empty prompt = load only
        return True

    def model_digest(self):
        """Pinned model version string for the manifest (R10)."""
        try:
            info = self._ollama.show(self.model)
            digest = (getattr(info, "modelinfo", None) or {}).get("general.basename")
            details = getattr(info, "details", None)
            return {"parameter_size": getattr(details, "parameter_size", None),
                    "quantization": getattr(details, "quantization_level", None),
                    "basename": digest}
        except Exception:
            return None

    def step(self, messages, tool_schemas):
        t0 = time.perf_counter()
        response = self._ollama.chat(
            model=self.model,
            messages=messages,
            tools=tool_schemas,
            options={"temperature": self.temperature, "seed": self.seed,
                     "num_ctx": self.num_ctx},
        )
        latency = time.perf_counter() - t0

        calls = [ToolCallReq(name=tc.function.name, args=dict(tc.function.arguments))
                 for tc in (response.message.tool_calls or [])]
        if not calls:
            calls = _tool_calls_from_text(
                response.message.content or "",
                {t["function"]["name"] for t in tool_schemas})
        try:
            raw = response.model_dump(mode="json")
        except Exception:
            raw = {"repr": repr(response)}
        return AgentStep(
            tool_calls=calls,
            content=response.message.content or "",
            tokens_in=getattr(response, "prompt_eval_count", None),
            tokens_out=getattr(response, "eval_count", None),
            latency_s=round(latency, 3),
            raw_response=raw,
        )


class DummyAgent:
    """Null/random baseline (R11): submits an answer on its first and only step."""

    CANNED = [
        "The negative rate is 50.0%.",
        "The top topic is price-value-for-money.",
        "There is no significant difference between the two groups.",
        "The most negative aspect is speed, at 33.3%.",
    ]

    def __init__(self, mode="null", seed=42):
        assert mode in ("null", "random")
        self.mode = mode
        self.rng = random.Random(seed)
        self.model = f"dummy-{mode}"
        self.temperature = 0.0
        self.seed = seed
        self.endpoint = "in-process (no endpoint)"
        self.agent_id = f"dummy-{mode}"

    def warmup(self):
        """No model to load. Returns False."""
        return False

    def model_digest(self):
        return None

    def step(self, messages, tool_schemas):
        text = "" if self.mode == "null" else self.rng.choice(self.CANNED)
        return AgentStep(
            tool_calls=[ToolCallReq(name="answer", args={"text": text})],
            tokens_in=0, tokens_out=0, latency_s=0.0,
            raw_response={"dummy": self.mode},
        )
