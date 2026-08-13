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
    cost_usd: float | None = None   # provider-reported USD cost; None if unreported
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


def _to_openai_messages(messages):
    """Translate the harness transcript into OpenAI chat format.

    The harness records tool calls the Ollama way (arguments as a dict, tool
    replies keyed by tool_name); OpenAI-compatible endpoints need arguments as a
    JSON string and tool replies linked by tool_call_id. One call per assistant
    turn (harness protocol), so ids can be assigned by turn order.
    """
    out, call_id = [], 0
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            call_id += 1
            out.append({"role": "assistant", "content": m.get("content") or "",
                        "tool_calls": [
                            {"id": f"call_{call_id}", "type": "function",
                             "function": {
                                 "name": tc["function"]["name"],
                                 "arguments": json.dumps(tc["function"]["arguments"]),
                             }} for tc in m["tool_calls"]]})
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": f"call_{call_id}",
                        "content": m["content"]})
        else:
            out.append({"role": m["role"], "content": m.get("content") or ""})
    return out


class OpenAICompatAgent:
    """The same single-model ReAct agent as OllamaAgent, served by a hosted
    OpenAI-compatible endpoint via the openai SDK: the Chattermill data-science
    gateway (Vertex AI models, which the gateway translates server-side) or
    OpenRouter. Pass the endpoint's own model id verbatim (e.g.
    'vertex_ai/gemini-2.5-flash', 'qwen/qwen-2.5-7b-instruct'). Hosted serving
    is not pinned the way a local quantised model is; model_digest records that
    caveat (R10)."""

    def __init__(self, model, temperature=0.0, seed=42, provider="openrouter",
                 api_key_env="OPENROUTER_API_KEY", api_base_env=None,
                 agent_id=None):
        from openai import OpenAI  # imported here so other agents run without it
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise SystemExit(f"{api_key_env} not set (.env or environment)")
        api_base = os.environ.get(api_base_env) if api_base_env else None
        if api_base_env and not api_base:
            raise SystemExit(f"{api_base_env} not set (.env or environment)")
        self.endpoint = api_base or "https://openrouter.ai/api/v1"
        self._client = OpenAI(base_url=self.endpoint, api_key=api_key,
                              max_retries=3)
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.provider = provider
        slug = (model.split("/", 1)[1] if "/" in model else model)
        self.agent_id = (agent_id
                         or f"{provider}-{slug.replace('/', '-').replace(':', '-')}")

    def warmup(self):
        """Nothing to load locally. Returns False."""
        return False

    def model_digest(self):
        return {"provider": self.provider,
                "note": "hosted; served model version not pinned by the API"}

    def step(self, messages, tool_schemas):
        t0 = time.perf_counter()
        raw_http = self._client.chat.completions.with_raw_response.create(
            model=self.model,
            messages=_to_openai_messages(messages),
            tools=tool_schemas,
            temperature=self.temperature,
            seed=self.seed,
        )
        latency = time.perf_counter() - t0
        response = raw_http.parse()
        # litellm-proxy gateways report the request's actual USD cost, computed
        # against the operator's price table, in a response header.
        try:
            cost = float(raw_http.headers.get("x-litellm-response-cost"))
        except (TypeError, ValueError):
            cost = None

        message = response.choices[0].message
        calls = []
        for tc in (message.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except ValueError:
                args = None
            if isinstance(args, dict):
                calls.append(ToolCallReq(name=tc.function.name, args=args))
        content = message.content or ""
        if not calls:
            calls = _tool_calls_from_text(
                content, {t["function"]["name"] for t in tool_schemas})

        usage = getattr(response, "usage", None)
        try:
            raw = response.model_dump(mode="json")
        except Exception:
            raw = {"repr": repr(response)}
        return AgentStep(
            tool_calls=calls,
            content=content,
            tokens_in=getattr(usage, "prompt_tokens", None),
            tokens_out=getattr(usage, "completion_tokens", None),
            latency_s=round(latency, 3),
            cost_usd=cost,
            raw_response=raw,
        )


# Distinctive question stems, checked in order, so the random dummy can shape
# its guess to the task template without any model. The stems come from
# build_question_set.py's fixed templates.
_TEMPLATE_STEMS = [
    ("T2", "sentiment breakdown"),
    ("T3", "top 3 topics"),
    ("T10", "cx report"),
    ("T6", "industry average on"),
    ("T7", "biggest driver"),
    ("T8", "address first"),
    ("T9", "why does"),
    ("T5", "significantly different"),
    ("T4", "higher in"),
    ("T1", "what proportion"),
]


def _template_of(question):
    q = question.lower()
    for template, stem in _TEMPLATE_STEMS:
        if stem in q:
            return template
    return "T1"


class DummyAgent:
    """Model-free floors (R11): one answer call on the first and only step.

    null    submits an empty answer every time — a probe that the scoring
            pipeline handles missing text correctly.
    random  abstains with probability 0.5; otherwise samples an answer in the
            task template's shape — uniform rates, labels drawn from the schema
            vocabularies. Chance-level performance with the right surface form.
    """

    def __init__(self, mode="null", seed=42):
        assert mode in ("null", "random")
        self.mode = mode
        self.rng = random.Random(seed)
        self.model = f"dummy-{mode}"
        self.temperature = 0.0
        self.seed = seed
        self.endpoint = "in-process (no endpoint)"
        # each random replicate is its own agent, so 20 seeds score as a distribution
        self.agent_id = ("dummy-null" if mode == "null"
                         else f"dummy-random-s{seed}")

    def warmup(self):
        """No model to load. Returns False."""
        return False

    def model_digest(self):
        return None

    def _guess(self, question):
        from .tool_schemas import ASPECTS, INDUSTRIES, ORGS
        rng = self.rng
        if rng.random() < 0.5:
            return {"text": "No answer. Insufficient data.", "abstain": True}
        template = _template_of(question)
        if template == "T2":
            cut_a, cut_b = sorted((rng.uniform(0, 100), rng.uniform(0, 100)))
            return {"text": f"Sentiment is {cut_a:.1f}% positive, "
                            f"{cut_b - cut_a:.1f}% negative and "
                            f"{100 - cut_b:.1f}% neutral.", "abstain": False}
        if template == "T3":
            picks = rng.sample(ASPECTS, rng.choice([1, 3]))
            return {"text": "The top topics are " + ", ".join(picks) + ".",
                    "abstain": False}
        if template == "T4":
            named = re.search(r"higher in (.+?) or (.+?)\?", question)
            side = rng.choice([named.group(1), named.group(2)] if named
                              else INDUSTRIES)
            return {"text": f"The negative sentiment rate is higher in {side}.",
                    "abstain": False}
        if template == "T5":
            significant = rng.choice([True, False])
            return {"text": "The difference is "
                            f"{'' if significant else 'not '}significant.",
                    "abstain": False}
        if template == "T6":
            listed = ", ".join(f"{org} ({rng.uniform(0, 100):.1f}%)"
                               for org in rng.sample(ORGS, 3))
            return {"text": f"Against an industry average of "
                            f"{rng.uniform(0, 100):.1f}%, the organisations "
                            f"compare as follows: {listed}.", "abstain": False}
        if template == "T7":
            return {"text": f"The biggest driver of dissatisfaction is "
                            f"{rng.choice(ASPECTS)} ({rng.uniform(0, 100):.1f}% "
                            "negative).", "abstain": False}
        if template == "T8":
            first, second = rng.sample(ASPECTS, 2)
            return {"text": f"Address {first} first, followed by {second}.",
                    "abstain": False}
        if template == "T9":
            return {"text": f"The higher complaint rate is driven by "
                            f"{rng.choice(ASPECTS)}; improve that first.",
                    "abstain": False}
        if template == "T10":
            top = rng.choice(ASPECTS)
            return {"text": f"CX report: overall sentiment is "
                            f"{rng.uniform(0, 100):.1f}% negative. The top issue "
                            f"is {top} ({rng.uniform(0, 100):.1f}% negative). "
                            f"Roadmap: address {top} first.", "abstain": False}
        return {"text": f"The rate is {rng.uniform(0, 100):.1f}%.",
                "abstain": False}

    def step(self, messages, tool_schemas):
        if self.mode == "null":
            args = {"text": ""}
        else:
            question = next((m["content"] for m in messages
                             if m["role"] == "user"), "")
            args = self._guess(question)
        return AgentStep(
            tool_calls=[ToolCallReq(name="answer", args=args)],
            tokens_in=0, tokens_out=0, latency_s=0.0,
            raw_response={"dummy": self.mode},
        )
