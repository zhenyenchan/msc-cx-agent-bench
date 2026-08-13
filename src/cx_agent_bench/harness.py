"""The single-agent ReAct loop (R1, R4, R6, R7, R8).

The harness owns everything except the next-action choice: it binds the fixed tool
module to the task's dataset, issues intermediate-result handles, validates every
ref argument, enforces the step cap and wall-clock timeout, and appends one JSONL
record per step as it happens. It computes no metrics (R9) and imports only the
public task loader (R3) — gold fields cannot reach a prompt from here.
"""

import json
import math
import time
import uuid
from pathlib import Path

from .canonical import canonical_call
from .tasks_public import TOOLS_MODULE_PATH, file_sha256, load_dataset
from .tool_schemas import (ALLOWED_PARAMS, PARAM_SPECS, REF_PARAMS, TOOL_NAMES,
                           TOOL_SCHEMAS)

# Step cap: ceil(2.5 x 15), the longest gold path in the task set, computed offline
# so the harness never touches the gold file at runtime (R6, R3).
STEP_CAP = math.ceil(2.5 * 15)
TIMEOUT_S = 600
STALL_LIMIT = 5  # consecutive identical no-tool-call replies before the run ends

# What each REF_PARAMS kind prefix means, for corrective bad_ref messages.
_HANDLE_KINDS = {"s": ("selection", "filter"), "t": ("table", "summarise")}

_JSON_TYPES = {"string": str, "integer": int, "boolean": bool,
               "array": list, "null": type(None)}


def _spec_problem(value, spec):
    """Why a value contradicts its schema property (type / enum / anyOf), or None.

    This mirrors exactly what the agent was shown in TOOL_SCHEMAS, so nothing is
    rejected here that the documentation allowed; without it a mistyped argument
    (group_by=['aspect'], top_k='all') crashes inside pandas with an opaque message.
    """
    if "anyOf" in spec:
        if any(_spec_problem(value, branch) is None for branch in spec["anyOf"]):
            return None
        kinds = " or ".join(b.get("type", "?") for b in spec["anyOf"])
        return f"must be a {kinds}, got {type(value).__name__} ({value!r})"
    declared = spec.get("type")
    if declared:
        types = declared if isinstance(declared, list) else [declared]
        matched = any(isinstance(value, _JSON_TYPES[t])
                      and not (t == "integer" and isinstance(value, bool))
                      for t in types)
        if not matched:
            return (f"must be of type {' or '.join(types)}, got "
                    f"{type(value).__name__} ({value!r})")
        if isinstance(value, list) and "items" in spec:
            for entry in value:
                problem = _spec_problem(entry, spec["items"])
                if problem:
                    return f"has an invalid entry: {problem}"
    if "enum" in spec and value not in spec["enum"]:
        return f"must be one of {spec['enum']}, got {value!r}"
    return None

ROLE = (
    "You are a data analyst answering one question about a dataset of customer "
    "reviews. The dataset holds labels only: each row records the industry, "
    "organisation (org), aspect (the topic mentioned) and sentiment "
    "(negative / neutral / positive) of one review. The review text is not in "
    "the data, so every finding comes from counts, rates and comparisons over these labels."
)

PROTOCOL = (
    "You cannot write or run code. All analysis is done through the tools "
    "provided. Every turn must be exactly one tool call, with no accompanying "
    "prose, then read the observation and decide the next call. Text outside a "
    "tool call is discarded and never reaches the user, so a final answer written "
    "as plain text is lost. Submit the final answer with the answer tool, which is the only way "
    "to finish. A failed call returns an observation with status 'error' and the "
    "information needed to correct it - fix the call and continue."
)

POLICY = (
    "Questions are phrased in natural language while the data uses fixed labels; "
    "map the question onto the field values listed in the tool descriptions. "
    "'Complaints' and 'issues' mean negative sentiment; 'praise' means positive.\n"
    "\n"
    "Report figures inline in the final answer, rates to one decimal place with "
    "counts. For example: 'The top complaints in Banking are app or website, "
    "(123 mentions), staff attitude (234 mentions) and ease of use (345 mentions).'\n"
    "\n"
    "A slice needs at least 30 mentions to be reported reliably. Below that, say so "
    "rather than reporting the figure. Competitor, general-satisfaction and reviews "
    "are non-actionable topics - exclude them when recommending what an organisation "
    "should fix, though they may still be reported as findings."
)

SYSTEM_PROMPT = f"{ROLE}\n\n{PROTOCOL}\n\n{POLICY}"

# Arguments that must be integers / booleans; a 7B model sometimes sends them as
# strings, which is an argument-format fix, not reasoning done for the agent.
_INT_ARGS = {"min_n", "top_k"}
_BOOL_ARGS = {"ascending", "abstain"}


def load_tools_module():
    """The single fixed tools module (R2), shipped inside the package."""
    from . import tools
    return tools


def build_messages(task):
    """The full agent-facing prompt for a task: system + question. Serialising this
    (plus TOOL_SCHEMAS) is everything the agent process ever receives (R3)."""
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task["question"]}]


def _coerce_args(args):
    """Fix argument types, not argument choices: '3' -> 3, 'true' -> True."""
    out = {}
    for key, value in args.items():
        if key in _INT_ARGS and isinstance(value, str) and value.strip().lstrip("-").isdigit():
            value = int(value.strip())
        elif key in _INT_ARGS and isinstance(value, float) and value.is_integer():
            value = int(value)
        elif key in _BOOL_ARGS and isinstance(value, str):
            if value.strip().lower() in ("true", "false"):
                value = value.strip().lower() == "true"
        if isinstance(value, str) and value.strip().lower() in ("none", "null"):
            value = None
        out[key] = value
    return out


def _validate(call, tools, offered=TOOL_NAMES):
    """Screen a requested call before dispatch.

    Returns (error_message, error_kind, violation). Only two failures are
    protocol violations (R4): a reply with no tool call (handled in the loop)
    and an unknown tool — including a real tool that was not offered to this
    run (the no-tool ablation offers answer only). Everything else — unknown or
    mistyped arguments, invented or wrong-kind handles — is a genuine agent
    mistake, fed back as a corrective observation (R7)."""
    if call.name not in offered:
        return (f"unknown tool '{call.name}'; available tools: {', '.join(offered)}",
                "unknown_tool", True)
    if not isinstance(call.args, dict):
        return (f"arguments must be an object, got {type(call.args).__name__}",
                "bad_args", False)
    unknown = set(call.args) - ALLOWED_PARAMS[call.name]
    if unknown:
        return (f"unknown argument(s) {sorted(unknown)} for {call.name}; allowed: "
                f"{sorted(ALLOWED_PARAMS[call.name])}", "bad_args", False)
    for param, value in call.args.items():
        if value is None:
            continue  # an explicit null means unset, like omitting the argument
        problem = _spec_problem(value, PARAM_SPECS[call.name][param])
        if problem:
            return (f"argument '{param}' of {call.name} {problem}",
                    "bad_args", False)
    for ref_arg, kind in REF_PARAMS.get(call.name, {}).items():
        if ref_arg not in call.args:
            return (f"{call.name} requires '{ref_arg}'", "bad_args", False)
        ref = call.args[ref_arg]
        if not isinstance(ref, str) or ref not in tools.store:
            issued = sorted(tools.store) or ["none yet — call filter first"]
            return (f"'{ref}' is not a handle the harness has issued. Handles are "
                    "opaque ids that cannot be sliced or subscripted; to work on "
                    "one group, build it as its own selection with filter. Existing "
                    f"handles: {', '.join(issued)}", "bad_ref", False)
        if not ref.startswith(kind):
            noun, producer = _HANDLE_KINDS[kind]
            have_noun, _ = _HANDLE_KINDS[ref[0]]
            return (f"{ref_arg} must be a {noun} handle produced by {producer}, "
                    f"like '{kind}1'; '{ref}' is a {have_noun} handle",
                    "bad_ref", False)
    if call.name == "answer" and "text" not in call.args:
        return ("answer requires 'text'", "bad_args", False)
    return None, None, False


def run_task(agent, task, out_dir, manifest=None, step_cap=STEP_CAP,
             timeout_s=TIMEOUT_S, tool_schemas=TOOL_SCHEMAS):
    """Run one agent on one task; write one append-only JSONL trace; return its path.

    tool_schemas is the tool set offered to the agent this run — the full five
    by default; the no-tool ablation passes the answer schema alone. Calls to
    tools outside the offered set are rejected as unknown.

    Terminal states (R6): submitted | stalled | step_cap | timeout | error.
    """
    offered = [schema["function"]["name"] for schema in tool_schemas]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{agent.agent_id}__{task['task_id']}__{uuid.uuid4().hex[:8]}"
    trace_path = out_dir / f"{run_id}.jsonl"

    tools_module = load_tools_module()
    tools = tools_module.Tools(load_dataset(task["dataset"]))
    messages = build_messages(task)

    def emit(record):
        with trace_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    emit({"type": "run_start", "run_id": run_id, "task_id": task["task_id"],
          "agent_id": agent.agent_id, "model": agent.model,
          "temperature": agent.temperature, "seed": agent.seed,
          "step_cap": step_cap, "timeout_s": timeout_s,
          "manifest": manifest, "ts": time.time()})

    terminal, error_note = None, None
    stall_content, stall_count = None, 0
    t_start = time.perf_counter()

    for idx in range(step_cap):
        if time.perf_counter() - t_start > timeout_s:
            terminal = "timeout"
            break

        t_step = time.perf_counter()
        try:
            agent_step = agent.step(messages, tool_schemas)
        except Exception as exc:  # provider/agent failure: the run ends, recorded
            terminal, error_note = "error", f"{type(exc).__name__}: {exc}"
            break

        step_latency = round(time.perf_counter() - t_step, 3)
        record = {"type": "step", "idx": idx, "tool": None, "args": None,
                  "handle": None, "canonical_signature": None, "output": None,
                  "status": None, "protocol_violation": False,
                  "tokens_in": agent_step.tokens_in,
                  "tokens_out": agent_step.tokens_out,
                  "latency_s": step_latency,
                  "cost_usd": agent_step.cost_usd,
                  "raw_response": agent_step.raw_response}

        # -- the model produced no structured tool call ------------------------------
        if not agent_step.tool_calls:
            # A repeat of the same reply gets escalating feedback; STALL_LIMIT
            # identical replies in a row end the run, because at temperature 0 an
            # unchanged reply to a barely-changed prompt will not change again.
            stall_count = (stall_count + 1
                           if agent_step.content == stall_content else 1)
            stall_content = agent_step.content
            if stall_count == 1:
                error_msg = ("no tool call in your reply; respond with exactly "
                             "one tool call (use answer to finish)")
            else:
                error_msg = (f"no tool call in your reply, {stall_count} times in "
                             "a row. If the text you just wrote is your final "
                             "answer, pass it to the answer tool as 'text'; "
                             "otherwise respond with exactly one tool call.")
            observation = {"status": "error", "error": error_msg}
            record.update(status="error", protocol_violation=True,
                          output=observation)
            emit(record)
            messages.append({"role": "assistant", "content": agent_step.content})
            messages.append({"role": "user",
                             "content": json.dumps(observation, default=str)})
            if stall_count >= STALL_LIMIT:
                terminal = "stalled"
                error_note = (f"{stall_count} consecutive identical replies "
                              "without a tool call")
                break
            continue

        stall_content, stall_count = None, 0
        call = agent_step.tool_calls[0]
        ignored = len(agent_step.tool_calls) - 1
        call.args = _coerce_args(call.args)
        record.update(tool=call.name, args=call.args)

        # -- validate, then dispatch into the tools module (R4, R7) ------------------
        error, error_kind, violation = _validate(call, tools, offered)
        if error is None:
            try:
                result = getattr(tools, call.name)(**call.args)
            except Exception as exc:
                result = {"error": f"{call.name} failed: {type(exc).__name__}: {exc}"}
                error_kind = "tool_exception"
        else:
            result = {"error": error, "error_kind": error_kind}

        failed = "error" in result
        observation = {"status": "error" if failed else "ok", **result}
        if ignored:
            observation["note"] = (f"{ignored} extra tool call(s) ignored; one call "
                                   "per turn")

        handle = result.get("id")
        produced_by = {c["result"]["id"]: c for c in tools.calls
                       if "id" in c.get("result", {})}
        record.update(
            handle=handle,
            canonical_signature=(canonical_call({"tool": call.name, "args": call.args},
                                                produced_by)
                                 if error is None else None),
            output=observation,
            status="error" if failed else "ok",
            protocol_violation=violation,
            error_kind=error_kind,
        )
        emit(record)

        # -- feed the observation back; answer is the single legal exit (R6) ---------
        messages.append({"role": "assistant", "content": agent_step.content,
                         "tool_calls": [{"function": {"name": call.name,
                                                      "arguments": call.args}}]})
        messages.append({"role": "tool", "tool_name": call.name,
                         "content": json.dumps(observation, default=str)})

        if call.name == "answer" and not failed:
            terminal = "submitted"
            break
    else:
        terminal = "step_cap"

    emit({"type": "run_end", "run_id": run_id, "terminal_state": terminal,
          "error": error_note, "final_answer": tools.final,
          "n_tool_calls": len(tools.calls),
          "wall_s": round(time.perf_counter() - t_start, 3), "ts": time.time()})
    return trace_path


# Public list prices, USD per 1M tokens, recorded into the manifest at run time
# (R10) so scoring can compute a reproducible list-price cost per task. Cached
# input is the provider's cache-read rate (Vertex implicit caching). Extend as
# the experiment's model roster grows.
LIST_PRICES = {
    "vertex_ai/gemini-2.5-flash": {"input": 0.30, "cached_input": 0.03,
                                   "output": 2.50},
}


def list_price_cost(steps, prices):
    """Total list-price USD cost over step records, or None without a price row.

    Per step: (input - cached) * p_in + cached * p_cached + output * p_out,
    with cached-token counts taken from the provider's usage block as stored in
    raw_response. Validated against the gateway's reported cost (exact match).
    """
    if not prices:
        return None
    total = 0.0
    for s in steps:
        raw = s.get("raw_response")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except ValueError:
                raw = {}
        usage = (raw.get("usage") or {}) if isinstance(raw, dict) else {}
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        tokens_in = s.get("tokens_in") or 0
        tokens_out = s.get("tokens_out") or 0
        total += ((tokens_in - cached) * prices["input"]
                  + cached * prices["cached_input"]
                  + tokens_out * prices["output"]) / 1e6
    return round(total, 6)


def build_manifest(agent):
    """Provenance for a batch of runs (R10): corpus hashes, tool module hash,
    generator commit, pinned model info."""
    import subprocess

    from .tasks_public import DATASET_CSVS, TASKS_PUBLIC_CSV
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"],
                                capture_output=True, text=True,
                                cwd=TOOLS_MODULE_PATH.parent).stdout.strip() or None
    except Exception:
        commit = None
    from . import HARNESS_VERSION
    return {
        "harness_version": HARNESS_VERSION,
        "generator_commit": commit,
        "tasks_public_sha256": file_sha256(TASKS_PUBLIC_CSV),
        "corpus_sha256": {name: file_sha256(path)
                          for name, path in DATASET_CSVS.items()},
        "tools_module_sha256": file_sha256(TOOLS_MODULE_PATH),
        "model": agent.model,
        "model_info": agent.model_digest(),
        "temperature": agent.temperature,
        "seed": agent.seed,
        "list_prices_usd_per_1m": LIST_PRICES.get(agent.model),
    }
