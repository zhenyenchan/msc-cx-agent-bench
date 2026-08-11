"""The single-agent ReAct loop (R1, R4, R6, R7, R8).

The harness owns everything except the next-action choice: it binds the fixed tool
module to the task's dataset, issues intermediate-result handles, validates every
ref argument, enforces the step cap and wall-clock timeout, and appends one JSONL
record per step as it happens. It computes no metrics (R9) and imports only the
public task loader (R3) — gold fields cannot reach a prompt from here.
"""

import importlib.util
import json
import math
import time
import uuid
from pathlib import Path

from .canonical import canonical_call
from .tasks_public import TOOLS_MODULE_PATH, file_sha256, load_dataset
from .tool_schemas import ALLOWED_PARAMS, REF_PARAMS, TOOL_NAMES, TOOL_SCHEMAS

# Step cap: ceil(2.5 x 15), the longest gold path in the task set, computed offline
# so the harness never touches the gold file at runtime (R6, R3).
STEP_CAP = math.ceil(2.5 * 15)
TIMEOUT_S = 600

SYSTEM_PROMPT = (
    "You are a data analyst answering one question about a corpus of customer "
    "reviews. Each review is labelled with industry, organisation (org), aspect "
    "(the topic it mentions) and sentiment (negative / neutral / positive).\n"
    "\n"
    "You cannot see the data directly and you cannot run code. Work only through "
    "the tools: filter selects reviews, summarise counts them and derives rates, "
    "rank orders a summary table, ztest compares the negative rates of two "
    "non-overlapping selections. Make exactly one tool call per turn, look at the "
    "observation that comes back, and decide the next call.\n"
    "\n"
    "Conventions: a rate is a share of all mentions (neutrals stay in the "
    "denominator); a slice needs at least 30 mentions to be reportable; "
    "significance is a two-sided z-test at alpha = 0.05; competitor, "
    "general-satisfaction and reviews are non-actionable topics — exclude them "
    "when recommending what an organisation should fix, but they may still be "
    "reported as findings.\n"
    "\n"
    "When you have the numbers you need, call answer with a concise final answer "
    "quoting them (rates to one decimal place, with counts). If the data cannot "
    "support the question — an empty or too-small slice, fewer qualifying groups "
    "than asked for — call answer with abstain=true and say why. The answer tool "
    "is the only way to finish."
)

# Arguments that must be integers / booleans; a 7B model sometimes sends them as
# strings, which is an argument-format fix, not reasoning done for the agent.
_INT_ARGS = {"min_n", "top_k"}
_BOOL_ARGS = {"ascending", "abstain"}


def load_tools_module():
    """Import the single fixed tools module from data/tools.py (R2)."""
    spec = importlib.util.spec_from_file_location("bench_tools", TOOLS_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def _validate(call, tools):
    """Screen a requested call before dispatch.

    Returns (error_message, error_kind, violation). Refs are checked against the
    handles the harness has actually issued; a handle the agent invents is a
    protocol violation (R4). Unknown tools and unknown arguments are tool
    failures fed back as observations (R7)."""
    if call.name not in TOOL_NAMES:
        return (f"unknown tool '{call.name}'; available tools: {', '.join(TOOL_NAMES)}",
                "unknown_tool", True)
    if not isinstance(call.args, dict):
        return (f"arguments must be an object, got {type(call.args).__name__}",
                "bad_args", False)
    unknown = set(call.args) - ALLOWED_PARAMS[call.name]
    if unknown:
        return (f"unknown argument(s) {sorted(unknown)} for {call.name}; allowed: "
                f"{sorted(ALLOWED_PARAMS[call.name])}", "bad_args", False)
    for ref_arg in REF_PARAMS.get(call.name, {}):
        if ref_arg not in call.args:
            return (f"{call.name} requires '{ref_arg}'", "bad_args", False)
        ref = call.args[ref_arg]
        if not isinstance(ref, str) or ref not in tools.store:
            issued = sorted(tools.store) or ["none yet — call filter first"]
            return (f"'{ref}' is not a handle the harness has issued; existing "
                    f"handles: {', '.join(issued)}", "bad_ref", True)
    if call.name == "answer" and "text" not in call.args:
        return ("answer requires 'text'", "bad_args", False)
    return None, None, False


def run_task(agent, task, out_dir, manifest=None, step_cap=STEP_CAP,
             timeout_s=TIMEOUT_S):
    """Run one agent on one task; write one append-only JSONL trace; return its path.

    Terminal states (R6): submitted | step_cap | timeout | error.
    """
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
    t_start = time.perf_counter()

    for idx in range(step_cap):
        if time.perf_counter() - t_start > timeout_s:
            terminal = "timeout"
            break

        t_step = time.perf_counter()
        try:
            agent_step = agent.step(messages, TOOL_SCHEMAS)
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
                  "raw_response": agent_step.raw_response}

        # -- the model produced no structured tool call ------------------------------
        if not agent_step.tool_calls:
            observation = {"status": "error",
                           "error": "no tool call in your reply; respond with exactly "
                                    "one tool call (use answer to finish)"}
            record.update(status="error", protocol_violation=True,
                          output=observation)
            emit(record)
            messages.append({"role": "assistant", "content": agent_step.content})
            messages.append({"role": "user",
                             "content": json.dumps(observation, default=str)})
            continue

        call = agent_step.tool_calls[0]
        ignored = len(agent_step.tool_calls) - 1
        call.args = _coerce_args(call.args)
        record.update(tool=call.name, args=call.args)

        # -- validate, then dispatch into the tools module (R4, R7) ------------------
        error, error_kind, violation = _validate(call, tools)
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
    }
