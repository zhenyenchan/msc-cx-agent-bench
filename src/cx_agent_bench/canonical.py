"""Shared canonicalisation of tool-call trajectories (R5).

Imported by both the task builder (run_gold_paths/gold_paths.ipynb) and the harness,
so gold paths and agent traces are reduced to the same normal form: a step is
(tool_name, sorted normalised arguments), with every ref argument expanded
recursively into the canonical signature of the step that produced it.

Two calls are equal when they did the same work to the same data, irrespective of
keyword ordering, quotation style, or handle naming.
"""

import json

# The three topics an organisation cannot fix. Mirrors tools.NON_ACTIONABLE; kept
# local so this module has no dependency on the agent-facing tools module.
NON_ACTIONABLE = ["competitor", "general-satisfaction", "reviews"]

# Arguments a tool fills in for itself. Dropping them leaves the call an agent would
# actually have to make, rather than the tool's full parameter list.
DEFAULT_ARGS = {("summarise", "min_n_column"): "total_count",
                ("rank", "ascending"): False,
                ("ztest", "alternative"): "two-sided"}

# Defaults dropped from the display signature, plus rank's top_k: 3 is the default
# and None means the whole ranking, so the two must stay distinguishable.
CANONICAL_DEFAULTS = {**DEFAULT_ARGS, ("rank", "top_k"): 3}

REF_ARGS = ("ref", "ref_a", "ref_b")            # arguments holding a handle
# Arguments naming a set of values. "aspect" is the only aspect field in this benchmark
# (the old parent_aspect/child_aspect split is gone); no call may use parent_aspect.
SET_ARGS = ("industry", "org", "aspect", "sentiment", "exclude_org", "exclude")
_UNSET = object()


def signature(call):
    """A logged call written back out as it was made: filter(industry='Banking', ...).

    Tools does not log arguments passed as None, so a rank call with no top_k in its log
    is one that asked for the whole ranking; it is restored here, since a path replayed
    without it would silently come back cut to the top 3.
    """
    tool, args = call["tool"], call["args"]
    shown = {k: v for k, v in args.items() if DEFAULT_ARGS.get((tool, k), object()) != v}
    if tool == "rank" and "top_k" not in args:
        shown["top_k"] = None
    return f"{tool}({', '.join(f'{k}={v!r}' for k, v in shown.items())})"


def _canonical_value(key, value):
    """A value stripped of everything that can differ without the meaning differing:
    3 and 3.0 agree, 'Banking' and ['Banking'] agree, and a set of values is sorted."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return int(value) if float(value).is_integer() else float(value)
    if key in SET_ARGS:
        values = value if isinstance(value, (list, tuple, set)) else [value]
        values = [str(v).strip() for v in values]
        if key == "exclude":
            # "non_actionable" is shorthand for three topics; expand it so the shorthand
            # and the spelled-out list are the same call
            values = [x for v in values
                      for x in (list(NON_ACTIONABLE) if v == "non_actionable" else [v])]
        return sorted(set(values))
    return str(value).strip()


def canonical_call(call, produced_by):
    """One call as {"tool": ..., "args": {...}}, with every handle resolved.

    A handle is a pointer, not content: an agent's 's1' is not gold's 's1', and the same
    name can sit on a completely different upstream filter. Each ref is therefore replaced
    by the canonical form of the call that produced it, recursively, so two calls are equal
    when they did the same work to the same data, whatever the intermediates were named.
    """
    tool, args = call["tool"], {}
    for key, value in call["args"].items():
        if key in REF_ARGS:
            producer = produced_by.get(value)
            args[key] = (canonical_call(producer, produced_by) if producer is not None
                         else {"unresolved_ref": value})
        elif CANONICAL_DEFAULTS.get((tool, key), _UNSET) != value:
            args[key] = _canonical_value(key, value)
    return {"tool": tool, "args": args}


def _calls(tools_or_calls):
    """Accept either a Tools instance (its .calls) or a plain list of call dicts."""
    return tools_or_calls if isinstance(tools_or_calls, list) else tools_or_calls.calls


def canonical_path(tools):
    """A whole trajectory as an ordered list of canonical calls. Run this over an agent's
    Tools log too: comparing its output with the gold one is the deterministic check."""
    calls = _calls(tools)
    produced_by = {c["result"]["id"]: c for c in calls if "id" in c.get("result", {})}
    return [canonical_call(call, produced_by) for call in calls]


def canonical_json(tools):
    """canonical_path as a stable string - keys sorted, no incidental whitespace - so
    equal paths serialise to equal text and survive a round trip through CSV."""
    return json.dumps(canonical_path(tools), sort_keys=True, separators=(",", ":"))
