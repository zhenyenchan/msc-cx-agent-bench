"""Run the benchmark agent on a hosted model: the five-model comparison roster
via OpenRouter or the Chattermill gateway, or any raw gateway model id.

    python -m cx_agent_bench.run_agent --agent gpt-5.6-terra --all-seeds   # the comparison
    python -m cx_agent_bench.run_agent --agent deepseek-v3.2 --all-seeds
    python -m cx_agent_bench.run_agent --agent claude-sonnet-5 --tasks T1-E1 T4-H2
    python -m cx_agent_bench.run_agent --model vertex_ai/gemini-2.5-flash       # raw gateway id
    python -m cx_agent_bench.run_agent --agent gpt-5.6-terra --provider openrouter  # fallback route

Every agent shares the same architecture (harness, tools, schemas, prompt, step
cap, timeout, token budget, seeds); only the model differs. The roster fixes
per model which inference controls its API accepts. The harness, traces and
reports are identical to run_baseline.py (everything shared through run_suite).
The default route is the Chattermill Vertex AI gateway (OPENAI_BASE_URL +
OPENAI_API_KEY in .env, as in notebooks/tests/test-api.ipynb); OpenRouter
(OPENROUTER_API_KEY) is a fallback route only. When the run finishes, scoring runs automatically: the run dir gets
its scores.csv and benchmark_outputs/results.csv is refreshed.
"""

import argparse
import time
from pathlib import Path

from .agents import OpenAICompatAgent
from .harness import STEP_CAP, TIMEOUT_S, TOKEN_BUDGET
from .run_baseline import SEEDS_PATH, load_seeds, run_suite, select_tasks
from .tasks_public import REPO_ROOT
from .tool_schemas import TOOL_SCHEMAS

GATEWAY_DEFAULT_MODEL = "vertex_ai/gemini-2.5-flash"

# The comparison roster: one agent architecture, five models. Per model: the id
# on each route, and which controls the API accepts. The Claude 5 family rejects
# temperature ("deprecated for this model") and has no seed parameter; GPT-5.6
# accepts both (OpenRouter's listing omits temperature, so it may be ignored
# upstream - the manifest records the listing). GPT-5.6 on /v1/chat/completions
# rejects function tools unless reasoning_effort is 'none' (400, probed
# 2026-09-04), so that is a required setting for this architecture: the model
# runs without its reasoning phase. Probed 2026-09-03/04.
GPT56_REQUIRED = {
    "extra": {"reasoning_effort": "none"},
    "note": ("reasoning_effort='none' sent: the chat completions endpoint "
             "rejects function tools at any other value for GPT-5.6, so the "
             "model runs without its reasoning phase"),
}
ROSTER = {
    "gpt-5.6-terra": {"openrouter": "openai/gpt-5.6-terra",
                      "chattermill": "gpt-5.6-terra",
                      "temperature": True, "seed": True, **GPT56_REQUIRED},
    "gpt-5.6-sol": {"openrouter": "openai/gpt-5.6-sol",
                    "chattermill": "gpt-5.6-sol",
                    "temperature": True, "seed": True, **GPT56_REQUIRED},
    "claude-fable-5": {"openrouter": "anthropic/claude-fable-5",
                       "chattermill": "claude-fable-5",
                       "temperature": False, "seed": False},
    "claude-sonnet-5": {"openrouter": "anthropic/claude-sonnet-5",
                        "chattermill": "claude-sonnet-5",
                        "temperature": False, "seed": False},
    "deepseek-v3.2": {"openrouter": "deepseek/deepseek-v3.2",
                      "chattermill": "vertex_ai/deepseek-ai/deepseek-v3.2-maas",
                      "temperature": True, "seed": True},
}

PROVIDERS = {
    "openrouter": {"api_key_env": "OPENROUTER_API_KEY", "api_base_env": None},
    "chattermill": {"api_key_env": "OPENAI_API_KEY",
                    "api_base_env": "OPENAI_BASE_URL"},
}


def make_hosted_agent(name=None, provider="chattermill", model=None, seed=42,
                      agent_id=None):
    """A roster agent (name) on a provider, or a raw model id on a provider."""
    if name:
        entry = ROSTER[name]
        model = entry[provider]
        controls = {"send_temperature": entry["temperature"],
                    "send_seed": entry["seed"],
                    "extra_options": entry.get("extra"),
                    "extra_note": entry.get("note")}
    else:
        controls = {}
    return OpenAICompatAgent(model=model, seed=seed, provider=provider,
                             agent_id=agent_id, **PROVIDERS[provider],
                             **controls)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=sorted(ROSTER), default=None,
                        help="a comparison-roster model (overrides --model)")
    parser.add_argument("--provider", choices=sorted(PROVIDERS),
                        default="chattermill",
                        help="serving route (default: the Chattermill gateway)")
    parser.add_argument("--model", default=GATEWAY_DEFAULT_MODEL,
                        help=f"raw model id (default: {GATEWAY_DEFAULT_MODEL})")
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="task ids to run (default: all)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all-seeds", action="store_true",
                        help=f"run the suite once per frozen seed in {SEEDS_PATH.name} "
                             "(one trace directory per seed, suffixed _s<seed>)")
    parser.add_argument("--step-cap", type=int, default=STEP_CAP)
    parser.add_argument("--timeout", type=int, default=TIMEOUT_S)
    parser.add_argument("--token-budget", type=int, default=TOKEN_BUDGET,
                        help="cumulative prompt+completion tokens per task")
    parser.add_argument("--no-tools", action="store_true",
                        help="offer only the answer tool: the prior-knowledge "
                             "floor (no data access), same prompt and step cap")
    parser.add_argument("--out", default=None,
                        help="trace directory (default: traces/gateway/<agent>_<ts>)")
    args = parser.parse_args()

    provider = args.provider
    if args.agent:
        # roster runs: traces/<provider>/<agent>_<ts>[_s<seed>]/<agent>__<task>__..
        short_name, provider_dir = args.agent, provider
    else:
        # raw ids: 'vertex_ai/gemini-2.5-flash' runs as
        # traces/vertex_ai/gemini-2.5-flash_<ts>/gemini-2.5-flash__<task>__..
        short_name = args.model.split("/")[-1]
        provider_dir = args.model.split("/")[0] if "/" in args.model else "gateway"
    if args.no_tools:
        short_name = f"no-tool-{short_name}"
    schemas = ([s for s in TOOL_SCHEMAS if s["function"]["name"] == "answer"]
               if args.no_tools else TOOL_SCHEMAS)
    tasks = select_tasks(args.tasks, args.limit)
    if not tasks:
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for seed in (load_seeds() if args.all_seeds else [args.seed]):
        agent = make_hosted_agent(args.agent, provider, model=args.model,
                                  seed=seed, agent_id=short_name)
        out_dir = Path(args.out) if args.out else (
            REPO_ROOT / "traces" / provider_dir / f"{short_name}_{stamp}")
        if args.all_seeds:
            out_dir = out_dir.with_name(f"{out_dir.name}_s{seed}")
        run_suite(agent, tasks, out_dir, args.step_cap, args.timeout,
                  tool_schemas=schemas, token_budget=args.token_budget)


if __name__ == "__main__":
    main()
