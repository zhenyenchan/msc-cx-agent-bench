"""Run the benchmark agent through the Chattermill gateway (Vertex AI models).

    python -m cx_agent_bench.run_agent                                # gemini-2.5-flash, all 50 tasks
    python -m cx_agent_bench.run_agent --tasks T1-E1 T4-H2
    python -m cx_agent_bench.run_agent --limit 3
    python -m cx_agent_bench.run_agent --model vertex_ai/gemini-2.5-pro

The harness, tools, schemas, traces and reports are identical to run_baseline.py
(everything shared through run_suite); only the serving endpoint differs. Needs
OPENAI_BASE_URL and OPENAI_API_KEY in .env, as in cx_agent_poc/llm.py.
Score afterwards with
    python -m cx_agent_bench.scoring <trace_dir> --out scores.csv
"""

import argparse
import time
from pathlib import Path

from .agents import OpenAICompatAgent
from .harness import STEP_CAP, TIMEOUT_S
from .run_baseline import run_suite, select_tasks
from .tasks_public import REPO_ROOT
from .tool_schemas import TOOL_SCHEMAS

GATEWAY_DEFAULT_MODEL = "vertex_ai/gemini-2.5-flash"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=GATEWAY_DEFAULT_MODEL,
                        help=f"gateway model id (default: {GATEWAY_DEFAULT_MODEL})")
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="task ids to run (default: all)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--step-cap", type=int, default=STEP_CAP)
    parser.add_argument("--timeout", type=int, default=TIMEOUT_S)
    parser.add_argument("--no-tools", action="store_true",
                        help="offer only the answer tool: the prior-knowledge "
                             "floor (no data access), same prompt and step cap")
    parser.add_argument("--out", default=None,
                        help="trace directory (default: traces/gateway/<agent>_<ts>)")
    args = parser.parse_args()

    # short name for trace files and directories: 'vertex_ai/gemini-2.5-flash'
    # runs as traces/vertex_ai/gemini-2.5-flash_<ts>/gemini-2.5-flash__<task>__..
    short_name = args.model.split("/")[-1]
    if args.no_tools:
        short_name = f"no-tool-{short_name}"
    provider_dir = args.model.split("/")[0] if "/" in args.model else "gateway"
    schemas = ([s for s in TOOL_SCHEMAS if s["function"]["name"] == "answer"]
               if args.no_tools else TOOL_SCHEMAS)
    agent = OpenAICompatAgent(model=args.model, temperature=0.0,
                              seed=args.seed, provider="chattermill",
                              api_key_env="OPENAI_API_KEY",
                              api_base_env="OPENAI_BASE_URL",
                              agent_id=short_name)
    tasks = select_tasks(args.tasks, args.limit)
    out_dir = Path(args.out) if args.out else (
        REPO_ROOT / "traces" / provider_dir
        / f"{short_name}_{time.strftime('%Y%m%d_%H%M%S')}")
    if tasks:
        run_suite(agent, tasks, out_dir, args.step_cap, args.timeout,
                  tool_schemas=schemas)


if __name__ == "__main__":
    main()
