"""cx_agent_bench — baseline agent harness for the CX analytics benchmark.

Modules
-------
    canonical     shared step canonicalisation (used by the task builder AND the harness)
    tasks_public  agent-facing task loader (question, task type only)
    tasks_gold    scorer-only task loader (gold answer, gold tool path)
    tool_schemas  the fixed tool schemas every agent receives
    agents        OllamaAgent (qwen2.5:7b-instruct) and DummyAgent
    harness       the single-agent ReAct loop; writes JSONL traces, computes nothing
    scoring       reads traces from disk and computes metrics
"""

HARNESS_VERSION = "0.2.0"  # 0.2: inference / stopping-rule / served-model logging
