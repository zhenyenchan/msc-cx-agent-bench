# msc-cx-agent-bench
UCL MSc Project - Benchmarking Agentic Systems for Customer Feedback Analytics 

## Proof of concept

What I've built:
- A minimal end-to-end CX analytics agent
- Dataset: FABSA (10500+ reviews, 12 aspect categories, 3 sentiment classes, 10 industries).
- LLM: Ollama qwen2.5:7b-instruct
- Architecture: ReAct from scratch
- The agent answers 3 types of questions (descriptive, inferential and reporting) by routing them through a small set of tools.

## Architecture

```mermaid
flowchart TD
    User([User question]) --> Loop

    subgraph Loop["agent.py (ReAct loop)"]
        direction LR
        Agent[Agent step] -->|calls| LLM[llm.py<br/>Qwen2.5-7B]
        LLM -->|decides tool + args| Tools[tools.py<br/>describe / infer / report]
        Tools -->|result| Agent
    end

    Loop --> Answer([Final answer])
    Loop -.->|records each step| Tracing[tracing.py<br/>JSON trace per run]
    Tracing --> Eval[evaluation.py<br/>metrics over trace records]

    classDef store fill:#E1F5EE,stroke:#0F6E56,color:#04342C
    classDef compute fill:#EEEDFE,stroke:#3C3489,color:#26215C
    classDef io fill:#F1EFE8,stroke:#5F5E5A,color:#2C2C2A
    class LLM,Tools,Agent compute
    class Tracing,Eval store
    class User,Answer io
```

## Benchmark metrics 

1. skill_correct: Did the agent pick the right tool?
2. path_efficient: Did it take the minimum number of steps?
3. num_loop_detected: Did the agent re-call a tool with identical arguments?
4. answer_has_markup: Did the final answer have JSON/markup characters?
5. latency_seconds: End-to-end response time

## Setup

```bash
conda env create -f environment.yml
conda activate cx-agent
pip install -e .

# Pull the model (requires Ollama running)
ollama pull qwen2.5:7b-instruct
```

## Usage

```bash
# Ask a single question
cx-agent ask "What are the top complaints in Trading?"

# Run the benchmark
cx-agent benchmark
```

## Repo structure

```
src/cx_agent/
├── data.py         # Load FABSA dataset
├── tools.py        # 3 analytics tools
├── llm.py          # LLM client and tool schemas
├── agent.py        # ReAct loop with loop detection
├── tracing.py      # Record the trace of each agent run
├── evaluation.py   # Run metrics and benchmark
└── cli.py          # Command-line interface

notebooks/          # EDA and development notebooks
traces/             # Saved trace JSONs (gitignored)
known_limitations.md  # Documented failure modes
```