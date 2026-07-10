"""Minimal Claude Code benchmark harness."""
import subprocess, json, time, csv, pathlib
from datetime import datetime

QUERIES_CSV  = "agent_sandbox/queries-1.csv"
SANDBOX_DIR  = "agent_sandbox"
RESULTS_DIR  = pathlib.Path("agent_sandbox/runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
ALLOWED_TOOLS = "mcp__fabsa__*"
SYSTEM_PROMPT = (
    "You are an analytics agent. Answer the user's question about customer "
    "feedback using ONLY the FABSA tools provided. Do not guess or use prior "
    "knowledge — every claim must come from a tool call."
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_query(qid: str, query: str) -> dict:
    cmd = [
        "claude", "-p", query,
        "--output-format", "stream-json",
        "--verbose",
        "--allowedTools", ALLOWED_TOOLS,
        "--disallowedTools", "Write,Read,Edit,Bash,Glob,Grep,WebSearch,WebFetch,NotebookEdit",  # block claude built-in tools
        "--permission-mode", "default",   
        "--max-turns", "20",
        "--append-system-prompt", SYSTEM_PROMPT,
    ]

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600,
        cwd=SANDBOX_DIR, shell=True,   # shell=True needed on Windows for claude.cmd
        encoding="utf-8", errors="replace",  # handle unknown characters in tool outputs
    )
    wall_ms = (time.perf_counter() - t0) * 1000

    (RESULTS_DIR / f"{qid}.jsonl").write_text(proc.stdout or "", encoding="utf-8")

    tool_path, final = [], None
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "assistant":
            for block in ev.get("message", {}).get("content", []):
                if block.get("type") == "tool_use":
                    tool_path.append(block.get("name"))
        if ev.get("type") == "result":
            final = ev

    if final is None:
        return {"qid": qid, "query": query, "error": "no result event",
                "stderr": proc.stderr[:500]}

    return {
        "qid": qid,
        "query": query,
        "final_answer": final.get("result", ""),
        "tool_path": " -> ".join(tool_path) if tool_path else "(none)",
        "num_tool_calls": len(tool_path),
        "num_turns": final.get("num_turns"),
        "input_tokens": final.get("usage", {}).get("input_tokens"),
        "output_tokens": final.get("usage", {}).get("output_tokens"),
        "cost_usd": final.get("total_cost_usd"),
        "latency_ms": round(wall_ms),
        "session_id": final.get("session_id"),
    }


def main():
    with open(QUERIES_CSV, encoding="utf-8") as f:
        queries = list(csv.DictReader(f))

    results = []
    for i, row in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] {row['id']}: {row['query'][:60]}...")
        try:
            res = run_query(row["id"], row["query"])
        except subprocess.TimeoutExpired:
            res = {"qid": row["id"], "query": row["query"], "error": "timeout"}
        results.append(res)

        # convert latency to minutes and seconds
        ms = res.get('latency_ms', 0) or 0
        mins, secs = divmod(ms / 1000, 60)
        latency_str = f"{int(mins)}m {secs:04.1f}s" if mins else f"{secs:.1f}s"
        print(f"    -> {res.get('num_tool_calls', '?')} tools, "
              f"${res.get('cost_usd', 0) or 0:.4f}, "
              f"{latency_str}")

    out = RESULTS_DIR / "results.csv"
    fieldnames = ["qid", "query", "final_answer", "tool_path",
                  "num_tool_calls", "num_turns", "input_tokens",
                  "output_tokens", "cost_usd", "latency_ms",
                  "session_id", "error"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"\nDone. Results: {out}")


if __name__ == "__main__":
    main()