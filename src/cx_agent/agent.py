"""ReAct agent loop that routes user questions to tools via LLM"""

import json
from cx_agent.llm import chat
from cx_agent.data import load_fabsa
from cx_agent.tools import describe, infer, report

# Map tool names to functions
TOOL_REGISTRY = {
    "describe": describe,
    "infer": infer,
    "report": report,
}

MAX_STEPS = 5  # prevent infinite loops


def _execute_tool(
    tool_name: str,
    arguments: dict,
    df_reviews,
    df_exploded,
) -> dict:
    """Execute a tool by name with parsed arguments."""
    if tool_name not in TOOL_REGISTRY:
        return {"error": f"Unknown tool: {tool_name}"}

    if tool_name == "report":
        return report(df_exploded, df_reviews, **arguments)
    elif tool_name == "describe":
        return describe(df_exploded, **arguments)
    elif tool_name == "infer":
        return infer(df_exploded, **arguments)


def run(question: str, df_reviews=None, df_exploded=None, verbose: bool = True) -> dict:
    """Run the agent on a single question.

    Returns a dict with:
      - answer: the LLM's final natural language response
      - trace: list of steps (for benchmarking)
    """
    # Load data if not provided
    if df_reviews is None or df_exploded is None:
        df_reviews, df_exploded = load_fabsa()

    messages = [{"role": "user", "content": question}]
    trace = []
    just_got_tool_result = False

    for step in range(MAX_STEPS):
        if verbose:
            print(f"\n--- Step {step + 1} ---")

        # Call LLM without tools after tool result
        response = chat(messages, use_tools=not just_got_tool_result)
        just_got_tool_result = False
        msg = response.choices[0].message

        # Case 1: LLM wants to call a tool
        if msg.tool_calls:
            tc = msg.tool_calls[0]
            tool_name = tc.function.name
            arguments = json.loads(tc.function.arguments)

            # Loop detection: same tool + same args as a previous step
            duplicate = False
            for prev in trace:
                if (prev["type"] == "tool_call"
                    and prev["tool"] == tool_name
                    and prev["arguments"] == arguments):
                    duplicate = True
                    break

            if duplicate:
                if verbose:
                    print(f"Loop detected: {tool_name} already called with same args")
                messages.append({
                    "role": "user",
                    "content": "You already called this tool with these exact arguments. Now summarise the result in plain English.",
                })
                trace.append({
                    "step": step + 1,
                    "type": "loop_detected",
                    "tool": tool_name,
                    "arguments": arguments,
                })
                continue

            if verbose:
                print(f"Tool: {tool_name}")
                print(f"Args: {arguments}")

            # Execute the tool
            try:
                tool_result = _execute_tool(
                    tool_name, arguments, df_reviews, df_exploded
                )
            except (ValueError, TypeError) as e:
                tool_result = {"error": str(e)}

            if verbose:
                print(f"Result: {json.dumps(tool_result, indent=2)[:500]}")

            # Record in trace
            trace.append({
                "step": step + 1,
                "type": "tool_call",
                "tool": tool_name,
                "arguments": arguments,
                "result": tool_result,
            })

            # Feed result back to LLM
            messages.append(msg.model_dump())
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(tool_result),
            })
            just_got_tool_result = True

        # Case 2: LLM gives a final text answer
        else:
            answer = msg.content
            if verbose:
                print(f"Answer: {answer}")

            trace.append({
                "step": step + 1,
                "type": "final_answer",
                "answer": answer,
            })

            return {"answer": answer, "trace": trace}

    # Hit max steps - use last tool result if available
    last_tool_result = None
    for t in reversed(trace):
        if t["type"] == "tool_call":
            last_tool_result = t["result"]
            break

    if last_tool_result:
        return {
            "answer": f"Agent reached max steps. Last tool result: {json.dumps(last_tool_result)[:500]}",
            "trace": trace,
        }

    return {
        "answer": "Agent reached maximum steps without a final answer.",
        "trace": trace,
    }