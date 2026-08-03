"""LLM client wrapper - supports Ollama (local) and OpenRouter (cloud)"""

import os
from dotenv import load_dotenv
from litellm import completion

load_dotenv()

# default model for poc
DEFAULT_MODEL = "ollama/qwen2.5:7b-instruct"
API_BASE = "http://localhost:11434"

SYSTEM_PROMPT = """You are an expert customer experience analyst. You analyse customer feedback data from the FABSA dataset.

SCOPE CHECK (do this FIRST, before anything else):
Before calling any tool, ask yourself: "Is this question about customer feedback, customer experience, reviews, sentiment, or aspects of products/services?"

- If NO (e.g. geography, weather, math, general knowledge, coding, history): Do NOT call any tool. 
Respond with exactly: "I can only answer questions about customer feedback data from the FABSA dataset. Please ask about industries, aspects, or sentiment."
- If YES: Proceed to use a tool.

Examples of OUT-OF-SCOPE questions (refuse these):
- "What is the capital of Italy?"
- "How do I cook pasta?"
- "Write me a poem."
- "What is 2+2?"

Examples of IN-SCOPE questions (use a tool):
- "What are the top complaints in Banking?"
- "Compare sentiment between Fashion and Trading."
- "Give me a summary report for Google Play reviews."

You have access to three tools:
1. describe — count and rank aspects/sentiments with filters
2. infer — statistically compare sentiment between two groups
3. report — generate a structured summary for a given scope

RULES:
- For in-scope questions, always use a tool. Do not make up data.
- Call ONE tool per step. Do not call the same tool twice with the same arguments.
- "Complaints" or "issues" always means sentiment="negative". "Praise" means sentiment="positive".
- After receiving a tool result, write a final answer in plain English.

RESPONSE FORMAT for your final answer:
- Maximum 3 sentences. Be direct and factual.
- NEVER include JSON, code blocks, "Tool Calls:", "### User:", or any markup.
- Include the key numbers from the tool result inline (e.g. "146 mentions").
- When reporting a statistical test, always include the p-value.
- Example good answer: "The top complaints in Banking are app-website (146 mentions), attitude-of-staff (109), and ease-of-use (107)"
- Example bad answer: any response containing "Tool Calls:" or multi-paragraph breakdowns with bullet points.

Valid filter values:
- industry: Banking, Consulting, Fashion, Groceries, Information Technology, Price Comparison, Ride Hailing, Streaming, Trading, Travel Booking
- parent_aspect: company-brand, logistics-rides, online-experience, purchase-booking-experience, staff-support, value, account-management
- child_aspect: account-access, app-website, attitude-of-staff, competitor, discounts-promotions, ease-of-use, email, general-satisfaction, phone, price-value-for-money, reviews, speed
- sentiment: positive, negative, neutral

"""

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "describe",
            "description": "Count and rank aspect+sentiment labels. Use for questions like: 'What are the top complaints in Banking?', 'How is sentiment distributed for app-website?', 'What aspects come up most?'",
            "parameters": {
                "type": "object",
                "properties": {
                    "group_by": {
                        "type": "string",
                        "enum": ["child_aspect", "parent_aspect", "industry"],
                        "description": "Dimension to group results by. Default: child_aspect",
                    },
                    "industry": {
                        "type": "string",
                        "description": "Filter by industry name",
                    },
                    "parent_aspect": {
                        "type": "string",
                        "description": "Filter by parent aspect category",
                    },
                    "child_aspect": {
                        "type": "string",
                        "description": "Filter by child aspect category",
                    },
                    "sentiment": {
                        "type": "string",
                        "enum": ["positive", "negative", "neutral"],
                        "description": "Filter by sentiment",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of top results to return. Default: 5",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "infer",
            "description": "Statistically compare sentiment distributions between two groups using a chi-squared test. Use for questions like: 'Is sentiment different between app-website and speed?', 'Do Banking and Fashion differ?'",
            "parameters": {
                "type": "object",
                "properties": {
                    "compare_field": {
                        "type": "string",
                        "enum": ["child_aspect", "parent_aspect", "industry"],
                        "description": "The dimension to compare across",
                    },
                    "group_a": {
                        "type": "string",
                        "description": "First group to compare",
                    },
                    "group_b": {
                        "type": "string",
                        "description": "Second group to compare",
                    },
                    "industry": {
                        "type": "string",
                        "description": "Optional: filter both groups by industry",
                    },
                },
                "required": ["compare_field", "group_a", "group_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "report",
            "description": "Generate a structured summary report for a given scope. Use for questions like: 'Give me a summary report for Fashion', 'Summarise feedback for Travel Booking'",
            "parameters": {
                "type": "object",
                "properties": {
                    "industry": {
                        "type": "string",
                        "description": "Filter by industry name",
                    },
                },
                "required": [],
            },
        },
    },
]


def chat(messages: list[dict], model: str = DEFAULT_MODEL) -> dict:
    """Send messages to the LLM with tool definitions.

    Args:
        messages: list of message dicts (role/content)
        model: LiteLLM model string. Examples:
            - "ollama/qwen2.5:7b-instruct"           (local)
            - "openrouter/openai/gpt-4o-mini"        (OpenAI via OpenRouter)
            - "openrouter/anthropic/claude-3.5-haiku" (Anthropic via OpenRouter)
            - "openrouter/meta-llama/llama-3.3-70b-instruct"
            - "openrouter/qwen/qwen-2.5-7b-instruct"
    """
    kwargs = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        "tools": TOOL_DEFINITIONS,
    }

    # Route by provider
    if model.startswith("ollama/"):
        kwargs["api_base"] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    elif model.startswith("openrouter/"):
        kwargs["api_key"] = os.getenv("OPENROUTER_API_KEY")
        if not kwargs["api_key"]:
            raise ValueError("OPENROUTER_API_KEY not set in .env")
    else:
        raise ValueError(f"Unsupported model prefix: {model}")

    return completion(**kwargs)