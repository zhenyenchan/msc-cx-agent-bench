"""LLM client wrapper with tool definitions"""

from litellm import completion

MODEL = "ollama/qwen2.5:3b-instruct"
API_BASE = "http://localhost:11434"

SYSTEM_PROMPT = """You are an expert customer experience analytics agent. You analyse customer feedback data from the FABSA dataset.

You have access to three tools:
1. describe — count and rank aspects/sentiments with filters
2. infer — statistically compare sentiment between two groups
3. report — generate a structured summary for a given scope

Always use a tool to answer questions. Do not make up data.

Valid filter values:
- industry: Banking, Consulting, Fashion, Groceries, Information Technology, Price Comparison, Ride Hailing, Streaming, Trading, Travel Booking
- data_source: Trustpilot, Google Play, Apple Store
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
                        "enum": ["child_aspect", "parent_aspect", "industry", "data_source"],
                        "description": "Dimension to group results by. Default: child_aspect",
                    },
                    "industry": {
                        "type": "string",
                        "description": "Filter by industry name",
                    },
                    "data_source": {
                        "type": "string",
                        "description": "Filter by data source",
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
                        "enum": ["child_aspect", "parent_aspect", "industry", "data_source"],
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
                    "data_source": {
                        "type": "string",
                        "description": "Optional: filter both groups by data source",
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
            "description": "Generate a structured summary report for a given scope. Use for questions like: 'Give me a summary report for Fashion', 'Summarise feedback from Google Play'",
            "parameters": {
                "type": "object",
                "properties": {
                    "industry": {
                        "type": "string",
                        "description": "Filter by industry name",
                    },
                    "data_source": {
                        "type": "string",
                        "description": "Filter by data source",
                    },
                },
                "required": [],
            },
        },
    },
]


def chat(messages: list[dict]) -> dict:
    """Send messages to the LLM with tool definitions.

    Returns the raw LiteLLM response object.
    """
    response = completion(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        tools=TOOL_DEFINITIONS,
        api_base=API_BASE,
    )
    return response