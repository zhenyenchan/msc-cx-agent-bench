"""The fixed tool set every agent receives (R2).

One schema per tool in cx_agent_bench/tools.py, in the provider-neutral function-calling
format (OpenAI/Ollama style). Descriptions and argument specifications are defined
once here and are identical for every agent under evaluation; no agent may add
tools or receive modified documentation.

Field vocabularies are part of the documentation because the aspect labels are
slugs that never appear verbatim in a question ("the app or website" is the
aspect 'app-website'); without the vocabulary no agent could name a slice.
"""

INDUSTRIES = ["Banking", "Consulting", "Fashion", "Groceries",
              "Information Technology", "Price Comparison", "Ride Hailing",
              "Streaming", "Trading", "Travel Booking"]

ASPECTS = ["account-access", "app-website", "attitude-of-staff", "competitor",
           "discounts-promotions", "ease-of-use", "email",
           "general-satisfaction", "phone", "price-value-for-money", "reviews",
           "speed"]

SENTIMENTS = ["negative", "neutral", "positive"]

ORGS = ["Alderline Advisory", "Ardent Systems", "Bluepath Technologies",
        "Castellan Partners", "CompareHive", "Corvex IT", "Freshbury",
        "Halden Savings", "Investa", "Journeo", "Kerbside", "Kestrel Bank",
        "Larkmead Market", "Lumora", "Marbrook", "Merrow & Pike",
        "Northeast Bank", "Northerly", "Northpeak Trading", "Oakpan Grocers",
        "Pinecast", "PricePilot", "Quantly", "Roamly", "Sable Row", "Streamly",
        "Swiftly Rides", "Tallywise", "Trippa", "Vanter Financial",
        "Vella & Co", "Wayfare", "Zeta Cabs"]

# Which parameters hold a handle to an earlier result, and what kind of handle.
REF_PARAMS = {"summarise": {"ref": "s"}, "rank": {"ref": "t"},
              "ztest": {"ref_a": "s", "ref_b": "s"}}

_STR_OR_LIST = {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "filter",
            "description": (
                "Subset the reviews. Fields combine with AND; a list within a field is "
                "OR. Returns a selection handle (e.g. 's1') and how many rows matched; "
                "n_rows = 0 is a valid result meaning the question names a slice that "
                "does not exist. A value that never occurs anywhere in its column is "
                "flagged in a note: fix the value rather than concluding the slice is empty."),
            "parameters": {
                "type": "object",
                "properties": {
                    "industry": {**_STR_OR_LIST,
                                 "description": f"One of: {', '.join(INDUSTRIES)}"},
                    "org": {**_STR_OR_LIST,
                            "description": f"Organisation name. One of: {', '.join(ORGS)}"},
                    "aspect": {**_STR_OR_LIST,
                               "description": f"Topic label. One of: {', '.join(ASPECTS)}"},
                    "sentiment": {**_STR_OR_LIST,
                                  "description": f"One of: {', '.join(SENTIMENTS)}"},
                    "exclude_org": {**_STR_OR_LIST,
                                    "description": "Organisation(s) to drop from the "
                                                   "selection. Same values as org."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarise",
            "description": (
                "Count reviews by sentiment, overall or per group, and derive the "
                "rates. Per row: total_count, neg_count, neu_count, pos_count, "
                "neg_rate (= neg_count / total_count; neutrals stay in the "
                "denominator), neu_rate, pos_rate, prevalence (= total_count / rows "
                "in the selection) and priority_score (= prevalence * neg_rate). "
                "Returns a table handle (e.g. 't1'). min_n drops groups below a "
                "volume floor before any rate is read; the benchmark convention is "
                "min_n=30 on total mentions for a slice to be reportable. Dropped "
                "groups are named in the result."),
            "parameters": {
                "type": "object",
                "properties": {
                    "ref": {"type": "string",
                            "description": "Selection handle from filter, e.g. 's1'."},
                    "group_by": {"type": "string",
                                 "enum": ["industry", "org", "aspect", "sentiment"],
                                 "description": "Column to group by; omit for a single "
                                                "row covering the whole selection."},
                    "min_n": {"type": "integer",
                              "description": "Volume floor; groups below it are dropped "
                                             "(convention: 30)."},
                    "min_n_column": {"type": "string",
                                     "enum": ["total_count", "neg_count", "neu_count",
                                              "pos_count"],
                                     "description": "What the floor applies to. Default "
                                                    "total_count."},
                },
                "required": ["ref"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rank",
            "description": (
                "Order a summarise table by one of its columns and keep the top rows. "
                "Ties share a rank and every row with rank <= top_k is kept, so a "
                "joint first place returns both rows. top_k=null returns every row, "
                "ordered. exclude drops rows by group name; pass 'non_actionable' "
                "for the topics an organisation cannot fix (competitor, "
                "general-satisfaction, reviews). Excluded groups are named in the "
                "result. Returns group, rank, the ranking column and its supporting "
                "counts per row; the other columns are in the summarise result."),
            "parameters": {
                "type": "object",
                "properties": {
                    "ref": {"type": "string",
                            "description": "Table handle from summarise, e.g. 't1'."},
                    "by": {"type": "string",
                           "enum": ["total_count", "neg_count", "neu_count", "pos_count",
                                    "neg_rate", "neu_rate", "pos_rate", "prevalence",
                                    "priority_score"],
                           "description": "Column to order by. Default neg_rate."},
                    "top_k": {"type": ["integer", "null"],
                              "description": "How many ranks to keep (default 3); null "
                                             "for the whole ranking."},
                    "ascending": {"type": "boolean",
                                  "description": "Default false (largest first)."},
                    "exclude": {**_STR_OR_LIST,
                                "description": "Group name(s) to drop, or the shorthand "
                                               "'non_actionable'."},
                },
                "required": ["ref"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ztest",
            "description": (
                "Two-proportion z-test on the negative sentiment rates of two "
                "selections (two-sided, alpha = 0.05). Reports the four cells, the "
                "gap in percentage points, z, p and significance. The two selections must not overlap. "
                "Each of the four cells should exceed 10 for the test to be reliable, and sufficiency is reported in the result."),
            "parameters": {
                "type": "object",
                "properties": {
                    "ref_a": {"type": "string",
                              "description": "Selection handle for group A, e.g. 's1'."},
                    "ref_b": {"type": "string",
                              "description": "Selection handle for group B, e.g. 's2'."},
                    "alternative": {"type": "string",
                                    "enum": ["two-sided", "smaller", "larger"],
                                    "description": "Default two-sided."},
                },
                "required": ["ref_a", "ref_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "answer",
            "description": (
                "Record the final natural-language answer and stop. This is the only "
                "way to end the run. State the key numbers (rates to one decimal "
                "place, with counts). Abstain when the data cannot support the question, e.g. when a data slice does not have the required volume. "
                "When abstaining, begin the text with 'No answer.' "
                "and then say why."),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string",
                             "description": "The final answer, with the supporting numbers."},
                    "abstain": {"type": "boolean",
                                "description": "True if the question cannot be answered "
                                               "from the data. Default false."},
                },
                "required": ["text"],
            },
        },
    },
]

TOOL_NAMES = [t["function"]["name"] for t in TOOL_SCHEMAS]

ALLOWED_PARAMS = {t["function"]["name"]: set(t["function"]["parameters"]["properties"])
                  for t in TOOL_SCHEMAS}

# Per-parameter property specs (type / enum / anyOf), for pre-dispatch screening:
# an argument that contradicts the schema the agent was shown is rejected with a
# corrective message instead of crashing inside the tool.
PARAM_SPECS = {t["function"]["name"]: t["function"]["parameters"]["properties"]
               for t in TOOL_SCHEMAS}
