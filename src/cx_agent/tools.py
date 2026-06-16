"""CX analytics tools for descriptive, inferential, reporting queries"""

import pandas as pd
from scipy import stats

from cx_agent.data import (
    VALID_INDUSTRIES,
    VALID_DATA_SOURCES,
    VALID_PARENT_ASPECTS,
    VALID_CHILD_ASPECTS,
    VALID_SENTIMENTS,
)

MIN_SAMPLE_SIZE = 30  


# Helpers

def _apply_filters(
    df: pd.DataFrame,
    industry: str | None = None,
    data_source: str | None = None,
    parent_aspect: str | None = None,
    child_aspect: str | None = None,
    sentiment: str | None = None,
) -> pd.DataFrame:
    """Filter df_exploded by any combination of dimensions."""
    if industry:
        _validate("industry", industry, VALID_INDUSTRIES)
        df = df[df["industry"] == industry]
    if data_source:
        _validate("data_source", data_source, VALID_DATA_SOURCES)
        df = df[df["data_source"] == data_source]
    if parent_aspect:
        _validate("parent_aspect", parent_aspect, VALID_PARENT_ASPECTS)
        df = df[df["parent_aspect"] == parent_aspect]
    if child_aspect:
        _validate("child_aspect", child_aspect, VALID_CHILD_ASPECTS)
        df = df[df["child_aspect"] == child_aspect]
    if sentiment:
        _validate("sentiment", sentiment, VALID_SENTIMENTS)
        df = df[df["sentiment"] == sentiment]
    return df


def _validate(field: str, value: str, valid: list[str]) -> None:
    """Raise ValueError if value is not in the valid list."""
    if value not in valid:
        raise ValueError(
            f"Invalid {field}: '{value}'. Must be one of: {valid}"
        )


# Tool 1: Descriptive

def describe(
    df_exploded: pd.DataFrame,
    group_by: str = "child_aspect",
    industry: str | None = None,
    data_source: str | None = None,
    parent_aspect: str | None = None,
    child_aspect: str | None = None,
    sentiment: str | None = None,
    top_n: int = 5,
) -> dict:
    """Count and rank aspect+sentiment labels.

    Example questions:
      - "What are the top complaints in Banking?"
      - "How do customers feel about app-website?"
      - "What are the most common aspects in Trustpilot reviews?"
    """
    filtered = _apply_filters(
        df_exploded, industry, data_source, parent_aspect, child_aspect, sentiment
    )

    if len(filtered) == 0:
        return {"error": "No data matches the given filters.", "count": 0}

    counts = (
        filtered.groupby([group_by, "sentiment"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    top = counts.head(top_n).to_dict(orient="records")

    sentiment_dist = filtered["sentiment"].value_counts(normalize=True).round(3).to_dict()

    return {
        "total_labels": len(filtered),
        "unique_reviews": filtered["id"].nunique(),
        "sentiment_distribution": sentiment_dist,
        "top": top,
        "filters_applied": {
            "industry": industry,
            "data_source": data_source,
            "parent_aspect": parent_aspect,
            "child_aspect": child_aspect,
            "sentiment": sentiment,
        },
    }


# Tool 2: Inferential

def infer(
    df_exploded: pd.DataFrame,
    compare_field: str = "child_aspect",
    group_a: str = "",
    group_b: str = "",
    industry: str | None = None,
    data_source: str | None = None,
) -> dict:
    """Compare sentiment distributions between 2 groups.

    Example questions:
      - "Is sentiment significantly different between app-website and speed?"
      - "How do Banking and Fashion differ in negative sentiment?"
    """
    filtered = _apply_filters(df_exploded, industry=industry, data_source=data_source)

    valid_fields = {
        "child_aspect": VALID_CHILD_ASPECTS,
        "parent_aspect": VALID_PARENT_ASPECTS,
        "industry": VALID_INDUSTRIES,
        "data_source": VALID_DATA_SOURCES,
    }

    if compare_field not in valid_fields:
        return {"error": f"compare_field must be one of: {list(valid_fields.keys())}"}

    _validate(f"{compare_field} group_a", group_a, valid_fields[compare_field])
    _validate(f"{compare_field} group_b", group_b, valid_fields[compare_field])

    a = filtered[filtered[compare_field] == group_a]
    b = filtered[filtered[compare_field] == group_b]

    warnings = []
    if len(a) < MIN_SAMPLE_SIZE:
        warnings.append(f"group_a '{group_a}' has only {len(a)} samples (< {MIN_SAMPLE_SIZE})")
    if len(b) < MIN_SAMPLE_SIZE:
        warnings.append(f"group_b '{group_b}' has only {len(b)} samples (< {MIN_SAMPLE_SIZE})")

    # Build sentiment count vectors in consistent order
    sent_order = ["positive", "negative", "neutral"]
    a_counts = a["sentiment"].value_counts().reindex(sent_order, fill_value=0)
    b_counts = b["sentiment"].value_counts().reindex(sent_order, fill_value=0)

    # Chi-squared test
    contingency = pd.DataFrame({"group_a": a_counts, "group_b": b_counts})
    chi2, p_value, dof, _ = stats.chi2_contingency(contingency)

    return {
        "group_a": {"name": group_a, "n": len(a), "sentiment": a_counts.to_dict()},
        "group_b": {"name": group_b, "n": len(b), "sentiment": b_counts.to_dict()},
        "test": "chi-squared",
        "chi2": round(chi2, 3),
        "p_value": round(p_value, 4),
        "significant": p_value < 0.05,
        "warnings": warnings,
    }


# Tool 3: Reporting 

def report(
    df_exploded: pd.DataFrame,
    df_reviews: pd.DataFrame,
    industry: str | None = None,
    data_source: str | None = None,
) -> dict:
    """Generate a structured summary for a given scope.

    Example questions:
      - "Give me a summary report for the Fashion industry"
      - "Summarise feedback from Google Play"
    """
    filtered = _apply_filters(df_exploded, industry=industry, data_source=data_source)

    if len(filtered) == 0:
        return {"error": "No data matches the given filters.", "count": 0}

    # Filter reviews to match
    review_ids = filtered["id"].unique()
    reviews = df_reviews[df_reviews["id"].isin(review_ids)]

    # Overall stats
    sentiment_dist = filtered["sentiment"].value_counts(normalize=True).round(3).to_dict()

    # Top positive and negative child aspects
    neg = (
        filtered[filtered["sentiment"] == "negative"]
        .groupby("child_aspect")
        .size()
        .sort_values(ascending=False)
        .head(3)
        .to_dict()
    )
    pos = (
        filtered[filtered["sentiment"] == "positive"]
        .groupby("child_aspect")
        .size()
        .sort_values(ascending=False)
        .head(3)
        .to_dict()
    )

    # Sample verbatims (one positive, one negative)
    verbatims = {}
    for sent in ["positive", "negative"]:
        subset = filtered[filtered["sentiment"] == sent]
        if len(subset) > 0:
            sample = subset.sample(1, random_state=42).iloc[0]
            verbatims[sent] = {
                "text": sample["text"][:300],
                "aspect": f"{sample['parent_aspect']}.{sample['child_aspect']}",
            }

    return {
        "scope": {"industry": industry, "data_source": data_source},
        "total_reviews": len(reviews),
        "total_labels": len(filtered),
        "avg_labels_per_review": round(len(filtered) / len(reviews), 1),
        "avg_words_per_review": round(reviews["word_count"].mean(), 0),
        "sentiment_distribution": sentiment_dist,
        "top_negative_aspects": neg,
        "top_positive_aspects": pos,
        "sample_verbatims": verbatims,
    }