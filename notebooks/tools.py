"""Modular CX analytics tools for the benchmark.

Each tool does one thing. Agents combine them into pipelines.

Data flow examples:
    filter_data / exclude          -> row-level slicing         (returns DataFrame)
    count_by / sentiment_breakdown -> aggregation               (returns DataFrame)
    add_shares                     -> derived metrics           (returns DataFrame)
    apply_min_volume               -> volume guard              (returns DataFrame)
    rank_top                       -> sort and select           (returns DataFrame)
    share_of / two_prop_test       -> scalar summaries & stats  (returns dict)
    sample_reviews                 -> verbatim evidence         (returns list[dict])
"""

import math
import pandas as pd
from scipy import stats
from scipy.stats import norm

MIN_SAMPLE_SIZE = 30
_SENTIMENT_ORDER = ["positive", "negative", "neutral"]


# ---- Row-level slicing ----

def filter_data(
    df: pd.DataFrame,
    industry: str | None = None,
    data_source: str | None = None,
    parent_aspect: str | None = None,
    child_aspect: str | None = None,
    sentiment: str | None = None,
) -> pd.DataFrame:
    """Keep only rows matching all provided filters.

    Example: filter_data(df, industry="Banking", child_aspect="app-website")
    """
    mask = pd.Series(True, index=df.index)
    if industry is not None:     mask &= df["industry"] == industry
    if data_source is not None:  mask &= df["data_source"] == data_source
    if parent_aspect is not None: mask &= df["parent_aspect"] == parent_aspect
    if child_aspect is not None: mask &= df["child_aspect"] == child_aspect
    if sentiment is not None:    mask &= df["sentiment"] == sentiment
    return df[mask].copy()


def exclude(
    df: pd.DataFrame,
    industry: str | None = None,
    data_source: str | None = None,
    parent_aspect: str | None = None,
    child_aspect: str | None = None,
    sentiment: str | None = None,
) -> pd.DataFrame:
    """Drop rows matching all provided filters (inverse of filter_data).

    Example: exclude(banking_slice, industry="Banking")  # everything except Banking
    """
    mask = pd.Series(True, index=df.index)
    if industry is not None:     mask &= df["industry"] == industry
    if data_source is not None:  mask &= df["data_source"] == data_source
    if parent_aspect is not None: mask &= df["parent_aspect"] == parent_aspect
    if child_aspect is not None: mask &= df["child_aspect"] == child_aspect
    if sentiment is not None:    mask &= df["sentiment"] == sentiment
    return df[~mask].copy()


# ---- Aggregation ----

def count_by(df: pd.DataFrame, group_by: str) -> pd.DataFrame:
    """Count rows per group, sorted descending.

    Returns DataFrame with columns [group_by, 'count'].

    Example: count_by(negative_reviews, group_by="child_aspect")
    """
    return (
        df.groupby(group_by).size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )


def sentiment_breakdown(df: pd.DataFrame, group_by: str) -> pd.DataFrame:
    """Pivot into sentiment-per-group table.

    Returns DataFrame: [group_by, 'positive', 'negative', 'neutral', 'total']

    Example: sentiment_breakdown(df, group_by="child_aspect")
    """
    pivot = pd.pivot_table(df, index=group_by, columns="sentiment",
                           aggfunc="size", fill_value=0)
    for s in _SENTIMENT_ORDER:
        if s not in pivot.columns:
            pivot[s] = 0
    pivot = pivot[_SENTIMENT_ORDER].copy()
    pivot["total"] = pivot.sum(axis=1)
    return pivot.reset_index()


# ---- Derived metrics ----

def add_shares(breakdown_df: pd.DataFrame) -> pd.DataFrame:
    """Add pos_share, neg_share, neu_share columns to a sentiment breakdown.

    Example: add_shares(sentiment_breakdown(df, "child_aspect"))
    """
    df = breakdown_df.copy()
    total = df["total"].replace(0, pd.NA)
    df["pos_share"] = (df["positive"] / total).fillna(0).round(3)
    df["neg_share"] = (df["negative"] / total).fillna(0).round(3)
    df["neu_share"] = (df["neutral"]  / total).fillna(0).round(3)
    return df


# ---- Selection ----

def apply_min_volume(
    df: pd.DataFrame,
    min_volume: int = MIN_SAMPLE_SIZE,
    col: str = "total",
) -> pd.DataFrame:
    """Drop rows where volume column < min_volume.

    Example: apply_min_volume(breakdown_df, min_volume=30)
    """
    return df[df[col] >= min_volume].copy()


def rank_top(
    df: pd.DataFrame,
    by: str,
    top_n: int = 3,
    ascending: bool = False,
) -> pd.DataFrame:
    """Sort by column and return top_n rows.

    Example: rank_top(scored_df, by="neg_share", top_n=3)
    """
    return df.sort_values(by, ascending=ascending).head(top_n).reset_index(drop=True)


# ---- Scalar summaries & stats ----

def share_of(df: pd.DataFrame, sentiment: str = "negative") -> dict:
    """Share of rows with the given sentiment.

    Returns {'n': total, 'count': matching, 'share': fraction}.

    Example: share_of(banking_appweb, sentiment="negative")
    """
    n = len(df)
    if n == 0:
        return {"n": 0, "count": 0, "share": None}
    count = int((df["sentiment"] == sentiment).sum())
    return {"n": n, "count": count, "share": round(count / n, 3)}


def two_prop_test(df_a: pd.DataFrame, df_b: pd.DataFrame,
                  sentiment: str = "negative") -> dict:
    """Two-proportion z-test: is the sentiment rate different between two slices?

    Returns {'p1', 'p2', 'z', 'p_value', 'significant', 'higher'}.

    Example: two_prop_test(banking_eou, travel_eou, sentiment="positive")
    """
    x1, n1 = int((df_a.sentiment == sentiment).sum()), len(df_a)
    x2, n2 = int((df_b.sentiment == sentiment).sum()), len(df_b)
    if n1 == 0 or n2 == 0:
        return {"error": "empty slice", "n1": n1, "n2": n2}
    p1, p2 = x1/n1, x2/n2
    p = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p * (1-p) * (1/n1 + 1/n2))
    z = (p1 - p2) / se if se > 0 else 0.0
    pv = float(2 * norm.sf(abs(z)))
    return {
        "p1": round(p1, 4), "p2": round(p2, 4),
        "z": round(z, 3), "p_value": round(pv, 4),
        "significant": pv < 0.05,
        "higher": "a" if p1 > p2 else "b",
    }


def chi_squared(df_a: pd.DataFrame, df_b: pd.DataFrame) -> dict:
    """Chi-squared test: do sentiment distributions differ between two slices?

    Example: chi_squared(org_slice, peers_slice)
    """
    a = df_a["sentiment"].value_counts().reindex(_SENTIMENT_ORDER, fill_value=0)
    b = df_b["sentiment"].value_counts().reindex(_SENTIMENT_ORDER, fill_value=0)
    contingency = pd.DataFrame({"a": a, "b": b})
    try:
        chi2, p_value, _, _ = stats.chi2_contingency(contingency)
        return {
            "n_a": int(a.sum()), "n_b": int(b.sum()),
            "chi2": round(float(chi2), 3),
            "p_value": round(float(p_value), 4),
            "significant": bool(p_value < 0.05),
        }
    except ValueError as e:
        return {"n_a": int(a.sum()), "n_b": int(b.sum()), "error": str(e)}


# ---- Evidence ----

def sample_reviews(
    df: pd.DataFrame,
    n: int = 3,
    random_state: int = 42,
) -> list[dict]:
    """Return n example reviews for verbatim evidence.

    Example: sample_reviews(negative_appweb_slice, n=3)
    """
    if len(df) == 0:
        return []
    sampled = df.sample(min(n, len(df)), random_state=random_state)
    return [
        {
            "text": row["text"][:300],
            "child_aspect": row["child_aspect"],
            "sentiment": row["sentiment"],
        }
        for _, row in sampled.iterrows()
    ]
