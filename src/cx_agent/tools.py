"""Modular CX analytics tools.

Each tool does one thing. Agents combine them into pipelines to answer
descriptive, inferential and reporting questions.

Example data flow:
    filter_data / exclude          -> row-level slicing         (returns DataFrame)
    count_by / sentiment_breakdown -> aggregation               (returns DataFrame)
    add_shares                     -> derived metrics           (returns DataFrame)
    apply_min_volume               -> volume guard              (returns DataFrame)
    rank_top                       -> sort and select           (returns DataFrame)
    share_of / chi_squared         -> scalar summaries & stats  (returns dict)

sample_reviews (verbatim evidence) is disabled: the corpus is labels-only, with no review text.
"""

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
_SENTIMENT_ORDER = ["positive", "negative", "neutral"]


# ---------------- Internal helpers ----------------

def _validate(field: str, value, valid) -> None:
    if value not in valid:
        raise ValueError(f"Invalid {field}: '{value}'. Must be one of: {valid}")


def _build_mask(df, org_index, industry, data_source, parent_aspect, child_aspect, sentiment, org=None):
    """Build a boolean mask that is True where all provided filters match."""
    mask = pd.Series(True, index=df.index)
    if org_index is not None:
        mask &= df["org_index"] == org_index
    if org is not None:
        mask &= df["org"] == org
    if industry is not None:
        _validate("industry", industry, VALID_INDUSTRIES)
        mask &= df["industry"] == industry
    if data_source is not None:
        _validate("data_source", data_source, VALID_DATA_SOURCES)
        mask &= df["data_source"] == data_source
    if parent_aspect is not None:
        _validate("parent_aspect", parent_aspect, VALID_PARENT_ASPECTS)
        mask &= df["parent_aspect"] == parent_aspect
    if child_aspect is not None:
        _validate("child_aspect", child_aspect, VALID_CHILD_ASPECTS)
        mask &= df["child_aspect"] == child_aspect
    if sentiment is not None:
        _validate("sentiment", sentiment, VALID_SENTIMENTS)
        mask &= df["sentiment"] == sentiment
    return mask


# ---------------- Row-level slicing ----------------

def filter_data(
    df: pd.DataFrame,
    org_index: int | None = None,
    industry: str | None = None,
    data_source: str | None = None,
    parent_aspect: str | None = None,
    child_aspect: str | None = None,
    sentiment: str | None = None,
    org: str | None = None,
) -> pd.DataFrame:
    """Keep only rows matching all provided filters.

    Any argument left as None is not applied. Returns a DataFrame of
    matching rows (same columns as input).

    Example: filter_data(df, industry="Banking", child_aspect="app-website")
             filter_data(df, org="BankA", child_aspect="app-website")  # single org
    """
    mask = _build_mask(df, org_index, industry, data_source, parent_aspect, child_aspect, sentiment, org)
    return df[mask].copy()


def exclude(
    df: pd.DataFrame,
    org_index: int | None = None,
    industry: str | None = None,
    data_source: str | None = None,
    parent_aspect: str | None = None,
    child_aspect: str | None = None,
    sentiment: str | None = None,
    org: str | None = None,
) -> pd.DataFrame:
    """Drop rows matching all provided filters (inverse of filter_data).

    Useful for building 'peer' or 'benchmark' slices that exclude a target.
    Requires at least one filter.

    Example: exclude(banking_slice, org_index=514)  # peers only
             exclude(banking_slice, org="BankA")    # Banking peers of BankA
    """
    if all(v is None for v in [org_index, industry, data_source, parent_aspect, child_aspect, sentiment, org]):
        raise ValueError("exclude requires at least one filter")
    mask = _build_mask(df, org_index, industry, data_source, parent_aspect, child_aspect, sentiment, org)
    return df[~mask].copy()


# ---------------- Aggregation ----------------

def count_by(df: pd.DataFrame, group_by: str) -> pd.DataFrame:
    """Count rows in each group, sorted descending.

    Returns a DataFrame with columns [group_by, 'count'].

    Example: count_by(negative_reviews, group_by="child_aspect")
    """
    return (
        df.groupby(group_by).size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )


def sentiment_breakdown(df: pd.DataFrame, group_by: str) -> pd.DataFrame:
    """Pivot rows into a sentiment-per-group table.

    Returns a DataFrame with columns:
      [group_by, 'positive', 'negative', 'neutral', 'total']

    Missing sentiment categories are filled with 0.

    Example: sentiment_breakdown(df, group_by="parent_aspect")
    """
    pivot = pd.pivot_table(df, index=group_by, columns="sentiment", aggfunc="size", fill_value=0)
    for s in _SENTIMENT_ORDER:
        if s not in pivot.columns:
            pivot[s] = 0
    pivot = pivot[_SENTIMENT_ORDER].copy()
    pivot["total"] = pivot.sum(axis=1)
    return pivot.reset_index()


# ---------------- Derived metrics ----------------

def add_shares(breakdown_df: pd.DataFrame) -> pd.DataFrame:
    """Add pos_share, neg_share, neu_share columns to a sentiment breakdown.

    Input must have 'positive', 'negative', 'neutral', 'total' columns
    (output of sentiment_breakdown). Rows with total=0 get 0 for all shares.

    Example: add_shares(sentiment_breakdown(df, group_by="parent_aspect"))
    """
    df = breakdown_df.copy()
    total = df["total"].replace(0, pd.NA)
    df["pos_share"] = (df["positive"] / total).fillna(0).round(3)
    df["neg_share"] = (df["negative"] / total).fillna(0).round(3)
    df["neu_share"] = (df["neutral"] / total).fillna(0).round(3)
    return df


def add_priority(breakdown_df: pd.DataFrame) -> pd.DataFrame:
    """Add a `priority` score combining complaint volume and severity (0-2 scale).

    priority = min-max normalised total volume + min-max normalised neg_share, so an
    aspect scores high only when it is BOTH high-volume and high-severity — the
    'prioritisation quadrant'. Requires neg_share (from add_shares) and total columns.

    Example: add_priority(add_shares(sentiment_breakdown(df, "child_aspect")))
    """
    df = breakdown_df.copy()
    def _norm(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng else s * 0.0
    df["priority"] = (_norm(df["total"]) + _norm(df["neg_share"])).round(3)
    return df


def add_polarisation(breakdown_df: pd.DataFrame) -> pd.DataFrame:
    """Add polarisation and pol_volume columns to a sentiment breakdown.

    Neutrals are excluded from both the score and its volume:
      - pol_volume   = positive + negative
      - polarisation = min(positive, negative) / pol_volume
        Peaks at 0.5 for a 50/50 split; floors at 0 for one-sided.

    Use pol_volume (not total) with apply_min_volume before ranking, so
    the volume guard matches the score's denominator.

    Example: add_polarisation(sentiment_breakdown(df, group_by="child_aspect"))
    """
    df = breakdown_df.copy()
    df["pol_volume"] = df["positive"] + df["negative"]
    denom = df["pol_volume"].replace(0, pd.NA)
    df["polarisation"] = (
        df[["positive", "negative"]].min(axis=1) / denom
    ).fillna(0).round(3)
    return df


# ---------------- Selection ----------------

def apply_min_volume(
    df: pd.DataFrame,
    min_volume: int = MIN_SAMPLE_SIZE,
    col: str = "total",
) -> pd.DataFrame:
    """Drop rows where the volume column is below min_volume.

    Use to exclude low-sample groups before ranking or comparing.
    Default column is 'total' (from sentiment_breakdown); use col='count'
    for output from count_by.

    Example: apply_min_volume(breakdown_df, min_volume=30)
    """
    return df[df[col] >= min_volume].copy()


def rank_top(
    df: pd.DataFrame,
    by: str,
    top_n: int = 3,
    ascending: bool = False,
) -> pd.DataFrame:
    """Sort by a column and return the top_n rows.

    Example: rank_top(scored_df, by="neg_share", top_n=3)
    """
    return df.sort_values(by, ascending=ascending).head(top_n).reset_index(drop=True)


# ---------------- Scalar summaries ----------------

def share_of(df: pd.DataFrame, sentiment: str = "negative") -> dict:
    """Return the share of rows with the given sentiment in a slice.

    Returns {'n': total, 'count': matching, 'share': fraction}.
    Returns share=None if the slice is empty.

    Example: share_of(banking_slice, sentiment="negative")
    """
    _validate("sentiment", sentiment, VALID_SENTIMENTS)
    n = len(df)
    if n == 0:
        return {"n": 0, "count": 0, "share": None}
    count = int((df["sentiment"] == sentiment).sum())
    return {"n": n, "count": count, "share": round(count / n, 3)}


def chi_squared(df_a: pd.DataFrame, df_b: pd.DataFrame) -> dict:
    """Test whether sentiment distributions differ between two slices.

    Runs a chi-squared test on the positive/negative/neutral counts of
    each slice. Returns test statistics and significance at p<0.05.

    Example: chi_squared(org_slice, peers_slice)
    """
    a = df_a["sentiment"].value_counts().reindex(_SENTIMENT_ORDER, fill_value=0)
    b = df_b["sentiment"].value_counts().reindex(_SENTIMENT_ORDER, fill_value=0)
    contingency = pd.DataFrame({"a": a, "b": b})
    try:
        chi2, p_value, _, _ = stats.chi2_contingency(contingency)
        return {
            "n_a": int(a.sum()),
            "n_b": int(b.sum()),
            "chi2": round(float(chi2), 3),
            "p_value": round(float(p_value), 4),
            "significant": bool(p_value < 0.05),
        }
    except ValueError as e:
        return {"n_a": int(a.sum()), "n_b": int(b.sum()), "error": str(e)}


# ---------------- Evidence ----------------

# DISABLED: the benchmark corpus is labels-only (no review text), so this tool has nothing to read.
# Kept commented rather than deleted in case review text is reinstated.
# def sample_reviews(
#     df: pd.DataFrame,
#     n: int = 3,
#     random_state: int = 42,
# ) -> list[dict]:
#     """Return n example reviews from a slice for verbatim evidence.
#
#     Each entry: {'text', 'parent_aspect', 'child_aspect', 'sentiment'}.
#     Returns an empty list if the slice is empty.
#
#     Example: sample_reviews(negative_appweb_slice, n=3)
#     """
#     if len(df) == 0:
#         return []
#     sampled = df.sample(min(n, len(df)), random_state=random_state)
#     return [
#         {
#             "text": row["text"][:300],
#             "parent_aspect": row["parent_aspect"],
#             "child_aspect": row["child_aspect"],
#             "sentiment": row["sentiment"],
#         }
#         for _, row in sampled.iterrows()
#     ]