"""MCP server exposing cx_agent.tools to Claude Code.

Data-source tools (filter_data, exclude, count_by, sentiment_breakdown) read
from the FABSA DataFrame loaded at startup.

Chainable tools (add_shares, add_polarisation, apply_min_volume, rank_top,
share_of, chi_squared) accept list[dict] inputs so results can be piped
between tool calls over MCP's JSON transport.

sample_reviews is disabled and not registered: the corpus is labels-only.
"""
import os
import pandas as pd
from fastmcp import FastMCP

from cx_agent import tools

# Load FABSA once at server startup
DF = pd.read_csv(os.environ["FABSA_CSV_PATH"])

mcp = FastMCP("fabsa")


# ---------------- Row-level slicing ----------------

@mcp.tool()
def filter_data(
    org_index: int | None = None,
    industry: str | None = None,
    parent_aspect: str | None = None,
    child_aspect: str | None = None,
    sentiment: str | None = None,
) -> list[dict]:
    """Keep only rows matching all provided filters. Any argument left as None
    is not applied. Returns matching rows as records.

    Example: filter_data(industry="Banking", child_aspect="app-website")
    """
    return tools.filter_data(
        DF, org_index, industry,
        parent_aspect, child_aspect, sentiment,
    ).to_dict("records")


@mcp.tool()
def exclude(
    rows: list[dict],
    org_index: int | None = None,
    industry: str | None = None,
    parent_aspect: str | None = None,
    child_aspect: str | None = None,
    sentiment: str | None = None,
) -> list[dict]:
    """Drop rows matching all provided filters from a prior slice. Useful for
    building peer/benchmark slices. Requires at least one filter.

    Example: exclude(banking_rows, org_index=514)  # peers only
    """
    return tools.exclude(
        pd.DataFrame(rows), org_index, industry,
        parent_aspect, child_aspect, sentiment,
    ).to_dict("records")


# ---------------- Aggregation ----------------

@mcp.tool()
def count_by(rows: list[dict], group_by: str) -> list[dict]:
    """Count rows in each group, sorted descending. Returns records with
    columns [group_by, 'count'].

    Example: count_by(negative_rows, group_by="child_aspect")
    """
    return tools.count_by(pd.DataFrame(rows), group_by).to_dict("records")


@mcp.tool()
def sentiment_breakdown(rows: list[dict], group_by: str) -> list[dict]:
    """Pivot rows into a sentiment-per-group table with columns
    [group_by, positive, negative, neutral, total].

    Example: sentiment_breakdown(banking_rows, group_by="parent_aspect")
    """
    return tools.sentiment_breakdown(pd.DataFrame(rows), group_by).to_dict("records")


# ---------------- Derived metrics ----------------

@mcp.tool()
def add_shares(breakdown: list[dict]) -> list[dict]:
    """Add pos_share, neg_share, neu_share to a sentiment breakdown. Input
    must have positive/negative/neutral/total columns (output of
    sentiment_breakdown).

    Example: add_shares(breakdown_rows)
    """
    return tools.add_shares(pd.DataFrame(breakdown)).to_dict("records")


@mcp.tool()
def add_polarisation(breakdown: list[dict]) -> list[dict]:
    """Add polarisation and pol_volume columns to a sentiment breakdown.
    Polarisation = min(positive, negative) / (positive + negative); peaks at
    0.5 for a 50/50 split. Use pol_volume with apply_min_volume before
    ranking.

    Example: add_polarisation(breakdown_rows)
    """
    return tools.add_polarisation(pd.DataFrame(breakdown)).to_dict("records")


# ---------------- Selection ----------------

@mcp.tool()
def apply_min_volume(
    rows: list[dict],
    min_volume: int = 30,
    col: str = "total",
) -> list[dict]:
    """Drop rows where the volume column is below min_volume. Use to exclude
    low-sample groups before ranking. Default col='total' (from
    sentiment_breakdown); use col='count' for count_by output, or
    col='pol_volume' after add_polarisation.

    Example: apply_min_volume(breakdown_rows, min_volume=30)
    """
    return tools.apply_min_volume(pd.DataFrame(rows), min_volume, col).to_dict("records")


@mcp.tool()
def rank_top(
    rows: list[dict],
    by: str,
    top_n: int = 3,
    ascending: bool = False,
) -> list[dict]:
    """Sort rows by a column and return the top N.

    Example: rank_top(scored_rows, by="neg_share", top_n=3)
    """
    return tools.rank_top(pd.DataFrame(rows), by, top_n, ascending).to_dict("records")


# ---------------- Scalar summaries ----------------

@mcp.tool()
def share_of(rows: list[dict], sentiment: str = "negative") -> dict:
    """Return the share of rows with the given sentiment in a slice. Returns
    {'n': total, 'count': matching, 'share': fraction} — share is None if
    the slice is empty.

    Example: share_of(banking_rows, sentiment="negative")
    """
    return tools.share_of(pd.DataFrame(rows), sentiment)


@mcp.tool()
def chi_squared(rows_a: list[dict], rows_b: list[dict]) -> dict:
    """Chi-squared test on the positive/negative/neutral counts of two
    slices. Returns test statistics and significance at p<0.05.

    Example: chi_squared(org_rows, peers_rows)
    """
    return tools.chi_squared(pd.DataFrame(rows_a), pd.DataFrame(rows_b))


# ---------------- Evidence ----------------

# DISABLED: the benchmark corpus is labels-only (no review text). The @mcp.tool() registration is
# commented out, so this tool is not advertised to the agent and cannot be called -- run_benchmark
# allows mcp__fabsa__*, which only covers tools this server actually registers.
# @mcp.tool()
# def sample_reviews(
#     rows: list[dict],
#     n: int = 3,
#     random_state: int = 42,
# ) -> list[dict]:
#     """Return n example reviews from a slice for verbatim evidence. Each
#     entry has text (truncated to 300 chars), parent_aspect, child_aspect,
#     sentiment.
#
#     Example: sample_reviews(negative_appweb_rows, n=3)
#     """
#     return tools.sample_reviews(pd.DataFrame(rows), n, random_state)


if __name__ == "__main__":
    mcp.run()