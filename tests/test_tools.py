"""R7: filter observations must be honest about *why* a selection is small.

Two failure modes observed in baseline traces made abstention unfalsifiable:
malformed values ('', {}, []) and unrecognised names ('Vella & Co.') both came
back as plain status-ok zero-row selections, indistinguishable from a slice
that legitimately does not exist.
"""

from cx_agent_bench.harness import load_tools_module
from cx_agent_bench.tasks_public import load_dataset


def _tools(dataset="hard"):
    return load_tools_module().Tools(load_dataset(dataset))


def test_malformed_filter_values_are_errors():
    t = _tools()
    for bad in ("", {}, [], {"enum": ["speed"]}, ["app-website", ""]):
        assert "error" in t.filter(aspect=bad)


def test_unrecognised_value_is_flagged_in_a_note():
    t = _tools()
    result = t.filter(org="Vella & Co.", industry="Price Comparison")
    assert result["n_rows"] == 0
    assert "'Vella & Co.'" in result["note"] and "org" in result["note"]

    result = t.filter(exclude_org="Vella & Co.")
    assert "note" in result


def test_empty_slice_from_recognised_labels_has_no_note():
    # Kestrel Bank exists and Fashion exists, but not together: a slice that
    # genuinely does not exist stays a plain zero-row ok result.
    result = _tools().filter(org="Kestrel Bank", industry="Fashion")
    assert result["n_rows"] == 0
    assert "note" not in result


def test_rank_echoes_only_the_ranking_columns():
    t = _tools()
    sel = t.filter(org="Vella & Co")
    table = t.summarise(sel["id"], group_by="aspect", min_n=30)
    ranked = t.rank(table["id"], by="neg_rate", top_k=3)
    assert set(ranked["rows"][0]) == {"group", "rank", "neg_rate", "neg_count",
                                      "total_count"}
    # the handle still holds the full table: re-ranking by another column works
    reranked = t.rank(ranked["id"], by="priority_score", top_k=1)
    assert "priority_score" in reranked["rows"][0]


def test_ztest_reports_cell_sufficiency():
    t = _tools()
    # T5-H3's slices: 20 and 25 mentions, thin cells -> unreliable test
    a = t.filter(org="Northpeak Trading", aspect="speed")
    b = t.filter(org="Quantly", aspect="speed")
    thin = t.ztest(a["id"], b["id"])
    assert thin["sufficient"] is False
    assert thin["p_value"] is not None       # still reported; the agent judges

    # T5-E1's slices: every cell exceeds MIN_CELL -> reliable
    a = t.filter(org="CompareHive", aspect="attitude-of-staff")
    b = t.filter(org="Tallywise", aspect="attitude-of-staff")
    assert t.ztest(a["id"], b["id"])["sufficient"] is True


def test_recognised_values_have_no_note():
    result = _tools().filter(org="Vella & Co")
    assert result["n_rows"] > 0
    assert "note" not in result
