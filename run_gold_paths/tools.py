"""
tools.py — the five tools an agent uses to answer the benchmark questions.

Each tool is one step from gold_answers_manual.ipynb:

    filter      df[(df.industry == ...) & (df.aspect == ...)]
    summarise   groupby(col).agg(size, neg/neu/pos counts), the 30-mention floor,
                and the derived rates
    rank        rank(method="min") / sort_values / head(k), minus non-actionable topics
    ztest       proportions_ztest, with the four-cell sufficiency check reported
    answer      records the final natural-language answer

The agent does the reasoning. These tools transform data and hand back numbers;
interpreting them, doing the arithmetic between them, and writing the answer is the
model's job, exactly as it was mine in the notebook.

Usage
-----
    df = pd.read_csv("labels.csv")
    t  = Tools(df)

    t.filter(org="Northeast Bank")                        # {'id': 's1', 'n_rows': 812}
    t.summarise("s1", group_by="aspect", min_n=30)        # {'id': 't1', 'rows': [...]}
    t.rank("t1", by="neg_rate", top_k=3)                  # {'id': 't2', 'rows': [...]}
    t.answer("Staff attitude is the biggest driver ...")

    t.path()       # ['filter', 'summarise', 'rank', 'answer']
    t.final        # {'text': ..., 'abstain': False}
"""

import math

import pandas as pd
from statsmodels.stats.proportion import proportions_ztest

# Conventions, from the notebook's "Conventions" cell.
MIN_N = 30          # minimum mentions for a slice to be reportable
MIN_CELL = 10       # each of the z-test's four cells must exceed this
ALPHA = 0.05        # two-sided significance level
NON_ACTIONABLE = ["competitor", "general-satisfaction", "reviews"]

FILTER_FIELDS = ["industry", "org", "aspect", "parent_aspect", "sentiment", "source"]


class Tools:
    """The five tools, bound to one dataframe. One instance per task."""

    def __init__(self, df):
        self.df = df
        self.store = {}    # id -> DataFrame: selections "s1.." and tables "t1.."
        self.calls = []    # trajectory: one dict per tool call
        self.final = None  # set by answer()

    # -- internals ---------------------------------------------------------------

    def _put(self, prefix, frame):
        new_id = f"{prefix}{sum(1 for k in self.store if k.startswith(prefix)) + 1}"
        self.store[new_id] = frame
        return new_id

    def _get(self, ref):
        return self.store.get(ref)

    def _log(self, tool, args, result):
        self.calls.append(
            {"step": len(self.calls), "tool": tool, "args": args, "result": result}
        )
        return result

    # -- 1. filter ---------------------------------------------------------------

    def filter(self, industry=None, org=None, aspect=None, parent_aspect=None,
               sentiment=None, source=None, exclude_org=None):
        """Subset the reviews. Fields combine with AND; a list within a field is OR.

        exclude_org drops an organisation, to build a "rest of the industry" group
        that does not overlap with the organisation being compared against it.

        Returns a selection id and how many rows matched. n_rows = 0 is a valid
        result: it means the question names a slice that does not exist.
        """
        args = {k: v for k, v in locals().items() if k != "self" and v is not None}
        sub = self.df
        for field in FILTER_FIELDS:
            value = args.get(field)
            if value is not None:
                values = value if isinstance(value, list) else [value]
                sub = sub[sub[field].isin(values)]
        if exclude_org is not None:
            drop = exclude_org if isinstance(exclude_org, list) else [exclude_org]
            sub = sub[~sub["org"].isin(drop)]

        ref = self._put("s", sub)
        return self._log("filter", args, {"id": ref, "n_rows": int(len(sub))})

    # -- 2. summarise ------------------------------------------------------------

    def summarise(self, ref, group_by=None, min_n=None, min_n_column="total_count"):
        """Count reviews by sentiment, overall or per group, and derive the rates.

        Per row: total_count, neg_count, neu_count, pos_count,
                 neg_rate       = neg_count / total_count  (neutrals stay in the denominator)
                 neu_rate, pos_rate   the same share for the other two sentiments, so a
                                      question asking for a positive rate or a full
                                      breakdown is answered from the table, not by hand
                 prevalence     = total_count / rows in the selection
                 priority_score = prevalence * neg_rate

        group_by=None gives a single row for the whole selection.

        min_n drops groups below a volume floor, applied here rather than at ranking
        so the qualifying set is fixed before any rate is read. min_n_column chooses
        what the floor applies to: total_count for "topics with at least 30 mentions",
        neg_count for a rule stated on complaints. Dropped groups are named in the
        result, so a group falling below the floor stays visible.
        """
        args = {k: v for k, v in locals().items() if k != "self" and v is not None}
        sub = self._get(ref)
        if sub is None:
            return self._log("summarise", args, {"error": f"no selection called '{ref}'"})
        total = len(sub)
        if total == 0:
            return self._log("summarise", args,
                             {"error": f"selection '{ref}' is empty"})

        if group_by is None:
            grouped = sub.assign(_all="all").groupby("_all")
        elif group_by not in sub.columns:
            return self._log("summarise", args, {"error": f"no column '{group_by}'"})
        else:
            grouped = sub.groupby(group_by)

        out = grouped.agg(
            total_count=("sentiment", "size"),
            neg_count=("sentiment", lambda x: (x == "negative").sum()),
            neu_count=("sentiment", lambda x: (x == "neutral").sum()),
            pos_count=("sentiment", lambda x: (x == "positive").sum()),
        )
        out["neg_rate"] = out["neg_count"] / out["total_count"]
        out["neu_rate"] = out["neu_count"] / out["total_count"]
        out["pos_rate"] = out["pos_count"] / out["total_count"]
        out["prevalence"] = out["total_count"] / total
        out["priority_score"] = out["prevalence"] * out["neg_rate"]
        out.index.name = "group"
        out = out.reset_index()

        below = []
        if min_n is not None:
            if min_n_column not in out.columns:
                return self._log("summarise", args, {"error": f"no column '{min_n_column}'"})
            below = out[out[min_n_column] < min_n]["group"].tolist()
            out = out[out[min_n_column] >= min_n]

        out = out.sort_values("total_count", ascending=False)
        table_id = self._put("t", out)

        result = {"id": table_id, "selection_rows": int(total), "rows": _rows(out)}
        if below:
            result["below_min_n"] = below
        return self._log("summarise", args, result)

    # -- 3. rank -----------------------------------------------------------------

    def rank(self, ref, by="neg_rate", top_k=3, ascending=False, exclude=None):
        """Order a summarise table by one of its columns.

        exclude drops rows by group name; pass "non_actionable" for the topics an
        organisation cannot fix (competitor, general-satisfaction, reviews). Excluded
        groups are named in the result, so they can still be mentioned as findings.

        Ties share a rank (method="min") and top_k keeps every row with rank <= top_k,
        so a joint first place returns both rows rather than the sort inventing a winner.
        top_k=None returns every row, ordered.
        """
        args = {k: v for k, v in locals().items() if k != "self" and v is not None}
        # top_k is the one argument whose None is not "unset": it means the whole ranking,
        # while omitting it means the top 3. Log it either way, so a trajectory cannot be
        # read as asking for one when it asked for the other.
        args["top_k"] = top_k
        table = self._get(ref)
        if table is None:
            return self._log("rank", args, {"error": f"no table called '{ref}'"})
        if by not in table.columns:
            return self._log("rank", args, {
                "error": f"no column '{by}'",
                "available": [c for c in table.columns if c != "group"],
            })

        out = table.copy()
        excluded = []
        if exclude is not None:
            names = NON_ACTIONABLE if exclude == "non_actionable" else (
                exclude if isinstance(exclude, list) else [exclude])
            excluded = [g for g in out["group"] if g in names]
            out = out[~out["group"].isin(names)]

        if len(out) == 0:
            return self._log("rank", args, {
                "rows": [], "excluded": excluded,
            })

        out["rank"] = out[by].rank(method="min", ascending=ascending).astype(int)
        out = out.sort_values("rank", kind="stable")
        n_available = len(out)
        if top_k is not None:
            out = out[out["rank"] <= top_k]

        table_id = self._put("t", out)
        result = {"id": table_id, "rows": _rows(out), "n_groups_available": n_available}
        if excluded:
            result["excluded"] = excluded
        return self._log("rank", args, result)

    # -- 4. ztest ----------------------------------------------------------------

    def ztest(self, ref_a, ref_b, alternative="two-sided"):
        """Two-proportion z-test on the negative sentiment rates of two selections.

        Always reports the four cells, z, p and the gap. The two selections must not overlap. To compare an organisation with its
        industry, build the second side with filter(..., exclude_org=<org>).
        """
        args = {"ref_a": ref_a, "ref_b": ref_b, "alternative": alternative}
        cells = {}
        for side, ref in (("a", ref_a), ("b", ref_b)):
            sub = self._get(ref)
            if sub is None:
                return self._log("ztest", args, {"error": f"no selection called '{ref}'"})
            total = len(sub)
            neg = int((sub["sentiment"] == "negative").sum())
            cells[side] = {"total": total, "neg": neg, "non_neg": total - neg,
                           "neg_rate": (neg / total) if total else None}

        a, b = cells["a"], cells["b"]
        if a["total"] == 0 or b["total"] == 0:
            return self._log("ztest", args, {
                "group_a": _round(a), "group_b": _round(b),
                "error": "one side has no reviews, so there is no rate to test",
            })
        z, p = proportions_ztest(count=[a["neg"], b["neg"]],
                                 nobs=[a["total"], b["total"]],
                                 alternative=alternative)
        z, p = float(z), float(p)
        usable = math.isfinite(z) and math.isfinite(p)

        result = {
            "group_a": _round(a),
            "group_b": _round(b),
            "gap_pp": round((a["neg_rate"] - b["neg_rate"]) * 100, 1),
            "z": round(z, 3) if usable else None,
            "p_value": round(p, 4) if usable else None,
            "significant": bool(p < ALPHA) if usable else None,
        }
        if not usable:
            result["note"] = ("both sides are entirely negative or entirely non-negative, "
                              "so no z-test is defined")
        return self._log("ztest", args, result)

    # -- 5. answer ---------------------------------------------------------------

    def answer(self, text, abstain=False):
        """Record the final answer and stop. Terminal call.

        Set abstain=True when the data cannot support the question as asked - an empty
        slice, a slice below the volume floor, a "top 3" where fewer than 3 topics
        exist, or four cells too thin to test - and say so in the text.
        """
        self.final = {"text": text, "abstain": bool(abstain)}
        return self._log("answer", {"abstain": bool(abstain)}, {"recorded": True})

    # -- trajectory --------------------------------------------------------------

    def path(self):
        """The tools called, in order: ['filter', 'summarise', 'rank', 'answer']."""
        return [c["tool"] for c in self.calls]


def _rows(frame, nd=4):
    """DataFrame -> list of plain dicts, floats rounded for readability."""
    records = []
    for rec in frame.to_dict("records"):
        row = {}
        for k, v in rec.items():
            if isinstance(v, str):
                row[k] = v
            elif isinstance(v, float):
                row[k] = round(v, nd)
            else:
                row[k] = int(v)
        records.append(row)
    return records


def _round(d, nd=4):
    return {k: (round(v, nd) if isinstance(v, float) else v) for k, v in d.items()}
