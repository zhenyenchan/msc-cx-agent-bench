"""Build the 50-question set for the v2 benchmark and validate every parameter
choice against the generated easy/hard label files.

Layout: 10 templates x 5 questions = 50 tasks, split 20 Easy / 20 Hard answerable
/ 10 Hard abstention. The per-template split and the reason each abstention fails
are declared in DISTRIBUTION and enforced by the validators below -- a question
whose parameters do not actually produce the declared failure is reported as a
problem and must be re-picked.
"""
import csv
import math

import pandas as pd
from scipy.stats import norm

BASE = r"C:\msc-cx-agent-bench\benchmark_outputs\v2"
OUT = BASE
MIN_N = 30
MAX_OFFSET = 0.25  # the largest planted offset; see data_generation_v2.ipynb

easy = pd.read_csv(BASE + r"\easy\labels.csv")
hard = pd.read_csv(BASE + r"\hard\labels.csv")
DATA = {"easy": easy, "hard": hard}

# Ground truth for T5. A planted null is the industry's tied aspect held at offset 0.0 for the
# first two orgs, so their negative rates are equal by construction and the cells are floored at
# MIN_CELL in hard too (protected from thinning) -- a true null with the power to be tested.
OFFSETS = pd.read_csv(BASE + r"\offsets.csv")
NULL_PAIRS = {ind: (g["child_aspect"].iloc[0], tuple(g["org"]))
              for ind, g in OFFSETS[OFFSETS["is_tied_aspect"]].groupby("industry")}
assert all(len(o) == 2 for _, o in NULL_PAIRS.values()), "each null pair must hold exactly 2 orgs"
OFFSET_OF = {(r.org, r.child_aspect): r.offset for r in OFFSETS.itertuples()}

# The industry x child_aspect groups the hard generator leaves empty (it reproduces
# FABSA's holes). Abstentions with reason ABSENT must point at one of these.
ALL_ASPECTS = sorted(hard["child_aspect"].unique())
_present = set(map(tuple, hard[["industry", "child_aspect"]].drop_duplicates().values))
ABSENT_GROUPS = {(i, a) for i in sorted(hard["industry"].unique())
                 for a in ALL_ASPECTS if (i, a) not in _present}
assert len(ABSENT_GROUPS) == 29, f"expected 29 absent groups, found {len(ABSENT_GROUPS)}"

# abstention reason codes -> the text used in the deliverables
LOW_N = "Cells have less than 30 reviews"
ABSENT = "Industry x aspect pair absent"
FEW_CATS = "Too few categories to rank"
PARTIAL = ("Partial - can still compute overall sentiment but driver ranking "
           "and industry comparison fails")

# T5 design codes: what the parameters are chosen to plant
NULL_PAIR = "Null pair, tied aspect"
MAX_VS_ZERO = f"Org at offset {MAX_OFFSET} vs peer at 0.0, same aspect"
INSUFFICIENT = "Insufficient data"

ASPECT = {
    "account-access": "account access",
    "app-website": "the app or website",
    "attitude-of-staff": "staff attitude",
    "competitor": "competitors",
    "discounts-promotions": "discounts and promotions",
    "ease-of-use": "ease of use",
    "email": "email support",
    "general-satisfaction": "general satisfaction",
    "phone": "phone support",
    "price-value-for-money": "price and value for money",
    "reviews": "reviews",
    "speed": "speed",
}

# What aspect and sentiment mean for templates that do not take them as a filter: the aspect is
# the dimension being ranked or compared across, and the sentiment is fixed by the metric the
# question names (a negative rate, a complaint count, or a full breakdown).
RANKED = "all 12 (ranked across)"
ALL_SENT = "all (breakdown)"
ASPECT_ROLE = {"T2": "all 12 (not filtered)", "T3": RANKED, "T7": RANKED, "T8": RANKED,
               "T9": RANKED, "T10": RANKED}
SENTIMENT_ROLE = {"T2": ALL_SENT, "T3": "negative", "T6": "negative", "T7": "negative",
                  "T8": "negative", "T9": "negative",
                  "T10": "all (overview), negative (drivers)"}

TASK_TYPE = {**{t: "Descriptive" for t in ("T1", "T2", "T3")},
             **{t: "Diagnostic" for t in ("T4", "T5", "T6", "T7")},
             **{t: "Prescriptive" for t in ("T8", "T9", "T10")}}

TEMPLATES = {
    "T1": ("T1- Single value retrieval",
           "What proportion of {seg} reviews that mention {aspect} are {sent}?"),
    "T2": ("T2- Distribution",
           "What is the sentiment breakdown across all {seg} reviews?"),
    "T3": ("T3- Ranking by volume",
           "What are the top 3 topics with the most number of complaints in {seg}?"),
    "T4": ("T4- Cross-segment comparison",
           "Is {sent} sentiment about {aspect} higher in {a} or {b}?"),
    "T5": ("T5- Statistical testing",
           "Do {a} and {b} have significantly different {sent} rates for {aspect}?"),
    "T6": ("T6- Organisation vs industry comparison",
           "How do individual organisations in {seg} compare to the industry average on {aspect}?"
           " Give me each organisation's negative rate and ignore any organisations with less than"
           " 30 mentions about {aspect}."),
    "T7": ("T7- Driver identification by severity",
           "What is the biggest driver of dissatisfaction at {seg}? Rank topics by negative rate"
           " and ignore any topics with less than 30 mentions."),
    "T8": ("T8- Prioritisation by a named rule",
           "Which 2 issues should {seg} address first? Define Priority = 0.5 x minmax(negative rate)"
           " + 0.5 x minmax(volume) and ignore any topics with fewer than 30 mentions."),
    "T9": ("T9- Comparative diagnosis and recommendation",
           "Why does {a} have a higher complaint rate than {b}? What should {a} improve first?"),
    "T10": ("T10- Full CX report",
            "Write a CX report for {seg}. Cover overall sentiment, top pain points ranked by"
            " negative rate, and how the top issue compares with the industry average. Also give"
            " me a prioritised improvement roadmap."),
}

# template -> (n easy, n hard answerable, n hard abstention, abstention reason)
DISTRIBUTION = {
    "T1": (2, 2, 1, LOW_N),
    "T2": (2, 3, 0, None),
    "T3": (2, 2, 1, FEW_CATS),
    "T4": (2, 1, 2, ABSENT),
    "T5": (2, 1, 2, LOW_N),
    "T6": (2, 1, 2, ABSENT),
    "T7": (2, 2, 1, LOW_N),
    "T8": (2, 3, 0, None),
    "T9": (2, 3, 0, None),
    "T10": (2, 2, 1, PARTIAL),
}

# (template, dataset, kind, params); kind: E = easy, H = hard answerable,
# or an abstention reason code. Hard answerable questions come before abstentions.
ROWS = [
 ("T1", "easy", "E", dict(seg="Banking", aspect="app-website", sent="negative")),
 ("T1", "easy", "E", dict(seg="Fashion", aspect="ease-of-use", sent="positive")),
 ("T1", "hard", "H", dict(seg="Fashion", aspect="speed", sent="negative")),
 ("T1", "hard", "H", dict(seg="Price Comparison", aspect="attitude-of-staff", sent="negative")),
 ("T1", "hard", LOW_N, dict(seg="Ride Hailing", aspect="discounts-promotions", sent="negative")),

 # T2 breaks the whole segment down by sentiment, so the aspect is not a parameter here
 ("T2", "easy", "E", dict(seg="Ride Hailing")),
 ("T2", "easy", "E", dict(seg="Marbrook")),
 ("T2", "hard", "H", dict(seg="Vanter Financial")),
 ("T2", "hard", "H", dict(seg="PricePilot")),
 ("T2", "hard", "H", dict(seg="Sable Row")),

 ("T3", "easy", "E", dict(seg="Travel Booking")),
 ("T3", "easy", "E", dict(seg="Groceries")),
 ("T3", "hard", "H", dict(seg="Fashion")),
 ("T3", "hard", "H", dict(seg="Price Comparison")),
 ("T3", "hard", FEW_CATS, dict(seg="Consulting")),

 ("T4", "easy", "E", dict(a="Banking", b="Ride Hailing", aspect="price-value-for-money", sent="negative")),
 ("T4", "easy", "E", dict(a="Fashion", b="Travel Booking", aspect="app-website", sent="negative")),
 ("T4", "hard", "H", dict(a="Groceries", b="Trading", aspect="ease-of-use", sent="negative")),
 # one side absent: the agent gets a real rate for Banking and must still abstain
 ("T4", "hard", ABSENT, dict(a="Banking", b="Consulting", aspect="ease-of-use", sent="negative")),
 # both sides absent
 ("T4", "hard", ABSENT, dict(a="Consulting", b="Streaming", aspect="price-value-for-money", sent="negative")),

 # T5 is tested against planted ground truth, so all five slots are org-vs-org.
 # Price Comparison's null carries the largest cells (103/118), so "not significant" there is a
 # real result rather than an underpowered one.
 ("T5", "easy", "E", dict(a="CompareHive", b="Tallywise", aspect="attitude-of-staff",
                          sent="negative", design=NULL_PAIR, answer="Not significant")),
 ("T5", "easy", "E", dict(a="Trippa", b="Roamly", aspect="general-satisfaction",
                          sent="negative", design=MAX_VS_ZERO, answer="Significant")),
 # in hard the null cells are protected from thinning, so both still clear the 40 floor
 ("T5", "hard", "H", dict(a="Swiftly Rides", b="Kerbside", aspect="email",
                          sent="negative", design=NULL_PAIR, answer="Not significant")),
 ("T5", "hard", LOW_N, dict(a="Halden Savings", b="Kestrel Bank",
                            aspect="price-value-for-money", sent="negative")),
 ("T5", "hard", LOW_N, dict(a="Northpeak Trading", b="Quantly", aspect="speed", sent="negative")),

 ("T6", "easy", "E", dict(seg="Banking", aspect="app-website")),
 ("T6", "easy", "E", dict(seg="Travel Booking", aspect="general-satisfaction")),
 ("T6", "hard", "H", dict(seg="Fashion", aspect="app-website")),
 ("T6", "hard", ABSENT, dict(seg="Consulting", aspect="app-website")),
 ("T6", "hard", ABSENT, dict(seg="Streaming", aspect="general-satisfaction")),

 ("T7", "easy", "E", dict(seg="Northeast Bank")),
 ("T7", "easy", "E", dict(seg="Wayfare")),
 ("T7", "hard", "H", dict(seg="Sable Row")),
 ("T7", "hard", "H", dict(seg="Investa")),
 ("T7", "hard", LOW_N, dict(seg="Pinecast")),

 ("T8", "easy", "E", dict(seg="Wayfare")),
 ("T8", "easy", "E", dict(seg="Larkmead Market")),
 ("T8", "hard", "H", dict(seg="Northerly")),
 ("T8", "hard", "H", dict(seg="CompareHive")),
 ("T8", "hard", "H", dict(seg="Quantly")),

 ("T9", "easy", "E", dict(a="Halden Savings", b="Kestrel Bank")),
 ("T9", "easy", "E", dict(a="CompareHive", b="PricePilot")),
 ("T9", "hard", "H", dict(a="Sable Row", b="Northerly")),
 ("T9", "hard", "H", dict(a="Larkmead Market", b="Oakpan Grocers")),
 ("T9", "hard", "H", dict(a="Halden Savings", b="Northeast Bank")),

 ("T10", "easy", "E", dict(seg="Trippa", industry="Travel Booking")),
 ("T10", "easy", "E", dict(seg="Kestrel Bank", industry="Banking")),
 ("T10", "hard", "H", dict(seg="Vella & Co", industry="Fashion")),
 ("T10", "hard", "H", dict(seg="Freshbury", industry="Groceries")),
 ("T10", "hard", PARTIAL, dict(seg="Lumora", industry="Streaming")),
]

problems = []


def seg_mask(df, seg):
    return df["industry"] == seg if seg in set(df["industry"]) else df["org"] == seg


def industry_of(mode, seg):
    """The industry a segment sits in: itself if it is an industry, else the org's."""
    df = DATA[mode]
    if seg in set(df["industry"]):
        return seg
    return df.loc[df["org"] == seg, "industry"].iloc[0]


def n_cell(mode, seg, aspect=None):
    df = DATA[mode]
    m = seg_mask(df, seg)
    if aspect:
        m &= df["child_aspect"] == aspect
    return int(m.sum())


def neg_rate(mode, seg):
    """(negative rate, negative count) for a segment."""
    df = DATA[mode]
    s = df[seg_mask(df, seg)]
    neg = int((s["sentiment"] == "negative").sum())
    return neg / len(s), neg


def two_prop(mode, a, b, aspect, sent):
    """Two-proportion z-test, matching notebooks/tools.py two_prop_test."""
    df = DATA[mode]
    sa = df[seg_mask(df, a) & (df["child_aspect"] == aspect)]
    sb = df[seg_mask(df, b) & (df["child_aspect"] == aspect)]
    n1, n2 = len(sa), len(sb)
    if not n1 or not n2:
        return None
    x1, x2 = int((sa["sentiment"] == sent).sum()), int((sb["sentiment"] == sent).sum())
    p1, p2 = x1 / n1, x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else 0.0
    pv = float(2 * norm.sf(abs(z)))
    return dict(n1=n1, n2=n2, p1=p1, p2=p2, p_value=pv, significant=pv < 0.05)


def check_design(tpl, mode, p):
    """For T5: assert the parameters really sit on the planted structure they claim, and that
    the test outcome matches the designed answer."""
    design, want_sig = p["design"], p["answer"] == "Significant"
    a, b, aspect = p["a"], p["b"], p["aspect"]
    ind = industry_of(mode, a)
    if design == NULL_PAIR:
        tied, orgs = NULL_PAIRS.get(ind, (None, ()))
        if (aspect, {a, b}) != (tied, set(orgs)):
            problems.append(f"{tpl} {design}: {ind} null pair is {orgs} on {tied}, got "
                            f"{(a, b)} on {aspect}")
    elif design == MAX_VS_ZERO:
        oa, ob = OFFSET_OF.get((a, aspect)), OFFSET_OF.get((b, aspect))
        if (oa, ob) != (MAX_OFFSET, 0.0):
            problems.append(f"{tpl} {design}: offsets are {oa}/{ob}, expected {MAX_OFFSET}/0.0")
    t = two_prop(mode, a, b, aspect, p["sent"])
    if t is None:
        problems.append(f"{tpl} {design}: empty slice for {a}/{b} x {aspect} in {mode}")
        return ""
    if t["significant"] != want_sig:
        problems.append(f'{tpl} {design}: p={t["p_value"]:.4f} gives significant='
                        f'{t["significant"]}, designed answer was "{p["answer"]}"')
    return (f'{design.lower()}; rates {t["p1"]:.0%} vs {t["p2"]:.0%}, '
            f'p={t["p_value"]:.4f} -> {p["answer"].lower()}')


def topics_over_min(mode, seg):
    df = DATA[mode]
    c = df[seg_mask(df, seg)].groupby("child_aspect").size()
    return sorted(c[c >= MIN_N].index)


def orgs_over_min(mode, industry, aspect):
    df = DATA[mode]
    c = df[(df["industry"] == industry) & (df["child_aspect"] == aspect)].groupby("org").size()
    return sorted(c[c >= MIN_N].index)


def segments(p):
    return [p["seg"]] if "seg" in p else [p["a"], p["b"]]


def render(tpl, p):
    q = dict(p)
    if "aspect" in q:
        q["aspect"] = ASPECT[q["aspect"]]
    return TEMPLATES[tpl][1].format(**q)


def evidence(tpl, mode, p):
    """(evidence note, answerable) for a parameter set."""
    if tpl == "T1":
        n = n_cell(mode, p["seg"], p["aspect"])
        return f'n({p["seg"]} x {p["aspect"]})={n}', n >= MIN_N
    if tpl == "T2":
        n = n_cell(mode, p["seg"])
        return f'n({p["seg"]})={n}', n >= MIN_N
    if tpl in ("T4", "T5"):
        na, nb = n_cell(mode, p["a"], p["aspect"]), n_cell(mode, p["b"], p["aspect"])
        return f'n({p["a"]})={na}, n({p["b"]})={nb}', na >= MIN_N and nb >= MIN_N
    if tpl == "T3":
        t = topics_over_min(mode, p["seg"])
        return (f'n({p["seg"]})={n_cell(mode, p["seg"])}, topics with n>={MIN_N}: {len(t)} of 12',
                len(t) >= 3)
    if tpl == "T6":
        o = orgs_over_min(mode, p["seg"], p["aspect"])
        n = n_cell(mode, p["seg"], p["aspect"])
        return (f'n({p["seg"]} x {p["aspect"]})={n}, orgs with n>={MIN_N}: {len(o)}'
                f'{" (" + ", ".join(o) + ")" if o else ""}', len(o) >= 2)
    if tpl in ("T7", "T8", "T10"):
        need = {"T7": 1, "T8": 2, "T10": 3}[tpl]
        t = topics_over_min(mode, p["seg"])
        return (f'n({p["seg"]})={n_cell(mode, p["seg"])}, topics with n>={MIN_N}: {len(t)}',
                len(t) >= need)
    if tpl == "T9":
        ta, tb = topics_over_min(mode, p["a"]), topics_over_min(mode, p["b"])
        shared = set(ta) & set(tb)
        ra, rb = neg_rate(mode, p["a"]), neg_rate(mode, p["b"])
        if not (ra[0] > rb[0] and ra[1] > rb[1]):  # the question presupposes a is worse
            problems.append(f'T9 premise fails ({mode}): {p["a"]} {ra} vs {p["b"]} {rb}')
        return (f'neg rate {p["a"]} {ra[0]:.0%} (n={ra[1]}) > {p["b"]} {rb[0]:.0%} (n={rb[1]}); '
                f'shared topics with n>={MIN_N}: {len(shared)}', len(shared) >= 2)
    raise ValueError(tpl)


def check_reason(tpl, mode, p, reason):
    """Assert the abstention fails for the declared reason, not some other way."""
    cells = [(s, p["aspect"]) for s in segments(p)] if "aspect" in p else []
    if reason == LOW_N:
        ns = [n_cell(mode, s, a) for s, a in cells] or [n_cell(mode, p["seg"])]
        if min(ns) == 0:
            problems.append(f"{tpl} {reason}: cell is absent, not thin -> {p}")
        elif min(ns) >= MIN_N:
            problems.append(f"{tpl} {reason}: no cell below {MIN_N} -> {p} {ns}")
    elif reason == ABSENT:
        hit = [(industry_of(mode, s), a) for s, a in cells
               if (industry_of(mode, s), a) in ABSENT_GROUPS]
        if not hit:
            problems.append(f"{tpl} {reason}: no side is a planted-absent group -> {p}")
        return "absent groups: " + "; ".join(f"{i} x {a}" for i, a in hit)
    elif reason in (FEW_CATS, PARTIAL):
        t = topics_over_min(mode, p["seg"])
        if n_cell(mode, p["seg"]) < MIN_N:
            problems.append(f"{tpl} {reason}: segment itself is below {MIN_N} -> {p}")
        if len(t) >= 3:
            problems.append(f"{tpl} {reason}: 3+ topics clear {MIN_N} -> {p}")
        if reason == PARTIAL:
            # the industry comparison fails downstream: with no ranked top issue there is
            # nothing to compare, even where the industry cell itself is large enough
            ind = [f"{p['industry']} x {a} n={n_cell(mode, p['industry'], a)}" for a in t]
            return ("overall sentiment computable; no ranked top issue "
                    f"({len(t)} topic(s) over {MIN_N}), so the industry comparison has no input"
                    + (" [" + "; ".join(ind) + "]" if ind else ""))
    return ""


# ---- build ----
meta, slot = [], {}
for tpl, mode, kind, p in ROWS:
    letter = "E" if kind == "E" else "H"  # slots are numbered within difficulty, per the task spec
    slot[tpl, letter] = slot.get((tpl, letter), 0) + 1
    q = render(tpl, p)
    note, answerable = evidence(tpl, mode, p)
    is_abstention = kind not in ("E", "H")
    if answerable == is_abstention:
        problems.append(f"{tpl} {kind}: answerable={answerable}, expected {not is_abstention} | {p}")
    extra = check_reason(tpl, mode, p, kind) if is_abstention else ""
    if "design" in p:
        extra = check_design(tpl, mode, p)
    meta.append(dict(
        task_id=f"{tpl}-{letter}{slot[tpl, letter]}",
        task_type=TASK_TYPE[tpl], template=TEMPLATES[tpl][0], dataset=mode,
        answerable="no (abstain)" if is_abstention else "yes",
        abstention_reason=kind if is_abstention else "",
        designed_answer=p.get("answer", INSUFFICIENT if is_abstention else ""),
        industry=" vs ".join(dict.fromkeys(industry_of(mode, s) for s in segments(p))),
        segment=p.get("seg") or f'{p["a"]} vs {p["b"]}',
        aspect=ASPECT[p["aspect"]] if "aspect" in p else ASPECT_ROLE[tpl],
        sentiment=p.get("sent") or SENTIMENT_ROLE[tpl],
        volume_check=note + (" | " + extra if extra else ""),
        question=q))

# declared distribution must match what ROWS actually contains
for tpl, (n_e, n_h, n_a, reason) in DISTRIBUTION.items():
    got = [r[2] for r in ROWS if r[0] == tpl]
    want = ["E"] * n_e + ["H"] * n_h + [reason] * n_a
    if sorted(map(str, got)) != sorted(map(str, want)):
        problems.append(f"{tpl} distribution: declared {want}, found {got}")

# ---- write ----
with open(OUT + r"\question_set_v2.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["task_id", "template", "question"])
    for m in meta:
        w.writerow([m["task_id"], m["template"], m["question"]])

pd.DataFrame(meta).to_csv(OUT + r"\question_set_v2_params.csv", index=False, encoding="utf-8-sig")

dist = [dict(zip(["Task type", "Template", "# tasks (Easy, all answerable)",
                  "# tasks (Hard, answerable)", "# tasks (Hard, abstention)", "Abstention reason"],
                 [TASK_TYPE[t], TEMPLATES[t][0], *DISTRIBUTION[t][:3], DISTRIBUTION[t][3] or ""]))
        for t in TEMPLATES]
dist.append({"Task type": "", "Template": "Total", "# tasks (Easy, all answerable)": 20,
             "# tasks (Hard, answerable)": 20, "# tasks (Hard, abstention)": 10,
             "Abstention reason": ""})
pd.DataFrame(dist).to_csv(OUT + r"\question_set_v2_distribution.csv", index=False, encoding="utf-8-sig")

# ---- report ----
counts = pd.DataFrame(meta).groupby(["dataset", "answerable"]).size()
print(f"rows: {len(ROWS)} | " + " | ".join(f"{k}: {v}" for k, v in counts.items()))
print("validation problems:", len(problems))
for x in problems:
    print("  !", x)
print()
for m in meta:
    print(f'{m["task_id"]:8} {m["dataset"]:5} {m["abstention_reason"][:34]:34} {m["volume_check"]}')
