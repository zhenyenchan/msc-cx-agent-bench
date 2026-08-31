"""LLM-judge for the insight score (0-5), via the Chattermill gateway.

Grades each agent answer against the gold answer with a reference-based
rubric prompt; the judge never sees the underlying data. Defaults to
Gemini 3.1 Pro at temperature 0.0 on all 50 rows of
tagging/baseline_answers.csv.

    python -m cx_agent_bench.run_judge
    python -m cx_agent_bench.run_judge --model vertex_ai/gemini-2.5-pro
    python -m cx_agent_bench.run_judge --limit 3

Needs OPENAI_BASE_URL and OPENAI_API_KEY in .env, as in run_agent.py.
Writes task_id, judge_score, judge_reasoning (the full response text) and
per-call cost to tagging/judge_scores.csv, which full_results.ipynb joins
onto the human annotation table.
"""

import argparse
import re
import time

import pandas as pd

from .tasks_public import REPO_ROOT

# The gateway pins europe-west4, which serves no gemini-3.x model (404); the
# per-request vertex_location=global override reaches the location that does.
JUDGE_DEFAULT_MODEL = "vertex_ai/gemini-3.1-pro-preview"
JUDGE_DEFAULT_VERTEX_LOCATION = "global"
ANSWERS_CSV = REPO_ROOT / "tagging" / "baseline_answers.csv"
SCORES_CSV = REPO_ROOT / "tagging" / "judge_scores.csv"

SYSTEM_PROMPT = """\
You are a senior data analyst evaluating an AI agent's response to an analytics query on customer feedback data. Your task is to grade the quality of the agent's provided answer by comparing it directly against a trusted, expert-written gold-standard answer. Assign an Insight Score (from 0 to 5) that measures the correctness and completeness against the gold answer.

You will not see the original customer data - you don’t need it because you’re not working out the right answer yourself. You're only comparing the agent's answer to the gold answer provided.

Every gold answer has the same structure below. Use this as your checklist when comparing:
● The verdict — the direct answer to the question (yes or no, a top-3 list, the single biggest problem etc.)
● The numbers — the supporting figures (percentages, counts).
● The catch — some questions specifically ask for a key point. For example, a gold answer might say whether a difference is big enough to be meaningful ("statistically significant"). Another gold answer might include a warning that one number is based on too few reviews to trust (“sample size is too small”). Not every question has a catch.
"""

USER_TEMPLATE = """\
Below is the user’s question, the trusted gold answer and the agent’s answer.

### Question:
{question}

### Gold answer:
{gold_answer}

### Agent answer to evaluate:
{agent_answer}

### Evaluation Rules:
● Getting the direction right matters most. Saying yes when the answer is no, naming the wrong problem, or putting a ranking in the wrong order is a big mistake, even if the answer reads well.
● If the agent doesn’t include the “catch” when the question asks for it, it can’t score higher than 3.
● If the agent doesn't answer the question, it can't score higher than 2.
● Extra claims are neither rewarded nor penalised unless they contradict the gold answer.


### Evaluation Rubric:
- 5: Relevant, accurate, and complete. Right verdict, all key numbers right, the "catch" and all requested parts included, nothing that contradicts the gold answer.
- 4: Relevant and mostly accurate. One non-essential part is wrong or missing, but the recommended actions are correct.
- 3: Relevant and correct direction, but a key number is wrong or missing, or the required "catch" is left out.
- 2: The answer sounds relevant but is inaccurate. Wrong verdict, no supporting numbers or a made-up/contradicting claim.
- 1: Barely relevant. The main answer is wrong or so vague that there is almost nothing you can check.
- 0: Doesn't answer the question asked. Wrong, contradicts itself, off-topic, a refusal, or blank.

### Evaluation Steps:
1. Read the question and the gold answer carefully to understand the exact insights that should have been discovered.
2. Analyse the agent's answer. Check for correctness and completeness.
3. Output a step-by-step reasoning trace (not more than 150 words) explaining why the agent's answer does or does not meet the gold-standard criteria. Note any specific factual errors, omissions or style variations (which does not affect the score).
4. Assign a final integer score from 0 to 5 based on the evaluation rubric. You must wrap the final score inside <insight_score></insight_score> tags.

### Output Format:
[Write your step-by-step reasoning here]

<insight_score>Your final integer score (0-5)</insight_score>
"""

SCORE_RE = re.compile(r"<insight_score>\s*(\d)\s*</insight_score>")


def make_client():
    import os

    from openai import OpenAI
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    for var in ("OPENAI_API_KEY", "OPENAI_BASE_URL"):
        if not os.environ.get(var):
            raise SystemExit(f"{var} not set (.env or environment)")
    return OpenAI(base_url=os.environ["OPENAI_BASE_URL"],
                  api_key=os.environ["OPENAI_API_KEY"], max_retries=3)


def judge_one(client, model, row, temperature=0.0, seed=42,
              vertex_location=JUDGE_DEFAULT_VERTEX_LOCATION):
    """One graded row -> dict with the parsed 0-5 score, raw text, cost."""
    t0 = time.perf_counter()
    raw_http = client.chat.completions.with_raw_response.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                question=row["question"],
                gold_answer=row["gold_answer"],
                agent_answer=row["agent_answer"])},
        ],
        temperature=temperature,
        seed=seed,
        extra_body={"vertex_location": vertex_location} if vertex_location else {},
    )
    latency = time.perf_counter() - t0
    response = raw_http.parse()
    try:
        cost = float(raw_http.headers.get("x-litellm-response-cost"))
    except (TypeError, ValueError):
        cost = None
    text = response.choices[0].message.content or ""
    m = SCORE_RE.search(text)
    score = int(m.group(1)) if m and 0 <= int(m.group(1)) <= 5 else None
    return {"task_id": row["task_id"], "judge_score": score,
            "judge_reasoning": text, "latency_s": round(latency, 2),
            "cost_usd": cost}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=JUDGE_DEFAULT_MODEL,
                        help=f"gateway model id (default: {JUDGE_DEFAULT_MODEL})")
    parser.add_argument("--answers", default=str(ANSWERS_CSV),
                        help="csv with task_id, question, agent_answer, gold_answer")
    parser.add_argument("--out", default=str(SCORES_CSV))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vertex-location", default=JUDGE_DEFAULT_VERTEX_LOCATION,
                        help="per-request Vertex location override; '' to use "
                             "the gateway's pinned region")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    answers = pd.read_csv(args.answers, encoding="utf-8-sig")
    if args.limit:
        answers = answers.head(args.limit)
    client = make_client()

    results = []
    for _, row in answers.iterrows():
        result = judge_one(client, args.model, row,
                           temperature=args.temperature, seed=args.seed,
                           vertex_location=args.vertex_location)
        results.append(result)
        score = result["judge_score"]
        print(f"{row['task_id']:<8} score={'?' if score is None else score}"
              f"  ({result['latency_s']:.1f}s)")

    out = pd.DataFrame(results)
    out.to_csv(args.out, index=False, encoding="utf-8")
    n_parsed = out["judge_score"].notna().sum()
    total_cost = out["cost_usd"].sum()
    print(f"\n{n_parsed}/{len(out)} scores parsed, "
          f"mean={out['judge_score'].mean():.2f}, "
          f"total cost=${total_cost:.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
