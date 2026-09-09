"""
Hand-assigned scores for eval/results/human_review_sample.jsonl, used to measure
LLM-judge/human agreement (eval/judge_agreement.py). Same rubric the judge uses
(agent/llm.py's judge system prompt / eval/llm_judge.py): grounded,
correct_helpful, tone_brand_fit, actionable, each 1-5.

Scored by the author, reading each (message, candidate reply, reference reply)
independently against the rubric text -- not by re-deriving the judge's own
rationale. Trivial-system rows share the same canned reply verbatim, so they're
scored on a consistent, near-identical baseline (grounded=2, correct_helpful=1,
tone=3, actionable=1) rather than re-litigated per row, with lower scores only
where the specific reference makes the mismatch starker (e.g. a reply that
promises to "look into" something that's actually a simple factual/licensing
question). See report.md §4/§6 for what the resulting agreement numbers do and
don't show.
"""

HUMAN_SCORES = {
    "trivial__g039": {"grounded": 2, "correct_helpful": 1, "tone_brand_fit": 3, "actionable": 1},
    "trivial__g150": {"grounded": 2, "correct_helpful": 1, "tone_brand_fit": 3, "actionable": 1},
    "trivial__g032": {"grounded": 2, "correct_helpful": 1, "tone_brand_fit": 3, "actionable": 1},
    "trivial__g015": {"grounded": 2, "correct_helpful": 1, "tone_brand_fit": 3, "actionable": 1},
    "trivial__g026": {"grounded": 2, "correct_helpful": 1, "tone_brand_fit": 3, "actionable": 1},
    "trivial__g172": {"grounded": 2, "correct_helpful": 1, "tone_brand_fit": 3, "actionable": 1},
    "trivial__g115": {"grounded": 1, "correct_helpful": 1, "tone_brand_fit": 2, "actionable": 1},
    "trivial__g029": {"grounded": 2, "correct_helpful": 1, "tone_brand_fit": 3, "actionable": 1},
    "trivial__g028": {"grounded": 2, "correct_helpful": 1, "tone_brand_fit": 3, "actionable": 1},
    "trivial__g170": {"grounded": 2, "correct_helpful": 1, "tone_brand_fit": 3, "actionable": 1},
    "trivial__g019": {"grounded": 2, "correct_helpful": 1, "tone_brand_fit": 3, "actionable": 1},
    "trivial__g185": {"grounded": 2, "correct_helpful": 1, "tone_brand_fit": 3, "actionable": 1},
    "trivial__g068": {"grounded": 2, "correct_helpful": 1, "tone_brand_fit": 3, "actionable": 1},
    "trivial__g093": {"grounded": 2, "correct_helpful": 1, "tone_brand_fit": 3, "actionable": 1},
    "trivial__g051": {"grounded": 2, "correct_helpful": 1, "tone_brand_fit": 3, "actionable": 1},

    "full__g023": {"grounded": 2, "correct_helpful": 3, "tone_brand_fit": 4, "actionable": 4},
    "full__g005": {"grounded": 4, "correct_helpful": 5, "tone_brand_fit": 5, "actionable": 5},
    "full__g001": {"grounded": 5, "correct_helpful": 5, "tone_brand_fit": 5, "actionable": 5},
    "full__g024": {"grounded": 5, "correct_helpful": 5, "tone_brand_fit": 5, "actionable": 5},
    "full__g030": {"grounded": 1, "correct_helpful": 1, "tone_brand_fit": 2, "actionable": 2},
    "full__g010": {"grounded": 5, "correct_helpful": 5, "tone_brand_fit": 5, "actionable": 5},
    "full__g034": {"grounded": 4, "correct_helpful": 4, "tone_brand_fit": 4, "actionable": 3},
    "full__g008": {"grounded": 3, "correct_helpful": 4, "tone_brand_fit": 4, "actionable": 4},
    "full__g021": {"grounded": 3, "correct_helpful": 3, "tone_brand_fit": 4, "actionable": 2},
    "full__g017": {"grounded": 1, "correct_helpful": 1, "tone_brand_fit": 2, "actionable": 1},
    "full__g015": {"grounded": 5, "correct_helpful": 5, "tone_brand_fit": 5, "actionable": 5},
    "full__g022": {"grounded": 4, "correct_helpful": 4, "tone_brand_fit": 4, "actionable": 4},
    "full__g012": {"grounded": 4, "correct_helpful": 4, "tone_brand_fit": 5, "actionable": 3},
    "full__g007": {"grounded": 4, "correct_helpful": 4, "tone_brand_fit": 4, "actionable": 4},
    "full__g009": {"grounded": 5, "correct_helpful": 5, "tone_brand_fit": 5, "actionable": 5},

    "simple__g092": {"grounded": 1, "correct_helpful": 1, "tone_brand_fit": 2, "actionable": 2},
    "simple__g104": {"grounded": 2, "correct_helpful": 2, "tone_brand_fit": 3, "actionable": 3},
    "simple__g119": {"grounded": 5, "correct_helpful": 5, "tone_brand_fit": 5, "actionable": 5},
    "simple__g095": {"grounded": 3, "correct_helpful": 4, "tone_brand_fit": 4, "actionable": 3},
    "simple__g108": {"grounded": 5, "correct_helpful": 5, "tone_brand_fit": 5, "actionable": 5},
    "simple__g046": {"grounded": 1, "correct_helpful": 1, "tone_brand_fit": 2, "actionable": 1},
    "simple__g023": {"grounded": 2, "correct_helpful": 2, "tone_brand_fit": 3, "actionable": 3},
    "simple__g146": {"grounded": 3, "correct_helpful": 3, "tone_brand_fit": 3, "actionable": 3},
    "simple__g195": {"grounded": 4, "correct_helpful": 4, "tone_brand_fit": 4, "actionable": 5},
    "simple__g155": {"grounded": 3, "correct_helpful": 3, "tone_brand_fit": 4, "actionable": 2},
    "simple__g198": {"grounded": 5, "correct_helpful": 5, "tone_brand_fit": 5, "actionable": 5},
    "simple__g152": {"grounded": 2, "correct_helpful": 2, "tone_brand_fit": 3, "actionable": 3},
    "simple__g161": {"grounded": 5, "correct_helpful": 5, "tone_brand_fit": 4, "actionable": 5},
    "simple__g030": {"grounded": 1, "correct_helpful": 1, "tone_brand_fit": 2, "actionable": 2},
    "simple__g082": {"grounded": 2, "correct_helpful": 2, "tone_brand_fit": 3, "actionable": 1},
}
