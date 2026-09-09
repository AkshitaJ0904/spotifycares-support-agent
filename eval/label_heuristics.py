"""
Weak-supervision helpers used ONLY to build a first-pass draft of the golden set
(never used at eval/inference time -- that would leak). Two independent signals:

1. `heuristic_intent`: a richer regex/keyword pass over the taxonomy than the
   'simple' baseline classifier (baselines/simple.py) uses -- deliberately kept as
   a *separate* function so the golden set's intent prior isn't literally the
   simple baseline's own predictions (which would make the simple baseline look
   artificially good in eval).
2. `outcome_action_prior`: derived from the REAL historical resolution (does any
   support turn in the actual thread ask to move to DM?), not from any model in
   this repo. This is real-world evidence of how Spotify itself handled it, and is
   the strongest single signal we have for what "should" happen -- though it is a
   description of what Spotify *did*, not a guarantee of what's *optimal*, which is
   why every row still gets manually reviewed (see LABELING_PROTOCOL.md).
"""
import json
import re
from pathlib import Path

TAXONOMY = json.load(open(Path(__file__).resolve().parent.parent / "agent" / "taxonomy.json"))

_PATTERNS = {
    "account_access": [r"\blog(ging)? ?in\b", r"\bpassword\b", r"\bcan'?t (log|sign) ?in\b",
                        r"\bhack(ed)?\b", r"\busername\b", r"\bcompromis", r"\bsign ?in\b", r"\breset\b.*\baccount\b"],
    "billing_subscription": [r"\bcharg", r"\bpremium\b", r"\bbill", r"\brefund", r"\bfamily plan\b",
                              r"\bstudent (discount|price)\b", r"\bpayment\b", r"\bsubscri", r"\binvite\b.*\bfamily\b"],
    "playback_technical_bug": [r"\bcrash", r"\bfreez", r"\bwon'?t play\b", r"\bbuffer", r"\bbug\b",
                                r"\bnot working\b", r"\bsync\b", r"\bdownload(ing|s)? (fail|stuck|stopp)",
                                r"\bkeeps (skipping|stopping|pausing)\b", r"\bglitch"],
    "content_availability": [r"\bnot (on|available)\b", r"\bremoved\b", r"\bmissing\b", r"\bwhere is\b",
                              r"\bcan'?t find\b.*\b(song|album|artist|podcast)\b", r"\bavailable in\b.*\bcountry\b",
                              r"\blicens"],
    "feature_request": [r"\bplease add\b", r"\bfeature\b", r"\bwish you\b", r"\bwould be (nice|great|amazing)\b",
                         r"\bcan you (add|make it)\b", r"\byou should\b"],
    "general_feedback_complaint": [r"\bhate\b", r"\bworst\b", r"\bterrible\b", r"\bdisappoint",
                                    r"\bwhy did you (change|remove|ruin)\b", r"\bannoying\b", r"\bsucks\b"],
    "general_inquiry": [r"\bhow do i\b", r"\bdoes spotify\b", r"\bjob\b", r"\bcareer\b", r"\bthanks?\b",
                         r"\bthank you\b", r"\?$"],
}

_DM_PATTERNS = [r"\bdm\b", r"\bdirect message\b", r"\bprivate message\b", r"\bbackstage\b",
                r"\bunder the hood\b", r"\bpm us\b", r"\breach out.*privat"]


def heuristic_intent(message: str) -> str:
    text = message.lower()
    scores = {intent: 0 for intent in _PATTERNS}
    for intent, pats in _PATTERNS.items():
        for p in pats:
            if re.search(p, text):
                scores[intent] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "other_unclear"


def outcome_action_prior(turns: list) -> str:
    """turns: list of [role, text]. Looks at every Support turn in the *actual*
    historical thread."""
    support_text = " ".join(t for r, t in turns if r == "Support").lower()
    for p in _DM_PATTERNS:
        if re.search(p, support_text):
            return "escalate"
    return "auto"
