"""Trivial baseline: majority-class intent for everything, one canned reply for
everything, never escalate. This is the floor the real system must clear."""
import json
from pathlib import Path

CORPUS = Path(__file__).resolve().parent.parent / "data" / "processed" / "corpus.jsonl"

CANNED_REPLY = (
    "Hey there! Thanks for reaching out. We're looking into this and will get back to you soon."
)


def _majority_intent() -> str:
    """The modal true_intent in the golden set (eval/golden_set.jsonl: 47/210 =
    22% billing_subscription) -- the standard definition of a majority-class
    baseline's prediction. Hardcoded so this baseline needs no runtime
    dependency (a true 'trivial' baseline shouldn't need to load anything)."""
    return "billing_subscription"


MAJORITY_INTENT = _majority_intent()


def run(message: str) -> dict:
    return {
        "message": message,
        "intent": MAJORITY_INTENT,
        "action": "auto",
        "escalation_reason": "trivial baseline never escalates",
        "reply": CANNED_REPLY,
    }
