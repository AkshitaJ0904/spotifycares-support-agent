"""
Deterministic, explainable auto-handle vs escalate policy.

Design principle: escalation should be governed by *rules over signals*, not a
second opaque LLM call -- the brief asks for a stated reason, and a rule engine's
reason is inherently auditable/reproducible in a way "the LLM said escalate" isn't.
See DECISIONS.md.

Thresholds were picked by hand after reading ~40 example messages during
development (not tuned against the golden set -- that would leak). They are
almost certainly not optimal; report.md's failure analysis flags this.
"""
import json
import re
from pathlib import Path

TAXONOMY = json.load(open(Path(__file__).resolve().parent / "taxonomy.json"))
PII_INTENTS = {i["id"] for i in TAXONOMY["intents"] if i.get("typically_needs_pii")}

RETRIEVAL_SIM_THRESHOLD = 0.15
CLASSIFIER_CONF_THRESHOLD = 0.55
ANGER_THRESHOLD = 2

PROFANITY = {
    "shit", "fuck", "fucking", "wtf", "damn", "hell", "crap", "ass", "bullshit",
    "pissed", "garbage", "trash", "scam", "worst", "hate", "sucks", "ridiculous",
}


def anger_score(message: str) -> int:
    text = message.lower()
    score = 0
    score += sum(1 for w in PROFANITY if re.search(rf"\b{re.escape(w)}\b", text))
    score += text.count("!!!")
    letters = [c for c in message if c.isalpha()]
    if letters:
        caps_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        if caps_ratio > 0.5 and len(letters) > 8:
            score += 2
    return score


def decide(message: str, intent: str, classifier_confidence: float, top_retrieval_similarity: float) -> dict:
    """Returns {"action": "auto" | "escalate", "reason": str, "signals": {...}}"""
    reasons = []
    anger = anger_score(message)

    if intent == "other_unclear":
        reasons.append("intent could not be confidently mapped to a known support topic")
    if intent in PII_INTENTS:
        reasons.append(
            f"intent '{intent}' typically requires account-specific verification (email/username) "
            f"that the brand's own historical playbook resolves via DM, not a public/automated reply"
        )
    if classifier_confidence < CLASSIFIER_CONF_THRESHOLD:
        reasons.append(f"intent-classification confidence {classifier_confidence:.2f} below {CLASSIFIER_CONF_THRESHOLD}")
    if top_retrieval_similarity < RETRIEVAL_SIM_THRESHOLD:
        reasons.append(
            f"no sufficiently similar historical precedent found (top similarity {top_retrieval_similarity:.2f} "
            f"< {RETRIEVAL_SIM_THRESHOLD}); an auto-reply here would be ungrounded"
        )
    if anger >= ANGER_THRESHOLD:
        reasons.append(f"message shows strong negative sentiment (anger score {anger}); human de-escalation preferred")

    signals = {
        "intent": intent,
        "classifier_confidence": classifier_confidence,
        "top_retrieval_similarity": top_retrieval_similarity,
        "anger_score": anger,
        "pii_required_by_intent": intent in PII_INTENTS,
    }

    if reasons:
        return {"action": "escalate", "reason": "; ".join(reasons), "signals": signals}
    return {
        "action": "auto",
        "reason": (
            f"intent '{intent}' does not require account verification, classifier confidence "
            f"{classifier_confidence:.2f} is sufficient, a close historical precedent was found "
            f"(similarity {top_retrieval_similarity:.2f}), and no strong negative sentiment detected"
        ),
        "signals": signals,
    }


if __name__ == "__main__":
    print(decide("I can't log in, my password isn't working!!! this is SO annoying", "account_access", 0.95, 0.7))
    print(decide("please add a dark mode toggle for the web player", "feature_request", 0.9, 0.6))
