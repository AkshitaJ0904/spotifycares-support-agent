"""Intent classification: the main (LLM) classifier and a rule-based classifier
used for the 'simple' baseline (see baselines/simple.py)."""
import json
import re
from pathlib import Path

from agent.llm import generate_json

TAXONOMY_PATH = Path(__file__).resolve().parent / "taxonomy.json"
TAXONOMY = json.load(open(TAXONOMY_PATH))
INTENT_IDS = [i["id"] for i in TAXONOMY["intents"]]

_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": INTENT_IDS},
        "confidence": {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": ["intent", "confidence", "rationale"],
}

_SYSTEM = (
    "You are an intent classifier for SpotifyCares, Spotify's Twitter customer-support team. "
    "Classify the customer's message into exactly one of the given intents. "
    "confidence is your own calibrated probability (0-1) that this is the correct intent -- "
    "use lower values for ambiguous or multi-issue messages, not just 0.9 by default."
)


def _taxonomy_block() -> str:
    lines = []
    for i in TAXONOMY["intents"]:
        lines.append(f"- {i['id']}: {i['name']} -- {i['description']}")
    return "\n".join(lines)


def classify_llm(message: str) -> dict:
    prompt = f"Intents:\n{_taxonomy_block()}\n\nCustomer message:\n{message!r}\n\nClassify it."
    result = generate_json(prompt, _SCHEMA, system=_SYSTEM)
    if result["intent"] not in INTENT_IDS:
        result["intent"] = "other_unclear"
    result["confidence"] = max(0.0, min(1.0, float(result["confidence"])))
    return result


def classify_rule_based(message: str) -> dict:
    """Keyword-match classifier: first intent (in taxonomy order) whose keyword list
    has a hit wins; ties broken by most keyword hits. No LLM call -- used by the
    'simple' baseline, and as an explainable fallback."""
    text = message.lower()
    best, best_hits = "other_unclear", 0
    for intent in TAXONOMY["intents"]:
        hits = sum(1 for kw in intent["keywords"] if kw in text)
        if hits > best_hits:
            best, best_hits = intent["id"], hits
    confidence = min(0.5 + 0.15 * best_hits, 0.95) if best_hits else 0.2
    return {"intent": best, "confidence": confidence, "rationale": f"{best_hits} keyword hit(s)"}


if __name__ == "__main__":
    import sys

    msg = " ".join(sys.argv[1:]) or "I can't log into my account, it keeps saying wrong password"
    print("LLM:", classify_llm(msg))
    print("Rule:", classify_rule_based(msg))
