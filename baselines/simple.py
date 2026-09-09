"""Simple baseline: keyword-rule intent classifier + verbatim reuse of the nearest
historical reply (no generation) + escalation based only on the intent's
typically_needs_pii flag (ignores sentiment/confidence/retrieval-quality signals
that the full system uses). No LLM calls at all -- fully deterministic and free."""
from agent.classify import classify_rule_based, TAXONOMY
from agent.retrieve import retrieve

PII_INTENTS = {i["id"] for i in TAXONOMY["intents"] if i.get("typically_needs_pii")}


def run(message: str, exclude_conversation_id: str = None) -> dict:
    cls = classify_rule_based(message)
    precedents = retrieve(message, k=1, exclude_conversation_id=exclude_conversation_id)
    reply = precedents[0]["support_reply"] if precedents else (
        "Hey there! Thanks for reaching out. We're looking into this and will get back to you soon."
    )
    action = "escalate" if cls["intent"] in PII_INTENTS else "auto"
    reason = (
        f"intent '{cls['intent']}' is in the fixed PII-requiring list"
        if action == "escalate"
        else f"intent '{cls['intent']}' is not in the fixed PII-requiring list"
    )
    return {
        "message": message,
        "intent": cls["intent"],
        "action": action,
        "escalation_reason": reason,
        "reply": reply,
        "grounding": [{"conversation_id": precedents[0]["conversation_id"], "similarity": precedents[0]["similarity"]}] if precedents else [],
    }


if __name__ == "__main__":
    import json, sys

    msg = " ".join(sys.argv[1:]) or "I can't log into my account, it says wrong password"
    print(json.dumps(run(msg), indent=2))
