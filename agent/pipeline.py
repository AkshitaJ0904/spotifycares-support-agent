"""End-to-end pipeline: classify -> retrieve grounding -> decide auto/escalate -> draft reply."""
import json
from pathlib import Path

from agent.classify import classify_llm, TAXONOMY
from agent.retrieve import retrieve
from agent.escalate import decide
from agent.draft_reply import draft_reply

_INTENT_NAME = {i["id"]: i["name"] for i in TAXONOMY["intents"]}


def run(message: str, k_precedents: int = 3, exclude_conversation_id: str = None) -> dict:
    cls = classify_llm(message)
    precedents = retrieve(message, k=k_precedents, exclude_conversation_id=exclude_conversation_id)
    top_sim = precedents[0]["similarity"] if precedents else 0.0

    decision = decide(
        message=message,
        intent=cls["intent"],
        classifier_confidence=cls["confidence"],
        top_retrieval_similarity=top_sim,
    )

    reply = draft_reply(
        message=message,
        intent_name=_INTENT_NAME.get(cls["intent"], cls["intent"]),
        action=decision["action"],
        precedents=precedents,
    )

    return {
        "message": message,
        "intent": cls["intent"],
        "intent_confidence": cls["confidence"],
        "intent_rationale": cls["rationale"],
        "action": decision["action"],
        "escalation_reason": decision["reason"],
        "escalation_signals": decision["signals"],
        "reply": reply,
        "grounding": [
            {"conversation_id": p["conversation_id"], "similarity": p["similarity"]} for p in precedents
        ],
    }


if __name__ == "__main__":
    import sys

    msg = " ".join(sys.argv[1:]) or "I can't log into my account, it says wrong password even though I know it's right"
    print(json.dumps(run(msg), indent=2))
