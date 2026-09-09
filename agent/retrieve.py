"""Nearest-neighbor retrieval over historically-resolved SpotifyCares threads."""
import json
from functools import lru_cache
from pathlib import Path

import joblib
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


@lru_cache(maxsize=1)
def _load():
    vec = joblib.load(DATA_DIR / "retrieval_vectorizer.joblib")
    X = joblib.load(DATA_DIR / "retrieval_matrix.joblib")
    records = json.load(open(DATA_DIR / "retrieval_records.json"))
    return vec, X, records


def retrieve(query: str, k: int = 3, exclude_conversation_id: str = None):
    vec, X, records = _load()
    q = vec.transform([query])
    sims = cosine_similarity(q, X)[0]
    order = sims.argsort()[::-1]
    out = []
    for idx in order:
        rec = records[idx]
        if exclude_conversation_id and rec["conversation_id"] == exclude_conversation_id:
            continue
        out.append(
            {
                "conversation_id": rec["conversation_id"],
                "similarity": float(sims[idx]),
                "customer_msg": rec["first_customer_msg"],
                "support_reply": rec["first_support_reply"],
                "turns": rec["turns"],
            }
        )
        if len(out) >= k:
            break
    return out


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "I can't log into my account, it says wrong password"
    for r in retrieve(q, k=3):
        print(f"sim={r['similarity']:.3f}  {r['customer_msg'][:80]!r}")
        print(f"   -> {r['support_reply'][:100]!r}")
