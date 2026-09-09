"""
Builds a TF-IDF nearest-neighbor index over the corpus's opening customer messages.
This is the "grounding" store: given a new message, we retrieve the most similar
historically-resolved threads and hand their actual support replies to the LLM as
few-shot precedent, rather than letting it free-generate.

TF-IDF (not embeddings) is a deliberate choice for the retrieval index -- see
DECISIONS.md: it's free, deterministic, fast to rebuild, and good enough for
short, vocabulary-heavy support tweets; embeddings are used only where they pay
for themselves (they weren't needed here since TF-IDF NN quality was already high
on manual spot-check).
"""
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

CORPUS = "data/processed/corpus.jsonl"
OUT_DIR = Path("data/processed")


def main():
    records = [json.loads(l) for l in open(CORPUS)]
    texts = [r["first_customer_msg"] for r in records]

    vec = TfidfVectorizer(max_features=20000, min_df=2, stop_words="english", ngram_range=(1, 2))
    X = vec.fit_transform(texts)
    print(f"index: {X.shape[0]} docs x {X.shape[1]} terms")

    joblib.dump(vec, OUT_DIR / "retrieval_vectorizer.joblib")
    joblib.dump(X, OUT_DIR / "retrieval_matrix.joblib")
    with open(OUT_DIR / "retrieval_records.json", "w") as f:
        json.dump(records, f)
    print("wrote retrieval_vectorizer.joblib, retrieval_matrix.joblib, retrieval_records.json")


if __name__ == "__main__":
    main()
