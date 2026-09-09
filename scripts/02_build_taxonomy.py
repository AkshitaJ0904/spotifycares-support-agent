"""
Bottom-up intent discovery: TF-IDF + KMeans over the customer's opening message,
across a sweep of k, to find a natural, human-nameable grouping. This script only
*proposes* clusters (top terms + example messages) to a review file; the actual
intent taxonomy (names/descriptions) is written by hand in agent/taxonomy.json
after reading scripts/out/cluster_review.md (see DECISIONS.md).
"""
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

CORPUS = "data/processed/corpus.jsonl"
OUT_DIR = Path("scripts/out")
SEED = 42


def main():
    records = [json.loads(l) for l in open(CORPUS)]
    texts = [r["first_customer_msg"] for r in records]
    print(f"clustering {len(texts)} opening messages")

    vec = TfidfVectorizer(max_features=4000, min_df=5, stop_words="english", ngram_range=(1, 2))
    X = vec.fit_transform(texts)
    terms = np.array(vec.get_feature_names_out())

    # silhouette on a subsample (full-matrix silhouette is O(n^2), too slow at 27k rows)
    rng = np.random.RandomState(SEED)
    sample_idx = rng.choice(X.shape[0], size=4000, replace=False)

    scores = {}
    for k in range(6, 13):
        km = KMeans(n_clusters=k, random_state=SEED, n_init=5)
        labels = km.fit_predict(X)
        s = silhouette_score(X[sample_idx], labels[sample_idx])
        scores[k] = s
        print(f"k={k:>2}  silhouette={s:.4f}")

    best_k = max(scores, key=scores.get)
    print(f"\nbest_k by silhouette = {best_k} (silhouette is a weak signal on sparse TF-IDF text; "
          f"used as a tie-breaker, not gospel — final k is chosen by reading the clusters)")

    # We fix k=9 regardless of the silhouette argmax: short tweets make TF-IDF silhouette
    # noisy/monotonic-ish, and 9 was the smallest k at which clusters stopped visibly
    # mixing unrelated topics on manual read (see DECISIONS.md).
    k = 9
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    labels = km.fit_predict(X)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "cluster_review.md", "w") as f:
        f.write(f"# Cluster review (k={k})\n\n")
        for c in range(k):
            idx = np.where(labels == c)[0]
            center = km.cluster_centers_[c]
            top_term_idx = center.argsort()[::-1][:15]
            f.write(f"## Cluster {c}  (n={len(idx)})\n\n")
            f.write("**top terms:** " + ", ".join(terms[top_term_idx]) + "\n\n")
            f.write("**examples:**\n")
            sample = rng.choice(idx, size=min(10, len(idx)), replace=False)
            for i in sample:
                f.write(f"- {texts[i]!r}\n")
            f.write("\n")

    # persist cluster assignment + vectorizer vocab for reuse / rule-baseline keyword export
    np.save(OUT_DIR / "cluster_labels.npy", labels)
    with open(OUT_DIR / "conversation_ids_order.json", "w") as f:
        json.dump([r["conversation_id"] for r in records], f)

    print(f"\nwrote {OUT_DIR/'cluster_review.md'} — read this, then hand-author agent/taxonomy.json")


if __name__ == "__main__":
    main()
