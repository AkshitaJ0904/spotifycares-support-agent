"""
Build the golden set: stratified sample from the held-out pool (data/processed/
golden_holdout_pool.jsonl, 260 threads never used for corpus/taxonomy/retrieval),
with a heuristic first-pass label attached to every row for manual review.

Sampling strategy (see LABELING_PROTOCOL.md for full rationale):
- Stratify by heuristic_intent so every taxonomy bucket has a floor of examples
  (rare intents like feature_request/general_feedback are naturally under-represented
  in raw frequency; a purely random sample would under-power failure analysis on them).
- Oversample "edge cases" (angry tone, ambiguous/other_unclear, long/multi-issue
  messages) into a dedicated slice, since those are exactly where systems break --
  a representative-only sample would bury the interesting failures.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

from agent.escalate import anger_score
from eval.label_heuristics import heuristic_intent, outcome_action_prior

POOL = "data/processed/golden_holdout_pool.jsonl"
OUT = "eval/golden_set_draft.jsonl"
SEED = 7
TARGET_TOTAL = 210
MIN_PER_BUCKET = 10


def main():
    records = [json.loads(l) for l in open(POOL)]
    for r in records:
        r["heuristic_intent"] = heuristic_intent(r["first_customer_msg"])
        r["outcome_action_prior"] = outcome_action_prior(r["turns"])
        r["anger_score"] = anger_score(r["first_customer_msg"])
        r["is_edge_case"] = (
            r["anger_score"] >= 2
            or r["heuristic_intent"] == "other_unclear"
            or len(r["first_customer_msg"]) > 220
        )

    by_bucket = defaultdict(list)
    for r in records:
        by_bucket[r["heuristic_intent"]].append(r)

    rng = random.Random(SEED)
    for v in by_bucket.values():
        rng.shuffle(v)

    selected, seen_ids = [], set()

    # 1) floor per bucket
    for bucket, items in by_bucket.items():
        take = items[:MIN_PER_BUCKET]
        for r in take:
            if r["conversation_id"] not in seen_ids:
                selected.append(r)
                seen_ids.add(r["conversation_id"])

    # 2) explicit edge-case slice (up to 30, on top of floor)
    edge_pool = [r for r in records if r["is_edge_case"] and r["conversation_id"] not in seen_ids]
    rng.shuffle(edge_pool)
    for r in edge_pool[:30]:
        selected.append(r)
        seen_ids.add(r["conversation_id"])

    # 3) fill remainder randomly from whatever's left, up to TARGET_TOTAL
    remaining = [r for r in records if r["conversation_id"] not in seen_ids]
    rng.shuffle(remaining)
    for r in remaining:
        if len(selected) >= TARGET_TOTAL:
            break
        selected.append(r)
        seen_ids.add(r["conversation_id"])

    rng.shuffle(selected)
    for i, r in enumerate(selected):
        r["golden_id"] = f"g{i:03d}"

    with open(OUT, "w") as f:
        for r in selected:
            f.write(json.dumps(r) + "\n")

    bucket_counts = defaultdict(int)
    for r in selected:
        bucket_counts[r["heuristic_intent"]] += 1
    print(f"selected {len(selected)} / {len(records)} pool examples")
    print("bucket counts:", dict(bucket_counts))
    print("edge cases:", sum(1 for r in selected if r["is_edge_case"]))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
