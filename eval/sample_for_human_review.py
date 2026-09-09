"""Samples replies (spread across all three systems, so both clearly-bad and
clearly-good replies are represented) for a human to score with the same
rubric the LLM judge uses, to measure judge/human agreement."""
import json
import random
from pathlib import Path

RESULTS_DIR = Path("eval/results")
SYSTEMS = ["trivial", "simple", "full"]
PER_SYSTEM = 15
SEED = 11
OUT = RESULTS_DIR / "human_review_sample.jsonl"


def main():
    rng = random.Random(SEED)
    sample = []
    for sys_name in SYSTEMS:
        path = RESULTS_DIR / f"{sys_name}_judged.jsonl"
        rows = [json.loads(l) for l in open(path)]
        rows = [r for r in rows if r.get("ok", True) and r.get("reply") and r["judge"].get("grounded") is not None]
        rng.shuffle(rows)
        for r in rows[:PER_SYSTEM]:
            sample.append(
                {
                    "review_id": f"{sys_name}__{r['golden_id']}",
                    "system": sys_name,
                    "golden_id": r["golden_id"],
                    "message": r["message"],
                    "reply": r["reply"],
                    "reference_reply": r["reference_reply"],
                    "judge_scores": r["judge"],
                }
            )
    rng.shuffle(sample)  # so the review order doesn't reveal which system produced which reply
    with open(OUT, "w") as f:
        for r in sample:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(sample)} rows to {OUT} ({PER_SYSTEM} per system)")


if __name__ == "__main__":
    main()
