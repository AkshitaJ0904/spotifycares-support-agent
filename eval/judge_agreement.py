"""Computes LLM-judge vs. human agreement on the sample produced by
sample_for_human_review.py, scored by hand in eval/human_scores.py."""
import json
from pathlib import Path
from statistics import correlation

from eval.human_scores import HUMAN_SCORES

SAMPLE = Path("eval/results/human_review_sample.jsonl")
DIMS = ["grounded", "correct_helpful", "tone_brand_fit", "actionable"]


def main():
    rows = [json.loads(l) for l in open(SAMPLE)]
    missing = [r["review_id"] for r in rows if r["review_id"] not in HUMAN_SCORES]
    if missing:
        raise SystemExit(f"missing human scores for: {missing}")

    per_dim = {}
    all_judge, all_human = [], []
    for dim in DIMS:
        judge_vals, human_vals = [], []
        for r in rows:
            j = r["judge_scores"].get(dim)
            h = HUMAN_SCORES[r["review_id"]].get(dim)
            if j is None or h is None:
                continue
            judge_vals.append(j)
            human_vals.append(h)
        all_judge.extend(judge_vals)
        all_human.extend(human_vals)

        n = len(judge_vals)
        exact = sum(1 for j, h in zip(judge_vals, human_vals) if j == h) / n
        within1 = sum(1 for j, h in zip(judge_vals, human_vals) if abs(j - h) <= 1) / n
        try:
            corr = correlation(judge_vals, human_vals)
        except Exception:
            corr = float("nan")
        per_dim[dim] = {"n": n, "pearson_r": corr, "exact_agreement": exact, "within_1_agreement": within1,
                         "judge_mean": sum(judge_vals) / n, "human_mean": sum(human_vals) / n}
        print(f"{dim:18s} n={n:3d}  r={corr:+.3f}  exact={exact:.0%}  within1={within1:.0%}  "
              f"judge_mean={per_dim[dim]['judge_mean']:.2f}  human_mean={per_dim[dim]['human_mean']:.2f}")

    overall_corr = correlation(all_judge, all_human)
    overall_exact = sum(1 for j, h in zip(all_judge, all_human) if j == h) / len(all_judge)
    overall_within1 = sum(1 for j, h in zip(all_judge, all_human) if abs(j - h) <= 1) / len(all_judge)
    print(f"\nOVERALL (all dims pooled, n={len(all_judge)}): r={overall_corr:+.3f}  "
          f"exact={overall_exact:.0%}  within1={overall_within1:.0%}")

    out = {"per_dimension": per_dim, "overall": {"n": len(all_judge), "pearson_r": overall_corr,
           "exact_agreement": overall_exact, "within_1_agreement": overall_within1}}
    with open("eval/results/judge_agreement.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
