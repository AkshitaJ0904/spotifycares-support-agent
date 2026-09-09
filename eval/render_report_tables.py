"""Renders the report.md results table (and judge scores) from
eval/results/automated_scores.json + *_judged.jsonl, so report.md's numbers
are pulled from the actual run output rather than retyped by hand."""
import json
from pathlib import Path
from statistics import mean

RESULTS_DIR = Path("eval/results")
SYSTEMS = ["trivial", "simple", "full"]
JUDGE_DIMS = ["grounded", "correct_helpful", "tone_brand_fit", "actionable"]


def judge_means(system: str) -> dict:
    path = RESULTS_DIR / f"{system}_judged.jsonl"
    if not path.exists():
        return {d: None for d in JUDGE_DIMS}
    rows = [json.loads(l) for l in open(path)]
    out = {}
    for d in JUDGE_DIMS:
        vals = [r["judge"][d] for r in rows if r.get("judge", {}).get(d) is not None]
        out[d] = mean(vals) if vals else None
    return out


def main():
    scores = json.load(open(RESULTS_DIR / "automated_scores.json"))
    jm = {s: judge_means(s) for s in SYSTEMS}

    def fmt(x, pct=False):
        if x is None:
            return "n/a"
        return f"{x:.0%}" if pct else f"{x:.2f}"

    rows = [
        ("Intent accuracy", lambda s: fmt(scores[s]["intent_accuracy"], pct=True)),
        ("Intent macro-F1", lambda s: fmt(scores[s]["intent_macro_f1"])),
        ("Escalation-decision accuracy", lambda s: fmt(scores[s]["action_accuracy"], pct=True)),
        ("Escalation precision (escalate)", lambda s: fmt(scores[s]["action_escalate_precision"])),
        ("Escalation recall (escalate)", lambda s: fmt(scores[s]["action_escalate_recall"])),
        ("Reply token-overlap-F1 vs. historical", lambda s: fmt(scores[s]["reply_token_overlap_f1_mean"])),
        ("Reply length (words, mean)", lambda s: fmt(scores[s]["reply_length_words_mean"])),
        ("LLM-judge: grounded", lambda s: fmt(jm[s]["grounded"])),
        ("LLM-judge: correct/helpful", lambda s: fmt(jm[s]["correct_helpful"])),
        ("LLM-judge: tone/brand fit", lambda s: fmt(jm[s]["tone_brand_fit"])),
        ("LLM-judge: actionable", lambda s: fmt(jm[s]["actionable"])),
    ]

    lines = ["| Metric | Trivial | Simple | Full |", "|---|---|---|---|"]
    for label, fn in rows:
        lines.append(f"| {label} | {fn('trivial')} | {fn('simple')} | {fn('full')} |")
    table = "\n".join(lines)
    print(table)

    agreement_path = RESULTS_DIR / "judge_agreement.json"
    if agreement_path.exists():
        agr = json.load(open(agreement_path))["overall"]
        print(f"\nJudge/human agreement: n={agr['n']}  r={agr['pearson_r']:+.3f}  "
              f"exact={agr['exact_agreement']:.0%}  within1={agr['within_1_agreement']:.0%}")

    with open(RESULTS_DIR / "report_table.md", "w") as f:
        f.write(table + "\n")
    print(f"\nwrote {RESULTS_DIR/'report_table.md'}")


if __name__ == "__main__":
    main()
