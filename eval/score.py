"""
Automated metrics over eval/results/<system>.jsonl (produced by run_systems.py):
intent accuracy/macro-F1, escalation-decision accuracy/precision/recall, and
cheap reply-quality proxies (token-overlap vs. the real historical reply,
reply length). LLM-judge scoring is separate (eval/llm_judge.py) since it's
the expensive/slow part.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from sklearn.metrics import classification_report, precision_recall_fscore_support

RESULTS_DIR = Path("eval/results")
SYSTEMS = ["trivial", "simple", "full"]
ID_FILTER = None  # set by main() from --ids-from, for a fair matched-subsample comparison


def token_overlap_f1(a: str, b: str) -> float:
    """Cheap, dependency-free proxy for lexical overlap with the real historical
    reply -- NOT a claim that the historical reply is the one 'correct' answer,
    just a sanity signal that the draft is in the right neighborhood. See
    report.md for why this number alone would be misleading."""
    ta, tb = set(a.lower().split()), set(b.lower().split())
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    p, r = inter / len(ta), inter / len(tb)
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


def score_system(name: str) -> dict:
    path = RESULTS_DIR / f"{name}.jsonl"
    rows = [json.loads(l) for l in open(path)]
    if ID_FILTER is not None:
        rows = [r for r in rows if r["golden_id"] in ID_FILTER]
    ok_rows = [r for r in rows if r.get("ok", True)]
    n_failed = len(rows) - len(ok_rows)

    y_true_intent = [r["true_intent"] for r in ok_rows]
    y_pred_intent = [r.get("intent", "other_unclear") for r in ok_rows]
    intent_report = classification_report(y_true_intent, y_pred_intent, output_dict=True, zero_division=0)

    y_true_action = [r["true_action"] for r in ok_rows]
    y_pred_action = [r.get("action", "auto") for r in ok_rows]
    p, r_, f1, _ = precision_recall_fscore_support(
        y_true_action, y_pred_action, labels=["escalate"], zero_division=0
    )
    action_acc = sum(a == b for a, b in zip(y_true_action, y_pred_action)) / len(ok_rows)

    overlaps = [token_overlap_f1(r.get("reply", ""), r["reference_reply"]) for r in ok_rows]
    lengths = [len(r.get("reply", "").split()) for r in ok_rows]

    return {
        "system": name,
        "n": len(rows),
        "n_failed": n_failed,
        "intent_accuracy": intent_report["accuracy"],
        "intent_macro_f1": intent_report["macro avg"]["f1-score"],
        "intent_per_class_f1": {
            k: v["f1-score"] for k, v in intent_report.items() if k not in ("accuracy", "macro avg", "weighted avg")
        },
        "action_accuracy": action_acc,
        "action_escalate_precision": float(p[0]),
        "action_escalate_recall": float(r_[0]),
        "action_escalate_f1": float(f1[0]),
        "action_true_dist": dict(Counter(y_true_action)),
        "action_pred_dist": dict(Counter(y_pred_action)),
        "reply_token_overlap_f1_mean": sum(overlaps) / len(overlaps),
        "reply_length_words_mean": sum(lengths) / len(lengths),
    }


def main():
    global ID_FILTER
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-from", type=str, default=None,
                     help="jsonl file whose golden_ids define a matched subsample to score all systems on")
    args = ap.parse_args()
    if args.ids_from:
        ID_FILTER = {json.loads(l)["golden_id"] for l in open(args.ids_from)}
        print(f"scoring restricted to {len(ID_FILTER)} matched golden_ids from {args.ids_from}")

    all_scores = {}
    for name in SYSTEMS:
        path = RESULTS_DIR / f"{name}.jsonl"
        if not path.exists():
            print(f"skip {name}: {path} not found (run eval/run_systems.py first)")
            continue
        s = score_system(name)
        all_scores[name] = s
        print(f"\n=== {name} (n={s['n']}, failed={s['n_failed']}) ===")
        print(f"intent accuracy={s['intent_accuracy']:.3f}  macro-F1={s['intent_macro_f1']:.3f}")
        print(f"action accuracy={s['action_accuracy']:.3f}  escalate P={s['action_escalate_precision']:.3f} "
              f"R={s['action_escalate_recall']:.3f} F1={s['action_escalate_f1']:.3f}")
        print(f"reply token-overlap-F1 vs historical={s['reply_token_overlap_f1_mean']:.3f}  "
              f"mean reply length={s['reply_length_words_mean']:.1f} words")

    with open(RESULTS_DIR / "automated_scores.json", "w") as f:
        json.dump(all_scores, f, indent=2)
    print(f"\nwrote {RESULTS_DIR/'automated_scores.json'}")


if __name__ == "__main__":
    main()
