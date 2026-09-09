"""Merge the sampled draft with the manually-reviewed labels into the final,
frozen golden set used by the eval harness."""
import json

from eval.golden_labels import LABELS

DRAFT = "eval/golden_set_draft.jsonl"
OUT = "eval/golden_set.jsonl"


def main():
    records = [json.loads(l) for l in open(DRAFT)]
    missing = [r["golden_id"] for r in records if r["golden_id"] not in LABELS]
    if missing:
        raise SystemExit(f"missing manual labels for: {missing}")

    n_intent_overridden = 0
    n_action_overridden = 0
    n_notes = 0
    final = []
    for r in records:
        intent_override, action_override, note = LABELS[r["golden_id"]]
        true_intent = intent_override or r["heuristic_intent"]
        true_action = action_override or r["outcome_action_prior"]
        if intent_override:
            n_intent_overridden += 1
        if action_override:
            n_action_overridden += 1
        if note:
            n_notes += 1
        final.append(
            {
                "golden_id": r["golden_id"],
                "conversation_id": r["conversation_id"],
                "message": r["first_customer_msg"],
                "true_intent": true_intent,
                "true_action": true_action,
                "reference_reply": r["first_support_reply"],
                "is_edge_case": r["is_edge_case"],
                "annotator_note": note,
            }
        )

    with open(OUT, "w") as f:
        for r in final:
            f.write(json.dumps(r) + "\n")

    print(f"wrote {len(final)} labeled examples to {OUT}")
    print(f"intent overridden from heuristic prior: {n_intent_overridden} ({n_intent_overridden/len(final):.0%})")
    print(f"action overridden from outcome prior: {n_action_overridden} ({n_action_overridden/len(final):.0%})")
    print(f"rows with an annotator note: {n_notes}")

    from collections import Counter
    print("final true_intent distribution:", Counter(r["true_intent"] for r in final).most_common())
    print("final true_action distribution:", Counter(r["true_action"] for r in final).most_common())


if __name__ == "__main__":
    main()
