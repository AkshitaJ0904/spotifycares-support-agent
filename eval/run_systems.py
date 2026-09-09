"""
Runs trivial / simple / full over the golden set (or a subsample) and writes
raw per-example outputs to eval/results/<system>.jsonl. Kept separate from
scoring (eval/score.py) so a slow/expensive LLM run only ever has to happen
once and can be re-scored freely.
"""
import argparse
import json
import time
from pathlib import Path

from agent.pipeline import run as run_full
from baselines.simple import run as run_simple
from baselines.trivial import run as run_trivial

GOLDEN = "eval/golden_set.jsonl"
OUT_DIR = Path("eval/results")

SYSTEMS = {"trivial": run_trivial, "simple": run_simple, "full": run_full}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="subsample size (default: all)")
    ap.add_argument("--systems", nargs="+", default=list(SYSTEMS), choices=list(SYSTEMS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="skip golden_ids already in the output file")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(GOLDEN)]
    if args.n:
        import random

        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.n]
    print(f"running {len(rows)} golden examples through: {args.systems}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for sys_name in args.systems:
        fn = SYSTEMS[sys_name]
        out_path = OUT_DIR / f"{sys_name}.jsonl"

        done_ids = set()
        if args.resume and out_path.exists():
            kept_lines = []
            for line in open(out_path):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # drop a possibly-truncated last line from an interrupted run
                kept_lines.append(line if line.endswith("\n") else line + "\n")
                done_ids.add(r["golden_id"])
            with open(out_path, "w") as f:
                f.writelines(kept_lines)
            print(f"[{sys_name}] resuming: {len(done_ids)} already done")

        remaining = [r for r in rows if r["golden_id"] not in done_ids]
        t0 = time.time()
        with open(out_path, "a") as f:
            for i, row in enumerate(remaining):
                kwargs = {"exclude_conversation_id": row["conversation_id"]} if sys_name != "trivial" else {}
                try:
                    result = fn(row["message"], **kwargs)
                    result["ok"] = True
                except Exception as e:
                    result = {"message": row["message"], "ok": False, "error": str(e)}
                result["golden_id"] = row["golden_id"]
                result["true_intent"] = row["true_intent"]
                result["true_action"] = row["true_action"]
                result["reference_reply"] = row["reference_reply"]
                f.write(json.dumps(result) + "\n")
                f.flush()
                if (i + 1) % 20 == 0:
                    print(f"  [{sys_name}] {i+1}/{len(remaining)}")
        dt = time.time() - t0
        n_done = len(remaining) or 1
        print(f"[{sys_name}] done: {len(remaining)} new examples in {dt:.1f}s ({dt/n_done:.2f}s/ex) -> {out_path}")


if __name__ == "__main__":
    main()
