"""
LLM-as-judge rubric for reply quality. Uses JUDGE_MODEL (a different model tier
than the agent's AGENT_MODEL -- see agent/llm.py) to score each drafted reply
1-5 on four dimensions, given the customer message and the REAL historical
SpotifyCares reply as a reference point for what a good, brand-grounded
resolution looks like for that exact message.
"""
import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from agent.llm import generate_json, JUDGE_MODEL, API_KEYS

RESULTS_DIR = Path("eval/results")
SYSTEMS = ["trivial", "simple", "full"]
# Round-robining keys only helps throughput if requests actually run concurrently --
# a sequential loop that alternates keys is still bottlenecked by one call's
# latency at a time. Use a small thread pool (I/O-bound: the GIL is released
# while waiting on the network) sized to the number of available keys.
MAX_WORKERS = 2  # dialed down further after even flash-lite started 429ing
# under sustained concurrent load late in the day's testing -- small remaining
# batches go sequential-ish and rely on agent/llm.py's backoff.

_SCHEMA = {
    "type": "object",
    "properties": {
        "grounded": {"type": "integer"},
        "correct_helpful": {"type": "integer"},
        "tone_brand_fit": {"type": "integer"},
        "actionable": {"type": "integer"},
        "rationale": {"type": "string"},
    },
    "required": ["grounded", "correct_helpful", "tone_brand_fit", "actionable", "rationale"],
}

_SYSTEM = (
    "You are grading a candidate customer-support reply for SpotifyCares (Spotify's Twitter support). "
    "You are given the customer's message, the candidate reply, and a REFERENCE reply that Spotify's real "
    "support team actually sent for this exact message (the reference is a real, brand-approved resolution "
    "style, not necessarily the only correct answer). Score the CANDIDATE reply 1-5 on each dimension:\n"
    "- grounded: does it follow the same resolution pattern/brand voice as the reference (e.g. routes to DM "
    "for account-specific issues the way the reference does, doesn't invent facts the reference/message don't support)?\n"
    "- correct_helpful: is it factually sound and actually useful given the message?\n"
    "- tone_brand_fit: warm, brief, casual-professional Twitter support voice, matching the reference's register?\n"
    "- actionable: does the customer know what to do next?\n"
    "1 = fails badly, 3 = acceptable, 5 = excellent. Be discriminating -- most replies should NOT be 5s."
)


def judge_one(message: str, reply: str, reference_reply: str, intent: str, action: str) -> dict:
    prompt = (
        f"Customer message: {message!r}\n"
        f"Classified intent: {intent}\n"
        f"System decision: {action}\n"
        f"REFERENCE (real historical) reply: {reference_reply!r}\n"
        f"CANDIDATE reply to grade: {reply!r}\n\n"
        f"Score the CANDIDATE reply."
    )
    return generate_json(prompt, _SCHEMA, system=_SYSTEM, model=JUDGE_MODEL, temperature=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", default=SYSTEMS, choices=SYSTEMS)
    ap.add_argument("--n", type=int, default=None, help="judge only the first N rows per system (default: all)")
    ap.add_argument("--ids-from", type=str, default=None,
                     help="jsonl file whose golden_ids define a matched subsample to judge across all systems")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    id_filter = None
    if args.ids_from:
        id_filter = {json.loads(l)["golden_id"] for l in open(args.ids_from)}
        print(f"judging restricted to {len(id_filter)} matched golden_ids from {args.ids_from}")

    for name in args.systems:
        in_path = RESULTS_DIR / f"{name}.jsonl"
        out_path = RESULTS_DIR / f"{name}_judged.jsonl"
        rows = [json.loads(l) for l in open(in_path)]
        if id_filter is not None:
            rows = [r for r in rows if r["golden_id"] in id_filter]
        if args.n:
            rows = rows[: args.n]

        done_ids = set()
        if args.resume and out_path.exists():
            kept = []
            n_retryable = 0
            for line in open(out_path):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("judge", {}).get("grounded") is None:
                    n_retryable += 1  # a prior failed judge call -- retry it, don't count as done
                    continue
                kept.append(line if line.endswith("\n") else line + "\n")
                done_ids.add(r["golden_id"])
            with open(out_path, "w") as f:
                f.writelines(kept)
            rows = [r for r in rows if r["golden_id"] not in done_ids]
            print(f"[{name}] resuming: {len(done_ids)} already judged, {n_retryable} failed (retrying), "
                  f"{len(rows)} left")

        print(f"[{name}] judging {len(rows)} replies with {JUDGE_MODEL} ({MAX_WORKERS} concurrent workers "
              f"across {len(API_KEYS)} key(s))...")

        def score_row(r):
            if not r.get("ok", True) or not r.get("reply"):
                scores = {"grounded": 1, "correct_helpful": 1, "tone_brand_fit": 1, "actionable": 1,
                          "rationale": "system failed to produce a reply"}
            else:
                try:
                    scores = judge_one(
                        r["message"], r["reply"], r["reference_reply"], r.get("intent", "?"), r.get("action", "?")
                    )
                except Exception as e:
                    scores = {"grounded": None, "correct_helpful": None, "tone_brand_fit": None,
                              "actionable": None, "rationale": f"judge error: {e}"}
            r["judge"] = scores
            return r

        write_lock = threading.Lock()
        n_done = 0
        with open(out_path, "a") as f, ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = [pool.submit(score_row, r) for r in rows]
            for fut in as_completed(futures):
                r = fut.result()
                with write_lock:
                    f.write(json.dumps(r) + "\n")
                    f.flush()
                    n_done += 1
                    if n_done % 20 == 0:
                        print(f"  [{name}] {n_done}/{len(rows)}")
        print(f"[{name}] wrote {out_path}")


if __name__ == "__main__":
    main()
