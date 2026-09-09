"""
Filter the raw Twitter customer-support dataset down to one brand (SpotifyCares),
parse each conversation into ordered (role, text) turns, clean the text, and split
into a retrieval/taxonomy CORPUS and a held-out GOLDEN pool that the rest of the
pipeline never touches during development.

Source data: HF mirror `TNE-AI/customer-support-on-twitter-conversation`, itself a
reconstruction of the Kaggle "Customer Support on Twitter" dataset (thoughtvector/
customer-support-on-twitter) into per-thread `Customer:` / `Support:` turns.
See DECISIONS.md item on dataset source.
"""
import json
import random
import re
from pathlib import Path

import pandas as pd

BRAND = "SpotifyCares"
RAW_PATH = "data/raw2/data/train-00000-of-00001.parquet"
OUT_DIR = Path("data/processed")
SEED = 42
HOLDOUT_SIZE = 260  # golden set is hand-labeled from a subsample of this, rest kept for buffer

URL_RE = re.compile(r"https?://\S+")
SIGNOFF_RE = re.compile(r"\s*[/\^][A-Za-z]{2,3}\s*$")  # trailing agent sign-off, e.g. " /CH", " ^KA"
MENTION_RE = re.compile(r"@\w+")
WS_RE = re.compile(r"\s+")


def clean_text(t: str) -> str:
    t = SIGNOFF_RE.sub("", t)
    t = URL_RE.sub("[link]", t)
    t = MENTION_RE.sub("", t)
    t = WS_RE.sub(" ", t).strip()
    return t


def parse_conversation(raw: str):
    """Split the `Role: text` blob into ordered turns. A turn can itself contain
    newlines (the label only appears at the start of a new turn), so we split on
    lines that begin a new `Customer:`/`Support:` block."""
    turns = []
    cur_role, cur_lines = None, []
    for line in raw.split("\n"):
        m = re.match(r"^(Customer|Support):\s?(.*)$", line)
        if m:
            if cur_role is not None:
                turns.append((cur_role, " ".join(cur_lines).strip()))
            cur_role, cur_lines = m.group(1), [m.group(2)]
        else:
            cur_lines.append(line)
    if cur_role is not None:
        turns.append((cur_role, " ".join(cur_lines).strip()))
    return turns


def main():
    df = pd.read_parquet(RAW_PATH)
    df = df[df["company"] == BRAND].reset_index(drop=True)
    print(f"[1] raw {BRAND} conversations: {len(df)}")

    records = []
    for row in df.itertuples():
        turns = parse_conversation(row.conversation)
        turns = [(role, clean_text(text)) for role, text in turns if clean_text(text)]
        # keep threads that open with the customer and get at least one support reply
        if len(turns) < 2 or turns[0][0] != "Customer" or not any(r == "Support" for r, _ in turns):
            continue
        first_customer = turns[0][1]
        first_support = next((t for r, t in turns if r == "Support"), "")
        if len(first_customer) < 8 or len(first_support) < 3:
            continue
        records.append(
            {
                "conversation_id": row.conversation_id,
                "turns": turns,
                "first_customer_msg": first_customer,
                "first_support_reply": first_support,
                "n_turns": len(turns),
            }
        )
    print(f"[2] usable parsed conversations: {len(records)}")

    rng = random.Random(SEED)
    rng.shuffle(records)

    holdout = records[:HOLDOUT_SIZE]
    corpus = records[HOLDOUT_SIZE:]
    print(f"[3] split -> corpus: {len(corpus)}  golden-holdout pool: {len(holdout)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "corpus.jsonl", "w") as f:
        for r in corpus:
            f.write(json.dumps(r) + "\n")
    with open(OUT_DIR / "golden_holdout_pool.jsonl", "w") as f:
        for r in holdout:
            f.write(json.dumps(r) + "\n")

    print("[4] wrote data/processed/corpus.jsonl and golden_holdout_pool.jsonl")


if __name__ == "__main__":
    main()
