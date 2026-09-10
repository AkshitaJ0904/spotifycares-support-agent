# SpotifyCares support agent (take-home)

An AI support agent for **SpotifyCares** (Spotify's Twitter support account),
built from the Customer Support on Twitter dataset. Given an incoming customer
message, it:

1. **Classifies intent** into one of 8 hand-derived intents (`agent/taxonomy.json`).
2. **Retrieves grounding**: the most similar historically-resolved SpotifyCares
   threads (`agent/retrieve.py`), and drafts a reply conditioned on how those
   were actually handled (`agent/draft_reply.py`) -- not on the model's own
   parametric guess.
3. **Decides auto-handle vs. escalate**, with a stated, rule-based reason
   (`agent/escalate.py`).

Full write-up, results, failure analysis, and the "what's misleading about my
headline number" section: **[report.md](report.md)**. Decision log:
[DECISIONS.md](DECISIONS.md). Golden-set sampling/labeling method:
[eval/LABELING_PROTOCOL.md](eval/LABELING_PROTOCOL.md).

## Repo layout

```
scripts/          data download, cleaning, taxonomy discovery, retrieval index build
agent/             the system under test: taxonomy, classify, retrieve, escalate, draft_reply, pipeline, llm client
baselines/         trivial (majority-class + canned reply) and simple (keyword classifier + verbatim reply reuse)
eval/              golden set construction + labels, automated scoring, LLM judge, judge/human agreement
eval/results/      checked-in outputs of the full run (all 3 systems) -- report.md's numbers come from here
report.md          the write-up
DECISIONS.md       decision log
```

## Setup (~2 min)

```bash
uv venv .venv --python 3.13 && source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env   # then put your Gemini API key in .env (GEMINI_API_KEY=...)
```

Get a free key at https://aistudio.google.com/apikey. No Kaggle account
needed -- data comes from a public Hugging Face mirror (see DECISIONS.md #2).

Free-tier quota is small and shared per Google account/project (see
DECISIONS.md #8 -- we hit this directly building this repo). `agent/llm.py`
round-robins across multiple comma-separated keys if you set
`GEMINI_API_KEYS=key1,key2,...` instead of a single `GEMINI_API_KEY` --
useful if you have more than one key, but note keys from the *same* Google
project share one quota pool, so this only helps with keys from different
accounts/projects.

## Reproduce the headline results

```bash
python3 scripts/00_download_data.py          # ~195MB, ONE-TIME -- see timing note below
python3 scripts/01_prepare_data.py           # filters to SpotifyCares, cleans, splits corpus/golden-holdout -- seconds
python3 scripts/02_build_taxonomy.py         # TF-IDF+KMeans cluster discovery -> scripts/out/cluster_review.md -- ~1 min
python3 scripts/03_build_retrieval_index.py  # builds the grounding index -- seconds

python3 -m eval.build_golden_set             # samples 210 from the held-out pool -- seconds
python3 -m eval.finalize_golden_set          # merges in the hand-reviewed labels (eval/golden_labels.py) -- seconds

# Quick end-to-end smoke test on a subsample:
python3 -m eval.run_systems --n 15           # trivial+simple are instant; full system makes 2 live LLM calls/example
python3 -m eval.score
```

**Timing note, measured, not assumed:** steps 2-6 above take under a minute
total (verified from a clean clone). Step 1 (the download) is the one
genuinely variable piece: it's a one-time, unauthenticated pull from the
Hugging Face Hub, and on a clean environment with no local cache we measured
**~10 minutes** for the 195MB file -- much slower than a cached re-run (which
completes in seconds, and every run after the first is cached). If your
network is faster or you set an `HF_TOKEN` env var (Hugging Face's own fix for
unauthenticated-request throttling, free to get), this will be quicker. Budget
for the download to be the long pole, not the pipeline itself -- everything
after step 1 plus the `--n 15` smoke test is fast (~1-2 min more). If you're
tight on the 15-minute window, run step 1 first and let steps 2-8 follow once
it lands.

This regenerates the taxonomy discovery, retrieval index, and golden set from
scratch, and proves the full pipeline runs end to end on a live subsample.

**The numbers in report.md** are from a 111-example matched subsample of the
210-example golden set -- all three systems scored on the identical 111
examples (`trivial`/`simple` also have full-210 runs checked in as bonus
evidence; `full` was capped at 111 for free-tier time/quota reasons, see
DECISIONS.md #14). Already run once and checked into `eval/results/*.jsonl` /
`automated_scores.json` / `*_judged.jsonl` so you don't have to re-run
anything to see them. To regenerate from scratch (slow -- free-tier latency
and quota, not compute; expect this to take a while and possibly need a
retry the next day if you hit a quota wall, see DECISIONS.md #8):

```bash
python3 -m eval.run_systems --systems trivial simple        # all 210, instant
python3 -m eval.run_systems --systems full --n 111 --resume  # capped + resumable if interrupted/quota-limited
python3 -m eval.score --ids-from eval/results/full.jsonl      # matches all systems to the same 111 ids

python3 -m eval.llm_judge --ids-from eval/results/full.jsonl --resume   # resumable; re-run to retry quota failures
python3 -m eval.sample_for_human_review       # samples 45 replies for the judge/human agreement check
python3 -m eval.judge_agreement               # eval/human_scores.py holds the hand-assigned scores already
python3 -m eval.render_report_tables          # regenerates eval/results/report_table.md from the above
```

## Try a single message

Requires the setup steps above (needs `data/processed/retrieval_*` from
`scripts/03_build_retrieval_index.py` to exist):

```bash
python3 -m agent.pipeline "I can't log into my account, it says wrong password"
python3 -m baselines.simple "why was my favorite album removed"
```

## What this is / isn't

Built and evaluated on subsamples throughout (per the assignment's own
instructions -- this isn't run against the full ~3M-tweet dataset). See
report.md for scope, what was deliberately left out, and honest limitations.
