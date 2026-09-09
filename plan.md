# Plan: AI Support Agent for a Twitter Customer-Support Brand

## 0. Read on the brief
Grading likely weights **proof over polish**: golden set quality, judge-human agreement, honest failure analysis, and the "misleading headline number" section are probably scrutinized harder than raw accuracy. Treat the report as the primary deliverable and the code as its supporting evidence.

## 1. Brand selection
Criteria: high volume, multi-turn threads where the brand *actually resolves* issues (not just "please DM us"), narrow-ish domain (fewer than 77 Banking77-style intents needed).
Candidates to check first: `AppleSupport`, `AmazonHelp`, `SpotifyCares`, `Uber_Support`, `Delta`, `ChipotleTweets`.
Action: pull tweet counts + avg thread depth per brand, pick top 1-2 by (volume × resolution signal), decide.

## 2. Data pipeline
- Reconstruct threads via `in_response_to_tweet_id` / `response_tweet_id` chains, keep only threads with ≥1 customer msg + ≥1 brand reply.
- Clean: strip handles/URLs, de-duplicate boilerplate ("We're sorry to hear..."), keep timestamps for turn order.
- Sample size: enough for a "historical resolution" retrieval corpus (thousands) + separate held-out golden set (150-250, no overlap).

## 3. Intent taxonomy
Bottom-up: embed customer first-turns → cluster (e.g. k-means or HDBSCAN over embeddings) → manually name clusters → collapse to 6-10 intents that are (a) distinct, (b) actionable, (c) map to different escalation policies. Document the derivation in report (not just "I decided on 8 intents").

## 4. System design
Pipeline per incoming message:
1. **Classify intent** (LLM few-shot, or fine-tuned/lightweight classifier — decide via baseline comparison).
2. **Retrieve grounding**: nearest historical resolved threads with same intent (embedding search over corpus from step 2) → few-shot context for reply drafting.
3. **Draft reply**: LLM generates reply conditioned on retrieved precedent, cites what pattern it followed.
4. **Escalation decision**: rule/LLM policy using intent + signals (anger/sentiment, account-security or payment keywords, low retrieval-similarity = no precedent, low classifier confidence) → auto-handle vs escalate + stated reason.

## 5. Baselines (mandatory: trivial + simple)
- **Trivial**: majority-class intent + single canned template reply + always-escalate (or never-escalate).
- **Simple**: keyword/regex intent classifier + verbatim-reuse of nearest historical reply (no generation) + fixed rule-based escalation.
- Compare full system against both on every metric in the harness.

## 6. Golden evaluation set (150-250 examples)
- Stratified sample across intents + brand, weighted toward edge cases (angry tone, ambiguous intent, multi-issue messages).
- Hand-label: intent, ideal escalate/auto decision, and a short "acceptable reply" note.
- Document sampling method + labeling protocol (who labeled, how disagreements resolved — likely just me, so say so honestly) directly in report.

## 7. Evaluation harness
- Automated: intent accuracy/F1 vs golden labels, escalation decision accuracy vs golden labels, retrieval hit-rate (did we pull a truly relevant precedent).
- LLM-as-judge: rubric (grounded-in-precedent, correctness, tone, actionability) scored 1-5, on a subset.
- **Judge-human agreement**: I hand-score ~30-50 of the same replies, compute agreement (e.g. weighted kappa / correlation) vs judge — required proof, not optional.

## 8. Report (≤6 pages)
Sections per brief: problem framing + scope cuts, results vs 2 baselines, top-5 failure modes w/ real examples + hypotheses, "what's misleading about my headline number" (candidates to seed: golden set is my own labels not brand's actual outcome; majority-class skew inflates intent accuracy; retrieval corpus and golden set overlap risk; judge may reward verbosity/politeness over correctness; small subsample ≠ full 3M-tweet distribution), next-week plan, decision log (10-15 bullets, e.g. why this brand, why this taxonomy size, why this escalation policy, model/embedding choices, sample sizes).

## 9. Repo structure (draft)
```
/data        - download + preprocessing scripts, sampled subsets only
/taxonomy    - clustering notebook + intent definitions
/agent       - classify.py, retrieve.py, draft_reply.py, escalate.py, pipeline.py
/eval        - golden_set.csv, harness.py, llm_judge.py, agreement.py
/baselines   - trivial.py, simple.py
README.md    - reproduce headline results in <15 min on subsample
report.md    - the 6-page report
```

## 10. Open decisions to make next
- Which brand (need quick data check).
- Classifier approach: pure LLM prompting vs small trained model (affects "simple baseline" contrast and cost/latency story).
- Embedding model + vector store (local/simple vs hosted).
- Which LLM for drafting + which (same or different) for judging (avoid self-preference bias — consider different model families).

## Next step
Run a quick data profiling pass on the Kaggle dataset to pick the brand, then start with §2-3 (pipeline + taxonomy) before touching modeling.
