# Decision log

Non-obvious calls made while building this, and why. Ordered roughly by when
they came up.

1. **Brand: SpotifyCares.** Checked volume/thread-depth across ~109 brands in
   the dataset first (`scripts` exploration, not committed as a script since it
   was one throwaway query). Picked Spotify over higher-volume options
   (AmazonHelp, AppleSupport) because its support domain is bounded enough for
   a small taxonomy to actually cover it, and it has a very clean, consistent
   "public troubleshoot, DM for anything account-specific" pattern that gives a
   natural, real signal for the escalation policy -- rather than inventing one.

2. **Dataset source: an HF parquet mirror, not the raw Kaggle CSV.** The direct
   Kaggle file needs thread reconstruction from `in_response_to_tweet_id`
   chains; `TNE-AI/customer-support-on-twitter-conversation` on Hugging Face is
   a public reconstruction of the same underlying Kaggle dataset already split
   into per-thread `Customer:`/`Support:` turns with a `company` field, which
   removes a whole error-prone step. Traded off some control over cleaning
   choices already baked into that mirror; noted as a limitation in report.md.
   (Tried a second mirror, `gorkemsevinc/...`, first -- it had stripped author/
   thread info down to just anonymized text, unusable for this task.)

3. **Taxonomy: hand-authored, informed by (not copied from) TF-IDF+KMeans
   clusters.** Clustering short tweets with TF-IDF is inherently noisy --
   `scripts/out/cluster_review.md` shows several clusters mixing unrelated
   topics and one catch-all holding ~38% of messages. Rather than force the
   raw clusters into the taxonomy, they were read and used as evidence to
   hand-write 8 intents in `agent/taxonomy.json`, each with a description used
   directly as the LLM classifier's decision rubric.

4. **8 intents, including an explicit `other_unclear` bucket.** A taxonomy that
   forces every message into a "real" category either becomes huge (Banking77-
   scale) or lies about confidence on the genuine long tail (praise, spam,
   fragments referencing unseen context -- ~7% of the golden set even after
   careful review). A named fallback bucket is more honest than a wrong
   confident guess, and it's a real routing decision (`other_unclear` always
   triggers escalation).

5. **Retrieval index: TF-IDF nearest-neighbor, not embeddings.** Free, fast to
   rebuild, deterministic, and on manual spot-checks (`agent/retrieve.py`)
   already surfaces near-duplicate historical issues with high cosine
   similarity for this vocabulary-heavy, short-text domain. Embeddings would
   add API cost/latency without a demonstrated quality gap here; flagged as a
   "what I'd try next" item rather than skipped for no reason.

6. **Escalation is a deterministic rule engine over signals, not a second LLM
   call.** The brief asks for "a stated reason" -- a rule engine's reason is
   reproducible and auditable in a way "the model said so" isn't. Signals:
   intent's PII flag, classifier confidence, top retrieval similarity, and a
   hand-rolled anger heuristic (profanity list + caps ratio + `!!!`). Thresholds
   were picked by reading ~40 examples during development, not tuned against
   the golden set (that would leak) -- almost certainly not optimal, and
   report.md's failure analysis names where they misfire.

7. **Per-intent PII flag is a static property of the taxonomy, and it's wrong
   sometimes.** Manually labeling the golden set surfaced real counterexamples
   in both directions: `content_availability` and `feature_request` messages
   that the real brand still routed to DM (`g054`, `g127`, `g207`), and an
   `account_access` (hacked account) case resolved with a public self-serve
   link, no DM at all (`g173`). Kept the static-flag design anyway for
   explainability, but this is the clearest concrete place the policy should
   get smarter next -- see report.md.

8. **Judge ended up on the same model as the agent -- not by design, by quota.**
   The original plan was two Gemini tiers (flash-lite for the agent,
   `gemini-3.5-flash`/`gemini-3.7-flash` for judging) for at least some
   separation between generator and grader. In practice, every `gemini-3.x-flash`
   judge model hit a hard "exceeded your current quota" 429 within the free
   tier partway through the judge run -- confirmed by curling the models
   directly (not just our own retry logic misbehaving). `gemini-flash-lite-latest`
   never once hit this in hours of use generating replies, so it clearly has a
   much larger free daily allowance than the bigger flash models. Also
   discovered along the way: the four API keys used for round-robining are
   evidently under one Google AI Studio project and share one quota pool --
   more keys from the same account bought no extra throughput, and concurrent
   requests across them just produced synchronized bursts of 429s. Net result:
   `JUDGE_MODEL` is now also `gemini-flash-lite-latest`. This is a real,
   disclosed limitation, not a hidden one -- see report.md's "what's misleading"
   section: the judge is, literally, grading its own family's output style, so
   the judge/human agreement number (§4) matters more here than it otherwise
   would, since same-model bias can't be ruled out by model choice alone.

9. **Golden set: 210 examples, stratified + edge-case-oversampled, single
   annotator, full manual read (not just an audited subsample).** See
   `eval/LABELING_PROTOCOL.md` for the full method. Chose to hand-label all 210
   rather than heuristic-label-all + audit-a-subset, because the "audit 20%"
   version would have meant reporting confidence in labels I hadn't actually
   checked -- with 210 short tweets this was feasible to do properly in one
   sitting instead.

10. **`true_action` in the golden set is a policy judgment, not a replay of
    history.** Using "did the real agent ask for DM" as the *default* is
    useful signal (it's evidence of how the brand itself handles similar
    cases) but isn't ground truth for "what should an automated agent do" --
    real agents are inconsistent too. Every row's action was still individually
    reviewed against a stated rule (can this be resolved well without further
    account verification?), and disagreements with the historical outcome are
    called out (`g069`, `g112`, `g173`, `g174`, `g177`, `g203`).

11. **Trivial baseline's majority-class intent is read off the golden set's
    true label distribution** (`billing_subscription`, 22%), the standard
    definition of a majority-class baseline -- not off a weak heuristic
    classifier's predictions on the corpus, which would have been a different
    (and less meaningful) number.

12. **Simple baseline's intent classifier is a *separate* keyword list from the
    one used to build the golden set's `heuristic_intent` prior**
    (`agent/classify.classify_rule_based` vs. `eval/label_heuristics`). Reusing
    the exact same keyword function for both would silently inflate the simple
    baseline's measured accuracy on the intents the golden set's own labels
    were seeded from.

13. **Reply cleaning artifacts leaking into generation, caught and fixed before
    eval.** Early manual testing (`agent/pipeline.py` smoke test) showed the
    LLM imitating our own redaction placeholders -- literally outputting the
    string `[link]` and inventing fake agent sign-off codes like `/PK` because
    those patterns are all over the few-shot precedent text. Fixed by telling
    the model explicitly, in-prompt, that those are redaction artifacts, not
    brand style. Worth a paragraph in report.md's failure analysis as a
    concrete "grounding can go wrong in an unexpected direction" example.

14. **`full` system evaluated on a 111-example matched subsample, not the
    entire 210 golden set -- a plan that changed under real constraints, not
    the original design.** The plan was the full 210 for every system; free-tier
    rate limits made the `full` system's ~10s/example (two LLM calls each)
    genuinely slow at that scale, so it was capped at 111 with `--n 111
    --resume` (`eval/run_systems.py`'s resume support -- keyed on `golden_id`,
    safe to interrupt and restart -- was added specifically because of this).
    `trivial` and `simple` still ran on the full 210 since they're free/instant;
    the 111 used for the report's headline table is the intersection, so every
    system is compared on identical examples (`eval/score.py --ids-from`).
    See DECISIONS.md #8 for the parallel story on why the LLM-judge stage hit
    the same wall even harder (and why `JUDGE_MODEL` changed as a result).

15. **No inline code comments explaining *what* code does** -- module
    docstrings carry the *why* (design rationale, tradeoffs), consistent with
    keeping the repo readable for a live walkthrough without redundant
    narration next to self-explanatory code.

16. **Caught (via actually re-running the README's own commands, not just
    reading them) that `run_systems.py`/`llm_judge.py` always appended to
    their output files regardless of `--resume`.** Without `--resume`, the
    code correctly computed a fresh subsample but still opened the output
    file in append mode -- meaning a second, no-`--resume` invocation (exactly
    what the README's own quick-repro section tells a reader to run) would
    have silently duplicated rows into the carefully-built full result files
    instead of overwriting them, as intended. Fixed: file mode is now `"a"`
    only under `--resume`, `"w"` otherwise. Caught by literally executing the
    README's reproduction steps end-to-end against the real results directory
    rather than trusting that the code matched the docs -- the corrupted
    files were then repaired by deduping to first-seen `golden_id` and
    re-verified against `eval/results/automated_scores.json` to confirm zero
    numeric drift from the fix.
