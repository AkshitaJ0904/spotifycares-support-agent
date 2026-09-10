# Report: an AI support agent for SpotifyCares

All numbers below are from the checked-in run in `eval/results/` (see README
for how to reproduce). Both automated metrics and LLM-judge scores run on the
same 111-example matched subsample of the 210-example golden set, identical
across all three systems (see §6 for why 111 and not the full 210 -- a real
free-tier rate-limit constraint, not a stylistic choice; the judge pass in
particular had to be finished in two sessions across a quota reset).

## 1. Problem framing

**Brand: SpotifyCares.** Chosen after profiling volume and thread structure
across the ~109 brands in the dataset (DECISIONS.md #1): high enough volume
(27.9k threads), a bounded product-support domain (unlike Amazon/Apple, which
span nearly everything), and a very consistent real-world pattern -- public
troubleshooting for generic issues, DM for anything account-specific -- that
gives a genuine signal to design the escalation policy around, instead of
inventing one from nothing.

**What "good" means here.** Three things, matching the brief's three asks:
1. **Intent classification** that's useful for routing/analytics, not just
   accurate on paper -- including an honest `other_unclear` bucket for the
   real long tail (praise, spam, fragments referencing unseen context) rather
   than a taxonomy that pretends every tweet is a clean support ticket.
2. **Grounded replies** that follow the brand's *actual* resolution pattern
   for similar past issues -- not a plausible-sounding generic answer. Concretely:
   if similar past issues were resolved by routing to DM, the drafted reply
   should do the same, not invent an on-the-spot fix.
3. **A defensible, auditable escalate/auto decision** -- explainable to a
   support-ops lead deciding whether to trust this in production, not just
   "the model said so."

**What I chose not to build:**
- **No multi-turn dialogue management.** The agent classifies/drafts/decides
  on the *opening* customer message only, matching how the golden set and
  most of the corpus are structured (first customer turn -> first support
  reply). A real deployment would need full thread state; out of scope here.
- **No fine-tuned classifier.** Given the free-tier/time budget, prompted
  LLM classification + a hand-authored taxonomy beat the effort of collecting
  enough clean labels to fine-tune a small model, especially with an escape
  hatch (`other_unclear`) for what it gets wrong. Discussed as a "next week"
  item.
- **No actual DM/private-channel integration.** "Escalate" means "flag for a
  human / route off the public-reply path with a stated reason" -- it does not
  attempt to simulate what happens after that.
- **No dense/embedding retrieval.** TF-IDF nearest-neighbor was good enough on
  inspection for this domain; see DECISIONS.md #5.

## 2. System design

`agent/pipeline.py`: classify (LLM, taxonomy-constrained) -> retrieve top-k
similar historically-resolved threads (TF-IDF cosine) -> decide auto/escalate
(deterministic rule engine over intent/confidence/retrieval-similarity/anger
signals, `agent/escalate.py`) -> draft reply (LLM, conditioned on the retrieved
precedents' actual text, `agent/draft_reply.py`).

## 3. Baselines

- **Trivial**: predicts the golden set's majority intent (`billing_subscription`,
  22%) for everything, a fixed canned reply for everything, never escalates.
- **Simple**: keyword-rule intent classifier (`agent/classify.classify_rule_based`,
  independent keyword list from the golden set's own labeling heuristic --
  DECISIONS.md #12) + verbatim reuse of the single nearest historical reply (no
  generation at all) + escalation from a fixed per-intent PII list only (no
  confidence/similarity/sentiment signals).
- **Full system**: as above (§2).

## 4. Results vs. baselines

*(111-example matched subsample of the golden set, all three systems scored on
the identical examples; see `eval/results/automated_scores.json` and
`eval/results/*_judged.jsonl` for raw numbers)*

| Metric | Trivial | Simple | Full |
|---|---|---|---|
| Intent accuracy | 15% | 47% | **75%** |
| Intent macro-F1 | 0.03 | 0.46 | **0.71** |
| Escalation-decision accuracy | 60% | 75% | 75% |
| Escalation precision (escalate class) | 0.00 | **0.77** | 0.73 |
| Escalation recall (escalate class) | 0.00 | 0.52 | **0.57** |
| Reply token-overlap-F1 vs. real historical reply | 0.14 | 0.29 | **0.30** |
| Reply length (words, mean) | 17.0 | 20.9 | 23.0 |
| LLM-judge: grounded (1-5) | 2.05 | 2.84 | **3.70** |
| LLM-judge: correct/helpful (1-5) | 1.92 | 2.85 | **3.71** |
| LLM-judge: tone/brand fit (1-5) | 3.00 | 3.31 | **4.09** |
| LLM-judge: actionable (1-5) | 1.50 | 3.29 | **3.79** |

**LLM-judge / human agreement** (45 replies, 15 per system, scored by the
author against the same rubric the judge uses, blind to the judge's own
rationale until after scoring -- `eval/judge_agreement.py`):
**Pearson r = +0.975, exact agreement = 88%, within-1 agreement = 100%**
across all four dimensions pooled (n=180 dimension-scores). Per-dimension,
`correct_helpful` had the lowest exact-match rate (69%) -- the judge trended
~0.3 points more generous than the human on that dimension specifically,
worth keeping in mind when reading the "correct/helpful" column above as
possibly a shade optimistic. See §6 for the bigger caveat on this number
(judge and agent share a model).

**Interpretation.** The system ordering is exactly what should happen if the
added machinery is doing real work: intent accuracy roughly doubles from
simple to full (47%→75%), and every judge dimension for `full` beats `simple`
by 0.5-0.9 points on a 5-point scale. The one metric that *doesn't* show a
large full-vs-simple gap is escalation-decision accuracy (75% vs. 75% --
identical), which looks like the fancier escalation policy bought nothing.
It didn't buy accuracy, but it did buy behavior: `full` has higher escalate-
recall (0.57 vs 0.52) at a small precision cost (0.73 vs 0.77), i.e. it
catches more of the genuinely-should-escalate cases and is slightly more
willing to over-escalate to do it -- a defensible tradeoff for a support
system (a missed escalation is worse than an unnecessary one), but invisible
in the single "accuracy" number. See §6 and §5 for why that number alone
would be misleading on its own.

## 5. Failure analysis: top 5 failure modes

1. **Static per-intent PII flag doesn't always match reality.** The escalation
   policy assumes `content_availability` and `feature_request` never need
   account verification, and `account_access` always does. Real counterexamples
   found while labeling the golden set: `g054`, `g127`, `g207` (those intents
   *did* need DM in practice) and `g173` (a hacked-account case resolved with a
   public self-serve link, no DM). Hypothesis: intent alone under-determines
   escalation need -- whether the *specific instance* references something
   account-specific (a screenshot, a persistent/recurring problem, an unusual
   account state) matters more than the intent category.

2. **The anger heuristic over-escalates on emphatic-but-simple messages, and
   this shows up live, not just in the golden labels.** Golden-set evidence:
   multiple `content_availability` complaints in all-caps or with profanity
   (`g112`, `g118`, `g174`, `g177`, `g203`) were resolved by the real brand with
   the *same* calm factual licensing explanation used for polite versions of
   the same question. Live confirmation from the actual full-system run:
   **`g035`** ("So when will be on? Bcs its about damn time!!! ... 🤓") has
   `true_intent=content_availability, true_action=auto` (a simple public
   licensing explanation, per the real historical reply), but the system
   classified it as `general_inquiry` and escalated with the stated reason
   *"message shows strong negative sentiment (anger score 2)"* -- caught by
   our own anger heuristic on the `!!!` and caps, not by anything about the
   actual issue. Sentiment should gate on whether de-escalation requires
   *judgment*, not just capitalization/punctuation.

3. **Grounding artifacts can leak into generation.** Caught during development
   (DECISIONS.md #13): the LLM initially imitated our own text-cleaning
   placeholders (`[link]`, invented fake sign-off codes) because they appear
   throughout the few-shot precedent examples. Fixed by naming the artifact
   explicitly in-prompt before any eval run used the output. General risk with
   retrieval-augmented generation over lightly-cleaned real data: the model
   imitates whatever's in the examples, including your own preprocessing
   artifacts, unless told not to.

4. **A "close enough" retrieval match can still be topically wrong, and
   correct intent + correct escalation decision don't save you from it.**
   `g030` ("...delete the 'dildos & more dildos' ad from the Classic Christmas
   playlist...") was classified correctly (`general_feedback_complaint`) and
   correctly auto-handled (classifier confidence 0.95, "no strong negative
   sentiment detected" -- accurate, the message is more incredulous than
   angry) -- but the drafted reply was *"Can you tell us more about what's
   happening? What device, operating system, and Spotify version are you
   on?"*, a device-troubleshooting question for a complaint about inappropriate
   ad content. The retrieval step found a precedent at similarity 0.42 (not a
   low score by this system's own numbers) that turned out to be topically
   unrelated. TF-IDF cosine similarity on short tweets rewards shared
   incidental vocabulary, not shared topic -- 0.42 reads as "plausible match"
   but wasn't. This is the single clearest case for why grounding quality
   needs its own check, independent of intent/escalation accuracy (see §7).

5. **Adjacent-intent confusion is the dominant classifier error mode.** 27/111
   (24%) of full-system intent predictions were wrong, and they cluster on
   boundaries between conceptually close categories rather than random noise:
   `g004` ("Did you get rid of your personalized Release Radar feature?") is
   `playback_technical_bug` (a feature silently breaking) but was classified
   `content_availability`; `g002`/`g027`/`g107`-style "when will X update
   happen"/"can you add setting Y" messages move between `general_inquiry` and
   `feature_request` depending on phrasing the taxonomy doesn't cleanly
   separate. Hypothesis: `content_availability` vs. `playback_technical_bug`
   and `general_inquiry` vs. `feature_request` are the two boundary pairs
   worth either merging or giving the classifier prompt explicit
   disambiguation examples for, rather than treating all 8 intents as equally
   separable.

## 6. What's misleading about my headline number

- **All 111 judge scores for `full` did complete, but only after hitting a
  real quota wall and resuming the next day** -- worth flagging even though
  it's resolved, because it's exactly the kind of thing that's easy to quietly
  patch over. Free-tier quota on all 4 rotated keys was exhausted mid-run
  (confirmed by direct curl returning "exceeded your current quota", not a
  transient rate limit; see DECISIONS.md #8), so the first pass judged only
  38/111 for `full` before erroring out cleanly (thanks to the resumable
  design -- failed rows are never silently counted as done) and the rest
  finished after quota reset. Had this report been written between those two
  sessions, the honest number would have been n=38, not n=111 -- a reminder
  that "the eval finished" is itself a claim worth being precise about, not
  an assumed background fact.
- **The judge and the agent are now the *same* model** (`gemini-flash-lite-latest`
  for both -- DECISIONS.md #8), not by design but because every larger Gemini
  judge model hit the same hard quota wall. The strong judge/human agreement
  (§4, r=+0.975) is reassuring evidence that the judge's *scores* track human
  judgment on this task, but it doesn't rule out the judge and the generator
  sharing systematic blind spots neither a human rater nor a same-family judge
  would catch (e.g. both trained toward similar "helpful-sounding" phrasing
  regardless of whether the content is actually right).
- **The golden set's `true_action` label is a policy judgment informed by, but
  not identical to, what the real brand actually did** (see
  `eval/LABELING_PROTOCOL.md`). A system that matches the label distribution
  well is matching *my* stated policy ("escalate only when public resolution
  genuinely isn't realistic"), not some external ground truth about what
  Spotify's real ops team would do today -- those can and do disagree
  (§5 item 1).
- **Escalation-decision accuracy (75% for both simple and full) hides a real
  behavioral difference** -- see §4's interpretation paragraph. Reporting only
  accuracy here, without precision/recall on the escalate class, would make
  the fancier policy look like it bought nothing.
- **Single annotator, single pass** on the golden set. Every label was
  hand-reviewed (stronger than an audited-subsample approach), but there's no
  independent second labeler and therefore no true inter-annotator-agreement
  number for the intent/action labels themselves -- only for *reply quality*
  (§4's judge/human agreement), which is a narrower claim about a different
  thing.
- **Token-overlap-vs-historical-reply is a weak proxy, included anyway for
  transparency.** A drafted reply can be *better* than the historical one (or
  equally valid but phrased differently) and still score low lexical overlap;
  conversely a reply that copies boilerplate phrasing scores well while adding
  little. It moved the least across systems (0.14→0.29→0.30) of any metric in
  the table -- that's the proxy being insensitive, not the systems being
  similar; the judge scores (which moved a lot, e.g. actionable 1.50→3.79) are
  the more trustworthy read on quality.
- **210 examples from one brand's Twitter threads, cleaned by someone else's
  redaction pipeline** (the HF mirror -- DECISIONS.md #2). Generalization to
  other brands, other channels (email/chat), or the raw uncleaned tweets isn't
  demonstrated.
- **The escalation-policy thresholds were hand-picked from ~40 examples during
  development, not tuned on data held apart from the golden set** (there is no
  third split to tune on without either shrinking the golden set or risking
  leakage) -- so the escalation numbers reflect one reasonable-looking but
  unoptimized threshold choice, not a ceiling on what the rule-engine approach
  could do.
- **Free-tier LLM infrastructure shaped the eval design more than intended.**
  Beyond the judge-sample-size issue above: results are cached from one run
  each, not averaged over repeated sampling, so `temperature=0.1-0.5`
  run-to-run variance isn't captured. Practically, building a real system on
  free-tier quota is itself a finding worth stating plainly: the eval
  methodology had to route around infrastructure limits live, and that's
  disclosed here rather than smoothed over.

## 7. What I'd do next with one more week

1. **Fix the concrete policy gaps found in §5.1/5.2** -- e.g. condition
   escalation on message-level signals (mentions "again"/"still"/a specific
   error code -> more likely needs account state) in addition to intent, and
   soften the anger-score gate for intents with a stable canned resolution.
2. **Add a grounding-relevance check independent of similarity score** (§5.4)
   -- e.g. a cheap LLM call that asks "is this precedent actually about the
   same issue?" before using it, since 0.42 TF-IDF similarity turned out not
   to guarantee topical relevance on `g030`.
3. **Second annotator on a subset of the golden set** to get a real
   inter-annotator-agreement number for the intent/action labels, not just the
   judge/human reply-quality check.
4. **Cross-provider judge** (e.g. a non-Gemini model) once available, to bound
   the shared-blind-spot risk named in §6.
5. **Multi-turn context**: extend beyond the opening message so the agent can
   use later turns (customer confirms device/OS, etc.) rather than only ever
   seeing the first message in isolation.
6. **Embedding retrieval A/B against the current TF-IDF index**, now that
   there's a golden set and judge harness to actually measure whether it's
   worth the added cost/latency, instead of assuming it from a manual
   spot-check.
7. **Non-English handling.** Several golden-set messages are non-English
   (`g016`, `g111`, `g139`, `g159`) and the current pipeline has no explicit
   language detection/routing -- worth deciding deliberately rather than
   leaving it to whatever the LLM happens to do.

## Decision log

See [DECISIONS.md](DECISIONS.md).

## Citations & attribution

- **Dataset**: Kaggle *Customer Support on Twitter*
  (`thoughtvector/customer-support-on-twitter`), accessed via a public Hugging
  Face reconstruction, `TNE-AI/customer-support-on-twitter-conversation`,
  which pre-splits the same underlying tweets into per-thread `Customer:`/
  `Support:` turns with a `company` field (see DECISIONS.md #2 for why this
  source was used instead of the raw Kaggle CSV, and its tradeoffs).
- **LLM**: Google Gemini API (`gemini-flash-lite-latest`), used for intent
  classification, reply drafting, and (per DECISIONS.md #8) also LLM-judge
  scoring, free tier.
- **Libraries**: scikit-learn (TF-IDF vectorization, KMeans clustering,
  classification-report/precision-recall metrics), pandas/pyarrow (data
  handling), `huggingface_hub` (dataset download), `requests` (Gemini REST
  calls) -- all standard open-source tooling, used as-is via their public
  APIs, no modified/vendored code.
- **Methodology**: retrieval-augmented generation (grounding replies in
  retrieved precedent rather than free-generation) and LLM-as-judge evaluation
  (using a model to score outputs against a rubric, validated against human
  agreement) are both established techniques in the field, not novel to this
  project -- applied here, not invented here. No specific paper's rubric or
  prompts were copied; the taxonomy, escalation rules, and judge rubric in
  this repo were authored from scratch against this dataset.
- **AI coding assistance**: used throughout development per the assignment's
  own rules ("you may use AI coding assistants freely... we will ask you to
  explain and modify your own code live") -- all design decisions, taxonomy
  choices, labeling judgments, and failure-mode analysis in this report and
  DECISIONS.md reflect the author's own review and reasoning over the actual
  data and run outputs, not unreviewed generated content.
