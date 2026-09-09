# Golden set: sampling & labeling protocol

**Pool.** `scripts/01_prepare_data.py` reserves 260 SpotifyCares threads as a
`golden_holdout_pool` that is never seen by the retrieval index, the taxonomy
clustering, or any baseline's fitting step (excluded before any of those run).
No leakage between corpus and golden set is possible by construction, not by
after-the-fact filtering.

**Sampling (`eval/build_golden_set.py`, 210 of 260 selected, seed=7).**
1. A cheap regex/keyword pass (`eval/label_heuristics.heuristic_intent`,
   *independent* of the baseline classifiers used at eval time) buckets every
   pooled thread into a taxonomy intent.
2. Every bucket gets a floor of 10 examples, so low-frequency intents
   (`feature_request`, `general_feedback_complaint`) aren't sampled down to
   nothing by pure random chance.
3. An explicit "edge case" slice (angry tone, heuristic-unclear, or long/
   multi-issue messages) is oversampled up to 30 on top of the floor -- a
   uniform random sample would under-represent exactly the messages that break
   systems, and those are what the failure analysis needs.
4. The remainder is filled randomly to 210.

**Labeling.** `eval/golden_set_review.txt` is the exact material read for every
row (message + full real historical reply thread) -- the labels in
`eval/golden_labels.py` were assigned directly from it, not from memory or a
smaller sample of it. Single annotator (the author), one pass, all 210 read in full --
customer message *and* the real historical SpotifyCares reply thread. Two
independent priors were computed first and then confirmed or corrected by hand
for every row (not just an audited sample):
- `heuristic_intent` -- keyword prior, corrected on ~48% of rows (mostly cases
  where the message doesn't contain the obvious keyword, e.g. "Bring Crazy
  Bitch back" is `content_availability` with no word like "available" in it).
- `outcome_action_prior` -- did the real thread's support turns mention DM/
  backstage/private message. Used as the default `true_action` (does resolving
  this well, in public, without further account verification, look realistic
  given how the brand itself actually handled *similar* cases), corrected on
  ~3% of rows where the signal was a false positive/negative (e.g. "backstage"
  used to mean "on our end" rather than "please DM us" -- see `g069`'s note).

This means `true_action` is a **policy judgment**, not a literal replay of what
the historical agent happened to do -- it asks "could this be resolved well
without collecting personal info", using the real thread as evidence, not as
an oracle. The two aren't always the same thing, and several rows are flagged
`policy_tension` in `eval/golden_labels.py` precisely where a message's *intent*
would suggest one answer but the *real* resolution needed the opposite
(`g054`, `g127`, `g207`: intents our own escalation policy assumes are always
safe to auto-handle, that in practice needed DM; `g173`: an `account_access`
case resolved via a public self-serve link, no DM at all).

**Known limitations (see report.md, "what's misleading about my headline
number").** Single annotator -- no independent second labeler, so there is no
true inter-annotator agreement number for the golden set itself (as distinct
from the LLM-judge/human agreement in `eval/judge_agreement.py`, which *is*
measured). Labels were produced by one person applying a documented rubric in
one sitting, which is internally consistent but can carry that person's
systematic blind spots. `other_unclear` still holds 15/210 (7%) genuinely
ambiguous or off-topic messages (praise, spam, fragments referencing unseen
prior context) -- a real system will see this long tail too, and it's part of
why the taxonomy needs a legitimate "doesn't fit" bucket rather than forcing
every message into a clean category.
