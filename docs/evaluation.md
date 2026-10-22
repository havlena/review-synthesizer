# Evaluation

Every detection signal gets a measured number before it is wired into the product, and
the numbers live in the repository next to the code that produced them.

## Golden tests (every signal, every commit)

`tests/` replays the capstone demo on the Harbor Light Inn fixture: the five planted
reviews are flagged, none of the twenty genuine ones are, and the rating moves from 3.8
displayed to 3.5 verified. These run on every push. They prove the demo reproduces; they
say nothing about accuracy on real data, because 25 synthetic reviews cannot.

## Deceptive language classifier (step 4b)

Data: the Ott Deceptive Opinion Spam Corpus v1.4, 1,600 labelled hotel reviews. Terms and
limits are in `data.md`.

Protocol: the five folds published with the corpus. Each fold holds the reviews of four
hotels, so no hotel appears in both training and test, which stops the model from learning
hotel names instead of language. The harness checks this and prints the result rather than
assuming it. For each fold the model is trained on the other four and scored on the held
out one; figures are reported per fold and as mean and sample standard deviation.

Two operating points:

- p ≥ 0.5, the usual decision, for accuracy, precision, recall, F1 and ROC AUC, with
  accuracy split by review polarity because the positive and negative halves of the
  corpus were collected differently;
- p ≥ 0.8, the conservative decision used in detection. Precision at this point is the
  figure that matters to an operator: a flag should be right. Recall says how much slips
  through, and some will.

Current result (`evals/results/ott_classifier.md`): 88.2% accuracy and ROC AUC 0.953 at
p ≥ 0.5; 99.1% precision at 42% recall at p ≥ 0.8.

```
python scripts/fetch_ott_corpus.py      # once; downloads to data/raw/ (git ignored)
python scripts/eval_ott.py
```

## Duplicate signals: word overlap versus embedding similarity (step 4c)

Data: `tests/fixtures/paraphrase_set.json`, 24 synthetic reviews written for this project:
three clusters of four rewritten duplicates (a positive campaign, a negative campaign, and
fabricated amenities), each written to keep the claims and change the words, plus twelve
distinct reviews that share topics with the clusters without sharing claims.

Protocol: pair level. Every pair inside a labelled cluster is a positive (18 pairs); every
other pair is a negative (258 pairs). For word overlap at the capstone threshold and for
embedding cosine at a sweep of thresholds from 0.30 to 0.90, the report gives pairs joined,
precision and recall, where the true pairs and the non duplicate pairs sit (min, mean, p95,
max, and per cluster), whether any threshold could separate them, and what each signal adds
on the same true pairs. The same script checks that embeddings recover the Harbor Light Inn
planted cluster and how close the nearest genuine review comes to it. Several models can be
run in one go; the summary compares them.

```
pip install -e .[embeddings]            # sentence-transformers, downloads the models once
python scripts/eval_similarity.py       # default and small model
python scripts/eval_similarity.py --model sentence-transformers/all-mpnet-base-v2 --model ...
```

Result (`evals/results/paraphrase_similarity.md`, three models, October 2026):

- Word overlap at 0.55 joins none of the 18 rewritten pairs. Expected: the set was written
  to defeat it.
- No model separates heavy paraphrase from honest reviews on the same topics. On all three
  the weakest true pair (0.32, 0.55, 0.56) scores below the strongest non duplicate pair
  (0.60, 0.65, 0.70). The best F1 points in the sweep (0.58 to 0.75, all at 0.40 to 0.60)
  would make a third or more of the flags wrong, which is unusable for an operator.
- The operating threshold is 0.85. It was 0.80 until the end to end run on Harbor Light
  (step 5) joined two genuine reviews, 10 and 16, at 0.801. That pair was already in the
  first version of this report as "most similar pair of genuine reviews 0.801", and nobody
  compared it with the threshold. At 0.80 the default model's only recall (3 of 18 pairs, in
  the fabricated amenities cluster) came from pairs at 0.80 to 0.81, the band honest reviews
  of one property reach, so those joins were not separable from honest ones.
- At 0.85 the signal joins none of the 18 heavy paraphrase pairs on any model. It still
  joins the Harbor Light planted cluster on every model (pairwise minimum 0.977 on the
  default `paraphrase-mpnet-base-v2`, 0.949 and 0.928 on the others). The default model
  keeps the widest margin: most similar genuine pair 0.801, nearest genuine review to a
  planted one 0.60. On `all-mpnet-base-v2` the most similar genuine pair is 0.835, a margin of
  only 0.015 under the threshold.
- Negative control, run on every model and failing the script if it fails: the most similar
  genuine Harbor Light pair and the most similar paraphrase set non duplicate pair must both
  sit below the threshold. At 0.85: PASS on all three (0.801 and 0.703 on the default model;
  0.698 and 0.598; 0.835 and 0.646).
- Light rewrites, the capstone case, are caught cleanly by every model (planted pairs 0.93
  and above). Rewrites built on concrete specifics are the most detectable; generic praise or
  complaint rewrites are the least.

Conclusion: the embedding signal detects light rewrites, nothing more. Cosine similarity
measures wording and topic, not the claims being made, and heavy paraphrase sits in the same
band as honest reviews of one property. Comparing claims across reviews is the job of the
language model step. The 0.85 threshold was set with the margin observed on synthetic data;
it must be re-checked on operator data, where honest reviews of one property will be more
alike than these.

## Claim duplicates: extraction plus a pairwise model judgment (step 5)

Same set, same pair level ground truth (18 true pairs, 258 negatives). The signal has two
stages, scored separately because the first caps the second: a candidate rule on the
extracted claims (pairs sharing at least N specific claims by aspect and polarity, no model
call), then one small model judgment per candidate. Measured on its own, with the detector's
skip rule off (word overlap joins none of the 18 pairs anyway).

```
python scripts/eval_claims.py                                          # current default
python scripts/eval_claims.py --min-shared 2 1 --judge strict lenient  # the ablation
```

Settings: claim extraction on `claude-sonnet-5-5`, judgments on `claude-haiku-5-5`, effort and
thinking at the API defaults, no temperature (the Messages API has rejected sampling
parameters since 2026-10-21, so stability is measured by repeating runs). Every run is
scored only if every call in it succeeded.

Baseline (`evals/results/claim_duplicates.md`, 2 shared claims, strict judge, two runs): 6
and 3 of 18 pairs joined, no wrong join in either run. Twelve or thirteen of the 18 pairs
were lost before any judgment, at the candidate rule. The positive campaign (cluster A)
yields no specific claims at all, so no candidate rule can reach it; the rating burst signal
is the one for that pattern. The runs agreed on 3 joined pairs: claim extraction itself
varies between runs.

Ablation (`evals/results/claim_duplicates_ablation.md`): each run extracts once per set and
all four cells (1 or 2 shared claims, strict or lenient judge) are scored on that same
extraction.

The first version used the paraphrase set alone. Its negatives were in sample and on
different topics, which cannot show the failure that matters most: honest reviews of one
property that mention the same features. No cell made a wrong join there, and the rule then
(no wrong join, highest recall) chose 1 shared claim with the lenient judge. The Harbor Light
end to end run at that default joined genuine reviews 10 and 16 on one shared claim, the
waterfront location.

The current version adds a control: the 20 genuine Harbor Light reviews, 190 honest pairs of
one property, every join wrong. The rule, fixed before the run: a cell qualifies only with no
wrong join on either set in either run; then highest mean paraphrase recall, ties to the
stricter setting; if none qualifies, nothing changes. Result: no cell made a wrong join on
either set in either run, so all four qualified. Mean paraphrase recall 0.333 (2, strict),
0.333 (2, lenient), 0.444 (1, strict), 0.472 (1, lenient). The rule kept 1 shared claim with
the lenient judge.

That result needs one caveat stated plainly. Pair 10 and 16 was a candidate in both control
runs and was judged not the same by both prompts both times; in the end to end run the
lenient prompt judged it the same. One "same" in five judgments of one pair is a judge that
is not stable on a borderline case, and two runs of 190 pairs do not catch a failure that
rare. Zero wrong joins on 190 pairs puts the per pair false join rate under about 1.6% at 95%
confidence (rule of three, pairs treated as independent, which they are not); at that rate a
batch of 500 reviews could still hold about 2,000 wrong joins. It bounds nothing about other
properties, longer reviews or other languages. The control prompts also named the property
with the evaluation placeholder rather than "Harbor Light Inn", a difference not measured.

Cost, from the realistic share: on honest reviews of one property 18% of pairs become
candidates at 1 shared claim (1.3% at 2), so a batch of 500 reviews needs about 22,650 small
model judgments, about $3, against about 1,640 at 2 shared claims. Claim duplicate flags
are now scored and banded by the evidence behind the join: the mean number of specific claims
shared by the pairs judged the same (1 is low, 2 medium, 3 or more high), so a join on one
shared claim reaches the operator as low.

Read the gain with care. Most of it comes from the candidate rule, and the lenient prompt's
examples of different names for one feature are the vocabulary of cluster C, which favours it
on recall. The setting was chosen on the sets it is reported on, so expect less on new data.
Any held out set must include honest reviews of one property that mention the same features.
The signal finds about four to five in ten rewritten pairs here; it is a partial signal whose
joins still go to a person.

Known behaviour, visible in the report: the rating burst rule is timing only, so a genuine
review posted within four days of a planted cluster is flagged with it. That is the Module
6 false positive risk, and it is why every flag goes to a person with its reason attached
rather than acting on its own. The v1 rule also merged every burst of one rating into a
single entry; v2 reports separate bursts separately, with the golden result unchanged.

## What a good number does not mean

The classifier learns the style of reviews written by paid crowd workers in 2011 to 2013
about large Chicago hotels. That is one kind of fake. A review by someone who stayed and
was paid to be generous, a paraphrased copy, or a coordinated burst are different problems,
and the other signals exist for them. Expect lower figures on independent lodging in
another region until the model is retrained on labelled data from it, and treat every
score as a signal for human review.

The paraphrase set is small and was written by the same person who wrote the detector. It
shows the mechanism and its limit and justifies the default threshold and model; it does not
measure field accuracy. A larger labelled set from real operator data is the next step once
there is an operator.

## Still to come

- A held out paraphrase set, written separately from the detector, to check the claim
  duplicate setting chosen above on data it was not chosen on.
- End to end evals of the graph: a golden set of batches with expected flags, themes and
  actions, scored on every run and traced in LangSmith.
