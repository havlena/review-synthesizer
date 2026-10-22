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
- At the 0.80 operating point, `paraphrase-mpnet-base-v2` is the only model with any recall
  (3 of 18 pairs, all in the fabricated amenities cluster) at a precision of 1.0, and it keeps
  the widest margin on Harbor Light (planted pairs at 0.98 and above, nearest genuine review
  0.60). It is the default for that reason; `all-MiniLM-L6-v2` stays as the small option.
- Light rewrites, the capstone case, are caught cleanly by every model (planted pairs 0.93
  and above). Rewrites built on concrete specifics are the most detectable; generic praise or
  complaint rewrites are the least.

Conclusion: the embedding signal is a precise detector of light rewrites and shared
specifics, nothing more. Cosine similarity measures wording and topic, not the claims being
made. Comparing claims across reviews is the job of the language model step, which is why
the LangGraph graph gets a claim extraction node rather than a lower embedding threshold.

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

Ablation (`evals/results/claim_duplicates_ablation.md`): each run extracts once and all four
cells (1 or 2 shared claims, strict or lenient judge) are scored on that same extraction.
No cell made a wrong join in either run. Mean judged recall: 0.222 (2, strict), 0.222
(2, lenient), 0.389 (1, strict), 0.417 (1, lenient). The rule fixed before the run (no wrong
join in either run, then highest recall, ties to the stricter setting, otherwise keep the
default) chose 1 shared claim with the lenient judge, and that is now the product default.
What it costs: about seven times the judgments on this set (12% of all pairs become
candidates, against 2%), roughly 15,000 small model judgments, about $2, on a batch of 500
reviews, and more on real batches where honest reviews share more specifics.

Read the gain with care. Most of it comes from the candidate rule (0.222 to 0.389 with the
same judge). The judge prompt's share is half a pair per run on average: lenient led strict
by two pairs in one run and trailed by one in the other. The lenient prompt's examples of
different names for one feature are the vocabulary of cluster C, which favours it on this
set. The setting was chosen on the same 24 reviews it is reported on, with no held out set,
so expect less on new data. The signal finds about four in ten rewritten pairs here, with
precision 1.0 in every run so far; it is a precise, partial signal, like the embedding one,
and the pairs it finds still go to a person.

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
