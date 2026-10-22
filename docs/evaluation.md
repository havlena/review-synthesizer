# Evaluation

Every detection signal gets a measured number before it is wired into the product, and
the numbers live in the repository next to the code that produced them.

## Golden tests (every signal, every commit)

`tests/` replays the capstone demo on the Harbor Light Inn fixture: the five planted
reviews are flagged, none of the twenty genuine ones are, and the rating moves from 3.8
displayed to 3.5 verified. These run in CI on every push. They prove the demo reproduces;
they say nothing about accuracy on real data, because 25 synthetic reviews cannot.

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

Run it:

```
pip install -e .[dev,ml]
python scripts/fetch_ott_corpus.py      # once; downloads to data/raw/ (git ignored)
python scripts/eval_ott.py
```

Outputs: `evals/results/ott_classifier.md` and `.json` (committed, so the figures are
versioned with the code), and `models/ott_deception.joblib` (not committed).

The report ends with the terms the model weights most in each direction and an
informational score of the Harbor Light Inn fixture. The fixture figures are a sanity check
on 25 synthetic reviews, not a result.

## What a good number does not mean

The classifier learns the style of reviews written by paid crowd workers in 2011 to 2013
about large Chicago hotels. That is one kind of fake. A review by someone who stayed and
was paid to be generous, a paraphrased copy, or a coordinated burst are different problems,
and the other signals (duplicate phrasing, rating bursts, embedding similarity) exist for
them. Expect lower figures on independent lodging in another region until the model is
retrained on labelled data from it, and treat every score as a signal for human review.

## Still to come

- Embedding similarity (step 4c): agreement with the Jaccard rule on the fixture, and a
  paraphrase test set built from the Ott corpus by rewriting a sample of reviews.
- End to end evals of the LangGraph graph (step 5 onward): a golden set of batches with
  expected flags, themes and actions, scored on every run and traced in LangSmith.
