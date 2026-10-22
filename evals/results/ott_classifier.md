# Deceptive language classifier: evaluation on the Ott corpus

Generated 2026-10-22T01:04:24+00:00 with scikit-learn 1.9.1 on Python 3.12.10. Protocol: published folds, leave one fold out.

Corpus: 1600 reviews, 800 deceptive and 800 truthful, 20 hotels, 5 folds. Folds are hotel disjoint: yes.

Model: TF-IDF over unigrams and bigrams, logistic regression. Parameters: C=10.0, ngram_range=(1, 2), min_df=2, max_features=None, sublinear_tf=True, random_state=0.

## Per fold (decision at p = 0.5; deceptive is the positive class)

| Fold | Test hotels | n | Accuracy | Precision | Recall | F1 | ROC AUC | Acc. positive | Acc. negative |
|---|---|---|---|---|---|---|---|---|---|
| 1 | hilton, james, monaco, sofitel | 320 | 0.850 | 0.837 | 0.869 | 0.853 | 0.927 | 0.850 | 0.850 |
| 2 | affinia, ambassador, hardrock, talbott | 320 | 0.875 | 0.905 | 0.838 | 0.870 | 0.954 | 0.900 | 0.850 |
| 3 | conrad, fairmont, hyatt, omni | 320 | 0.891 | 0.903 | 0.875 | 0.889 | 0.957 | 0.906 | 0.875 |
| 4 | homewood, knickerbocker, sheraton, swissotel | 320 | 0.887 | 0.919 | 0.850 | 0.883 | 0.959 | 0.894 | 0.881 |
| 5 | allegro, amalfi, intercontinental, palmer | 320 | 0.909 | 0.923 | 0.894 | 0.908 | 0.970 | 0.912 | 0.906 |
| **Mean ± sd** | | | **0.882 ± 0.022** | 0.897 ± 0.035 | 0.865 ± 0.022 | 0.881 ± 0.021 | 0.953 ± 0.016 | 0.893 | 0.872 |

## Conservative operating point (flag only when p ≥ 0.8)

This is how the signal is used in detection. Precision is what matters to an operator (a flag should be right); recall says how much slips through.

| Fold | Flagged | Precision | Recall |
|---|---|---|---|
| 1 | 72 of 320 | 0.972 | 0.438 |
| 2 | 61 of 320 | 0.984 | 0.375 |
| 3 | 80 of 320 | 1.000 | 0.500 |
| 4 | 62 of 320 | 1.000 | 0.388 |
| 5 | 66 of 320 | 1.000 | 0.412 |
| **Mean ± sd** | | **0.991 ± 0.013** | 0.422 ± 0.050 |

## Most indicative terms (model trained on all folds)

Toward deceptive: chicago (+7.50), my (+5.42), luxury (+4.49), my stay (+3.47), vacation (+3.17), experience (+3.11), chicago hotel (+2.93), when (+2.80), the food (+2.77), luxurious (+2.75), my room (+2.75), hilton (+2.74), food (+2.65), definitely (+2.61), recently (+2.59)

Toward truthful: location (-4.37), floor (-4.12), we (-4.00), small (-3.48), rate (-3.25), breakfast (-2.79), street (-2.77), great (-2.75), priceline (-2.72), bathroom (-2.63), on (-2.59), reviews (-2.54), no (-2.48), concierge (-2.40), large (-2.37)

## Additional checks

- Harbor Light Inn fixture (informational, 25 synthetic reviews): mean p(deceptive) planted 0.723 vs genuine 0.189; lowest planted 0.585, highest genuine 0.581

## Read this before quoting the numbers

- The deceptive reviews were written by paid crowd workers between 2011 and 2013 for 20 large Chicago hotels, in English. The model learns that style. Coordinated campaigns, paraphrased duplicates and reviews written by people who did stay are different problems, covered by the other signals.
- Hotel disjoint folds stop the model from memorising hotel names, but all 20 hotels are in one city and one market segment. Expect lower figures on independent lodging in a different region until the model is retrained on data from it.
- The corpus is CC BY-NC-SA 3.0. A model trained on it is for research and demonstration; see docs/data.md.
