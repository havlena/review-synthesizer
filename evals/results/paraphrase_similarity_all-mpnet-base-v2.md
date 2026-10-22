# Duplicate signals: word overlap versus embedding similarity

Generated 2026-10-22T22:28:32+00:00 on Python 3.12.10. Embedding model: `sentence-transformers/all-mpnet-base-v2`.

Set: 24 reviews, 3 labelled clusters of rewritten duplicates (18 true duplicate pairs) and 258 pairs that should not be joined. Synthetic; see tests/fixtures/paraphrase_set.json.

## Pair level results

| Signal | Threshold | Pairs joined | Correct | Wrong | Missed | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| Word overlap (Jaccard) | 0.55 | 0 | 0 | 0 | 18 | 0.000 | 0.000 | 0.000 |
| Embedding cosine | 0.30 | 198 | 18 | 180 | 0 | 0.091 | 1.000 | 0.167 |
| Embedding cosine | 0.40 | 116 | 18 | 98 | 0 | 0.155 | 1.000 | 0.269 |
| Embedding cosine | 0.50 | 59 | 18 | 41 | 0 | 0.305 | 1.000 | 0.468 |
| Embedding cosine | 0.60 | 22 | 15 | 7 | 3 | 0.682 | 0.833 | 0.750 |
| Embedding cosine | 0.70 | 8 | 8 | 0 | 10 | 1.000 | 0.444 | 0.615 |
| Embedding cosine | 0.80 | 0 | 0 | 0 | 18 | 0.000 | 0.000 | 0.000 |
| Embedding cosine | 0.90 | 0 | 0 | 0 | 18 | 0.000 | 0.000 | 0.000 |

## Where the similarities fall

| Pairs | Min | Mean | p95 | Max |
|---|---|---|---|---|
| True duplicates (18) | 0.548 | 0.681 | | 0.774 |
| Everything else (258) | | 0.367 | 0.566 | 0.646 |

A threshold that separates the two exists only if the duplicates' minimum is above the others' maximum: no on this set.

| Cluster | Written as | Pairs | Min | Mean | Max |
|---|---|---|---|---|---|
| A | best hotel in the region, amazing staff, perfect rooms, top rating (positive campaign) | 6 | 0.548 | 0.663 | 0.732 |
| B | rude manager, dirty room, overcharged, never again (negative campaign) | 6 | 0.561 | 0.661 | 0.769 |
| C | lakefront balcony, gourmet breakfast, spa, best vacation ever (fabricated amenities) | 6 | 0.615 | 0.719 | 0.774 |

## What each signal adds (embedding threshold 0.85)

- True duplicate pairs caught by both signals: 0
- Caught by embeddings only: 0
- Caught by word overlap only: 0
- Missed by both: 18

Nearest non duplicate pair: C3 and G12 at cosine 0.646. The gap between that and the threshold is the margin against false positives on this set.

True duplicates still missed at the default threshold:

- C3 and C4 (cosine 0.774)
- B1 and B2 (cosine 0.769)
- C1 and C2 (cosine 0.743)
- C1 and C4 (cosine 0.736)
- C1 and C3 (cosine 0.733)
- A1 and A4 (cosine 0.732)
- C2 and C4 (cosine 0.715)
- B1 and B4 (cosine 0.706)
- A2 and A4 (cosine 0.695)
- A1 and A2 (cosine 0.686)
- B2 and B3 (cosine 0.671)
- B2 and B4 (cosine 0.665)
- A3 and A4 (cosine 0.664)
- A2 and A3 (cosine 0.653)
- C2 and C3 (cosine 0.615)
- B3 and B4 (cosine 0.593)
- B1 and B3 (cosine 0.561)
- A1 and A3 (cosine 0.548)

## Additional checks

- Harbor Light Inn fixture: planted cluster pairwise cosine min 0.928, mean 0.965; fully joined at 0.85: True; nearest genuine review to a planted one 0.747; most similar pair of genuine reviews 0.835
- Rating burst rule on this set (timing only, no text): 4 1-star reviews, 2026-07-10 to 2026-07-12; 4 5-star reviews, 2026-07-02 to 2026-07-05; 6 5-star reviews, 2026-07-19 to 2026-07-26, genuine reviews swept in: G9, G11
- Negative control: PASS: most similar genuine Harbor Light pair 0.835, most similar paraphrase set non duplicate pair 0.646, threshold 0.85

## Read this before quoting the numbers

- Twenty four synthetic reviews written for this project. The clusters were written to share meaning and avoid shared words, which is the case word overlap is known to miss; the distinct reviews were written to share topics without sharing claims. The figures show the mechanism and the chosen threshold's margin, not field accuracy.
- Embedding similarity joins reviews that say the same thing. Two honest guests can say the same thing. The signal is one input to a human decision, never a verdict, and the Module 6 false positive risk (a cluster of honest praise after a good weekend) is exactly what the nearest non duplicate figure watches.
- If the duplicates' minimum sits below the others' maximum, no cosine threshold on this model separates rewritten fakes from genuine reviews on the same topic. Cosine similarity measures wording and topic; it does not compare the claims being made. Claim level comparison is the job of the language model step.
- The operating threshold was set with the margin observed on this synthetic data and the Harbor Light fixture. It must be re-checked on operator data, where honest reviews of one property will be more alike than these.
