# Duplicate signals: word overlap versus embedding similarity

Generated 2026-10-22T19:37:07+00:00 on Python 3.12.10. Embedding model: `sentence-transformers/paraphrase-mpnet-base-v2`.

Set: 24 reviews, 3 labelled clusters of rewritten duplicates (18 true duplicate pairs) and 258 pairs that should not be joined. Synthetic; see tests/fixtures/paraphrase_set.json.

## Pair level results

| Signal | Threshold | Pairs joined | Correct | Wrong | Missed | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| Word overlap (Jaccard) | 0.55 | 0 | 0 | 0 | 18 | 0.000 | 0.000 | 0.000 |
| Embedding cosine | 0.30 | 162 | 18 | 144 | 0 | 0.111 | 1.000 | 0.200 |
| Embedding cosine | 0.40 | 87 | 18 | 69 | 0 | 0.207 | 1.000 | 0.343 |
| Embedding cosine | 0.50 | 47 | 18 | 29 | 0 | 0.383 | 1.000 | 0.554 |
| Embedding cosine | 0.60 | 20 | 13 | 7 | 5 | 0.650 | 0.722 | 0.684 |
| Embedding cosine | 0.70 | 10 | 9 | 1 | 9 | 0.900 | 0.500 | 0.643 |
| Embedding cosine **(default)** | 0.80 | 3 | 3 | 0 | 15 | 1.000 | 0.167 | 0.286 |
| Embedding cosine | 0.90 | 0 | 0 | 0 | 18 | 0.000 | 0.000 | 0.000 |

## Where the similarities fall

| Pairs | Min | Mean | p95 | Max |
|---|---|---|---|---|
| True duplicates (18) | 0.559 | 0.686 | | 0.812 |
| Everything else (258) | | 0.330 | 0.544 | 0.703 |

A threshold that separates the two exists only if the duplicates' minimum is above the others' maximum: no on this set.

| Cluster | Written as | Pairs | Min | Mean | Max |
|---|---|---|---|---|---|
| A | best hotel in the region, amazing staff, perfect rooms, top rating (positive campaign) | 6 | 0.559 | 0.618 | 0.724 |
| B | rude manager, dirty room, overcharged, never again (negative campaign) | 6 | 0.561 | 0.653 | 0.741 |
| C | lakefront balcony, gourmet breakfast, spa, best vacation ever (fabricated amenities) | 6 | 0.729 | 0.786 | 0.812 |

## What each signal adds (embedding threshold 0.80)

- True duplicate pairs caught by both signals: 0
- Caught by embeddings only: 3
- Caught by word overlap only: 0
- Missed by both: 15

Nearest non duplicate pair: G1 and G12 at cosine 0.703. The gap between that and the threshold is the margin against false positives on this set.

True duplicates still missed at the default threshold:

- C1 and C3 (cosine 0.783)
- C2 and C4 (cosine 0.78)
- B1 and B2 (cosine 0.741)
- C2 and C3 (cosine 0.729)
- A3 and A4 (cosine 0.724)
- B1 and B4 (cosine 0.717)
- B2 and B3 (cosine 0.66)
- B1 and B3 (cosine 0.643)
- A1 and A2 (cosine 0.632)
- A1 and A4 (cosine 0.626)
- B2 and B4 (cosine 0.599)
- A2 and A3 (cosine 0.584)
- A1 and A3 (cosine 0.581)
- B3 and B4 (cosine 0.561)
- A2 and A4 (cosine 0.559)

## Additional checks

- Harbor Light Inn fixture: planted cluster pairwise cosine min 0.977, mean 0.985; fully joined at 0.80: True; nearest genuine review to a planted one 0.599; most similar pair of genuine reviews 0.801
- Rating burst rule on this set (timing only, no text): 4 1-star reviews, 2026-07-10 to 2026-07-12; 4 5-star reviews, 2026-07-02 to 2026-07-05; 6 5-star reviews, 2026-07-19 to 2026-07-26, genuine reviews swept in: G9, G11

## Read this before quoting the numbers

- Twenty four synthetic reviews written for this project. The clusters were written to share meaning and avoid shared words, which is the case word overlap is known to miss; the distinct reviews were written to share topics without sharing claims. The figures show the mechanism and the chosen threshold's margin, not field accuracy.
- Embedding similarity joins reviews that say the same thing. Two honest guests can say the same thing. The signal is one input to a human decision, never a verdict, and the Module 6 false positive risk (a cluster of honest praise after a good weekend) is exactly what the nearest non duplicate figure watches.
- If the duplicates' minimum sits below the others' maximum, no cosine threshold on this model separates rewritten fakes from genuine reviews on the same topic. Cosine similarity measures wording and topic; it does not compare the claims being made. Claim level comparison is the job of the language model step.
