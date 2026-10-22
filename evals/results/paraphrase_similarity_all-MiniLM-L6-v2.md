# Duplicate signals: word overlap versus embedding similarity

Generated 2026-10-22T19:37:09+00:00 on Python 3.12.10. Embedding model: `sentence-transformers/all-MiniLM-L6-v2`.

Set: 24 reviews, 3 labelled clusters of rewritten duplicates (18 true duplicate pairs) and 258 pairs that should not be joined. Synthetic; see tests/fixtures/paraphrase_set.json.

## Pair level results

| Signal | Threshold | Pairs joined | Correct | Wrong | Missed | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| Word overlap (Jaccard) | 0.55 | 0 | 0 | 0 | 18 | 0.000 | 0.000 | 0.000 |
| Embedding cosine | 0.30 | 88 | 18 | 70 | 0 | 0.205 | 1.000 | 0.340 |
| Embedding cosine | 0.40 | 37 | 16 | 21 | 2 | 0.432 | 0.889 | 0.582 |
| Embedding cosine | 0.50 | 15 | 8 | 7 | 10 | 0.533 | 0.444 | 0.485 |
| Embedding cosine | 0.60 | 3 | 3 | 0 | 15 | 1.000 | 0.167 | 0.286 |
| Embedding cosine | 0.70 | 0 | 0 | 0 | 18 | 0.000 | 0.000 | 0.000 |
| Embedding cosine **(default)** | 0.80 | 0 | 0 | 0 | 18 | 0.000 | 0.000 | 0.000 |
| Embedding cosine | 0.90 | 0 | 0 | 0 | 18 | 0.000 | 0.000 | 0.000 |

## Where the similarities fall

| Pairs | Min | Mean | p95 | Max |
|---|---|---|---|---|
| True duplicates (18) | 0.316 | 0.498 | | 0.632 |
| Everything else (258) | | 0.248 | 0.446 | 0.598 |

A threshold that separates the two exists only if the duplicates' minimum is above the others' maximum: no on this set.

| Cluster | Written as | Pairs | Min | Mean | Max |
|---|---|---|---|---|---|
| A | best hotel in the region, amazing staff, perfect rooms, top rating (positive campaign) | 6 | 0.316 | 0.429 | 0.530 |
| B | rude manager, dirty room, overcharged, never again (negative campaign) | 6 | 0.415 | 0.466 | 0.522 |
| C | lakefront balcony, gourmet breakfast, spa, best vacation ever (fabricated amenities) | 6 | 0.553 | 0.600 | 0.632 |

## What each signal adds (embedding threshold 0.80)

- True duplicate pairs caught by both signals: 0
- Caught by embeddings only: 0
- Caught by word overlap only: 0
- Missed by both: 18

Nearest non duplicate pair: G3 and G8 at cosine 0.598. The gap between that and the threshold is the margin against false positives on this set.

True duplicates still missed at the default threshold:

- C1 and C4 (cosine 0.632)
- C1 and C2 (cosine 0.626)
- C3 and C4 (cosine 0.61)
- C1 and C3 (cosine 0.59)
- C2 and C4 (cosine 0.587)
- C2 and C3 (cosine 0.553)
- A1 and A4 (cosine 0.53)
- B1 and B3 (cosine 0.522)
- B1 and B2 (cosine 0.487)
- B1 and B4 (cosine 0.481)
- B2 and B3 (cosine 0.476)
- A1 and A3 (cosine 0.465)
- A3 and A4 (cosine 0.465)
- A1 and A2 (cosine 0.422)
- B2 and B4 (cosine 0.417)
- B3 and B4 (cosine 0.415)
- A2 and A3 (cosine 0.377)
- A2 and A4 (cosine 0.316)

## Additional checks

- Harbor Light Inn fixture: planted cluster pairwise cosine min 0.949, mean 0.976; fully joined at 0.80: True; nearest genuine review to a planted one 0.669; most similar pair of genuine reviews 0.698
- Rating burst rule on this set (timing only, no text): 4 1-star reviews, 2026-07-10 to 2026-07-12; 4 5-star reviews, 2026-07-02 to 2026-07-05; 6 5-star reviews, 2026-07-19 to 2026-07-26, genuine reviews swept in: G9, G11

## Read this before quoting the numbers

- Twenty four synthetic reviews written for this project. The clusters were written to share meaning and avoid shared words, which is the case word overlap is known to miss; the distinct reviews were written to share topics without sharing claims. The figures show the mechanism and the chosen threshold's margin, not field accuracy.
- Embedding similarity joins reviews that say the same thing. Two honest guests can say the same thing. The signal is one input to a human decision, never a verdict, and the Module 6 false positive risk (a cluster of honest praise after a good weekend) is exactly what the nearest non duplicate figure watches.
- If the duplicates' minimum sits below the others' maximum, no cosine threshold on this model separates rewritten fakes from genuine reviews on the same topic. Cosine similarity measures wording and topic; it does not compare the claims being made. Claim level comparison is the job of the language model step.
