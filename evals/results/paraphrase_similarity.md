# Duplicate signals: embedding similarity across models

Set: 24 synthetic reviews, 3 labelled clusters of rewritten duplicates (see tests/fixtures/paraphrase_set.json). Operating threshold 0.85. Per model detail, including the threshold sweep, is in the linked reports.

| Model | Default | True pairs min / mean | Non duplicate max | Separable | At threshold: precision, recall, pairs | Best F1 (threshold) | Harbor Light planted min / nearest genuine / genuine pair max | Report |
|---|---|---|---|---|---|---|---|---|
| paraphrase-mpnet-base-v2 | yes | 0.559 / 0.686 | 0.703 | no | 0.00, 0.00, 0 | 0.684 (0.60) | 0.977 / 0.599 / 0.801 | [paraphrase_similarity_paraphrase-mpnet-base-v2.md](paraphrase_similarity_paraphrase-mpnet-base-v2.md) |
| all-MiniLM-L6-v2 |  | 0.316 / 0.498 | 0.598 | no | 0.00, 0.00, 0 | 0.582 (0.40) | 0.949 / 0.669 / 0.698 | [paraphrase_similarity_all-MiniLM-L6-v2.md](paraphrase_similarity_all-MiniLM-L6-v2.md) |
| all-mpnet-base-v2 |  | 0.548 / 0.681 | 0.646 | no | 0.00, 0.00, 0 | 0.750 (0.60) | 0.928 / 0.747 / 0.835 | [paraphrase_similarity_all-mpnet-base-v2.md](paraphrase_similarity_all-mpnet-base-v2.md) |

## Negative control

The closest honest reviews come to each other in the fixtures (the most similar genuine Harbor Light pair, and the most similar non duplicate pair of the paraphrase set) must both sit below the operating threshold.

- paraphrase-mpnet-base-v2: PASS: most similar genuine Harbor Light pair 0.801, most similar paraphrase set non duplicate pair 0.703, threshold 0.85
- all-MiniLM-L6-v2: PASS: most similar genuine Harbor Light pair 0.698, most similar paraphrase set non duplicate pair 0.598, threshold 0.85
- all-mpnet-base-v2: PASS: most similar genuine Harbor Light pair 0.835, most similar paraphrase set non duplicate pair 0.646, threshold 0.85

Lowest within cluster cosine per model (how detectable each kind of rewrite is):

| Model | A: best hotel in the region, amazing staff, perfect rooms, top rating (positive campaign) | B: rude manager, dirty room, overcharged, never again (negative campaign) | C: lakefront balcony, gourmet breakfast, spa, best vacation ever (fabricated amenities) |
|---|---|---|---|
| paraphrase-mpnet-base-v2 | 0.559 | 0.561 | 0.729 |
| all-MiniLM-L6-v2 | 0.316 | 0.415 | 0.553 |
| all-mpnet-base-v2 | 0.548 | 0.561 | 0.615 |

## Reading

- Light rewrites (a few words changed, the capstone case) score high on every model and are caught cleanly at the operating threshold. The embedding signal detects light rewrites, nothing more.
- At 0.85 the signal joins no heavy paraphrase pair on any model: the rewritten pairs that cleared 0.80 sat at 0.80 to 0.81, the band honest reviews of one property reach. Finding heavy paraphrase is the claim duplicate signal's job (claim_duplicates_ablation.md).
- The threshold was set with the margin observed on this synthetic data and must be re-checked on operator data, where honest reviews of one property will be more alike than these.
- Heavy paraphrases that keep the claims and change every word overlap honest reviews on the same topics on every model tested: the weakest true pair scores below the strongest non duplicate pair, so no threshold separates them. Cosine similarity measures wording and topic, not the claims being made.
- Rewrites built on concrete specifics (named amenities) are the most detectable; generic praise or generic complaint rewrites are the least, because generic claims look like generic honest reviews.
- The default model is the one with the best behaviour at the operating point: any recall with no false positives on this set, and the widest margin between planted and genuine reviews on Harbor Light.
- Claim level comparison of reviews is the job of the language model step, not of this signal.
