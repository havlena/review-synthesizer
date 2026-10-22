# End to end graph run: Harbor Light Inn

Generated 2026-10-22T22:54:26+00:00 on Python 3.12.10. 25 reviews, 5 of them planted (21, 22, 23, 24, 25). 1 runs of the whole graph. Brief of run 1: brief_harbor_light.md.

## Headline

| | Run 1 | Golden |
|---|---|---|
| Displayed rating | 3.8 | 3.8 |
| Verified rating | 3.4 | 3.5 |
| Flags | 6 | |
| Genuine reviews flagged | 10 (claim_duplicate), 16 (claim_duplicate) | none |

**At least one run does not reproduce the golden ratings.** **Genuine reviews were flagged: 10, 16.**

## Run 1 (`harbor-light-run1`)

| Signal | Score | Band | Reviews | Decision |
|---|---|---|---|---|
| duplicate_phrasing | 0.92 | high | 21, 22, 23, 24, 25 | confirmed |
| rating_burst | 0.83 | high | 21, 22, 23, 24, 25 | confirmed |
| embedding_similarity | 1.00 | high | 21, 22, 23, 24, 25 | confirmed |
| deceptive_language | 0.82 | low | 21 | confirmed |
| deceptive_language | 0.82 | low | 23 | confirmed |
| claim_duplicate | 0.33 | low | 10, 16 | confirmed |

Notes from the run and the output checks:
- none

## Differences between runs

One run only.

## Cost (measured tokens)

| Run | Model | Calls | Input tokens | Output tokens |
|---|---|---|---|---|
| 1 | claude-sonnet-5-5 | 27 | 34,818 | 8,248 |
| 1 | claude-haiku-5-5 | 29 | 24,981 | 2,596 |

Output tokens include thinking tokens.
Estimated cost of all runs: $0.16 at list prices per million tokens as of 2026-10-06 (claude-sonnet-5-5 $2.00 input, $10.00 output; claude-haiku-5-5 $0.10 input, $0.50 output).

## Methods

- Signals: duplicate_phrasing, rating_burst, embedding_similarity, deceptive_language, claim_duplicate.
- duplicate_phrasing: Jaccard word overlap at or above 0.55.
- rating_burst: 3 or more same rating reviews within 4 days.
- embedding_similarity: cosine similarity at or above 0.85.
- deceptive_language: classifier probability at or above 0.80.
- claim_duplicate: at least 1 shared specific claim, then a model judgment per pair (lenient prompt).
- Models: main claude-sonnet-5-5, small claude-haiku-5-5, embeddings sentence-transformers/paraphrase-mpnet-base-v2.
- Flag decisions: automatic: every flag confirmed without review (evaluation run).

## Limits

- One fictional property with 25 synthetic reviews, five of them a planted campaign. The run checks whether the graph reproduces the capstone result with every signal on; it says nothing about accuracy on real batches.
- Every flag was confirmed automatically. In the product a person decides each flag; here the golden result assumes all planted reviews are set aside.
- Two runs show what moves between runs; they are not a distribution.
- Cost is the token count the API reported; the dollar figure is an estimate at list prices on the stated date.
