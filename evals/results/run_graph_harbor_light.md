# End to end graph run: Harbor Light Inn

Generated 2026-10-22T22:02:40+00:00 on Python 3.12.10. 25 reviews, 5 of them planted (21, 22, 23, 24, 25). 2 runs of the whole graph. Brief of run 1: brief_harbor_light.md.

## Headline

| | Run 1 | Run 2 | Golden |
|---|---|---|---|
| Displayed rating | 3.8 | 3.8 | 3.8 |
| Verified rating | 3.4 | 3.4 | 3.5 |
| Flags | 6 | 6 | |
| Genuine reviews flagged | 10 (embedding_similarity), 16 (embedding_similarity) | 10 (embedding_similarity), 16 (embedding_similarity) | none |

**At least one run does not reproduce the golden ratings.** **Genuine reviews were flagged: 10, 16.**

## Run 1 (`harbor-light-run1`)

| Signal | Score | Band | Reviews | Decision |
|---|---|---|---|---|
| duplicate_phrasing | 0.92 | high | 21, 22, 23, 24, 25 | confirmed |
| rating_burst | 0.83 | high | 21, 22, 23, 24, 25 | confirmed |
| embedding_similarity | 0.80 | high | 10, 16 | confirmed |
| embedding_similarity | 1.00 | high | 21, 22, 23, 24, 25 | confirmed |
| deceptive_language | 0.82 | high | 21 | confirmed |
| deceptive_language | 0.82 | high | 23 | confirmed |

Notes from the run and the output checks:
- none

## Run 2 (`harbor-light-run2`)

| Signal | Score | Band | Reviews | Decision |
|---|---|---|---|---|
| duplicate_phrasing | 0.92 | high | 21, 22, 23, 24, 25 | confirmed |
| rating_burst | 0.83 | high | 21, 22, 23, 24, 25 | confirmed |
| embedding_similarity | 0.80 | high | 10, 16 | confirmed |
| embedding_similarity | 1.00 | high | 21, 22, 23, 24, 25 | confirmed |
| deceptive_language | 0.82 | high | 21 | confirmed |
| deceptive_language | 0.82 | high | 23 | confirmed |

Notes from the run and the output checks:
- none

## Differences between runs

- Flags: run 1 only none; run 2 only none.
- Themes: run 1 only "Inconsistent breakfast", "Spotless, well-kept rooms", "Warm, helpful front desk", "Comfortable beds and rooms", "Unresponsive staff", "Mold, musty smell, dirty rooms", "Dated rooms, heater problems", "Value for the price", "Lakeview room not delivered"; run 2 only "Cleanliness varies by room", "Front desk staff service", "Inconsistent breakfast offering", "Dated rooms and maintenance", "Comfortable beds".
  - "Harbor views and location": 7 reviews, positive in run 1; 8, positive in run 2.
  - "Thin walls and noise": 3 reviews, negative in run 1; 6, mixed in run 2.
- Aspect scores that differ: cleanliness 1 vs 3; breakfast 3 vs 2; parking 1 vs 2; amenities None vs 5.
- Actions: run 1 only "Standardize breakfast with hot options and restock until ten", "Inspect and remediate mold and damp in bathrooms and carpets", "Service and test every room heater before cold-season stays", "Set up a staffed phone line and a late-night noise protocol", "Upgrade wifi coverage and mark parking limits before arrival"; run 2 only "Add a room-ready checklist and supervisor spot check after housekeeping", "Set up a repair and room-move process for heater failures and noise complaints", "Inspect every room for mold and dampness, and fix the source", "Set a standard breakfast menu with hot items and restocking", "Upgrade the wifi and test evening connection reliability". Same order in both runs: no.

## Cost (measured tokens)

| Run | Model | Calls | Input tokens | Output tokens |
|---|---|---|---|---|
| 1 | claude-sonnet-5-5 | 27 | 34,947 | 9,889 |
| 1 | claude-haiku-5-5 | 30 | 25,886 | 2,712 |
| 2 | claude-sonnet-5-5 | 27 | 34,881 | 9,137 |
| 2 | claude-haiku-5-5 | 28 | 24,162 | 2,614 |

Output tokens include thinking tokens.
Estimated cost of all runs: $0.34 at list prices per million tokens as of 2026-10-06 (claude-sonnet-5-5 $2.00 input, $10.00 output; claude-haiku-5-5 $0.10 input, $0.50 output).

## Methods

- Signals: duplicate_phrasing, rating_burst, embedding_similarity, deceptive_language, claim_duplicate.
- duplicate_phrasing: Jaccard word overlap at or above 0.55.
- rating_burst: 3 or more same rating reviews within 4 days.
- embedding_similarity: cosine similarity at or above 0.80.
- deceptive_language: classifier probability at or above 0.80.
- claim_duplicate: at least 1 shared specific claim, then a model judgment per pair (lenient prompt).
- Models: main claude-sonnet-5-5, small claude-haiku-5-5, embeddings sentence-transformers/paraphrase-mpnet-base-v2.
- Flag decisions: automatic: every flag confirmed without review (evaluation run).

## Limits

- One fictional property with 25 synthetic reviews, five of them a planted campaign. The run checks whether the graph reproduces the capstone result with every signal on; it says nothing about accuracy on real batches.
- Every flag was confirmed automatically. In the product a person decides each flag; here the golden result assumes all planted reviews are set aside.
- Two runs show what moves between runs; they are not a distribution.
- Cost is the token count the API reported; the dollar figure is an estimate at list prices on the stated date.
