# End to end graph run: Harbor Light Inn

Generated 2026-10-23T00:23:18+00:00 on Python 3.12.10. 25 reviews, 5 of them planted (21, 22, 23, 24, 25). 2 runs of the whole graph. Brief of run 1: brief_harbor_light.md.

## Headline

| | Run 1 | Run 2 | Golden |
|---|---|---|---|
| Displayed rating | 3.8 | 3.8 | 3.8 |
| Verified rating | 3.5 | 3.5 | 3.5 |
| Flags | 5 | 5 | |
| Genuine reviews flagged | none | none | none |

Every run reproduces the golden ratings.

## Run 1 (`harbor-light-run1`)

| Signal | Score | Band | Reviews | Decision |
|---|---|---|---|---|
| duplicate_phrasing | 0.92 | high | 21, 22, 23, 24, 25 | confirmed |
| rating_burst | 0.83 | high | 21, 22, 23, 24, 25 | confirmed |
| embedding_similarity | 1.00 | high | 21, 22, 23, 24, 25 | confirmed |
| deceptive_language | 0.82 | low | 21 | confirmed |
| deceptive_language | 0.82 | low | 23 | confirmed |

Notes from the run and the output checks:
- none

## Run 2 (`harbor-light-run2`)

| Signal | Score | Band | Reviews | Decision |
|---|---|---|---|---|
| duplicate_phrasing | 0.92 | high | 21, 22, 23, 24, 25 | confirmed |
| rating_burst | 0.83 | high | 21, 22, 23, 24, 25 | confirmed |
| embedding_similarity | 1.00 | high | 21, 22, 23, 24, 25 | confirmed |
| deceptive_language | 0.82 | low | 21 | confirmed |
| deceptive_language | 0.82 | low | 23 | confirmed |

Notes from the run and the output checks:
- none

## Differences between runs

- Flags: run 1 only none; run 2 only none.
- Themes: run 1 only "Clean rooms, some lapses", "Comfortable beds, dated rooms", "Walkable harbor location", "Friendly but inconsistent service", "Uneven breakfast offering", "Quiet rooms, thin walls"; run 2 only "Harbor location and views", "Clean, spotless rooms", "Inconsistent breakfast offering", "Friendly, helpful staff", "Dated rooms, heater trouble", "Thin walls and noise".
- Aspect scores that differ: none.
- Actions: run 1 only "Add a room inspection checklist to housekeeping routines", "Inspect and remediate mold and damp in affected bathrooms and carpets", "Expand breakfast with hot items and restock before 8 am", "Set front desk standards for phone answering and noise complaints", "Service and replace faulty room heaters before the cold season"; run 2 only "Standardize breakfast with a hot item and restock before 9am", "Inspect and mold-treat bathrooms and musty rooms, then dehumidify", "Service or replace rattling and failing room heaters before cold season", "Set quiet hours and a staffed process for night noise complaints", "Upgrade wifi capacity and coverage for evening and work-call use". Same order in both runs: no.

## Synthesis variance, before and after

Same input (the 20 verified Harbor Light reviews) in every run. Before: the model chose aspect scores; after: scores are computed from the claims, serious issues are listed separately, and themes need one aspect, two verified reviews, at most eight.

| | before 1 (7a2b779) | before 2 | after 1 | after 2 |
|---|---|---|---|---|
| Themes | 9 | 10 | 8 | 8 |
| cleanliness | 3 | 3 | 3 | 3 |
| service | 3 | 3 | 3 | 3 |
| breakfast | 3 | 3 | 2 | 2 |
| room comfort | 3 | 3 | 3 | 3 |
| value | 3 | 3 | 3 | 3 |
| location | 5 | 5 | 5 | 5 |
| noise | 3 | 3 | 3 | 3 |
| parking | 2 | 1 | 2 | 2 |
| wifi | 1 | 1 | 1 | 1 |
| other | 3 | 4 | - | - |
| amenities | - | - | unscored | unscored |
| Serious issues | 0 | 0 | 1 | 1 |
| Actions | 5 | 5 | 5 | 5 |

Within each pair of runs:

- before 1 (7a2b779) and before 2: 8 theme(s) with identical review id sets (of 9 and 10); 8 of 10 aspect scores equal (differ: other 3 vs 4; parking 2 vs 1); serious issues in both: none; identical action titles: 0.
- after 1 and after 2: 3 theme(s) with identical review id sets (of 8 and 8); 10 of 10 aspect scores equal; serious issues in both: health, review 5; identical action titles: 0.
- Serious issues, after 1: health, review 5.
- Serious issues, after 2: health, review 5.

## Cost (measured tokens)

| Run | Model | Calls | Input tokens | Output tokens |
|---|---|---|---|---|
| 1 | claude-sonnet-5-5 | 27 | 35,670 | 7,236 |
| 1 | claude-haiku-5-5 | 2 | 1,706 | 223 |
| 2 | claude-sonnet-5-5 | 27 | 35,659 | 6,937 |
| 2 | claude-haiku-5-5 | 3 | 2,543 | 319 |

Output tokens include thinking tokens.
Estimated cost of all runs: $0.29 at list prices per million tokens as of 2026-10-06 (claude-sonnet-5-5 $2.00 input, $10.00 output; claude-haiku-5-5 $0.10 input, $0.50 output).

## Methods

- Signals: duplicate_phrasing, rating_burst, embedding_similarity, deceptive_language, claim_duplicate.
- duplicate_phrasing: Jaccard word overlap at or above 0.55.
- rating_burst: 3 or more same rating reviews within 4 days.
- embedding_similarity: cosine similarity at or above 0.85.
- deceptive_language: classifier probability at or above 0.80.
- claim_duplicate: at least 2 shared specific claims, then a model judgment per pair (strict prompt).
- Models: main claude-sonnet-5-5, small claude-haiku-5-5, embeddings sentence-transformers/paraphrase-mpnet-base-v2.
- Flag decisions: automatic: every flag confirmed without review (evaluation run).

## Limits

- One fictional property with 25 synthetic reviews, five of them a planted campaign. The run checks whether the graph reproduces the capstone result with every signal on; it says nothing about accuracy on real batches.
- Every flag was confirmed automatically. In the product a person decides each flag; here the golden result assumes all planted reviews are set aside.
- Two runs show what moves between runs; they are not a distribution.
- Cost is the token count the API reported; the dollar figure is an estimate at list prices on the stated date.
