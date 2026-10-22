# Claim duplicates: extraction plus pairwise judgment

Generated 2026-10-22T21:24:06+00:00 on Python 3.12.10. Main model `claude-sonnet-5-5` (claim extraction), small model `claude-haiku-5-5` (pair judgments).

Set: 24 synthetic reviews, 3 labelled clusters of rewritten duplicates (18 true pairs) and 258 pairs that should not be joined (tests/fixtures/paraphrase_set.json). Same set and pair level scoring as the embedding evaluation (paraphrase_similarity.md).

## Stages

Candidate stage: pairs sharing at least 2 specific claims by aspect and polarity. Judged stage: the small model's yes or no on each candidate. Candidate recall is the ceiling on judged recall.

| | Run 1 | Run 2 |
|---|---|---|
| Candidate pairs | 6 | 5 |
| True pairs that became candidates (of 18) | 6 | 5 |
| Non duplicate pairs that became candidates | 0 | 0 |
| Candidate recall | 0.333 | 0.278 |
| Pairs judged the same | 6 | 3 |
| True candidates judged the same | 6 of 6 | 3 of 5 |
| Non duplicate pairs judged the same | 0 | 0 |
| Judged precision | 1.000 | 1.000 |
| Judged recall (of 18) | 0.333 | 0.167 |
| Grouped pairs (through chains): correct / wrong | 7 / 0 | 6 / 0 |

## Per cluster

| Cluster | What the rewrites keep | True pairs | Mean specific claims per review | Candidates | Judged the same |
|---|---|---|---|---|---|
| A | best hotel in the region, amazing staff, perfect rooms, top rating (positive campaign) | 6 | 0.0 / 0.0 | 0 / 0 | 0 / 0 |
| B | rude manager, dirty room, overcharged, never again (negative campaign) | 6 | 1.8 / 1.5 | 1 / 1 | 1 / 0 |
| C | lakefront balcony, gourmet breakfast, spa, best vacation ever (fabricated amenities) | 6 | 2.8 / 2.5 | 5 / 4 | 5 / 3 |

Values separated by a slash are run 1 / run 2.

## Run to run agreement

Pairs judged the same in every run: 3 (C1-C2, C1-C3, C2-C4).

- Judged the same in run 1 only: 3 (B2-B3, C2-C3, C3-C4).
- Judged the same in run 2 only: 0 (none).

Candidate pairs in every run: 5.
- Candidate in run 1 only: 1 (C3-C4).
- Candidate in run 2 only: 0 (none).

## Run 1: true pairs missed and wrong joins

| Pair | Lost at | Shared specific claims | Judgment |
|---|---|---|---|
| A1-A2 | not a candidate | 0 |  |
| A1-A3 | not a candidate | 0 |  |
| A1-A4 | not a candidate | 0 |  |
| A2-A3 | not a candidate | 0 |  |
| A2-A4 | not a candidate | 0 |  |
| A3-A4 | not a candidate | 0 |  |
| B1-B2 | not a candidate | 1 |  |
| B1-B3 | not a candidate | 1 |  |
| B1-B4 | not a candidate | 1 |  |
| B2-B4 | not a candidate | 1 |  |
| B3-B4 | not a candidate | 1 |  |
| C1-C4 | not a candidate | 1 |  |

No non duplicate pair judged the same.

## Run 2: true pairs missed and wrong joins

| Pair | Lost at | Shared specific claims | Judgment |
|---|---|---|---|
| A1-A2 | not a candidate | 0 |  |
| A1-A3 | not a candidate | 0 |  |
| A1-A4 | not a candidate | 0 |  |
| A2-A3 | not a candidate | 0 |  |
| A2-A4 | not a candidate | 0 |  |
| A3-A4 | not a candidate | 0 |  |
| B1-B2 | not a candidate | 1 |  |
| B1-B3 | not a candidate | 1 |  |
| B1-B4 | not a candidate | 1 |  |
| B2-B3 | judged different | 2 | Both reviews report the final bill exceeding the quoted rate, but B2 says grime covers the suite while B3 says housekeeping skipped the unit, which are different cleanliness claims. |
| B2-B4 | not a candidate | 1 |  |
| B3-B4 | not a candidate | 1 |  |
| C1-C4 | not a candidate | 1 |  |
| C2-C3 | judged different | 2 | Both praise lake views, breakfast and wellness/massage, but C2 specifically claims a private terrace, chef-quality breakfast and a massage, while C3 claims a generic deck, five-star dining and a world-class wellness centre, so the concrete details differ. |
| C3-C4 | not a candidate | 1 |  |

No non duplicate pair judged the same.

## Cost (measured tokens)

| Run | Model | Calls | Input tokens | Output tokens |
|---|---|---|---|---|
| 1 | claude-sonnet-5-5 | 24 | 27,798 | 3,020 |
| 1 | claude-haiku-5-5 | 6 | 5,001 | 435 |
| 2 | claude-sonnet-5-5 | 24 | 27,798 | 3,063 |
| 2 | claude-haiku-5-5 | 5 | 4,097 | 398 |
| all | claude-sonnet-5-5 | 48 | 55,596 | 6,083 |
| all | claude-haiku-5-5 | 11 | 9,098 | 833 |

Output tokens include thinking tokens.
Estimated cost of all runs: $0.17 at list prices per million tokens as of 2026-10-06 (claude-sonnet-5-5 $2.00 input, $10.00 output; claude-haiku-5-5 $0.10 input, $0.50 output).

## Methods

- Effort: not set; the API default for each model (claude-sonnet-5-5 high, claude-haiku-5-5 medium).
- Thinking: not set; the API default (adaptive) for each model.
- Temperature: not sent. The Messages API has rejected temperature, top_p and top_k with an error on every model since 2026-10-21, so run to run stability is measured by repeating the run, not by fixing sampling.
- Structured output: native JSON schema output (langchain-anthropic, method json_schema).
- Max output tokens per call: 16,000.
- Concurrent calls per stage: 8.
- Packages: langchain-anthropic 1.7.6, anthropic 1.17.0, langgraph 1.2.16.
- Word overlap at 0.55 joins 0 of the 18 true pairs, so the graph's rule of skipping pairs the detector already joined would not change this result; it was switched off to measure the claim signal on its own.
- A run is scored only if every extraction and judgment call succeeded.

## Limits

- Twenty four synthetic reviews written for this project by the author of the detector. The figures show the mechanism and where it loses pairs, not field accuracy.
- Two runs show whether answers move between runs; they are not a distribution.
- The claim signal is measured on its own. In the graph it runs alongside the detector, and pairs the detector already joined are not judged again.
- The fictional property name in the prompts is a placeholder; the set names none.
- Cost is the token count the API reported; the dollar figure is an estimate at list prices on the stated date.
