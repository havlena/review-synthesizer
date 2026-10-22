# Claim duplicates: ablation of the candidate rule and the judge prompt

Generated 2026-10-22T21:35:18+00:00 on Python 3.12.10. Main model `claude-sonnet-5-5` (claim extraction), small model `claude-haiku-5-5` (pair judgments).

Set: 24 synthetic reviews, 18 true pairs, 258 pairs that should not be joined (tests/fixtures/paraphrase_set.json). The baseline report for the current default is claim_duplicates.md.

## Design

Each run extracts claims once (saved under `data/interim/claims/extraction_run1.json`, `data/interim/claims/extraction_run2.json`, not committed). Candidates are taken at both rules from that extraction, the union of candidate pairs is judged once with each prompt, and every cell is scored on the same extraction and judgments, so within a run the cells differ only in the lever.

Decision rule, fixed before the run: the product default becomes the cell with no wrong join in either run and the highest recall (mean judged recall over the runs); ties go to the stricter setting (2 shared claims before 1, then the strict prompt); if no cell beats the current default without a wrong join, the default stays. Current default: 2 shared, strict.

## Grid

| Cell | Run | Candidates: true (of 18) / non duplicate | Judged the same: true / wrong | Precision | Recall | Estimated cost |
|---|---|---|---|---|---|---|
| 2 shared, strict | 1 | 7 / 0 | 6 / 0 | 1.000 | 0.333 | $0.088 |
| 2 shared, strict | 2 | 3 / 0 | 2 / 0 | 1.000 | 0.111 | $0.087 |
| 2 shared, lenient | 1 | 7 / 0 | 6 / 0 | 1.000 | 0.333 | $0.088 |
| 2 shared, lenient | 2 | 3 / 0 | 2 / 0 | 1.000 | 0.111 | $0.087 |
| 1 shared, strict | 1 | 12 / 23 | 7 / 0 | 1.000 | 0.389 | $0.091 |
| 1 shared, strict | 2 | 12 / 20 | 7 / 0 | 1.000 | 0.389 | $0.091 |
| 1 shared, lenient | 1 | 12 / 23 | 9 / 0 | 1.000 | 0.500 | $0.092 |
| 1 shared, lenient | 2 | 12 / 20 | 6 / 0 | 1.000 | 0.333 | $0.091 |

Estimated cost per cell: the run's measured extraction tokens plus the cell's number of judgments at the measured mean tokens per judgment for its prompt.

| Cell | Wrong joins per run | Mean recall | Lost before judging / at judging (per run) |
|---|---|---|---|
| 2 shared, strict | 0, 0 | 0.222 | 11 / 1, 15 / 1 |
| 2 shared, lenient | 0, 0 | 0.222 | 11 / 1, 15 / 1 |
| 1 shared, strict | 0, 0 | 0.389 | 6 / 5, 6 / 5 |
| 1 shared, lenient | 0, 0 | 0.417 | 6 / 3, 6 / 6 |

## Decision

**1 shared, lenient**. No wrong join in either run and the highest mean recall (0.417, against 0.222 for the current default).

## Wrong joins

None in any cell or run.

## Run to run agreement

- 2 shared, strict: joined in every run 2 (C1-C2, C2-C4); run 1 only: C1-C3, C1-C4, C2-C3, C3-C4; run 2 only: none.
- 2 shared, lenient: joined in every run 2 (C1-C2, C2-C4); run 1 only: C1-C3, C1-C4, C2-C3, C3-C4; run 2 only: none.
- 1 shared, strict: joined in every run 6 (B1-B2, C1-C2, C1-C3, C1-C4, C2-C4, C3-C4); run 1 only: C2-C3; run 2 only: B2-B4.
- 1 shared, lenient: joined in every run 6 (B1-B2, C1-C2, C1-C3, C1-C4, C2-C4, C3-C4); run 1 only: B1-B4, B2-B4, C2-C3; run 2 only: none.
- Candidates at each rule, every run / one run only: 2 shared 3 / 4; 1 shared 28 / 11.

## Cost (measured tokens)

| Run | Stage | Model | Calls | Input tokens | Output tokens |
|---|---|---|---|---|---|
| 1 | extraction | claude-sonnet-5-5 | 24 | 27,798 | 3,144 |
| 1 | judge strict | claude-haiku-5-5 | 35 | 28,519 | 3,179 |
| 1 | judge lenient | claude-haiku-5-5 | 35 | 30,444 | 3,123 |
| 2 | extraction | claude-sonnet-5-5 | 24 | 27,798 | 3,109 |
| 2 | judge strict | claude-haiku-5-5 | 32 | 25,856 | 2,880 |
| 2 | judge lenient | claude-haiku-5-5 | 32 | 27,616 | 2,978 |
| all | all | claude-sonnet-5-5 | 48 | 55,596 | 6,253 |
| all | all | claude-haiku-5-5 | 134 | 112,435 | 12,160 |

Output tokens include thinking tokens.
Measured total for the whole ablation: $0.19 at list prices per million tokens as of 2026-10-06 (claude-sonnet-5-5 $2.00 input, $10.00 output; claude-haiku-5-5 $0.10 input, $0.50 output).

## How judgments scale with batch size

Pairs grow with the square of the batch, and every candidate pair costs one small model judgment, so the candidate rule sets the cost on a real batch. Projection from the share of pairs that became candidates here:

| Rule | Candidate share of pairs | Judgments at 25 reviews | Judgments at 100 reviews | Judgments at 500 reviews |
|---|---|---|---|---|
| 2 shared | 0.018 | 5 ($0.00) | 90 ($0.01) | 2,260 ($0.29) |
| 1 shared | 0.121 | 36 ($0.00) | 601 ($0.08) | 15,142 ($1.96) |

Dollar figures are small model judgments only, at the measured mean tokens per judgment. The share comes from a set where half the reviews are rewrites of each other and the rest are spread over topics; honest reviews of one real property mention the same features more often, so expect a higher share there.

## Methods

- Effort: not set; the API default for each model (claude-sonnet-5-5 high, claude-haiku-5-5 medium).
- Thinking: not set; the API default (adaptive) for each model.
- Temperature: not sent. The Messages API has rejected temperature, top_p and top_k with an error on every model since 2026-10-21, so run to run stability is measured by repeating the run, not by fixing sampling.
- Structured output: native JSON schema output (langchain-anthropic, method json_schema).
- Max output tokens per call: 16,000.
- Concurrent calls per stage: 8.
- Packages: langchain-anthropic 1.7.6, anthropic 1.17.0, langgraph 1.2.16.
- The judge prompts are recorded in full in the JSON report.
- An ablation is scored only if every extraction and judgment call succeeded.

## Limits

- The setting is chosen on the same 24 reviews it is reported on. There is no held out set, so the chosen cell's figures are optimistic for new data.
- The lenient prompt's examples of different names for one feature (a deck, a terrace, a balcony; a massage, a spa treatment, a wellness centre) are the vocabulary of cluster C in this set, which favours the lenient prompt here.
- Two runs show whether answers move between runs; they are not a distribution.
- The claim signal is measured on its own. In the graph it runs alongside the detector, and pairs the detector already joined are not judged again.
- The fictional property name in the prompts is a placeholder; the set names none.
- Cost is the token count the API reported; the dollar figure is an estimate at list prices on the stated date.
- Per cell cost is estimated from measured tokens: judgments ran once on the union of candidates, not once per cell.
