# Claim duplicates at the product default (2 shared, strict), with the same property control

Generated 2026-10-22T23:45:01+00:00 on Python 3.12.10. Main model `claude-sonnet-5-5` (claim extraction), small model `claude-haiku-5-5` (pair judgments).

Recall set: 24 synthetic reviews, 18 true pairs, 258 pairs that should not be joined, on different topics (tests/fixtures/paraphrase_set.json). Control set: the 20 genuine Harbor Light reviews, 190 honest pairs of one property, every join wrong (tests/fixtures/harbor_light_inn.json, ids 1 to 20).

## Context

- The product default, 2 shared claims with the strict judge, is an author decision that overrides the ablation's recall rule; the reasons are in claim_duplicates_ablation.md and docs/evaluation.md.
- This run uses the corrected prompts: until this report both sets were sent with a placeholder property line, while the graph sends the real one.

## Design

Each run extracts claims once per set (saved under `data/interim/claims/claim_duplicates_default/extraction_run1.json`, `data/interim/claims/claim_duplicates_default/control_extraction_run1.json`, `data/interim/claims/claim_duplicates_default/extraction_run2.json`, `data/interim/claims/claim_duplicates_default/control_extraction_run2.json`, not committed). Candidates are taken at both rules from that extraction, the union of candidate pairs is judged once with each prompt, and every cell is scored on the same extraction and judgments, so within a run the cells differ only in the lever.

Property line in the prompts: paraphrase set `Property: Birch Point Hotel (hotel, fictional, Northern Michigan)`; control set `Property: Harbor Light Inn (inn, fictional, Northern Michigan)`.

## Grid

| Cell | Run | Paraphrase candidates: true (of 18) / non duplicate | Judged the same: true / wrong | Precision | Recall | Control candidates (of 190) | Control wrong joins |
|---|---|---|---|---|---|---|---|
| 2 shared, strict | 1 | 4 / 0 | 3 / 0 | 1.000 | 0.167 | 2 | 0 |
| 2 shared, strict | 2 | 3 / 0 | 3 / 0 | 1.000 | 0.167 | 2 | 0 |

| Cell | Wrong joins per run: paraphrase / control | Qualifies | Mean paraphrase recall |
|---|---|---|---|
| 2 shared, strict | 0 / 0, 0 / 0 | yes | 0.167 |

## Wrong joins

None in any cell or run.

## Run to run agreement (paraphrase set joins)

- 2 shared, strict: joined in every run 3 (C1-C2, C1-C4, C2-C4); run 1 only: none; run 2 only: none.
- Candidates at each rule, every run / one run only: 2 shared: paraphrase 3 / 1, control 2 / 0.

## Cost (measured tokens)

| Run | Stage | Model | Calls | Input tokens | Output tokens |
|---|---|---|---|---|---|
| 1 | paraphrase extraction | claude-sonnet-5-5 | 24 | 27,966 | 3,343 |
| 1 | paraphrase judge strict | claude-haiku-5-5 | 4 | 3,390 | 342 |
| 1 | control extraction | claude-sonnet-5-5 | 20 | 23,257 | 2,885 |
| 1 | control judge strict | claude-haiku-5-5 | 2 | 1,705 | 228 |
| 2 | paraphrase extraction | claude-sonnet-5-5 | 24 | 27,966 | 3,276 |
| 2 | paraphrase judge strict | claude-haiku-5-5 | 3 | 2,529 | 212 |
| 2 | control extraction | claude-sonnet-5-5 | 20 | 23,257 | 2,887 |
| 2 | control judge strict | claude-haiku-5-5 | 2 | 1,708 | 235 |
| all | all | claude-sonnet-5-5 | 88 | 102,446 | 12,391 |
| all | all | claude-haiku-5-5 | 11 | 9,332 | 1,017 |

Output tokens include thinking tokens.
Measured total for the whole ablation: $0.33 at list prices per million tokens as of 2026-10-06 (claude-sonnet-5-5 $2.00 input, $10.00 output; claude-haiku-5-5 $0.10 input, $0.50 output).

## How judgments scale with batch size

Pairs grow with the square of the batch, and every candidate pair costs one small model judgment, so the candidate rule sets the cost on a real batch. The projection uses the Harbor Light candidate share: honest reviews of one property, the realistic case. The paraphrase share is shown for comparison.

| Rule | Harbor Light share | Paraphrase share | Judgments at 25 reviews | Judgments at 100 reviews | Judgments at 500 reviews |
|---|---|---|---|---|---|
| 2 shared | 0.011 | 0.013 | 3 ($0.00) | 52 ($0.01) | 1,313 ($0.17) |

Dollar figures are small model judgments only, at the measured mean tokens per judgment over every judging stage.

## What the control bounds

Zero wrong joins on 190 honest pairs of one property, in one run, puts the per pair false join rate under about 1.6% at 95% confidence (rule of three, treating pairs as independent, which they are not: they share reviews). That is a loose bound: at 1.6% a batch of 500 reviews (124,750 pairs) could still hold about 1,970 wrong joins. It says nothing about other properties, longer reviews, other languages, or reviews written to share more features than these do.

## Methods

- Effort: not set; the API default for each model (claude-sonnet-5-5 high, claude-haiku-5-5 medium).
- Thinking: not set; the API default (adaptive) for each model.
- Temperature: not sent. The Messages API has rejected temperature, top_p and top_k with an error on every model since 2026-10-21, so run to run stability is measured by repeating the run, not by fixing sampling.
- Structured output: native JSON schema output (langchain-anthropic, method json_schema).
- Max output tokens per call: 16,000.
- Concurrent calls per stage: 8.
- Packages: langchain-anthropic 1.7.6, anthropic 1.17.0, langgraph 1.2.16.
- Each set's prompts are built by the graph's own prompt builders, property line included (see Design).
- The claim signal is measured on its own: the detector's skip rule is off on both sets, so no pair is hidden by an earlier join (the end to end run showed that it can be).
- The judge prompts are recorded in full in the JSON report.
- An ablation is scored only if every extraction and judgment call succeeded.

## Limits

- The sets are small, synthetic and written by the author of the detector: 24 reviews for recall, 20 genuine reviews of one fictional property as the control.
- Any held out set must include honest reviews of one property that mention the same features. Negatives on different topics cannot show the failure that matters most.
- Two runs show whether answers move between runs; they are not a distribution.
- The claim signal is measured on its own. In the graph it runs alongside the detector, and pairs the detector already joined are not judged again.
- Cost is the token count the API reported; the dollar figure is an estimate at list prices on the stated date.
