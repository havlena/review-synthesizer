# Claim duplicates: ablation with a same property negative control

Generated 2026-10-22T22:51:37+00:00 on Python 3.12.10. Main model `claude-sonnet-5-5` (claim extraction), small model `claude-haiku-5-5` (pair judgments).

Recall set: 24 synthetic reviews, 18 true pairs, 258 pairs that should not be joined, on different topics (tests/fixtures/paraphrase_set.json). Control set: the 20 genuine Harbor Light reviews, 190 honest pairs of one property, every join wrong (tests/fixtures/harbor_light_inn.json, ids 1 to 20).

## Earlier version of this ablation

- First version (2026-10-22, commit 61ab971): paraphrase set only. Its negatives were in sample and on different topics. Mean paraphrase recall 0.222 (2 shared, strict), 0.222 (2 shared, lenient), 0.389 (1 shared, strict), 0.417 (1 shared, lenient); no wrong join in any cell. The rule then (no wrong join, highest recall) chose 1 shared claim with the lenient judge, and it became the product default.
- The Harbor Light end to end run at that default (commit 8173b18) joined genuine reviews 10 and 16 on one shared claim, the waterfront location. The pair had been hidden in earlier runs because embedding similarity joined it first and the graph does not judge pairs the detector already joined.
- This version adds the 20 genuine Harbor Light reviews as a control set of honest pairs of one property, every join wrong, and a rule that disqualifies any cell with a wrong join on either set.

## Design

Each run extracts claims once per set (saved under `data/interim/claims/extraction_run1.json`, `data/interim/claims/control_extraction_run1.json`, `data/interim/claims/extraction_run2.json`, `data/interim/claims/control_extraction_run2.json`, not committed). Candidates are taken at both rules from that extraction, the union of candidate pairs is judged once with each prompt, and every cell is scored on the same extraction and judgments, so within a run the cells differ only in the lever.

Property line in the prompts: paraphrase set `Property: Lakeshore hotel (fictional) (hotel) (placeholder; the graph sends the real name)`; control set `Property: Lakeshore hotel (fictional) (hotel) (placeholder; the graph sends the real name)`.

Decision rule, fixed before the run: a cell qualifies only with no wrong join on either set in either run. Among qualifying cells, the highest mean paraphrase recall wins; ties go to the stricter setting (2 shared claims before 1, then the strict prompt). If no cell qualifies, nothing changes. Default before this run: 1 shared, lenient.

## Grid

| Cell | Run | Paraphrase candidates: true (of 18) / non duplicate | Judged the same: true / wrong | Precision | Recall | Control candidates (of 190) | Control wrong joins |
|---|---|---|---|---|---|---|---|
| 2 shared, strict | 1 | 6 / 0 | 6 / 0 | 1.000 | 0.333 | 2 | 0 |
| 2 shared, strict | 2 | 7 / 0 | 6 / 0 | 1.000 | 0.333 | 3 | 0 |
| 2 shared, lenient | 1 | 6 / 0 | 6 / 0 | 1.000 | 0.333 | 2 | 0 |
| 2 shared, lenient | 2 | 7 / 0 | 6 / 0 | 1.000 | 0.333 | 3 | 0 |
| 1 shared, strict | 1 | 12 / 21 | 7 / 0 | 1.000 | 0.389 | 36 | 0 |
| 1 shared, strict | 2 | 12 / 24 | 9 / 0 | 1.000 | 0.500 | 33 | 0 |
| 1 shared, lenient | 1 | 12 / 21 | 9 / 0 | 1.000 | 0.500 | 36 | 0 |
| 1 shared, lenient | 2 | 12 / 24 | 8 / 0 | 1.000 | 0.444 | 33 | 0 |

| Cell | Wrong joins per run: paraphrase / control | Qualifies | Mean paraphrase recall |
|---|---|---|---|
| 2 shared, strict | 0 / 0, 0 / 0 | yes | 0.333 |
| 2 shared, lenient | 0 / 0, 0 / 0 | yes | 0.333 |
| 1 shared, strict | 0 / 0, 0 / 0 | yes | 0.444 |
| 1 shared, lenient | 0 / 0, 0 / 0 | yes | 0.472 |

## Decision by the rule

**1 shared, lenient**. Qualifying cells (no wrong join on either set in either run): 2 shared, strict, 2 shared, lenient, 1 shared, strict, 1 shared, lenient. Highest mean paraphrase recall: 1 shared, lenient (0.472).

## Decision by the author (overrides the rule)

The product default is 2 shared claims with the strict judge, set by the author on 2026-10-22. The rule's record stands as measured: it chose 1 shared claim with the lenient judge in both versions of this ablation, and the production run disqualified it. Reasons: the end to end runs are runs of the production prompt, and under 1 shared, lenient they joined honest reviews (Harbor Light 10 and 16) in both runs. The 190 pair control cannot bound a judge's wrong join rate at the volume 1 shared claim produces: 18% of honest pairs become candidates, about 22,650 judgments per 500 reviews. The judge's verdict on the same pair moved with the property line in the prompt, so the only setting the control can speak for is the one with the smallest exposure: 2 shared claims, 1.3% of honest pairs, fourteen times fewer judgments. The recall given up here is two to three of 18 synthetic pairs, on a set whose figures moved by 0.05 to 0.11 between ablations from extraction variance alone. The default is measured with corrected prompts in claim_duplicates_default.md. No further tuning of this signal in step 5.

## Wrong joins

None in any cell or run.

## Run to run agreement (paraphrase set joins)

- 2 shared, strict: joined in every run 6 (C1-C2, C1-C3, C1-C4, C2-C3, C2-C4, C3-C4); run 1 only: none; run 2 only: none.
- 2 shared, lenient: joined in every run 6 (C1-C2, C1-C3, C1-C4, C2-C3, C2-C4, C3-C4); run 1 only: none; run 2 only: none.
- 1 shared, strict: joined in every run 7 (B2-B4, C1-C2, C1-C3, C1-C4, C2-C3, C2-C4, C3-C4); run 1 only: none; run 2 only: B1-B2, B1-B4.
- 1 shared, lenient: joined in every run 8 (B1-B2, B1-B4, C1-C2, C1-C3, C1-C4, C2-C3, C2-C4, C3-C4); run 1 only: B2-B4; run 2 only: none.
- Candidates at each rule, every run / one run only: 2 shared: paraphrase 6 / 1, control 2 / 1; 1 shared: paraphrase 33 / 3, control 32 / 5.

## Cost (measured tokens)

| Run | Stage | Model | Calls | Input tokens | Output tokens |
|---|---|---|---|---|---|
| 1 | paraphrase extraction | claude-sonnet-5-5 | 24 | 27,798 | 3,079 |
| 1 | paraphrase judge strict | claude-haiku-5-5 | 33 | 26,720 | 3,077 |
| 1 | paraphrase judge lenient | claude-haiku-5-5 | 33 | 28,535 | 3,060 |
| 1 | control extraction | claude-sonnet-5-5 | 20 | 23,137 | 2,895 |
| 1 | control judge strict | claude-haiku-5-5 | 36 | 28,635 | 3,298 |
| 1 | control judge lenient | claude-haiku-5-5 | 36 | 30,615 | 3,350 |
| 2 | paraphrase extraction | claude-sonnet-5-5 | 24 | 27,798 | 3,116 |
| 2 | paraphrase judge strict | claude-haiku-5-5 | 36 | 29,293 | 3,260 |
| 2 | paraphrase judge lenient | claude-haiku-5-5 | 36 | 31,273 | 3,343 |
| 2 | control extraction | claude-sonnet-5-5 | 20 | 23,137 | 2,917 |
| 2 | control judge strict | claude-haiku-5-5 | 33 | 26,368 | 3,063 |
| 2 | control judge lenient | claude-haiku-5-5 | 33 | 28,183 | 3,050 |
| all | all | claude-sonnet-5-5 | 88 | 101,870 | 12,007 |
| all | all | claude-haiku-5-5 | 276 | 229,622 | 25,501 |

Output tokens include thinking tokens.
Measured total for the whole ablation: $0.36 at list prices per million tokens as of 2026-10-06 (claude-sonnet-5-5 $2.00 input, $10.00 output; claude-haiku-5-5 $0.10 input, $0.50 output).

## How judgments scale with batch size

Pairs grow with the square of the batch, and every candidate pair costs one small model judgment, so the candidate rule sets the cost on a real batch. The projection uses the Harbor Light candidate share: honest reviews of one property, the realistic case. The paraphrase share is shown for comparison.

| Rule | Harbor Light share | Paraphrase share | Judgments at 25 reviews | Judgments at 100 reviews | Judgments at 500 reviews |
|---|---|---|---|---|---|
| 2 shared | 0.013 | 0.024 | 4 ($0.00) | 65 ($0.01) | 1,641 ($0.21) |
| 1 shared | 0.182 | 0.125 | 54 ($0.01) | 899 ($0.12) | 22,652 ($2.93) |

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

- The setting is chosen on the same reviews it is reported on, both sets. There is no held out set, so the chosen cell's figures are optimistic for new data.
- Any future held out set must include honest reviews of one property that mention the same features. Negatives on different topics, as in the first version of this ablation, cannot show the failure that matters most.
- The lenient prompt's examples of different names for one feature (a deck, a terrace, a balcony; a massage, a spa treatment, a wellness centre) are the vocabulary of cluster C in the paraphrase set, which favours the lenient prompt on recall.
- The control is one fictional property with 20 short reviews written by the author of the detector.
- Two runs show whether answers move between runs; they are not a distribution.
- The claim signal is measured on its own. In the graph it runs alongside the detector, and pairs the detector already joined are not judged again.
- The fictional property name in the prompts is a placeholder; the set names none.
- Cost is the token count the API reported; the dollar figure is an estimate at list prices on the stated date.
- The pair that motivated the control, genuine reviews 10 and 16, was a candidate in both runs at 1 shared claim and was judged not the same by both prompts in both runs. In the Harbor Light end to end run (commit 8173b18) the lenient prompt judged the same pair the same, on the shared location claim. The judge is not stable on this pair: one 'same' in five judgments across the two reports. Zero wrong joins here does not show that failure is gone, only that it is not frequent enough to appear in two runs.
- The control prompts named the property with the evaluation placeholder ('Lakeshore hotel (fictional)'); the end to end run named it 'Harbor Light Inn (inn, fictional, Northern Michigan)'. Whether that difference changes the judgments was not measured.
