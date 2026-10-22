# Review Synthesizer

Turns a batch of guest reviews for an independent lodging property into an operator brief: ranked themes with sentiment, an aspect scorecard, a prioritized action list, and an authenticity layer that flags coordinated review manipulation and computes a verified rating from the reviews that pass.

Started as the capstone for the M.S. in Artificial Intelligence in Business at Arizona State University (CIS 576, summer 2026). This repository carries that prototype forward into a production style build.

## Status

| Component | Capstone (v1) | v2 (this repository) |
|---|---|---|
| Detector | Rule based, in browser JavaScript: Jaccard near duplicate text at 0.55, rating bursts of 3 same rating reviews within 4 days | Python package with tests, sentence embeddings for paraphrased duplicates, a classifier trained on labeled hotel reviews, and a stated reason on every flag |
| Synthesis | Keyword rules in the demo; the LLM version lived in a Google AI Studio system prompt | Claude API inside a LangGraph graph with a human approval step |
| Data | 25 synthetic reviews for a fictional property | CSV import, Postgres with pgvector, public hotel review datasets for development and evaluation |
| Backend | Designed in Module 4 (relational schema with row level security), never wired to the UI | FastAPI service, Docker Compose, MCP server for the detector |
| Evaluation | Six structured test cases | Labeled golden set, LangSmith tracing, tests and evals in CI |

The v1 artifacts are kept unchanged under `prototype/`.

## Layout

```
src/review_synthesizer/authenticity/   the detector: rules, embedding similarity, classifier, one detect() call
src/review_synthesizer/data/           fetch and normalise public evaluation data (step 4a)
src/review_synthesizer/evals/          evaluation harnesses; results are committed under evals/results/ (step 4b)
scripts/                                command line entry points
tests/                                  golden tests on the capstone data set
prototype/                              v1 single file prototype and the AI Studio system prompt
db/                                     Postgres schema (Module 4 base) and v2 migrations
docs/                                   design notes; docs/data.md lists every data source and its terms
```

## Run the tests

```
python -m venv .venv
.venv\Scripts\activate        (Windows)   |   source .venv/bin/activate   (macOS, Linux)
pip install -e .[dev,ml]
pytest
```

The golden tests replay the capstone demo: the five planted reviews are flagged, none of the twenty genuine ones are, and the rating moves from 3.8 displayed to 3.5 verified.

## Detector signals

| Signal | What it catches | How it is measured |
|---|---|---|
| `duplicate_phrasing` | Copy and paste with light edits: Jaccard word overlap at or above 0.55 | Golden tests on the capstone data set |
| `rating_burst` | Three or more same rating reviews within four days; separate bursts reported separately | Golden tests |
| `embedding_similarity` | Light rewrites: cosine similarity of sentence embeddings at or above 0.85 (paraphrase-mpnet-base-v2, runs locally). Measured limit: heavy paraphrase is not separable from honest reviews on the same topic by any of three models tested, and at 0.85 none of it is joined | `evals/results/paraphrase_similarity.md` |
| `deceptive_language` | Language that resembles reviews written without a stay; flagged only at p ≥ 0.8 | 99.1% precision, 42% recall on the Ott corpus, `evals/results/ott_classifier.md` |

Every flag carries a score, a confidence band and a written reason, and is a signal for human review rather than a verdict. Try it on the capstone data set:

```
python scripts/detect.py tests/fixtures/harbor_light_inn.json
python scripts/detect.py tests/fixtures/paraphrase_set.json --embeddings --classifier models/ott_deception.joblib
```

## Evaluation

The deceptive language classifier is scored on the Ott corpus with the published, hotel disjoint folds; the current figures are in `evals/results/ott_classifier.md` and the protocol in `docs/evaluation.md`. Reproduce them with:

```
python scripts/fetch_ott_corpus.py
python scripts/eval_ott.py
```

The duplicate signals are compared on a labelled paraphrase set across three embedding models with `python scripts/eval_similarity.py` (needs `pip install -e .[embeddings]`); the result is a measured limit as much as a measured capability. The LangGraph graph and the API follow as the build proceeds.

## Notes

- The demo property, Harbor Light Inn, is fictional. No real business or reviewer data was used in the prototype.
- Authenticity output is a probabilistic signal for human review, not a verdict. Nothing in this system accuses a reviewer or triggers a takedown.
- Secrets are never committed. Copy `.env.example` to `.env` locally.
