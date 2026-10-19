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
src/review_synthesizer/authenticity/   rule based signals (v1 port, step 3)
tests/                                  golden tests on the capstone data set
prototype/                              v1 single file prototype and the AI Studio system prompt
db/                                     Postgres schema (Module 4 base, extended in v2)
docs/                                   design notes and capstone materials
```

## Run the tests

```
python -m venv .venv
.venv\Scripts\activate        (Windows)   |   source .venv/bin/activate   (macOS, Linux)
pip install -e .[dev]
pytest
```

The golden tests replay the capstone demo: the five planted reviews are flagged, none of the twenty genuine ones are, and the rating moves from 3.8 displayed to 3.5 verified.

Embeddings, the classifier, the LangGraph graph, the API and the evaluation harness follow as the build proceeds.

## Notes

- The demo property, Harbor Light Inn, is fictional. No real business or reviewer data was used in the prototype.
- Authenticity output is a probabilistic signal for human review, not a verdict. Nothing in this system accuses a reviewer or triggers a takedown.
- Secrets are never committed. Copy `.env.example` to `.env` locally.
