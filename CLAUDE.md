# Review Synthesizer: working rules for this repository

Read `README.md`, `docs/build-plan.md` and `docs/evaluation.md` before changing anything.
The build plan says what comes next and why; the evaluation doc says how results are
measured and reported.

## What this is

An operator brief from guest reviews for independent lodging, with an authenticity layer
that flags coordinated review manipulation. Started as an M.S. capstone (ASU, CIS 576,
summer 2026); v2 is a production style rebuild. Steps 1 to 4 are done: the detector
(`src/review_synthesizer/authenticity/`) runs four signals through one `detect()` call,
each measured, each flag carrying a score, a band and a written reason. Step 5 onward is
the LangGraph graph, the API, Docker, tracing, CI, deployment and the operator UI.

## Environment

- Python 3.12, virtual environment at `.venv`, src layout, `pip install -e .[dev,ml,embeddings]`.
- Run tests with `python -m pytest` (or `.venv\Scripts\python -m pytest` from PowerShell).
  Lint with `ruff check src tests scripts`.
- Secrets live in `.env` only (copy from `.env.example`). Never commit a key, and never
  print one. Load with `python-dotenv` or read the environment.
- Large files stay out of git: `data/`, `models/`, downloaded corpora and model weights
  are ignored. Only `evals/results/` reports are committed.
- Windows host. Prefer cross platform paths (`pathlib`), LF line endings
  (`.gitattributes` enforces them), and commands that work in PowerShell and bash.

## Non negotiable rules

1. The golden tests must stay green. `tests/test_rules.py` and `tests/test_detector.py`
   replay the capstone demo (five planted reviews flagged, no genuine review flagged, 3.8
   displayed to 3.5 verified). Changing a threshold or a rule without that result holding is
   a bug unless the change is deliberate and documented.
2. Every flag is a signal for human review, never a verdict. Nothing in this system accuses
   a reviewer, posts anything, or takes a review down. The graph has a human approval step
   before a brief is final; keep it.
3. Every new capability gets a measured number before it is wired into the product, and
   the number is committed under `evals/results/` with its limits stated next to it. A
   negative result gets the same treatment as a positive one (see the embedding similarity
   result in `docs/evaluation.md` for the tone).
4. No real reviews, reviewers, platforms or businesses in fixtures, tests, docs or demos.
   Fictional property names only (Harbor Light Inn is the house example). The Ott corpus is
   used under its terms and never redistributed (`docs/data.md`).
5. Tests run without network and without API keys. Anything that calls a model gets a fake
   in tests (a stub that returns fixed structured output) and a separate live evaluation
   script, named as such, for real runs.
6. Keep dependencies optional where they are heavy: the detector imports scikit-learn and
   sentence-transformers on first use; the rule based path needs nothing. Follow that pattern.

## Code conventions

- Dataclasses for records, type hints everywhere, docstrings that say what a module is for
  and what it cannot do. Functions that compute return values; printing happens in `scripts/`.
- Thresholds and model names are module level constants with a one line docstring saying
  where the value came from (an eval result, a capstone decision).
- Rationale strings are written for an operator, not a developer: what was found, the
  number, the threshold, in one or two sentences.
- Scripts in `scripts/` take arguments with `argparse`, document their inputs and outputs
  in the module docstring, and write reports as both `.md` (for people) and `.json` (for
  machines) under `evals/results/`.

## Writing conventions for docs and commit messages

- Plain language, measured claims, numbers with their protocol. No marketing words.
- Commit subjects: `Step N: what changed`, imperative, under 72 characters. Body optional.
  Commits made with Claude Code carry one trailer only, naming the model that did the work,
  for example `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. No
  `Claude-Session` line (commits before step 5 have one; that convention is retired).
- Do not commit on the author's behalf without being asked; stage, show the diff summary,
  and let the author run the commit, unless told otherwise in the session.

## Accounts and services (set up by the author, one at a time)

Separate account per service, each with its own email and password login. No "sign in
with GitHub" or other third party sign ins. Existing: GitHub (`havlena`), Anthropic Console
(API key in `.env`). Coming with their steps: LangSmith (step 7), a cloud host (step 8), a
new Supabase org for the production database (step 8).
