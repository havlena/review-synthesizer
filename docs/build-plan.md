# Build plan

Nine steps. One at a time, each ending in a commit with tests green and, where a capability
is added, a measured result under `evals/results/`.

| Step | What | Status |
|---|---|---|
| 1 | Rescue the capstone schema and prototype | done |
| 2 | Repository and git | done |
| 3 | Port the detector rules to Python with golden tests | done |
| 4 | Ott corpus, deceptive language classifier, embedding similarity, composite detector, evals | done |
| 5 | LangGraph graph: claims, verified set, themes, actions, human approval | next |
| 6 | FastAPI service, Postgres with migrations, Docker Compose, MCP server for the detector | |
| 7 | LangSmith tracing, GitHub Actions CI running tests and fixture evals | |
| 8 | Deployment: container on a cloud host, production database on a new Supabase org | |
| 9 | Operator UI: upload a batch, review flags with reasons, approve, read the brief | |

## Step 5: the graph

Package `src/review_synthesizer/graph/`. Framework: LangGraph with `langchain-anthropic`.
Model names come from the environment (`ANTHROPIC_MODEL` for the main calls,
`ANTHROPIC_MODEL_SMALL` for cheap pairwise judgments), with sensible defaults in one
constants module; never hard code a model name in a node.

### Purpose

Turn a batch of reviews for one property into an operator brief with four parts (ranked
themes with sentiment, an aspect scorecard, a prioritized action list, and the
authenticity findings with a verified rating), with a person approving the authenticity
findings before the brief is final. This is the capstone's product, rebuilt on the v2
detector, with one new capability the detector cannot provide: claim level comparison of
reviews, which is what catches rewritten duplicates (see `docs/evaluation.md`, the
embedding result).

### State

One typed state object (a `TypedDict` or Pydantic model) carrying: the property (name,
type, region), the input reviews, the `DetectionResult` from the detector, extracted
claims per review, claim duplicate groups, the verified review ids, themes, aspect scores,
actions, the operator's approval decisions, and the final brief. Keep raw model text out of
the state; store parsed, validated structures.

### Nodes

1. `load_batch`: validate input (ids unique, ratings 1 to 5, dates parse, text non empty),
   normalise into `Review` objects.
2. `authenticity`: call `review_synthesizer.authenticity.detect()` with the embedder and the
   classifier when configured. Deterministic, no model call. Store the result.
3. `extract_claims`: one model call per review, structured output. Each claim has an
   `aspect` from a fixed list (cleanliness, service, breakfast, room comfort, value, location,
   noise, parking, wifi, amenities, other), a `polarity` (positive, negative, mixed), a flag
   `specific` (names a concrete thing: a view, a dish, a price, a fixture) and a short
   canonical statement. Temperature 0. Fail soft: a review whose extraction fails gets an
   empty claim list and a note, and the run continues.
4. `claim_duplicates`: find reviews that make the same specific claims in different words.
   Candidate pairs are reviews sharing at least two specific claims by aspect and polarity;
   each candidate pair gets one small model judgment ("do these two reviews make the same
   specific claims about the same things?" with a one sentence reason). Pairs judged the
   same join clusters (reuse `authenticity.clustering.union_clusters`). Each cluster becomes
   a `Flag` with `signal_type="claim_duplicate"` (add it to the database in migration
   0002), the reason, and the pair judgments in `details`. Pairs already joined by the
   detector's duplicate signals are skipped.
5. `verified_set`: review ids not covered by any flag (detector flags plus claim
   duplicates). Compute the displayed and verified ratings with
   `authenticity.rating_summary`.
6. `themes_and_scores`: one model call over the verified reviews only. Output: themes
   (name, count of supporting reviews, sentiment, two or three quoted phrases as evidence)
   and an aspect scorecard (score 1 to 5 per aspect, review count, one line of evidence).
   The system prompt is derived from `prototype/harbor_light_inn_prompt.json` (the capstone's
   anchored rubric): keep its scoring bands and its rule that only verified reviews feed
   synthesis.
7. `actions`: one model call producing a ranked action list (title, why, impact, effort,
   supporting review ids). Impact and effort use the capstone's High, Medium, Low.
8. `human_review`: a LangGraph interrupt. Present every flag with its rationale; the
   operator confirms or dismisses each one. On resume, dismissed flags release their
   reviews back into the verified set and steps 5 to 7 rerun. Nothing is final before this
   node has run.
9. `compose_brief`: assemble the brief as JSON and as Markdown (the four parts plus a
   methods note listing which signals ran, thresholds, model names and the run id).

Nodes 2 and 3 run in parallel after node 1 (fan out), joined at node 4. Node 4 depends on
both. Everything downstream of node 5 is sequential.

### Evaluation (before the graph is wired to anything)

- Unit tests with a fake model client that returns fixed structured outputs, so the graph
  runs in CI without a key. Cover: the fan out and join, the interrupt and resume, dismissed
  flags releasing reviews, and `compose_brief` on the Harbor Light Inn fixture.
- `scripts/eval_claims.py` (live, needs the key): run `extract_claims` and
  `claim_duplicates` on `tests/fixtures/paraphrase_set.json` and score the 18 true pairs and
  258 negatives the same way `evals/similarity.py` does. Report precision, recall, the
  pairs missed, the false positive pairs, model name, and cost (tokens). Write
  `evals/results/claim_duplicates.md` and `.json`. Run it twice to show run to run
  stability at temperature 0.
- `scripts/run_graph.py` (live): run the whole graph on the Harbor Light Inn fixture with
  automatic approval of every flag, save the brief to `evals/results/brief_harbor_light.md`,
  and compare the verified rating with the golden 3.5.

### Done when

Tests green, both live evals committed with their numbers and limits stated, the README's
"Agent design" section written (see below), and the graph callable from one function
`run_brief(reviews, property, approve=callback)`.

### README section to add: "Agent design"

Job application forms ask for these five things, so the README answers them in order:

- what the agent does and who it is for;
- how it is built: framework, single graph with parallel branches and a human interrupt,
  which steps are deterministic tools and which call a model, no retrieval (state only),
  with a Mermaid diagram of the graph;
- one key design decision and its trade off: deterministic detection and a human gate
  before anything is final, trading throughput for zero unreviewed output;
- how it was tested and evaluated, with links to `evals/results/`;
- demo or deployment link once step 8 exists.

## Step 6: service and storage

- FastAPI: `POST /batches` (upload reviews for a property), `GET /batches/{id}` (status,
  flags with reasons), `POST /batches/{id}/review` (operator decisions, resumes the graph),
  `GET /batches/{id}/brief`. Pydantic models mirror the graph state.
- Postgres 17 with pgvector in Docker Compose for local development. Apply `db/schema.sql`
  then `db/migrations/` in order; add migration 0002 (`claim_duplicate` signal type,
  `source`, `external_id` and `language` on reviews, an embedding column) and the row level
  security policies the capstone left out (`db/README.md`, gap 1).
- Persist batches, reviews, flags with `score` and `details`, and analyses.
- An MCP server exposing `detect` as a tool, so any agent or client can call the detector.
- `Dockerfile` for the service; `docker compose up` brings up the database and the API.

## Step 7: tracing and CI

- LangSmith tracing on every graph run (project `review-synthesizer`), with the run id in
  the brief's methods note.
- GitHub Actions: on every push, install, `ruff`, `pytest`, and the fake model graph tests.
  Live evals stay manual (they need keys and cost money).

## Step 8: deployment

- One container on a cloud host (decide then: Azure Container Apps lines up with the
  AZ-900 study plan; Cloud Run is the simplest free tier). Database on a new Supabase org
  created under the author's own email, with the schema and migrations applied and RLS
  policies in place. Secrets in the host's secret store, never in the image.
- Health check, structured logs, and a smoke test that posts the Harbor Light Inn fixture
  and reads the brief back.

## Step 9: operator UI

- A small web app (React with Vite, or plain HTML if it stays small): upload a CSV, see the
  flags with reasons, confirm or dismiss, read the brief, download it. The v1 prototype's
  look (harbor navy, teal, amber) carries over. Every number on screen links to its
  evidence.

## Standing rules

- One step at a time. Finish, measure, commit, then the next.
- Sanitise before publishing: fictional properties, no real reviewers, no employer data.
- A finished capability earns one measured line in the README; nothing is claimed before it
  is measured.
