#!/usr/bin/env python
"""Run the whole graph on the Harbor Light Inn fixture, live, with every local signal.

    python scripts/run_graph.py              # two runs
    python scripts/run_graph.py --runs 1

Signals: the rules, the embedder at its default model, the Ott classifier from
models/ott_deception.joblib when that file exists, and claim duplicates at the current
defaults. Every flag is confirmed automatically (``approve_all``), and the brief's methods
note says so: this is an evaluation run against the golden result, not an operator review.

Needs ANTHROPIC_API_KEY (read from .env) and the network; it spends API credit. The
embedding model is downloaded into the Hugging Face cache on first use.

Inputs: tests/fixtures/harbor_light_inn.json (or --fixture).
Outputs, under evals/results/ (or --out-dir):
  brief_harbor_light.md and .json          the brief of run 1
  run_graph_harbor_light.md and .json      the run report over all runs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from review_synthesizer.authenticity import SentenceTransformerEmbedder
from review_synthesizer.evals.claims import usage_of
from review_synthesizer.evals.graph_run import summarize
from review_synthesizer.graph import (
    AnthropicModelClient,
    ModelConfig,
    Property,
    approve_all,
    run_brief,
)
from review_synthesizer.graph.run import AUTO_APPROVED

ROOT = Path(__file__).resolve().parents[1]

HARBOR = Property(name="Harbor Light Inn", type="inn", region="fictional, Northern Michigan")

LIMITS = [
    (
        "One fictional property with 25 synthetic reviews, five of them a planted campaign. "
        "The run checks whether the graph reproduces the capstone result with every signal on; it "
        "says nothing about accuracy on real batches."
    ),
    (
        "Every flag was confirmed automatically. In the product a person decides each flag; "
        "here the golden result assumes all planted reviews are set aside."
    ),
    "Two runs show what moves between runs; they are not a distribution.",
    (
        "Cost is the token count the API reported; the dollar figure is an estimate at list "
        "prices on the stated date."
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--fixture", type=Path, default=ROOT / "tests/fixtures/harbor_light_inn.json"
    )
    parser.add_argument("--classifier", type=Path, default=ROOT / "models/ott_deception.joblib")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "evals/results")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set (looked in the environment and .env).")
        return 1

    rows = json.loads(args.fixture.read_text(encoding="utf-8"))["reviews"]
    embedder = SentenceTransformerEmbedder()
    classifier = None
    if args.classifier.exists():
        from review_synthesizer.authenticity.classifier import DeceptionClassifier

        classifier = DeceptionClassifier.load(args.classifier)
    else:
        print(f"{args.classifier.relative_to(ROOT)} not found; running without the classifier.")
    config = ModelConfig.from_env()
    print(
        f"{len(rows)} reviews, {args.runs} run(s), models {config.main} / {config.small}, "
        f"embedder {embedder.name}, classifier {'on' if classifier else 'off'}"
    )

    briefs, usages = [], []
    for k in range(1, args.runs + 1):
        client = AnthropicModelClient(config)
        brief = run_brief(
            rows,
            HARBOR,
            approve_all,
            client=client,
            embedder=embedder,
            classifier=classifier,
            run_id=f"harbor-light-run{k}",
            approved_by=AUTO_APPROVED,
        )
        briefs.append(brief)
        usages.append(usage_of(client))
        r = brief.data["rating"]
        print(
            f"run {k}: displayed {r['displayed']}, verified {r['verified']}, "
            f"{len(brief.data['authenticity']['flags'])} flags"
        )

    report = summarize(rows, [b.data for b in briefs], usages, limits=LIMITS)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "brief_harbor_light.md": briefs[0].markdown,
        "brief_harbor_light.json": json.dumps(briefs[0].data, indent=2) + "\n",
        "run_graph_harbor_light.md": report.to_markdown(),
        "run_graph_harbor_light.json": report.to_json() + "\n",
    }
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8", newline="\n")

    h = report.headline
    print(
        f"verified {h['verified_ratings']} (golden 3.5), displayed {h['displayed_ratings']} (golden 3.8)"
    )
    print(f"genuine reviews flagged: {h['genuine_reviews_flagged'] or 'none'}")
    for model, u in report.tokens["total"].items():
        print(f"{model}: {u['calls']} calls, {u['input_tokens']:,} in, {u['output_tokens']:,} out")
    if report.tokens["usd"] is not None:
        print(f"estimated ${report.tokens['usd']:.2f}")
    print("wrote " + ", ".join(f"{out.relative_to(ROOT).as_posix()}/{n}" for n in files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
