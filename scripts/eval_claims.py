#!/usr/bin/env python
"""Live evaluation of the claim duplicate signal on the labelled paraphrase set.

    python scripts/eval_claims.py                 # two runs, default models from .env
    python scripts/eval_claims.py --runs 3

Needs ANTHROPIC_API_KEY (read from .env) and the network; it spends API credit. Model names
come from ANTHROPIC_MODEL and ANTHROPIC_MODEL_SMALL, with the defaults in graph/config.py.

Inputs: tests/fixtures/paraphrase_set.json (or --set).
Outputs: evals/results/claim_duplicates.md and .json (or --out-dir, --name), written only
if every run completed. If any extraction or judgment call fails, the script prints the
failures, writes nothing and exits with status 2.
"""

from __future__ import annotations

import argparse
import os
import sys
from importlib.metadata import version
from pathlib import Path

from dotenv import load_dotenv

from review_synthesizer.evals import claims
from review_synthesizer.evals.similarity import load_paraphrase_set
from review_synthesizer.graph import AnthropicModelClient, ModelConfig
from review_synthesizer.graph.config import MAX_PARALLEL_CALLS, MAX_TOKENS

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_EFFORT = {"claude-sonnet-5-5": "high", "claude-haiku-5-5": "medium"}
"""API default effort per model, from Anthropic's model documentation; not sent by us."""

LIMITS = [
    (
        "Twenty four synthetic reviews written for this project by the author of the detector. "
        "The figures show the mechanism and where it loses pairs, not field accuracy."
    ),
    "Two runs show whether answers move between runs; they are not a distribution.",
    (
        "The claim signal is measured on its own. In the graph it runs alongside the detector, "
        "and pairs the detector already joined are not judged again."
    ),
    "The fictional property name in the prompts is a placeholder; the set names none.",
    (
        "Cost is the token count the API reported; the dollar figure is an estimate at list "
        "prices on the stated date."
    ),
]


def settings(config: ModelConfig) -> dict[str, str]:
    effort = ", ".join(
        f"{m} {DEFAULT_EFFORT[m]}" for m in (config.main, config.small) if m in DEFAULT_EFFORT
    )
    return {
        "Effort": "not set; the API default for each model"
        + (f" ({effort})" if effort else "")
        + ".",
        "Thinking": "not set; the API default (adaptive) for each model.",
        "Temperature": (
            "not sent. The Messages API has rejected temperature, top_p and top_k with an "
            "error on every model since 2026-10-21, so run to run stability is measured by "
            "repeating the run, not by fixing sampling."
        ),
        "Structured output": "native JSON schema output (langchain-anthropic, method json_schema).",
        "Max output tokens per call": f"{MAX_TOKENS:,}.",
        "Concurrent calls per stage": f"{MAX_PARALLEL_CALLS}.",
        "Packages": (
            f"langchain-anthropic {version('langchain-anthropic')}, "
            f"anthropic {version('anthropic')}, langgraph {version('langgraph')}."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", type=Path, default=ROOT / "tests/fixtures/paraphrase_set.json")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "evals/results")
    parser.add_argument("--name", default="claim_duplicates")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set (looked in the environment and .env).")
        return 1

    rows, clusters = load_paraphrase_set(args.set)
    config = ModelConfig.from_env()
    print(f"{len(rows)} reviews, {args.runs} run(s), models {config.main} / {config.small}")

    try:
        report = claims.evaluate(
            rows,
            clusters,
            lambda: AnthropicModelClient(config),
            runs=args.runs,
            settings=settings(config),
            limits=LIMITS,
        )
    except claims.CallFailure as err:
        print(f"Stopped: {len(err.failures)} call(s) failed during {err.stage}. Nothing written.")
        for f in err.failures:
            print(f"  - {f}")
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    md = args.out_dir / f"{args.name}.md"
    js = args.out_dir / f"{args.name}.json"
    md.write_text(report.to_markdown(), encoding="utf-8")
    js.write_text(report.to_json() + "\n", encoding="utf-8")

    for r in report.runs:
        c, j = r["candidates"], r["judged"]
        print(
            f"run {r['run']}: candidates {c['true']}/{report.truth_pairs} true, {c['false']} false; "
            f"judged same {j['true_positive']} true, {j['false_positive']} false; "
            f"precision {j['precision']:.3f}, recall {j['recall']:.3f}"
        )
    for model, u in report.tokens["total"].items():
        print(f"{model}: {u['calls']} calls, {u['input_tokens']:,} in, {u['output_tokens']:,} out")
    if report.tokens["usd"] is not None:
        print(f"estimated ${report.tokens['usd']:.2f} at list prices as of {claims.PRICES_AS_OF}")
    print(f"wrote {md.relative_to(ROOT)} and {js.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
