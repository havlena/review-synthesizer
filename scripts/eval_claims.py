#!/usr/bin/env python
"""Live evaluation of the claim duplicate signal: recall set plus same property control.

    python scripts/eval_claims.py                                          # product default
    python scripts/eval_claims.py --min-shared 2 1 --judge strict lenient  # the ablation

Needs ANTHROPIC_API_KEY (read from .env) and the network; it spends API credit. Model names
come from ANTHROPIC_MODEL and ANTHROPIC_MODEL_SMALL, with the defaults in graph/config.py.

Two sets, always: the paraphrase set for recall (18 true pairs), and the genuine Harbor Light
reviews (--control) as honest pairs of one property, where every join is wrong. Each set's
prompts carry its own fictional property line, built by the graph's own prompt builders, so
the model sees what it sees in the product (evals/fixtures.py).

--min-shared (1 or 2) is the candidate rule: specific claims two reviews must share by aspect
and polarity. --judge (strict or lenient) picks the judge prompt. Each run extracts once per
set, saves the extractions under data/interim/claims/ (git ignored), judges the union of
candidates with every prompt, and scores every setting on that same extraction. With one
value each the report is claim_duplicates_default; with several it is the ablation,
claim_duplicates_ablation (or --name).

Inputs: tests/fixtures/paraphrase_set.json (or --set); tests/fixtures/harbor_light_inn.json
(or --control), reviews whose intent is genuine.
Outputs: evals/results/<name>.md and .json (or --out-dir), written only if every run
completed. If any extraction or judgment call fails, the script prints the failures, writes
no report and exits with status 2.
"""

from __future__ import annotations

import argparse
import os
import sys
from importlib.metadata import version
from pathlib import Path

from dotenv import load_dotenv

from review_synthesizer.evals import claims, claims_ablation
from review_synthesizer.evals.fixtures import HARBOR_PROPERTY, PARAPHRASE_PROPERTY, harbor_control
from review_synthesizer.evals.similarity import load_paraphrase_set
from review_synthesizer.graph import AnthropicModelClient, ModelConfig
from review_synthesizer.graph.config import MAX_PARALLEL_CALLS, MAX_TOKENS
from review_synthesizer.graph.duplicates import JUDGE_STYLE, MIN_SHARED_SPECIFIC

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_EFFORT = {"claude-sonnet-5-5": "high", "claude-haiku-5-5": "medium"}
"""API default effort per model, from Anthropic's model documentation; not sent by us."""

LIMITS = [
    (
        "The sets are small, synthetic and written by the author of the detector: 24 reviews "
        "for recall, 20 genuine reviews of one fictional property as the control."
    ),
    (
        "Any held out set must include honest reviews of one property that mention the same "
        "features. Negatives on different topics cannot show the failure that matters most."
    ),
    "Two runs show whether answers move between runs; they are not a distribution.",
    (
        "The claim signal is measured on its own. In the graph it runs alongside the detector, "
        "and pairs the detector already joined are not judged again."
    ),
    (
        "Cost is the token count the API reported; the dollar figure is an estimate at list "
        "prices on the stated date."
    ),
]

ABLATION_LIMITS = [
    (
        "The setting is chosen on the same reviews it is reported on, both sets. There is no "
        "held out set, so the chosen cell's figures are optimistic for new data."
    ),
    (
        "The lenient prompt's examples of different names for one feature (a deck, a terrace, "
        "a balcony; a massage, a spa treatment, a wellness centre) are the vocabulary of "
        "cluster C in the paraphrase set, which favours the lenient prompt on recall."
    ),
    *LIMITS,
]

DEFAULT_HISTORY = [
    (
        "The product default, 2 shared claims with the strict judge, is an author decision "
        "that overrides the ablation's recall rule; the reasons are in "
        "claim_duplicates_ablation.md and docs/evaluation.md."
    ),
    (
        "This run uses the corrected prompts: until this report both sets were sent with a "
        "placeholder property line, while the graph sends the real one."
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
    parser.add_argument(
        "--control", type=Path, default=ROOT / "tests/fixtures/harbor_light_inn.json"
    )
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument(
        "--min-shared", type=int, nargs="+", choices=[1, 2], default=[MIN_SHARED_SPECIFIC]
    )
    parser.add_argument("--judge", nargs="+", choices=["strict", "lenient"], default=[JUDGE_STYLE])
    parser.add_argument("--out-dir", type=Path, default=ROOT / "evals/results")
    parser.add_argument("--extraction-dir", type=Path, default=ROOT / "data/interim/claims")
    parser.add_argument("--name")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set (looked in the environment and .env).")
        return 1

    rows, _ = load_paraphrase_set(args.set)
    control = harbor_control(args.control)
    config = ModelConfig.from_env()
    min_shared = list(dict.fromkeys(args.min_shared))
    judges = list(dict.fromkeys(args.judge))
    grid = len(min_shared) * len(judges) > 1
    name = args.name or ("claim_duplicates_ablation" if grid else "claim_duplicates_default")
    print(
        f"{len(rows)} + {len(control)} reviews, {args.runs} run(s), models {config.main} / "
        f"{config.small}, min shared {min_shared}, judge {judges}"
    )

    try:
        report = claims_ablation.evaluate_grid(
            rows,
            control,
            lambda: AnthropicModelClient(config),
            runs=args.runs,
            min_shared=min_shared,
            judges=judges,
            save_dir=args.extraction_dir / name,
            prop=PARAPHRASE_PROPERTY,
            control_prop=HARBOR_PROPERTY,
            settings=settings(config),
            history=[] if grid else DEFAULT_HISTORY,
            limits=ABLATION_LIMITS if grid else LIMITS,
        )
    except claims.CallFailure as err:
        print(f"Stopped: {len(err.failures)} call(s) failed during {err.stage}. No report written.")
        for f in err.failures:
            print(f"  - {f}")
        return 2

    # Record the saved extractions relative to the repository, not the local disk.
    report.extraction_files = [
        Path(os.path.relpath(p, ROOT)).as_posix() for p in report.extraction_files
    ]
    for cell in report.cells:
        for r in report.runs:
            c, k = r["cells"][cell], r["control_cells"][cell]
            print(
                f"{cell:<20} run {r['run']}: candidates {c['candidates']['true']} true, "
                f"{c['candidates']['false']} non duplicate; judged same "
                f"{c['judged']['true_positive']} true, {c['judged']['false_positive']} wrong; "
                f"recall {c['judged']['recall']:.3f}; control {k['candidates']} candidates, "
                f"{len(k['wrong_joins'])} wrong"
            )
    if grid:
        print(f"rule: {report.decision['chosen']}. {report.decision['reason']}")
    for model, u in report.tokens["total"].items():
        print(f"{model}: {u['calls']} calls, {u['input_tokens']:,} in, {u['output_tokens']:,} out")
    if report.tokens["usd"] is not None:
        print(f"estimated ${report.tokens['usd']:.2f} at list prices as of {claims.PRICES_AS_OF}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    md, js = args.out_dir / f"{name}.md", args.out_dir / f"{name}.json"
    md.write_text(report.to_markdown(), encoding="utf-8", newline="\n")
    js.write_text(report.to_json() + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {md.relative_to(ROOT).as_posix()} and {js.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
