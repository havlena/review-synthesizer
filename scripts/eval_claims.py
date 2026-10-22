#!/usr/bin/env python
"""Live evaluation of the claim duplicate signal on the labelled paraphrase set.

    python scripts/eval_claims.py                               # current default, two runs
    python scripts/eval_claims.py --min-shared 1 --judge lenient   # one other setting
    python scripts/eval_claims.py --min-shared 2 1 --judge strict lenient   # the ablation

Needs ANTHROPIC_API_KEY (read from .env) and the network; it spends API credit. Model names
come from ANTHROPIC_MODEL and ANTHROPIC_MODEL_SMALL, with the defaults in graph/config.py.

--min-shared (1 or 2) is the candidate rule: specific claims two reviews must share by
aspect and polarity. --judge (strict or lenient) picks the judge prompt. With one value
each, every run extracts and judges for that one setting and the report is
claim_duplicates (or --name). With several, the script runs the ablation on two sets: the
paraphrase set for recall, and the genuine Harbor Light reviews (--control) as honest pairs
of one property where every join is wrong. Each run extracts once per set, saves the
extractions under data/interim/claims/ (git ignored), judges the union of candidates with
every prompt, and scores every cell on that same extraction; the report is
claim_duplicates_ablation (or --name).

Inputs: tests/fixtures/paraphrase_set.json (or --set); tests/fixtures/harbor_light_inn.json
(or --control), reviews whose intent is genuine.
Outputs: evals/results/<name>.md and .json (or --out-dir), written only if every run
completed. If any extraction or judgment call fails, the script prints the failures, writes
no report and exits with status 2.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path

from dotenv import load_dotenv

from review_synthesizer.authenticity import Review
from review_synthesizer.evals import claims, claims_ablation
from review_synthesizer.evals.similarity import LabelledReview, load_paraphrase_set
from review_synthesizer.graph import AnthropicModelClient, ModelConfig
from review_synthesizer.graph.config import MAX_PARALLEL_CALLS, MAX_TOKENS
from review_synthesizer.graph.duplicates import JUDGE_STYLE, MIN_SHARED_SPECIFIC

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

ABLATION_LIMITS = [
    (
        "The setting is chosen on the same reviews it is reported on, both sets. There is no "
        "held out set, so the chosen cell's figures are optimistic for new data."
    ),
    (
        "Any future held out set must include honest reviews of one property that mention the "
        "same features. Negatives on different topics, as in the first version of this "
        "ablation, cannot show the failure that matters most."
    ),
    (
        "The lenient prompt's examples of different names for one feature (a deck, a terrace, "
        "a balcony; a massage, a spa treatment, a wellness centre) are the vocabulary of "
        "cluster C in the paraphrase set, which favours the lenient prompt on recall."
    ),
    (
        "The control is one fictional property with 20 short reviews written by the author of "
        "the detector."
    ),
    *LIMITS[1:],
]

ABLATION_HISTORY = [
    (
        "First version (2026-10-22, commit 61ab971): paraphrase set only. Its negatives were "
        "in sample and on different topics. Mean paraphrase recall 0.222 (2 shared, strict), "
        "0.222 (2 shared, lenient), 0.389 (1 shared, strict), 0.417 (1 shared, lenient); no "
        "wrong join in any cell. The rule then (no wrong join, highest recall) chose 1 shared "
        "claim with the lenient judge, and it became the product default."
    ),
    (
        "The Harbor Light end to end run at that default (commit 8173b18) joined genuine "
        "reviews 10 and 16 on one shared claim, the waterfront location. The pair had been "
        "hidden in earlier runs because embedding similarity joined it first and the graph "
        "does not judge pairs the detector already joined."
    ),
    (
        "This version adds the 20 genuine Harbor Light reviews as a control set of honest "
        "pairs of one property, every join wrong, and a rule that disqualifies any cell with a "
        "wrong join on either set."
    ),
]


def harbor_control(path: Path) -> list[LabelledReview]:
    """The genuine Harbor Light reviews (the fixture's intent labels), as a negatives only set."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        LabelledReview(Review.from_dict(r), None)
        for r in data["reviews"]
        if r.get("intent") == "genuine"
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


def write(report, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    md, js = out_dir / f"{name}.md", out_dir / f"{name}.json"
    md.write_text(report.to_markdown(), encoding="utf-8", newline="\n")
    js.write_text(report.to_json() + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {md.relative_to(ROOT)} and {js.relative_to(ROOT)}")


def print_tokens(tokens: dict) -> None:
    for model, u in tokens["total"].items():
        print(f"{model}: {u['calls']} calls, {u['input_tokens']:,} in, {u['output_tokens']:,} out")
    if tokens["usd"] is not None:
        print(f"estimated ${tokens['usd']:.2f} at list prices as of {claims.PRICES_AS_OF}")


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

    rows, clusters = load_paraphrase_set(args.set)
    config = ModelConfig.from_env()
    min_shared = list(dict.fromkeys(args.min_shared))
    judges = list(dict.fromkeys(args.judge))
    grid = len(min_shared) * len(judges) > 1
    print(
        f"{len(rows)} reviews, {args.runs} run(s), models {config.main} / {config.small}, "
        f"min shared {min_shared}, judge {judges}"
    )

    try:
        if grid:
            report = claims_ablation.evaluate_grid(
                rows,
                harbor_control(args.control),
                lambda: AnthropicModelClient(config),
                runs=args.runs,
                min_shared=min_shared,
                judges=judges,
                save_dir=args.extraction_dir,
                settings=settings(config),
                history=ABLATION_HISTORY,
                limits=ABLATION_LIMITS,
            )
        else:
            report = claims.evaluate(
                rows,
                clusters,
                lambda: AnthropicModelClient(config),
                runs=args.runs,
                min_shared=min_shared[0],
                judge=judges[0],
                settings=settings(config),
                limits=LIMITS,
            )
    except claims.CallFailure as err:
        print(f"Stopped: {len(err.failures)} call(s) failed during {err.stage}. No report written.")
        for f in err.failures:
            print(f"  - {f}")
        return 2

    if grid:
        # Record the saved extractions relative to the repository, not the local disk.
        report.extraction_files = [
            Path(os.path.relpath(p, ROOT)).as_posix() for p in report.extraction_files
        ]
        for name in report.cells:
            for r in report.runs:
                c, k = r["cells"][name], r["control_cells"][name]
                print(
                    f"{name:<20} run {r['run']}: candidates {c['candidates']['true']} true, "
                    f"{c['candidates']['false']} non duplicate; judged same "
                    f"{c['judged']['true_positive']} true, {c['judged']['false_positive']} wrong; "
                    f"recall {c['judged']['recall']:.3f}; control {k['candidates']} candidates, "
                    f"{len(k['wrong_joins'])} wrong"
                )
        print(f"decision: {report.decision['chosen']}. {report.decision['reason']}")
    else:
        for r in report.runs:
            c, j = r["candidates"], r["judged"]
            print(
                f"run {r['run']}: candidates {c['true']}/{report.truth_pairs} true, "
                f"{c['false']} false; judged same {j['true_positive']} true, "
                f"{j['false_positive']} false; precision {j['precision']:.3f}, "
                f"recall {j['recall']:.3f}"
            )
    print_tokens(report.tokens)
    write(
        report,
        args.out_dir,
        args.name or ("claim_duplicates_ablation" if grid else "claim_duplicates"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
