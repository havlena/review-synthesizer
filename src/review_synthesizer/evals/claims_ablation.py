"""Ablation of the claim duplicate signal's two levers on one shared extraction per run.

Levers: the candidate rule (at least 1 or 2 shared specific claims by aspect and polarity)
and the judge prompt (strict or lenient, ``graph.prompts.JUDGE_SYSTEMS``). Extraction varies
between runs (the baseline report showed it), so each run extracts once and every cell is
scored on that same extraction: candidates at 1 shared claim include those at 2, the union
is judged once with each prompt, and a cell's joins are its own candidates judged the same
by its own prompt. Within a run the four cells differ only in the lever.

The product default is chosen by a rule fixed before the run (:func:`choose_default`): the
cell with no wrong join in either run and the highest mean judged recall; ties go to the
stricter setting; if no such cell beats the current default, the default stays.

What this cannot do: generalise. The setting is chosen on the same 24 synthetic reviews it
is reported on, and two runs are not a distribution.
"""

from __future__ import annotations

import datetime as _dt
import json
import platform
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..graph.duplicates import JUDGE_STYLE, MIN_SHARED_SPECIFIC
from ..graph.model import ModelClient
from ..graph.prompts import JUDGE_SYSTEMS
from ..graph.state import Claim, PairJudgment
from .claims import (
    PRICES_AS_OF,
    PRICES_PER_MTOK,
    Pair,
    candidates_at,
    extract_all,
    judge_pairs,
    usage_of,
)
from .similarity import LabelledReview, pair_metrics, truth_pairs

BATCH_SIZES = (25, 100, 500)
"""Batch sizes for the judgment count projection: the capstone batch, then larger ones."""

Cell = tuple[int, str]


def strictness(cell: Cell) -> tuple[int, int]:
    """Higher is stricter: more shared claims first, then the strict prompt."""
    return (cell[0], 1 if cell[1] == "strict" else 0)


def cell_name(cell: Cell) -> str:
    return f"{cell[0]} shared, {cell[1]}"


@dataclass
class AblationRun:
    claims: dict[Any, list[Claim]]
    candidates: dict[int, set[Pair]]
    judgments: dict[str, dict[Pair, PairJudgment]]
    stage_usage: dict[str, dict[str, dict[str, int]]]
    """Measured tokens per stage: extraction, then one judging stage per prompt."""


def _diff(before: dict, after: dict) -> dict[str, dict[str, int]]:
    out = {}
    for model, a in after.items():
        b = before.get(model, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
        d = {k: a[k] - b[k] for k in a}
        if d["calls"]:
            out[model] = d
    return out


def _save_claims(path: Path, claims: dict[Any, list[Claim]], model: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "saved_at": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "claims": {str(k): [asdict(c) for c in v] for k, v in claims.items()},
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def ablate_once(
    rows: Sequence[LabelledReview],
    client: ModelClient,
    min_shared: Sequence[int],
    judges: Sequence[str],
    save_to: Path | None = None,
) -> AblationRun:
    """One extraction, candidates at each rule, the union judged with each prompt."""
    before = usage_of(client)
    claims = extract_all(rows, client)
    stage_usage = {"extraction": _diff(before, usage_of(client))}
    if save_to is not None:
        _save_claims(save_to, claims, client.config.main)

    candidates = {m: candidates_at(rows, claims, m) for m in min_shared}
    union: set[Pair] = set().union(*candidates.values())
    judgments = {}
    for j in judges:
        before = usage_of(client)
        judgments[j] = judge_pairs(rows, claims, union, client, j)
        stage_usage[f"judge {j}"] = _diff(before, usage_of(client))
    return AblationRun(claims, candidates, judgments, stage_usage)


def _label(rows: Sequence[LabelledReview], p: Pair) -> str:
    return f"{rows[p[0]].review.id}-{rows[p[1]].review.id}"


def _usd(model: str, input_tokens: float, output_tokens: float) -> float | None:
    if model not in PRICES_PER_MTOK:
        return None
    p_in, p_out = PRICES_PER_MTOK[model]
    return input_tokens / 1e6 * p_in + output_tokens / 1e6 * p_out


def _per_judgment(stage: dict[str, dict[str, int]]) -> tuple[str, float, float] | None:
    """Model and mean input and output tokens per judgment in one judging stage."""
    for model, u in stage.items():
        if u["calls"]:
            return model, u["input_tokens"] / u["calls"], u["output_tokens"] / u["calls"]
    return None


def score_cell(
    rows: Sequence[LabelledReview], truth: set[Pair], run: AblationRun, cell: Cell
) -> dict[str, Any]:
    m, j = cell
    cand = run.candidates[m]
    same = {p for p in cand if run.judgments[j][p].same_claims}
    metrics = pair_metrics(same, truth, float(m))
    wrong = [
        {
            "pair": _label(rows, p),
            "clusters": f"{rows[p[0]].cluster or 'distinct'} / {rows[p[1]].cluster or 'distinct'}",
            "reason": run.judgments[j][p].reason,
        }
        for p in sorted(same - truth)
    ]
    # Estimated cost of this cell alone: the run's extraction plus this cell's judgments at
    # the measured mean tokens per judgment for its prompt. Calls ran once on the union.
    cost = 0.0
    priced = True
    for model, u in run.stage_usage["extraction"].items():
        c = _usd(model, u["input_tokens"], u["output_tokens"])
        priced &= c is not None
        cost += c or 0.0
    per = _per_judgment(run.stage_usage[f"judge {j}"])
    if per and cand:
        c = _usd(per[0], per[1] * len(cand), per[2] * len(cand))
        priced &= c is not None
        cost += c or 0.0
    return {
        "candidates": {
            "total": len(cand),
            "true": len(cand & truth),
            "false": len(cand - truth),
        },
        "judged": asdict(metrics),
        "wrong_joins": wrong,
        "missed_not_candidate": len(truth - cand),
        "missed_judged_different": len((truth & cand) - same),
        "judged_same_pairs": sorted(_label(rows, p) for p in same),
        "estimated_usd": cost if priced else None,
    }


def choose_default(
    summary: dict[str, dict[str, Any]], cells: Sequence[Cell], current: Cell
) -> dict[str, Any]:
    """The fixed rule. ``summary`` maps cell name to its wrong joins per run and mean recall."""

    def ok(c: Cell) -> bool:
        return all(n == 0 for n in summary[cell_name(c)]["wrong_joins_per_run"])

    def recall(c: Cell) -> float:
        return summary[cell_name(c)]["mean_recall"]

    eligible = [c for c in cells if ok(c)]
    best = max(eligible, key=lambda c: (recall(c), strictness(c)), default=None)
    # A current default that itself made a wrong join is beaten by any clean cell.
    current_recall = recall(current) if current in eligible else -1.0
    if best is None or best == current or recall(best) <= current_recall:
        return {
            "chosen": cell_name(current),
            "changed": False,
            "reason": (
                "No cell without a wrong join beats the current default's mean recall "
                f"({recall(current):.3f}); the default stays."
            ),
        }
    return {
        "chosen": cell_name(best),
        "changed": True,
        "reason": (
            f"No wrong join in either run and the highest mean recall ({recall(best):.3f}, "
            f"against {recall(current):.3f} for the current default)"
            + (
                "; tied cells resolved to the stricter setting."
                if sum(1 for c in eligible if recall(c) == recall(best)) > 1
                else "."
            )
        ),
        "min_shared": best[0],
        "judge": best[1],
    }


@dataclass
class AblationReport:
    created_at: str
    python_version: str
    models: dict[str, str]
    settings: dict[str, str]
    reviews: int
    truth_pairs: int
    negative_pairs: int
    cells: list[str]
    current_default: str
    runs: list[dict[str, Any]]
    summary: dict[str, dict[str, Any]]
    agreement: dict[str, dict[str, Any]]
    decision: dict[str, Any]
    tokens: dict[str, Any]
    scaling: list[dict[str, Any]]
    extraction_files: list[str]
    judge_prompts: dict[str, str]
    limits: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_markdown(self) -> str:
        t = self.truth_pairs
        lines = [
            "# Claim duplicates: ablation of the candidate rule and the judge prompt",
            "",
            (
                f"Generated {self.created_at} on Python {self.python_version}. Main model "
                f"`{self.models['main']}` (claim extraction), small model `{self.models['small']}` "
                "(pair judgments)."
            ),
            "",
            (
                f"Set: {self.reviews} synthetic reviews, {t} true pairs, {self.negative_pairs} "
                "pairs that should not be joined (tests/fixtures/paraphrase_set.json). The "
                "baseline report for the current default is claim_duplicates.md."
            ),
            "",
            "## Design",
            "",
            "Each run extracts claims once (saved under "
            + ", ".join(f"`{p}`" for p in self.extraction_files)
            + ", not committed). Candidates are taken at both rules from that extraction, "
            "the union of candidate pairs is judged once with each prompt, and every cell is "
            "scored on the same extraction and judgments, so within a run the cells differ "
            "only in the lever.",
            "",
            (
                "Decision rule, fixed before the run: the product default becomes the cell with no "
                "wrong join in either run and the highest recall (mean judged recall over the "
                "runs); ties go to the stricter setting (2 shared claims before 1, then the "
                "strict prompt); if no cell beats the current default without a wrong join, the "
                f"default stays. Current default: {self.current_default}."
            ),
            "",
            "## Grid",
            "",
            (
                f"| Cell | Run | Candidates: true (of {t}) / non duplicate | Judged the same: true / "
                "wrong | Precision | Recall | Estimated cost |"
            ),
            "|---|---|---|---|---|---|---|",
        ]
        for name in self.cells:
            for r in self.runs:
                c = r["cells"][name]
                usd = "" if c["estimated_usd"] is None else f"${c['estimated_usd']:.3f}"
                lines.append(
                    f"| {name} | {r['run']} | {c['candidates']['true']} / "
                    f"{c['candidates']['false']} | {c['judged']['true_positive']} / "
                    f"{c['judged']['false_positive']} | {c['judged']['precision']:.3f} | "
                    f"{c['judged']['recall']:.3f} | {usd} |"
                )
        lines += [
            "",
            (
                "Estimated cost per cell: the run's measured extraction tokens plus the cell's "
                "number of judgments at the measured mean tokens per judgment for its prompt."
            ),
            "",
            "| Cell | Wrong joins per run | Mean recall | Lost before judging / at judging (per run) |",
            "|---|---|---|---|",
        ]
        for name in self.cells:
            s = self.summary[name]
            lost = ", ".join(f"{a} / {b}" for a, b in s["lost_per_run"])
            lines.append(
                f"| {name} | {', '.join(map(str, s['wrong_joins_per_run']))} | "
                f"{s['mean_recall']:.3f} | {lost} |"
            )
        d = self.decision
        lines += [
            "",
            "## Decision",
            "",
            f"**{d['chosen']}**. {d['reason']}",
            "",
            "## Wrong joins",
            "",
        ]
        any_wrong = False
        for name in self.cells:
            for r in self.runs:
                for w in r["cells"][name]["wrong_joins"]:
                    any_wrong = True
                    reason = w["reason"].replace("|", "\\|")
                    lines.append(
                        f"- {name}, run {r['run']}: {w['pair']} ({w['clusters']}). {reason}"
                    )
        if not any_wrong:
            lines.append("None in any cell or run.")

        lines += ["", "## Run to run agreement", ""]
        for name in self.cells:
            a = self.agreement[name]
            only = "; ".join(
                f"run {k} only: {', '.join(v) or 'none'}" for k, v in a["only_in"].items()
            )
            lines.append(
                f"- {name}: joined in every run {len(a['in_all'])} "
                f"({', '.join(a['in_all']) or 'none'}); {only}."
            )
        lines.append(
            "- Candidates at each rule, every run / one run only: "
            + "; ".join(
                f"{m} shared {len(v['in_all'])} / {sum(len(x) for x in v['only_in'].values())}"
                for m, v in self.agreement["candidates"].items()
            )
            + "."
        )

        lines += [
            "",
            "## Cost (measured tokens)",
            "",
            "| Run | Stage | Model | Calls | Input tokens | Output tokens |",
            "|---|---|---|---|---|---|",
        ]
        for r in self.runs:
            for stage, by_model in r["stage_usage"].items():
                for model, u in by_model.items():
                    lines.append(
                        f"| {r['run']} | {stage} | {model} | {u['calls']} | "
                        f"{u['input_tokens']:,} | {u['output_tokens']:,} |"
                    )
        tok = self.tokens
        for model, u in tok["total"].items():
            lines.append(
                f"| all | all | {model} | {u['calls']} | {u['input_tokens']:,} | "
                f"{u['output_tokens']:,} |"
            )
        lines += ["", "Output tokens include thinking tokens."]
        if tok["usd"] is not None:
            prices = "; ".join(
                f"{m} ${p[0]:.2f} input, ${p[1]:.2f} output"
                for m, p in tok["prices_per_mtok"].items()
            )
            lines.append(
                f"Measured total for the whole ablation: ${tok['usd']:.2f} at list prices per "
                f"million tokens as of {tok['prices_as_of']} ({prices})."
            )

        lines += [
            "",
            "## How judgments scale with batch size",
            "",
            (
                "Pairs grow with the square of the batch, and every candidate pair costs one small "
                "model judgment, so the candidate rule sets the cost on a real batch. Projection "
                "from the share of pairs that became candidates here:"
            ),
            "",
            "| Rule | Candidate share of pairs | "
            + " | ".join(f"Judgments at {n} reviews" for n in BATCH_SIZES)
            + " |",
            "|---|---|" + "---|" * len(BATCH_SIZES),
        ]
        for s in self.scaling:
            cells = " | ".join(
                f"{p['judgments']:,}" + (f" (${p['usd']:.2f})" if p["usd"] is not None else "")
                for p in s["projection"]
            )
            lines.append(f"| {s['min_shared']} shared | {s['share']:.3f} | {cells} |")
        lines += [
            "",
            (
                "Dollar figures are small model judgments only, at the measured mean tokens per "
                "judgment. The share comes from a set where half the reviews are rewrites of each "
                "other and the rest are spread over topics; honest reviews of one real property "
                "mention the same features more often, so expect a higher share there."
            ),
            "",
            "## Methods",
            "",
        ]
        lines += [f"- {k}: {v}" for k, v in self.settings.items()]
        lines += [
            "- The judge prompts are recorded in full in the JSON report.",
            "- An ablation is scored only if every extraction and judgment call succeeded.",
            "",
            "## Limits",
            "",
        ]
        lines += [f"- {x}" for x in self.limits]
        return "\n".join(lines) + "\n"


def evaluate_grid(
    rows: Sequence[LabelledReview],
    make_client: Callable[[], ModelClient],
    *,
    runs: int = 2,
    min_shared: Sequence[int] = (2, 1),
    judges: Sequence[str] = ("strict", "lenient"),
    save_dir: Path | None = None,
    current: Cell | None = None,
    settings: dict[str, str] | None = None,
    limits: Sequence[str] = (),
) -> AblationReport:
    """Score every cell on shared extractions. ``current`` defaults to the product setting."""
    truth = truth_pairs(rows)
    n_pairs = len(rows) * (len(rows) - 1) // 2
    cells: list[Cell] = [(m, j) for m in min_shared for j in judges]
    current = current or (MIN_SHARED_SPECIFIC, JUDGE_STYLE)

    results: list[AblationRun] = []
    files: list[str] = []
    models: dict[str, str] = {}
    for k in range(1, runs + 1):
        client = make_client()
        models = {"main": client.config.main, "small": client.config.small}
        path = save_dir / f"extraction_run{k}.json" if save_dir else None
        results.append(ablate_once(rows, client, min_shared, judges, path))
        if path:
            files.append(path.as_posix())

    run_dicts = []
    for k, run in enumerate(results, start=1):
        run_dicts.append(
            {
                "run": k,
                "cells": {cell_name(c): score_cell(rows, truth, run, c) for c in cells},
                "stage_usage": run.stage_usage,
                "claims": {str(i): [asdict(c) for c in v] for i, v in run.claims.items()},
                "judgments": {
                    j: [
                        {"pair": _label(rows, p), "same_claims": v.same_claims, "reason": v.reason}
                        for p, v in sorted(by_pair.items())
                    ]
                    for j, by_pair in run.judgments.items()
                },
            }
        )

    summary = {}
    agreement: dict[str, Any] = {}
    for c in cells:
        name = cell_name(c)
        per_run = [r["cells"][name] for r in run_dicts]
        summary[name] = {
            "wrong_joins_per_run": [x["judged"]["false_positive"] for x in per_run],
            "recall_per_run": [x["judged"]["recall"] for x in per_run],
            "mean_recall": sum(x["judged"]["recall"] for x in per_run) / len(per_run),
            "lost_per_run": [
                (x["missed_not_candidate"], x["missed_judged_different"]) for x in per_run
            ],
        }
        sets = [set(x["judged_same_pairs"]) for x in per_run]
        both = set.intersection(*sets)
        agreement[name] = {
            "in_all": sorted(both),
            "only_in": {str(k): sorted(s - both) for k, s in enumerate(sets, start=1)},
        }
    cand_agree = {}
    for m in min_shared:
        sets = [{_label(rows, p) for p in run.candidates[m]} for run in results]
        both = set.intersection(*sets)
        cand_agree[str(m)] = {
            "in_all": sorted(both),
            "only_in": {str(k): sorted(s - both) for k, s in enumerate(sets, start=1)},
        }
    agreement["candidates"] = cand_agree

    total: dict[str, dict[str, int]] = {}
    for run in results:
        for stage in run.stage_usage.values():
            for model, u in stage.items():
                t = total.setdefault(model, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
                for key in t:
                    t[key] += u[key]
    costs = [_usd(m, u["input_tokens"], u["output_tokens"]) for m, u in total.items()]
    usd = sum(costs) if costs and None not in costs else None

    # Mean tokens per judgment over every judging stage of every run.
    j_calls = j_in = j_out = 0
    j_model = models.get("small", "")
    for run in results:
        for stage, by_model in run.stage_usage.items():
            if stage.startswith("judge"):
                for model, u in by_model.items():
                    j_model = model
                    j_calls += u["calls"]
                    j_in += u["input_tokens"]
                    j_out += u["output_tokens"]
    scaling = []
    for m in min_shared:
        share = sum(len(run.candidates[m]) for run in results) / (len(results) * n_pairs)
        projection = []
        for n in BATCH_SIZES:
            judgments = round(share * n * (n - 1) / 2)
            cost = (
                _usd(j_model, j_in / j_calls * judgments, j_out / j_calls * judgments)
                if j_calls
                else None
            )
            projection.append({"reviews": n, "judgments": judgments, "usd": cost})
        scaling.append({"min_shared": m, "share": share, "projection": projection})

    return AblationReport(
        created_at=_dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        python_version=platform.python_version(),
        models=models,
        settings=dict(settings or {}),
        reviews=len(rows),
        truth_pairs=len(truth),
        negative_pairs=n_pairs - len(truth),
        cells=[cell_name(c) for c in cells],
        current_default=cell_name(current),
        runs=run_dicts,
        summary=summary,
        agreement=agreement,
        decision=choose_default(summary, cells, current),
        tokens={
            "total": total,
            "usd": usd,
            "prices_per_mtok": {m: PRICES_PER_MTOK[m] for m in total if m in PRICES_PER_MTOK},
            "prices_as_of": PRICES_AS_OF,
        },
        scaling=scaling,
        extraction_files=files,
        judge_prompts={j: JUDGE_SYSTEMS[j] for j in judges},
        limits=list(limits),
    )
