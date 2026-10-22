"""Ablation of the claim duplicate signal's two levers, with a same property negative control.

Levers: the candidate rule (at least 1 or 2 shared specific claims by aspect and polarity)
and the judge prompt (strict or lenient, ``graph.prompts.JUDGE_SYSTEMS``).

Two sets. The paraphrase set measures recall (18 true pairs) and has negatives on different
topics. The control set is honest reviews of one property (the 20 genuine Harbor Light
reviews, 190 pairs): every join there is wrong, and these pairs mention the same features,
which is where a lenient setting fails. Extraction varies between runs, so each run extracts
once per set and every cell is scored on that same extraction: candidates at 1 shared claim
include those at 2, the union is judged once with each prompt, and a cell's joins are its own
candidates judged the same by its own prompt.

The product default is chosen by a rule fixed before the run (:func:`choose_default`): a cell
qualifies only with no wrong join on either set in either run; among qualifying cells, the
highest mean paraphrase recall wins, ties to the stricter setting; if none qualifies, nothing
changes, because the signal then needs a design change, not a setting.

What this cannot do: generalise. The setting is chosen on the sets it is reported on, and
two runs are not a distribution.
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
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")


def ablate_once(
    rows: Sequence[LabelledReview],
    client: ModelClient,
    min_shared: Sequence[int],
    judges: Sequence[str],
    save_to: Path | None = None,
    label: str = "paraphrase",
) -> AblationRun:
    """One extraction, candidates at each rule, the union judged with each prompt."""
    before = usage_of(client)
    claims = extract_all(rows, client)
    stage_usage = {f"{label} extraction": _diff(before, usage_of(client))}
    if save_to is not None:
        _save_claims(save_to, claims, client.config.main)

    candidates = {m: candidates_at(rows, claims, m) for m in min_shared}
    union: set[Pair] = set().union(*candidates.values())
    judgments = {}
    for j in judges:
        before = usage_of(client)
        judgments[j] = judge_pairs(rows, claims, union, client, j)
        stage_usage[f"{label} judge {j}"] = _diff(before, usage_of(client))
    return AblationRun(claims, candidates, judgments, stage_usage)


def _label(rows: Sequence[LabelledReview], p: Pair) -> str:
    return f"{rows[p[0]].review.id}-{rows[p[1]].review.id}"


def _usd(model: str, input_tokens: float, output_tokens: float) -> float | None:
    if model not in PRICES_PER_MTOK:
        return None
    p_in, p_out = PRICES_PER_MTOK[model]
    return input_tokens / 1e6 * p_in + output_tokens / 1e6 * p_out


def score_cell(
    rows: Sequence[LabelledReview], truth: set[Pair], run: AblationRun, cell: Cell
) -> dict[str, Any]:
    """Paraphrase set: candidates, joins, precision and recall, wrong joins with reasons."""
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
    return {
        "candidates": {"total": len(cand), "true": len(cand & truth), "false": len(cand - truth)},
        "judged": asdict(metrics),
        "wrong_joins": wrong,
        "missed_not_candidate": len(truth - cand),
        "missed_judged_different": len((truth & cand) - same),
        "judged_same_pairs": sorted(_label(rows, p) for p in same),
    }


def score_control(rows: Sequence[LabelledReview], run: AblationRun, cell: Cell) -> dict[str, Any]:
    """Control set: every pair is honest, so every join is wrong."""
    m, j = cell
    cand = run.candidates[m]
    same = sorted(p for p in cand if run.judgments[j][p].same_claims)
    return {
        "candidates": len(cand),
        "wrong_joins": [
            {"pair": _label(rows, p), "reason": run.judgments[j][p].reason} for p in same
        ],
    }


def choose_default(summary: dict[str, dict[str, Any]], cells: Sequence[Cell]) -> dict[str, Any]:
    """The fixed rule: no wrong join on either set in either run, then the highest mean
    paraphrase recall, ties to the stricter setting; no qualifying cell means no change."""

    def qualifies(c: Cell) -> bool:
        s = summary[cell_name(c)]
        return not any(s["wrong_joins_per_run"]) and not any(s["control_wrong_joins_per_run"])

    def recall(c: Cell) -> float:
        return summary[cell_name(c)]["mean_recall"]

    qualifying = [c for c in cells if qualifies(c)]
    if not qualifying:
        return {
            "chosen": None,
            "changed": False,
            "qualifying": [],
            "reason": (
                "No cell qualifies: every setting made a wrong join on at least one set. "
                "Nothing is changed; the signal needs a design change, not a setting."
            ),
        }
    best = max(qualifying, key=lambda c: (recall(c), strictness(c)))
    tied = sum(1 for c in qualifying if recall(c) == recall(best)) > 1
    return {
        "chosen": cell_name(best),
        "changed": True,
        "qualifying": [cell_name(c) for c in qualifying],
        "reason": (
            f"Qualifying cells (no wrong join on either set in either run): "
            f"{', '.join(cell_name(c) for c in qualifying)}. Highest mean paraphrase recall: "
            f"{cell_name(best)} ({recall(best):.3f})"
            + ("; tied cells resolved to the stricter setting." if tied else ".")
        ),
        "min_shared": best[0],
        "judge": best[1],
    }


def _agreement(sets: list[set[str]]) -> dict[str, Any]:
    both = set.intersection(*sets) if sets else set()
    return {
        "in_all": sorted(both),
        "only_in": {str(k): sorted(s - both) for k, s in enumerate(sets, start=1)},
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
    control_reviews: int
    control_pairs: int
    cells: list[str]
    current_default: str
    runs: list[dict[str, Any]]
    summary: dict[str, dict[str, Any]]
    agreement: dict[str, Any]
    decision: dict[str, Any]
    tokens: dict[str, Any]
    scaling: list[dict[str, Any]]
    extraction_files: list[str]
    judge_prompts: dict[str, str]
    history: list[str] = field(default_factory=list)
    limits: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def bound(self) -> str:
        """What zero wrong joins on the control pairs does and does not bound."""
        n = self.control_pairs
        upper = 3 / n if n else 0.0
        big = 500 * 499 // 2
        return (
            f"Zero wrong joins on {n} honest pairs of one property, in one run, puts the per pair "
            f"false join rate under about {upper:.1%} at 95% confidence (rule of three, treating "
            f"pairs as independent, which they are not: they share reviews). That is a loose "
            f"bound: at {upper:.1%} a batch of 500 reviews ({big:,} pairs) could still hold "
            f"about {round(upper * big):,} wrong joins. It says nothing about other properties, "
            "longer reviews, other languages, or reviews written to share more features than "
            "these do."
        )

    def to_markdown(self) -> str:
        t = self.truth_pairs
        lines = [
            "# Claim duplicates: ablation with a same property negative control",
            "",
            (
                f"Generated {self.created_at} on Python {self.python_version}. Main model "
                f"`{self.models['main']}` (claim extraction), small model `{self.models['small']}` "
                "(pair judgments)."
            ),
            "",
            (
                f"Recall set: {self.reviews} synthetic reviews, {t} true pairs, "
                f"{self.negative_pairs} pairs that should not be joined, on different topics "
                "(tests/fixtures/paraphrase_set.json). Control set: the "
                f"{self.control_reviews} genuine Harbor Light reviews, {self.control_pairs} "
                "honest pairs of one property, every join wrong "
                "(tests/fixtures/harbor_light_inn.json, ids 1 to 20)."
            ),
        ]
        if self.history:
            lines += ["", "## Earlier version of this ablation", ""]
            lines += [f"- {h}" for h in self.history]
        lines += [
            "",
            "## Design",
            "",
            (
                "Each run extracts claims once per set (saved under "
                + ", ".join(f"`{p}`" for p in self.extraction_files)
                + ", not committed). Candidates are taken at both rules from that extraction, "
                "the union of candidate pairs is judged once with each prompt, and every cell is "
                "scored on the same extraction and judgments, so within a run the cells differ "
                "only in the lever."
            ),
            "",
            (
                "Decision rule, fixed before the run: a cell qualifies only with no wrong join on "
                "either set in either run. Among qualifying cells, the highest mean paraphrase "
                "recall wins; ties go to the stricter setting (2 shared claims before 1, then the "
                "strict prompt). If no cell qualifies, nothing changes. Default before this run: "
                f"{self.current_default}."
            ),
            "",
            "## Grid",
            "",
            (
                f"| Cell | Run | Paraphrase candidates: true (of {t}) / non duplicate | Judged "
                "the same: true / wrong | Precision | Recall | Control candidates (of "
                f"{self.control_pairs}) | Control wrong joins |"
            ),
            "|---|---|---|---|---|---|---|---|",
        ]
        for name in self.cells:
            for r in self.runs:
                c, k = r["cells"][name], r["control_cells"][name]
                lines.append(
                    f"| {name} | {r['run']} | {c['candidates']['true']} / "
                    f"{c['candidates']['false']} | {c['judged']['true_positive']} / "
                    f"{c['judged']['false_positive']} | {c['judged']['precision']:.3f} | "
                    f"{c['judged']['recall']:.3f} | {k['candidates']} | "
                    f"{len(k['wrong_joins'])} |"
                )
        lines += [
            "",
            "| Cell | Wrong joins per run: paraphrase / control | Qualifies | Mean paraphrase recall |",
            "|---|---|---|---|",
        ]
        for name in self.cells:
            s = self.summary[name]
            per_run = ", ".join(
                f"{a} / {b}"
                for a, b in zip(
                    s["wrong_joins_per_run"], s["control_wrong_joins_per_run"], strict=True
                )
            )
            ok = "yes" if name in self.decision.get("qualifying", []) else "no"
            lines.append(f"| {name} | {per_run} | {ok} | {s['mean_recall']:.3f} |")
        d = self.decision
        lines += [
            "",
            "## Decision",
            "",
            (f"**{d['chosen']}**. " if d["chosen"] else "**No change.** ") + d["reason"],
            "",
            "## Wrong joins",
            "",
        ]
        any_wrong = False
        for name in self.cells:
            for r in self.runs:
                for w in r["cells"][name]["wrong_joins"]:
                    any_wrong = True
                    lines.append(
                        f"- {name}, run {r['run']}, paraphrase set: {w['pair']} ({w['clusters']}). "
                        + w["reason"].replace("|", "\\|")
                    )
                for w in r["control_cells"][name]["wrong_joins"]:
                    any_wrong = True
                    lines.append(
                        f"- {name}, run {r['run']}, Harbor Light control: {w['pair']}. "
                        + w["reason"].replace("|", "\\|")
                    )
        if not any_wrong:
            lines.append("None in any cell or run.")

        lines += ["", "## Run to run agreement (paraphrase set joins)", ""]
        for name in self.cells:
            a = self.agreement["paraphrase"][name]
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
                f"{m} shared: paraphrase {len(v['in_all'])} / "
                f"{sum(len(x) for x in v['only_in'].values())}, control "
                f"{len(self.agreement['control_candidates'][m]['in_all'])} / "
                f"{sum(len(x) for x in self.agreement['control_candidates'][m]['only_in'].values())}"
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
                "model judgment, so the candidate rule sets the cost on a real batch. The "
                "projection uses the Harbor Light candidate share: honest reviews of one property, "
                "the realistic case. The paraphrase share is shown for comparison."
            ),
            "",
            "| Rule | Harbor Light share | Paraphrase share | "
            + " | ".join(f"Judgments at {n} reviews" for n in BATCH_SIZES)
            + " |",
            "|---|---|---|" + "---|" * len(BATCH_SIZES),
        ]
        for s in self.scaling:
            proj = " | ".join(
                f"{p['judgments']:,}" + (f" (${p['usd']:.2f})" if p["usd"] is not None else "")
                for p in s["projection"]
            )
            lines.append(
                f"| {s['min_shared']} shared | {s['control_share']:.3f} | "
                f"{s['paraphrase_share']:.3f} | {proj} |"
            )
        lines += [
            "",
            (
                "Dollar figures are small model judgments only, at the measured mean tokens per "
                "judgment over every judging stage."
            ),
            "",
            "## What the control bounds",
            "",
            self.bound(),
            "",
            "## Methods",
            "",
        ]
        lines += [f"- {k}: {v}" for k, v in self.settings.items()]
        lines += [
            (
                "- The claim signal is measured on its own: the detector's skip rule is off on both "
                "sets, so no pair is hidden by an earlier join (the end to end run showed that it "
                "can be)."
            ),
            "- The judge prompts are recorded in full in the JSON report.",
            "- An ablation is scored only if every extraction and judgment call succeeded.",
            "",
            "## Limits",
            "",
        ]
        lines += [f"- {x}" for x in self.limits]
        return "\n".join(lines) + "\n"


def _judgment_rows(rows: Sequence[LabelledReview], run: AblationRun) -> dict[str, list[dict]]:
    return {
        j: [
            {"pair": _label(rows, p), "same_claims": v.same_claims, "reason": v.reason}
            for p, v in sorted(by_pair.items())
        ]
        for j, by_pair in run.judgments.items()
    }


def evaluate_grid(
    rows: Sequence[LabelledReview],
    control_rows: Sequence[LabelledReview],
    make_client: Callable[[], ModelClient],
    *,
    runs: int = 2,
    min_shared: Sequence[int] = (2, 1),
    judges: Sequence[str] = ("strict", "lenient"),
    save_dir: Path | None = None,
    current: Cell | None = None,
    settings: dict[str, str] | None = None,
    history: Sequence[str] = (),
    limits: Sequence[str] = (),
) -> AblationReport:
    """Score every cell on shared extractions of both sets; apply the fixed rule."""
    truth = truth_pairs(rows)
    n_pairs = len(rows) * (len(rows) - 1) // 2
    n_control = len(control_rows) * (len(control_rows) - 1) // 2
    cells: list[Cell] = [(m, j) for m in min_shared for j in judges]
    current = current or (MIN_SHARED_SPECIFIC, JUDGE_STYLE)

    para: list[AblationRun] = []
    ctrl: list[AblationRun] = []
    files: list[str] = []
    models: dict[str, str] = {}
    for k in range(1, runs + 1):
        client = make_client()
        models = {"main": client.config.main, "small": client.config.small}
        p_path = save_dir / f"extraction_run{k}.json" if save_dir else None
        c_path = save_dir / f"control_extraction_run{k}.json" if save_dir else None
        para.append(ablate_once(rows, client, min_shared, judges, p_path, "paraphrase"))
        ctrl.append(ablate_once(control_rows, client, min_shared, judges, c_path, "control"))
        files += [p.as_posix() for p in (p_path, c_path) if p]

    run_dicts = []
    for k, (p, c) in enumerate(zip(para, ctrl, strict=True), start=1):
        run_dicts.append(
            {
                "run": k,
                "cells": {cell_name(x): score_cell(rows, truth, p, x) for x in cells},
                "control_cells": {cell_name(x): score_control(control_rows, c, x) for x in cells},
                "stage_usage": {**p.stage_usage, **c.stage_usage},
                "claims": {str(i): [asdict(x) for x in v] for i, v in p.claims.items()},
                "control_claims": {str(i): [asdict(x) for x in v] for i, v in c.claims.items()},
                "judgments": _judgment_rows(rows, p),
                "control_judgments": _judgment_rows(control_rows, c),
            }
        )

    summary = {}
    para_agree = {}
    for x in cells:
        name = cell_name(x)
        per_run = [r["cells"][name] for r in run_dicts]
        per_ctrl = [r["control_cells"][name] for r in run_dicts]
        summary[name] = {
            "wrong_joins_per_run": [r["judged"]["false_positive"] for r in per_run],
            "control_wrong_joins_per_run": [len(r["wrong_joins"]) for r in per_ctrl],
            "control_candidates_per_run": [r["candidates"] for r in per_ctrl],
            "recall_per_run": [r["judged"]["recall"] for r in per_run],
            "mean_recall": sum(r["judged"]["recall"] for r in per_run) / len(per_run),
            "lost_per_run": [
                (r["missed_not_candidate"], r["missed_judged_different"]) for r in per_run
            ],
        }
        para_agree[name] = _agreement([set(r["judged_same_pairs"]) for r in per_run])
    agreement = {
        "paraphrase": para_agree,
        "candidates": {
            str(m): _agreement([{_label(rows, q) for q in r.candidates[m]} for r in para])
            for m in min_shared
        },
        "control_candidates": {
            str(m): _agreement([{_label(control_rows, q) for q in r.candidates[m]} for r in ctrl])
            for m in min_shared
        },
    }

    total: dict[str, dict[str, int]] = {}
    j_calls = j_in = j_out = 0
    j_model = models.get("small", "")
    for run in [*para, *ctrl]:
        for stage, by_model in run.stage_usage.items():
            for model, u in by_model.items():
                t = total.setdefault(model, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
                for key in t:
                    t[key] += u[key]
                if " judge " in stage:
                    j_model = model
                    j_calls += u["calls"]
                    j_in += u["input_tokens"]
                    j_out += u["output_tokens"]
    costs = [_usd(m, u["input_tokens"], u["output_tokens"]) for m, u in total.items()]
    usd = sum(costs) if costs and None not in costs else None

    scaling = []
    for m in min_shared:
        control_share = sum(len(r.candidates[m]) for r in ctrl) / (len(ctrl) * n_control)
        paraphrase_share = sum(len(r.candidates[m]) for r in para) / (len(para) * n_pairs)
        projection = []
        for n in BATCH_SIZES:
            judgments = round(control_share * n * (n - 1) / 2)
            cost = (
                _usd(j_model, j_in / j_calls * judgments, j_out / j_calls * judgments)
                if j_calls
                else None
            )
            projection.append({"reviews": n, "judgments": judgments, "usd": cost})
        scaling.append(
            {
                "min_shared": m,
                "control_share": control_share,
                "paraphrase_share": paraphrase_share,
                "projection": projection,
            }
        )

    return AblationReport(
        created_at=_dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        python_version=platform.python_version(),
        models=models,
        settings=dict(settings or {}),
        reviews=len(rows),
        truth_pairs=len(truth),
        negative_pairs=n_pairs - len(truth),
        control_reviews=len(control_rows),
        control_pairs=n_control,
        cells=[cell_name(x) for x in cells],
        current_default=cell_name(current),
        runs=run_dicts,
        summary=summary,
        agreement=agreement,
        decision=choose_default(summary, cells),
        tokens={
            "total": total,
            "usd": usd,
            "prices_per_mtok": {m: PRICES_PER_MTOK[m] for m in total if m in PRICES_PER_MTOK},
            "prices_as_of": PRICES_AS_OF,
        },
        scaling=scaling,
        extraction_files=files,
        judge_prompts={j: JUDGE_SYSTEMS[j] for j in judges},
        history=list(history),
        limits=list(limits),
    )
