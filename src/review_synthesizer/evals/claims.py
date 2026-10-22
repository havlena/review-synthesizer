"""Evaluation of the claim duplicate signal on the labelled paraphrase set.

Question: do claim extraction plus a pairwise model judgment find the rewritten duplicates
that word overlap and embedding similarity miss, without joining honest reviews that share
a topic?

Same ground truth as the similarity evaluation: every pair inside a labelled cluster is a
positive (18), every other pair a negative (258). The signal has two stages, scored
separately because the first caps the second:

* candidate stage (no model call beyond extraction): pairs sharing at least two specific
  claims by aspect and polarity. Candidate recall is the ceiling on judged recall.
* judged stage: the small model's yes or no per candidate pair. A pair is joined when it is
  judged the same (pair level, as in the similarity evaluation). Pairs that end up in the
  same flag group through a chain are reported separately as grouped pairs.

The detector's skip rule is switched off here (empty detection), so the claim signal is
measured on its own; word overlap joins none of the 18 true pairs anyway, which the report
checks. Every run is scored only if every model call in it succeeded: a failed call raises
:class:`CallFailure` instead of being scored as an empty answer.

The set is small, synthetic, and written by the same person as the detector. Two runs show
whether answers move between runs, not a distribution.
"""

from __future__ import annotations

import datetime as _dt
import json
import platform
import re
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any

from ..authenticity.models import DetectionResult
from ..authenticity.rules import DUP_THRESHOLD, duplicate_pairs
from ..graph.config import MAX_PARALLEL_CALLS
from ..graph.duplicates import (
    JUDGE_STYLE,
    MIN_SHARED_SPECIFIC,
    candidate_pairs,
    claim_flags,
    shared_specific,
)
from ..graph.model import ModelClient
from ..graph.nodes import extract_claims
from ..graph.prompts import JUDGE_SYSTEMS, judge_prompt
from ..graph.schemas import PairVerdict
from ..graph.state import Claim, PairJudgment, Property
from .similarity import LabelledReview, pair_metrics, truth_pairs

PRICES_PER_MTOK = {
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}
"""Anthropic list prices in US dollars per million tokens (input, output), standard API,
as of PRICES_AS_OF. Haiku 5.5 at its rate for prompts up to 100K tokens."""

PRICES_AS_OF = "2026-10-06"

EVAL_PROPERTY = Property(name="Lakeshore hotel (fictional)", type="hotel", region="")
"""The paraphrase set names no property; the prompts get this neutral placeholder."""

Pair = tuple[int, int]


class CallFailure(RuntimeError):
    """A model call failed during an evaluation run; the run is not scored."""

    def __init__(self, stage: str, failures: list[str]) -> None:
        self.stage = stage
        self.failures = failures
        super().__init__(f"{len(failures)} call(s) failed during {stage}: " + " | ".join(failures))


class _Recorder:
    """Passes calls through and records every failure, which the nodes would swallow."""

    def __init__(self, inner: ModelClient) -> None:
        self.inner = inner
        self.config = inner.config
        self.failures: list[str] = []

    def structured(self, schema, *, system, prompt, small=False):
        try:
            return self.inner.structured(schema, system=system, prompt=prompt, small=small)
        except Exception as exc:
            ids = ", ".join(re.findall(r'<review id="([^"]+)"', prompt))
            self.failures.append(
                f"{schema.__name__} for review(s) {ids}: {type(exc).__name__}: {exc}"
            )
            raise


@dataclass
class RunResult:
    """One live pass: what was extracted, which pairs became candidates, which were judged."""

    extracted: dict[Any, list[Claim]]
    """Claims per review id, as extracted in this run."""
    candidates: set[Pair]
    judged_same: set[Pair]
    grouped: set[Pair]
    reasons: dict[Pair, str]
    usage: dict[str, dict[str, int]]

    def specific(self, review_id: Any) -> int:
        return sum(c.specific for c in self.extracted[review_id])


NO_SKIP = DetectionResult(flags=[], flagged_ids=set(), peak_similarity=0.0, signals=[])
"""Empty detection: the claim signal is measured without the detector's skip rule."""


def usage_of(client: ModelClient) -> dict[str, dict[str, int]]:
    """The token usage the client has recorded so far, as plain numbers (none for fakes)."""
    return {
        name: {"calls": u.calls, "input_tokens": u.input_tokens, "output_tokens": u.output_tokens}
        for name, u in getattr(client, "usage", {}).items()
    }


def extract_all(rows: Sequence[LabelledReview], client: ModelClient) -> dict[Any, list[Claim]]:
    """Claims for every review, through the graph's own node; raise on any failed call."""
    rec = _Recorder(client)
    state = {"property": EVAL_PROPERTY, "reviews": [r.review for r in rows]}
    claims = extract_claims(state, client=rec)["claims"]
    if rec.failures:
        raise CallFailure("claim extraction", rec.failures)
    return dict(claims)


def candidates_at(
    rows: Sequence[LabelledReview], claims: dict[Any, list[Claim]], min_shared: int
) -> set[Pair]:
    """Candidate pairs, as row index pairs, under the candidate rule at ``min_shared``."""
    reviews = [r.review for r in rows]
    index = {r.id: i for i, r in enumerate(reviews)}
    pairs = candidate_pairs(reviews, claims, NO_SKIP, min_shared=min_shared)
    return {(index[a.id], index[b.id]) for a, b in pairs}


def judge_pairs(
    rows: Sequence[LabelledReview],
    claims: dict[Any, list[Claim]],
    pairs: set[Pair],
    client: ModelClient,
    judge: str,
) -> dict[Pair, PairJudgment]:
    """One small model judgment per pair with the named prompt; raise on any failed call.

    Same prompt builder, schema and model as the graph's ``claim_duplicates`` node.
    """
    rec = _Recorder(client)
    system = JUDGE_SYSTEMS[judge]

    def one(p: Pair) -> PairJudgment:
        a, b = rows[p[0]].review, rows[p[1]].review
        v = rec.structured(
            PairVerdict,
            system=system,
            prompt=judge_prompt(EVAL_PROPERTY, a, b, claims[a.id], claims[b.id]),
            small=True,
        )
        return PairJudgment(a.id, b.id, v.same_claims, v.reason.strip())

    ordered = sorted(pairs)
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_CALLS) as pool:
        futures = [pool.submit(one, p) for p in ordered]
    results: dict[Pair, PairJudgment] = {}
    for p, fut in zip(ordered, futures, strict=True):
        if fut.exception() is None:
            results[p] = fut.result()
    if rec.failures:
        raise CallFailure(f"pair judgment ({judge} prompt)", rec.failures)
    return results


def run_once(
    rows: Sequence[LabelledReview],
    client: ModelClient,
    *,
    min_shared: int = MIN_SHARED_SPECIFIC,
    judge: str = JUDGE_STYLE,
) -> RunResult:
    """Extract claims and judge candidate pairs; raise :class:`CallFailure` on any failure."""
    reviews = [r.review for r in rows]
    claims = extract_all(rows, client)
    candidates = candidates_at(rows, claims, min_shared)
    judged = judge_pairs(rows, claims, candidates, client, judge)
    index = {r.id: i for i, r in enumerate(reviews)}
    grouped: set[Pair] = set()
    for f in claim_flags(reviews, list(judged.values()), client.config.small, claims):
        grouped.update(combinations(sorted(index[i] for i in f.review_ids), 2))
    return RunResult(
        extracted=claims,
        candidates=candidates,
        judged_same={p for p, j in judged.items() if j.same_claims},
        grouped=grouped,
        reasons={p: j.reason for p, j in judged.items()},
        usage=usage_of(client),
    )


@dataclass
class ClaimReport:
    created_at: str
    python_version: str
    models: dict[str, str]
    settings: dict[str, str]
    reviews: int
    clusters: dict[str, str]
    truth_pairs: int
    negative_pairs: int
    word_overlap_true_pairs: int
    runs: list[dict[str, Any]]
    per_cluster: list[dict[str, Any]]
    agreement: dict[str, Any]
    tokens: dict[str, Any]
    limits: list[str] = field(default_factory=list)
    min_shared: int = MIN_SHARED_SPECIFIC
    judge: str = JUDGE_STYLE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_markdown(self) -> str:
        n = len(self.runs)
        cols = " | ".join(f"Run {r['run']}" for r in self.runs)
        sep = "|---|" + "---|" * n
        t = self.truth_pairs

        def row(label: str, values: Sequence[Any]) -> str:
            return f"| {label} | " + " | ".join(str(v) for v in values) + " |"

        lines = [
            "# Claim duplicates: extraction plus pairwise judgment",
            "",
            (
                f"Generated {self.created_at} on Python {self.python_version}. Main model "
                f"`{self.models['main']}` (claim extraction), small model `{self.models['small']}` "
                "(pair judgments)."
            ),
            "",
            (
                f"Set: {self.reviews} synthetic reviews, {len(self.clusters)} labelled clusters of "
                f"rewritten duplicates ({t} true pairs) and {self.negative_pairs} pairs that should "
                "not be joined (tests/fixtures/paraphrase_set.json). Same set and pair level "
                "scoring as the embedding evaluation (paraphrase_similarity.md)."
            ),
            "",
            "## Stages",
            "",
            (
                f"Candidate stage: pairs sharing at least {self.min_shared} specific claims by "
                f"aspect and polarity. Judged stage: the small model's yes or no on each "
                f"candidate ({self.judge} judge prompt). Candidate recall is the ceiling on "
                "judged recall."
            ),
            "",
            f"| | {cols} |",
            sep,
            row("Candidate pairs", [r["candidates"]["total"] for r in self.runs]),
            row(
                f"True pairs that became candidates (of {t})",
                [r["candidates"]["true"] for r in self.runs],
            ),
            row(
                "Non duplicate pairs that became candidates",
                [r["candidates"]["false"] for r in self.runs],
            ),
            row("Candidate recall", [f"{r['candidates']['recall']:.3f}" for r in self.runs]),
            row("Pairs judged the same", [r["judged"]["joined"] for r in self.runs]),
            row(
                "True candidates judged the same",
                [f"{r['judged']['true_positive']} of {r['candidates']['true']}" for r in self.runs],
            ),
            row(
                "Non duplicate pairs judged the same",
                [r["judged"]["false_positive"] for r in self.runs],
            ),
            row("Judged precision", [f"{r['judged']['precision']:.3f}" for r in self.runs]),
            row(f"Judged recall (of {t})", [f"{r['judged']['recall']:.3f}" for r in self.runs]),
            row(
                "Grouped pairs (through chains): correct / wrong",
                [
                    f"{r['grouped']['true_positive']} / {r['grouped']['false_positive']}"
                    for r in self.runs
                ],
            ),
            "",
            "## Per cluster",
            "",
            (
                "| Cluster | What the rewrites keep | True pairs | Mean specific claims per review | "
                "Candidates | Judged the same |"
            ),
            "|---|---|---|---|---|---|",
        ]
        for c in self.per_cluster:
            lines.append(
                f"| {c['cluster']} | {c['claim']} | {c['pairs']} | "
                + " / ".join(f"{v:.1f}" for v in c["mean_specific"])
                + " | "
                + " / ".join(str(v) for v in c["candidates"])
                + " | "
                + " / ".join(str(v) for v in c["judged_same"])
                + " |"
            )
        lines += ["", "Values separated by a slash are run 1 / run 2.", ""]

        a = self.agreement
        lines += [
            "## Run to run agreement",
            "",
            (
                f"Pairs judged the same in every run: {len(a['judged_in_all'])} "
                f"({_pairs_text(a['judged_in_all'])})."
            ),
            "",
        ]
        for k, pairs in a["judged_only_in"].items():
            lines.append(f"- Judged the same in run {k} only: {len(pairs)} ({_pairs_text(pairs)}).")
        lines += [
            "",
            f"Candidate pairs in every run: {len(a['candidates_in_all'])}.",
        ]
        for k, pairs in a["candidates_only_in"].items():
            lines.append(f"- Candidate in run {k} only: {len(pairs)} ({_pairs_text(pairs)}).")

        for r in self.runs:
            lines += ["", f"## Run {r['run']}: true pairs missed and wrong joins", ""]
            if r["missed"]:
                lines += [
                    "| Pair | Lost at | Shared specific claims | Judgment |",
                    "|---|---|---|---|",
                ]
                for m in r["missed"]:
                    lines.append(
                        f"| {m['pair']} | {m['stage']} | {m['shared_specific']} | "
                        f"{_cell(m.get('reason', ''))} |"
                    )
            else:
                lines.append("No true pair missed.")
            lines.append("")
            if r["false_positives"]:
                lines += ["| Wrongly joined pair | Clusters | Judgment |", "|---|---|---|"]
                for fp in r["false_positives"]:
                    lines.append(f"| {fp['pair']} | {fp['clusters']} | {_cell(fp['reason'])} |")
            else:
                lines.append("No non duplicate pair judged the same.")

        tok = self.tokens
        lines += [
            "",
            "## Cost (measured tokens)",
            "",
            "| Run | Model | Calls | Input tokens | Output tokens |",
            "|---|---|---|---|---|",
        ]
        for r in self.runs:
            for model, u in r["usage"].items():
                lines.append(
                    f"| {r['run']} | {model} | {u['calls']} | {u['input_tokens']:,} | "
                    f"{u['output_tokens']:,} |"
                )
        for model, u in tok["total"].items():
            lines.append(
                f"| all | {model} | {u['calls']} | {u['input_tokens']:,} | {u['output_tokens']:,} |"
            )
        lines += ["", "Output tokens include thinking tokens."]
        if tok["usd"] is not None:
            prices = "; ".join(
                f"{m} ${p[0]:.2f} input, ${p[1]:.2f} output"
                for m, p in tok["prices_per_mtok"].items()
            )
            lines.append(
                f"Estimated cost of all runs: ${tok['usd']:.2f} at list prices per million tokens "
                f"as of {tok['prices_as_of']} ({prices})."
            )

        lines += ["", "## Methods", ""]
        lines += [f"- {k}: {v}" for k, v in self.settings.items()]
        lines += [
            (
                f"- Word overlap at {DUP_THRESHOLD:.2f} joins {self.word_overlap_true_pairs} of the "
                f"{t} true pairs, so the graph's rule of skipping pairs the detector already "
                "joined would not change this result; it was switched off to measure the claim "
                "signal on its own."
            ),
            "- A run is scored only if every extraction and judgment call succeeded.",
            "",
            "## Limits",
            "",
        ]
        lines += [f"- {x}" for x in self.limits]
        return "\n".join(lines) + "\n"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _pairs_text(pairs: Sequence[str]) -> str:
    return ", ".join(pairs) if pairs else "none"


def _label(rows: Sequence[LabelledReview], p: Pair) -> str:
    return f"{rows[p[0]].review.id}-{rows[p[1]].review.id}"


def _run_summary(
    k: int, rows: Sequence[LabelledReview], truth: set[Pair], run: RunResult
) -> dict[str, Any]:
    true_cand = run.candidates & truth
    judged = pair_metrics(run.judged_same, truth, 0.0)
    grouped = pair_metrics(run.grouped, truth, 0.0)
    missed = []
    for p in sorted(truth - run.judged_same):
        a, b = rows[p[0]].review.id, rows[p[1]].review.id
        entry = {
            "pair": _label(rows, p),
            "shared_specific": shared_specific(run.extracted[a], run.extracted[b]),
        }
        if p in run.candidates:
            entry |= {"stage": "judged different", "reason": run.reasons.get(p, "")}
        else:
            entry |= {"stage": "not a candidate"}
        missed.append(entry)
    fps = [
        {
            "pair": _label(rows, p),
            "clusters": f"{rows[p[0]].cluster or 'distinct'} / {rows[p[1]].cluster or 'distinct'}",
            "reason": run.reasons.get(p, ""),
        }
        for p in sorted(run.judged_same - truth)
    ]
    return {
        "run": k,
        "candidates": {
            "total": len(run.candidates),
            "true": len(true_cand),
            "false": len(run.candidates - truth),
            "recall": len(true_cand) / len(truth) if truth else 0.0,
        },
        "judged": asdict(judged),
        "grouped": asdict(grouped),
        "missed": missed,
        "false_positives": fps,
        "claims": {str(r.review.id): [asdict(c) for c in run.extracted[r.review.id]] for r in rows},
        "candidate_pairs": sorted(_label(rows, p) for p in run.candidates),
        "judged_same_pairs": sorted(_label(rows, p) for p in run.judged_same),
        "usage": run.usage,
    }


def evaluate(
    rows: Sequence[LabelledReview],
    clusters: dict[str, str],
    make_client: Callable[[], ModelClient],
    *,
    runs: int = 2,
    min_shared: int = MIN_SHARED_SPECIFIC,
    judge: str = JUDGE_STYLE,
    settings: dict[str, str] | None = None,
    limits: Sequence[str] = (),
) -> ClaimReport:
    """Run one setting ``runs`` times, each with a fresh client and its own extraction."""
    truth = truth_pairs(rows)
    reviews = [r.review for r in rows]
    n_pairs = len(rows) * (len(rows) - 1) // 2

    results: list[RunResult] = []
    models: dict[str, str] = {}
    for _ in range(runs):
        client = make_client()
        models = {"main": client.config.main, "small": client.config.small}
        results.append(run_once(rows, client, min_shared=min_shared, judge=judge))

    summaries = [_run_summary(k, rows, truth, r) for k, r in enumerate(results, start=1)]

    per_cluster = []
    for name in sorted(clusters):
        members = [i for i, r in enumerate(rows) if r.cluster == name]
        pairs = set(combinations(members, 2))
        per_cluster.append(
            {
                "cluster": name,
                "claim": clusters[name],
                "pairs": len(pairs),
                "mean_specific": [
                    sum(r.specific(rows[i].review.id) for i in members) / len(members)
                    for r in results
                ],
                "candidates": [len(r.candidates & pairs) for r in results],
                "judged_same": [len(r.judged_same & pairs) for r in results],
            }
        )

    judged_sets = [r.judged_same for r in results]
    cand_sets = [r.candidates for r in results]
    in_all = set.intersection(*judged_sets) if judged_sets else set()
    cand_all = set.intersection(*cand_sets) if cand_sets else set()
    agreement = {
        "judged_in_all": sorted(_label(rows, p) for p in in_all),
        "judged_only_in": {
            str(k): sorted(_label(rows, p) for p in s - in_all)
            for k, s in enumerate(judged_sets, start=1)
        },
        "candidates_in_all": sorted(_label(rows, p) for p in cand_all),
        "candidates_only_in": {
            str(k): sorted(_label(rows, p) for p in s - cand_all)
            for k, s in enumerate(cand_sets, start=1)
        },
    }

    total: dict[str, dict[str, int]] = {}
    for s in summaries:
        for model, u in s["usage"].items():
            t = total.setdefault(model, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
            for key in t:
                t[key] += u[key]
    priced = {m: PRICES_PER_MTOK[m] for m in total if m in PRICES_PER_MTOK}
    usd = None
    if total and len(priced) == len(total):
        usd = sum(
            u["input_tokens"] / 1e6 * priced[m][0] + u["output_tokens"] / 1e6 * priced[m][1]
            for m, u in total.items()
        )

    word_pairs = {(i, j) for i, j, _ in duplicate_pairs(reviews, DUP_THRESHOLD)}
    return ClaimReport(
        created_at=_dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        python_version=platform.python_version(),
        models=models,
        settings=dict(settings or {}),
        reviews=len(rows),
        clusters=dict(clusters),
        truth_pairs=len(truth),
        negative_pairs=n_pairs - len(truth),
        word_overlap_true_pairs=len(word_pairs & truth),
        runs=summaries,
        per_cluster=per_cluster,
        agreement=agreement,
        tokens={
            "total": total,
            "usd": usd,
            "prices_per_mtok": priced,
            "prices_as_of": PRICES_AS_OF,
        },
        limits=list(limits),
        min_shared=min_shared,
        judge=judge,
    )
