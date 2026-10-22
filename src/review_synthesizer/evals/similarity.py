"""Evaluation of the two duplicate signals on the labelled paraphrase set.

Question: does embedding similarity find rewritten duplicates that word overlap
misses, without joining distinct genuine reviews that happen to share a topic?

Ground truth is pair level. Every pair of reviews in the same labelled cluster
is a positive; every other pair is a negative. For each signal and threshold we
count the pairs it joins and report precision (joined pairs that are true
duplicates) and recall (true duplicate pairs it joined). The embedding threshold
is swept so the default is chosen from data rather than guessed, and the two
signals are compared on the same pairs: caught by both, by one, or by neither.

The set is small and synthetic (see tests/fixtures/paraphrase_set.json), so the
figures show the mechanism, not field accuracy.
"""

from __future__ import annotations

import datetime as _dt
import json
import platform
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from ..authenticity.embeddings import Embedder, similarity_matrix
from ..authenticity.models import Review
from ..authenticity.rules import DUP_THRESHOLD, duplicate_pairs

DEFAULT_SWEEP = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)


@dataclass(frozen=True)
class LabelledReview:
    review: Review
    cluster: str | None


def load_paraphrase_set(path: str | Path) -> tuple[list[LabelledReview], dict[str, str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = [LabelledReview(Review.from_dict(r), r.get("cluster")) for r in data["reviews"]]
    return rows, data.get("clusters", {})


def truth_pairs(rows: Sequence[LabelledReview]) -> set[tuple[int, int]]:
    return {
        (i, j)
        for i, j in combinations(range(len(rows)), 2)
        if rows[i].cluster is not None and rows[i].cluster == rows[j].cluster
    }


@dataclass(frozen=True)
class PairMetrics:
    threshold: float
    joined: int
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float


def pair_metrics(
    joined: set[tuple[int, int]], truth: set[tuple[int, int]], threshold: float
) -> PairMetrics:
    tp = len(joined & truth)
    fp = len(joined - truth)
    fn = len(truth - joined)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return PairMetrics(threshold, len(joined), tp, fp, fn, precision, recall, f1)


@dataclass
class SimilarityReport:
    model: str
    reviews: int
    clusters: dict[str, str]
    truth_pairs: int
    negative_pairs: int
    jaccard: PairMetrics
    embedding_sweep: list[PairMetrics]
    chosen_threshold: float
    chosen: PairMetrics
    overlap: dict[str, int]
    """At the chosen threshold: pairs caught by both, by embeddings only, by Jaccard only, by neither."""
    false_positive_pairs: list[list[str]]
    """Review ids wrongly joined at the chosen threshold, with the similarity."""
    missed_pairs: list[list[str]]
    """True duplicate pairs still missed at the chosen threshold, with the similarity."""
    nearest_negative: list[Any]
    """The most similar pair of reviews that are not duplicates: the margin the threshold has."""
    within_cluster: dict[str, float]
    """Cosine over true duplicate pairs: min, mean, max."""
    negatives: dict[str, float]
    """Cosine over non duplicate pairs: mean, p95, max."""
    per_cluster: list[dict[str, Any]]
    """Per labelled cluster: pair count and min, mean, max cosine inside it."""
    separable: bool
    """True when every true pair scores above every negative pair (a threshold exists)."""
    created_at: str
    python_version: str
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_markdown(self) -> str:
        j, c = self.jaccard, self.chosen
        lines = [
            "# Duplicate signals: word overlap versus embedding similarity",
            "",
            f"Generated {self.created_at} on Python {self.python_version}. Embedding model: "
            f"`{self.model}`.",
            "",
            f"Set: {self.reviews} reviews, {len(self.clusters)} labelled clusters of rewritten "
            f"duplicates ({self.truth_pairs} true duplicate pairs) and {self.negative_pairs} "
            "pairs that should not be joined. Synthetic; see tests/fixtures/paraphrase_set.json.",
            "",
            "## Pair level results",
            "",
            "| Signal | Threshold | Pairs joined | Correct | Wrong | Missed | Precision | Recall | F1 |",
            "|---|---|---|---|---|---|---|---|---|",
            f"| Word overlap (Jaccard) | {j.threshold:.2f} | {j.joined} | {j.true_positive} | "
            f"{j.false_positive} | {j.false_negative} | {j.precision:.3f} | {j.recall:.3f} | {j.f1:.3f} |",
        ]
        for m in self.embedding_sweep:
            mark = " **(default)**" if abs(m.threshold - self.chosen_threshold) < 1e-9 else ""
            lines.append(
                f"| Embedding cosine{mark} | {m.threshold:.2f} | {m.joined} | {m.true_positive} | "
                f"{m.false_positive} | {m.false_negative} | {m.precision:.3f} | {m.recall:.3f} | {m.f1:.3f} |"
            )
        w, ng = self.within_cluster, self.negatives
        lines += [
            "",
            "## Where the similarities fall",
            "",
            "| Pairs | Min | Mean | p95 | Max |",
            "|---|---|---|---|---|",
            f"| True duplicates ({self.truth_pairs}) | {w['min']:.3f} | {w['mean']:.3f} | | {w['max']:.3f} |",
            f"| Everything else ({self.negative_pairs}) | | {ng['mean']:.3f} | {ng['p95']:.3f} | {ng['max']:.3f} |",
            "",
            "A threshold that separates the two exists only if the duplicates' minimum is above the "
            f"others' maximum: {'yes' if self.separable else 'no'} on this set.",
            "",
            "| Cluster | Written as | Pairs | Min | Mean | Max |",
            "|---|---|---|---|---|---|",
        ]
        for c in self.per_cluster:
            lines.append(
                f"| {c['cluster']} | {c['claim']} | {c['pairs']} | {c['min']:.3f} | {c['mean']:.3f} | {c['max']:.3f} |"
            )
        o = self.overlap
        lines += [
            "",
            f"## What each signal adds (embedding threshold {self.chosen_threshold:.2f})",
            "",
            f"- True duplicate pairs caught by both signals: {o['both']}",
            f"- Caught by embeddings only: {o['embedding_only']}",
            f"- Caught by word overlap only: {o['jaccard_only']}",
            f"- Missed by both: {o['neither']}",
            "",
            f"Nearest non duplicate pair: {self.nearest_negative[0]} and {self.nearest_negative[1]} "
            f"at cosine {self.nearest_negative[2]:.3f}. The gap between that and the threshold is "
            "the margin against false positives on this set.",
            "",
        ]
        if self.false_positive_pairs:
            lines += ["Wrongly joined at the default threshold:", ""]
            lines += [f"- {a} and {b} (cosine {s})" for a, b, s in self.false_positive_pairs]
            lines.append("")
        if self.missed_pairs:
            lines += ["True duplicates still missed at the default threshold:", ""]
            lines += [f"- {a} and {b} (cosine {s})" for a, b, s in self.missed_pairs]
            lines.append("")
        if self.extras:
            lines += ["## Additional checks", ""]
            lines += [f"- {k}: {v}" for k, v in self.extras.items()]
            lines.append("")
        lines += [
            "## Read this before quoting the numbers",
            "",
            "- Twenty four synthetic reviews written for this project. The clusters were written "
            "to share meaning and avoid shared words, which is the case word overlap is known to "
            "miss; the distinct reviews were written to share topics without sharing claims. The "
            "figures show the mechanism and the chosen threshold's margin, not field accuracy.",
            "- Embedding similarity joins reviews that say the same thing. Two honest guests can "
            "say the same thing. The signal is one input to a human decision, never a verdict, "
            "and the Module 6 false positive risk (a cluster of honest praise after a good "
            "weekend) is exactly what the nearest non duplicate figure watches.",
            "- If the duplicates' minimum sits below the others' maximum, no cosine threshold on "
            "this model separates rewritten fakes from genuine reviews on the same topic. Cosine "
            "similarity measures wording and topic; it does not compare the claims being made. "
            "Claim level comparison is the job of the language model step.",
            "",
        ]
        return "\n".join(lines)


def evaluate(
    rows: Sequence[LabelledReview],
    clusters: dict[str, str],
    embedder: Embedder,
    *,
    chosen_threshold: float = 0.80,
    sweep: Sequence[float] = DEFAULT_SWEEP,
    jaccard_threshold: float = DUP_THRESHOLD,
) -> SimilarityReport:
    reviews = [r.review for r in rows]
    truth = truth_pairs(rows)
    n_pairs = len(reviews) * (len(reviews) - 1) // 2

    jac = {(i, j) for i, j, _ in duplicate_pairs(reviews, jaccard_threshold)}
    jaccard_m = pair_metrics(jac, truth, jaccard_threshold)

    # One embedding pass; the full matrix gives every pair, not only those above a threshold.
    sims = similarity_matrix(embedder.embed([r.text for r in reviews]))
    n = len(reviews)
    sim_of = {(i, j): float(sims[i, j]) for i in range(n) for j in range(i + 1, n)}

    sweep_metrics = []
    for t in sweep:
        joined = {p for p, s_ in sim_of.items() if s_ >= t}
        sweep_metrics.append(pair_metrics(joined, truth, t))
    chosen_joined = {p for p, s_ in sim_of.items() if s_ >= chosen_threshold}
    chosen_m = pair_metrics(chosen_joined, truth, chosen_threshold)

    overlap = {
        "both": len(truth & jac & chosen_joined),
        "embedding_only": len((truth & chosen_joined) - jac),
        "jaccard_only": len((truth & jac) - chosen_joined),
        "neither": len(truth - jac - chosen_joined),
    }

    def ids(p):
        return [str(reviews[p[0]].id), str(reviews[p[1]].id)]

    pos_vals = np.array([sim_of[p] for p in truth]) if truth else np.zeros(0)
    neg_pairs = [p for p in sim_of if p not in truth]
    neg_vals = np.array([sim_of[p] for p in neg_pairs]) if neg_pairs else np.zeros(0)
    within = {
        "min": float(pos_vals.min()) if pos_vals.size else 0.0,
        "mean": float(pos_vals.mean()) if pos_vals.size else 0.0,
        "max": float(pos_vals.max()) if pos_vals.size else 0.0,
    }
    negatives = {
        "mean": float(neg_vals.mean()) if neg_vals.size else 0.0,
        "p95": float(np.percentile(neg_vals, 95)) if neg_vals.size else 0.0,
        "max": float(neg_vals.max()) if neg_vals.size else 0.0,
    }
    separable = bool(pos_vals.size and neg_vals.size and within["min"] > negatives["max"])

    per_cluster = []
    for name in sorted(clusters):
        members = [i for i, r in enumerate(rows) if r.cluster == name]
        vals = [sim_of[(a, b)] for a, b in combinations(members, 2)]
        if vals:
            per_cluster.append(
                {
                    "cluster": name,
                    "claim": clusters[name],
                    "pairs": len(vals),
                    "min": float(min(vals)),
                    "mean": float(sum(vals) / len(vals)),
                    "max": float(max(vals)),
                }
            )

    fps = sorted(chosen_joined - truth, key=lambda p: -sim_of[p])
    missed = sorted(truth - chosen_joined, key=lambda p: -sim_of[p])
    if neg_pairs:
        p = max(neg_pairs, key=lambda q: sim_of[q])
        nearest = ids(p) + [round(sim_of[p], 3)]
    else:
        nearest = ["none", "none", 0.0]

    return SimilarityReport(
        model=embedder.name,
        reviews=len(reviews),
        clusters=dict(clusters),
        truth_pairs=len(truth),
        negative_pairs=n_pairs - len(truth),
        jaccard=jaccard_m,
        embedding_sweep=sweep_metrics,
        chosen_threshold=chosen_threshold,
        chosen=chosen_m,
        overlap=overlap,
        false_positive_pairs=[ids(p) + [round(sim_of[p], 3)] for p in fps],
        missed_pairs=[ids(p) + [round(sim_of[p], 3)] for p in missed],
        nearest_negative=nearest,
        within_cluster=within,
        negatives=negatives,
        per_cluster=per_cluster,
        separable=separable,
        created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        python_version=platform.python_version(),
    )
