"""Rule based authenticity signals.

Faithful port of the capstone prototype's ``detect()``:

* near duplicate phrasing: any pair of reviews whose token sets have a Jaccard
  similarity at or above ``DUP_THRESHOLD`` flags both reviews;
* rating bursts: for each star rating, any forward window of ``BURST_DAYS`` days
  that holds at least ``BURST_MIN`` reviews flags every review in the window.

What is new in this port is structure: pairs are joined into clusters, each
signal is returned as a :class:`Flag` with a score, a confidence band and a
written rationale, and the rating summary is a function rather than a side
effect of rendering. Thresholds are unchanged so the capstone figures reproduce.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .models import DetectionResult, Flag, RatingSummary, Review, ReviewId, confidence_band
from .text import jaccard, tokenize

DUP_THRESHOLD = 0.55
"""Near duplicate phrasing: pairwise Jaccard similarity at or above this value."""
BURST_MIN = 3
"""Rating burst: at least this many same rating reviews ..."""
BURST_DAYS = 4
"""... posted within this many days of each other."""


@dataclass(frozen=True)
class DuplicateCluster:
    review_ids: tuple[ReviewId, ...]
    peak_similarity: float
    mean_similarity: float
    """Mean over the pairs that met the threshold."""


@dataclass(frozen=True)
class RatingBurst:
    rating: int
    review_ids: tuple[ReviewId, ...]
    start: str
    end: str
    span_days: int


def find_duplicate_clusters(
    reviews: Sequence[Review], threshold: float = DUP_THRESHOLD
) -> list[DuplicateCluster]:
    """Group reviews joined by near duplicate pairs (transitively) into clusters."""
    toks = [tokenize(r.text) for r in reviews]
    n = len(reviews)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    qualifying: dict[int, list[float]] = defaultdict(list)
    pairs: list[tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = jaccard(toks[i], toks[j])
            if sim >= threshold:
                pairs.append((i, j, sim))
                parent[find(i)] = find(j)

    for i, j, sim in pairs:
        qualifying[find(i)].append(sim)

    members: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        members[find(i)].append(i)

    clusters = []
    for root, idxs in members.items():
        if len(idxs) < 2:
            continue
        sims = qualifying[root]
        clusters.append(
            DuplicateCluster(
                review_ids=tuple(reviews[i].id for i in idxs),
                peak_similarity=max(sims),
                mean_similarity=sum(sims) / len(sims),
            )
        )
    clusters.sort(key=lambda c: min(reviews.index(r) for r in reviews if r.id in c.review_ids))
    return clusters


def find_rating_bursts(
    reviews: Sequence[Review], min_size: int = BURST_MIN, window_days: int = BURST_DAYS
) -> list[RatingBurst]:
    """Find same rating reviews posted in dense forward windows, one burst per rating."""
    by_rating: dict[int, list[Review]] = defaultdict(list)
    for r in reviews:
        by_rating[r.rating].append(r)

    bursts = []
    for rating in sorted(by_rating):
        ordered = sorted(by_rating[rating], key=lambda r: r.posted)  # stable, like v1
        hit: dict[ReviewId, Review] = {}
        for anchor in ordered:
            window = [
                r
                for r in ordered
                if r.posted >= anchor.posted and (r.posted - anchor.posted).days <= window_days
            ]
            if len(window) >= min_size:
                for r in window:
                    hit[r.id] = r
        if hit:
            dates = sorted(r.posted for r in hit.values())
            bursts.append(
                RatingBurst(
                    rating=rating,
                    review_ids=tuple(r.id for r in ordered if r.id in hit),
                    start=dates[0].isoformat(),
                    end=dates[-1].isoformat(),
                    span_days=(dates[-1] - dates[0]).days,
                )
            )
    return bursts


def detect(
    reviews: Sequence[Review],
    *,
    dup_threshold: float = DUP_THRESHOLD,
    burst_min: int = BURST_MIN,
    burst_days: int = BURST_DAYS,
) -> DetectionResult:
    """Run every rule and return flags with reasons, plus the union of flagged ids."""
    flags: list[Flag] = []

    for c in find_duplicate_clusters(reviews, dup_threshold):
        score = c.peak_similarity
        flags.append(
            Flag(
                signal_type="duplicate_phrasing",
                review_ids=list(c.review_ids),
                score=score,
                confidence=confidence_band(score),
                rationale=(
                    f"{len(c.review_ids)} reviews share near duplicate phrasing: peak Jaccard "
                    f"similarity {c.peak_similarity:.2f} (mean {c.mean_similarity:.2f}) against a "
                    f"threshold of {dup_threshold:.2f}."
                ),
                details={
                    "peak_similarity": c.peak_similarity,
                    "mean_similarity": c.mean_similarity,
                    "threshold": dup_threshold,
                },
            )
        )

    for b in find_rating_bursts(reviews, burst_min, burst_days):
        count = len(b.review_ids)
        score = min(1.0, count / (2 * burst_min))  # provisional heuristic; see docs
        flags.append(
            Flag(
                signal_type="rating_burst",
                review_ids=list(b.review_ids),
                score=score,
                confidence=confidence_band(score),
                rationale=(
                    f"{count} {b.rating}-star reviews posted within {b.span_days} days "
                    f"({b.start} to {b.end}); threshold is {burst_min} within {burst_days} days."
                ),
                details={
                    "rating": b.rating,
                    "start": b.start,
                    "end": b.end,
                    "span_days": b.span_days,
                    "min_size": burst_min,
                    "window_days": burst_days,
                },
            )
        )

    flagged: set[ReviewId] = set()
    for f in flags:
        flagged.update(f.review_ids)

    # Peak similarity across the whole flagged group, as the v1 panel displayed it.
    toks = {r.id: tokenize(r.text) for r in reviews if r.id in flagged}
    ids = list(toks)
    peak = 0.0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            peak = max(peak, jaccard(toks[ids[i]], toks[ids[j]]))

    return DetectionResult(flags=flags, flagged_ids=flagged, peak_similarity=peak)


def round_half_up(value: float, ndigits: int = 1) -> float:
    """Round like JavaScript's ``Math.round(n * 10) / 10`` (half away from zero)."""
    factor = 10**ndigits
    return math.floor(value * factor + 0.5) / factor


def rating_summary(reviews: Iterable[Review], flagged_ids: set[ReviewId]) -> RatingSummary:
    """Displayed mean over all reviews versus verified mean over unflagged ones."""
    reviews = list(reviews)
    passed = [r for r in reviews if r.id not in flagged_ids]
    displayed = round_half_up(sum(r.rating for r in reviews) / len(reviews))
    verified = round_half_up(sum(r.rating for r in passed) / len(passed)) if passed else None
    return RatingSummary(
        total=len(reviews),
        removed=len(reviews) - len(passed),
        displayed=displayed,
        verified=verified,
    )
