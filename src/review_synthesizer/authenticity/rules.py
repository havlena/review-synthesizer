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

:func:`detect` here runs the rules only. :func:`detector.detect` runs every
signal, including embeddings and the classifier, and is what the rest of the
system calls.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .clustering import union_clusters
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
    pair_count: int = 0


@dataclass(frozen=True)
class RatingBurst:
    rating: int
    review_ids: tuple[ReviewId, ...]
    start: str
    end: str
    span_days: int


def duplicate_pairs(
    reviews: Sequence[Review], threshold: float = DUP_THRESHOLD
) -> list[tuple[int, int, float]]:
    """Index pairs ``(i, j, similarity)`` with Jaccard similarity at or above ``threshold``."""
    toks = [tokenize(r.text) for r in reviews]
    out = []
    for i in range(len(reviews)):
        for j in range(i + 1, len(reviews)):
            sim = jaccard(toks[i], toks[j])
            if sim >= threshold:
                out.append((i, j, sim))
    return out


def find_duplicate_clusters(
    reviews: Sequence[Review], threshold: float = DUP_THRESHOLD
) -> list[DuplicateCluster]:
    """Group reviews joined by near duplicate pairs (transitively) into clusters."""
    return [
        DuplicateCluster(
            review_ids=tuple(reviews[i].id for i in c.members),
            peak_similarity=c.peak,
            mean_similarity=c.mean,
            pair_count=c.pair_count,
        )
        for c in union_clusters(len(reviews), duplicate_pairs(reviews, threshold))
    ]


def find_rating_bursts(
    reviews: Sequence[Review], min_size: int = BURST_MIN, window_days: int = BURST_DAYS
) -> list[RatingBurst]:
    """Find same rating reviews posted in dense forward windows.

    Every forward window of ``window_days`` that holds at least ``min_size`` reviews
    of one rating flags the reviews in it, as in v1. Windows that share a review
    chain into one burst; windows that do not are reported separately (v1 merged
    everything of one rating into a single entry, which hid the gap between two
    distinct bursts). A genuine review posted inside a window is flagged with it:
    timing alone cannot tell them apart, which is why this is a signal for review.
    """
    by_rating: dict[int, list[int]] = defaultdict(list)
    for i, r in enumerate(reviews):
        by_rating[r.rating].append(i)

    bursts = []
    for rating in sorted(by_rating):
        ordered = sorted(by_rating[rating], key=lambda i: reviews[i].posted)  # stable, like v1
        pairs: list[tuple[int, int, float]] = []
        for a in ordered:
            anchor = reviews[a].posted
            window = [
                i
                for i in ordered
                if reviews[i].posted >= anchor and (reviews[i].posted - anchor).days <= window_days
            ]
            if len(window) >= min_size:
                pairs += [(a, i, 1.0) for i in window if i != a]
        for c in union_clusters(len(reviews), pairs):
            members = [i for i in ordered if i in c.members]
            dates = sorted(reviews[i].posted for i in members)
            bursts.append(
                RatingBurst(
                    rating=rating,
                    review_ids=tuple(reviews[i].id for i in members),
                    start=dates[0].isoformat(),
                    end=dates[-1].isoformat(),
                    span_days=(dates[-1] - dates[0]).days,
                )
            )
    return bursts


def rule_flags(
    reviews: Sequence[Review],
    *,
    dup_threshold: float = DUP_THRESHOLD,
    burst_min: int = BURST_MIN,
    burst_days: int = BURST_DAYS,
) -> list[Flag]:
    """Flags from the two rules, each with a score, a band and a written reason."""
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
                    "pairs": c.pair_count,
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
    return flags


def peak_jaccard(reviews: Sequence[Review], ids: set[ReviewId]) -> float:
    """Highest pairwise Jaccard similarity among the given reviews (the v1 panel figure)."""
    toks = [tokenize(r.text) for r in reviews if r.id in ids]
    peak = 0.0
    for i in range(len(toks)):
        for j in range(i + 1, len(toks)):
            peak = max(peak, jaccard(toks[i], toks[j]))
    return peak


def detect(
    reviews: Sequence[Review],
    *,
    dup_threshold: float = DUP_THRESHOLD,
    burst_min: int = BURST_MIN,
    burst_days: int = BURST_DAYS,
) -> DetectionResult:
    """Run the two rules only and return flags with reasons, plus the union of flagged ids."""
    flags = rule_flags(
        reviews, dup_threshold=dup_threshold, burst_min=burst_min, burst_days=burst_days
    )
    flagged: set[ReviewId] = set()
    for f in flags:
        flagged.update(f.review_ids)
    return DetectionResult(
        flags=flags,
        flagged_ids=flagged,
        peak_similarity=peak_jaccard(reviews, flagged),
        signals=["duplicate_phrasing", "rating_burst"],
    )


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
