"""Data types for the authenticity layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

SignalType = Literal[
    "duplicate_phrasing",  # near duplicate wording (Jaccard over word sets)
    "rating_burst",  # same rating, dense window in time
    "embedding_similarity",  # near duplicate meaning (cosine over sentence embeddings)
    "deceptive_language",  # language resembles reviews written without a stay (classifier)
    "claim_duplicate",  # same specific claims in different words (language model, graph)
]
Confidence = Literal["low", "medium", "high"]

ReviewId = int | str


@dataclass(frozen=True)
class Review:
    """One guest review. Only ``text``, ``rating`` and ``posted`` feed detection."""

    id: ReviewId
    text: str
    rating: int
    posted: date
    author: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Review:
        posted = d.get("posted", d.get("date"))
        if isinstance(posted, str):
            posted = date.fromisoformat(posted)
        return cls(
            id=d["id"],
            text=d["text"],
            rating=int(d["rating"]),
            posted=posted,
            author=d.get("author"),
        )


def confidence_band(score: float) -> Confidence:
    """Map a 0 to 1 score to the three bands used by the database schema.

    For scores that are not a measurement against a threshold: the rating burst (count over
    twice the minimum size) and the claim duplicate share of pairs judged the same. Signals
    with a threshold use :func:`margin_band`.
    """
    if score >= 0.75:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"


MARGIN_BANDS = (0.25, 0.60)
"""Margin over a signal's threshold, as a share of the room above it, at which medium and
high start. Author decision after the Harbor Light end to end run (October 2026), where a
pair 0.001 over its threshold read as high; not tuned on data."""


def margin_band(score: float, threshold: float) -> Confidence:
    """Band from how far a score clears its signal's own threshold, not from the raw score.

    margin = (score - threshold) / (1 - threshold): 0 at the threshold, 1 at a perfect score.
    Low below 0.25, medium below 0.60, high at or above. A flag that only just clears its
    threshold is low, whatever the threshold is.
    """
    room = 1.0 - threshold
    # Rounded so a score exactly on a band edge is not pushed below it by float error.
    margin = round((score - threshold) / room, 9) if room > 0 else 1.0
    if margin >= MARGIN_BANDS[1]:
        return "high"
    if margin >= MARGIN_BANDS[0]:
        return "medium"
    return "low"


@dataclass
class Flag:
    """One authenticity signal covering one or more reviews, with its reasoning."""

    signal_type: SignalType
    review_ids: list[ReviewId]
    score: float
    confidence: Confidence
    rationale: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionResult:
    flags: list[Flag]
    flagged_ids: set[ReviewId]
    peak_similarity: float
    """Highest pairwise Jaccard similarity among the flagged reviews (shown on the v1 panel)."""
    signals: list[str] = field(default_factory=list)
    """Which signals ran, in order."""

    def flags_for(self, review_id: ReviewId) -> list[Flag]:
        return [f for f in self.flags if review_id in f.review_ids]


@dataclass(frozen=True)
class RatingSummary:
    total: int
    removed: int
    displayed: float
    """Mean rating over every review, rounded to one decimal."""
    verified: float | None
    """Mean rating over reviews that were not flagged; ``None`` if nothing passed."""
