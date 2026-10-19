"""Data types for the authenticity layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

SignalType = Literal["duplicate_phrasing", "rating_burst"]
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
    def from_dict(cls, d: dict[str, Any]) -> "Review":
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
    """Map a 0 to 1 score to the three bands used by the database schema."""
    if score >= 0.75:
        return "high"
    if score >= 0.5:
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
    """Highest pairwise similarity among the flagged reviews (shown on the v1 panel)."""


@dataclass(frozen=True)
class RatingSummary:
    total: int
    removed: int
    displayed: float
    """Mean rating over every review, rounded to one decimal."""
    verified: float | None
    """Mean rating over reviews that were not flagged; ``None`` if nothing passed."""
