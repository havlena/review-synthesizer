"""The graph's state: one typed dictionary of parsed, validated records.

Records are dataclasses, like the detector's. Model output is validated against a schema
in :mod:`.model` before a node turns it into one of these, so no raw model text is
stored. The state carries no client, key or connection, so a checkpoint can be saved and
resumed as it is.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypedDict

from review_synthesizer.authenticity import DetectionResult, Flag, RatingSummary, Review, ReviewId

Aspect = Literal[
    "cleanliness",
    "service",
    "breakfast",
    "room comfort",
    "value",
    "location",
    "noise",
    "parking",
    "wifi",
    "amenities",
    "other",
]
Polarity = Literal["positive", "negative", "mixed"]
Level = Literal["High", "Medium", "Low"]


@dataclass(frozen=True)
class Property:
    """The property the batch is about. Fictional in every fixture and demo."""

    name: str
    type: str = "inn"
    region: str = ""


@dataclass(frozen=True)
class Claim:
    """One thing a review asserts about one aspect of the stay."""

    aspect: Aspect
    polarity: Polarity
    specific: bool
    """Names a concrete thing: a view, a dish, a price, a fixture."""
    statement: str
    """Short canonical statement, so the same claim in other words reads the same."""


@dataclass(frozen=True)
class PairJudgment:
    """The small model's answer for one candidate pair of reviews."""

    a: ReviewId
    b: ReviewId
    same_claims: bool
    reason: str


@dataclass(frozen=True)
class Theme:
    name: str
    count: int
    sentiment: Polarity
    evidence: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AspectScore:
    aspect: Aspect
    score: float
    review_count: int
    evidence: str


@dataclass(frozen=True)
class Action:
    title: str
    why: str
    impact: Level
    effort: Level
    review_ids: list[ReviewId] = field(default_factory=list)


@dataclass(frozen=True)
class Brief:
    """The final output: the same content as data and as Markdown."""

    data: dict[str, Any]
    markdown: str


class BriefState(TypedDict, total=False):
    """Everything a run knows. Each node returns only the keys it changes."""

    property: Property
    raw_reviews: list[dict[str, Any]]
    """Input as given; ``load_batch`` validates it into ``reviews``."""
    reviews: list[Review]
    detection: DetectionResult
    claims: dict[ReviewId, list[Claim]]
    claim_judgments: list[PairJudgment]
    claim_flags: list[Flag]
    verified_ids: list[ReviewId]
    rating: RatingSummary
    themes: list[Theme]
    aspect_scores: list[AspectScore]
    actions: list[Action]
    decisions: dict[str, bool]
    """Operator decision per flag key: ``True`` confirmed, ``False`` dismissed."""
    brief: Brief
    notes: Annotated[list[str], operator.add]
    """Things the operator should know about the run itself (an extraction that failed).
    Appended to, never replaced, so parallel branches can both write."""
