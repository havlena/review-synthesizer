"""The aspect scorecard, computed from the extracted claims of verified reviews.

The model does not choose the scores. For each aspect (except ``other``), every verified
review with a claim on it takes one stance: positive or negative if all its claims on the
aspect agree, mixed otherwise. The score follows from the share of negative stances, mixed
counting half:

    share 0           5
    up to 1/4         4
    up to 1/2         3
    up to 3/4         2
    above 3/4         1

An aspect needs at least two reviews to be scored; with one it is listed as mentioned once,
unscored. The bands are a design choice in the style of the capstone's anchored rubric, not
measured against anything. A serious problem (health, safety, a stay cut short) does not
change a score; it is listed on its own (``SeriousIssue``).

What this cannot do: weigh how bad a complaint is, or notice a claim extraction missed. The
score is only as good as the claims behind it, and the counts are shown so an operator can
see what it rests on.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from review_synthesizer.authenticity import Review, ReviewId

from .state import Aspect, AspectScore, Claim, Polarity

SCORED_ASPECTS: tuple[Aspect, ...] = (
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
)
"""Scorecard order: the capstone scorecard's six aspects first, then the v2 additions.
``other`` is never scored: it is not one thing."""

MIN_REVIEWS_TO_SCORE = 2
"""Reviews an aspect needs to be scored; one review is an anecdote. Design choice."""

SCORE_BANDS = ((0.0, 5), (0.25, 4), (0.5, 3), (0.75, 2))
"""(largest negative share, score) in order; above the last, 1. Design choice, not measured."""


def stance(polarities: Sequence[Polarity]) -> Polarity:
    """One review's stance on an aspect from its claims on it."""
    found = set(polarities)
    if found == {"positive"}:
        return "positive"
    if found == {"negative"}:
        return "negative"
    return "mixed"


def score_for(negative: int, mixed: int, total: int) -> int:
    """Score from the share of negative stances, mixed counting half."""
    share = (negative + 0.5 * mixed) / total
    for limit, score in SCORE_BANDS:
        if share <= limit + 1e-9:
            return score
    return 1


def compute_scorecard(
    verified: Sequence[Review],
    claims: Mapping[ReviewId, Sequence[Claim]],
    evidence: Mapping[str, str] | None = None,
) -> list[AspectScore]:
    """Scores for every aspect at least one verified review has a claim on, in fixed order.

    ``evidence`` maps an aspect to the model's one line of evidence, already checked.
    """
    evidence = evidence or {}
    out = []
    for aspect in SCORED_ASPECTS:
        stances: dict[ReviewId, Polarity] = {}
        for r in verified:
            pol = [c.polarity for c in claims.get(r.id, []) if c.aspect == aspect]
            if pol:
                stances[r.id] = stance(pol)
        if not stances:
            continue
        counts = {
            p: sum(1 for s in stances.values() if s == p) for p in ("positive", "mixed", "negative")
        }
        n = len(stances)
        out.append(
            AspectScore(
                aspect=aspect,
                score=score_for(counts["negative"], counts["mixed"], n)
                if n >= MIN_REVIEWS_TO_SCORE
                else None,
                review_count=n,
                evidence=evidence.get(aspect, ""),
                review_ids=list(stances),
                positive=counts["positive"],
                mixed=counts["mixed"],
                negative=counts["negative"],
            )
        )
    return out
