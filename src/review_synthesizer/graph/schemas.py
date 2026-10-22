"""Schemas the model must answer in. Validated before anything reaches the state.

These are the model facing shapes (Pydantic, so they become JSON schemas for structured
output). Nodes convert them into the dataclass records of :mod:`.state` and apply the
checks a schema cannot express (ids that exist, quotes that appear in the cited review).
Kept free of length and range constraints, which not every structured output backend
enforces; nodes check those in code.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .state import Aspect, Level, Polarity


class ClaimOut(BaseModel):
    aspect: Aspect
    polarity: Polarity
    specific: bool = Field(
        description="True when the claim names a concrete, checkable thing; false for "
        "general praise or complaint."
    )
    statement: str = Field(description="The claim in at most twelve plain words.")


class ExtractedClaims(BaseModel):
    """Every claim one review makes. An empty list is a valid answer."""

    claims: list[ClaimOut]


class PairVerdict(BaseModel):
    """Whether two reviews make the same specific claims about the same things."""

    same_claims: bool
    reason: str = Field(description="One sentence: the shared claims, or what differs.")


class ThemeOut(BaseModel):
    name: str = Field(description="Two to five words, for example 'Thin walls and noise'.")
    sentiment: Polarity
    review_ids: list[str] = Field(description="Ids of every review that supports the theme.")
    evidence: list[str] = Field(
        description="Two or three short phrases copied exactly from the cited reviews."
    )


class AspectScoreOut(BaseModel):
    aspect: Aspect
    score: int = Field(description="1 to 5 on the bands in the instructions.")
    review_ids: list[str] = Field(description="Ids of the reviews that speak to this aspect.")
    evidence: str = Field(description="One line saying what the reviews report.")


class Synthesis(BaseModel):
    themes: list[ThemeOut]
    aspect_scores: list[AspectScoreOut]


class ActionOut(BaseModel):
    title: str = Field(description="Imperative, under twelve words.")
    why: str = Field(description="One sentence tying the action to what the reviews say.")
    impact: Level
    effort: Level
    review_ids: list[str] = Field(description="Ids of the reviews that support the action.")


class ActionList(BaseModel):
    actions: list[ActionOut]
