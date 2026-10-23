"""Schemas the model must answer in. Validated before anything reaches the state.

These are the model facing shapes (Pydantic, so they become JSON schemas for structured
output). Nodes convert them into the dataclass records of :mod:`.state` and apply the
checks a schema cannot express (ids that exist, quotes that appear in the cited review).
Kept free of length and range constraints, which not every structured output backend
enforces; nodes check those in code.
"""

from __future__ import annotations

from typing import Literal

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
    aspect: Aspect = Field(description="The one aspect this theme is about.")
    sentiment: Polarity
    review_ids: list[str] = Field(
        description="Ids of every review that supports the theme; at least two."
    )
    evidence: list[str] = Field(
        description="Two or three short phrases copied exactly from the cited reviews."
    )


class AspectEvidenceOut(BaseModel):
    aspect: Aspect
    evidence: str = Field(description="One line saying what the reviews report on it.")
    review_ids: list[str] = Field(description="Ids of the reviews the line draws on.")


class SeriousIssueOut(BaseModel):
    kind: Literal["health", "safety", "stay cut short"]
    review_id: str
    quote: str = Field(description="The guest's words, copied exactly from that review.")


class Synthesis(BaseModel):
    themes: list[ThemeOut]
    aspect_evidence: list[AspectEvidenceOut]
    serious_issues: list[SeriousIssueOut]


class ActionOut(BaseModel):
    title: str = Field(description="Imperative, under twelve words.")
    why: str = Field(description="One sentence tying the action to what the reviews say.")
    impact: Level
    effort: Level
    review_ids: list[str] = Field(description="Ids of the reviews that support the action.")


class ActionList(BaseModel):
    actions: list[ActionOut]
