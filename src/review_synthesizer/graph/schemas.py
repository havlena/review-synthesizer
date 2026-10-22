"""Schemas the model must answer in. Validated before anything reaches the state.

These are the model facing shapes (Pydantic, so they become JSON schemas for structured
output). Nodes convert them into the dataclass records of :mod:`.state` and apply the
checks a schema cannot express (ids that exist, quotes that appear in the cited review).
Kept free of length and range constraints, which not every structured output backend
enforces; nodes check those in code.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .state import Aspect, Polarity


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
