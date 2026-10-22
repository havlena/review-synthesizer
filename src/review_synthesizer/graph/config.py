"""Model names and call settings for the graph, in one place.

Nodes never name a model: they ask the client for the main or the small model, and the
client takes the names from here. The environment overrides the defaults
(``ANTHROPIC_MODEL``, ``ANTHROPIC_MODEL_SMALL``).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

DEFAULT_MODEL = "claude-sonnet-5-5"
"""Main model for claim extraction and synthesis; the value in ``.env.example`` (step 5)."""

DEFAULT_MODEL_SMALL = "claude-haiku-5-5"
"""Small model for the pairwise claim duplicate judgments; the value in ``.env.example``."""

MAX_TOKENS = 16000
"""Output ceiling per call. Structured outputs are short; this only guards against cut offs."""

MAX_PARALLEL_CALLS = 8
"""Concurrent model calls within one node (per review, per pair). Not tuned; keeps a batch
of a few dozen reviews to seconds without approaching rate limits."""


@dataclass(frozen=True)
class ModelConfig:
    """Which models a run uses. Recorded in the brief's methods note."""

    main: str = DEFAULT_MODEL
    small: str = DEFAULT_MODEL_SMALL

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ModelConfig:
        env = os.environ if env is None else env
        return cls(
            main=env.get("ANTHROPIC_MODEL") or DEFAULT_MODEL,
            small=env.get("ANTHROPIC_MODEL_SMALL") or DEFAULT_MODEL_SMALL,
        )
