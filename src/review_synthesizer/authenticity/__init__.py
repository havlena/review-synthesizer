"""Authenticity layer: signals that a set of reviews may be coordinated or manipulated.

Everything here is a signal for human review, not a verdict. The rule based
layer in :mod:`rules` is a faithful port of the capstone prototype (July 2026)
and reproduces its results exactly on the original synthetic data set.
"""

from .models import DetectionResult, Flag, RatingSummary, Review
from .rules import (
    BURST_DAYS,
    BURST_MIN,
    DUP_THRESHOLD,
    detect,
    find_duplicate_clusters,
    find_rating_bursts,
    rating_summary,
)
from .text import jaccard, tokenize

__all__ = [
    "BURST_DAYS",
    "BURST_MIN",
    "DUP_THRESHOLD",
    "DetectionResult",
    "Flag",
    "RatingSummary",
    "Review",
    "detect",
    "find_duplicate_clusters",
    "find_rating_bursts",
    "jaccard",
    "rating_summary",
    "tokenize",
]
