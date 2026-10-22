"""Tokenisation and similarity, ported from the capstone prototype."""

from __future__ import annotations

import re

STOP_WORDS: frozenset[str] = frozenset(
    [
        "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at", "for",
        "with", "is", "are", "was", "were", "be", "been", "it", "we", "our", "us",
        "you", "your", "they", "them", "this", "that", "i", "my", "me", "so", "all",
    ]
)
_NON_ALNUM = re.compile(r"[^a-z0-9\s]")


def tokenize(text: str) -> frozenset[str]:
    """Lower case, strip punctuation, drop stop words, return the set of remaining words."""
    cleaned = _NON_ALNUM.sub(" ", text.lower())
    return frozenset(w for w in cleaned.split() if w and w not in STOP_WORDS)


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard similarity of two token sets; 0.0 when both are empty."""
    inter = len(a & b)
    union = len(a) + len(b) - inter
    return inter / union if union else 0.0
