"""Authenticity layer: signals that a set of reviews may be coordinated or manipulated.

Everything here is a signal for human review, not a verdict. Four signals:

* ``duplicate_phrasing`` and ``rating_burst``: the capstone rules (``rules``),
  reproduced exactly on the original synthetic data set;
* ``embedding_similarity``: paraphrased duplicates by sentence embedding
  (``embeddings``, optional dependency);
* ``deceptive_language``: a classifier trained on labelled hotel reviews
  (``classifier``, optional dependency).

:func:`detect` runs whichever of them it is given.
"""

from .detector import CLASSIFIER_THRESHOLD, classifier_flags, detect, embedding_flags
from .embeddings import (
    DEFAULT_MODEL,
    EMBED_THRESHOLD,
    SMALL_MODEL,
    Embedder,
    SentenceTransformerEmbedder,
    find_embedding_clusters,
    similar_pairs,
)
from .models import DetectionResult, Flag, RatingSummary, Review, ReviewId, confidence_band
from .rules import (
    BURST_DAYS,
    BURST_MIN,
    DUP_THRESHOLD,
    duplicate_pairs,
    find_duplicate_clusters,
    find_rating_bursts,
    rating_summary,
    rule_flags,
)
from .text import jaccard, tokenize

__all__ = [
    "BURST_DAYS",
    "BURST_MIN",
    "CLASSIFIER_THRESHOLD",
    "DEFAULT_MODEL",
    "DUP_THRESHOLD",
    "EMBED_THRESHOLD",
    "SMALL_MODEL",
    "DetectionResult",
    "Embedder",
    "Flag",
    "RatingSummary",
    "Review",
    "ReviewId",
    "SentenceTransformerEmbedder",
    "classifier_flags",
    "confidence_band",
    "detect",
    "duplicate_pairs",
    "embedding_flags",
    "find_duplicate_clusters",
    "find_embedding_clusters",
    "find_rating_bursts",
    "jaccard",
    "rating_summary",
    "rule_flags",
    "similar_pairs",
    "tokenize",
]
