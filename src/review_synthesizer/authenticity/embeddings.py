"""Embedding similarity: near duplicates by meaning rather than by shared words.

The word overlap rule catches copy and paste with light edits. A fake rewritten
with different vocabulary slips past it. Sentence embeddings place reviews with
the same content close together regardless of wording, so cosine similarity
between embeddings finds paraphrased duplicates. Same shape as the word rule:
pairs at or above a threshold join clusters, and each cluster becomes one flag
with a stated reason.

Default model: ``sentence-transformers/paraphrase-mpnet-base-v2`` (110M
parameters, about 420 MB, downloaded once). Chosen over the smaller
``all-MiniLM-L6-v2`` because at the 0.80 operating point it is the only model
tested that finds any rewritten duplicates with no false positives, and it keeps
the widest margin between planted and genuine reviews (evals/results). A batch
of a few hundred reviews embeds in seconds on a CPU, and nothing leaves the
machine. ``SMALL_MODEL`` is the lightweight alternative. Install with
``pip install -e .[embeddings]``.

What this signal can and cannot do, measured on three models: light rewrites
(a few words changed) score 0.93 and above and are caught cleanly; heavy
paraphrases that keep the claims and change every word score 0.55 to 0.81,
overlapping honest reviews on the same topics, so no threshold separates them.
Cosine similarity measures wording and topic, not the claims being made. Claim
level comparison belongs to the language model step.

Any object with ``name`` and ``embed(texts) -> (n, d) array of unit vectors``
works as an embedder, so the tests use a small fake and the API can swap in a
hosted model later without touching this module.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from .clustering import union_clusters
from .models import Review, ReviewId

DEFAULT_MODEL = "sentence-transformers/paraphrase-mpnet-base-v2"
SMALL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_THRESHOLD = 0.80
"""Near duplicate meaning: cosine similarity at or above this value. See evals/results."""


@runtime_checkable
class Embedder(Protocol):
    name: str

    def embed(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - protocol
        """Return one L2 normalised vector per text, shape ``(len(texts), dim)``."""
        ...


class SentenceTransformerEmbedder:
    """Local sentence-transformers model, loaded on first use."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None) -> None:
        self.name = model_name
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:  # pragma: no cover - depends on environment
                raise ImportError(
                    "sentence-transformers is not installed; run pip install -e .[embeddings]"
                ) from e
            self._model = SentenceTransformer(self.name, device=self.device)
        return self._model

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0))
        vectors = self._load().encode(
            list(texts), batch_size=64, convert_to_numpy=True, normalize_embeddings=True
        )
        return np.asarray(vectors, dtype=np.float32)


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """L2 normalise each row; zero rows stay zero."""
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity of unit vectors."""
    v = normalize_rows(vectors)
    return v @ v.T


def similar_pairs(
    reviews: Sequence[Review], embedder: Embedder, threshold: float = EMBED_THRESHOLD
) -> list[tuple[int, int, float]]:
    """Index pairs ``(i, j, cosine)`` at or above ``threshold``."""
    if len(reviews) < 2:
        return []
    sims = similarity_matrix(embedder.embed([r.text for r in reviews]))
    n = len(reviews)
    return [
        (i, j, float(sims[i, j]))
        for i in range(n)
        for j in range(i + 1, n)
        if sims[i, j] >= threshold
    ]


@dataclass(frozen=True)
class EmbeddingCluster:
    review_ids: tuple[ReviewId, ...]
    peak_similarity: float
    mean_similarity: float
    pair_count: int
    model: str


def find_embedding_clusters(
    reviews: Sequence[Review], embedder: Embedder, threshold: float = EMBED_THRESHOLD
) -> list[EmbeddingCluster]:
    """Group reviews joined by near duplicate meaning (transitively) into clusters."""
    return [
        EmbeddingCluster(
            review_ids=tuple(reviews[i].id for i in c.members),
            peak_similarity=c.peak,
            mean_similarity=c.mean,
            pair_count=c.pair_count,
            model=embedder.name,
        )
        for c in union_clusters(len(reviews), similar_pairs(reviews, embedder, threshold))
    ]
