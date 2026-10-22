"""The composite detector: every authenticity signal through one call.

Signals, in the order they run:

1. ``duplicate_phrasing``: near duplicate wording (rules, always on)
2. ``rating_burst``: same rating, dense window in time (rules, always on)
3. ``embedding_similarity``: near duplicate meaning (when an embedder is given)
4. ``deceptive_language``: language resembling reviews written without a stay
   (when a classifier is given; flags only above a conservative threshold)

Each flag carries a score, a confidence band and a written rationale. The union
of flagged ids drives the verified rating. With no embedder and no classifier
the result is identical to the capstone rules, which the golden tests check.

Dependencies are injected rather than imported here, so this module stays
importable without scikit-learn or sentence-transformers installed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from .embeddings import EMBED_THRESHOLD, Embedder, find_embedding_clusters
from .models import DetectionResult, Flag, Review, ReviewId, confidence_band
from .rules import (
    BURST_DAYS,
    BURST_MIN,
    DUP_THRESHOLD,
    duplicate_pairs,
    peak_jaccard,
    rule_flags,
)

CLASSIFIER_THRESHOLD = 0.80
"""Flag a review's language only at or above this probability. See evals/results."""


@runtime_checkable
class Classifier(Protocol):
    def predict_proba(self, texts: Sequence[str]) -> Any:  # pragma: no cover - protocol
        """Probability of deceptive language per text, shape ``(n,)``."""
        ...


def embedding_flags(
    reviews: Sequence[Review],
    embedder: Embedder,
    *,
    threshold: float = EMBED_THRESHOLD,
    dup_threshold: float = DUP_THRESHOLD,
) -> list[Flag]:
    """Clusters of near duplicate meaning, noting how many pairs word overlap missed."""
    index_of = {r.id: i for i, r in enumerate(reviews)}
    word_pairs = {(i, j) for i, j, _ in duplicate_pairs(reviews, dup_threshold)}
    flags: list[Flag] = []
    for c in find_embedding_clusters(reviews, embedder, threshold):
        idx = sorted(index_of[rid] for rid in c.review_ids)
        member_pairs = [(a, b) for k, a in enumerate(idx) for b in idx[k + 1 :]]
        new_pairs = sum(1 for p in member_pairs if p not in word_pairs)
        score = c.peak_similarity
        flags.append(
            Flag(
                signal_type="embedding_similarity",
                review_ids=list(c.review_ids),
                score=score,
                confidence=confidence_band(score),
                rationale=(
                    f"{len(c.review_ids)} reviews share near duplicate meaning: peak cosine "
                    f"similarity {c.peak_similarity:.2f} (mean {c.mean_similarity:.2f}) against a "
                    f"threshold of {threshold:.2f}; {new_pairs} of {len(member_pairs)} pairs in "
                    f"the group were not caught by word overlap."
                ),
                details={
                    "peak_similarity": c.peak_similarity,
                    "mean_similarity": c.mean_similarity,
                    "pairs": c.pair_count,
                    "pairs_missed_by_word_overlap": new_pairs,
                    "threshold": threshold,
                    "model": c.model,
                },
            )
        )
    return flags


def classifier_flags(
    reviews: Sequence[Review],
    classifier: Classifier,
    *,
    threshold: float = CLASSIFIER_THRESHOLD,
    k: int = 3,
) -> list[Flag]:
    """One flag per review whose language scores at or above ``threshold``."""
    if not reviews:
        return []
    probs = classifier.predict_proba([r.text for r in reviews])
    explain = getattr(classifier, "explain", None)
    flags: list[Flag] = []
    for review, p in zip(reviews, probs):
        p = float(p)
        if p < threshold:
            continue
        details: dict[str, Any] = {"probability": p, "threshold": threshold}
        cue_text = ""
        if explain is not None:
            exp = explain(review.text, k=k)
            up = [t.term for t in exp.toward_deceptive]
            down = [t.term for t in exp.toward_truthful]
            details["toward_deceptive"] = up
            details["toward_truthful"] = down
            cue_text = (
                f" Cues toward deceptive: {', '.join(repr(t) for t in up) or 'none'};"
                f" toward truthful: {', '.join(repr(t) for t in down) or 'none'}."
            )
        flags.append(
            Flag(
                signal_type="deceptive_language",
                review_ids=[review.id],
                score=p,
                confidence=confidence_band(p),
                rationale=(
                    f"Language resembles reviews written without a stay: p = {p:.2f} "
                    f"(threshold {threshold:.2f}).{cue_text}"
                ),
                details=details,
            )
        )
    return flags


def detect(
    reviews: Sequence[Review],
    *,
    embedder: Embedder | None = None,
    classifier: Classifier | None = None,
    dup_threshold: float = DUP_THRESHOLD,
    burst_min: int = BURST_MIN,
    burst_days: int = BURST_DAYS,
    embed_threshold: float = EMBED_THRESHOLD,
    classifier_threshold: float = CLASSIFIER_THRESHOLD,
) -> DetectionResult:
    """Run every available signal and return all flags plus the union of flagged ids."""
    flags = rule_flags(
        reviews, dup_threshold=dup_threshold, burst_min=burst_min, burst_days=burst_days
    )
    signals = ["duplicate_phrasing", "rating_burst"]
    if embedder is not None:
        flags += embedding_flags(
            reviews, embedder, threshold=embed_threshold, dup_threshold=dup_threshold
        )
        signals.append("embedding_similarity")
    if classifier is not None:
        flags += classifier_flags(reviews, classifier, threshold=classifier_threshold)
        signals.append("deceptive_language")

    flagged: set[ReviewId] = set()
    for f in flags:
        flagged.update(f.review_ids)
    return DetectionResult(
        flags=flags,
        flagged_ids=flagged,
        peak_similarity=peak_jaccard(reviews, flagged),
        signals=signals,
    )
