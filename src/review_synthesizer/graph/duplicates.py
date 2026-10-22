"""Claim duplicates: reviews that make the same specific claims in different words.

The detector's duplicate signals compare wording (Jaccard) and meaning (embeddings); the
paraphrase evaluation showed neither separates a heavy rewrite from an honest review on the
same topic (``docs/evaluation.md``). This signal compares claims instead:

1. candidate pairs are reviews sharing at least ``MIN_SHARED_SPECIFIC`` specific claims by
   aspect and polarity (a cheap filter on the extracted claims, no model call);
2. each candidate pair gets one judgment from the small model, with a one sentence reason;
3. pairs judged the same are joined into groups, and each group becomes one flag.

Pairs the detector already joined by wording or meaning are not judged again.

What this cannot do: catch rewrites that share no specific claim (generic praise campaigns
carry none; the rating burst signal is the one for those), or recover from a failed claim
extraction. On the paraphrase set it finds about four in ten rewritten pairs, and which
ones moves between runs (``evals/results/claim_duplicates_ablation.md``).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from review_synthesizer.authenticity import DetectionResult, Flag, Review, ReviewId
from review_synthesizer.authenticity.clustering import union_clusters
from review_synthesizer.authenticity.models import Confidence

from .state import Claim, PairJudgment

MIN_SHARED_SPECIFIC = 2
"""Specific claims (same aspect and polarity) two reviews must share to be judged. An author
decision (October 2026) that overrides the ablation's recall rule, which twice chose 1 with
the lenient judge: the production prompt joined genuine Harbor Light reviews 10 and 16 under
that setting in both end to end runs, and a 190 pair control cannot bound a judge's wrong
join rate at the volume 1 produces (18% of honest pairs; 1.3% at 2, fourteen times fewer
judgments). Recall given up: two to three of 18 synthetic pairs. See docs/evaluation.md and
evals/results/claim_duplicates_ablation.md. Not tuned further in step 5."""

JUDGE_STYLE = "strict"
"""Which judge prompt the product uses (``prompts.JUDGE_SYSTEMS``). Set with
``MIN_SHARED_SPECIFIC`` by the same author decision: the lenient prompt's verdict on the
same pair moved with the property line in the prompt, and the strict prompt never joined a
control pair. Measured at this default in evals/results/claim_duplicates_default.md."""

DETECTOR_DUPLICATE_SIGNALS = ("duplicate_phrasing", "embedding_similarity")
"""Detector signals whose pairs are already joined, so the model is not asked about them."""


def shared_specific(a: Sequence[Claim], b: Sequence[Claim]) -> int:
    """Number of specific claims the two lists share by aspect and polarity (multiset)."""
    ka = Counter((c.aspect, c.polarity) for c in a if c.specific)
    kb = Counter((c.aspect, c.polarity) for c in b if c.specific)
    return sum((ka & kb).values())


def detector_joined_pairs(
    reviews: Sequence[Review], detection: DetectionResult
) -> set[tuple[ReviewId, ReviewId]]:
    """Every pair inside a detector duplicate flag, as (earlier id, later id) in batch order."""
    order = {r.id: i for i, r in enumerate(reviews)}
    joined: set[tuple[ReviewId, ReviewId]] = set()
    for flag in detection.flags:
        if flag.signal_type not in DETECTOR_DUPLICATE_SIGNALS:
            continue
        ids = sorted(flag.review_ids, key=order.__getitem__)
        joined.update((a, b) for k, a in enumerate(ids) for b in ids[k + 1 :])
    return joined


def candidate_pairs(
    reviews: Sequence[Review],
    claims: Mapping[ReviewId, Sequence[Claim]],
    detection: DetectionResult,
    *,
    min_shared: int = MIN_SHARED_SPECIFIC,
) -> list[tuple[Review, Review]]:
    """Pairs worth a model judgment, in batch order."""
    skip = detector_joined_pairs(reviews, detection)
    out = []
    for i, a in enumerate(reviews):
        for b in reviews[i + 1 :]:
            if (a.id, b.id) in skip:
                continue
            if shared_specific(claims.get(a.id, []), claims.get(b.id, [])) >= min_shared:
                out.append((a, b))
    return out


def claim_band(mean_shared: float) -> Confidence:
    """Band from the specific claims behind a join: 1 shared is low, 2 medium, 3 or more high.

    One shared specific claim is weak evidence: honest guests of one property share one
    readily (the Harbor Light run joined two genuine reviews on the location alone).
    """
    if mean_shared >= 3:
        return "high"
    if mean_shared >= 2:
        return "medium"
    return "low"


def claim_flags(
    reviews: Sequence[Review],
    judgments: Sequence[PairJudgment],
    model: str,
    claims: Mapping[ReviewId, Sequence[Claim]],
) -> list[Flag]:
    """One flag per group of reviews joined by pairs judged to make the same claims.

    Score and band rest on the evidence: the mean number of specific claims shared by the
    pairs judged the same, banded by :func:`claim_band`; the score is that mean over 3,
    capped at 1, to stay on the 0 to 1 scale of the other signals.
    """
    index = {r.id: i for i, r in enumerate(reviews)}
    same = [j for j in judgments if j.same_claims]
    clusters = union_clusters(len(reviews), [(index[j.a], index[j.b], 1.0) for j in same])

    flags = []
    for c in clusters:
        ids = [reviews[i].id for i in c.members]
        members = set(ids)
        inside = [j for j in judgments if j.a in members and j.b in members]
        judged_same = [j for j in inside if j.same_claims]
        member_pairs = len(ids) * (len(ids) - 1) // 2
        shared = [shared_specific(claims.get(j.a, []), claims.get(j.b, [])) for j in judged_same]
        mean_shared = sum(shared) / len(shared)
        example = judged_same[0]
        flags.append(
            Flag(
                signal_type="claim_duplicate",
                review_ids=ids,
                score=min(mean_shared / 3, 1.0),
                confidence=claim_band(mean_shared),
                rationale=(
                    f"{len(ids)} reviews make the same specific claims in different words: "
                    f"{len(judged_same)} of {member_pairs} pairs in the group were judged to "
                    f"describe the same things, sharing {mean_shared:.1f} specific "
                    f"claim{'' if mean_shared == 1 else 's'} on average, none of them already "
                    f"joined by the detector's wording or meaning signals. Reviews {example.a} "
                    f"and {example.b}: {example.reason}"
                ),
                details={
                    "pairs_judged_same": len(judged_same),
                    "member_pairs": member_pairs,
                    "mean_shared_specific": mean_shared,
                    "shared_specific_per_pair": [
                        {"a": j.a, "b": j.b, "shared": s}
                        for j, s in zip(judged_same, shared, strict=True)
                    ],
                    "min_shared_specific": MIN_SHARED_SPECIFIC,
                    "model": model,
                    "judgments": [
                        {"a": j.a, "b": j.b, "same_claims": j.same_claims, "reason": j.reason}
                        for j in inside
                    ],
                },
            )
        )
    return flags
