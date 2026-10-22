"""Check and rank what the synthesis and action calls return.

The model's answer is validated against a schema first; these functions then apply the
checks a schema cannot: every cited review must be in the verified set, every quoted
phrase must appear in a cited review, every score must be on the 1 to 5 scale. What fails
is left out and counted in a note, never repaired by guessing. Ranking is done here, not
by the model, so the same items always come out in the same order.

What this cannot do: tell whether a theme or an action is a good reading of the reviews.
That is what the operator, and the end to end evaluation, are for.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from review_synthesizer.authenticity import Review, ReviewId

from .schemas import ActionOut, AspectScoreOut, ThemeOut
from .state import Action, AspectScore, Theme

ASPECT_ORDER = (
    "cleanliness",
    "service",
    "breakfast",
    "room comfort",
    "value",
    "location",
    "noise",
    "parking",
    "wifi",
    "amenities",
    "other",
)
"""Scorecard order: the capstone scorecard's six aspects first, then the v2 additions."""

MAX_ACTIONS = 5
"""Length of the action list. A design choice for a one page brief, not measured."""

LEVEL_RANK = {"High": 3, "Medium": 2, "Low": 1}

_SPACE = re.compile(r"\s+")
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def _norm(text: str) -> str:
    return _SPACE.sub(" ", text.translate(_QUOTES).lower()).strip()


def _quoted_in(phrase: str, texts: Sequence[str]) -> bool:
    p = _norm(phrase).strip(" \"'.,;:!?")
    return bool(p) and any(p in _norm(t) for t in texts)


class _Ids:
    """Maps the model's string ids back to verified review ids, counting the misses."""

    def __init__(self, verified: Sequence[Review]) -> None:
        self.by_str = {str(r.id): r for r in verified}
        self.unknown = 0

    def resolve(self, ids: Sequence[str]) -> list[ReviewId]:
        out: list[ReviewId] = []
        for s in ids:
            r = self.by_str.get(str(s).strip())
            if r is None:
                self.unknown += 1
            elif r.id not in out:
                out.append(r.id)
        return out

    def texts(self, ids: Sequence[ReviewId]) -> list[str]:
        return [self.by_str[str(i)].text for i in ids]


def _unknown_note(n: int, where: str) -> list[str]:
    if not n:
        return []
    return [f"{n} review reference(s) in the {where} were not verified reviews and were left out."]


def clean_themes(
    raw: Sequence[ThemeOut], verified: Sequence[Review]
) -> tuple[list[Theme], list[str]]:
    """Themes with verified support only, quotes checked, ranked by supporting reviews."""
    ids = _Ids(verified)
    themes: list[Theme] = []
    dropped_quotes = dropped_themes = 0
    for t in raw:
        review_ids = ids.resolve(t.review_ids)
        if not review_ids or not t.name.strip():
            dropped_themes += 1
            continue
        texts = ids.texts(review_ids)
        evidence = [q.strip() for q in t.evidence if _quoted_in(q, texts)]
        dropped_quotes += len(t.evidence) - len(evidence)
        themes.append(Theme(t.name.strip(), len(review_ids), t.sentiment, evidence, review_ids))
    themes.sort(key=lambda t: -t.count)  # stable: the model's order breaks ties

    notes = _unknown_note(ids.unknown, "themes")
    if dropped_themes:
        notes.append(
            f"{dropped_themes} theme(s) had no verified supporting review and were left out."
        )
    if dropped_quotes:
        notes.append(
            f"{dropped_quotes} quoted phrase(s) did not appear in the cited reviews and were "
            "left out of the theme evidence."
        )
    return themes, notes


def clean_scores(
    raw: Sequence[AspectScoreOut], verified: Sequence[Review]
) -> tuple[list[AspectScore], list[str]]:
    """One score per aspect on the 1 to 5 scale, with verified support, in scorecard order."""
    ids = _Ids(verified)
    by_aspect: dict[str, AspectScore] = {}
    dropped = 0
    for s in raw:
        review_ids = ids.resolve(s.review_ids)
        if s.aspect in by_aspect or not 1 <= s.score <= 5 or not review_ids:
            dropped += 1
            continue
        by_aspect[s.aspect] = AspectScore(
            s.aspect, s.score, len(review_ids), s.evidence.strip(), review_ids
        )
    scores = [by_aspect[a] for a in ASPECT_ORDER if a in by_aspect]

    notes = _unknown_note(ids.unknown, "aspect scorecard")
    if dropped:
        notes.append(
            f"{dropped} aspect score(s) were left out: off the 1 to 5 scale, repeated, or "
            "without a verified supporting review."
        )
    return scores, notes


def clean_actions(
    raw: Sequence[ActionOut], verified: Sequence[Review]
) -> tuple[list[Action], list[str]]:
    """Actions with verified support, ranked by impact then lower effort, at most five."""
    ids = _Ids(verified)
    actions: list[Action] = []
    unsupported = 0
    for a in raw:
        review_ids = ids.resolve(a.review_ids)
        if not review_ids or not a.title.strip():
            unsupported += 1
            continue
        actions.append(Action(a.title.strip(), a.why.strip(), a.impact, a.effort, review_ids))
    # The capstone demo's rule: highest impact first, then the cheapest to do.
    actions.sort(key=lambda a: (-LEVEL_RANK[a.impact], LEVEL_RANK[a.effort]))

    notes = _unknown_note(ids.unknown, "actions")
    if unsupported:
        notes.append(f"{unsupported} action(s) cited no verified review and were left out.")
    if len(actions) > MAX_ACTIONS:
        notes.append(
            f"{len(actions) - MAX_ACTIONS} lower ranked action(s) were left out to keep the "
            f"list to {MAX_ACTIONS}."
        )
    return actions[:MAX_ACTIONS], notes
