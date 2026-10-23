"""Check and rank what the synthesis and action calls return.

The model's answer is validated against a schema first; these functions then apply the
checks a schema cannot: every cited review must be in the verified set, every quoted
phrase must appear in a cited review, a theme needs two verified reviews. What fails
is left out and counted in a note, never repaired by guessing. Ranking is done here, not
by the model, so the same items always come out in the same order.

What this cannot do: tell whether a theme or an action is a good reading of the reviews.
That is what the operator, and the end to end evaluation, are for.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from review_synthesizer.authenticity import Review, ReviewId

from .schemas import ActionOut, AspectEvidenceOut, SeriousIssueOut, ThemeOut
from .state import Action, SeriousIssue, Theme

MIN_THEME_REVIEWS = 2
"""Verified reviews a theme must cite; one review is an anecdote, not a theme. Design choice."""

MAX_THEMES = 8
"""Themes kept, largest first. A design choice for a one page brief, not measured."""

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
    """Themes about one aspect, each with at least two verified supporting reviews and quotes
    checked, ranked by supporting reviews, at most eight."""
    ids = _Ids(verified)
    themes: list[Theme] = []
    dropped_quotes = too_thin = 0
    for t in raw:
        review_ids = ids.resolve(t.review_ids)
        if len(review_ids) < MIN_THEME_REVIEWS or not t.name.strip():
            too_thin += 1
            continue
        texts = ids.texts(review_ids)
        evidence = [q.strip() for q in t.evidence if _quoted_in(q, texts)]
        dropped_quotes += len(t.evidence) - len(evidence)
        themes.append(
            Theme(t.name.strip(), len(review_ids), t.sentiment, evidence, review_ids, t.aspect)
        )
    themes.sort(key=lambda t: -t.count)  # stable: the model's order breaks ties
    extra = max(0, len(themes) - MAX_THEMES)

    notes = _unknown_note(ids.unknown, "themes")
    if too_thin:
        notes.append(
            f"{too_thin} theme(s) cited fewer than {MIN_THEME_REVIEWS} verified reviews and "
            "were left out."
        )
    if extra:
        notes.append(f"{extra} smaller theme(s) beyond the first {MAX_THEMES} were left out.")
    if dropped_quotes:
        notes.append(
            f"{dropped_quotes} quoted phrase(s) did not appear in the cited reviews and were "
            "left out of the theme evidence."
        )
    return themes[:MAX_THEMES], notes


def clean_aspect_evidence(
    raw: Sequence[AspectEvidenceOut], verified: Sequence[Review]
) -> tuple[dict[str, str], list[str]]:
    """One evidence line per scored aspect, kept only if it cites a verified review."""
    ids = _Ids(verified)
    lines: dict[str, str] = {}
    dropped = 0
    for e in raw:
        if e.aspect == "other" or e.aspect in lines or not ids.resolve(e.review_ids):
            dropped += 1
            continue
        lines[e.aspect] = e.evidence.strip()
    notes = _unknown_note(ids.unknown, "aspect evidence")
    if dropped:
        notes.append(
            f"{dropped} aspect evidence line(s) were left out: repeated, about 'other', or "
            "without a verified review."
        )
    return lines, notes


def clean_serious_issues(
    raw: Sequence[SeriousIssueOut], verified: Sequence[Review]
) -> tuple[list[SeriousIssue], list[str]]:
    """Serious issues whose review is verified and whose quote appears in it, once each."""
    ids = _Ids(verified)
    issues: list[SeriousIssue] = []
    dropped = 0
    for i in raw:
        found = ids.resolve([i.review_id])
        if not found or not _quoted_in(i.quote, ids.texts(found)):
            dropped += 1
            continue
        issue = SeriousIssue(i.kind, found[0], i.quote.strip())
        if issue not in issues:
            issues.append(issue)
    notes = _unknown_note(ids.unknown, "serious issues")
    if dropped:
        notes.append(
            f"{dropped} serious issue report(s) were left out: the review was not verified or "
            "the quote did not appear in it."
        )
    return issues, notes


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
