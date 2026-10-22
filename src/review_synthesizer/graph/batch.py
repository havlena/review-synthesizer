"""Validate an uploaded batch of reviews before anything runs on it.

Checks: at least one review; every id present and unique; rating a whole number from 1
to 5; date present and in ISO form (``YYYY-MM-DD``); text present and not blank. Every
problem is collected, so an operator fixing a file sees all of them at once.

What this cannot do: tell whether a review is genuine, or whether a date is plausible.
It checks shape only; the detector and the operator do the rest.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from datetime import date
from typing import Any

from review_synthesizer.authenticity import Review

RATING_MIN = 1
RATING_MAX = 5
"""The 1 to 5 star scale of the capstone schema (``db/schema.sql``)."""


class BatchError(ValueError):
    """The batch cannot be processed. ``problems`` lists each one in operator terms."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__(f"{len(problems)} problem(s) in the batch: " + "; ".join(problems))


def _rating(value: Any) -> int | None:
    """Whole number 1 to 5, as an int or a string of digits (CSV); otherwise ``None``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        n = value
    elif isinstance(value, float) and value.is_integer():
        n = int(value)
    elif isinstance(value, str) and value.strip().isdigit():
        n = int(value.strip())
    else:
        return None
    return n if RATING_MIN <= n <= RATING_MAX else None


def _posted(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip())
        except ValueError:
            return None
    return None


def validate_batch(raw: Sequence[dict[str, Any]]) -> list[Review]:
    """Return the reviews as ``Review`` objects, or raise :class:`BatchError`.

    The date may be given as ``date`` or ``posted``. Fields other than id, text, rating,
    date and author are ignored.
    """
    if not raw:
        raise BatchError(["the batch has no reviews"])

    problems: list[str] = []
    ids = [r.get("id") for r in raw]
    repeated = {i for i, n in Counter(ids).items() if n > 1 and i is not None}

    reviews: list[Review] = []
    for row, r in enumerate(raw, start=1):
        rid = r.get("id")
        where = f"row {row}" if rid is None else f"review {rid} (row {row})"
        ok = True

        if rid is None or isinstance(rid, bool) or not isinstance(rid, int | str) or rid == "":
            problems.append(f"{where}: missing id")
            ok = False
        elif rid in repeated:
            problems.append(f"{where}: id {rid} appears more than once")
            ok = False

        rating = _rating(r.get("rating"))
        if rating is None:
            problems.append(
                f"{where}: rating {r.get('rating')!r} is not a whole number "
                f"from {RATING_MIN} to {RATING_MAX}"
            )
            ok = False

        raw_date = r.get("posted", r.get("date"))
        posted = _posted(raw_date)
        if posted is None:
            problems.append(f"{where}: date {raw_date!r} is missing or not YYYY-MM-DD")
            ok = False

        text = r.get("text")
        if not isinstance(text, str) or not text.strip():
            problems.append(f"{where}: review text is missing or blank")
            ok = False

        if ok:
            reviews.append(
                Review(id=rid, text=text, rating=rating, posted=posted, author=r.get("author"))
            )

    if problems:
        raise BatchError(problems)
    return reviews
