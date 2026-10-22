"""The operator's review of flags: what is asked, what is answered, what it changes.

Every flag goes to a person with its reason. The answer per flag is ``True`` (confirm: its
reviews stay out of the verified set) or ``False`` (dismiss: its reviews come back, unless
another flag still holds them). Until a flag is answered it holds its reviews, so the draft
brief shown alongside the questions is the cautious one.

What this cannot do: decide anything itself. It records decisions; it does not remove,
report or answer any review.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from review_synthesizer.authenticity import Flag, ReviewId

from .state import BriefState

REVIEW_NOTE = (
    "Each flag is a signal for review, not a verdict. Confirm keeps its reviews out of the "
    "verified rating and the synthesis; dismiss returns them."
)


def flag_key(flag: Flag) -> str:
    """Stable name for a flag within a run: signal and the reviews it covers."""
    return f"{flag.signal_type}:{','.join(str(i) for i in flag.review_ids)}"


def all_flags(state: BriefState) -> list[Flag]:
    """Detector flags first, then claim duplicate flags, as presented to the operator."""
    detection = state.get("detection")
    return [*(detection.flags if detection else []), *state.get("claim_flags", [])]


def held_ids(flags: Sequence[Flag], decisions: Mapping[str, bool]) -> set[ReviewId]:
    """Reviews covered by at least one flag that has not been dismissed."""
    held: set[ReviewId] = set()
    for f in flags:
        if decisions.get(flag_key(f)) is not False:
            held.update(f.review_ids)
    return held


def review_request(flags: Sequence[Flag], state: BriefState) -> dict[str, Any]:
    """What the operator is shown: every undecided flag with its reason, and the rating."""
    rating = state.get("rating")
    return {
        "note": REVIEW_NOTE,
        "flags": [
            {
                "key": flag_key(f),
                "signal_type": f.signal_type,
                "review_ids": list(f.review_ids),
                "score": f.score,
                "confidence": f.confidence,
                "rationale": f.rationale,
            }
            for f in flags
        ],
        "rating": None
        if rating is None
        else {"displayed": rating.displayed, "verified": rating.verified},
    }


def parse_decisions(answer: Any, expected: Sequence[str]) -> dict[str, bool]:
    """Check the operator's answer covers exactly the flags asked about, each True or False."""
    if not isinstance(answer, Mapping):
        raise TypeError("The review answer must map each flag key to true or false.")
    missing = [k for k in expected if k not in answer]
    unknown = [k for k in answer if k not in expected]
    not_bool = [k for k, v in answer.items() if k in expected and not isinstance(v, bool)]
    problems = []
    if missing:
        problems.append(f"no decision for {', '.join(missing)}")
    if unknown:
        problems.append(f"unknown flags {', '.join(map(str, unknown))}")
    if not_bool:
        problems.append(f"decision is not true or false for {', '.join(not_bool)}")
    if problems:
        raise ValueError("The review answer cannot be applied: " + "; ".join(problems) + ".")
    return {k: answer[k] for k in expected}
