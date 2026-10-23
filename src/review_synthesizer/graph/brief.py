"""Assemble the operator brief from a finished run, as data and as Markdown.

Four parts in the build plan's order: ranked themes, the aspect scorecard, the action list,
and the authenticity findings with the verified rating; then notes about the run and a
methods note (signals, thresholds, models, run id). Every number in the brief carries the
review ids behind it, so the operator UI can link each one to its evidence.

What this cannot do: add anything. It formats what the nodes produced and the operator
decided; it calls no model.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from datetime import date
from typing import Any

from review_synthesizer.authenticity import (
    BURST_DAYS,
    BURST_MIN,
    CLASSIFIER_THRESHOLD,
    DUP_THRESHOLD,
    EMBED_THRESHOLD,
    Flag,
)

from .approval import REVIEW_NOTE, all_flags, flag_key, held_ids
from .config import ModelConfig
from .duplicates import JUDGE_STYLE, MIN_SHARED_SPECIFIC
from .state import Brief, BriefState

GRAPH_SIGNALS = ("claim_duplicate",)
"""Signals the graph adds to whatever the detector ran."""

SCORECARD_NOTE = (
    "Scores are computed from the verified reviews' claims, not chosen by a model: each review "
    "takes one stance per aspect, and the share of negative stances (mixed counting half) sets "
    "the score, 0 is 5, up to a quarter 4, up to half 3, up to three quarters 2, above that 1. "
    "An aspect needs two reviews to be scored. Serious issues are listed below, not scored."
)


def _jsonable(value: Any) -> Any:
    """Plain JSON types only: dates as ISO strings, sets sorted, dataclasses as dicts."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, set | frozenset):
        return [_jsonable(v) for v in sorted(value, key=str)]
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, date):
        return value.isoformat()
    if hasattr(value, "item") and callable(value.item):  # numpy scalar
        return value.item()
    return value


def _thresholds(signals: Sequence[str]) -> dict[str, str]:
    known = {
        "duplicate_phrasing": f"Jaccard word overlap at or above {DUP_THRESHOLD:.2f}",
        "rating_burst": f"{BURST_MIN} or more same rating reviews within {BURST_DAYS} days",
        "embedding_similarity": f"cosine similarity at or above {EMBED_THRESHOLD:.2f}",
        "deceptive_language": f"classifier probability at or above {CLASSIFIER_THRESHOLD:.2f}",
        "claim_duplicate": (
            f"at least {MIN_SHARED_SPECIFIC} shared specific "
            f"claim{'' if MIN_SHARED_SPECIFIC == 1 else 's'}, then a model judgment per pair "
            f"({JUDGE_STYLE} prompt)"
        ),
    }
    return {s: known[s] for s in signals if s in known}


def _dedupe(items: Sequence[str]) -> list[str]:
    """Notes in first seen order; a rerun after a dismissal can repeat one."""
    return list(dict.fromkeys(items))


def _decision(flag: Flag, decisions: dict[str, bool]) -> str:
    value = decisions.get(flag_key(flag))
    return "confirmed" if value is True else "dismissed" if value is False else "undecided"


def build_data(
    state: BriefState, models: ModelConfig, embedder_name: str | None = None
) -> dict[str, Any]:
    decisions = state.get("decisions") or {}
    flags = all_flags(state)
    detection = state.get("detection")
    signals = [*(detection.signals if detection else []), *GRAPH_SIGNALS]
    rating = state.get("rating")
    model_names = {"main": models.main, "small": models.small}
    if embedder_name and "embedding_similarity" in signals:
        model_names["embeddings"] = embedder_name
    data = {
        "property": state.get("property"),
        "run_id": state.get("run_id"),
        "rating": rating,
        "themes": state.get("themes", []),
        "aspect_scores": state.get("aspect_scores", []),
        "serious_issues": state.get("serious_issues", []),
        "actions": state.get("actions", []),
        "authenticity": {
            "note": REVIEW_NOTE,
            "flags": [
                {
                    "key": flag_key(f),
                    "decision": _decision(f, decisions),
                    "signal_type": f.signal_type,
                    "review_ids": f.review_ids,
                    "score": f.score,
                    "confidence": f.confidence,
                    "rationale": f.rationale,
                    "details": f.details,
                }
                for f in flags
            ],
            "verified_review_ids": state.get("verified_ids", []),
            "held_review_ids": [
                r.id for r in state.get("reviews", []) if r.id in held_ids(flags, decisions)
            ],
        },
        "notes": _dedupe(state.get("notes", [])),
        "methods": {
            "signals": signals,
            "thresholds": _thresholds(signals),
            "models": model_names,
            "flag_decisions_by": state.get("approved_by"),
            "run_id": state.get("run_id"),
        },
    }
    return _jsonable(data)


def _cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _ids(ids: Sequence[Any]) -> str:
    return ", ".join(str(i) for i in ids)


def render_markdown(data: dict[str, Any]) -> str:
    prop = data["property"] or {}
    rating = data["rating"] or {}
    lines = [f"# Operator brief: {prop.get('name', 'unnamed property')}", ""]
    if data.get("run_id"):
        lines += [f"Run `{data['run_id']}`.", ""]

    lines += ["## Verified rating", ""]
    if rating.get("verified") is None:
        lines.append(
            f"No verified rating: none of the {rating.get('total', 0)} reviews passed. "
            f"Displayed rating {rating.get('displayed')}."
        )
    else:
        kept = rating["total"] - rating["removed"]
        lines.append(
            f"**{rating['verified']}** verified, from {kept} of {rating['total']} reviews "
            f"(displayed rating {rating['displayed']}). {rating['removed']} review(s) are held "
            "out by confirmed flags; see part 4."
        )

    lines += ["", "## 1. Themes", ""]
    if data["themes"]:
        lines += [
            "| Theme | Aspect | Sentiment | Reviews | Evidence |",
            "|---|---|---|---|---|",
        ]
        for t in data["themes"]:
            quotes = "; ".join(f'"{q}"' for q in t["evidence"]) or "none quoted"
            lines.append(
                f"| {_cell(t['name'])} | {t.get('aspect', '')} | {t['sentiment']} | "
                f"{t['count']} ({_ids(t['review_ids'])}) | {_cell(quotes)} |"
            )
    else:
        lines.append("No themes produced; see notes.")

    lines += ["", "## 2. Aspect scorecard", ""]
    if data["aspect_scores"]:
        lines += [
            SCORECARD_NOTE,
            "",
            "| Aspect | Score (1 to 5) | Positive / mixed / negative | Reviews | Evidence |",
            "|---|---|---|---|---|",
        ]
        for s in data["aspect_scores"]:
            score = "mentioned once, unscored" if s["score"] is None else s["score"]
            lines.append(
                f"| {s['aspect']} | {score} | {s['positive']} / {s['mixed']} / {s['negative']} | "
                f"{s['review_count']} ({_ids(s['review_ids'])}) | {_cell(s['evidence']) or '-'} |"
            )
    else:
        lines.append("No aspect scores produced; see notes.")

    issues = data.get("serious_issues", [])
    lines += ["", "### Serious issues reported", ""]
    if issues:
        lines.append("Listed, not scored: each is one guest's report and needs a look.")
        lines.append("")
        lines += [
            f'- {i["kind"].capitalize()}, review {i["review_id"]}: "{i["quote"]}"' for i in issues
        ]
    else:
        lines.append("None reported in the verified reviews.")

    lines += ["", "## 3. Actions", ""]
    if data["actions"]:
        for n, a in enumerate(data["actions"], start=1):
            lines.append(
                f"{n}. **{a['title']}** Impact {a['impact']}, effort {a['effort']}. "
                f"{a['why']} Reviews {_ids(a['review_ids'])}."
            )
    else:
        lines.append("No actions produced; see notes.")

    auth = data["authenticity"]
    lines += ["", "## 4. Authenticity findings", "", auth["note"], ""]
    if auth["flags"]:
        for f in auth["flags"]:
            lines.append(
                f"- **{f['decision'].capitalize()}**, {f['signal_type']}, reviews "
                f"{_ids(f['review_ids'])}, {f['confidence']} ({f['score']:.2f}). {f['rationale']}"
            )
    else:
        lines.append("No flags were raised.")

    if data["notes"]:
        lines += ["", "## Notes on this run", ""]
        lines += [f"- {n}" for n in data["notes"]]

    methods = data["methods"]
    lines += ["", "## Methods", ""]
    lines += [f"- Signals: {', '.join(methods['signals'])}."]
    lines += [f"- {s}: {t}." for s, t in methods["thresholds"].items()]
    lines += [f"- Models: {', '.join(f'{k} {v}' for k, v in methods['models'].items())}."]
    if methods.get("flag_decisions_by"):
        lines += [f"- Flag decisions: {methods['flag_decisions_by']}."]
    lines += [f"- Run id: {methods['run_id']}."]
    return "\n".join(lines) + "\n"


def build_brief(state: BriefState, models: ModelConfig, embedder_name: str | None = None) -> Brief:
    data = build_data(state, models, embedder_name)
    return Brief(data=data, markdown=render_markdown(data))
