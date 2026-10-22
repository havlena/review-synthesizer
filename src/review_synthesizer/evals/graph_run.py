"""Report on end to end graph runs over the Harbor Light Inn fixture.

Takes the briefs from repeated runs (with every flag confirmed automatically) and states:
every flag with its signal, score, band and reviews; whether a genuine review was flagged,
and by which signal; the displayed and verified ratings against the golden 3.8 and 3.5; the
notes the output checks produced; what differed between runs; and the tokens spent.

The fixture's ``intent`` labels (genuine or planted) are ground truth for this report only;
nothing in the graph reads them.

What this cannot do: say whether the themes and actions are a good reading of the reviews.
It shows what changed between runs and what the checks dropped; judging quality is the
operator's job and a later golden set's.
"""

from __future__ import annotations

import datetime as _dt
import json
import platform
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from .claims import PRICES_AS_OF, PRICES_PER_MTOK

GOLDEN_DISPLAYED = 3.8
"""Displayed rating of the capstone demo on Harbor Light Inn (tests/test_rules.py)."""

GOLDEN_VERIFIED = 3.5
"""Verified rating of the capstone demo once the five planted reviews are set aside."""


def _usd(usage: Mapping[str, Mapping[str, int]]) -> float | None:
    total = 0.0
    for model, u in usage.items():
        if model not in PRICES_PER_MTOK:
            return None
        p_in, p_out = PRICES_PER_MTOK[model]
        total += u["input_tokens"] / 1e6 * p_in + u["output_tokens"] / 1e6 * p_out
    return total


def _run(k: int, data: dict[str, Any], intent: dict[Any, str], usage) -> dict[str, Any]:
    flags = data["authenticity"]["flags"]
    genuine: dict[Any, list[str]] = {}
    for f in flags:
        for rid in f["review_ids"]:
            if intent.get(rid) == "genuine":
                genuine.setdefault(rid, []).append(f["signal_type"])
    rating = data["rating"]
    return {
        "run": k,
        "run_id": data["run_id"],
        "rating": rating,
        "matches_golden": rating["displayed"] == GOLDEN_DISPLAYED
        and rating["verified"] == GOLDEN_VERIFIED,
        "flags": [
            {
                "key": f["key"],
                "signal_type": f["signal_type"],
                "score": f["score"],
                "confidence": f["confidence"],
                "review_ids": f["review_ids"],
                "decision": f["decision"],
            }
            for f in flags
        ],
        "genuine_flagged": [
            {"review_id": rid, "signals": sigs} for rid, sigs in sorted(genuine.items(), key=str)
        ],
        "notes": data["notes"],
        "themes": [
            {"name": t["name"], "count": t["count"], "sentiment": t["sentiment"]}
            for t in data["themes"]
        ],
        "scores": {s["aspect"]: s["score"] for s in data["aspect_scores"]},
        "actions": [
            {"title": a["title"], "impact": a["impact"], "effort": a["effort"]}
            for a in data["actions"]
        ],
        "usage": dict(usage),
        "usd": _usd(usage),
    }


def _only(a: Sequence[str], b: Sequence[str]) -> dict[str, list[str]]:
    return {"run 1 only": [x for x in a if x not in b], "run 2 only": [x for x in b if x not in a]}


def _differences(r1: dict[str, Any], r2: dict[str, Any]) -> dict[str, Any]:
    t1 = {t["name"]: t for t in r1["themes"]}
    t2 = {t["name"]: t for t in r2["themes"]}
    a1 = [a["title"] for a in r1["actions"]]
    a2 = [a["title"] for a in r2["actions"]]
    return {
        "flags": _only([f["key"] for f in r1["flags"]], [f["key"] for f in r2["flags"]]),
        "themes": _only(list(t1), list(t2)),
        "theme_changes": [
            {
                "name": n,
                "run 1": [t1[n]["count"], t1[n]["sentiment"]],
                "run 2": [t2[n]["count"], t2[n]["sentiment"]],
            }
            for n in t1
            if n in t2
            and (t1[n]["count"], t1[n]["sentiment"]) != (t2[n]["count"], t2[n]["sentiment"])
        ],
        "scores": [
            {"aspect": a, "run 1": r1["scores"].get(a), "run 2": r2["scores"].get(a)}
            for a in dict.fromkeys([*r1["scores"], *r2["scores"]])
            if r1["scores"].get(a) != r2["scores"].get(a)
        ],
        "actions": _only(a1, a2),
        "same_action_order": a1 == a2,
    }


@dataclass
class GraphRunReport:
    created_at: str
    python_version: str
    property: str
    reviews: int
    planted: list[Any]
    methods: dict[str, Any]
    golden: dict[str, float]
    runs: list[dict[str, Any]]
    differences: dict[str, Any] | None
    tokens: dict[str, Any]
    limits: list[str] = field(default_factory=list)

    @property
    def headline(self) -> dict[str, Any]:
        return {
            "verified_ratings": [r["rating"]["verified"] for r in self.runs],
            "displayed_ratings": [r["rating"]["displayed"] for r in self.runs],
            "all_runs_match_golden": all(r["matches_golden"] for r in self.runs),
            "genuine_reviews_flagged": sorted(
                {g["review_id"] for r in self.runs for g in r["genuine_flagged"]}, key=str
            ),
            "flags_per_run": [len(r["flags"]) for r in self.runs],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "headline": self.headline}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_markdown(self) -> str:
        h = self.headline
        g = self.golden
        lines = [
            f"# End to end graph run: {self.property}",
            "",
            (
                f"Generated {self.created_at} on Python {self.python_version}. {self.reviews} reviews, "
                f"{len(self.planted)} of them planted ({', '.join(map(str, self.planted))}). "
                f"{len(self.runs)} runs of the whole graph. Brief of run 1: brief_harbor_light.md."
            ),
            "",
            "## Headline",
            "",
            "| | " + " | ".join(f"Run {r['run']}" for r in self.runs) + " | Golden |",
            "|---|" + "---|" * len(self.runs) + "---|",
            "| Displayed rating | "
            + " | ".join(str(x) for x in h["displayed_ratings"])
            + f" | {g['displayed']} |",
            "| Verified rating | "
            + " | ".join(str(x) for x in h["verified_ratings"])
            + f" | {g['verified']} |",
            "| Flags | " + " | ".join(str(x) for x in h["flags_per_run"]) + " | |",
            "| Genuine reviews flagged | "
            + " | ".join(
                ", ".join(
                    f"{x['review_id']} ({', '.join(x['signals'])})" for x in r["genuine_flagged"]
                )
                or "none"
                for r in self.runs
            )
            + " | none |",
            "",
            (
                "Every run reproduces the golden ratings."
                if h["all_runs_match_golden"]
                else "**At least one run does not reproduce the golden ratings.**"
            )
            + (
                ""
                if not h["genuine_reviews_flagged"]
                else " **Genuine reviews were flagged: "
                + ", ".join(map(str, h["genuine_reviews_flagged"]))
                + ".**"
            ),
        ]
        for r in self.runs:
            lines += [
                "",
                f"## Run {r['run']} (`{r['run_id']}`)",
                "",
                "| Signal | Score | Band | Reviews | Decision |",
                "|---|---|---|---|---|",
            ]
            for f in r["flags"]:
                lines.append(
                    f"| {f['signal_type']} | {f['score']:.2f} | {f['confidence']} | "
                    f"{', '.join(map(str, f['review_ids']))} | {f['decision']} |"
                )
            lines += ["", "Notes from the run and the output checks:"]
            lines += [f"- {n}" for n in r["notes"]] or ["- none"]
        d = self.differences
        lines += ["", "## Differences between runs", ""]
        if d is None:
            lines.append("One run only.")
        else:

            def show(xs):
                return ", ".join(f'"{x}"' for x in xs) or "none"

            lines += [
                (
                    f"- Flags: run 1 only {show(d['flags']['run 1 only'])}; "
                    f"run 2 only {show(d['flags']['run 2 only'])}."
                ),
                (
                    f"- Themes: run 1 only {show(d['themes']['run 1 only'])}; "
                    f"run 2 only {show(d['themes']['run 2 only'])}."
                ),
            ]
            for c in d["theme_changes"]:
                lines.append(
                    f'  - "{c["name"]}": {c["run 1"][0]} reviews, {c["run 1"][1]} in run 1; '
                    f"{c['run 2'][0]}, {c['run 2'][1]} in run 2."
                )
            lines.append(
                "- Aspect scores that differ: "
                + (
                    "; ".join(f"{s['aspect']} {s['run 1']} vs {s['run 2']}" for s in d["scores"])
                    or "none"
                )
                + "."
            )
            lines += [
                (
                    f"- Actions: run 1 only {show(d['actions']['run 1 only'])}; "
                    f"run 2 only {show(d['actions']['run 2 only'])}. Same order in both runs: "
                    f"{'yes' if d['same_action_order'] else 'no'}."
                ),
            ]
        lines += [
            "",
            "## Cost (measured tokens)",
            "",
            "| Run | Model | Calls | Input tokens | Output tokens |",
            "|---|---|---|---|---|",
        ]
        for r in self.runs:
            for model, u in r["usage"].items():
                lines.append(
                    f"| {r['run']} | {model} | {u['calls']} | {u['input_tokens']:,} | "
                    f"{u['output_tokens']:,} |"
                )
        lines += ["", "Output tokens include thinking tokens."]
        if self.tokens["usd"] is not None:
            prices = "; ".join(
                f"{m} ${p[0]:.2f} input, ${p[1]:.2f} output"
                for m, p in self.tokens["prices_per_mtok"].items()
            )
            lines.append(
                f"Estimated cost of all runs: ${self.tokens['usd']:.2f} at list prices per "
                f"million tokens as of {self.tokens['prices_as_of']} ({prices})."
            )
        m = self.methods
        lines += [
            "",
            "## Methods",
            "",
            f"- Signals: {', '.join(m['signals'])}.",
            *[f"- {s}: {t}." for s, t in m["thresholds"].items()],
            f"- Models: {', '.join(f'{k} {v}' for k, v in m['models'].items())}.",
            f"- Flag decisions: {m['flag_decisions_by']}.",
            *[f"- {k}: {v}" for k, v in m.get("settings", {}).items()],
            "",
            "## Limits",
            "",
            *[f"- {x}" for x in self.limits],
        ]
        return "\n".join(lines) + "\n"


def summarize(
    fixture_rows: Sequence[Mapping[str, Any]],
    briefs_data: Sequence[dict[str, Any]],
    usages: Sequence[Mapping[str, Mapping[str, int]]],
    *,
    settings: Mapping[str, str] | None = None,
    limits: Sequence[str] = (),
) -> GraphRunReport:
    intent = {r["id"]: r.get("intent") for r in fixture_rows}
    runs = [
        _run(k, data, intent, usage)
        for k, (data, usage) in enumerate(zip(briefs_data, usages, strict=True), start=1)
    ]
    total: dict[str, dict[str, int]] = {}
    for usage in usages:
        for model, u in usage.items():
            t = total.setdefault(model, {"calls": 0, "input_tokens": 0, "output_tokens": 0})
            for key in t:
                t[key] += u[key]
    first = briefs_data[0]
    return GraphRunReport(
        created_at=_dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        python_version=platform.python_version(),
        property=first["property"]["name"],
        reviews=len(fixture_rows),
        planted=[r["id"] for r in fixture_rows if r.get("intent") == "planted"],
        methods={**first["methods"], "settings": dict(settings or {})},
        golden={"displayed": GOLDEN_DISPLAYED, "verified": GOLDEN_VERIFIED},
        runs=runs,
        differences=_differences(runs[0], runs[1]) if len(runs) >= 2 else None,
        tokens={
            "total": total,
            "usd": _usd(total) if total else None,
            "prices_per_mtok": {m: PRICES_PER_MTOK[m] for m in total if m in PRICES_PER_MTOK},
            "prices_as_of": PRICES_AS_OF,
        },
        limits=list(limits),
    )
