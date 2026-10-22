"""verified_set and human_review: the pause, the answer, and what a dismissal releases.

Harbor Light Inn has two flags, duplicate phrasing and a rating burst, both on reviews 21
to 25. Confirming keeps the golden 3.8 displayed to 3.5 verified.
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("langgraph")

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from review_synthesizer.authenticity import Flag
from review_synthesizer.graph import FakeModelClient, Property, build_graph
from review_synthesizer.graph.approval import REVIEW_NOTE, flag_key, held_ids, parse_decisions

FIXTURES = Path(__file__).parent / "fixtures"
HARBOR = Property(name="Harbor Light Inn", type="inn")
PLANTED = [21, 22, 23, 24, 25]
DUP = "duplicate_phrasing:21,22,23,24,25"
BURST = "rating_burst:21,22,23,24,25"


def harbor_input():
    rows = json.loads((FIXTURES / "harbor_light_inn.json").read_text(encoding="utf-8"))["reviews"]
    return {"property": HARBOR, "raw_reviews": rows}


def runs_of(graph, config, node):
    """How many times ``node`` was scheduled in this thread's history."""
    return sum(1 for s in graph.get_state_history(config) if node in s.next)


def test_run_pauses_at_human_review_with_every_flag_and_its_reason():
    graph = build_graph(FakeModelClient(), checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "pause"}}
    out = graph.invoke(harbor_input(), config)

    assert graph.get_state(config).next == ("human_review",)
    assert "brief" not in out
    request = out["__interrupt__"][0].value
    assert request["note"] == REVIEW_NOTE
    assert [f["key"] for f in request["flags"]] == [DUP, BURST]
    assert all(f["rationale"] and f["review_ids"] == PLANTED for f in request["flags"])
    assert request["rating"] == {"displayed": 3.8, "verified": 3.5}


def test_confirming_every_flag_keeps_the_golden_rating_and_does_not_rerun(run_graph):
    out, graph, config, requests = run_graph(FakeModelClient(), harbor_input())
    assert len(requests) == 1
    assert out["decisions"] == {DUP: True, BURST: True}
    assert out["rating"].displayed == 3.8 and out["rating"].verified == 3.5
    assert out["verified_ids"] == list(range(1, 21))
    assert runs_of(graph, config, "themes_and_scores") == 1
    assert "brief" in out


def test_dismissing_every_flag_releases_the_reviews_and_reruns_synthesis(run_graph):
    out, graph, config, requests = run_graph(
        FakeModelClient(),
        harbor_input(),
        decide=lambda req: {f["key"]: False for f in req["flags"]},
    )
    assert len(requests) == 1  # the rerun does not ask again
    assert out["verified_ids"] == list(range(1, 26))
    assert out["rating"].removed == 0
    assert out["rating"].verified == out["rating"].displayed == 3.8
    assert runs_of(graph, config, "verified_set") == 2
    assert runs_of(graph, config, "themes_and_scores") == 2
    assert runs_of(graph, config, "actions") == 2


def test_a_review_held_by_another_confirmed_flag_stays_out(run_graph):
    out, graph, config, _ = run_graph(
        FakeModelClient(), harbor_input(), decide=lambda req: {DUP: False, BURST: True}
    )
    assert out["verified_ids"] == list(range(1, 21))
    assert out["rating"].verified == 3.5
    assert runs_of(graph, config, "themes_and_scores") == 1


@pytest.mark.parametrize(
    ("answer", "error", "message"),
    [
        ({DUP: True}, ValueError, f"no decision for {BURST}"),
        ({DUP: True, BURST: True, "x:1": True}, ValueError, "unknown flags x:1"),
        ({DUP: True, BURST: "yes"}, ValueError, f"decision is not true or false for {BURST}"),
        ([True, True], TypeError, "must map each flag key"),
    ],
)
def test_an_incomplete_or_malformed_answer_is_refused(answer, error, message):
    graph = build_graph(FakeModelClient(), checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "bad-answer"}}
    graph.invoke(harbor_input(), config)
    with pytest.raises(error, match=message):
        graph.invoke(Command(resume=answer), config)
    assert "brief" not in graph.get_state(config).values


def test_a_batch_without_flags_is_not_paused():
    rows = [
        {"id": i, "text": f"Distinct stay number {i} {w}", "rating": r, "date": d}
        for i, w, r, d in [
            (1, "quiet harbor room", 4, "2026-01-02"),
            (2, "cold breakfast eggs", 2, "2026-02-14"),
            (3, "friendly front desk", 5, "2026-03-30"),
        ]
    ]
    graph = build_graph(FakeModelClient())  # no checkpointer: an interrupt would fail
    out = graph.invoke({"property": HARBOR, "raw_reviews": rows})
    assert out["decisions"] == {} and out["verified_ids"] == [1, 2, 3]
    assert "brief" in out


def test_held_ids_releases_only_dismissed_flags():
    a = Flag("rating_burst", [1, 2], 0.8, "high", "r")
    b = Flag("claim_duplicate", [2, 3], 1.0, "high", "r")
    assert held_ids([a, b], {}) == {1, 2, 3}
    assert held_ids([a, b], {flag_key(a): False}) == {2, 3}
    assert held_ids([a, b], {flag_key(a): False, flag_key(b): False}) == set()
    assert flag_key(b) == "claim_duplicate:2,3"


def test_parse_decisions_returns_only_the_asked_keys():
    assert parse_decisions({"a": True, "b": False}, ["a", "b"]) == {"a": True, "b": False}
