"""load_batch validation and the authenticity node inside the graph.

The graph must give exactly the detector's answer: the golden five planted reviews on
Harbor Light Inn, and the optional signals only when they are configured. Skipped when
the ``graph`` extra is not installed.
"""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("langgraph")

from langgraph.checkpoint.memory import InMemorySaver

from review_synthesizer.authenticity import Review, detect
from review_synthesizer.graph import (
    BatchError,
    FakeModelClient,
    Property,
    build_graph,
    validate_batch,
)

FIXTURES = Path(__file__).parent / "fixtures"


def harbor_rows():
    return json.loads((FIXTURES / "harbor_light_inn.json").read_text(encoding="utf-8"))["reviews"]


def row(id_=1, text="Clean room, kind staff.", rating=4, date_="2026-03-02", **extra):
    return {"id": id_, "text": text, "rating": rating, "date": date_, **extra}


# validate_batch ---------------------------------------------------------------------


def test_harbor_fixture_validates_to_the_same_reviews_as_the_detector_loader():
    rows = harbor_rows()
    assert validate_batch(rows) == [Review.from_dict(r) for r in rows]


def test_accepts_posted_key_date_objects_and_csv_style_strings():
    reviews = validate_batch(
        [
            row(1, rating="5", date_=None, posted="2026-03-02"),
            row("b-2", rating=3.0, date_=date(2026, 3, 3)),
        ]
    )
    assert [(r.id, r.rating, r.posted) for r in reviews] == [
        (1, 5, date(2026, 3, 2)),
        ("b-2", 3, date(2026, 3, 3)),
    ]


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        (row(rating=0), "rating 0 is not a whole number from 1 to 5"),
        (row(rating=6), "rating 6 is not a whole number from 1 to 5"),
        (row(rating=4.5), "rating 4.5 is not a whole number"),
        (row(rating=True), "rating True is not a whole number"),
        (row(rating="four"), "rating 'four' is not a whole number"),
        (row(rating=None), "rating None is not a whole number"),
        (row(date_="03/02/2026"), "date '03/02/2026' is missing or not YYYY-MM-DD"),
        (row(date_=None), "date None is missing or not YYYY-MM-DD"),
        (row(text="   "), "review text is missing or blank"),
        (row(text=None), "review text is missing or blank"),
        ({"text": "x", "rating": 3, "date": "2026-03-02"}, "row 1: missing id"),
    ],
)
def test_each_problem_is_named_for_the_operator(bad, message):
    with pytest.raises(BatchError) as err:
        validate_batch([bad])
    assert len(err.value.problems) == 1
    assert message in err.value.problems[0]


def test_every_problem_is_reported_at_once_including_duplicate_ids():
    with pytest.raises(BatchError) as err:
        validate_batch([row(7), row(7, rating=9), row(8, text="")])
    assert err.value.problems == [
        "review 7 (row 1): id 7 appears more than once",
        "review 7 (row 2): id 7 appears more than once",
        "review 7 (row 2): rating 9 is not a whole number from 1 to 5",
        "review 8 (row 3): review text is missing or blank",
    ]


def test_empty_batch_is_rejected():
    with pytest.raises(BatchError, match="no reviews"):
        validate_batch([])


# authenticity node in the graph -----------------------------------------------------


class SameVector:
    """Every text gets the same embedding, so every pair is a perfect match."""

    name = "fake-same"

    def embed(self, texts):
        return np.ones((len(texts), 3)) / np.sqrt(3)


class FixedProbability:
    """Deceptive language probability from a table keyed by text; 0 otherwise."""

    def __init__(self, table):
        self.table = table

    def predict_proba(self, texts):
        return np.array([self.table.get(t, 0.0) for t in texts])


def run(rows, **kwargs):
    g = build_graph(FakeModelClient(), **kwargs)
    return g.invoke({"property": Property("Harbor Light Inn"), "raw_reviews": rows})


def test_graph_detection_is_the_detector_result_and_keeps_the_golden_five():
    rows = harbor_rows()
    out = run(rows)
    direct = detect([Review.from_dict(r) for r in rows])
    assert out["detection"] == direct
    planted = {r["id"] for r in rows if r["intent"] == "planted"}
    assert out["detection"].flagged_ids == planted == {21, 22, 23, 24, 25}
    assert out["detection"].signals == ["duplicate_phrasing", "rating_burst"]


def test_optional_signals_run_only_when_configured():
    rows = harbor_rows()
    first_text = rows[0]["text"]
    out = run(rows, embedder=SameVector(), classifier=FixedProbability({first_text: 0.95}))
    signals = out["detection"].signals
    assert signals == [
        "duplicate_phrasing",
        "rating_burst",
        "embedding_similarity",
        "deceptive_language",
    ]
    deceptive = [f for f in out["detection"].flags if f.signal_type == "deceptive_language"]
    assert [f.review_ids for f in deceptive] == [[rows[0]["id"]]]


def test_invalid_batch_stops_the_run_before_any_node_after_load_batch():
    g = build_graph(FakeModelClient(), checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "bad"}}
    with pytest.raises(BatchError, match="rating 9"):
        g.invoke({"property": Property("Harbor Light Inn"), "raw_reviews": [row(rating=9)]}, config)
    assert "detection" not in g.get_state(config).values
