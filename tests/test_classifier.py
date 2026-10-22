"""Classifier and evaluation harness tests on synthetic text. No corpus text is used."""

import random
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("sklearn")

from review_synthesizer.authenticity.classifier import (  # noqa: E402
    DECEPTIVE,
    TRUTHFUL,
    DeceptionClassifier,
    to_binary,
)
from review_synthesizer.evals import ott  # noqa: E402

TRUTHFUL_PHRASES = [
    "our room on the fourth floor faced the parking garage",
    "the bathroom was small but the shower pressure was fine",
    "the elevator took a while at breakfast time",
    "we walked to the river from the lobby in ten minutes",
    "checkout was at noon and the front desk gave us a map",
    "the lobby bar closed at ten on a weeknight",
    "the window did not open and the heater clicked",
    "we parked in the structure across the street for thirty dollars",
]
DECEPTIVE_PHRASES = [
    "my husband and i had the most amazing vacation of our lives",
    "this is truly the best hotel in the entire city",
    "everything about it was luxurious and perfect in every way",
    "i would highly recommend this hotel to absolutely everyone",
    "the staff were incredible and made us feel like royalty",
    "a wonderful experience we will never ever forget",
    "five stars is not enough for this gorgeous property",
    "we cannot wait to come back to this perfect place",
]
NEUTRAL = ["we stayed two nights", "the hotel is downtown", "it rained on saturday"]


def synthetic_docs(n_per_class: int, seed: int = 7) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    texts, labels = [], []
    for label, pool in ((TRUTHFUL, TRUTHFUL_PHRASES), (DECEPTIVE, DECEPTIVE_PHRASES)):
        for _ in range(n_per_class):
            parts = rng.sample(pool, k=rng.randint(3, 5)) + rng.sample(NEUTRAL, k=1)
            rng.shuffle(parts)
            texts.append(". ".join(parts) + ".")
            labels.append(label)
    return texts, labels


def test_to_binary_accepts_strings_ints_and_bools():
    assert list(to_binary(["deceptive", "truthful", 1, 0, True, False])) == [1, 0, 1, 0, 1, 0]
    with pytest.raises(ValueError):
        to_binary(["maybe"])


def test_fit_predict_explain_and_round_trip():
    texts, labels = synthetic_docs(60)
    train_t, train_l = texts[:50] + texts[60:110], labels[:50] + labels[60:110]
    test_t, test_l = texts[50:60] + texts[110:120], labels[50:60] + labels[110:120]

    clf = DeceptionClassifier(min_df=1).fit(train_t, train_l)
    assert clf.is_fitted and clf.meta["n_train"] == 100 and clf.meta["vocabulary_size"] > 0

    p = clf.predict_proba(test_t)
    assert p.shape == (20,) and ((0.0 <= p) & (p <= 1.0)).all()
    pred = [DECEPTIVE if x >= 0.5 else TRUTHFUL for x in p]
    accuracy = sum(a == b for a, b in zip(pred, test_l)) / len(test_l)
    assert accuracy >= 0.9

    exp = clf.explain(DECEPTIVE_PHRASES[0] + ". " + TRUTHFUL_PHRASES[0] + ".", k=3)
    assert 0.0 <= exp.probability <= 1.0
    assert exp.toward_deceptive and exp.toward_truthful
    assert all(t.weight > 0 for t in exp.toward_deceptive)
    assert all(t.weight < 0 for t in exp.toward_truthful)
    assert "p(deceptive)" in exp.summary()

    top = clf.top_terms(5)
    assert len(top[DECEPTIVE]) == 5 and len(top[TRUTHFUL]) == 5
    assert top[DECEPTIVE][0].weight > 0 > top[TRUTHFUL][0].weight

    with tempfile.TemporaryDirectory() as tmp:
        path = clf.save(Path(tmp) / "m.joblib")
        again = DeceptionClassifier.load(path)
    assert again.params == clf.params and again.meta == clf.meta
    assert list(again.predict_proba(test_t)) == list(p)


def test_fit_rejects_single_class_and_mismatched_lengths():
    with pytest.raises(ValueError):
        DeceptionClassifier(min_df=1).fit(["a b", "a c"], [DECEPTIVE, DECEPTIVE])
    with pytest.raises(ValueError):
        DeceptionClassifier(min_df=1).fit(["a b"], [DECEPTIVE, TRUTHFUL])


def _synthetic_rows(n_per_class: int = 45) -> list[ott.CorpusRow]:
    texts, labels = synthetic_docs(n_per_class, seed=11)
    rows = []
    for i, (t, l) in enumerate(zip(texts, labels)):
        hotel = f"hotel{i % 6}"
        rows.append(
            ott.CorpusRow(
                id=f"r{i}",
                polarity="positive" if i % 2 else "negative",
                label=l,
                source="synthetic",
                hotel=hotel,
                fold=(i % 6) % 3 + 1,
                text=t,
            )
        )
    return rows


def test_evaluate_on_published_style_folds():
    rows = _synthetic_rows()
    assert ott.hotel_disjoint(rows)
    report = ott.evaluate(rows, threshold=0.8, min_df=1)
    assert report.protocol.startswith("published folds")
    assert [f.fold for f in report.folds] == [1, 2, 3]
    assert report.hotel_disjoint is True
    assert report.corpus["reviews"] == 90 and report.corpus["hotels"] == 6
    assert report.mean["accuracy"] >= 0.85
    assert 0.0 <= report.mean["roc_auc"] <= 1.0
    assert set(report.top_terms) == {DECEPTIVE, TRUTHFUL}
    md = report.to_markdown()
    assert "| Fold |" in md and "Mean ± sd" in md and "p ≥ 0.8" in md
    assert '"ngram_range": [' in report.to_json()


def test_evaluate_assigns_hotel_folds_when_missing():
    rows = [ott.CorpusRow(**{**r.__dict__, "fold": 0}) for r in _synthetic_rows()]
    report = ott.evaluate(rows, k_folds=3, min_df=1)
    assert report.protocol.startswith("3 fold")
    assert len(report.folds) == 3 and report.hotel_disjoint is True


def test_load_corpus_reads_the_step_4a_csv_layout():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "c.csv"
        p.write_text(
            "id,polarity,label,source,hotel,fold,text\n"
            "pos_t_h_1,positive,truthful,truthful_from_TripAdvisor,h,1,\"one, two\"\n"
            "neg_d_h_1,negative,deceptive,deceptive_from_MTurk,h,2,three\n",
            encoding="utf-8",
        )
        rows = ott.load_corpus(p)
    assert len(rows) == 2 and rows[0].text == "one, two" and rows[1].fold == 2
