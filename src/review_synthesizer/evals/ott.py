"""Evaluation of the deceptive language classifier on the Ott corpus.

Protocol: the five folds published with the corpus. Each fold holds the reviews
of four hotels, so no hotel is seen in both training and test; the report
checks that rather than assuming it. For each fold the model is trained on the
other four and scored on the held out one. Figures are reported per fold and as
mean and sample standard deviation across folds.

Two operating points are reported:

* the default decision at p = 0.5, for accuracy, precision, recall, F1 and ROC AUC;
* a conservative decision at ``threshold`` (0.8 by default), which is how the
  signal is used in detection: flag only when the model is confident, and
  accept that some deceptive reviews are missed.

The corpus is CC BY-NC-SA 3.0 (see docs/data.md). Results are committed under
``evals/results/``; the corpus and the model file are not.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import platform
import statistics
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from ..authenticity.classifier import DECEPTIVE, DeceptionClassifier, to_binary

DEFAULT_THRESHOLD = 0.8


@dataclass(frozen=True)
class CorpusRow:
    """One row of ``data/interim/ott_corpus.csv`` as written by step 4a."""

    id: str
    polarity: str
    label: str
    source: str
    hotel: str
    fold: int
    text: str


def load_corpus(path: str | Path) -> list[CorpusRow]:
    rows: list[CorpusRow] = []
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append(
                CorpusRow(
                    id=r["id"],
                    polarity=r["polarity"],
                    label=r["label"],
                    source=r.get("source", ""),
                    hotel=r["hotel"],
                    fold=int(r.get("fold") or 0),
                    text=r["text"],
                )
            )
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def assign_hotel_folds(rows: Sequence[CorpusRow], k: int = 5) -> list[CorpusRow]:
    """Give rows without published folds a hotel disjoint k fold assignment."""
    hotels = sorted({r.hotel for r in rows})
    fold_of = {h: (i % k) + 1 for i, h in enumerate(hotels)}
    return [CorpusRow(**{**asdict(r), "fold": fold_of[r.hotel]}) for r in rows]


def hotel_disjoint(rows: Sequence[CorpusRow]) -> bool:
    seen: dict[str, set[int]] = defaultdict(set)
    for r in rows:
        seen[r.hotel].add(r.fold)
    return all(len(folds) == 1 for folds in seen.values())


@dataclass(frozen=True)
class FoldResult:
    fold: int
    n_train: int
    n_test: int
    hotels_test: list[str]
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    accuracy_positive: float
    accuracy_negative: float
    precision_at_threshold: float
    recall_at_threshold: float
    flagged_at_threshold: int


_METRICS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "accuracy_positive",
    "accuracy_negative",
    "precision_at_threshold",
    "recall_at_threshold",
)


@dataclass
class EvalReport:
    protocol: str
    threshold: float
    params: dict[str, Any]
    corpus: dict[str, Any]
    hotel_disjoint: bool
    folds: list[FoldResult]
    mean: dict[str, float]
    std: dict[str, float]
    top_terms: dict[str, list[list[Any]]]
    created_at: str
    sklearn_version: str
    python_version: str
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["params"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.params.items()}
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_markdown(self) -> str:
        c = self.corpus
        lines = [
            "# Deceptive language classifier: evaluation on the Ott corpus",
            "",
            f"Generated {self.created_at} with scikit-learn {self.sklearn_version} on "
            f"Python {self.python_version}. Protocol: {self.protocol}.",
            "",
            f"Corpus: {c['reviews']} reviews, {c['deceptive']} deceptive and {c['truthful']} "
            f"truthful, {c['hotels']} hotels, {c['folds']} folds. "
            f"Folds are hotel disjoint: {'yes' if self.hotel_disjoint else 'NO'}.",
            "",
            "Model: TF-IDF over unigrams and bigrams, logistic regression. Parameters: "
            + ", ".join(f"{k}={v}" for k, v in self.params.items())
            + ".",
            "",
            "## Per fold (decision at p = 0.5; deceptive is the positive class)",
            "",
            "| Fold | Test hotels | n | Accuracy | Precision | Recall | F1 | ROC AUC | Acc. positive | Acc. negative |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for f in self.folds:
            lines.append(
                f"| {f.fold} | {', '.join(f.hotels_test)} | {f.n_test} | {f.accuracy:.3f} | "
                f"{f.precision:.3f} | {f.recall:.3f} | {f.f1:.3f} | {f.roc_auc:.3f} | "
                f"{f.accuracy_positive:.3f} | {f.accuracy_negative:.3f} |"
            )
        m, s = self.mean, self.std
        lines.append(
            f"| **Mean ± sd** | | | **{m['accuracy']:.3f} ± {s['accuracy']:.3f}** | "
            f"{m['precision']:.3f} ± {s['precision']:.3f} | {m['recall']:.3f} ± {s['recall']:.3f} | "
            f"{m['f1']:.3f} ± {s['f1']:.3f} | {m['roc_auc']:.3f} ± {s['roc_auc']:.3f} | "
            f"{m['accuracy_positive']:.3f} | {m['accuracy_negative']:.3f} |"
        )
        lines += [
            "",
            f"## Conservative operating point (flag only when p ≥ {self.threshold})",
            "",
            "This is how the signal is used in detection. Precision is what matters to an "
            "operator (a flag should be right); recall says how much slips through.",
            "",
            "| Fold | Flagged | Precision | Recall |",
            "|---|---|---|---|",
        ]
        for f in self.folds:
            lines.append(
                f"| {f.fold} | {f.flagged_at_threshold} of {f.n_test} | "
                f"{f.precision_at_threshold:.3f} | {f.recall_at_threshold:.3f} |"
            )
        lines.append(
            f"| **Mean ± sd** | | **{m['precision_at_threshold']:.3f} ± "
            f"{s['precision_at_threshold']:.3f}** | {m['recall_at_threshold']:.3f} ± "
            f"{s['recall_at_threshold']:.3f} |"
        )
        lines += [
            "",
            "## Most indicative terms (model trained on all folds)",
            "",
            "Toward deceptive: "
            + ", ".join(f"{t} ({w:+.2f})" for t, w in self.top_terms.get(DECEPTIVE, [])),
            "",
            "Toward truthful: "
            + ", ".join(f"{t} ({w:+.2f})" for t, w in self.top_terms.get("truthful", [])),
            "",
        ]
        if self.extras:
            lines += ["## Additional checks", ""]
            for key, value in self.extras.items():
                lines.append(f"- {key}: {value}")
            lines.append("")
        lines += [
            "## Read this before quoting the numbers",
            "",
            "- The deceptive reviews were written by paid crowd workers between 2011 and 2013 "
            "for 20 large Chicago hotels, in English. The model learns that style. Coordinated "
            "campaigns, paraphrased duplicates and reviews written by people who did stay are "
            "different problems, covered by the other signals.",
            "- Hotel disjoint folds stop the model from memorising hotel names, but all 20 "
            "hotels are in one city and one market segment. Expect lower figures on independent "
            "lodging in a different region until the model is retrained on data from it.",
            "- The corpus is CC BY-NC-SA 3.0. A model trained on it is for research and "
            "demonstration; see docs/data.md.",
            "",
        ]
        return "\n".join(lines)


def _fold_metrics(
    fold: int,
    train: Sequence[CorpusRow],
    test: Sequence[CorpusRow],
    threshold: float,
    clf_params: dict[str, Any],
) -> FoldResult:
    clf = DeceptionClassifier(**clf_params).fit([r.text for r in train], [r.label for r in train])
    y = to_binary([r.label for r in test])
    p = clf.predict_proba([r.text for r in test])
    pred = (p >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", pos_label=1, zero_division=0
    )
    pol = np.asarray([r.polarity for r in test])
    pos, neg = pol == "positive", pol == "negative"
    pred_t = (p >= threshold).astype(int)
    prec_t, rec_t, _, _ = precision_recall_fscore_support(
        y, pred_t, average="binary", pos_label=1, zero_division=0
    )
    return FoldResult(
        fold=fold,
        n_train=len(train),
        n_test=len(test),
        hotels_test=sorted({r.hotel for r in test}),
        accuracy=float(accuracy_score(y, pred)),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        roc_auc=float(roc_auc_score(y, p)) if len(set(y)) == 2 else float("nan"),
        accuracy_positive=float(accuracy_score(y[pos], pred[pos])) if pos.any() else float("nan"),
        accuracy_negative=float(accuracy_score(y[neg], pred[neg])) if neg.any() else float("nan"),
        precision_at_threshold=float(prec_t),
        recall_at_threshold=float(rec_t),
        flagged_at_threshold=int(pred_t.sum()),
    )


def evaluate(
    rows: Sequence[CorpusRow],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    k_folds: int = 5,
    top_k: int = 15,
    **clf_params: Any,
) -> EvalReport:
    """Cross validate on the published folds (or hotel disjoint folds when absent)."""
    if any(r.fold <= 0 for r in rows):
        rows = assign_hotel_folds(rows, k_folds)
        protocol = f"{k_folds} fold cross validation, folds assigned by hotel"
    else:
        protocol = "published folds, leave one fold out"
    folds = sorted({r.fold for r in rows})
    results: list[FoldResult] = []
    for k in folds:
        train = [r for r in rows if r.fold != k]
        test = [r for r in rows if r.fold == k]
        results.append(_fold_metrics(k, train, test, threshold, clf_params))

    def agg(fn, name):
        vals = [getattr(r, name) for r in results if not np.isnan(getattr(r, name))]
        return float(fn(vals)) if vals else float("nan")

    mean = {m: agg(statistics.fmean, m) for m in _METRICS}
    std = {
        m: (agg(statistics.stdev, m) if len(results) > 1 else 0.0) for m in _METRICS
    }

    final = fit_final(rows, **clf_params)
    top = {
        label: [[t.term, round(t.weight, 3)] for t in terms]
        for label, terms in final.top_terms(top_k).items()
    }
    labels = [r.label for r in rows]
    report = EvalReport(
        protocol=protocol,
        threshold=threshold,
        params=dict(final.params),
        corpus={
            "reviews": len(rows),
            "deceptive": sum(1 for x in labels if x == DECEPTIVE),
            "truthful": sum(1 for x in labels if x == "truthful"),
            "hotels": len({r.hotel for r in rows}),
            "folds": len(folds),
            "duplicate_texts": len(rows) - len({r.text for r in rows}),
        },
        hotel_disjoint=hotel_disjoint(rows),
        folds=results,
        mean=mean,
        std=std,
        top_terms=top,
        created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        sklearn_version=sklearn.__version__,
        python_version=platform.python_version(),
    )
    return report


def fit_final(rows: Sequence[CorpusRow], **clf_params: Any) -> DeceptionClassifier:
    """Train on every row, for use in detection and for the term tables."""
    clf = DeceptionClassifier(**clf_params)
    clf.fit([r.text for r in rows], [r.label for r in rows])
    clf.meta["trained_on"] = "Ott Deceptive Opinion Spam Corpus v1.4, all folds"
    return clf
