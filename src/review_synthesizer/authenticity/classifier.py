"""Deceptive language classifier.

A linear model over word and bigram TF-IDF features, trained on labelled hotel
reviews (the Ott corpus, see docs/data.md). It scores one review at a time and
says *why*: the terms that pushed the score toward "deceptive" and the terms
that pushed it back. That is what operators asked for in the capstone feedback
(the Mom Test, Module 5): not a number, the reason behind the number.

Scope: this is the demonstration baseline. It learns the style of reviews that
were written without a stay (Ott et al. found them heavy on superlatives and
light on spatial detail), which is one kind of fake and not every kind. Scores
are a signal for human review, never a verdict on a reviewer.

Dependencies (``pip install -e .[ml]``): scikit-learn, numpy, joblib.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

DECEPTIVE = "deceptive"
TRUTHFUL = "truthful"

_POSITIVE_LABELS = {DECEPTIVE, "1", "true", "fake", "planted"}
_NEGATIVE_LABELS = {TRUTHFUL, "0", "false", "genuine", "real"}


def to_binary(labels: Iterable[Any]) -> np.ndarray:
    """Map labels to ints: 1 means deceptive, 0 means truthful."""
    out = []
    for label in labels:
        if isinstance(label, (bool, np.bool_)):
            out.append(int(label))
        elif isinstance(label, (int, np.integer)):
            if label not in (0, 1):
                raise ValueError(f"integer labels must be 0 or 1, got {label!r}")
            out.append(int(label))
        else:
            key = str(label).strip().lower()
            if key in _POSITIVE_LABELS:
                out.append(1)
            elif key in _NEGATIVE_LABELS:
                out.append(0)
            else:
                raise ValueError(f"unrecognised label {label!r}")
    return np.asarray(out, dtype=int)


@dataclass(frozen=True)
class TermContribution:
    term: str
    weight: float
    """Signed contribution to the log odds of "deceptive" for one review."""


@dataclass(frozen=True)
class Explanation:
    probability: float
    """Probability that the review's language is deceptive, 0 to 1."""
    toward_deceptive: list[TermContribution]
    toward_truthful: list[TermContribution]

    def summary(self, k: int = 3) -> str:
        up = ", ".join(f'"{t.term}"' for t in self.toward_deceptive[:k]) or "none"
        down = ", ".join(f'"{t.term}"' for t in self.toward_truthful[:k]) or "none"
        return (
            f"p(deceptive) = {self.probability:.2f}; strongest cues toward deceptive: {up}; "
            f"toward truthful: {down}."
        )


class DeceptionClassifier:
    """TF-IDF (unigrams and bigrams) with L2 regularised logistic regression."""

    def __init__(
        self,
        *,
        C: float = 10.0,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_features: int | None = None,
        sublinear_tf: bool = True,
        random_state: int = 0,
    ) -> None:
        self.params: dict[str, Any] = {
            "C": C,
            "ngram_range": tuple(ngram_range),
            "min_df": min_df,
            "max_features": max_features,
            "sublinear_tf": sublinear_tf,
            "random_state": random_state,
        }
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=tuple(ngram_range),
                        min_df=min_df,
                        max_features=max_features,
                        sublinear_tf=sublinear_tf,
                        lowercase=True,
                        strip_accents="unicode",
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=C, solver="liblinear", max_iter=5000, random_state=random_state
                    ),
                ),
            ]
        )
        self.meta: dict[str, Any] = {}

    # ----- training and scoring -----

    @property
    def _vectorizer(self) -> TfidfVectorizer:
        return self.pipeline.named_steps["tfidf"]

    @property
    def _model(self) -> LogisticRegression:
        return self.pipeline.named_steps["clf"]

    @property
    def is_fitted(self) -> bool:
        return hasattr(self._model, "coef_")

    def fit(self, texts: Sequence[str], labels: Iterable[Any]) -> DeceptionClassifier:
        y = to_binary(labels)
        if len(y) != len(texts):
            raise ValueError("texts and labels differ in length")
        if y.min() == y.max():
            raise ValueError("training data must contain both classes")
        self.pipeline.fit(list(texts), y)
        self.meta.update(
            {
                "n_train": len(y),
                "n_deceptive": int(y.sum()),
                "vocabulary_size": len(self._vectorizer.vocabulary_),
                "trained_at": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
                "sklearn_version": sklearn.__version__,
            }
        )
        return self

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Probability of deceptive language for each text, shape ``(n,)``."""
        if not texts:
            return np.zeros(0)
        return self.pipeline.predict_proba(list(texts))[:, 1]

    def explain(self, text: str, k: int = 5) -> Explanation:
        """Score one review and list the terms that moved the score each way."""
        x = self._vectorizer.transform([text])
        coef = np.asarray(self._model.coef_[0])
        contrib = x.multiply(coef).tocoo()
        terms = self._vectorizer.get_feature_names_out()
        pairs = [
            TermContribution(str(terms[j]), float(v))
            for j, v in zip(contrib.col, contrib.data)
            if v != 0.0
        ]
        up = sorted((p for p in pairs if p.weight > 0), key=lambda p: -p.weight)[:k]
        down = sorted((p for p in pairs if p.weight < 0), key=lambda p: p.weight)[:k]
        prob = float(self.pipeline.predict_proba([text])[0, 1])
        return Explanation(probability=prob, toward_deceptive=up, toward_truthful=down)

    def top_terms(self, k: int = 20) -> dict[str, list[TermContribution]]:
        """Globally most indicative terms for each class, from the model weights."""
        coef = np.asarray(self._model.coef_[0])
        terms = self._vectorizer.get_feature_names_out()
        order = np.argsort(coef)
        return {
            DECEPTIVE: [TermContribution(str(terms[i]), float(coef[i])) for i in order[::-1][:k]],
            TRUTHFUL: [TermContribution(str(terms[i]), float(coef[i])) for i in order[:k]],
        }

    # ----- persistence -----

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"format": 1, "params": self.params, "meta": self.meta, "pipeline": self.pipeline},
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> DeceptionClassifier:
        obj = joblib.load(path)
        if obj.get("format") != 1:
            raise ValueError(f"unsupported model file format: {obj.get('format')!r}")
        inst = cls(**obj["params"])
        inst.pipeline = obj["pipeline"]
        inst.meta = dict(obj["meta"])
        return inst
