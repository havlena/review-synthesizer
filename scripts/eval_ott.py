#!/usr/bin/env python
"""Evaluate the deceptive language classifier on the Ott corpus and write the report.

    python scripts/eval_ott.py                 # data/interim/ott_corpus.csv -> evals/results/
    python scripts/eval_ott.py --threshold 0.9 --C 1.0

Writes evals/results/ott_classifier.md and .json (committed), saves the model
trained on all folds to models/ott_deception.joblib (git ignored), and scores the
Harbor Light Inn fixture as a sanity check.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from review_synthesizer.authenticity.classifier import DeceptionClassifier
from review_synthesizer.evals import ott

ROOT = Path(__file__).resolve().parents[1]


def score_fixture(clf: DeceptionClassifier, path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    reviews = data["reviews"]
    probs = clf.predict_proba([r["text"] for r in reviews])
    planted = [p for p, r in zip(probs, reviews) if r.get("intent") == "planted"]
    genuine = [p for p, r in zip(probs, reviews) if r.get("intent") != "planted"]
    return {
        "planted_mean_p": round(statistics.fmean(planted), 3) if planted else float("nan"),
        "genuine_mean_p": round(statistics.fmean(genuine), 3) if genuine else float("nan"),
        "genuine_max_p": round(float(max(genuine)), 3) if genuine else float("nan"),
        "planted_min_p": round(float(min(planted)), 3) if planted else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=ROOT / "data/interim/ott_corpus.csv")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "evals/results")
    ap.add_argument("--threshold", type=float, default=ott.DEFAULT_THRESHOLD)
    ap.add_argument("--C", type=float, default=10.0, help="inverse regularisation strength")
    ap.add_argument("--min-df", type=int, default=2)
    ap.add_argument("--save-model", type=Path, default=ROOT / "models/ott_deception.joblib")
    ap.add_argument("--no-model", action="store_true", help="do not save the model file")
    args = ap.parse_args()

    if not args.csv.exists():
        print(f"{args.csv} not found; run scripts/fetch_ott_corpus.py first", file=sys.stderr)
        return 2

    rows = ott.load_corpus(args.csv)
    print(f"loaded {len(rows)} rows from {args.csv}")
    report = ott.evaluate(rows, threshold=args.threshold, C=args.C, min_df=args.min_df)

    final = ott.fit_final(rows, C=args.C, min_df=args.min_df)
    fixture = score_fixture(final, ROOT / "tests/fixtures/harbor_light_inn.json")
    if fixture:
        report.extras["Harbor Light Inn fixture (informational, 25 synthetic reviews)"] = (
            f"mean p(deceptive) planted {fixture['planted_mean_p']} vs genuine "
            f"{fixture['genuine_mean_p']}; lowest planted {fixture['planted_min_p']}, "
            f"highest genuine {fixture['genuine_max_p']}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    md = args.out_dir / "ott_classifier.md"
    js = args.out_dir / "ott_classifier.json"
    md.write_text(report.to_markdown(), encoding="utf-8")
    js.write_text(report.to_json(), encoding="utf-8")

    if not args.no_model:
        final.save(args.save_model)
        print(f"model saved to {args.save_model}")

    m, s = report.mean, report.std
    print(f"protocol: {report.protocol}; hotel disjoint folds: {report.hotel_disjoint}")
    print(
        f"accuracy {m['accuracy']:.3f} ± {s['accuracy']:.3f}  "
        f"precision {m['precision']:.3f}  recall {m['recall']:.3f}  "
        f"F1 {m['f1']:.3f}  ROC AUC {m['roc_auc']:.3f}"
    )
    print(
        f"at p >= {args.threshold}: precision {m['precision_at_threshold']:.3f} ± "
        f"{s['precision_at_threshold']:.3f}, recall {m['recall_at_threshold']:.3f}"
    )
    if fixture:
        print(f"fixture: {fixture}")
    print(f"report: {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
