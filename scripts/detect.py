#!/usr/bin/env python
"""Run the authenticity detector on a file of reviews and print the flags with reasons.

    python scripts/detect.py tests/fixtures/harbor_light_inn.json
    python scripts/detect.py reviews.csv --embeddings --classifier models/ott_deception.joblib
    python scripts/detect.py reviews.csv --json

Input: the fixture JSON layout ({"reviews": [{id, text, rating, date}, ...]}) or a
CSV with columns id, text, rating, date (ISO). Rules always run; embeddings and
the classifier run when asked for. Output is a signal for human review.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

from review_synthesizer.authenticity import Review, detect, rating_summary


def load_reviews(path: Path) -> list[Review]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data["reviews"] if isinstance(data, dict) else data
        return [Review.from_dict(r) for r in rows]
    with open(path, encoding="utf-8", newline="") as fh:
        return [Review.from_dict(r) for r in csv.DictReader(fh)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path)
    ap.add_argument("--embeddings", action="store_true", help="run the embedding similarity signal")
    ap.add_argument("--model", default=None, help="sentence-transformers model name")
    ap.add_argument("--classifier", type=Path, default=None, help="path to a saved DeceptionClassifier")
    ap.add_argument("--json", action="store_true", help="machine readable output")
    args = ap.parse_args()

    reviews = load_reviews(args.path)
    embedder = classifier = None
    if args.embeddings:
        from review_synthesizer.authenticity import DEFAULT_MODEL, SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder(args.model or DEFAULT_MODEL)
    if args.classifier:
        from review_synthesizer.authenticity.classifier import DeceptionClassifier

        classifier = DeceptionClassifier.load(args.classifier)

    result = detect(reviews, embedder=embedder, classifier=classifier)
    summary = rating_summary(reviews, result.flagged_ids)

    if args.json:
        out = {
            "signals": result.signals,
            "summary": asdict(summary),
            "flagged_ids": sorted(result.flagged_ids, key=str),
            "flags": [asdict(f) for f in result.flags],
        }
        print(json.dumps(out, indent=2, default=str))
        return 0

    print(f"{len(reviews)} reviews; signals run: {', '.join(result.signals)}")
    print(
        f"displayed rating {summary.displayed:.1f}, verified rating "
        f"{'n/a' if summary.verified is None else f'{summary.verified:.1f}'} "
        f"after removing {summary.removed} flagged review(s)"
    )
    if not result.flags:
        print("no flags")
        return 0
    for n, f in enumerate(result.flags, 1):
        ids = ", ".join(str(i) for i in f.review_ids)
        print(f"\n{n}. {f.signal_type}  [{f.confidence}, score {f.score:.2f}]  reviews: {ids}")
        print(f"   {f.rationale}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
