#!/usr/bin/env python
"""Fetch the Ott Deceptive Opinion Spam Corpus and write data/interim/ott_corpus.csv.

The corpus is CC BY-NC-SA 3.0 and is not part of this repository. Three ways in:

    python scripts/fetch_ott_corpus.py                      # try to download the archive
    python scripts/fetch_ott_corpus.py --archive path.tar.gz   # use a copy you downloaded
    python scripts/fetch_ott_corpus.py --csv deceptive-opinion.csv   # the community CSV
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
from pathlib import Path

from review_synthesizer.data import ott


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", type=Path, help="existing op_spam_v1.4 .tar.gz or .zip")
    ap.add_argument("--csv", type=Path, help="community CSV repackaging instead of the archive")
    ap.add_argument("--url", default=ott.OTT_URL, help="archive URL to download when --archive is absent")
    ap.add_argument("--out", type=Path, default=ott.OUT_PATH)
    args = ap.parse_args()

    if args.csv:
        rows = list(ott.iter_kaggle_csv(args.csv))
        origin = str(args.csv)
    else:
        archive = args.archive
        if archive is None:
            if ott.RAW_PATH.exists():
                archive = ott.RAW_PATH
            else:
                try:
                    archive = ott.download(args.url, ott.RAW_PATH)
                    print(f"downloaded {archive} ({archive.stat().st_size:,} bytes)")
                except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
                    print(f"download failed: {e}\n", file=sys.stderr)
                    print(
                        "Download the corpus by hand from "
                        f"{ott.OTT_PAGE} and place it at {ott.RAW_PATH}, then rerun. "
                        "Or pass --archive <file> or --csv <file>.",
                        file=sys.stderr,
                    )
                    return 2
        rows = list(ott.iter_archive_reviews(archive))
        origin = f"{archive} sha256={ott.sha256(archive)[:16]}..."

    if not rows:
        print("no reviews found; is this the right file?", file=sys.stderr)
        return 1

    n = ott.write_csv(rows, args.out)
    summary = ott.summarize(rows)
    print(f"source: {origin}")
    print(f"wrote {n} rows to {args.out}")
    print(json.dumps(summary, indent=2))
    print(f"\ncite: {ott.CITATION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
