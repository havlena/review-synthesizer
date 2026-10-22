"""Deceptive Opinion Spam Corpus v1.4 (Ott et al. 2011, 2013).

1,600 hotel reviews for 20 Chicago hotels: 400 truthful positive (TripAdvisor),
400 deceptive positive (Mechanical Turk), 400 truthful negative (Expedia,
Hotels.com, Orbitz, Priceline, TripAdvisor, Yelp), 400 deceptive negative
(Mechanical Turk). Licensed CC BY-NC-SA 3.0 by its authors, so it is fetched on
demand and never committed to this repository. See docs/data.md.

Archive layout::

    op_spam_v1.4/<polarity>_polarity/<source>/fold<n>/<c>_<hotel>_<i>.txt

where ``<c>`` is ``t`` (truthful) or ``d`` (deceptive).
"""

from __future__ import annotations

import csv
import hashlib
import re
import tarfile
import urllib.request
import zipfile
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

OTT_URL = "https://myleott.com/op_spam_v1.4.tar.gz"
OTT_PAGE = "https://myleott.com/op-spam.html"
RAW_PATH = Path("data/raw/op_spam_v1.4.tar.gz")
OUT_PATH = Path("data/interim/ott_corpus.csv")

CITATION = (
    "Ott, Choi, Cardie, Hancock (2011), Finding Deceptive Opinion Spam by Any Stretch of the "
    "Imagination, ACL; Ott, Cardie, Hancock (2013), Negative Deceptive Opinion Spam, NAACL."
)

_MEMBER = re.compile(
    r"(?P<polarity>positive|negative)_polarity/"
    r"(?P<source>[A-Za-z_]+)/"
    r"fold(?P<fold>\d+)/"
    r"(?P<cls>[td])_(?P<hotel>[a-z0-9]+)_(?P<idx>\d+)\.txt$"
)


@dataclass(frozen=True)
class OttReview:
    id: str
    polarity: str  # positive | negative
    label: str  # truthful | deceptive
    source: str  # e.g. truthful_from_TripAdvisor, deceptive_from_MTurk, truthful_from_Web
    hotel: str
    fold: int  # 1..5 as published; 0 when unknown (Kaggle CSV)
    text: str


def parse_member_name(name: str) -> dict | None:
    """Return the fields encoded in an archive member path, or ``None`` if it is not a review."""
    name = name.replace("\\", "/")
    base = name.rsplit("/", 1)[-1]
    if base.startswith("._"):  # macOS resource forks inside the tarball
        return None
    m = _MEMBER.search(name)
    if not m:
        return None
    cls = m.group("cls")
    return {
        "polarity": m.group("polarity"),
        "label": "deceptive" if cls == "d" else "truthful",
        "source": m.group("source"),
        "hotel": m.group("hotel"),
        "fold": int(m.group("fold")),
        "idx": int(m.group("idx")),
    }


def _make_review(meta: dict, raw: bytes) -> OttReview:
    text = raw.decode("utf-8", errors="replace").strip()
    rid = f"{meta['polarity'][:3]}_{meta['label'][:1]}_{meta['hotel']}_{meta['idx']}"
    return OttReview(
        id=rid,
        polarity=meta["polarity"],
        label=meta["label"],
        source=meta["source"],
        hotel=meta["hotel"],
        fold=meta["fold"],
        text=text,
    )


def iter_archive_reviews(archive: Path) -> Iterator[OttReview]:
    """Yield reviews from the published ``.tar.gz`` (or a ``.zip`` mirror of it)."""
    archive = Path(archive)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                meta = parse_member_name(info.filename)
                if meta and not info.is_dir():
                    yield _make_review(meta, z.read(info))
        return
    with tarfile.open(archive, "r:*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            meta = parse_member_name(member.name)
            if meta:
                f = tar.extractfile(member)
                if f is not None:
                    yield _make_review(meta, f.read())


def iter_kaggle_csv(path: Path) -> Iterator[OttReview]:
    """Read the community CSV repackaging (columns: deceptive, hotel, polarity, source, text)."""
    counters: Counter[tuple[str, str, str]] = Counter()
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            label = row["deceptive"].strip().lower()
            polarity = row["polarity"].strip().lower()
            hotel = row["hotel"].strip().lower()
            key = (polarity, label, hotel)
            counters[key] += 1
            yield OttReview(
                id=f"{polarity[:3]}_{label[:1]}_{hotel}_{counters[key]}",
                polarity=polarity,
                label=label,
                source=row.get("source", "").strip(),
                hotel=hotel,
                fold=0,
                text=row["text"].strip(),
            )


def download(url: str = OTT_URL, dest: Path = RAW_PATH, timeout: int = 60) -> Path:
    """Download the archive to ``dest`` (skipped if it already exists)."""
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "review-synthesizer/0.2 (research)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as out:
        while chunk := resp.read(1 << 16):
            out.write(chunk)
    return dest


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(rows: Iterable[OttReview], out: Path = OUT_PATH) -> int:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["id", "polarity", "label", "source", "hotel", "fold", "text"]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))
            n += 1
    return n


def summarize(rows: list[OttReview]) -> dict:
    words = [len(r.text.split()) for r in rows]
    return {
        "reviews": len(rows),
        "by_label": dict(Counter(r.label for r in rows)),
        "by_polarity": dict(Counter(r.polarity for r in rows)),
        "by_source": dict(Counter(r.source for r in rows)),
        "hotels": len({r.hotel for r in rows}),
        "folds": sorted({r.fold for r in rows}),
        "mean_words": round(sum(words) / len(words), 1) if words else 0.0,
        "duplicate_texts": len(rows) - len({r.text for r in rows}),
    }
