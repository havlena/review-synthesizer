"""Parser tests on a synthetic archive. No corpus text is used or stored here."""

import io
import tarfile
import tempfile
from pathlib import Path

from review_synthesizer.data import ott


def test_parse_member_name_reads_the_published_layout():
    meta = ott.parse_member_name("op_spam_v1.4/positive_polarity/deceptive_from_MTurk/fold3/d_hilton_12.txt")
    assert meta == {
        "polarity": "positive",
        "label": "deceptive",
        "source": "deceptive_from_MTurk",
        "hotel": "hilton",
        "fold": 3,
        "idx": 12,
    }
    meta = ott.parse_member_name("negative_polarity/truthful_from_Web/fold5/t_allegro_7.txt")
    assert meta["label"] == "truthful" and meta["polarity"] == "negative" and meta["fold"] == 5


def test_parse_member_name_ignores_non_reviews():
    assert ott.parse_member_name("op_spam_v1.4/README.txt") is None
    assert ott.parse_member_name("op_spam_v1.4/positive_polarity/deceptive_from_MTurk/fold1/._d_hilton_1.txt") is None
    assert ott.parse_member_name("op_spam_v1.4/positive_polarity/deceptive_from_MTurk/fold1/") is None


def _synthetic_archive(path: Path) -> None:
    files = {
        "op_spam_v1.4/README.txt": "not a review",
        "op_spam_v1.4/positive_polarity/truthful_from_TripAdvisor/fold1/t_alpha_1.txt": "synthetic truthful one",
        "op_spam_v1.4/positive_polarity/deceptive_from_MTurk/fold1/d_alpha_1.txt": "synthetic deceptive one",
        "op_spam_v1.4/negative_polarity/truthful_from_Web/fold2/t_beta_3.txt": "synthetic truthful two",
        "op_spam_v1.4/negative_polarity/deceptive_from_MTurk/fold2/._d_beta_3.txt": "resource fork junk",
        "op_spam_v1.4/negative_polarity/deceptive_from_MTurk/fold2/d_beta_3.txt": "synthetic deceptive two  \n",
    }
    with tarfile.open(path, "w:gz") as tar:
        for name, text in files.items():
            data = text.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def test_iter_archive_reviews_yields_only_reviews_with_clean_text():
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "op_spam_v1.4.tar.gz"
        _synthetic_archive(archive)
        rows = list(ott.iter_archive_reviews(archive))
    assert len(rows) == 4
    ids = {r.id for r in rows}
    assert ids == {"pos_t_alpha_1", "pos_d_alpha_1", "neg_t_beta_3", "neg_d_beta_3"}
    by_id = {r.id: r for r in rows}
    assert by_id["neg_d_beta_3"].text == "synthetic deceptive two"
    assert by_id["neg_d_beta_3"].fold == 2 and by_id["neg_d_beta_3"].source == "deceptive_from_MTurk"


def test_summarize_counts():
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "a.tar.gz"
        _synthetic_archive(archive)
        rows = list(ott.iter_archive_reviews(archive))
        out = Path(tmp) / "out.csv"
        assert ott.write_csv(rows, out) == 4
        assert out.read_text(encoding="utf-8").startswith("id,polarity,label,source,hotel,fold,text")
    s = ott.summarize(rows)
    assert s["reviews"] == 4
    assert s["by_label"] == {"truthful": 2, "deceptive": 2}
    assert s["by_polarity"] == {"positive": 2, "negative": 2}
    assert s["hotels"] == 2 and s["folds"] == [1, 2] and s["duplicate_texts"] == 0
