import json
import logging
from pathlib import Path

import numpy as np
import pytest

from machine_learning.machine_learning import (
    BUNDLE_PATH,
    MAX_PDF_BYTES,
    bm25_from_counts,
    clean_text,
    extract_pdf_text,
    load_bundle,
    score_uploaded,
    screen,
    strip_leading_title,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = PROJECT_ROOT / "outputs" / "parity_fixture.json"


def _fixture() -> dict:
    if not FIXTURE_PATH.exists():
        pytest.fail(f"Missing parity fixture: {FIXTURE_PATH}")
    if not BUNDLE_PATH.exists():
        pytest.fail(f"Missing screening bundle: {BUNDLE_PATH}")
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _embed_stub(fixture: dict) -> callable:
    vectors = {
        clean_text(fixture["job_description"]): np.asarray(fixture["job_vector"], dtype=float)
    }
    vectors.update(
        {
            clean_text(strip_leading_title(resume["text"])): np.asarray(vector, dtype=float)
            for resume, vector in zip(fixture["resumes"], fixture["resume_vectors"], strict=True)
        }
    )

    def embed(texts: list[str]) -> np.ndarray:
        return np.vstack([vectors[text] for text in texts])

    return embed


def test_parity_against_fixture() -> None:
    fixture = _fixture()
    bundle = load_bundle()
    embed_fn = _embed_stub(fixture)
    actual = screen(fixture["job_description"], fixture["resumes"], embed_fn, bundle)
    scored = score_uploaded(
        bundle,
        fixture["job_description"],
        [resume["text"] for resume in fixture["resumes"]],
        embed_fn,
    )

    assert actual["jd_skills"] == fixture["expected_jd_skill_weights"]
    for result, expected in zip(scored["results"], fixture["expected_results"], strict=True):
        assert result["rank"] == expected["rank"]
        assert result["matched"] == expected["matched"]
        assert result["missing_required"] == expected["missing_required"]
        assert result["missing_preferred"] == expected["missing_preferred"]
        assert result["parts"] == expected["parts"]
        assert result["skill_match_pct"] == expected["skill_match_pct"]
        assert result["score"] == pytest.approx(expected["score"], abs=0.05)

    actual_by_name = {row["name"]: row for row in actual["results"]}
    for resume, expected in zip(fixture["resumes"], fixture["expected_results"], strict=True):
        row = actual_by_name[resume["name"]]
        for key in ("rank", "skill_match_pct", "matched", "missing_required", "missing_preferred"):
            assert row[key] == expected[key]
        assert row["score"] == pytest.approx(expected["score"], abs=0.05)

    config = bundle["config"]
    query_terms = np.unique(
        bundle["vectorizer"].transform([clean_text(fixture["job_description"])]).indices
    )
    reference_scores = bm25_from_counts(
        bundle["ref_tf"],
        bundle["doc_len"],
        query_terms,
        bundle["idf"],
        bundle["doc_len"].mean(),
        config["bm25_k1"],
        config["bm25_b"],
    )
    top_ids = bundle["ref_ids"][np.argsort(-reference_scores)[:5]].tolist()
    assert top_ids == fixture["reference_bm25_top5_ids"]
    assert sum(score_uploaded(
        bundle,
        fixture["job_description"],
        [resume["text"] for resume in fixture["resumes"]],
        embed_fn,
        return_reference=True,
    )["reference"]["skills"]) == pytest.approx(
        fixture["reference_skill_coverage_sum"], abs=1e-9
    )


def test_invariants_and_determinism() -> None:
    fixture = _fixture()
    bundle = load_bundle()
    first = screen(fixture["job_description"], fixture["resumes"], _embed_stub(fixture), bundle)
    second = screen(fixture["job_description"], fixture["resumes"], _embed_stub(fixture), bundle)
    assert first == second
    rows = first["results"]
    assert sorted(row["rank"] for row in rows) == list(range(1, len(rows) + 1))
    assert [row["rank"] for row in sorted(
        rows, key=lambda row: row["score"], reverse=True
    )] == list(range(1, len(rows) + 1))
    skills = set(first["jd_skills"])
    for row in rows:
        assert 0 <= row["score"] <= 100
        assert 0 <= row["skill_match_pct"] <= 100
        assert not set(row["matched"]) & set(row["missing_required"])
        assert not set(row["matched"]) & set(row["missing_preferred"])
        assert (
            set(row["matched"]) | set(row["missing_required"]) | set(row["missing_preferred"])
        ) <= skills
    for text in [fixture["job_description"], *(resume["text"] for resume in fixture["resumes"])]:
        cleaned = clean_text(text)
        assert clean_text(cleaned) == cleaned


@pytest.mark.parametrize(
    ("job_description", "resumes", "message_text"),
    [
        ("short", [{"name": "a", "text": "x" * 50}], "short"),
        ("x" * 30, [], ""),
        ("x" * 30, [{"name": "a", "text": "x" * 50}] * 51, ""),
        ("x" * 30, [{"name": "a", "text": "tiny"}], "tiny"),
        ("x" * 30, [{"name": "a", "text": "x" * 20001}], ""),
        ("x" * 10001, [{"name": "a", "text": "x" * 50}], ""),
    ],
)
def test_input_validation(job_description: str, resumes: list[dict], message_text: str) -> None:
    with pytest.raises(ValueError) as error:
        screen(job_description, resumes, embed_fn=lambda _: np.zeros((len(resumes) + 1, 384)))
    if message_text:
        assert message_text not in str(error.value)


def test_pdf_extraction_and_rejections() -> None:
    def build_pdf(content: bytes) -> bytes:
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Count 1 /Kids [3 0 R] >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
            b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n"
            + content
            + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        ]
        pdf = bytearray(b"%PDF-1.4\n")
        offsets = [0]
        for number, body in enumerate(objects, start=1):
            offsets.append(len(pdf))
            pdf.extend(f"{number} 0 obj\n".encode() + body + b"\nendobj\n")
        startxref = len(pdf)
        pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode())
        pdf.extend(b"0000000000 65535 f \n")
        pdf.extend(b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:]))
        pdf.extend(
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{startxref}\n%%EOF".encode()
        )
        return bytes(pdf)

    pdf = build_pdf(b"BT /F1 12 Tf 20 20 Td (python sql) Tj ET")
    extracted = extract_pdf_text(pdf)
    assert "python" in extracted.lower()
    assert "sql" in extracted.lower()
    with pytest.raises(ValueError):
        extract_pdf_text(b"garbage")
    with pytest.raises(ValueError):
        extract_pdf_text(build_pdf(b""))
    with pytest.raises(ValueError):
        extract_pdf_text(b"x" * (MAX_PDF_BYTES + 1))


def test_privacy_logging(caplog: pytest.LogCaptureFixture) -> None:
    fixture = _fixture()
    with caplog.at_level(logging.DEBUG):
        screen(fixture["job_description"], fixture["resumes"], _embed_stub(fixture), load_bundle())
    logs = caplog.text
    assert fixture["job_description"] not in logs
    assert all(resume["text"] not in logs for resume in fixture["resumes"])
