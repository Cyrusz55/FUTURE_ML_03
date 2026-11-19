import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from apps import model_loader
from apps.main import app
from apps.routes import get_screening_context
from machine_learning import machine_learning

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / "outputs" / "parity_fixture.json").read_text(encoding="utf-8"))
BUNDLE = machine_learning.load_bundle()


def _embed_stub(texts: list[str]) -> np.ndarray:
    vectors = {
        machine_learning.clean_text(FIXTURE["job_description"]): np.asarray(
            FIXTURE["job_vector"], dtype=float
        )
    }
    vectors.update(
        {
            machine_learning.clean_text(
                machine_learning.strip_leading_title(resume["text"])
            ): np.asarray(vector, dtype=float)
            for resume, vector in zip(
                FIXTURE["resumes"], FIXTURE["resume_vectors"], strict=True
            )
        }
    )
    return np.vstack([vectors[text] for text in texts])


def _hash_embed(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 384))
    rows = []
    for text in texts:
        seed = sum(text.encode("utf-8")) % (2**32)
        vector = np.random.default_rng(seed).random(384)
        rows.append(vector / np.linalg.norm(vector))
    return np.vstack(rows)


def _api_embed(texts: list[str]) -> np.ndarray:
    known = {
        machine_learning.clean_text(FIXTURE["job_description"]): np.asarray(
            FIXTURE["job_vector"], dtype=float
        )
    }
    known.update(
        {
            machine_learning.clean_text(
                machine_learning.strip_leading_title(resume["text"])
            ): np.asarray(vector, dtype=float)
            for resume, vector in zip(
                FIXTURE["resumes"], FIXTURE["resume_vectors"], strict=True
            )
        }
    )
    missing = [text for text in texts if text not in known]
    hashed = dict(zip(missing, _hash_embed(missing), strict=True))
    return np.vstack([known[text] if text in known else hashed[text] for text in texts])


@pytest.fixture
def client() -> TestClient:
    app.dependency_overrides[get_screening_context] = lambda: (BUNDLE, _api_embed)
    model_loader._BUNDLE = BUNDLE
    model_loader._EMBED_FN = _api_embed
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _raw_pdf(content: bytes) -> bytes:
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


def test_health_model_info_and_index(client: TestClient) -> None:
    health = client.get("/api/v1/health")
    assert health.status_code == 200
    assert health.json()["model_loaded"] is True
    info = client.get("/api/v1/model-info")
    assert info.status_code == 200
    body = info.json()
    assert {"hf_repo_id", "hf_revision", "embedding_model_id", "limits"} <= body.keys()
    assert {"max_resumes", "max_pdf_bytes", "max_total_upload_bytes"} <= body["limits"].keys()
    assert client.get("/").status_code == 200
    assert "Resume screening" in client.get("/").text
    uptime_response = client.head("/")
    assert uptime_response.status_code == 200
    assert uptime_response.content == b""


def test_json_parity(client: TestClient) -> None:
    response = client.post(
        "/api/v1/screen",
        json={"job_description": FIXTURE["job_description"], "resumes": FIXTURE["resumes"]},
    )
    assert response.status_code == 200
    actual_rows = {row["name"]: row for row in response.json()["results"]}
    for resume, expected in zip(FIXTURE["resumes"], FIXTURE["expected_results"], strict=True):
        actual = actual_rows[resume["name"]]
        assert actual["rank"] == expected["rank"]
        assert actual["score"] == pytest.approx(expected["score"], abs=0.05)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"job_description": "   ", "resumes": []},
        {"job_description": "x" * 29, "resumes": []},
        {"job_description": "x" * 10001, "resumes": []},
        {"job_description": "x" * 30, "resumes": []},
        {"job_description": "x" * 30, "resumes": [{"text": "x" * 49}]},
        {"job_description": "x" * 30, "resumes": [{"text": "x" * 20001}]},
        {"job_description": "x" * 30, "resumes": [{"text": 42}]},
        {"job_description": "x" * 30, "resumes": [{"text": "x" * 50}] * 51},
    ],
)
def test_json_validation(client: TestClient, payload: dict) -> None:
    assert client.post("/api/v1/screen", json=payload).status_code == 422


def test_multipart_files_and_skipped_items(client: TestClient) -> None:
    text = "python sql " * 20
    response = client.post(
        "/api/v1/screen/files",
        data={"job_description": FIXTURE["job_description"], "pasted_texts": text},
        files=[
            (
                "files",
                (
                    "resume.pdf",
                    _raw_pdf(
                        b"BT /F1 12 Tf (python sql python sql "
                        b"python sql python sql python sql) Tj ET"
                    ),
                    "application/pdf",
                ),
            ),
            ("files", ("resume.txt", text.encode(), "text/plain")),
            ("files", ("bad.pdf", b"not a pdf", "application/pdf")),
            ("files", ("large.txt", b"x" * (machine_learning.MAX_PDF_BYTES + 1), "text/plain")),
        ],
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 3
    assert {item["name"] for item in body["skipped"]} == {"bad.pdf", "large.txt"}


def test_unreadable_files_and_content_length(client: TestClient) -> None:
    response = client.post(
        "/api/v1/screen/files",
        data={"job_description": FIXTURE["job_description"]},
        files=[("files", ("bad.pdf", b"garbage", "application/pdf"))],
    )
    assert response.status_code == 422
    response = client.post(
        "/api/v1/screen/files",
        data={"job_description": FIXTURE["job_description"]},
        headers={"content-length": str(25 * 1024 * 1024 + 1)},
    )
    assert response.status_code == 413


def test_filename_and_response_extra_key(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from apps.routes import _safe_filename

    assert _safe_filename(r"..\secret/\x00resume.txt") == "x00resume.txt"
    original = machine_learning.screen

    def extra(*args, **kwargs):
        result = original(*args, **kwargs)
        result["unexpected"] = "hidden"
        return result

    monkeypatch.setattr(machine_learning, "screen", extra)
    response = client.post(
        "/api/v1/screen",
        json={"job_description": FIXTURE["job_description"], "resumes": FIXTURE["resumes"][:1]},
    )
    assert response.status_code == 200
    assert "unexpected" not in response.json()


def test_privacy_logging(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("DEBUG"):
        client.post(
            "/api/v1/screen",
            json={"job_description": FIXTURE["job_description"], "resumes": FIXTURE["resumes"][:1]},
        )
    assert FIXTURE["job_description"] not in caplog.text
    assert FIXTURE["resumes"][0]["text"] not in caplog.text
    assert "data_analyst" not in caplog.text
