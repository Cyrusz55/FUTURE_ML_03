import json
from pathlib import Path

import numpy as np
import pytest
from huggingface_hub import hf_hub_download

from apps.model_loader import HF_FILENAME, HF_REPO_ID, HF_REVISION
from machine_learning import machine_learning

pytestmark = pytest.mark.hf

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / "outputs" / "parity_fixture.json").read_text(encoding="utf-8"))
LOCAL_BUNDLE_PATH = ROOT / "model" / HF_FILENAME


def _fixture_embed(texts: list[str]) -> np.ndarray:
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


def test_downloaded_bundle_matches_local_bundle(tmp_path: Path) -> None:
    downloaded = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
        revision=HF_REVISION,
        local_dir=tmp_path,
    )
    local_raw = np.load(LOCAL_BUNDLE_PATH, allow_pickle=False)
    remote_raw = np.load(downloaded, allow_pickle=False)
    with local_raw, remote_raw:
        assert local_raw.files == remote_raw.files
        for key in local_raw.files:
            np.testing.assert_equal(local_raw[key], remote_raw[key])

    bundle = machine_learning.load_bundle(Path(downloaded))
    local_bundle = machine_learning.load_bundle(LOCAL_BUNDLE_PATH)
    actual = machine_learning.screen(
        FIXTURE["job_description"],
        FIXTURE["resumes"],
        embed_fn=_fixture_embed,
        bundle=bundle,
    )
    expected = machine_learning.screen(
        FIXTURE["job_description"],
        FIXTURE["resumes"],
        embed_fn=_fixture_embed,
        bundle=local_bundle,
    )
    assert [(row["rank"], row["score"]) for row in actual["results"]] == [
        (row["rank"], row["score"]) for row in expected["results"]
    ]


def test_pinned_public_embedder_matches_reference() -> None:
    bundle = machine_learning.load_bundle(LOCAL_BUNDLE_PATH)
    embedder = machine_learning.get_embedder(bundle)
    vector = embedder([machine_learning.clean_text(FIXTURE["job_description"])])[0]
    reference = np.asarray(FIXTURE["job_vector"], dtype=float)
    assert float(vector @ reference) >= 0.999
    result = machine_learning.screen(
        FIXTURE["job_description"],
        FIXTURE["resumes"],
        embed_fn=embedder,
        bundle=bundle,
    )
    assert result["results"][0]["name"] == "data_analyst"
