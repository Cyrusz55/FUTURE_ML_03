"""Validate and publish the screening bundle to Hugging Face."""

from __future__ import annotations

import json
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from huggingface_hub import HfApi

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from apps.model_loader import HF_FILENAME, HF_REPO_ID, HF_REVISION  # noqa: E402

EXPECTED_KEYS = {
    "tf_data",
    "tf_indices",
    "tf_indptr",
    "tf_shape",
    "vocabulary",
    "idf",
    "doc_len",
    "ref_embeddings",
    "ref_ids",
    "ref_categories",
    "ref_skill_sets_json",
    "skill_aliases_json",
    "config_json",
}


def _read_bundle(path: Path) -> tuple[dict[str, tuple[int, ...]], dict[str, Any]]:
    try:
        raw = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Unable to load bundle: {exc}") from exc
    with raw:
        keys = set(raw.files)
        if keys != EXPECTED_KEYS:
            missing = sorted(EXPECTED_KEYS - keys)
            extra = sorted(keys - EXPECTED_KEYS)
            raise ValueError(
                f"Bundle keys do not match expected keys; missing={missing}, extra={extra}"
            )
        try:
            config = json.loads(str(raw["config_json"]))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("Bundle config_json is not valid JSON") from exc
        revision = config.get("embedding_model_revision")
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("Bundle config_json must contain a non-empty embedding_model_revision")
        shapes = {key: tuple(raw[key].shape) for key in raw.files}
    return shapes, config


def _model_card(
    shapes: dict[str, tuple[int, ...]],
    config: dict[str, Any],
    metrics: str,
    results: str,
    requirements: str,
) -> str:
    shape_lines = "\n".join(f"- `{key}`: `{shape}`" for key, shape in shapes.items())
    return f"""---
library_name: numpy
pipeline_tag: sentence-similarity
license: other
tags:
- resume-screening
- ranking
- reference-bundle
---

# Resume screening reference bundle

This file is a reference bundle for scoring uploaded resumes. It contains derived
numbers from 2,481 Kaggle resumes and never contains resume text.

## Bundle contents

{shape_lines}

## Use

Load the file with `numpy.load(..., allow_pickle=False)` and pass the resulting
bundle through `machine_learning.score_uploaded`.

The hybrid weights are `{json.dumps(config["weights"], sort_keys=True)}`. The
embedding model is `{config["embedding_model_id"]}` at revision
`{config["embedding_model_revision"]}`.

## Test metrics

The following values are read from `outputs/metrics.csv`:

```csv
{metrics.strip()}
```

Confidence intervals are read from `outputs/readme_results.md`:

```text
{results.strip()}
```

Pinned library versions are read from `requirements.txt`:

```text
{requirements.strip()}
```

## Limitations

Category labels are a proxy, the ADVOCATE category was excluded, and job
descriptions are synthetic. Scores are relative to the reference pool. Source
dataset licences were not verified.
"""


def main() -> int:
    """Validate and upload the pinned bundle revision."""
    load_dotenv(PROJECT_ROOT / ".env", encoding="utf-8-sig")
    token = os.getenv("HF_TOKEN")
    if not token:
        print("HF_TOKEN is missing from .env", file=sys.stderr)
        return 1

    bundle_path = PROJECT_ROOT / "model" / HF_FILENAME
    try:
        shapes, config = _read_bundle(bundle_path)
        metrics = (PROJECT_ROOT / "outputs" / "metrics.csv").read_text(encoding="utf-8")
        results = (PROJECT_ROOT / "outputs" / "readme_results.md").read_text(encoding="utf-8")
        requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"Bundle validation failed: {exc}", file=sys.stderr)
        return 1

    api = HfApi(token=token)
    try:
        api.create_repo(
            repo_id=HF_REPO_ID,
            repo_type="model",
            private=False,
            exist_ok=True,
        )
        refs = api.list_repo_refs(repo_id=HF_REPO_ID, repo_type="model")
        if any(tag.name == HF_REVISION for tag in refs.tags):
            raise RuntimeError(
                f"Tag {HF_REVISION} already exists; bump HF_REVISION in "
                "apps/model_loader.py instead"
            )
        api.upload_file(
            path_or_fileobj=str(bundle_path),
            path_in_repo=HF_FILENAME,
            repo_id=HF_REPO_ID,
            repo_type="model",
            commit_message=f"Upload {HF_FILENAME}",
        )
        card = _model_card(shapes, config, metrics, results, requirements)
        commit = api.upload_file(
            path_or_fileobj=BytesIO(card.encode("utf-8")),
            path_in_repo="README.md",
            repo_id=HF_REPO_ID,
            repo_type="model",
            commit_message="Add model card",
        )
        api.create_tag(
            repo_id=HF_REPO_ID,
            tag=HF_REVISION,
            revision=commit.oid,
            repo_type="model",
            tag_message=f"Release {HF_REVISION}",
        )
    except Exception as exc:
        print(f"Hugging Face upload failed: {exc}", file=sys.stderr)
        return 1
    print(f"Uploaded {HF_FILENAME} to {HF_REPO_ID} at tag {HF_REVISION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
