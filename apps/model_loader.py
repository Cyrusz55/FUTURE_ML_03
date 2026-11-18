"""Lazy loading for the local screening bundle and embedding model."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from machine_learning import machine_learning

HF_REPO_ID = "cyrusnx/resume_screening_bundle"
HF_FILENAME = "screening_bundle.npz"
HF_REVISION = "v1"

_BUNDLE: dict[str, Any] | None = None
_EMBED_FN: Callable | None = None


def get_bundle() -> dict[str, Any]:
    """Load the local bundle, downloading the pinned public artefact only if absent."""
    global _BUNDLE
    if _BUNDLE is None:
        bundle_path = machine_learning.BUNDLE_PATH
        if not bundle_path.exists():
            from huggingface_hub import hf_hub_download

            bundle_path = Path(
                hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename=HF_FILENAME,
                    revision=HF_REVISION,
                    local_dir=machine_learning.PROJECT_ROOT / "model",
                )
            )
        _BUNDLE = machine_learning.load_bundle(bundle_path)
    return _BUNDLE


def get_embed_fn() -> Callable:
    """Return the cached embedding function."""
    global _EMBED_FN
    if _EMBED_FN is None:
        _EMBED_FN = machine_learning.get_embedder(get_bundle())
    return _EMBED_FN


def is_ready() -> bool:
    """Report readiness without triggering any model loading."""
    return _BUNDLE is not None and _EMBED_FN is not None
