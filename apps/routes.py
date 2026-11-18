"""HTTP routes for resume screening."""

from __future__ import annotations

import logging
import re
import time
from pathlib import PurePath
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from machine_learning import machine_learning

from . import model_loader
from .schemas import (
    MODEL_LIMITS,
    HealthResponse,
    ModelInfoResponse,
    ScreenRequest,
    ScreenResponse,
    SkippedItem,
    library_versions,
)

LOGGER = logging.getLogger(__name__)
MAX_TOTAL_UPLOAD_BYTES = 25 * 1024 * 1024
router = APIRouter()


def get_screening_context() -> tuple[dict[str, Any], Any]:
    """Load scoring dependencies or return a short service-unavailable error."""
    try:
        return model_loader.get_bundle(), model_loader.get_embed_fn()
    except Exception as exc:
        LOGGER.warning("Model loading failed")
        raise HTTPException(status_code=503, detail="Screening model is unavailable") from exc


def check_content_length(request: Request) -> None:
    """Reject requests that exceed the multipart body limit."""
    value = request.headers.get("content-length")
    if value is not None:
        try:
            too_large = int(value) > MAX_TOTAL_UPLOAD_BYTES
        except ValueError:
            too_large = False
        if too_large:
            raise HTTPException(status_code=413, detail="Upload is too large")


def _safe_filename(filename: str | None) -> str:
    name = PurePath((filename or "").replace("\\", "/")).name
    name = re.sub(r"[\x00-\x1f\x7f]", "", name).strip()
    return name[:100] or "Uploaded file"


def _screen(
    job_description: str,
    resumes: list[dict[str, str]],
    skipped: list[SkippedItem],
    context: tuple[dict[str, Any], Any],
) -> ScreenResponse:
    """Run the sole scoring path and map expected input errors to HTTP 422."""
    try:
        output = machine_learning.screen(
            job_description,
            resumes,
            embed_fn=context[1],
            bundle=context[0],
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    output["skipped"] = skipped
    return ScreenResponse.model_validate(output)


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    loaded = model_loader.is_ready()
    return HealthResponse(status="ok" if loaded else "degraded", model_loaded=loaded)


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    try:
        bundle = model_loader.get_bundle()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Screening model is unavailable") from exc
    config = bundle["config"]
    return ModelInfoResponse(
        hf_repo_id=model_loader.HF_REPO_ID,
        hf_revision=model_loader.HF_REVISION,
        embedding_model_id=config["embedding_model_id"],
        embedding_model_revision=config["embedding_model_revision"],
        reference_resumes=config["reference_resumes"],
        weights=config["weights"],
        min_skill_weight=config["min_skill_weight"],
        limits={
            **MODEL_LIMITS,
            "max_total_upload_bytes": MAX_TOTAL_UPLOAD_BYTES,
        },
        library_versions=library_versions(),
    )


@router.post("/screen", response_model=ScreenResponse)
def screen_json(
    request: ScreenRequest,
    _content_length: Annotated[None, Depends(check_content_length)],
    context: Annotated[tuple[dict[str, Any], Any], Depends(get_screening_context)],
) -> ScreenResponse:
    resumes = [
        {"name": resume.name or f"Pasted resume {index}", "text": resume.text}
        for index, resume in enumerate(request.resumes, start=1)
    ]
    return _screen(request.job_description, resumes, [], context)


@router.post("/screen/files", response_model=ScreenResponse)
def screen_files(
    job_description: Annotated[str, Form()],
    files: Annotated[list[UploadFile] | None, File()] = None,
    pasted_texts: Annotated[list[str] | None, Form()] = None,
    _content_length: Annotated[None, Depends(check_content_length)] = None,
    context: Annotated[tuple[dict[str, Any], Any], Depends(get_screening_context)] = None,
) -> ScreenResponse:
    if context is None:
        raise HTTPException(status_code=503, detail="Screening model is unavailable")
    files = files or []
    pasted_texts = pasted_texts or []
    job_description = job_description.strip()
    if not job_description:
        raise HTTPException(status_code=422, detail="Job description cannot be blank")
    total_inputs = len(files) + len(pasted_texts)
    if total_inputs < 1 or total_inputs > machine_learning.MAX_RESUMES:
        raise HTTPException(
            status_code=422,
            detail=f"Provide between 1 and {machine_learning.MAX_RESUMES} resumes",
        )

    resumes: list[dict[str, str]] = []
    skipped: list[SkippedItem] = []
    started = time.perf_counter()
    for upload in files:
        name = _safe_filename(upload.filename)
        try:
            data = upload.file.read(machine_learning.MAX_PDF_BYTES + 1)
            if len(data) > machine_learning.MAX_PDF_BYTES:
                raise ValueError("file is too large")
            suffix = PurePath(name).suffix.lower()
            if suffix == ".pdf":
                if not data.startswith(b"%PDF"):
                    raise ValueError("invalid PDF")
                text = machine_learning.extract_pdf_text(data)[: machine_learning.MAX_RESUME_CHARS]
            elif suffix == ".txt":
                text = data.decode("utf-8", errors="replace")
            else:
                raise ValueError("unsupported file type")
            text = text.strip()
            if not machine_learning.MIN_RESUME_CHARS <= len(
                text
            ) <= machine_learning.MAX_RESUME_CHARS:
                raise ValueError("resume length is outside the allowed range")
            resumes.append({"name": name, "text": text})
        except (OSError, UnicodeError, ValueError):
            skipped.append(SkippedItem(name=name, reason="File could not be screened"))

    for index, text in enumerate(pasted_texts, start=1):
        cleaned = text.strip()
        name = f"Pasted resume {index}"
        if not machine_learning.MIN_RESUME_CHARS <= len(
            cleaned
        ) <= machine_learning.MAX_RESUME_CHARS:
            skipped.append(
                SkippedItem(name=name, reason="Resume length is outside the allowed range")
            )
        else:
            resumes.append({"name": name, "text": cleaned})

    if not resumes:
        raise HTTPException(status_code=422, detail="No readable resumes were provided")
    response = _screen(job_description, resumes, skipped, context)
    LOGGER.debug(
        "Processed %d files, %d pasted resumes in %.3fs",
        len(files),
        len(pasted_texts),
        time.perf_counter() - started,
    )
    return response
