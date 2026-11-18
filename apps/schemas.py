"""API request and response models."""

from __future__ import annotations

import importlib.metadata
import sys
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from machine_learning.machine_learning import (
    MAX_JD_CHARS,
    MAX_PDF_BYTES,
    MAX_PDF_PAGES,
    MAX_RESUME_CHARS,
    MAX_RESUMES,
    MIN_JD_CHARS,
    MIN_RESUME_CHARS,
)

MODEL_LIMITS = {
    "max_resumes": MAX_RESUMES,
    "min_resume_chars": MIN_RESUME_CHARS,
    "max_resume_chars": MAX_RESUME_CHARS,
    "min_jd_chars": MIN_JD_CHARS,
    "max_jd_chars": MAX_JD_CHARS,
    "max_pdf_bytes": MAX_PDF_BYTES,
    "max_pdf_pages": MAX_PDF_PAGES,
}


class ResumeIn(BaseModel):
    """A pasted resume."""

    model_config = ConfigDict(extra="forbid")

    name: Annotated[str | None, Field(max_length=100)] = None
    text: Annotated[str, Field(min_length=MIN_RESUME_CHARS, max_length=MAX_RESUME_CHARS)]

    @field_validator("name", mode="before")
    @classmethod
    def clean_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("Resume name must be a string")
        value = value.strip()
        return value or None

    @field_validator("text", mode="before")
    @classmethod
    def clean_text(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("Resume text must be a string")
        value = value.strip()
        if not value:
            raise ValueError("Resume text cannot be blank")
        return value


class ScreenRequest(BaseModel):
    """JSON screening request."""

    model_config = ConfigDict(extra="forbid")

    job_description: Annotated[
        str, Field(min_length=MIN_JD_CHARS, max_length=MAX_JD_CHARS)
    ]
    resumes: Annotated[list[ResumeIn], Field(min_length=1, max_length=MAX_RESUMES)]

    @field_validator("job_description", mode="before")
    @classmethod
    def clean_job_description(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("Job description must be a string")
        value = value.strip()
        if not value:
            raise ValueError("Job description cannot be blank")
        return value


class ResultItem(BaseModel):
    """A ranked candidate result."""

    model_config = ConfigDict(extra="ignore")

    name: str
    rank: int
    score: Annotated[float, Field(ge=0, le=100)]
    skill_match_pct: Annotated[float, Field(ge=0, le=100)]
    matched: list[str]
    missing_required: list[str]
    missing_preferred: list[str]
    strongest_signal: str
    summary: str


class SkippedItem(BaseModel):
    """An upload that could not be screened."""

    name: str
    reason: str


class ScreenResponse(BaseModel):
    """Screening response."""

    jd_skills: dict[str, float]
    results: list[ResultItem]
    skipped: list[SkippedItem] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Service health response."""

    status: Literal["ok", "degraded"]
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    """Model metadata and frontend limits."""

    hf_repo_id: str
    hf_revision: str
    embedding_model_id: str
    embedding_model_revision: str
    reference_resumes: int
    weights: dict[str, float]
    min_skill_weight: float
    limits: dict[str, int]
    library_versions: dict[str, str]


def library_versions() -> dict[str, str]:
    """Return versions used by the API without importing optional model code."""
    distributions = {
        "numpy": "numpy",
        "scipy": "scipy",
        "scikit_learn": "scikit-learn",
        "spacy": "spacy",
        "sentence_transformers": "sentence-transformers",
        "torch": "torch",
    }
    return {
        "python": sys.version.split()[0],
        **{
            key: importlib.metadata.version(distribution)
            for key, distribution in distributions.items()
        },
    }
