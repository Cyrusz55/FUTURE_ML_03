"""Bundle-backed resume screening logic."""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import spacy
from pypdf import PdfReader
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from spacy.matcher import PhraseMatcher

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = PROJECT_ROOT / "model" / "screening_bundle.npz"
MAX_RESUMES = 50
MIN_RESUME_CHARS = 50
MAX_RESUME_CHARS = 20_000
MIN_JD_CHARS = 30
MAX_JD_CHARS = 10_000
MAX_PDF_BYTES = 5 * 1024 * 1024
MAX_PDF_PAGES = 20
EMBED_BATCH_SIZE = 32

_LOGGER = logging.getLogger(__name__)
_EMBEDDER_MODEL: Any | None = None


def strip_leading_title(text: str, max_tokens: int = 15) -> str:
    """Remove the run of ALL CAPS words at the start of a resume."""
    tokens = str(text).split()
    i = 0
    while i < len(tokens) and i < max_tokens and tokens[i] == tokens[i].upper():
        i += 1
    return " ".join(tokens[i:])


def clean_text(text: str) -> str:
    """Lowercase text and preserve only terms used by the saved vectorizers."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z0-9+# ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def skill_coverage(resume_skills: set[str], weights: dict[str, float]) -> float:
    """Return the weighted share of job skills covered by a resume."""
    total = sum(weights.values())
    if total == 0:
        return 0.0
    covered = sum(weight for skill, weight in weights.items() if skill in resume_skills)
    return covered / total


def normalise_rows(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to 0..1 so different scorers can be blended fairly."""
    low = matrix.min(axis=1, keepdims=True)
    high = matrix.max(axis=1, keepdims=True)
    span = np.where(high - low == 0, 1, high - low)
    return (matrix - low) / span


def bm25_from_counts(
    counts_csc: sparse.spmatrix,
    doc_len: np.ndarray,
    query_terms: np.ndarray,
    idf: np.ndarray,
    avg_len: float,
    k1: float,
    b: float,
) -> np.ndarray:
    """Calculate BM25 scores from a count matrix."""
    scores = np.zeros(counts_csc.shape[0])
    length_norm = k1 * (1 - b + b * doc_len / avg_len)
    for term in query_terms:
        tf_column = counts_csc[:, term].toarray().ravel()
        scores += idf[term] * (tf_column * (k1 + 1)) / (tf_column + length_norm)
    return scores


def load_bundle(path: Path = BUNDLE_PATH) -> dict[str, Any]:
    """Load the immutable screening bundle without pickle support."""
    raw = np.load(path, allow_pickle=False)
    aliases = json.loads(str(raw["skill_aliases_json"]))

    skill_nlp = spacy.blank("en")
    skill_matcher = PhraseMatcher(skill_nlp.vocab, attr="LOWER")
    for canonical, alias_list in aliases.items():
        patterns = [skill_nlp.make_doc(clean_text(name)) for name in [canonical] + alias_list]
        skill_matcher.add(canonical, patterns)

    vocab = {term: i for i, term in enumerate(raw["vocabulary"].tolist())}
    ref_tf = sparse.csr_matrix(
        (raw["tf_data"], raw["tf_indices"], raw["tf_indptr"]),
        shape=tuple(int(x) for x in raw["tf_shape"]),
    ).tocsc()
    return {
        "config": json.loads(str(raw["config_json"])),
        "nlp": skill_nlp,
        "matcher": skill_matcher,
        "vectorizer": CountVectorizer(stop_words="english", vocabulary=vocab),
        "ref_tf": ref_tf,
        "idf": raw["idf"],
        "doc_len": raw["doc_len"],
        "ref_emb": raw["ref_embeddings"],
        "ref_ids": raw["ref_ids"],
        "ref_categories": raw["ref_categories"],
        "ref_skill_sets": [set(s) for s in json.loads(str(raw["ref_skill_sets_json"]))],
    }


_BUNDLE: dict[str, Any] | None = None


def get_bundle() -> dict[str, Any]:
    """Return the process-cached screening bundle."""
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = load_bundle()
    return _BUNDLE


def bundle_extract_skills(bundle: dict[str, Any], text: str) -> set[str]:
    """Extract canonical skills using the aliases stored in the bundle."""
    doc = bundle["nlp"].make_doc(clean_text(text))
    return {
        bundle["nlp"].vocab.strings[match_id]
        for match_id, _start, _end in bundle["matcher"](doc)
    }


def bundle_jd_weights(bundle: dict[str, Any], jd_text: str) -> dict[str, float]:
    """Assign required or preferred weights to skills in a job description."""
    config = bundle["config"]
    weights: dict[str, float] = {}
    for sentence in re.split(r"(?<=[.!?\n])\s+", str(jd_text)):
        is_preferred = any(cue in sentence.lower() for cue in config["preferred_cues"])
        weight = config["preferred_weight"] if is_preferred else config["required_weight"]
        for skill in bundle_extract_skills(bundle, sentence):
            weights[skill] = max(weights.get(skill, 0.0), weight)
    return weights


def score_uploaded(
    bundle: dict[str, Any],
    jd_text: str,
    resume_texts: list[str],
    embed_fn: Callable[[list[str]], np.ndarray],
    return_reference: bool = False,
) -> dict[str, Any]:
    """Score uploaded resumes against the reference pool."""
    started = time.perf_counter()
    config = bundle["config"]
    weights = config["weights"]
    n_ref = len(bundle["ref_skill_sets"])
    bodies = [strip_leading_title(text) for text in resume_texts]
    cleaned = [clean_text(body) for body in bodies]
    jd_clean = clean_text(jd_text)

    vectors = embed_fn([jd_clean] + cleaned)
    query_vec, upload_emb = vectors[0], vectors[1:]
    ref_embed = bundle["ref_emb"] @ query_vec
    embed_raw = np.concatenate([ref_embed, upload_emb @ query_vec])

    vectorizer = bundle["vectorizer"]
    query_terms = np.unique(vectorizer.transform([jd_clean]).indices)
    avg_len = bundle["doc_len"].mean()
    k1, b = config["bm25_k1"], config["bm25_b"]
    ref_bm25 = bm25_from_counts(
        bundle["ref_tf"], bundle["doc_len"], query_terms, bundle["idf"], avg_len, k1, b
    )
    upload_counts = vectorizer.transform(cleaned).tocsc()
    upload_len = np.asarray(upload_counts.sum(axis=1)).ravel()
    upload_bm25 = bm25_from_counts(
        upload_counts, upload_len, query_terms, bundle["idf"], avg_len, k1, b
    )
    bm25_raw = np.concatenate([ref_bm25, upload_bm25])

    jd_weights = bundle_jd_weights(bundle, jd_text)
    upload_skills = [bundle_extract_skills(bundle, body) for body in bodies]
    coverage = np.array(
        [skill_coverage(s, jd_weights) for s in bundle["ref_skill_sets"] + upload_skills]
    )

    parts = {
        "bm25": normalise_rows(bm25_raw[None, :])[0],
        "embedding": normalise_rows(embed_raw[None, :])[0],
        "skills": normalise_rows(coverage[None, :])[0],
    }
    final = sum(weights[name] * parts[name] for name in parts)
    required = {s for s, w in jd_weights.items() if w == config["required_weight"]}
    preferred = {s for s, w in jd_weights.items() if w == config["preferred_weight"]}

    results = []
    for i, have in enumerate(upload_skills):
        idx = n_ref + i
        results.append(
            {
                "score": round(float(final[idx]) * 100, 1),
                "skill_match_pct": int(round(float(coverage[idx]) * 100)),
                "matched": sorted(have & set(jd_weights)),
                "missing_required": sorted(required - have),
                "missing_preferred": sorted(preferred - have),
                "parts": {
                    name: round(float(weights[name] * parts[name][idx]), 4) for name in parts
                },
            }
        )
    order = sorted(range(len(results)), key=lambda i: (-results[i]["score"], i))
    for rank, i in enumerate(order, start=1):
        results[i]["rank"] = rank

    _LOGGER.debug("Scored %d resumes in %.3fs", len(resume_texts), time.perf_counter() - started)
    output: dict[str, Any] = {"results": results, "jd_skill_weights": jd_weights}
    if return_reference:
        output["reference"] = {
            "bm25": ref_bm25,
            "embedding": ref_embed,
            "skills": coverage[:n_ref],
        }
    return output


def get_embedder(bundle: dict[str, Any]) -> Callable[[list[str]], np.ndarray]:
    """Load and configure the bundle's exact embedding model lazily."""
    global _EMBEDDER_MODEL
    config = bundle["config"]
    revision = config.get("embedding_model_revision")
    if revision is None:
        raise ValueError("Bundle embedding_model_revision must not be None")
    if _EMBEDDER_MODEL is None:
        from sentence_transformers import SentenceTransformer

        _EMBEDDER_MODEL = SentenceTransformer(
            config["embedding_model_id"], revision=revision
        )
        _EMBEDDER_MODEL.max_seq_length = config["embedding_max_seq_length"]
    model = _EMBEDDER_MODEL

    def embed(texts: list[str]) -> np.ndarray:
        return np.asarray(
            model.encode(
                texts,
                batch_size=EMBED_BATCH_SIZE,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )

    return embed


def extract_pdf_text(data: bytes) -> str:
    """Extract text from a small, unencrypted PDF without OCR."""
    if len(data) > MAX_PDF_BYTES:
        raise ValueError("PDF exceeds the 5 MiB size limit")
    try:
        reader = PdfReader(BytesIO(data))
    except Exception as exc:
        raise ValueError("Unable to read PDF") from exc
    if reader.is_encrypted:
        raise ValueError("Encrypted PDFs are not supported")
    if len(reader.pages) > MAX_PDF_PAGES:
        raise ValueError("PDF exceeds the 20 page limit")
    try:
        text = "\n".join((page.extract_text() or "") for page in reader.pages).strip()
    except Exception as exc:
        raise ValueError("Unable to extract PDF text") from exc
    if not text:
        raise ValueError("PDF contains no extractable text; scanned PDFs are not supported")
    return text


def explain_candidate(result: dict[str, Any]) -> tuple[str, str]:
    """Return the strongest score signal and the notebook's plain-language summary."""
    contributions = result["parts"]
    labels = {"bm25": "keyword match", "embedding": "meaning match", "skills": "skill coverage"}
    strongest = labels[max(contributions, key=contributions.get)]
    text = (
        f"Ranked {result['rank']} with a fit score of {result['score']}. "
        f"Strongest signal: {strongest}. "
    )
    text += f"Skill match: {result['skill_match_pct']} percent. "
    matches = ", ".join(result["matched"]) if result["matched"] else "no listed skills"
    text += f"Matches: {matches}. "
    if result["missing_required"]:
        text += "Missing required skills: " + ", ".join(result["missing_required"]) + ". "
    if result["missing_preferred"]:
        text += "Missing preferred skills: " + ", ".join(result["missing_preferred"]) + "."
    return strongest, text.strip()


def _validate_screen_input(job_description: str, resumes: list[dict[str, str]]) -> None:
    if not isinstance(job_description, str) or not MIN_JD_CHARS <= len(
        job_description
    ) <= MAX_JD_CHARS:
        raise ValueError(
            f"Job description must contain {MIN_JD_CHARS} to {MAX_JD_CHARS} characters"
        )
    if not 1 <= len(resumes) <= MAX_RESUMES:
        raise ValueError(f"Provide between 1 and {MAX_RESUMES} resumes")
    for resume in resumes:
        text = resume.get("text", "")
        if not isinstance(text, str) or not MIN_RESUME_CHARS <= len(text) <= MAX_RESUME_CHARS:
            raise ValueError(
                f"Each resume must contain {MIN_RESUME_CHARS} to {MAX_RESUME_CHARS} characters"
            )
        if not isinstance(resume.get("name"), str):
            raise ValueError("Each resume must have a name")


def screen(
    job_description: str,
    resumes: list[dict[str, str]],
    embed_fn: Callable[[list[str]], np.ndarray] | None = None,
    bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and rank pasted resume texts using one scoring path."""
    _validate_screen_input(job_description, resumes)
    active_bundle = bundle or get_bundle()
    active_embed_fn = embed_fn or get_embedder(active_bundle)
    scored = score_uploaded(
        active_bundle,
        job_description,
        [resume["text"] for resume in resumes],
        active_embed_fn,
    )
    results = []
    for resume, result in zip(resumes, scored["results"], strict=True):
        strongest, summary = explain_candidate(result)
        results.append(
            {
                "name": resume["name"],
                "rank": result["rank"],
                "score": result["score"],
                "skill_match_pct": result["skill_match_pct"],
                "matched": result["matched"],
                "missing_required": result["missing_required"],
                "missing_preferred": result["missing_preferred"],
                "strongest_signal": strongest,
                "summary": summary,
            }
        )
    results.sort(key=lambda result: result["rank"])
    return {"jd_skills": scored["jd_skill_weights"], "results": results}
