"""Create offline evaluation figures from the checked-in result files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = PROJECT_ROOT / "docs" / "images"
TARGETS = {
    "ci": IMAGES_DIR / "ci_graded_ndcg.png",
    "strict": IMAGES_DIR / "strict_vs_graded.png",
    "weights": IMAGES_DIR / "blend_weights.png",
}


def _hashes(directory: Path) -> dict[Path, str]:
    return {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.iterdir()
        if path.is_file()
    }


def _parse_ci_table(markdown: str) -> pd.DataFrame:
    heading = "### Graded NDCG@10 with 95% bootstrap confidence intervals"
    if heading not in markdown:
        raise ValueError(f"Missing section: {heading}")
    section = markdown.split(heading, 1)[1]
    lines = section.splitlines()
    table_lines: list[str] = []
    started = False
    for line in lines:
        if line.startswith("|"):
            table_lines.append(line)
            started = True
        elif started:
            break
    if len(table_lines) < 3:
        raise ValueError("The confidence interval section does not contain a markdown table")
    headers = [part.strip() for part in table_lines[0].strip("|").split("|")]
    expected = ["method", "NDCG@10 graded", "CI low", "CI high"]
    if headers != expected:
        raise ValueError(f"Confidence interval columns must be {expected}, got {headers}")
    rows = []
    for line in table_lines[2:]:
        values = [part.strip() for part in line.strip("|").split("|")]
        if len(values) != len(expected):
            raise ValueError("Malformed confidence interval table row")
        rows.append(values)
    if not rows:
        raise ValueError("Confidence interval table is empty")
    frame = pd.DataFrame(rows, columns=expected)
    for column in expected[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


def _save_figure(path: Path, figure: plt.Figure) -> None:
    if path.exists():
        print(f"Skipping existing figure: {path.name}")
    else:
        figure.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(figure)


def main() -> None:
    """Read metrics and create figures without contacting external services."""
    metrics = pd.read_csv(PROJECT_ROOT / "outputs" / "metrics.csv")
    required_metrics = {"method", "NDCG@10", "NDCG@10 graded"}
    if not required_metrics <= set(metrics.columns):
        missing = sorted(required_metrics - set(metrics.columns))
        raise ValueError(f"metrics.csv is missing {missing}")
    with (PROJECT_ROOT / "outputs" / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config.get("weights"), dict) or not config["weights"]:
        raise ValueError("config.json is missing non-empty weights")
    ci = _parse_ci_table(
        (PROJECT_ROOT / "outputs" / "readme_results.md").read_text(encoding="utf-8")
    )
    before = _hashes(IMAGES_DIR)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    methods = metrics["method"].astype(str).tolist()
    positions = np.arange(len(methods))
    ci_values = ci["NDCG@10 graded"].to_numpy()
    ci_low = ci_values - ci["CI low"].to_numpy()
    ci_high = ci["CI high"].to_numpy() - ci_values
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.errorbar(
        np.arange(len(ci)),
        ci_values,
        yerr=[ci_low, ci_high],
        fmt="o",
        capsize=4,
        color="#2563eb",
    )
    axis.set_xticks(np.arange(len(ci)), ci["method"], rotation=30, ha="right")
    axis.set_ylabel("Graded NDCG@10")
    axis.set_xlabel("Method")
    axis.set_title("Held-out test graded NDCG@10 with 95% bootstrap confidence intervals")
    axis.grid(axis="y", alpha=0.25)
    _save_figure(TARGETS["ci"], figure)

    width = 0.38
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.bar(positions - width / 2, metrics["NDCG@10"], width, label="Strict NDCG@10")
    axis.bar(positions + width / 2, metrics["NDCG@10 graded"], width, label="Graded NDCG@10")
    axis.set_xticks(positions, methods, rotation=30, ha="right")
    axis.set_ylabel("NDCG@10")
    axis.set_xlabel("Method")
    axis.set_title("Held-out test strict versus graded NDCG@10")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    _save_figure(TARGETS["strict"], figure)

    weights = config["weights"]
    labels = [str(key).upper() for key in weights]
    values = list(weights.values())
    figure, axis = plt.subplots(figsize=(6, 4.5))
    bars = axis.bar(labels, values, color=["#2563eb", "#0891b2", "#7c3aed"])
    axis.set_ylabel("Weight")
    axis.set_xlabel("Hybrid component")
    axis.set_title("Hybrid blend weights for held-out test scoring")
    axis.set_ylim(0, max(values) * 1.25)
    for bar, value in zip(bars, values, strict=True):
        axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:g}", ha="center", va="bottom")
    axis.grid(axis="y", alpha=0.25)
    _save_figure(TARGETS["weights"], figure)

    after = _hashes(IMAGES_DIR)
    for path, digest in before.items():
        if after.get(path) != digest:
            raise RuntimeError(f"Pre-existing image changed: {path.name}")
    print("Verified SHA-256 for all pre-existing images.")


if __name__ == "__main__":
    main()
