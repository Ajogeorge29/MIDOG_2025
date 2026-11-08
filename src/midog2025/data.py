"""Small, dependency-light dataset loaders and leakage-safe split helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

TRACK2_LABEL_TO_ID = {"NMF": 0, "AMF": 1}
TRACK2_SCANNER_ALIASES = {"Hamammatsu XR": "Hamamatsu XR"}


def load_track2_table(track2_dir: Path, normalize_domains: bool = True) -> pd.DataFrame:
    """Load Track 2 metadata and attach validated local image paths.

    The source columns are preserved. Normalized values are added under new
    names so provenance remains visible.
    """

    csv_path = track2_dir / "MIDOG25_Atypical_Classification_Train_Set.csv"
    image_dir = track2_dir / "MIDOG25_Binary_Classification_Train_Set"
    frame = pd.read_csv(csv_path, dtype=str)
    frame["image_path"] = frame["image_id"].map(lambda value: str(image_dir / value))
    frame["label_id"] = frame["majority"].map(TRACK2_LABEL_TO_ID)
    if frame["label_id"].isna().any():
        unknown = sorted(frame.loc[frame["label_id"].isna(), "majority"].dropna().unique())
        raise ValueError(f"Unknown Track 2 labels: {unknown}")
    frame["label_id"] = frame["label_id"].astype("int8")
    frame["scanner_normalized"] = (
        frame["Scanner"].replace(TRACK2_SCANNER_ALIASES)
        if normalize_domains
        else frame["Scanner"]
    )
    frame["image_exists"] = frame["image_path"].map(lambda value: Path(value).is_file())
    return frame


def load_track2_image(row: pd.Series) -> tuple[Image.Image, int]:
    """Load one RGB patch and integer target from a metadata row."""

    with Image.open(row["image_path"]) as image:
        rgb = image.convert("RGB").copy()
    return rgb, int(row["label_id"])


def load_track1_tables(track1_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, str]]:
    """Return image, annotation, and category tables from MIDOGpp JSON."""

    with (track1_dir / "MIDOGpp.json").open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)
    images = pd.DataFrame(payload["images"])
    annotations = pd.DataFrame(payload["annotations"])
    categories = {int(item["id"]): str(item["name"]) for item in payload["categories"]}
    return images, annotations, categories


def corners_to_xywh(box: list[float]) -> list[float]:
    """Convert MIDOG's [x0,y0,x1,y1] annotation to standard COCO xywh."""

    x0, y0, x1, y1 = box
    return [x0, y0, x1 - x0, y1 - y0]


def assign_group_splits(
    frame: pd.DataFrame,
    group_column: str = "filename",
    stratify_column: str = "Tumor",
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    seed: int = 2025,
) -> pd.DataFrame:
    """Assign stable, domain-stratified splits without leaking source WSIs."""

    if train_fraction <= 0 or validation_fraction < 0:
        raise ValueError("Split fractions must be non-negative and train must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train + validation fractions must be less than one")
    group_domains = frame[[group_column, stratify_column]].drop_duplicates()
    domain_counts = group_domains.groupby(group_column)[stratify_column].nunique()
    if (domain_counts > 1).any():
        raise ValueError("A split group belongs to more than one stratification domain")

    assignments: dict[str, str] = {}
    for domain, subset in group_domains.groupby(stratify_column, dropna=False):
        groups = subset[group_column].astype(str).tolist()
        groups.sort(
            key=lambda group: hashlib.sha256(
                f"{seed}|{domain}|{group}".encode()
            ).hexdigest()
        )
        count = len(groups)
        train_end = round(count * train_fraction)
        validation_end = train_end + round(count * validation_fraction)
        for index, group in enumerate(groups):
            assignments[group] = (
                "train"
                if index < train_end
                else "validation"
                if index < validation_end
                else "test"
            )

    result = frame.copy()
    result["split"] = result[group_column].astype(str).map(assignments)
    return result
