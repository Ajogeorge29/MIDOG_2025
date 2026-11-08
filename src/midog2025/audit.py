"""Deep integrity and distribution audit for MIDOG 2025 training data."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import html
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageFile

from .data import TRACK2_SCANNER_ALIASES, assign_group_splits
from .paths import DATA_ROOT, REPORT_ROOT

ImageFile.LOAD_TRUNCATED_IMAGES = False
sns.set_theme(style="whitegrid", context="notebook")

REQUIRED_TRACK2_COLUMNS = {
    "image_id",
    "filename",
    "coordinateX",
    "coordinateY",
    "Tumor",
    "Scanner",
    "Origin",
    "Species",
    "expert1",
    "expert2",
    "expert3",
    "majority",
}
LABELS = {"AMF", "NMF"}
COLORS = {"AMF": "#8e44ad", "NMF": "#18a999"}


@dataclass
class Issue:
    severity: str
    scope: str
    finding: str
    metric: str
    recommendation: str


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def cohen_kappa(left: pd.Series, right: pd.Series) -> dict[str, float | int | None]:
    mask = left.notna() & right.notna() & left.isin(LABELS) & right.isin(LABELS)
    a, b = left[mask], right[mask]
    n = len(a)
    if n == 0:
        return {"n": 0, "agreement": None, "kappa": None}
    agreement = float((a == b).mean())
    expected = sum(float((a == label).mean() * (b == label).mean()) for label in LABELS)
    kappa = (agreement - expected) / (1 - expected) if expected < 1 else 1.0
    return {"n": n, "agreement": agreement, "kappa": float(kappa)}


def strict_vote_majority(row: pd.Series) -> str | None:
    votes = [row[key] for key in ("expert1", "expert2", "expert3") if row[key] in LABELS]
    if not votes:
        return None
    counts = Counter(votes)
    label, count = counts.most_common(1)[0]
    return label if count > len(votes) / 2 else None


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_patch(path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "image_id": path.name,
        "path": str(path),
        "file_bytes": path.stat().st_size,
        "sha256": None,
        "average_hash": None,
        "width": None,
        "height": None,
        "mode": None,
        "mean_r": None,
        "mean_g": None,
        "mean_b": None,
        "std_intensity": None,
        "tissue_fraction": None,
        "error": None,
    }
    try:
        row["sha256"] = _hash_file(path)
        with Image.open(path) as image:
            image.load()
            row["width"], row["height"] = image.size
            row["mode"] = image.mode
            sample_image = image.convert("RGB")
            sample_image.thumbnail((96, 96), Image.Resampling.LANCZOS)
            sample = np.asarray(sample_image, dtype=np.float32)
        means = sample.mean(axis=(0, 1))
        row["mean_r"], row["mean_g"], row["mean_b"] = [float(x) for x in means]
        row["std_intensity"] = float(sample.std())
        row["tissue_fraction"] = float((sample.mean(axis=2) < 235).mean())
        gray = np.asarray(
            Image.fromarray(sample.astype(np.uint8)).convert("L").resize((8, 8)),
            dtype=np.float32,
        )
        bits = (gray >= gray.mean()).flatten()
        row["average_hash"] = f"{sum(int(bit) << index for index, bit in enumerate(bits)):016x}"
    except Exception as exc:  # audit must record corrupt files rather than stop
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def inspect_patch_directory(image_dir: Path, workers: int) -> pd.DataFrame:
    files = sorted(path for path in image_dir.glob("*.png") if path.is_file())
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(inspect_patch, files))
    return pd.DataFrame(rows)


def _ratio_to_float(value: Any) -> float | None:
    try:
        if isinstance(value, tuple) and len(value) == 2:
            return float(value[0]) / float(value[1])
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def inspect_tiff_header(path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "file_name": path.name,
        "file_bytes": path.stat().st_size,
        "width": None,
        "height": None,
        "mode": None,
        "frames": None,
        "compression": None,
        "resolution_unit": None,
        "x_resolution": None,
        "y_resolution": None,
        "mpp_x": None,
        "mpp_y": None,
        "error": None,
    }
    try:
        with Image.open(path) as image:
            row["width"], row["height"] = image.size
            row["mode"] = image.mode
            row["frames"] = getattr(image, "n_frames", 1)
            row["compression"] = image.info.get("compression")
            tags = getattr(image, "tag_v2", {})
            row["x_resolution"] = _ratio_to_float(tags.get(282))
            row["y_resolution"] = _ratio_to_float(tags.get(283))
            row["resolution_unit"] = int(tags.get(296, 1))
        units_per_micron = {2: 1 / 25400, 3: 1 / 10000}
        factor = units_per_micron.get(row["resolution_unit"])
        if factor and row["x_resolution"]:
            row["mpp_x"] = 1 / (row["x_resolution"] * factor)
        if factor and row["y_resolution"]:
            row["mpp_y"] = 1 / (row["y_resolution"] * factor)
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def inspect_tiff_directory(image_dir: Path, workers: int = 8) -> pd.DataFrame:
    files = sorted(
        path for path in image_dir.glob("*.tiff") if path.is_file() and path.stat().st_size > 0
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(inspect_tiff_header, files))
    return pd.DataFrame(rows)


def _save_count_plot(
    series: pd.Series,
    path: Path,
    title: str,
    xlabel: str = "Count",
    palette: list[str] | None = None,
) -> None:
    counts = series.value_counts(dropna=False).sort_values()
    fig_height = max(3.2, 0.45 * len(counts) + 1.5)
    fig, axis = plt.subplots(figsize=(9, fig_height))
    colors = palette or sns.color_palette("viridis", len(counts))
    counts.plot.barh(ax=axis, color=colors)
    axis.set_title(title, loc="left", weight="bold")
    axis.set_xlabel(xlabel)
    axis.set_ylabel("")
    for container in axis.containers:
        axis.bar_label(container, padding=3, fmt="{:,.0f}")
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def audit_track2(track2_dir: Path, report_dir: Path, workers: int) -> tuple[dict[str, Any], list[Issue], pd.DataFrame]:
    csv_path = track2_dir / "MIDOG25_Atypical_Classification_Train_Set.csv"
    image_dir = track2_dir / "MIDOG25_Binary_Classification_Train_Set"
    if not csv_path.exists():
        raise FileNotFoundError(f"Track 2 CSV not found: {csv_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Track 2 image directory not found: {image_dir}")

    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=True)
    missing_columns = sorted(REQUIRED_TRACK2_COLUMNS - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Track 2 CSV is missing columns: {missing_columns}")
    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].str.strip()
    for column in ("coordinateX", "coordinateY"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    inventory = inspect_patch_directory(image_dir, workers)
    inventory.to_csv(report_dir / "track2_image_inventory.csv", index=False)
    file_names = set(inventory["image_id"])
    csv_names = set(frame["image_id"].dropna())
    missing_files = sorted(csv_names - file_names)
    orphan_files = sorted(file_names - csv_names)
    duplicate_id_rows = frame[frame.duplicated("image_id", keep=False)].sort_values("image_id")
    duplicate_id_rows.to_csv(report_dir / "track2_duplicate_image_ids.csv", index=False)
    duplicate_coordinate_rows = int(
        frame.duplicated(["filename", "coordinateX", "coordinateY"], keep=False).sum()
    )
    exact_duplicate_files = int(inventory.duplicated("sha256", keep=False).sum())
    perceptual_collision_files = int(inventory.duplicated("average_hash", keep=False).sum())
    corrupt_files = int(inventory["error"].notna().sum())

    frame["computed_majority"] = frame.apply(strict_vote_majority, axis=1)
    frame["scanner_normalized"] = frame["Scanner"].replace(TRACK2_SCANNER_ALIASES)
    split_frame = assign_group_splits(frame)
    split_frame[
        [
            "image_id",
            "filename",
            "coordinateX",
            "coordinateY",
            "Tumor",
            "scanner_normalized",
            "Origin",
            "Species",
            "majority",
            "split",
        ]
    ].to_csv(report_dir / "track2_recommended_split.csv", index=False)
    checkable = frame["computed_majority"].notna()
    inconsistent_majority = int(
        (frame.loc[checkable, "computed_majority"] != frame.loc[checkable, "majority"]).sum()
    )
    invalid_labels = {
        column: sorted(set(frame[column].dropna()) - LABELS)
        for column in ("expert1", "expert2", "expert3", "majority")
    }
    invalid_labels = {key: values for key, values in invalid_labels.items() if values}
    scanner_values = set(frame["Scanner"].dropna())

    kappas: dict[str, dict[str, float | int | None]] = {}
    experts = ["expert1", "expert2", "expert3"]
    for index, left in enumerate(experts):
        for right in experts[index + 1 :]:
            kappas[f"{left} vs {right}"] = cohen_kappa(frame[left], frame[right])

    vote_counts = frame[experts].isin(LABELS).sum(axis=1)
    unanimous = frame[experts].apply(
        lambda row: len({value for value in row if value in LABELS}) == 1, axis=1
    )
    class_counts = frame["majority"].value_counts().to_dict()
    amf = int(class_counts.get("AMF", 0))
    nmf = int(class_counts.get("NMF", 0))
    imbalance_ratio = nmf / amf if amf else math.inf

    issues: list[Issue] = []
    if len(frame) != frame["image_id"].nunique(dropna=True):
        issues.append(
            Issue(
                "high",
                "Track 2 metadata",
                "CSV image identifiers are not one-to-one with rows.",
                f"{len(frame):,} rows; {frame['image_id'].nunique():,} unique IDs; "
                f"{len(duplicate_id_rows):,} rows involved",
                "Resolve duplicate IDs before splitting or training; use the exported duplicate table.",
            )
        )
    if missing_files or orphan_files:
        issues.append(
            Issue(
                "high",
                "Track 2 files",
                "CSV-to-file membership is not bijective.",
                f"{len(missing_files)} referenced files missing; {len(orphan_files)} orphan files",
                "Reconcile IDs and do not silently drop unmatched rows/files.",
            )
        )
    if corrupt_files:
        issues.append(
            Issue(
                "critical",
                "Track 2 files",
                "One or more PNG files cannot be decoded.",
                f"{corrupt_files} corrupt files",
                "Re-download the archive and verify its MD5 before training.",
            )
        )
    if inconsistent_majority:
        issues.append(
            Issue(
                "high",
                "Track 2 labels",
                "Published majority disagrees with a strict majority of available expert votes.",
                f"{inconsistent_majority} rows",
                "Review these rows and retain published labels only with documented adjudication.",
            )
        )
    if duplicate_coordinate_rows:
        issues.append(
            Issue(
                "medium",
                "Track 2 metadata",
                "Multiple rows share the same source WSI coordinate.",
                f"{duplicate_coordinate_rows} rows involved",
                "Deduplicate by source filename and coordinate before evaluation splits.",
            )
        )
    if {"Hamamatsu XR", "Hamammatsu XR"}.issubset(scanner_values):
        typo_count = int((frame["Scanner"] == "Hamammatsu XR").sum())
        issues.append(
            Issue(
                "medium",
                "Track 2 domains",
                "A scanner name typo creates a false extra domain.",
                f"Hamammatsu XR: {typo_count:,} rows",
                "Normalize `Hamammatsu XR` to `Hamamatsu XR` in a derived metadata column.",
            )
        )
    issues.append(
        Issue(
            "high",
            "Model validation",
            "Patches from the same source WSI are strongly correlated.",
            f"{frame['filename'].nunique():,} source WSIs for {len(frame):,} rows",
            "Split by source filename (and preferably domain), never randomly by patch.",
        )
    )
    issues.append(
        Issue(
            "medium",
            "Track 2 labels",
            "The binary target is substantially imbalanced.",
            f"NMF:AMF = {imbalance_ratio:.2f}:1 ({nmf:,} vs {amf:,})",
            "Report balanced accuracy/AUROC and use class-aware sampling or loss weighting.",
        )
    )

    _save_count_plot(
        frame["majority"],
        report_dir / "track2_class_distribution.png",
        "Track 2 class distribution",
        palette=[COLORS.get(label, "#777") for label in frame["majority"].value_counts().sort_values().index],
    )
    _save_count_plot(frame["Tumor"], report_dir / "track2_tumor_distribution.png", "Patches by tumor domain")
    _save_count_plot(frame["Scanner"], report_dir / "track2_scanner_distribution.png", "Patches by scanner")

    domain_class = pd.crosstab(frame["Tumor"], frame["majority"], normalize="index")
    fig, axis = plt.subplots(figsize=(10, max(4, 0.55 * len(domain_class))))
    domain_class.reindex(columns=["NMF", "AMF"], fill_value=0).plot.barh(
        stacked=True, ax=axis, color=[COLORS["NMF"], COLORS["AMF"]]
    )
    axis.set_title("Class balance within each tumor domain", loc="left", weight="bold")
    axis.set_xlabel("Share of patches")
    axis.set_ylabel("")
    axis.legend(title="Majority label", bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(report_dir / "track2_class_by_tumor.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "csv_rows": len(frame),
        "unique_image_ids": int(frame["image_id"].nunique(dropna=True)),
        "png_files": len(inventory),
        "source_wsis": int(frame["filename"].nunique(dropna=True)),
        "tumor_domains": int(frame["Tumor"].nunique(dropna=True)),
        "class_counts": {str(key): int(value) for key, value in class_counts.items()},
        "nmf_to_amf_ratio": imbalance_ratio,
        "missing_files": missing_files,
        "orphan_files": orphan_files,
        "duplicate_id_rows": len(duplicate_id_rows),
        "duplicate_coordinate_rows": duplicate_coordinate_rows,
        "exact_duplicate_file_rows": exact_duplicate_files,
        "perceptual_hash_collision_rows": perceptual_collision_files,
        "corrupt_files": corrupt_files,
        "invalid_labels": invalid_labels,
        "strict_majority_checkable_rows": int(checkable.sum()),
        "inconsistent_majority_rows": inconsistent_majority,
        "expert_vote_coverage": {str(key): int(value) for key, value in vote_counts.value_counts().sort_index().items()},
        "unanimous_available_votes": int(unanimous.sum()),
        "pairwise_agreement": kappas,
        "dimensions": {
            f"{width}x{height}": int(count)
            for (width, height), count in inventory.groupby(
                ["width", "height"], dropna=False
            ).size().items()
        },
        "mean_rgb": {
            channel: float(inventory[f"mean_{channel}"].mean()) for channel in ("r", "g", "b")
        },
        "mean_tissue_fraction": float(inventory["tissue_fraction"].mean()),
        "domains": {
            column: {str(key): int(value) for key, value in frame[column].value_counts().items()}
            for column in ("Tumor", "Scanner", "Origin", "Species")
        },
    }
    return summary, issues, frame


def audit_track1(
    track1_dir: Path,
    report_dir: Path,
    track2_frame: pd.DataFrame,
) -> tuple[dict[str, Any], list[Issue]]:
    metadata_path = track1_dir / "MIDOGpp.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Track 1 metadata not found: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)

    images = pd.DataFrame(coco.get("images", []))
    annotations = pd.DataFrame(coco.get("annotations", []))
    categories = {item["id"]: item["name"] for item in coco.get("categories", [])}
    if images.empty or annotations.empty:
        raise ValueError("Track 1 JSON does not contain images and annotations")
    annotations["category"] = annotations["category_id"].map(categories).fillna("unknown")

    image_by_id = images.set_index("id")
    missing_image_refs = int((~annotations["image_id"].isin(image_by_id.index)).sum())
    duplicate_image_ids = int(images.duplicated("id", keep=False).sum())
    duplicate_annotation_ids = int(annotations.duplicated("id", keep=False).sum())
    unknown_category_refs = int((~annotations["category_id"].isin(categories)).sum())

    def bbox_issue(row: pd.Series) -> str | None:
        bbox = row["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            return "invalid_shape"
        if not all(isinstance(value, (int, float)) and np.isfinite(value) for value in bbox):
            return "non_numeric"
        x0, y0, x1, y1 = bbox
        if x1 <= x0 or y1 <= y0:
            return "non_positive_extent"
        if row["image_id"] not in image_by_id.index:
            return "missing_image"
        image = image_by_id.loc[row["image_id"]]
        if x0 < 0 or y0 < 0 or x1 > image["width"] or y1 > image["height"]:
            return "out_of_bounds"
        return None

    annotations["bbox_issue"] = annotations.apply(bbox_issue, axis=1)
    bbox_issues = annotations["bbox_issue"].value_counts(dropna=True).to_dict()
    annotations["vote_count"] = annotations["labels"].apply(
        lambda values: len(values) if isinstance(values, list) else 0
    )
    annotations["vote_unanimous"] = annotations["labels"].apply(
        lambda values: isinstance(values, list) and bool(values) and len(set(values)) == 1
    )

    image_stats = images.copy()
    per_image = annotations.groupby(["image_id", "category"]).size().unstack(fill_value=0)
    image_stats = image_stats.merge(per_image, left_on="id", right_index=True, how="left")
    for category in categories.values():
        if category not in image_stats:
            image_stats[category] = 0
    image_stats.to_csv(report_dir / "track1_image_inventory.csv", index=False)

    image_dir = track1_dir / "images"
    expected_files = set(images["file_name"])
    manifest_path = track1_dir / "download_manifest.json"
    if manifest_path.exists():
        manifest_values = json.loads(manifest_path.read_text(encoding="utf-8")).values()
        downloadable_files = {name for name in manifest_values if name.lower().endswith(".tiff")}
    else:
        downloadable_files = expected_files
    local_files = {
        path.name for path in image_dir.glob("*.tiff") if path.is_file() and path.stat().st_size > 0
    } if image_dir.exists() else set()
    zero_byte_files = sorted(
        path.name for path in image_dir.glob("*.tiff") if path.is_file() and path.stat().st_size == 0
    ) if image_dir.exists() else []
    missing_local_images = sorted(downloadable_files - local_files)
    metadata_only_images = sorted(expected_files - downloadable_files)
    unexpected_local_images = sorted(local_files - expected_files)
    tiff_inventory = inspect_tiff_directory(image_dir) if local_files else pd.DataFrame()
    if not tiff_inventory.empty:
        tiff_inventory.to_csv(report_dir / "track1_tiff_inventory.csv", index=False)
    corrupt_tiff_headers = (
        int(tiff_inventory["error"].notna().sum()) if not tiff_inventory.empty else 0
    )

    # The MIDOG JSON uses corner coordinates [x0, y0, x1, y1], despite the
    # field being named `bbox`. Match Track 2 coordinates to category-1 centers.
    mitosis_category_ids = {
        category_id
        for category_id, name in categories.items()
        if name.lower().strip() == "mitotic figure"
    }
    mitoses = annotations[annotations["category_id"].isin(mitosis_category_ids)].copy()
    mitoses["filename"] = mitoses["image_id"].map(image_by_id["file_name"])
    mitoses["coordinateX"] = mitoses["bbox"].apply(lambda box: (box[0] + box[2]) / 2)
    mitoses["coordinateY"] = mitoses["bbox"].apply(lambda box: (box[1] + box[3]) / 2)
    track1_keys = set(
        zip(
            mitoses["filename"],
            mitoses["coordinateX"].round(4),
            mitoses["coordinateY"].round(4),
            strict=False,
        )
    )
    track2_keys = set(
        zip(
            track2_frame["filename"],
            track2_frame["coordinateX"].round(4),
            track2_frame["coordinateY"].round(4),
            strict=False,
        )
    )
    overlap = track1_keys & track2_keys

    issues: list[Issue] = []
    if bbox_issues:
        issues.append(
            Issue(
                "high",
                "Track 1 annotations",
                "Some detection boxes fail geometry or boundary validation.",
                ", ".join(f"{key}: {value}" for key, value in bbox_issues.items()),
                "Clip boxes to ROI bounds (or exclude after review) before patch extraction and training.",
            )
        )
    if duplicate_image_ids or duplicate_annotation_ids:
        issues.append(
            Issue(
                "high",
                "Track 1 identifiers",
                "COCO identifiers are not unique.",
                f"image rows: {duplicate_image_ids}; annotation rows: {duplicate_annotation_ids}",
                "Re-key duplicates before joining tables.",
            )
        )
    if missing_local_images:
        severity = "medium" if local_files else "info"
        issues.append(
            Issue(
                severity,
                "Track 1 files",
                "The local image set is not yet complete.",
                f"{len(local_files):,}/{len(downloadable_files):,} downloadable TIFFs present; "
                f"{len(zero_byte_files):,} zero-byte placeholders",
                "Resume `midog-download --track track1-images --i-understand-size`.",
            )
        )
    if metadata_only_images:
        issues.append(
            Issue(
                "info",
                "Track 1 manifest",
                "The COCO metadata references more TIFF names than the official downloader publishes.",
                f"{len(expected_files):,} metadata names; {len(downloadable_files):,} downloadable TIFFs; "
                f"{len(metadata_only_images):,} metadata-only names",
                "Treat the official 503-TIFF manifest as the downloadable training corpus and retain this distinction in provenance.",
            )
        )
    if corrupt_tiff_headers:
        issues.append(
            Issue(
                "critical",
                "Track 1 files",
                "One or more downloaded TIFF headers cannot be decoded.",
                f"{corrupt_tiff_headers} TIFFs",
                "Remove the affected files and resume the downloader.",
            )
        )
    if overlap:
        issues.append(
            Issue(
                "critical",
                "Cross-track leakage",
                "Track 2 patches are derived from Track 1 mitosis coordinates.",
                f"{len(overlap):,}/{len(track2_keys):,} unique Track 2 coordinates match Track 1",
                "Use one shared WSI-level split manifest across both tracks; never split tracks independently.",
            )
        )

    _save_count_plot(
        images["tumor_type"],
        report_dir / "track1_images_by_tumor.png",
        "Track 1 images by tumor type",
        xlabel="ROI images",
    )
    counts = annotations.groupby(["image_id", "category"]).size().reset_index(name="count")
    counts = counts.merge(images[["id", "tumor_type"]], left_on="image_id", right_on="id")
    tumor_counts = counts.groupby(["tumor_type", "category"])["count"].sum().unstack(fill_value=0)
    fig, axis = plt.subplots(figsize=(10, max(4, 0.6 * len(tumor_counts))))
    tumor_counts.plot.barh(ax=axis, color=sns.color_palette("Set2", len(tumor_counts.columns)))
    axis.set_title("Track 1 annotations by tumor and category", loc="left", weight="bold")
    axis.set_xlabel("Annotations")
    axis.set_ylabel("")
    axis.legend(title="Category", bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(report_dir / "track1_annotations_by_tumor.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "metadata_images": len(images),
        "annotations": len(annotations),
        "categories": {str(key): value for key, value in categories.items()},
        "category_counts": {
            str(key): int(value) for key, value in annotations["category"].value_counts().items()
        },
        "tumor_domains": int(images["tumor_type"].nunique()),
        "tumor_image_counts": {
            str(key): int(value) for key, value in images["tumor_type"].value_counts().items()
        },
        "missing_image_references": missing_image_refs,
        "unknown_category_references": unknown_category_refs,
        "duplicate_image_id_rows": duplicate_image_ids,
        "duplicate_annotation_id_rows": duplicate_annotation_ids,
        "bbox_issues": {str(key): int(value) for key, value in bbox_issues.items()},
        "annotation_vote_coverage": {
            str(key): int(value) for key, value in annotations["vote_count"].value_counts().sort_index().items()
        },
        "unanimous_annotation_votes": int(annotations["vote_unanimous"].sum()),
        "metadata_referenced_tiff_files": len(expected_files),
        "download_manifest_tiff_files": len(downloadable_files),
        "metadata_only_tiff_files": metadata_only_images,
        "local_nonempty_tiff_files": len(local_files),
        "zero_byte_tiff_files": len(zero_byte_files),
        "missing_local_tiff_files": len(missing_local_images),
        "unexpected_local_tiff_files": unexpected_local_images,
        "corrupt_tiff_headers": corrupt_tiff_headers,
        "local_mpp_values": sorted(
            {
                round(float(value), 6)
                for value in (tiff_inventory["mpp_x"].dropna() if not tiff_inventory.empty else [])
            }
        ),
        "cross_track_unique_coordinate_overlap": len(overlap),
        "track2_unique_coordinates": len(track2_keys),
        "track1_unique_mitosis_coordinates": len(track1_keys),
    }
    return summary, issues


def _format_table(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        frame = frame.head(max_rows)
    return frame.to_html(index=False, border=0, classes="data-table", escape=True)


def render_report(summary: dict[str, Any], issues: list[Issue], report_dir: Path) -> Path:
    track1 = summary["track1"]
    track2 = summary["track2"]
    issue_frame = pd.DataFrame([asdict(item) for item in issues])
    issue_frame.to_csv(report_dir / "data_quality_issues.csv", index=False)
    severity_order = pd.CategoricalDtype(
        ["critical", "high", "medium", "low", "info"], ordered=True
    )
    issue_frame["severity"] = issue_frame["severity"].astype(severity_order)
    issue_frame = issue_frame.sort_values(["severity", "scope"])

    class_frame = pd.DataFrame(
        [{"label": key, "count": value} for key, value in track2["class_counts"].items()]
    )
    domain_frame = pd.DataFrame(
        [
            {"tumor": key, "patches": value}
            for key, value in track2["domains"]["Tumor"].items()
        ]
    )
    agreement_frame = pd.DataFrame(
        [
            {
                "pair": pair,
                "shared rows": values["n"],
                "agreement": values["agreement"],
                "Cohen kappa": values["kappa"],
            }
            for pair, values in track2["pairwise_agreement"].items()
        ]
    )
    for column in ("agreement", "Cohen kappa"):
        agreement_frame[column] = agreement_frame[column].map(
            lambda value: "—" if value is None else f"{value:.3f}"
        )

    cards = [
        ("Track 1 ROI images", f"{track1['metadata_images']:,}"),
        ("Track 1 annotations", f"{track1['annotations']:,}"),
        ("Track 2 CSV rows", f"{track2['csv_rows']:,}"),
        ("Track 2 PNGs", f"{track2['png_files']:,}"),
        ("Source WSIs", f"{track2['source_wsis']:,}"),
        ("NMF : AMF", f"{track2['nmf_to_amf_ratio']:.2f} : 1"),
    ]
    card_html = "".join(
        f'<div class="card"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>'
        for label, value in cards
    )
    plot_files = [
        ("track2_class_distribution.png", "Track 2 class distribution"),
        ("track2_class_by_tumor.png", "Class balance by tumor domain"),
        ("track2_tumor_distribution.png", "Patches by tumor domain"),
        ("track2_scanner_distribution.png", "Patches by scanner"),
        ("track1_images_by_tumor.png", "Track 1 image domains"),
        ("track1_annotations_by_tumor.png", "Track 1 annotation domains"),
    ]
    plots_html = "".join(
        f'<figure><img src="{name}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'
        for name, caption in plot_files
        if (report_dir / name).exists()
    )
    generated = summary["generated_at"]
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MIDOG 2025 data audit</title>
<style>
:root {{ --ink:#19212b; --muted:#64748b; --panel:#f7f7fb; --purple:#6d3fc0; --teal:#159b8b; --line:#e2e8f0; }}
* {{ box-sizing:border-box; }} body {{ margin:0; color:var(--ink); font:15px/1.55 Inter,Segoe UI,Arial,sans-serif; background:#fff; }}
main {{ max-width:1180px; margin:auto; padding:48px 28px 80px; }} h1 {{ font-size:42px; letter-spacing:-1.4px; margin:.1em 0; }}
h2 {{ margin-top:48px; border-bottom:1px solid var(--line); padding-bottom:8px; }} h3 {{ margin-top:28px; }}
.eyebrow {{ color:var(--purple); font-weight:700; text-transform:uppercase; letter-spacing:.12em; }} .muted {{ color:var(--muted); }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; margin:28px 0; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px; }} .card span {{ display:block; color:var(--muted); }} .card strong {{ font-size:25px; }}
.callout {{ border-left:5px solid var(--purple); background:#f7f2ff; padding:18px 20px; border-radius:8px; }}
.data-table {{ width:100%; border-collapse:collapse; margin:16px 0 30px; }} .data-table th,.data-table td {{ text-align:left; padding:10px 12px; border-bottom:1px solid var(--line); vertical-align:top; }} .data-table th {{ background:var(--panel); }}
.gallery {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); gap:20px; }} figure {{ margin:0; border:1px solid var(--line); border-radius:12px; overflow:hidden; }} figure img {{ width:100%; display:block; }} figcaption {{ padding:10px 14px; color:var(--muted); }}
code {{ background:#f1f5f9; padding:2px 5px; border-radius:4px; }} a {{ color:var(--purple); }}
@media(max-width:600px) {{ h1 {{ font-size:32px; }} main {{ padding:28px 16px; }} .gallery {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body><main>
<div class="eyebrow">MIDOG 2025 · reproducible audit</div>
<h1>Training data audit</h1>
<p class="muted">Generated {html.escape(generated)}. Full-image statistics reflect files present locally at audit time.</p>
<div class="cards">{card_html}</div>

<div class="callout"><strong>Resolution answer:</strong> do not assume every test image is exactly 0.25&nbsp;µm/px.
The official challenge documentation lists native scanner resolutions of 0.22, 0.23 and 0.25&nbsp;µm/px; the
challenge paper additionally reports 0.12&nbsp;µm/px for two final-test domains. They are broadly similar in scale,
but not identical. Read TIFF spacing/DPI metadata per image and resample explicitly when a model requires a fixed MPP.
See the <a href="https://zenodo.org/records/15077361">challenge specification</a> and
<a href="https://arxiv.org/abs/2606.07368">challenge paper</a>.</div>

<h2>Priority findings</h2>
{_format_table(issue_frame)}

<h2>Track 2: atypical classification</h2>
<p>The archive contains {track2['png_files']:,} PNG files while the CSV contains {track2['csv_rows']:,} rows and
{track2['unique_image_ids']:,} unique image IDs. The source grouping column identifies {track2['source_wsis']:,}
WSIs. Use that field as the minimum split unit.</p>
<h3>Class distribution</h3>{_format_table(class_frame)}
<h3>Domain distribution</h3>{_format_table(domain_frame)}
<h3>Expert agreement</h3>{_format_table(agreement_frame)}

<h2>Track 1: mitosis detection</h2>
<p>The COCO-style metadata contains {track1['metadata_images']:,} ROI images and {track1['annotations']:,}
annotations across {track1['tumor_domains']} tumor domains. Its <code>bbox</code> values are corner coordinates
<code>[x0,y0,x1,y1]</code>, not standard COCO <code>[x,y,width,height]</code>; treating them as standard COCO silently
corrupts box sizes and centers.</p>
<p>Local TIFF status at audit time: {track1['local_nonempty_tiff_files']:,}/{track1['download_manifest_tiff_files']:,}
non-empty files, {track1['zero_byte_tiff_files']:,} zero-byte files.</p>

<h2>Cross-track relationship</h2>
<p>{track1['cross_track_unique_coordinate_overlap']:,} of {track1['track2_unique_coordinates']:,} unique Track 2
source coordinates match Track 1 mitotic annotations. This is expected dataset derivation, but it creates severe
leakage if the tracks are split independently. Maintain one WSI-level split manifest for both tasks.</p>

<h2>Visual audit</h2><div class="gallery">{plots_html}</div>

<h2>Artifacts and methodology</h2>
<p>The report folder also contains machine-readable <code>audit_summary.json</code>, issue records,
per-image inventories, and duplicate-ID rows. Patch checks include full-file SHA-256, decode integrity, dimensions,
mode, downsampled RGB/tissue statistics, and average-hash collisions. Track 1 metadata checks include referential
integrity, category validity, box geometry/bounds, annotation vote coverage, domain counts, and cross-track coordinate matching.</p>
<p class="muted">Average-hash collisions are candidates, not proof of duplication. TIFF pixel statistics are deferred
until the 65 GB image download completes; rerunning the command refreshes this report from the current local state.</p>
</main></body></html>"""
    output = report_dir / "data_audit_report.html"
    output.write_text(document, encoding="utf-8")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.report_root.mkdir(parents=True, exist_ok=True)
    track2, track2_issues, track2_frame = audit_track2(
        args.data_root / "track2", args.report_root, args.workers
    )
    track1, track1_issues = audit_track1(
        args.data_root / "track1", args.report_root, track2_frame
    )
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "track1": track1,
        "track2": track2,
    }
    with (args.report_root / "audit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=_json_default)
    output = render_report(summary, track1_issues + track2_issues, args.report_root)
    print(f"Audit complete: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
