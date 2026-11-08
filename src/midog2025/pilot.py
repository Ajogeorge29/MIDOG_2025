"""Resource-bounded MitoGuard candidate-level pilot.

This validates the complete verifier -> subtype -> hierarchical decision path on
held-out specimens. The proposal coordinates come from annotations, so results
are candidate-level sanity metrics, not whole-slide detector performance.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import tomllib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import TRACK2_LABEL_TO_ID, assign_group_splits, load_track1_tables, load_track2_table
from .paths import DATA_ROOT, REPO_ROOT, REPORT_ROOT


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def patch_features(image: Image.Image, size: int) -> np.ndarray:
    """Compact HOG + color-statistics features with bounded memory."""

    resized = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    rgb = np.asarray(resized, dtype=np.float32) / 255.0
    gray = rgb2gray(rgb)
    hog_vector = hog(
        gray,
        orientations=8,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)
    moments = np.concatenate(
        [
            rgb.mean(axis=(0, 1)),
            rgb.std(axis=(0, 1)),
            np.quantile(rgb, [0.1, 0.5, 0.9], axis=(0, 1)).reshape(-1),
        ]
    ).astype(np.float32)
    histograms = np.concatenate(
        [np.histogram(rgb[:, :, channel], bins=12, range=(0, 1), density=True)[0] for channel in range(3)]
    ).astype(np.float32)
    return np.concatenate([hog_vector, moments, histograms]).astype(np.float32)


def crop_center(image: Image.Image, x: float, y: float, size: int = 128) -> Image.Image:
    half = size // 2
    left, top = int(round(x)) - half, int(round(y)) - half
    right, bottom = left + size, top + size
    if left >= 0 and top >= 0 and right <= image.width and bottom <= image.height:
        return image.crop((left, top, right, bottom)).convert("RGB")
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    source_box = (max(0, left), max(0, top), min(image.width, right), min(image.height, bottom))
    region = image.crop(source_box).convert("RGB")
    canvas.paste(region, (max(0, -left), max(0, -top)))
    return canvas


def stratified_cap(frame: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    if len(frame) <= cap:
        return frame.sample(frac=1, random_state=seed)
    label_groups = list(frame.groupby("majority"))
    per_group = max(1, cap // len(label_groups))
    sampled = [
        group.sample(n=min(len(group), per_group), random_state=seed + index)
        for index, (_, group) in enumerate(label_groups)
    ]
    result = pd.concat(sampled)
    if len(result) < cap:
        remainder = frame.drop(result.index)
        result = pd.concat(
            [result, remainder.sample(n=min(cap - len(result), len(remainder)), random_state=seed)]
        )
    return result.sample(frac=1, random_state=seed).head(cap)


def png_features(frame: pd.DataFrame, size: int) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    targets: list[int] = []
    for row in frame.itertuples(index=False):
        with Image.open(row.image_path) as image:
            rows.append(patch_features(image, size))
        targets.append(TRACK2_LABEL_TO_ID[row.majority])
    return np.stack(rows), np.asarray(targets, dtype=np.int8)


def annotation_centers(annotations: pd.DataFrame, images: pd.DataFrame) -> pd.DataFrame:
    result = annotations.copy()
    result["x"] = result["bbox"].map(lambda box: (float(box[0]) + float(box[2])) / 2)
    result["y"] = result["bbox"].map(lambda box: (float(box[1]) + float(box[3])) / 2)
    result = result.merge(images[["id", "file_name", "tumor_type"]], left_on="image_id", right_on="id")
    return result


def choose_files(
    candidates: pd.DataFrame,
    split_map: dict[str, str],
    split: str,
    cap: int,
    seed: int,
) -> list[str]:
    available = sorted(
        filename
        for filename in candidates["file_name"].unique()
        if split_map.get(str(filename)) == split
    )
    rng = np.random.default_rng(seed)
    rng.shuffle(available)
    return available[:cap]


def wsi_candidate_features(
    centers: pd.DataFrame,
    image_dir: Path,
    filenames: list[str],
    category_ids: set[int],
    per_class_cap: int | None,
    total_cap: int | None,
    feature_size: int,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    selected = centers[
        centers["file_name"].isin(filenames) & centers["category_id"].isin(category_ids)
    ].copy()
    rng = np.random.default_rng(seed)
    if per_class_cap is not None:
        chunks = []
        for _category_id, group in selected.groupby("category_id"):
            indices = rng.choice(group.index, size=min(per_class_cap, len(group)), replace=False)
            chunks.append(group.loc[indices])
        selected = pd.concat(chunks) if chunks else selected.iloc[0:0]
    if total_cap is not None and len(selected) > total_cap:
        indices = rng.choice(selected.index, size=total_cap, replace=False)
        selected = selected.loc[indices]
    selected = selected.sort_values(["file_name", "id_x"]).reset_index(drop=True)

    features: list[np.ndarray] = []
    kept_rows: list[pd.Series] = []
    for filename, group in selected.groupby("file_name", sort=True):
        path = image_dir / str(filename)
        if not path.exists():
            continue
        with Image.open(path) as image:
            for _, row in group.iterrows():
                crop = crop_center(image, row["x"], row["y"])
                features.append(patch_features(crop, feature_size))
                kept_rows.append(row)
    if not features:
        raise RuntimeError("No WSI candidate features were extracted")
    return np.stack(features), pd.DataFrame(kept_rows).reset_index(drop=True)


def best_threshold(y_true: np.ndarray, scores: np.ndarray, metric: str) -> tuple[float, float]:
    best_value, best_score = 0.5, -np.inf
    for threshold in np.linspace(0.1, 0.9, 81):
        predicted = (scores >= threshold).astype(np.int8)
        score = (
            balanced_accuracy_score(y_true, predicted)
            if metric == "balanced_accuracy"
            else f1_score(y_true, predicted, zero_division=0)
        )
        if score > best_score:
            best_value, best_score = float(threshold), float(score)
    return best_value, best_score


def classifier(config: dict[str, Any], seed: int) -> Any:
    settings = config["model"]
    return make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss=settings["loss"],
            alpha=float(settings["alpha"]),
            max_iter=int(settings["max_iter"]),
            early_stopping=bool(settings["early_stopping"]),
            class_weight="balanced",
            random_state=seed,
            n_jobs=int(settings["n_jobs"]),
            validation_fraction=0.15,
            n_iter_no_change=12,
        ),
    )


def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    return float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else None


def save_confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], path: Path) -> None:
    fig, axis = plt.subplots(figsize=(5.5, 5))
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
    ConfusionMatrixDisplay(matrix, display_labels=labels).plot(ax=axis, cmap="Purples", colorbar=False)
    axis.set_title("Held-out hierarchical pilot")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "pilot.toml")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path, default=REPORT_ROOT / "pilot")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    config = load_config(args.config)
    limits = config["pilot"]
    seed, feature_size = int(limits["seed"]), int(limits["feature_size"])

    track2 = load_track2_table(args.data_root / "track2")
    track2 = assign_group_splits(track2)
    if track2["image_id"].duplicated().any() or not track2["image_exists"].all():
        raise RuntimeError("Track 2 Gate A failed: duplicate IDs or missing image files")
    split_map = track2.drop_duplicates("filename").set_index("filename")["split"].to_dict()

    subtype_train = stratified_cap(
        track2[track2["split"] == "train"], int(limits["subtype_train_rows"]), seed
    )
    subtype_validation = stratified_cap(
        track2[track2["split"] == "validation"],
        int(limits["subtype_validation_rows"]),
        seed + 1,
    )
    print(f"Subtype features: train={len(subtype_train)}, validation={len(subtype_validation)}")
    x_sub_train, y_sub_train = png_features(subtype_train, feature_size)
    x_sub_val, y_sub_val = png_features(subtype_validation, feature_size)
    subtype_model = classifier(config, seed)
    subtype_model.fit(x_sub_train, y_sub_train)
    subtype_val_scores = subtype_model.predict_proba(x_sub_val)[:, 1]
    subtype_threshold, _ = best_threshold(y_sub_val, subtype_val_scores, "balanced_accuracy")

    images, annotations, categories = load_track1_tables(args.data_root / "track1")
    centers = annotation_centers(annotations, images)
    mitosis_id = next(key for key, value in categories.items() if value == "mitotic figure")
    negative_id = next(key for key, value in categories.items() if value == "not mitotic figure")
    train_files = choose_files(
        centers, split_map, "train", int(limits["negative_train_wsi_files"]), seed
    )
    validation_files = choose_files(
        centers, split_map, "validation", int(limits["negative_validation_wsi_files"]), seed
    )
    test_files = choose_files(
        centers, split_map, "test", int(limits["joint_test_wsi_files"]), seed
    )
    if not train_files or not validation_files or not test_files:
        raise RuntimeError("Specimen-level WSI file selection produced an empty split")

    # Positive verifier examples use released Track 2 crops; negatives are
    # extracted from independent annotated hard-negative coordinates.
    verifier_positive_train = subtype_train.sample(
        n=min(int(limits["verifier_train_per_class"]), len(subtype_train)), random_state=seed
    )
    x_ver_pos_train, _ = png_features(verifier_positive_train, feature_size)
    x_ver_neg_train, _ = wsi_candidate_features(
        centers,
        args.data_root / "track1" / "images",
        train_files,
        {negative_id},
        int(limits["verifier_train_per_class"]),
        None,
        feature_size,
        seed,
    )
    train_count = min(len(x_ver_pos_train), len(x_ver_neg_train))
    x_ver_train = np.concatenate([x_ver_pos_train[:train_count], x_ver_neg_train[:train_count]])
    y_ver_train = np.concatenate(
        [np.ones(train_count, dtype=np.int8), np.zeros(train_count, dtype=np.int8)]
    )

    verifier_positive_val = subtype_validation.sample(
        n=min(int(limits["verifier_validation_per_class"]), len(subtype_validation)),
        random_state=seed,
    )
    x_ver_pos_val, _ = png_features(verifier_positive_val, feature_size)
    x_ver_neg_val, _ = wsi_candidate_features(
        centers,
        args.data_root / "track1" / "images",
        validation_files,
        {negative_id},
        int(limits["verifier_validation_per_class"]),
        None,
        feature_size,
        seed + 1,
    )
    validation_count = min(len(x_ver_pos_val), len(x_ver_neg_val))
    x_ver_val = np.concatenate([x_ver_pos_val[:validation_count], x_ver_neg_val[:validation_count]])
    y_ver_val = np.concatenate(
        [np.ones(validation_count, dtype=np.int8), np.zeros(validation_count, dtype=np.int8)]
    )
    verifier_model = classifier(config, seed + 1)
    verifier_model.fit(x_ver_train, y_ver_train)
    verifier_val_scores = verifier_model.predict_proba(x_ver_val)[:, 1]
    verifier_threshold, _ = best_threshold(y_ver_val, verifier_val_scores, "f1")

    x_joint, joint_rows = wsi_candidate_features(
        centers,
        args.data_root / "track1" / "images",
        test_files,
        {mitosis_id, negative_id},
        None,
        int(limits["joint_test_candidates"]),
        feature_size,
        seed + 2,
    )
    subtype_lookup = {
        (str(row.filename), round(float(row.coordinateX), 4), round(float(row.coordinateY), 4)): row.majority
        for row in track2.itertuples(index=False)
    }
    true_classes: list[int] = []
    usable_indices: list[int] = []
    unmatched_positive = 0
    for index, row in joint_rows.iterrows():
        if int(row["category_id"]) == negative_id:
            true_classes.append(0)
            usable_indices.append(index)
            continue
        label = subtype_lookup.get(
            (str(row["file_name"]), round(float(row["x"]), 4), round(float(row["y"]), 4))
        )
        if label is None:
            unmatched_positive += 1
            continue
        true_classes.append(2 if label == "AMF" else 1)
        usable_indices.append(index)
    x_joint = x_joint[usable_indices]
    joint_rows = joint_rows.iloc[usable_indices].reset_index(drop=True)
    y_joint = np.asarray(true_classes, dtype=np.int8)

    verification_scores = verifier_model.predict_proba(x_joint)[:, 1]
    subtype_scores = subtype_model.predict_proba(x_joint)[:, 1]
    predicted_joint = np.zeros(len(x_joint), dtype=np.int8)
    accepted = verification_scores >= verifier_threshold
    predicted_joint[accepted] = np.where(subtype_scores[accepted] >= subtype_threshold, 2, 1)

    verifier_truth = (y_joint > 0).astype(np.int8)
    verifier_pred = (verification_scores >= verifier_threshold).astype(np.int8)
    verifier_precision, verifier_recall, verifier_f1, _ = precision_recall_fscore_support(
        verifier_truth, verifier_pred, average="binary", zero_division=0
    )
    positive_mask = y_joint > 0
    subtype_truth = (y_joint[positive_mask] == 2).astype(np.int8)
    subtype_pred = (subtype_scores[positive_mask] >= subtype_threshold).astype(np.int8)
    amf_precision, amf_recall, amf_f1, _ = precision_recall_fscore_support(
        y_joint == 2, predicted_joint == 2, average="binary", zero_division=0
    )

    metrics = {
        "scope": "annotation-candidate pilot; not whole-slide proposal detection",
        "runtime_minutes": (time.perf_counter() - started) / 60,
        "limits": limits,
        "splits": {
            "subtype_train_rows": len(subtype_train),
            "subtype_validation_rows": len(subtype_validation),
            "verifier_train_candidates": len(x_ver_train),
            "verifier_validation_candidates": len(x_ver_val),
            "joint_test_candidates": len(y_joint),
            "joint_test_wsi_files": test_files,
            "unmatched_positive_candidates_excluded": unmatched_positive,
        },
        "thresholds": {
            "verification": verifier_threshold,
            "subtype": subtype_threshold,
        },
        "verification": {
            "precision": float(verifier_precision),
            "recall": float(verifier_recall),
            "f1": float(verifier_f1),
            "roc_auc": safe_auc(verifier_truth, verification_scores),
        },
        "subtype_oracle_positive_candidates": {
            "balanced_accuracy": float(balanced_accuracy_score(subtype_truth, subtype_pred)),
            "roc_auc": safe_auc(subtype_truth, subtype_scores[positive_mask]),
            "atypical_count": int(subtype_truth.sum()),
            "normal_count": int((subtype_truth == 0).sum()),
        },
        "joint": {
            "macro_f1": float(f1_score(y_joint, predicted_joint, average="macro", zero_division=0)),
            "amf_precision": float(amf_precision),
            "amf_recall": float(amf_recall),
            "amf_f1": float(amf_f1),
            "report": classification_report(
                y_joint,
                predicted_joint,
                labels=[0, 1, 2],
                target_names=["background", "normal", "atypical"],
                output_dict=True,
                zero_division=0,
            ),
        },
    }
    with (args.output / "pilot_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    joblib.dump(verifier_model, args.output / "verifier_sgd.joblib")
    joblib.dump(subtype_model, args.output / "subtype_sgd.joblib")
    prediction_table = joint_rows[["file_name", "x", "y", "category_id"]].copy()
    prediction_table["truth"] = y_joint
    prediction_table["verification_probability"] = verification_scores
    prediction_table["atypia_probability"] = subtype_scores
    prediction_table["prediction"] = predicted_joint
    prediction_table.to_csv(args.output / "pilot_predictions.csv", index=False)
    save_confusion(
        y_joint,
        predicted_joint,
        ["background", "normal", "atypical"],
        args.output / "pilot_confusion_matrix.png",
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
