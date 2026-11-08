import pandas as pd

from midog2025.data import assign_group_splits, corners_to_xywh


def test_corner_conversion() -> None:
    assert corners_to_xywh([10, 20, 60, 70]) == [10, 20, 50, 50]


def test_group_split_has_no_leakage() -> None:
    frame = pd.DataFrame(
        {
            "filename": ["a", "a", "b", "c", "d", "e"],
            "Tumor": ["x", "x", "x", "x", "y", "y"],
        }
    )
    split = assign_group_splits(frame)
    assert split.groupby("filename")["split"].nunique().max() == 1
