from pathlib import Path

import pandas as pd
from PIL import Image

from midog2025.audit import cohen_kappa, inspect_patch, strict_vote_majority


def test_strict_vote_majority() -> None:
    row = pd.Series({"expert1": "AMF", "expert2": "AMF", "expert3": "NMF"})
    assert strict_vote_majority(row) == "AMF"


def test_strict_vote_majority_rejects_tie() -> None:
    row = pd.Series({"expert1": "AMF", "expert2": "NMF", "expert3": None})
    assert strict_vote_majority(row) is None


def test_cohen_kappa_perfect_agreement() -> None:
    result = cohen_kappa(pd.Series(["AMF", "NMF"]), pd.Series(["AMF", "NMF"]))
    assert result["agreement"] == 1.0
    assert result["kappa"] == 1.0


def test_inspect_patch(tmp_path: Path) -> None:
    path = tmp_path / "example.png"
    Image.new("RGB", (32, 24), (120, 80, 160)).save(path)
    result = inspect_patch(path)
    assert result["error"] is None
    assert (result["width"], result["height"]) == (32, 24)
    assert result["sha256"]
