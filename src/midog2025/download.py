"""Resumable, explicit dataset download helpers.

Track 1 images are intentionally guarded because the upstream dataset is about
65 GB. Metadata and the Track 2 patch dataset are small enough for normal use.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

from .paths import DATA_ROOT

TRACK2_CSV_URL = (
    "https://zenodo.org/records/16044804/files/"
    "MIDOG25_Atypical_Classification_Train_Set.csv?download=1"
)
TRACK2_ARCHIVE_URL = (
    "https://midog2025.deepmicroscopy.org/data/"
    "MIDOG25_Atypical_Classification_Train_Set.zip"
)
TRACK1_METADATA_URL = "https://figshare.com/ndownloader/files/40283737"
TRACK1_MANIFEST_SCRIPT_URL = (
    "https://raw.githubusercontent.com/DeepMicroscopy/"
    "MIDOG_2025_Guide/main/download_MIDOGpp.py"
)

TRACK2_CSV_MD5 = "6f31b448ba88dc7595882c23a1d7e388"
TRACK2_ARCHIVE_MD5 = "c68b09c51dda154aef798e8df094b8dc"


def file_digest(path: Path, algorithm: str = "md5") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, expected_md5: str | None = None) -> Path:
    """Download to a `.part` file and resume when the server supports ranges."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == 0:
        destination.unlink()
    if destination.exists():
        if expected_md5 is None or file_digest(destination) == expected_md5:
            print(f"Using existing {destination}")
            return destination
        raise RuntimeError(f"Checksum mismatch for existing file: {destination}")

    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={"User-Agent": "midog-2025/0.1"})
    if offset:
        request.add_header("Range", f"bytes={offset}-")

    with urllib.request.urlopen(request, timeout=90) as response:  # noqa: S310
        if response.headers.get("Content-Length") == "0":
            raise RuntimeError(f"Upstream returned an empty response for {url}")
        resumed = offset > 0 and getattr(response, "status", None) == 206
        mode = "ab" if resumed else "wb"
        if offset and not resumed:
            offset = 0
        with partial.open(mode) as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
    partial.replace(destination)

    if expected_md5 and file_digest(destination) != expected_md5:
        raise RuntimeError(f"Checksum mismatch after download: {destination}")
    return destination


def curl_download(url: str, destination: Path) -> Path:
    """Resume a large file with bounded retries and low-speed detection."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == 0:
        destination.unlink()
    if destination.exists():
        return destination
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if curl is None:
        return download(url, destination)
    partial = destination.with_suffix(destination.suffix + ".part")
    command = [
        curl,
        "-L",
        "--fail",
        "--retry",
        "0",
        "--connect-timeout",
        "30",
        "--speed-limit",
        "10240",
        "--speed-time",
        "30",
        "--max-time",
        "180",
        "-C",
        "-",
        "-o",
        str(partial),
        url,
    ]
    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        message = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown error"
        raise RuntimeError(f"curl failed for {destination.name}: {message}")
    if not partial.exists() or partial.stat().st_size == 0:
        raise RuntimeError(f"curl returned an empty file for {destination.name}")
    partial.replace(destination)
    return destination


def safe_extract(archive: Path, destination: Path) -> None:
    """Extract a ZIP while rejecting path traversal entries."""

    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if root != target and root not in target.parents:
                raise RuntimeError(f"Unsafe archive path: {member.filename}")
        bundle.extractall(destination)


def _track1_manifest() -> dict[str, str]:
    request = urllib.request.Request(
        TRACK1_MANIFEST_SCRIPT_URL, headers={"User-Agent": "midog-2025/0.1"}
    )
    with urllib.request.urlopen(request, timeout=90) as response:  # noqa: S310
        tree = ast.parse(response.read().decode("utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "FILENAMES" for target in node.targets):
                manifest = ast.literal_eval(node.value)
                if isinstance(manifest, dict):
                    return {str(key): str(value) for key, value in manifest.items()}
    raise RuntimeError("Could not find FILENAMES in the official MIDOG download script")


def download_track2(root: Path) -> None:
    target = root / "track2"
    csv_path = download(
        TRACK2_CSV_URL,
        target / "MIDOG25_Atypical_Classification_Train_Set.csv",
        TRACK2_CSV_MD5,
    )
    archive = download(
        TRACK2_ARCHIVE_URL,
        target / "MIDOG25_Atypical_Classification_Train_Set.zip",
        TRACK2_ARCHIVE_MD5,
    )
    image_dir = target / "MIDOG25_Binary_Classification_Train_Set"
    if not image_dir.exists():
        safe_extract(archive, target)
    print(f"Track 2 ready: {csv_path.parent}")


def download_track1_metadata(root: Path) -> None:
    path = download(TRACK1_METADATA_URL, root / "track1" / "MIDOGpp.json")
    print(f"Track 1 metadata ready: {path}")


def write_track1_manifest(root: Path) -> dict[str, str]:
    manifest = _track1_manifest()
    manifest_path = root / "track1" / "download_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def download_track1_images(root: Path, acknowledged: bool, workers: int = 4) -> None:
    if not acknowledged:
        raise SystemExit(
            "Track 1 is approximately 65 GB. Re-run with --i-understand-size "
            "after checking disk space and the upstream license."
        )
    image_dir = root / "track1" / "images"
    manifest = write_track1_manifest(root)
    download_items = [
        item for item in manifest.items() if Path(item[1]).suffix.lower() in {".tiff", ".json"}
    ]
    print(
        f"Downloading {len(download_items)} MIDOG++ image/JSON files to {image_dir} "
        f"with {workers} workers",
        flush=True,
    )

    def fetch(item: tuple[str, str]) -> Path:
        file_id, filename = item
        url = f"https://ndownloader.figshare.com/files/{file_id}"
        destination = (
            image_dir / filename
            if Path(filename).suffix.lower() == ".tiff"
            else root / "track1" / filename
        )
        last_error: Exception | None = None
        for attempt in range(1, 21):
            try:
                return curl_download(url, destination)
            except RuntimeError as exc:
                last_error = exc
                print(f"Retry {attempt}/20 for {filename}: {exc}", flush=True)
                time.sleep(min(attempt * 2, 30))
        raise RuntimeError(f"Exhausted retries for {filename}: {last_error}")

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch, item) for item in download_items]
        for future in concurrent.futures.as_completed(futures):
            path = future.result()
            completed += 1
            print(f"Completed [{completed}/{len(futures)}]: {path.name}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--track",
        choices=(
            "track2",
            "track1-metadata",
            "track1-manifest",
            "track1-images",
            "all-metadata",
        ),
        default="all-metadata",
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--i-understand-size", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.track in {"track2", "all-metadata"}:
        download_track2(args.data_root)
    if args.track in {"track1-metadata", "all-metadata"}:
        download_track1_metadata(args.data_root)
    if args.track == "track1-manifest":
        manifest = write_track1_manifest(args.data_root)
        print(f"Track 1 manifest ready: {len(manifest)} entries")
    if args.track == "track1-images":
        if args.workers < 1 or args.workers > 32:
            raise SystemExit("--workers must be between 1 and 32")
        download_track1_images(args.data_root, args.i_understand_size, args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
