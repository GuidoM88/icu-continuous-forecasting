#!/usr/bin/env python3
"""Download and extract PhysioNet/CinC Challenge 2012 sets A/B/C.

Uses the archives published by PhysioNet. Patient filenames are record IDs
(e.g. 132539.txt), not sequential integers.
"""
from __future__ import annotations

import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

BASE_URL = "https://physionet.org/files/challenge-2012/1.0.0"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

ARCHIVES = {
    "set-a": "set-a.zip",
    "set-b": "set-b.zip",
    "set-c": "set-c.tar.gz",
}
OUTCOMES = ["Outcomes-a.txt", "Outcomes-b.txt", "Outcomes-c.txt"]


def fetch(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "icu-continuous-forecasting/1.0"})
    with urllib.request.urlopen(req, timeout=120) as response, open(destination, "wb") as f:
        shutil.copyfileobj(response, f)


def extract_archive(path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if path.name.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            zf.extractall(destination)
    else:
        with tarfile.open(path, "r:gz") as tf:
            tf.extractall(destination)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for split, archive_name in ARCHIVES.items():
            split_dir = RAW_DIR / split
            if len(list(split_dir.glob("*.txt"))) == 4000:
                print(f"{split}: already contains 4000 records; skipping.")
                continue
            archive = tmp / archive_name
            fetch(f"{BASE_URL}/{archive_name}", archive)
            extract_archive(archive, RAW_DIR)
            n = len(list(split_dir.glob("*.txt")))
            if n != 4000:
                raise RuntimeError(f"{split}: expected 4000 records after extraction, found {n}")

        for name in OUTCOMES:
            dst = RAW_DIR / name
            if not dst.exists():
                fetch(f"{BASE_URL}/{name}", dst)

    print("PhysioNet 2012 A/B/C download complete.")


if __name__ == "__main__":
    main()
