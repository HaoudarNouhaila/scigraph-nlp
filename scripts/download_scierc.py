"""Download the official SciERC processed release and extract the JSON splits.

Source: https://nlp.cs.washington.edu/sciIE/ (Luan et al., EMNLP 2018).

The archive is about 695 MB because it also contains ELMo embeddings; only the three
JSON files are kept. If the download fails in your environment (for example behind a
TLS-intercepting proxy), download the archive manually and pass ``--archive``.

Usage::

    python scripts/download_scierc.py --out data/raw/scierc
    python scripts/download_scierc.py --archive path/to/sciERC_processed.tar.gz
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

URL = "https://nlp.cs.washington.edu/sciIE/data/sciERC_processed.tar.gz"
# SHA-256 of the archive as downloaded on 2026-10-16; a mismatch is reported, not fatal.
EXPECTED_SHA256 = "bce752fb7ebe4acf570937d76ffb27c239cfc907e477a69075dc7082d0e72e9b"
MEMBERS = {
    "processed_data/json/train.json": "train.json",
    "processed_data/json/dev.json": "dev.json",
    "processed_data/json/test.json": "test.json",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_archive(path: Path) -> None:
    digest = sha256(path)
    print(f"archive sha256 {digest}")
    if digest != EXPECTED_SHA256:
        print("WARNING: archive checksum differs from the one recorded for this project.")


def download(url: str, dest: Path) -> None:
    print(f"Downloading {url} -> {dest}")
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as out:  # noqa: S310
        shutil.copyfileobj(resp, out, length=1 << 20)


def extract(archive: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    found = set()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            target = MEMBERS.get(member.name.lstrip("./"))
            if target is None or not member.isfile():
                continue
            src = tar.extractfile(member)
            assert src is not None
            with open(out_dir / target, "wb") as f:
                shutil.copyfileobj(src, f)
            found.add(target)
    missing = set(MEMBERS.values()) - found
    if missing:
        raise SystemExit(f"Archive does not contain expected files: {sorted(missing)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=Path("data/raw/scierc"))
    p.add_argument("--archive", type=Path, default=None, help="use a local archive")
    args = p.parse_args()

    if args.archive is None:
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "sciERC_processed.tar.gz"
            download(URL, archive)
            check_archive(archive)
            extract(archive, args.out)
    else:
        check_archive(args.archive)
        extract(args.archive, args.out)
    for name in MEMBERS.values():
        path = args.out / name
        print(f"{path}  sha256={sha256(path)}  bytes={path.stat().st_size}")


if __name__ == "__main__":
    main()
