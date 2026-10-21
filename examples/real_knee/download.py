"""Download the two archives the real knee example needs from Zenodo
(10.5281/zenodo.19720803, CC BY 4.0), check their MD5 and unpack them.

python download.py data

About 1.1 GB to download and 3 GB unpacked. A download cut off part way is
resumed when the script is run again."""

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

RECORD = "https://zenodo.org/api/records/19720803/files"
FILES = {
    "02_markerbased_data.zip": "6438a2afd60c71f6ec6b07fd0a0f60e2",
    "03_markerless_data.zip": "611e13bf6299d5a1796cfb27ea3bc342",
}


def fetch(url: str, path: Path) -> None:
    have = path.stat().st_size if path.exists() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"})
    with urllib.request.urlopen(request) as response:
        mode = "ab" if response.status == 206 else "wb"
        with open(path, mode) as f:
            while chunk := response.read(1 << 20):
                f.write(chunk)


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def main(folder: str) -> None:
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        path = out / name
        if not (path.exists() and md5(path) == expected):
            print(f"downloading {name}")
            fetch(f"{RECORD}/{name}/content", path)
        if md5(path) != expected:
            sys.exit(f"{name}: MD5 does not match; delete it and run again")
        with zipfile.ZipFile(path) as archive:
            archive.extractall(out)
        print(f"{name}: checked and unpacked")


if __name__ == "__main__":
    main(*sys.argv[1:])
