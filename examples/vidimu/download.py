"""Download what the VIDIMU example needs from Zenodo
(10.5281/zenodo.15088423, CC BY 4.0), check the MD5 and unpack it.

python download.py data

`jointangles.zip` (460 MB, 1.5 GB unpacked) holds the 3D keypoints of the
four video models and the IMU inverse kinematics; `results.zip` (4.5 MB)
holds the authors' published summary tables. A download cut off part way
is resumed when the script is run again."""

import hashlib
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

RECORD = "https://zenodo.org/api/records/15088423/files"
SIZES = {"jointangles.zip": 459179524, "results.zip": 4519210}
FILES = {
    "jointangles.zip": "3f80d368163d4d0a35028ac028a9db24",
    "results.zip": "1de2d2ab1173d1304f6baebdd41a6e0d",
}


def fetch(url: str, path: Path) -> None:
    have = path.stat().st_size if path.exists() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"})
    try:
        response = urllib.request.urlopen(request)
    except urllib.error.HTTPError as error:
        if error.code != 416:  # 416: nothing left past `have`, so start over
            raise
        path.unlink()
        response = urllib.request.urlopen(url)
    with response:
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
            # a complete file that fails the check is fetched again whole
            if path.exists() and path.stat().st_size >= SIZES[name]:
                path.unlink()
            print(f"downloading {name}")  # resumes a part left by an earlier run
            fetch(f"{RECORD}/{name}/content", path)
        if md5(path) != expected:
            sys.exit(f"{name}: MD5 does not match yet; run again to resume or refetch")
        with zipfile.ZipFile(path) as archive:
            archive.extractall(out)
        print(f"{name}: checked and unpacked")


if __name__ == "__main__":
    main(*sys.argv[1:])
