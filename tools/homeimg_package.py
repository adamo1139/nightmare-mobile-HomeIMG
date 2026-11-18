"""Validate an epoch49 Diffusers directory against the bundled, trusted v1 manifest.

files maps each required relative path to its exact byte count and SHA-256.
The digest covers configuration and full tensor headers (names, shapes, dtypes,
offsets) as well as data. Unknown model revisions require a separate manifest.
"""
import argparse
import hashlib
import json
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[1] / "backend/homeimg/epoch49.json"


def validate(root):
    manifest = json.loads(MANIFEST.read_text())
    for name, expected in manifest["files"].items():
        path = root / name
        if path.stat().st_size != expected["bytes"]:
            raise ValueError(f"{name}: incorrect byte count")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != expected["sha256"]:
            raise ValueError(f"{name}: checksum mismatch")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    print(validate(parser.parse_args().root)["revision"])
