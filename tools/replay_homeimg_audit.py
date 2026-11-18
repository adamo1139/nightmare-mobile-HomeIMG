#!/usr/bin/env python3
"""Reconstruct the audited candidate source tree, without modifying the input.

This is NOT a claim to reproduce the released Android binary. The manifest
records a source-compatible candidate, including its explicit exclusions.
Requires an initialized Local Dream checkout at the manifest's source pins.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def git(path, *args):
    return subprocess.check_output(
        ["git", "-C", str(path), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="New directory, must not already exist")
    args = parser.parse_args()
    source = args.local_dream.resolve()
    output = args.output.resolve()
    repo = Path(__file__).resolve().parent.parent
    manifest_path = repo / "tools/homeimg-engine-patches.json"
    manifest = json.loads(manifest_path.read_text())
    cpp_rel = Path("app/src/main/cpp")
    sd_rel = cpp_rel / "3rdparty/stable-diffusion.cpp"
    roots = {"local_dream": Path("."), "stable_diffusion_cpp": sd_rel,
             "ggml": sd_rel / "ggml"}

    if output.exists() or source == output or source in output.parents:
        parser.error("output must be a new directory outside the input checkout")
    for name, relative in roots.items():
        path = source / relative
        if git(path, "rev-parse", "HEAD") != manifest["source_pins"][name]:
            parser.error(f"wrong {name} revision; see manifest source_pins")
        if git(path, "status", "--porcelain", "--untracked-files=no"):
            parser.error(f"{name} has tracked changes; use a clean audit input")

    patches = []
    for entry in manifest["apply_order"]:
        name = entry["patch"]
        if name.startswith("upstream/dit/"):
            patch = source / cpp_rel / "dit" / Path(name).name
            target = sd_rel / "ggml" if patch.name == "ggml.patch" else sd_rel
        else:
            patch = repo / "backend-patches/dit" / name
            target = (sd_rel / "ggml" if name.endswith(".ggml.patch") else
                      sd_rel if name.endswith(".sdcpp.patch") else cpp_rel)
        if hashlib.sha256(patch.read_bytes()).hexdigest() != entry["sha256"]:
            parser.error(f"patch hash mismatch: {name}")
        patches.append((name, patch, target))

    shutil.copytree(source, output, ignore=shutil.ignore_patterns(".git"))
    git(output, "init", "-q")
    # One temporary repository lets --directory route patches without gitlink
    # ambiguity. No commit, staging, reset, network or input-tree write occurs.
    for name, patch, target in patches:
        git(output, "apply", "--check", f"--directory={target}", str(patch))
        git(output, "apply", f"--directory={target}", str(patch))
        print(f"applied {name}")
    header = (output / cpp_rel / "include/DitEngine.h").read_text()
    if "#define DIT_ENGINE_ABI_VERSION 111" not in header:
        raise RuntimeError("candidate header did not reach expected ABI 111")
    print(f"Candidate sources: {output}")
    print("Android/v73 build and released-binary equivalence remain unverified.")


if __name__ == "__main__":
    main()
