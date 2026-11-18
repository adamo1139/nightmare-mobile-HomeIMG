"""Stage untouched Nightmare 1.6.307 components for a side-by-side HomeIMG debug build.

Usage: python tools/stage_homeimg_base.py nightmare-1.6.307.apk
Then gradlew :app:assembleDebug -PhomeImgReleaseBase=true. Build Foxlet and
the HomeIMG Android targets first. Release signing and publishing stay separate.
"""
import argparse
import hashlib
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = "5bef11a4844430d748a852c8c0f1855007f56ac5548e018f1ab0928a4788a07f"
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("apk",type=Path)
args=p.parse_args()
if hashlib.sha256(args.apk.read_bytes()).hexdigest()!=EXPECTED:
    p.error("wrong source APK; expected upstream v1.6.307")
dst=ROOT/"build/homeimg-release-base"
lib=dst/"jniLibs/arm64-v8a"
lib.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(args.apk) as archive:
    for name in ("libnmqnn.so","libnmjs.so","libstable_diffusion_core.so"):
        (lib/name).write_bytes(archive.read("lib/arm64-v8a/"+name))
    for key in archive.namelist():
        if key.startswith(("assets/npu/","assets/ditlibs/")) and not key.endswith("/"):
            out=dst/key
            out.parent.mkdir(parents=True,exist_ok=True)
            out.write_bytes(archive.read(key))
shutil.copy2(ROOT/"build/homeimg-arm64/libhomeimg_cpu.so",lib/"libhomeimg_cpu.so")
