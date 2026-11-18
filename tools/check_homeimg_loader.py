"""Compare every native mapped tensor to the official safetensors PyTorch reader.

From the repository root (C++17, nlohmann/json and OpenSSL development headers):
    c++ -O2 -std=c++17 tools/check_homeimg_loader.cpp -lcrypto -o /tmp/check_homeimg_loader
    python tools/check_homeimg_loader.py MODEL_DIRECTORY /tmp/check_homeimg_loader

OpenSSL is used only by the test to fingerprint the mapped bytes.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from safetensors import safe_open
import torch
from homeimg_package import validate

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("root",type=Path)
p.add_argument("driver",type=Path)
args = p.parse_args()
for file in validate(args.root)["files"]:
    if not file.endswith(".safetensors"):
        continue
    path = args.root/file
    lines = subprocess.check_output([args.driver,args.root,path.parent.name],text=True).splitlines()
    with safe_open(path,framework="pt",device="cpu") as reader:
        assert len(lines)==len(reader.keys())
        for line in lines:
            name,dtype,shape,digest = line.split()
            tensor = reader.get_tensor(name)
            assert reader.get_slice(name).get_dtype()==dtype and list(tensor.shape)==json.loads(shape), name
            raw = tensor.reshape(-1).view(torch.uint8).numpy()
            assert hashlib.sha256(memoryview(raw)).hexdigest()==digest, name
    print(file,len(lines),flush=True)
