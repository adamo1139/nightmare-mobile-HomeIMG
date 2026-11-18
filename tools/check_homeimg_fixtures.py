"""Verify the recorded CPU or CUDA fixtures, including the Euler trajectory."""
import argparse
import json
from pathlib import Path
import torch
from safetensors.torch import load_file

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("directory",type=Path)
root = p.parse_args().directory
metadata = json.loads((root / "metadata.json").read_text())
texts = {}
for label in metadata["prompts"]:
    tensors = load_file(root / f"text-{label}.safetensors")
    ids, valid = tensors["ids"], tensors["valid"]
    assert ids.shape == (1,256) and ids[0,0] == 1
    assert torch.equal(valid, ids != 65535)
    count = int(valid.sum())
    assert tensors["lm"].shape == (1,count,2048)
    assert tensors["context"].shape == (1,count,1280)
    assert valid[0,:count].all() and tensors["context_valid"].all()
    assert torch.isfinite(tensors["lm"]).all() and torch.isfinite(tensors["context"]).all()
    texts[label] = tensors
assert texts["empty"]["valid"].sum() == 1 and texts["long"]["valid"].sum() == 256
for resolution in metadata["resolutions"]:
    d = load_file(root / f"image-{resolution}.safetensors")
    z, times = d["noise"], d["times"]
    assert z.shape == (1,128,resolution//16,resolution//16)
    assert times[0].isclose(torch.tensor(1.)) and times[-1]==0 and (times.diff()<0).all()
    assert torch.isfinite(d["context"]).all() and list(d["valid"].sum(1)) == [1,texts[metadata.get("image_prompt","pl")]["valid"].sum()]
    for i, dt in enumerate(times.diff()):
        neg,pos = d[f"velocity_pair_{i}"].chunk(2)
        z = z + dt*(neg + metadata["cfg"]*(pos-neg)).float()
        torch.testing.assert_close(z,d[f"euler_{i}"],atol=metadata["atol"],rtol=metadata["rtol"])
    image = d["decoded"]
    assert image.shape==(1,3,resolution,resolution) and (image.abs()<=1).all()
    print(resolution, "OK")
