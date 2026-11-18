"""Run real epoch49 text parity checks, including first dense and routed layers.

Build the driver using:
    c++ -O2 -march=native -std=c++17 -I/usr/include/eigen3 tools/check_homeimg_text.cpp -o /tmp/homeimg-text
Run with MODEL_DIRECTORY CPU_FIXTURES /tmp/homeimg-text in the fixture generator environment.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import torch
from safetensors.torch import load_file, save_file
from homeimg_package import validate

p = argparse.ArgumentParser(description=__doc__)
for arg in ("root", "fixtures", "driver"):
    p.add_argument(arg, type=Path)
args = p.parse_args()
metadata = json.loads((args.fixtures / "metadata.json").read_text())
assert metadata["device"] == "cpu" and validate(args.root)["revision"] == metadata["revision"]
spec = importlib.util.spec_from_file_location("epoch49_reference", args.root / "pipeline.py")
ref = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ref
spec.loader.exec_module(ref)
encoder = ref.AuroraLMTextEncoder.from_pretrained(args.root/"text_encoder", local_files_only=True).eval()
for label, prompt in metadata["prompts"].items():
    fixture = args.fixtures / f"text-{label}.safetensors"
    command = [args.driver.resolve(), args.root, fixture]
    with TemporaryDirectory() as tmp, torch.inference_mode():
        if label == "pl":
            data = load_file(fixture)
            ids = data["ids"][:, :int(data["valid"].sum())]
            lm = encoder.lm
            first, second = lm.layers[:2]
            x = lm.word_embeddings(ids).float()
            attention = first.attention(first.input_layernorm(x))
            x = x + attention
            dense = first.mlp(first.post_attention_layernorm(x))
            x = x + dense
            x = second.post_attention_layernorm(x + second.attention(second.input_layernorm(x)))
            gate = second.mlp.gate
            rank = torch.nn.functional.linear(x,gate.weight.float()).sigmoid() + gate.expert_bias.float()
            groups = rank.reshape(-1,8,16).topk(2).values.sum(-1).topk(2).indices
            mask = torch.zeros_like(rank.reshape(-1,8,16)[:,:,0], dtype=torch.bool).scatter_(1,groups,True)
            experts = rank.view(-1,8,16).masked_fill(~mask[:,:,None],-torch.inf).flatten(1).topk(8).indices
            path = Path(tmp)/"layers.safetensors"
            save_file(dict(attention=attention,dense=dense,experts=experts.float(),moe=second.mlp(x)),path)
            command.append(path)
        print(label, flush=True)
        subprocess.run(command,input=prompt,text=True,check=True)
