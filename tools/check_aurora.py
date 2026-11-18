"""Usage: python check_aurora.py path/to/hf_pipeline.py compiled-check_aurora"""
import ast
import math
from pathlib import Path
from torch import nn
import torch.nn.functional as F
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
import subprocess
import sys
import numpy as np
import torch

names = {"AuroraDenoiser", "DenoiserBlock", "SelfAttention", "CrossAttention", "RMSNorm",
         "SwiGLU", "build_rope_2d", "_angles", "rotate_half_pairs", "apply_rope",
         "timestep_embedding", "micro_cond_embedding", "default_micro_cond"}
source = ast.parse(Path(sys.argv[1]).read_text())
source.body = [node for node in source.body if getattr(node, "name", "") in names]
exec(compile(source, sys.argv[1], "exec"))
torch.manual_seed(49)
torch.set_num_threads(1)
model = AuroraDenoiser(latent_channels=4, latent_size=3, dim=16, depth=2,
    heads=2, mlp_hidden=24, text_dim=12, rope_ref=64, micro_cond=256)
with torch.no_grad():
    for block in model.blocks:
        block.mod_table.normal_(std=0.2)
    latent = torch.randn(1,4,3,3)
    text = torch.randn(1,5,12)
    mask = torch.tensor([[True,False,True,True,False]])
    tensors = dict(model.state_dict(), latent=latent.flatten(2)[0].T, text=text[mask])
    payload = bytearray()
    for key, tensor in tensors.items():
        matrix = tensor.reshape(tensor.shape[0], -1)
        payload.extend(f"{key} {matrix.shape[0]} {matrix.shape[1]}\n".encode())
        payload.extend(matrix.numpy().tobytes())
    output = subprocess.run([sys.argv[2]], input=payload, capture_output=True, check=True)
    actual = np.frombuffer(output.stdout, dtype=np.float32).reshape(3,9,4)
    for index, t in enumerate((0.,0.37,1.)):
        expected = model(latent,torch.tensor([t]),text,mask).flatten(2)[0].T.numpy()
        np.testing.assert_allclose(actual[index], expected, atol=2e-5, rtol=2e-5)
        print(t, np.abs(actual[index]-expected).max())
