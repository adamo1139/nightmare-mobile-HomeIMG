"""Compare the native token kernel with PoziomkaMoE from a supplied pipeline.py."""
import ast
import ctypes
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F

library, reference = sys.argv[1:]
source = ast.parse(Path(reference).read_text())
names = {"_lin", "PoziomkaMLP", "PoziomkaGate", "PoziomkaMoE"}
source.body = [n for n in source.body if getattr(n, "name", "") in names]
exec(compile(source, reference, "exec"))
torch.manual_seed(49)
torch.set_num_threads(1)
d, h, shared = 32, 12, 24
model = PoziomkaMoE(d, h, 128, 8, 8, 2, 2.5, shared).bfloat16()
with torch.no_grad():
    for value in model.parameters():
        value.uniform_(-0.3, 0.3)
    for name, value in model.named_buffers():
        if value.dtype == torch.int8:
            value.random_(-100, 100)
        else:
            value.uniform_(0.01, 0.1)
    bias = model.gate.expert_bias.float().contiguous()
    kernel = ctypes.CDLL(library).poziomka_moe_token
    kernel.argtypes = [ctypes.c_int]*3 + [ctypes.c_void_p]*12
    kernel.restype = None
    for x in torch.randn(32, d).bfloat16():
        indices = torch.empty(8, dtype=torch.int32)
        actual = torch.empty_like(x)
        weights = model.shared_experts
        tensors = [x, model.gate.weight, bias, model._wgu, model._wgu_s,
                   model._wd, model._wd_s, weights.gate_proj.weight,
                   weights.up_proj.weight, weights.down_proj.weight, indices, actual]
        kernel(d, h, shared, *(t.data_ptr() for t in tensors))
        rank = F.linear(x.float(), model.gate.weight.float()).sigmoid() + bias
        groups = rank.view(8, 16).topk(2).values.sum(-1).topk(2).indices
        masked = torch.full_like(rank.view(8, 16), -torch.inf)
        masked[groups] = rank.view(8, 16)[groups]
        assert torch.equal(indices, masked.flatten().topk(8).indices)
        torch.testing.assert_close(actual, model(x[None])[0], rtol=0, atol=0)
print("32 tokens matched reference routing and BF16 outputs exactly")
