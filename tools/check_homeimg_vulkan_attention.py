"""Compare Vulkan self-attention with stable NumPy softmax, including the 1024 px grid."""
import argparse
import ctypes as C
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("library", type=Path)
args = parser.parse_args()
lib = C.CDLL(str(args.library.resolve()))
lib.homeimg_gpu_open.restype = C.c_void_p
lib.homeimg_gpu_close.argtypes = [C.c_void_p]
lib.homeimg_gpu_attention.argtypes = [C.c_void_p]*4 + [C.c_int, C.c_int, C.c_float, C.c_void_p]
lib.homeimg_gpu_project.argtypes = [C.c_void_p]*3 + [C.c_int]*3 + [C.c_void_p]
device = lib.homeimg_gpu_open()
assert device
try:
    rng = np.random.default_rng(49)
    for tokens, amplitude in [(32, 1), (544, 6), (4096, 1)]:
        q, k, v = rng.standard_normal((3, tokens, 128), dtype=np.float32)
        q *= amplitude
        k *= amplitude
        scale = np.float32(128**-0.5)
        output = np.empty_like(v)
        assert lib.homeimg_gpu_attention(device, q.ctypes.data, k.ctypes.data, v.ctypes.data,
                                         tokens, 128, scale, output.ctypes.data)
        for row in range(0, tokens, 64):
            score = (q[row:row+64] @ k.T)*scale
            prob = np.exp(score-score.max(axis=1, keepdims=True))
            expected = (prob/prob.sum(axis=1, keepdims=True)) @ v
            np.testing.assert_allclose(output[row:row+64], expected, atol=1e-4, rtol=1e-4)
        # Aurora interleaves the two kernels using the same mapped scratch buffers.
        weight = rng.standard_normal((128, 128), dtype=np.float32)
        projected = np.empty_like(output)
        assert lib.homeimg_gpu_project(device, output.ctypes.data, weight.ctypes.data, tokens,
                                       128, 128, projected.ctypes.data)
        np.testing.assert_allclose(projected, output @ weight.T, atol=5e-5, rtol=1e-4)
        print(tokens, "tokens passed")
finally:
    lib.homeimg_gpu_close(device)
