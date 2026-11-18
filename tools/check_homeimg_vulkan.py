"""Compare host builds of vulkan.cpp with NumPy (same SPIR-V as Android)."""
import argparse
import ctypes as C
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("libraries", nargs="+", type=Path)
    parser.add_argument("--full-qkv", action="store_true",
                        help="also exercise the 1024-pixel attention projection")
    parser.add_argument("--full-cfg", action="store_true",
                        help="also exercise joint positive/negative QKV and MLP projections at 1024 px")
    args = parser.parse_args()
    shapes = [(32, 128, 128), (271, 117, 137), (3, 4096, 2081), (4096, 128, 2560)]
    if args.full_qkv:
        shapes.append((4096, 2560, 7680))
    if args.full_cfg:
        shapes.extend([(8192, 2560, 7680), (8192, 2560, 13824)])
    for path in args.libraries:
        lib = C.CDLL(str(path.resolve()))
        lib.homeimg_gpu_open.restype = C.c_void_p
        lib.homeimg_gpu_close.argtypes = [C.c_void_p]
        lib.homeimg_gpu_project.argtypes = [C.c_void_p] * 3 + [C.c_int] * 3 + [C.c_void_p]
        dispatches = getattr(lib, "homeimg_gpu_dispatches", None)
        if dispatches:
            dispatches.argtypes = [C.c_void_p]
            dispatches.restype = C.c_int64
        device = lib.homeimg_gpu_open()
        if not device:
            raise RuntimeError("Cannot open Vulkan compute device")
        try:
            rng = np.random.default_rng(49)
            for rows, width, channels in shapes:
                x = rng.standard_normal((rows, width), dtype=np.float32) * 0.01
                w = rng.standard_normal((channels, width), dtype=np.float32) * 0.01
                result = np.empty((rows, channels), dtype=np.float32)
                before = dispatches(device) if dispatches else 0
                start = time.perf_counter()
                if not lib.homeimg_gpu_project(device, x.ctypes.data, w.ctypes.data,
                                               rows, width, channels, result.ctypes.data):
                    raise RuntimeError("Projection fell back to CPU")
                elapsed = time.perf_counter() - start
                for i in range(0, rows, 32):
                    np.testing.assert_allclose(result[i:i+32], x[i:i+32] @ w.T, rtol=1e-4, atol=3e-6)
                count = f"{dispatches(device)-before} dispatches" if dispatches else ""
                print(path.name, (rows, width, channels), f"{elapsed:.3f} s", count, flush=True)
        finally:
            lib.homeimg_gpu_close(device)


if __name__ == "__main__":
    main()
