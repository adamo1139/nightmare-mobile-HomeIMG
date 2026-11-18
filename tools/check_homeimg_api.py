"""Verify the C API against a saved epoch49 fixture, including cancellation and errors."""
import argparse
import ctypes as C
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

parser=argparse.ArgumentParser(description=__doc__)
for name in ("library","model","fixtures"):
    parser.add_argument(name,type=Path)
args=parser.parse_args()
meta=json.loads((args.fixtures/"metadata.json").read_text())
with safe_open(args.fixtures/"image-256.safetensors",framework="numpy") as f:
    z=np.ascontiguousarray(f.get_tensor("noise")[0].reshape(128,-1).T)
    ref=f.get_tensor("decoded")[0].transpose(1,2,0)
    steps=len(f.get_tensor("times"))-1
callback=C.CFUNCTYPE(C.c_int,C.c_void_p,C.c_int,C.c_int)
api=C.CDLL(str(args.library.resolve()))
assert api.homeimg_api_version()==1 and api.homeimg_capabilities()==1
api.homeimg_generate.argtypes=[C.c_char_p]*3+[C.c_int,C.c_int,C.c_float,C.c_void_p,C.c_size_t,
    C.c_void_p,C.c_size_t,callback,C.c_void_p,C.c_char_p,C.c_size_t]
rgb=np.full(ref.shape,17,np.uint8)
error=C.create_string_buffer(512)
events=[]
def call(count,cb):
    return api.homeimg_generate(str(args.model).encode(),meta["prompts"][meta.get("image_prompt","pl")].encode(),
        b"",16,steps,meta["cfg"],z.ctypes.data,count,rgb.ctypes.data,rgb.size,callback(cb),None,error,len(error))
assert call(1,lambda *a:1)==1 and error.value
assert call(z.size,lambda *a:0)==2 and np.all(rgb==17)
def proceed(_,n,total):
    events.append((n,total))
    return 1
assert call(z.size,proceed)==0,error.value
assert events==[(n,steps) for n in range(steps+1)]
delta=np.abs(rgb-np.rint((ref+1)*127.5))
assert delta.max()<=1,delta.max()
print("RGB8 max difference",delta.max(),"changed channels",np.count_nonzero(delta),"of",rgb.size)
