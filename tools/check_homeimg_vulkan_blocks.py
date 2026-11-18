"""Check Vulkan Aurora and VAE kernels against NumPy on the same Vulkan session."""
import argparse
import ctypes as C
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("library", type=Path)
args=parser.parse_args()
lib=C.CDLL(str(args.library.resolve()))
ptr=C.c_void_p
integer=C.c_int
scalar=C.c_float
lib.homeimg_gpu_open.restype=ptr
lib.homeimg_gpu_close.argtypes=[ptr]
lib.homeimg_gpu_mlp.argtypes=[ptr]*2+[integer]*3+[ptr]*6
lib.homeimg_gpu_conv.argtypes=[ptr]*3+[integer]*4+[ptr]
lib.homeimg_gpu_spatial.argtypes=[ptr,ptr,integer,integer,integer,scalar,ptr,ptr,ptr]
lib.homeimg_gpu_masked_attention.argtypes=[ptr]*4+[integer]*3+[scalar,ptr,integer]
lib.homeimg_gpu_group.argtypes=[ptr]*4+[integer]*3+[ptr]
lib.homeimg_gpu_expert.argtypes=[ptr]*4+[integer]*3+[ptr]
lib.homeimg_gpu_weights.argtypes=[ptr,ptr,C.c_size_t]
device=lib.homeimg_gpu_open()
assert device
rng=np.random.default_rng(49)
def matrix(*shape):
    return rng.standard_normal(shape,dtype=np.float32)*.1
def buffers(*arrays):
    return [x.ctypes.data for x in arrays]
def attention(q,k,v,causal=False):
    scores=q@k.T / np.sqrt(np.float32(q.shape[1]))
    if causal:
        scores[np.triu_indices(len(q),1)]=-np.inf
    exp=np.exp(scores-scores.max(axis=1,keepdims=True))
    return (exp/exp.sum(axis=1,keepdims=True))@v
try:
    for rows,width,h in [(32,128,256),(75,113,167),(640,512,1280)]:
        x=matrix(rows,width)
        gate,down=matrix(2*h,width),matrix(width,h)
        norm,scale,bias,residual=matrix(4,width)
        pre=(x/np.sqrt((x*x).mean(1,keepdims=True)+1e-6)*norm)*(1+scale)+bias
        g,u=np.split(pre@gate.T,2,axis=1)
        ref=x+((g/(1+np.exp(-g))*u)@down.T)*residual
        assert lib.homeimg_gpu_mlp(device,x.ctypes.data,rows,width,h,*buffers(gate,down,norm,scale,bias,residual))
        np.testing.assert_allclose(x,ref,atol=2e-5,rtol=1e-4)
    for side,cin,cout,scale in [(17,3,7,1),(67,10,16,2)]:
        x,w=matrix(side,side,cin),matrix(cout,cin,3,3)
        up=np.pad(x.repeat(scale,0).repeat(scale,1),((1,1),(1,1),(0,0)))
        patches=np.lib.stride_tricks.sliding_window_view(up,(3,3),axis=(0,1))
        ref=patches.reshape(-1,cin*9)@w.reshape(cout,-1).T
        y=np.empty_like(ref)
        assert lib.homeimg_gpu_conv(device,*buffers(x,w),side,cin,cout,scale,y.ctypes.data)
        np.testing.assert_allclose(y,ref,atol=1e-5,rtol=1e-4)
    for rows,keys,width,causal in [(10,10,128,1),(4096,11,128,0),(256,256,512,0)]:
        q,k,v=matrix(rows,width),matrix(keys,width),matrix(keys,width)
        y=np.empty_like(q)
        assert lib.homeimg_gpu_masked_attention(device,*buffers(q,k,v),rows,keys,width,width**-.5,y.ctypes.data,causal)
        np.testing.assert_allclose(y,attention(q,k,v,causal),atol=1e-5,rtol=1e-4)
    side,heads=16,3
    tokens=side*side
    raw=matrix(tokens,3,heads,128)
    nq,nk=matrix(2,128)
    y=np.empty((tokens,heads,128),np.float32)
    assert lib.homeimg_gpu_spatial(device,raw.ctypes.data,side,heads,64,128**-.5,*buffers(nq,nk,y))
    row=np.arange(tokens)[:,None]
    pair=np.arange(64)[None,:]
    angles=((np.where(pair<32,row//side,row%side)+.5)*64/side-.5)/10000**((pair%32)/32)
    def rotated(x,w):
        x=x/np.sqrt((x*x).mean(1,keepdims=True)+1e-6)*w
        a,b=x[:,::2],x[:,1::2]
        return np.stack([a*np.cos(angles)-b*np.sin(angles),a*np.sin(angles)+b*np.cos(angles)],axis=-1).reshape(tokens,128)
    for head in range(heads):
        ref=attention(rotated(raw[:,0,head],nq),rotated(raw[:,1,head],nk),raw[:,2,head])
        np.testing.assert_allclose(y[:,head],ref,atol=1e-5,rtol=1e-4)
    for rows,cols in [(65,64),(33031,128)]:
        x=matrix(rows,cols); w,b=matrix(2,cols)
        y=np.empty_like(x)
        assert lib.homeimg_gpu_group(device,*buffers(x,w,b),rows,cols,1,y.ctypes.data)
        g=x.reshape(rows,32,-1).astype(np.float64)
        norm=(g-g.mean((0,2),keepdims=True))/np.sqrt(g.var((0,2),keepdims=True)+1e-6)
        n=norm.reshape(x.shape)*w+b
        np.testing.assert_allclose(y,n/(1+np.exp(-n)),atol=2e-5,rtol=1e-4)
    x=matrix(3,2048)
    packed=rng.integers(-128,128,(2048,640),dtype=np.int8)
    scales=matrix(640);y=np.empty((3,640),np.float32)
    assert lib.homeimg_gpu_expert(device,*buffers(x,packed,scales),3,2048,640,y.ctypes.data)
    np.testing.assert_allclose(y,(x@packed.astype(np.float32))*scales,atol=3e-4,rtol=1e-4)
    # A later checkpoint may reuse the same virtual addresses with new contents.
    # Unregistering the previous mapping must invalidate all cached shards.
    w=matrix(128,128);x=matrix(64,128);y=np.empty_like(x)
    lib.homeimg_gpu_project.argtypes=[ptr]*3+[integer]*3+[ptr]
    for iteration in range(2):
        lib.homeimg_gpu_weights(device,w.ctypes.data,w.nbytes)
        for repeat in range(2):
            assert lib.homeimg_gpu_project(device,*buffers(x,w),64,128,128,y.ctypes.data)
            np.testing.assert_allclose(y,x@w.T,atol=1e-5,rtol=1e-4)
        lib.homeimg_gpu_weights(device,None,0)
        w[:]*=-1
    print("MLP, convolution, causal/cross attention, spatial QKV and head joining passed")
finally:
    lib.homeimg_gpu_close(device)
