"""Desktop epoch49 txt2img. Build the CPU worker from the repo root:
cmake -S backend/homeimg -B build/homeimg -DCMAKE_BUILD_TYPE=Release
cmake --build build/homeimg
python tools/homeimg_txt2img.py /path/to/HomeIMG "Rudy lis w lesie" fox.png
"""
import argparse
import os
from pathlib import Path
import subprocess

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from homeimg_package import validate


def sample(args, noise):
    side = args.resolution // 16
    if noise.dtype != torch.float32 or noise.shape != (1,128,side,side):
        raise ValueError("noise must be FP32 [1,128,resolution/16,resolution/16]")
    command = [args.worker.resolve(),args.model,str(side),str(args.steps),str(args.cfg),
               args.prompt,args.negative]
    env = dict(os.environ, OPENBLAS_NUM_THREADS=str(args.threads))
    payload = noise.flatten(2)[0].T.contiguous().numpy().tobytes()
    with subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,env=env) as worker:
        worker.stdin.write(payload)
        worker.stdin.close()
        for _ in range(args.steps):
            raw = worker.stdout.read(len(payload))
            if len(raw)!=len(payload):
                raise RuntimeError(f"incomplete sampling output (exit {worker.wait()})")
            tokens = torch.from_numpy(np.frombuffer(raw,np.float32).copy()).reshape(side*side,128).T
            yield tokens.reshape_as(noise)
        if worker.wait():
            raise RuntimeError("sampler failed")


def decode(args,z):
    tokens=z.flatten(2)[0].T.contiguous()
    rgb=subprocess.check_output([args.worker.resolve(),"decode",args.model,str(z.shape[-1])],
                                input=tokens.numpy().tobytes(),
                                env=dict(os.environ,OPENBLAS_NUM_THREADS=str(args.threads)))
    return torch.from_numpy(np.frombuffer(rgb,np.float32).copy()).reshape(
        1,args.resolution,args.resolution,3).permute(0,3,1,2)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model",type=Path)
    p.add_argument("prompt")
    p.add_argument("output",type=Path)
    p.add_argument("--worker",type=Path,default=Path(__file__).resolve().parents[1]/"build/homeimg/homeimg_sample")
    p.add_argument("--negative",default="")
    p.add_argument("--resolution",type=int,choices=(256,512,1024),default=256)
    p.add_argument("--steps",type=int,default=50)
    p.add_argument("--cfg",type=float,default=3)
    p.add_argument("--seed",type=int,default=49)
    p.add_argument("--threads",type=int,default=8)
    p.add_argument("--noise",type=Path,help="safetensors containing the input tensor 'noise'")
    p.add_argument("--check-reference",action="store_true",help="compare steps and decoded output to --noise fixture")
    args=p.parse_args()
    if args.steps<1 or args.threads<1 or (args.check_reference and not args.noise):
        p.error("steps and threads must be positive; reference checks require --noise")
    validate(args.model)
    torch.set_num_threads(args.threads)
    side=args.resolution//16
    noise=torch.randn(1,128,side,side,generator=torch.Generator().manual_seed(args.seed))
    if args.noise:
        with safe_open(args.noise,framework="pt") as f:
            noise=f.get_tensor("noise")
    failures=[]
    def check(name,actual):
        if args.check_reference:
            with safe_open(args.noise,framework="pt") as f:
                ref=f.get_tensor(name)
            error=(actual-ref).abs()
            failed=(~torch.isclose(actual,ref,atol=1e-4,rtol=1e-4)).sum().item()
            print(name,"max abs",error.max().item(),"RMS",error.square().mean().sqrt().item(),
                  "outside tolerance",failed,"/",actual.numel(),flush=True)
            if failed:
                failures.append(name)
    for i,z in enumerate(sample(args,noise)):
        check(f"euler_{i}",z)
    image=decode(args,z)
    check("decoded",image)
    rgb=((image[0]+1)*127.5).round().byte().permute(1,2,0).numpy()
    Image.fromarray(rgb).save(args.output)
    if failures:
        raise SystemExit("Reference tolerance exceeded: "+", ".join(failures))


if __name__=="__main__":
    main()
