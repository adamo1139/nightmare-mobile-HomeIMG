"""Generate epoch49 intermediate tensors with the pinned Python implementation.

    python tools/homeimg_fixtures.py MODEL_DIRECTORY OUTPUT_DIRECTORY --device cpu
    python tools/check_homeimg_fixtures.py OUTPUT_DIRECTORY/cpu-epoch49

Tokens and masks are exact. Compare floating tensors in their recorded profile.
The FP32 (1e-4) and BF16 (0.02) atol/rtol values are initial regression
thresholds; calibration across device kernels requires target measurements.
"""
import argparse
import importlib.util
import json
import tempfile
import sys
from pathlib import Path

import torch
from diffusers import AutoencoderKLFlux2
from safetensors.torch import save_file
from transformers import PreTrainedTokenizerFast
from homeimg_package import validate

PROMPTS = {"pl": "Rudy lis w zaśnieżonym lesie.", "en": "A red fox in a snowy forest.",
           "empty": "", "long": "las " * 320}


@torch.inference_mode()
def generate(args):
    suffix = "" if args.image_prompt == "pl" else f"-{args.image_prompt}"
    destination = args.output / f"{args.device}-epoch49{suffix}"
    if destination.exists():
        raise FileExistsError(f"fixture directory already exists: {destination}")
    root = args.root.resolve()
    manifest = validate(root)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("CUDA BF16 required for this profile")
    spec = importlib.util.spec_from_file_location("epoch49_pipeline", root / "pipeline.py")
    ref = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ref
    spec.loader.exec_module(ref)
    def load(cls, part):
        return cls.from_pretrained(root / part, local_files_only=True).eval().to(device)
    pipe = ref.AuroraIMG2Pipeline(
        PreTrainedTokenizerFast.from_pretrained(root / "tokenizer", local_files_only=True),
        load(ref.AuroraLMTextEncoder, "text_encoder"), load(ref.AuroraDenoiser, "transformer"),
        load(AutoencoderKLFlux2, "vae"))
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output) as temporary:
        stage = Path(temporary)
        for label, prompt in PROMPTS.items():
            data = {}
            hook = pipe.text_encoder.lm.register_forward_hook(
                lambda m, inputs, out: data.update(lm=out.detach().cpu()))
            ids = pipe._tokenize([prompt])
            valid = ids.ne(pipe.text_encoder.config.pad_id)
            try:
                with pipe._autocast():
                    context, mask = pipe.text_encoder(ids[:, :int(valid.sum())].to(device))
            finally:
                hook.remove()
            data.update(ids=ids, valid=valid, context=context.cpu(), context_valid=mask.cpu())
            save_file(data, stage / f"text-{label}.safetensors")
        context, mask = pipe.encode_prompt([PROMPTS[args.image_prompt]], [""], True, device)
        for res in args.resolutions:
            side = res // 16
            z = torch.randn(1,128,side,side,generator=torch.Generator().manual_seed(49)).to(device)
            data = {"noise":z.cpu(), "context":context.cpu(), "valid":mask.cpu()}
            u = torch.linspace(1.,0.,4,device=device)
            shift = res / pipe.transformer.config.time_shift_ref
            ts = shift*u/(1+(shift-1)*u)
            data["times"] = ts.cpu()
            for i, t in enumerate(ts[:-1]):
                with pipe._autocast():
                    v = pipe.transformer(torch.cat([z,z]), t.expand(2), context, mask)
                    neg,pos = v.chunk(2)
                    velocity = neg + 3*(pos-neg)
                z = z + (ts[i+1]-t)*velocity.float()
                data[f"velocity_pair_{i}"] = v.cpu()
                data[f"euler_{i}"] = z.cpu()
            data["decoded"] = pipe.decode_latents(z).cpu()
            save_file({key: value.contiguous() for key, value in data.items()}, stage / f"image-{res}.safetensors")
        metadata = dict(revision=manifest["revision"], device=args.device, torch=str(torch.__version__),
                        resolutions=args.resolutions, image_prompt=args.image_prompt, prompts=PROMPTS, negative_prompt="", cfg=3, seed=49,
                        atol=1e-4 if args.device=="cpu" else 0.02, rtol=1e-4 if args.device=="cpu" else 0.02)
        (stage/"metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
        stage.rename(destination)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--image-prompt", choices=["pl", "en"], default="pl")
    p.add_argument("--device", choices=["cpu","cuda"],default="cpu")
    p.add_argument("--resolutions",nargs="+",type=int,choices=[256,512,1024],default=[256,512,1024])
    generate(p.parse_args())
