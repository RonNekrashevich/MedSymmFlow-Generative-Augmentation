"""Sample class-conditioned synthetic images from a MedSymmFlow checkpoint, for any
of the four MedMNIST datasets the paper covers.

Generalises generate_pneumoniamnist.py: `--per_class` samples for every class rather
than the binary normal/pneumonia pair, and the UNet architecture is inferred from the
checkpoint (see msf_arch.py) instead of hard-coded.

    python project/generate_medmnist.py --checkpoint <path> --dataset bloodmnist \
        --n_classes 8 --per_class 200 --image_size 32 --rgb_mask \
        --solver euler --step_size 0.04 --output_dir /tmp/gen
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
MEDSYMMFLOW_ROOT = SRC_ROOT / "medsymmflow"
for candidate in [str(SRC_ROOT), str(MEDSYMMFLOW_ROOT), str(Path(__file__).resolve().parent)]:
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from medsymmflow.models.SymmFMClass import SymmFMClass
from medsymmflow.data.Dataloaders import pick_dataset
from medsymmflow.utils.util import parse_args_SymmetricFlowMatchingClass
from msf_arch import add_arch_cli, resolve_arch, apply_arch

CLASS_NAMES = {
    "pneumoniamnist": ["normal", "pneumonia"],
    "bloodmnist": ["basophil", "eosinophil", "erythroblast", "ig",
                   "lymphocyte", "monocyte", "neutrophil", "platelet"],
    "dermamnist": ["actinic", "bcc", "keratosis", "dermatofibroma",
                   "melanoma", "nevus", "vascular"],
    "retinamnist": ["grade0", "grade1", "grade2", "grade3", "grade4"],
}


def class_names_for(dataset, n_classes):
    names = CLASS_NAMES.get(dataset, [])
    if len(names) == n_classes:
        return names
    return [f"class_{i}" for i in range(n_classes)]


def build_arg_parser():
    p = argparse.ArgumentParser(description="Generate synthetic MedMNIST samples with MedSymmFlow")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset", type=str, default="pneumoniamnist")
    p.add_argument("--n_classes", type=int, default=2)
    p.add_argument("--per_class", type=int, default=100, help="samples generated per class")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--beta", type=float, default=4.0, help="label-noise amplitude")
    p.add_argument("--image_size", type=int, default=32)
    p.add_argument("--store_size", type=int, default=None,
                   help="resize saved PNGs to this (default: keep image_size)")
    p.add_argument("--solver", type=str, default="euler")
    p.add_argument("--step_size", type=float, default=0.04)
    p.add_argument("--solver_lib", type=str, default="torchdiffeq")
    p.add_argument("--batch", type=int, default=100, help="samples per forward pass")
    p.add_argument("--rgb_mask", action="store_true", default=False)
    p.add_argument("--mask_code", default="rgb", choices=["rgb", "onehot", "thermometer"])
    p.add_argument("--cfg_w", type=float, default=0.0,
                   help=">0: classifier-free guidance weight (model must be cfg-trained)")
    p.add_argument("--latent", action="store_true", default=False)
    p.add_argument("--vae_id", type=str, default=None)
    p.add_argument("--num_workers", type=int, default=0)
    add_arch_cli(p)
    return p


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = build_arg_parser().parse_args()
    set_seed(args.seed)

    arch = resolve_arch(args)
    print("architecture:", {k: arch[k] for k in
                            ("model_channels", "num_res_blocks", "channel_mult",
                             "attention_resolutions", "in_channels")})

    output_root = Path(args.output_dir)
    names = class_names_for(args.dataset, args.n_classes)
    for n in names:
        (output_root / n).mkdir(parents=True, exist_ok=True)

    image_shape, channels, _ = pick_dataset(args.dataset, "val", args.image_size, 1, args.num_workers)

    original_argv = sys.argv.copy()
    try:
        sys.argv = [sys.argv[0]]
        model_args = parse_args_SymmetricFlowMatchingClass()
    finally:
        sys.argv = original_argv

    model_args.train = False
    model_args.sample = True
    model_args.dataset = args.dataset
    model_args.checkpoint = args.checkpoint
    model_args.num_samples = 1
    model_args.n_classes = args.n_classes
    model_args.batch_size = 1
    model_args.num_workers = args.num_workers
    model_args.solver_lib = args.solver_lib
    model_args.solver = args.solver
    model_args.step_size = args.step_size
    model_args.beta = args.beta
    model_args.rgb_mask = args.rgb_mask or arch["rgb_mask"]
    model_args.latent = args.latent
    model_args.vae_id = args.vae_id
    model_args.size = args.image_size
    model_args.mask_code = args.mask_code
    apply_arch(model_args, arch)

    model = SymmFMClass(model_args, image_shape, channels)
    model.load_checkpoint(args.checkpoint)
    model.eval()

    store = args.store_size or args.image_size
    rows = []
    for label in range(args.n_classes):
        name = names[label]
        made = 0
        while made < args.per_class:
            n = min(args.batch, args.per_class - made)
            labels = torch.full((n,), label, dtype=torch.long, device=model.device)
            if args.cfg_w > 0:
                samples = model.sample_guided(n, labels, w=args.cfg_w)
            elif model.vae is not None and model.mask_code == "rgb":
                # Published latent path: sample() expects an ALREADY-ENCODED palette
                # mask (the repo's classification.py does the same encode). Latent-
                # native K-codes need no encode and go through the labels path below.
                mask = model.dequantize_class(labels).to(model.device)
                with torch.no_grad():
                    mask = model.encode(mask).latent_dist.sample().mul_(model.vae_scale)
                samples = model.sample(n, mask=mask, train=False, fid=True)
            else:
                samples = model.sample(n, labels=labels, train=False, fid=True)
            for j, sample in enumerate(samples):
                arr = sample.detach().cpu().float().clamp(0, 1).permute(1, 2, 0).numpy()
                if arr.shape[2] == 1:
                    img = Image.fromarray((arr[:, :, 0] * 255).astype("uint8"), mode="L")
                else:
                    img = Image.fromarray((arr * 255).astype("uint8"))
                if img.size != (store, store):
                    img = img.resize((store, store), Image.LANCZOS)
                path = output_root / name / f"{name}_{made + j:05d}.png"
                img.save(path)
                rows.append({"image_path": str(path), "label": label, "class_name": name,
                             "gen_seed": args.seed, "beta": args.beta,
                             "step_size": args.step_size, "checkpoint": args.checkpoint})
            made += n
        print(f"  class {label} ({name}): {made} samples")

    meta_path = output_root / "metadata.csv"
    with meta_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} images to {output_root}")


if __name__ == "__main__":
    main()
