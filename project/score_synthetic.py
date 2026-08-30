"""Score a synthetic pool by the GENERATOR'S OWN reverse-flow classification.

For every generated image, the generator that made it classifies it back:
    pred    = the class the generator recognises in its own output
    margin  = confidence (runner-up distance minus best distance; higher = surer)
    match   = pred equals the requested label

Used by the self-consistency filter: keep, per class, round-trip matches in the
top-q fraction by margin. Writes <synthetic-dir>/self_scores.csv aligned to the
pool's metadata.csv image_path column.

    python project/score_synthetic.py --checkpoint <gen.pt> \
        --synthetic-dir <run>/synthetic_28 --dataset retinamnist --n_classes 5
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
MEDSYMMFLOW_ROOT = SRC_ROOT / "medsymmflow"
for candidate in [str(SRC_ROOT), str(MEDSYMMFLOW_ROOT), str(Path(__file__).resolve().parent)]:
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from medsymmflow.models.SymmFMClass import SymmFMClass
from medsymmflow.utils.util import parse_args_SymmetricFlowMatchingClass
from msf_arch import resolve_arch, apply_arch, add_arch_cli
from datasets_meta import dataset_meta


def build_arg_parser():
    p = argparse.ArgumentParser(description="Generator self-consistency scoring")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--synthetic-dir", required=True)
    p.add_argument("--dataset", default="retinamnist")
    p.add_argument("--n_classes", type=int, default=5)
    p.add_argument("--image_size", type=int, default=32)
    p.add_argument("--beta", type=float, default=4.0)
    p.add_argument("--mask_code", default="rgb", choices=["rgb", "onehot", "thermometer"])
    p.add_argument("--solver", default="euler")
    p.add_argument("--step_size", type=float, default=0.04)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--out-name", default="self_scores.csv",
                   help="output CSV name inside the synthetic dir (use a distinct "
                        "name for step-size variants so score files never collide)")
    p.add_argument("--rgb_mask", action="store_true", default=True)
    p.add_argument("--latent", action="store_true", default=False)
    p.add_argument("--vae_id", type=str, default=None)
    add_arch_cli(p)
    return p


def main():
    args = build_arg_parser().parse_args()
    meta = dataset_meta(args.dataset)
    channels = meta["channels"]
    arch = resolve_arch(args)

    original_argv = sys.argv.copy()
    try:
        sys.argv = [sys.argv[0]]
        model_args = parse_args_SymmetricFlowMatchingClass()
    finally:
        sys.argv = original_argv
    model_args.train = False
    model_args.dataset = args.dataset
    model_args.n_classes = args.n_classes
    model_args.solver_lib = "torchdiffeq"
    model_args.solver = args.solver
    model_args.step_size = args.step_size
    model_args.beta = args.beta
    model_args.rgb_mask = True
    model_args.latent = args.latent
    model_args.vae_id = args.vae_id
    model_args.size = args.image_size
    model_args.mask_code = args.mask_code
    apply_arch(model_args, arch)

    model = SymmFMClass(model_args, args.image_size, channels)
    model.load_checkpoint(args.checkpoint)
    model.eval()

    mode = "L" if channels == 1 else "RGB"
    tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    syn_dir = Path(args.synthetic_dir)
    pool = pd.read_csv(syn_dir / "metadata.csv")
    print(f"scoring {len(pool)} synthetic images with their own generator")

    preds, margins, dmins, dtrues = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(pool), args.batch):
            chunk = pool.iloc[i:i + args.batch]
            xs = torch.stack([tf(Image.open(p).convert(mode)) for p in chunk.image_path])
            xs = xs.to(model.device)
            if model.vae is not None:   # latent path: segment() expects encoded input
                if xs.shape[1] == 1:
                    xs = torch.cat((xs, xs, xs), dim=1)
                xs = model.encode(xs).latent_dist.sample().mul_(model.vae_scale)
            mask = model.segment(xs.shape[0], xs, train=False, eval=True)
            d = model.distance_to_classes(mask).float().cpu().numpy()
            s = np.sort(d, axis=1)
            preds.extend(d.argmin(axis=1).tolist())
            margins.extend((s[:, 1] - s[:, 0]).tolist())   # runner-up minus best
            dmins.extend(s[:, 0].tolist())                 # distance to the winner
            dtrues.extend(d[np.arange(len(d)), chunk.label.values].tolist())
            if (i // args.batch) % 5 == 0:
                print(f"  {i + len(chunk)}/{len(pool)}")

    out = pool[["image_path", "label"]].copy()
    out["pred"] = preds
    out["margin"] = margins
    out["dmin"] = dmins        # published convention: small = confident
    out["dtrue"] = dtrues      # distance to the requested class
    out["match"] = (out.pred == out.label).astype(int)
    out.to_csv(syn_dir / args.out_name, index=False)
    match_rate = out.groupby("label")["match"].mean().round(3).to_dict()
    print("round-trip match rate per class:", match_rate)
    print("->", syn_dir / args.out_name)


if __name__ == "__main__":
    main()
