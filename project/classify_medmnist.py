"""Classify a real MedMNIST split with MedSymmFlow's own reverse-flow direction.

This is the *discriminative* half of the symmetric velocity field: a real image is
concatenated with noise in the mask channels and integrated backwards in time, and
the recovered mask is decoded to a class by nearest palette colour.

Writes per-image predictions AND the soft distance-to-class scores, so downstream
code can compute AUC and compare decision functions, not just accuracy.

    python project/classify_medmnist.py --checkpoint <path> --dataset pneumoniamnist \
        --n_classes 2 --image_size 32 --rgb_mask --output_csv msf_test.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision import transforms

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

import medmnist

DATASET_CLASS = {
    "pneumoniamnist": "PneumoniaMNIST",
    "bloodmnist": "BloodMNIST",
    "dermamnist": "DermaMNIST",
    "retinamnist": "RetinaMNIST",
}


def build_arg_parser():
    p = argparse.ArgumentParser(description="Classify MedMNIST with MedSymmFlow's reverse flow")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output_csv", type=str, required=True)
    p.add_argument("--dataset", type=str, default="pneumoniamnist")
    p.add_argument("--n_classes", type=int, default=2)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--image_size", type=int, default=32)
    p.add_argument("--source_size", type=int, default=64,
                   help="MedMNIST source resolution to download before resizing")
    p.add_argument("--beta", type=float, default=4.0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--solver", type=str, default="euler")
    p.add_argument("--step_size", type=float, default=0.04)
    p.add_argument("--solver_lib", type=str, default="torchdiffeq")
    p.add_argument("--rgb_mask", action="store_true", default=False)
    p.add_argument("--mask_code", default="rgb", choices=["rgb", "onehot", "thermometer"])
    p.add_argument("--latent", action="store_true", default=False)
    p.add_argument("--vae_id", type=str, default=None)
    p.add_argument("--seg_t_start", type=float, default=1.0,
                   help="start time of the reverse integration (default 1.0)")
    p.add_argument("--seg_t_end", type=float, default=0.0,
                   help="time at which the class code is read (default 0.0; try 0.1 "
                        "for generators trained with logit-normal timesteps, whose "
                        "endpoints are weakly trained)")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="debug: only classify N images")
    p.add_argument("--ensemble", type=int, default=1,
                   help=">1: average class distances over N reverse-flow noise draws "
                        "(the paper is single-pass; this is our extension)")
    add_arch_cli(p)
    return p


def main():
    args = build_arg_parser().parse_args()
    arch = resolve_arch(args)
    print("architecture:", {k: arch[k] for k in
                            ("model_channels", "num_res_blocks", "channel_mult",
                             "attention_resolutions", "in_channels")})

    image_shape, channels, _ = pick_dataset(args.dataset, "val", args.image_size, 1, args.num_workers)

    original_argv = sys.argv.copy()
    try:
        sys.argv = [sys.argv[0]]
        model_args = parse_args_SymmetricFlowMatchingClass()
    finally:
        sys.argv = original_argv

    model_args.train = False
    model_args.sample = False
    model_args.classification = True
    model_args.dataset = args.dataset
    model_args.checkpoint = args.checkpoint
    model_args.n_classes = args.n_classes
    model_args.batch_size = args.batch_size
    model_args.num_workers = args.num_workers
    model_args.solver_lib = args.solver_lib
    model_args.solver = args.solver
    model_args.step_size = args.step_size
    model_args.beta = args.beta
    model_args.rgb_mask = args.rgb_mask or arch["rgb_mask"]
    model_args.latent = args.latent
    model_args.vae_id = args.vae_id
    model_args.seg_t_start = args.seg_t_start
    model_args.seg_t_end = args.seg_t_end
    model_args.size = args.image_size
    model_args.mask_code = args.mask_code
    apply_arch(model_args, arch)

    model = SymmFMClass(model_args, image_shape, channels)
    model.load_checkpoint(args.checkpoint)
    model.eval()

    # Mirror the repo's own preprocessing: resize to image_size, the dataset's
    # native channel count (grayscale only for 1-channel datasets), [-1, 1].
    tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        *([transforms.Grayscale(num_output_channels=1)] if channels == 1 else []),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    ds_cls = getattr(medmnist, DATASET_CLASS[args.dataset])
    ds = ds_cls(split=args.split, download=True, size=args.source_size)
    n_total = len(ds) if args.limit is None else min(args.limit, len(ds))

    trues, mean_preds, mode_preds, dists = [], [], [], []
    buf_x, buf_y = [], []

    @torch.no_grad()
    def flush():
        if not buf_x:
            return
        x = torch.stack(buf_x).to(model.device)
        if model.vae is not None:   # latent path: segment() expects encoded input
            if x.shape[1] == 1:
                x = torch.cat((x, x, x), dim=1)
            x = model.encode(x).latent_dist.sample().mul_(model.vae_scale)
        mask = model.segment(x.shape[0], x, train=False, eval=True)
        mean_p, mode_p = model.quantize_class(mask)
        d = model.distance_to_classes(mask)
        for _ in range(args.ensemble - 1):
            m2 = model.segment(x.shape[0], x, train=False, eval=True)
            d = d + model.distance_to_classes(m2)
        if args.ensemble > 1:
            d = d / args.ensemble
            mean_p = d.argmin(dim=1)      # ensemble decision from averaged distances
        mean_preds.extend(mean_p.long().cpu().numpy().tolist())
        mode_preds.extend(mode_p.long().cpu().numpy().tolist())
        dists.append(d.float().cpu().numpy())
        trues.extend(buf_y)
        buf_x.clear(); buf_y.clear()

    for i in range(n_total):
        img, label = ds[i]
        buf_x.append(tf(img))
        buf_y.append(int(np.asarray(label).reshape(-1)[0]))
        if len(buf_x) >= args.batch_size:
            flush()
    flush()

    dist = np.concatenate(dists, axis=0)
    out = pd.DataFrame({"index": np.arange(len(trues)), "true": trues,
                        "msf_pred": mean_preds, "msf_pred_mode": mode_preds})
    # Smaller distance = closer to that class colour -> negate for a score-like column.
    for c in range(args.n_classes):
        out[f"msf_negdist_{c}"] = -dist[:, c]
    out.to_csv(args.output_csv, index=False)

    acc = float((np.array(trues) == np.array(mean_preds)).mean())
    print(f"MSF {args.dataset} [{args.split}] accuracy: {acc:.4f} over {len(trues)} images")
    print("->", args.output_csv)


if __name__ == "__main__":
    main()
