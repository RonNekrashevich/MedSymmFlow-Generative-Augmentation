"""Train the MSF generator on an EXTERNAL corpus npz (e.g. APTOS 2019).

The external corpus shares the task's label space (5 DR grades) but comes from
a different source than the benchmark, so a generator trained here has seen
ZERO benchmark images — the strongest possible leakage claim. The resulting
checkpoint plugs into the pipeline either as-is (run_experiment
--external-checkpoint) or as a warm start for disjoint fine-tuning
(run_experiment --gen-init-checkpoint).

    python project/train_msf_external.py --npz /storage/medsymm/external/aptos_28.npz \
        --out /storage/medsymm/weights/scratch/FM_external_aptos_e300_beta4.0_rgb.pt \
        --epochs 300 --balance-classes
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
MEDSYMMFLOW_ROOT = SRC_ROOT / "medsymmflow"
for candidate in [str(SRC_ROOT), str(MEDSYMMFLOW_ROOT)]:
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import numpy as np
from PIL import Image

from medsymmflow.models.SymmFMClass import SymmFMClass
from medsymmflow.utils.util import parse_args_SymmetricFlowMatchingClass
from config import models_dir


class NpzDataset(Dataset):
    def __init__(self, images, labels, tf):
        self.images, self.labels, self.tf = images, labels, tf

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.tf(Image.fromarray(self.images[i])), int(self.labels[i])


def build_arg_parser():
    p = argparse.ArgumentParser(description="Train MSF on an external corpus npz")
    p.add_argument("--npz", required=True, help="npz with images uint8 (N,H,W,3), labels int")
    p.add_argument("--out", required=True, help="final checkpoint destination (.pt)")
    p.add_argument("--tag", default="external", help="name used in snapshot files/manifest")
    p.add_argument("--n-classes", type=int, default=5)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=4.0)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--snapshots", type=int, default=10)
    p.add_argument("--sample-freq", type=int, default=50)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--size", type=int, default=32,
                   help="training resolution (256 with --latent; the npz must hold "
                        "images at >= this size, e.g. preprocess_aptos --size 256)")
    p.add_argument("--latent", action="store_true",
                   help="LatMSF: flow in VAE latent space (use --size 256)")
    p.add_argument("--mask-code", default="rgb", choices=["rgb", "onehot", "thermometer"])
    p.add_argument("--model-channels", type=int, default=64)
    p.add_argument("--t-lognorm", action="store_true")
    p.add_argument("--vae-id", type=str, default=None)
    p.add_argument("--repa-weight", type=float, default=0.0)
    p.add_argument("--repa-teacher", type=str, default=None)
    p.add_argument("--balance-classes", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=None, help="debug: cap corpus size")
    return p


def main():
    args = build_arg_parser().parse_args()
    assert args.epochs >= 2, "need at least 2 epochs (OneCycleLR warmup + anneal)"
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data = np.load(args.npz)
    images, labels = data["images"], data["labels"].astype(int)
    if args.limit:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(len(labels), size=min(args.limit, len(labels)), replace=False)
        images, labels = images[keep], labels[keep]
    counts = np.bincount(labels, minlength=args.n_classes).tolist()
    print(f"external corpus {Path(args.npz).name}: {len(labels)} images, per class {counts}")

    tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Resize(args.size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    ds = NpzDataset(images, labels, tf)
    sampler = None
    if args.balance_classes:
        class_w = 1.0 / np.bincount(labels, minlength=args.n_classes).clip(min=1)
        sampler = WeightedRandomSampler(class_w[labels], num_samples=len(ds), replacement=True)
        print("class-balanced sampling ON")
    train_loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler,
                              shuffle=(sampler is None), pin_memory=True,
                              num_workers=args.num_workers)
    # tiny held-out slice purely for the periodic sample visualisation
    val_loader = DataLoader(ds, batch_size=16, shuffle=True, pin_memory=True)

    original_argv = sys.argv.copy()
    try:
        sys.argv = [sys.argv[0]]
        model_args = parse_args_SymmetricFlowMatchingClass()
    finally:
        sys.argv = original_argv
    model_args.train = True
    model_args.dataset = args.tag
    model_args.n_classes = args.n_classes
    model_args.batch_size = args.batch_size
    model_args.n_epochs = args.epochs
    model_args.lr = args.lr
    model_args.warmup = min(args.warmup, max(1, args.epochs // 2))
    model_args.beta = args.beta
    model_args.rgb_mask = True
    model_args.size = args.size
    model_args.no_wandb = True
    model_args.num_workers = args.num_workers
    model_args.snapshots = max(1, min(args.snapshots, args.epochs))
    model_args.sample_and_save_freq = args.sample_freq
    model_args.latent = args.latent
    model_args.mask_code = args.mask_code
    model_args.t_lognorm = args.t_lognorm
    model_args.vae_id = args.vae_id
    model_args.repa_weight = args.repa_weight
    model_args.repa_teacher = args.repa_teacher
    model_args.model_channels = args.model_channels
    model_args.num_res_blocks = 2
    model_args.channel_mult = (1, 2, 2, 2)
    model_args.num_heads = 4
    model_args.num_head_channels = 64
    model_args.attention_resolutions = (2,)
    model_args.dropout = args.dropout

    model = SymmFMClass(model_args, args.size, 3)
    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"training: {n_params/1e6:.1f}M params, {args.epochs} epochs, "
          f"{len(train_loader)} steps/epoch")

    start = time.time()
    model.train_model(train_loader, val_loader)

    snap_dir = Path(models_dir) / "SymmetricalFlowMatchingClass"
    pattern = f"{'LatFM' if args.latent else 'FM'}_{args.tag}_beta{args.beta}_rgb_epoch*.pt"
    snaps = [p for p in snap_dir.glob(pattern) if p.stat().st_mtime >= start - 60]
    assert snaps, f"no fresh snapshot matching {pattern} in {snap_dir}"
    latest = max(snaps, key=lambda p: int(p.stem.split("epoch")[1]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(latest, out)
    out.with_suffix(".json").write_text(json.dumps({
        "corpus": Path(args.npz).name, "tag": args.tag, "n_train": len(labels),
        "per_class": counts, "epochs": args.epochs, "batch_size": args.batch_size,
        "lr": args.lr, "dropout": args.dropout, "balance_classes": args.balance_classes,
        "size": args.size, "latent": args.latent, "mask_code": args.mask_code,
        "model_channels": args.model_channels, "t_lognorm": args.t_lognorm,
        "vae_id": args.vae_id or "", "repa_weight": args.repa_weight,
        "repa_teacher": args.repa_teacher or "",
        "beta": args.beta, "seed": args.seed, "source_snapshot": latest.name,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))
    print(f"checkpoint -> {out}  ({(time.time() - start)/3600:.2f} h)")


if __name__ == "__main__":
    main()
