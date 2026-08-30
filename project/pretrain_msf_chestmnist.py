"""Pretrain the MSF generator on ChestMNIST before fine-tuning on PneumoniaMNIST.

ChestMNIST (NIH ChestX-ray14, 28px) supplies ~43k chest X-rays to teach the flow
model the CXR manifold; train_msf_scratch.py then fine-tunes on the (much smaller)
PneumoniaMNIST generator half via --init-checkpoint. Only the two semantically
clean groups are used, mapped onto the same 2-class mask conditioning:

    class 0 = no finding        (all 14 labels zero, ~42k)
    class 1 = pneumonia-positive (~1k)

Images with other findings but no pneumonia are EXCLUDED: calling a mass or an
effusion "normal" would poison the class-0 mask. Classes are rebalanced with a
weighted sampler so the class-1 conditioning pathway actually trains.

Note the domain gap: ChestMNIST is adult NIH data, PneumoniaMNIST is pediatric.
That is exactly why this is a pretraining stage and not extra labeled data --
the fine-tune re-anchors both classes to the pediatric domain.
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
from medmnist import ChestMNIST

from medsymmflow.models.SymmFMClass import SymmFMClass
from medsymmflow.utils.util import parse_args_SymmetricFlowMatchingClass
from config import models_dir

PNEUMONIA_COL = 6  # ChestMNIST label column for 'pneumonia'


class BinaryChest(Dataset):
    """ChestMNIST restricted to (no-finding | pneumonia), with int binary labels."""

    def __init__(self, base, idx, labels):
        self.base, self.idx, self.labels = base, list(idx), labels

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        x, _ = self.base[self.idx[i]]
        return x, int(self.labels[i])


def build_arg_parser():
    p = argparse.ArgumentParser(description="Pretrain MSF on ChestMNIST (no-finding vs pneumonia)")
    p.add_argument("--out", type=str, required=True, help="pretrained checkpoint destination (.pt)")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=4.0)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--snapshots", type=int, default=6)
    p.add_argument("--sample-freq", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="debug: cap the dataset size")
    return p


def main():
    args = build_arg_parser().parse_args()
    assert args.epochs >= 2, "need at least 2 epochs (OneCycleLR warmup + anneal)"
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tf = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    base = ChestMNIST(split="train", download=True, transform=tf)
    L = np.array(base.labels)
    no_finding = L.sum(axis=1) == 0
    pneu = L[:, PNEUMONIA_COL] == 1
    keep = np.where(no_finding | pneu)[0]
    labels = pneu[keep].astype(int)
    if args.limit:
        rng = np.random.default_rng(args.seed)
        sub = rng.choice(len(keep), size=min(args.limit, len(keep)), replace=False)
        keep, labels = keep[sub], labels[sub]
    ds = BinaryChest(base, keep, labels)
    print(f"chestmnist pretrain set: {len(ds)} "
          f"(no-finding {int((labels == 0).sum())} / pneumonia {int((labels == 1).sum())}), "
          f"excluded other-findings-only images")

    # 50/50 sampler: without it class-1 masks appear once every ~44 batches.
    class_w = 1.0 / np.bincount(labels, minlength=2).clip(min=1)
    sampler = WeightedRandomSampler(class_w[labels], num_samples=len(ds), replacement=True)
    train_loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler,
                              pin_memory=True, num_workers=args.num_workers)
    val_base = ChestMNIST(split="val", download=True, transform=tf)
    Lv = np.array(val_base.labels)
    keep_v = np.where((Lv.sum(axis=1) == 0) | (Lv[:, PNEUMONIA_COL] == 1))[0]
    val_ds = BinaryChest(val_base, keep_v, (Lv[keep_v, PNEUMONIA_COL] == 1).astype(int))
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=True, pin_memory=True)

    original_argv = sys.argv.copy()
    try:
        sys.argv = [sys.argv[0]]
        model_args = parse_args_SymmetricFlowMatchingClass()
    finally:
        sys.argv = original_argv
    model_args.train = True
    model_args.dataset = "chestmnist"
    model_args.n_classes = 2
    model_args.batch_size = args.batch_size
    model_args.n_epochs = args.epochs
    model_args.lr = args.lr
    model_args.warmup = min(args.warmup, max(1, args.epochs // 2))
    model_args.beta = args.beta
    model_args.rgb_mask = True
    model_args.size = 32
    model_args.no_wandb = True
    model_args.num_workers = args.num_workers
    model_args.snapshots = max(1, min(args.snapshots, args.epochs))
    model_args.sample_and_save_freq = args.sample_freq
    # Architecture of the published RGB_28 checkpoint (must match train_msf_scratch.py).
    model_args.model_channels = 64
    model_args.num_res_blocks = 2
    model_args.channel_mult = (1, 2, 2, 2)
    model_args.num_heads = 4
    model_args.num_head_channels = 64
    model_args.attention_resolutions = (2,)

    model = SymmFMClass(model_args, 32, 1)
    print(f"pretraining: {args.epochs} epochs, {len(train_loader)} steps/epoch, "
          f"batch {args.batch_size}")

    start = time.time()
    model.train_model(train_loader, val_loader)

    snap_dir = Path(models_dir) / "SymmetricalFlowMatchingClass"
    pattern = f"FM_chestmnist_beta{args.beta}_rgb_epoch*.pt"
    snaps = [p for p in snap_dir.glob(pattern) if p.stat().st_mtime >= start - 60]
    assert snaps, f"no fresh snapshot matching {pattern} in {snap_dir}"
    latest = max(snaps, key=lambda p: int(p.stem.split("epoch")[1]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(latest, out)
    manifest = {
        "pretrain_dataset": "chestmnist", "n_train": len(ds),
        "n_class0": int((labels == 0).sum()), "n_class1": int((labels == 1).sum()),
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "beta": args.beta, "seed": args.seed, "source_snapshot": latest.name,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    print(f"pretrained checkpoint -> {out}  ({(time.time() - start)/3600:.2f} h)")


if __name__ == "__main__":
    main()
