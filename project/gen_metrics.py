"""Generative-quality metrics + t-SNE coordinates for the report.

Computes FID and KID (torchmetrics, InceptionV3-2048 features, standard 299px
preprocessing) for each synthetic pool against the real train split, and
2-D t-SNE coordinates of ImageNet-ResNet18 penultimate features for a small
sample per group. Everything is printed to stdout: a metrics table plus CSV
blocks (panel,group,x,y) that the report figures are built from.

    python project/gen_metrics.py --sample 1500 --tsne-per-group 300
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "project"))

import medmnist

POOLS = {
    "pixel-28": "/storage/medsymm/runs/retina-g75/synthetic_28/metadata.csv",
    "pixel-64": "/storage/medsymm/runs/retina-64px/synthetic_28/metadata.csv",
    "latent-224 (paper recipe)": "/storage/medsymm/runs/retina-lat256hr/synthetic_224/metadata.csv",
    "latent-224 (ours v3e)": "/storage/medsymm/runs/retina-latv3ehr/synthetic_224/metadata.csv",
    "latent-224 (ours v2)": "/storage/medsymm/runs/retina-latv2hr/synthetic_224/metadata.csv",
}

TSNE_PANELS = [
    ("A", ["real", "latent-224 (paper recipe)", "latent-224 (ours v3e)"]),
    ("B", ["real", "pixel-28", "pixel-64"]),
]


class Paths224(Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = list(paths), tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


class Arrays224(Dataset):
    def __init__(self, arrays, tf):
        self.arrays, self.tf = arrays, tf

    def __len__(self):
        return len(self.arrays)

    def __getitem__(self, i):
        return self.tf(Image.fromarray(self.arrays[i]).convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=1500, help="images per pool for FID/KID")
    ap.add_argument("--tsne-per-group", type=int, default=300)
    ap.add_argument("--kid-subset", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance

    to_uint8_299 = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.PILToTensor(),      # uint8, what torchmetrics expects with normalize=False
    ])

    real_ds = getattr(medmnist, "RetinaMNIST")(split="train", download=True, size=224)
    real_arrays = [np.asarray(real_ds[i][0]) for i in range(len(real_ds))]
    real224 = Arrays224(real_arrays, to_uint8_299)
    print(f"real train images: {len(real224)} (224px source)")

    def feed(metric, ds, real):
        loader = DataLoader(ds, batch_size=64, num_workers=2)
        for batch in loader:
            metric.update(batch.to(device), real=real)

    print("\n===== FID / KID vs real train (Inception-2048, 299px) =====")
    print("pool, n_images, FID, KID_mean, KID_std")
    for name, meta in POOLS.items():
        if not Path(meta).exists():
            print(f"{name}, MISSING")
            continue
        paths = pd.read_csv(meta).image_path.tolist()
        if len(paths) > args.sample:
            paths = [paths[i] for i in rng.choice(len(paths), args.sample, replace=False)]
        fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
        kid = KernelInceptionDistance(subset_size=args.kid_subset, normalize=False).to(device)
        feed(fid, real224, True)
        feed(fid, Paths224(paths, to_uint8_299), False)
        feed(kid, real224, True)
        feed(kid, Paths224(paths, to_uint8_299), False)
        km, ks = kid.compute()
        print(f"{name}, {len(paths)}, {float(fid.compute()):.2f}, "
              f"{float(km):.4f}, {float(ks):.4f}")
        del fid, kid
        torch.cuda.empty_cache() if device.type == "cuda" else None

    # ---------------- t-SNE coordinates (ResNet18 ImageNet features) ----------
    enc = resnet18(weights=ResNet18_Weights.DEFAULT)
    enc.fc = torch.nn.Identity()
    enc = enc.to(device).eval()
    feat_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    @torch.no_grad()
    def features(ds):
        out = []
        for batch in DataLoader(ds, batch_size=64, num_workers=2):
            out.append(enc(batch.to(device)).cpu().numpy())
        return np.concatenate(out)

    group_feats = {}
    idx = rng.choice(len(real_arrays), min(args.tsne_per_group, len(real_arrays)), replace=False)
    group_feats["real"] = features(Arrays224([real_arrays[i] for i in idx], feat_tf))
    for name, meta in POOLS.items():
        if not Path(meta).exists():
            continue
        paths = pd.read_csv(meta).image_path.tolist()
        sel = [paths[i] for i in rng.choice(len(paths),
                                            min(args.tsne_per_group, len(paths)),
                                            replace=False)]
        group_feats[name] = features(Paths224(sel, feat_tf))

    from sklearn.manifold import TSNE
    for panel, groups in TSNE_PANELS:
        groups = [g for g in groups if g in group_feats]
        X = np.concatenate([group_feats[g] for g in groups])
        labels = sum([[g] * len(group_feats[g]) for g in groups], [])
        Y = TSNE(n_components=2, perplexity=30, random_state=args.seed,
                 init="pca").fit_transform(X)
        print(f"\n===== TSNE PANEL {panel} =====")
        print("group,x,y")
        for g, (x, y) in zip(labels, Y):
            print(f"{g},{x:.3f},{y:.3f}")


if __name__ == "__main__":
    main()
