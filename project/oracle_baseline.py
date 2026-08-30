"""Upper-bound control: train the classifier directly on ALL real training data.

The disjoint protocol reserves 810 of the 1,080 RetinaMNIST training images for
the generator, so no classifier in the main study may use them. That raises a
fair objection: the synthetic settings indirectly benefit from those 810 images,
while the real-only baselines never see them. This script answers the objection
by breaking the protocol on purpose and training the same ResNet-18, with the
same recipe, on the full training split. It is a reference number, not part of
the leakage-free study, and is reported as such.

    python project/oracle_baseline.py --image-size 224 --seeds 0 1 2 3 4
    python project/oracle_baseline.py --image-size 224 --scratch-clf --budgets 810 1080
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score, cohen_kappa_score
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import medmnist
from datasets_meta import dataset_meta

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_model(n_classes, image_size, pretrained, device):
    model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    if image_size <= 64:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model.to(device)


def evaluate(model, loader, device, n_classes):
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            p = torch.softmax(model(x.to(device)), dim=1).cpu().numpy()
            probs.append(p)
            ys.append(np.asarray(y).reshape(-1))
    p = np.concatenate(probs)
    y = np.concatenate(ys)
    pred = p.argmax(1)
    auc = (roc_auc_score(y, p, multi_class="ovr", average="macro") if n_classes > 2
           else roc_auc_score(y, p[:, 1]))
    return auc, accuracy_score(y, pred), cohen_kappa_score(y, pred, weights="quadratic")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="retinamnist")
    ap.add_argument("--image-size", type=int, default=224, choices=[28, 64, 224])
    ap.add_argument("--budgets", type=int, nargs="+", default=None,
                    help="number of real images to train on (default: the full split)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--scratch-clf", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = dataset_meta(args.dataset)
    n_classes = meta["n_classes"]
    ds_cls = getattr(medmnist, meta["medmnist_class"])

    to3 = ([transforms.Grayscale(num_output_channels=3)] if meta["channels"] == 1 else [])
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        *to3, transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    eval_tf = transforms.Compose([
        *to3, transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    train = ds_cls(split="train", transform=train_tf, download=True, size=args.image_size)
    val = ds_cls(split="val", transform=eval_tf, download=True, size=args.image_size)
    test = ds_cls(split="test", transform=eval_tf, download=True, size=args.image_size)
    labels = np.array(train.labels).reshape(-1)
    budgets = args.budgets or [len(labels)]

    val_loader = DataLoader(val, batch_size=64, num_workers=2)
    test_loader = DataLoader(test, batch_size=64, num_workers=2)

    print(f"{args.dataset} at {args.image_size}px, "
          f"{'from scratch' if args.scratch_clf else 'ImageNet-pretrained'}, "
          f"full train split = {len(labels)} images")
    print("budget, seed, test_auc, test_acc, test_qwk")

    for budget in budgets:
        rows = []
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            if budget >= len(labels):
                idx = np.arange(len(labels))
            else:                                   # stratified subset, as in the study
                idx = []
                for k in range(n_classes):
                    c = np.flatnonzero(labels == k)
                    take = max(1, round(budget * len(c) / len(labels)))
                    idx.extend(rng.choice(c, size=min(take, len(c)), replace=False).tolist())
                idx = np.array(sorted(idx))
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = build_model(n_classes, args.image_size, not args.scratch_clf, device)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            crit = nn.CrossEntropyLoss()
            loader = DataLoader(Subset(train, idx), batch_size=args.batch_size,
                                shuffle=True, num_workers=2, drop_last=len(idx) % args.batch_size == 1)
            best_auc, best_state = -1.0, None
            for _ in range(args.epochs):
                model.train()
                for x, y in loader:
                    x, y = x.to(device), torch.as_tensor(np.asarray(y).reshape(-1)).to(device)
                    opt.zero_grad()
                    loss = crit(model(x), y)
                    loss.backward()
                    opt.step()
                vauc, _, _ = evaluate(model, val_loader, device, n_classes)
                if vauc > best_auc:
                    best_auc = vauc
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(best_state)
            auc, acc, qwk = evaluate(model, test_loader, device, n_classes)
            print(f"{len(idx)}, {seed}, {auc:.4f}, {acc:.4f}, {qwk:.4f}")
            rows.append((auc, acc, qwk))
        a = np.array(rows)
        print(f"MEAN budget={budget}: AUC {100*a[:,0].mean():.1f} "
              f"ACC {100*a[:,1].mean():.1f} QWK {100*a[:,2].mean():.1f} "
              f"(n={len(rows)})")


if __name__ == "__main__":
    main()
