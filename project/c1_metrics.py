"""Recompute reverse-flow classification metrics with BOTH conventions.

The published MedSymmFlow code scores AUC per class as
    roc_auc_score(gt != i, distance_to_class_i)
averaged over classes, and quantifies uncertainty with the distance to the
PREDICTED class (large distance = uncertain). Our harness instead softmaxes
the negative distances and calls sklearn's macro one-vs-rest AUC, and uses
the margin between the two nearest class codes as confidence.

This script reports both so the comparison against published numbers is
like for like, and so the difference between the two confidence measures
can be stated rather than assumed.

    python project/c1_metrics.py --glob "/storage/medsymm/runs/c1eval/*.csv"
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def coverage_acc(correct, conf, frac, higher_is_better=True):
    order = np.argsort(-conf if higher_is_better else conf)
    k = max(1, int(len(order) * frac))
    return float(correct[order[:k]].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", nargs="+", required=True)
    args = ap.parse_args()

    files = sorted({f for pat in args.glob for f in glob.glob(pat)})
    print("file, n, ACC, AUC_paper, AUC_ours, acc@20_margin, acc@20_paperdist")
    for f in files:
        d = pd.read_csv(f)
        cols = [c for c in d.columns if c.startswith("msf_negdist")]
        if not cols or "true" not in d.columns:
            continue
        nd = d[cols].values
        dist = -nd
        y = d["true"].values.astype(int)
        K = dist.shape[1]
        present = np.unique(y)
        if len(present) < 2:
            continue
        auc_paper = float(np.mean([roc_auc_score(y != i, dist[:, i])
                                   for i in range(K) if 0 < (y == i).sum() < len(y)]))
        e = np.exp(nd)
        p = e / e.sum(axis=1, keepdims=True)
        if K > 2 and len(present) == K:
            auc_ours = float(roc_auc_score(y, p, multi_class="ovr", average="macro"))
        else:
            auc_ours = float(roc_auc_score(y, p[:, 1]))
        pred = dist.argmin(axis=1)
        correct = (pred == y)
        s = np.sort(dist, axis=1)
        margin = s[:, 1] - s[:, 0]
        dmin = dist[np.arange(len(y)), pred]
        print(f"{os.path.basename(f)}, {len(y)}, {correct.mean():.4f}, "
              f"{auc_paper:.4f}, {auc_ours:.4f}, "
              f"{coverage_acc(correct, margin, 0.2, True):.4f}, "
              f"{coverage_acc(correct, dmin, 0.2, False):.4f}")


if __name__ == "__main__":
    main()
