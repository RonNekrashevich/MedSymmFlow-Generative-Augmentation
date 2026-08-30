"""Paired significance testing for the augmentation sweep.

Deliberately dependency-light (numpy / pandas / scipy only, all preinstalled on
Colab) so it can be run standalone against a saved results.csv without importing
torch, medmnist or the rest of the training stack -- i.e. no retraining and no
GPU session required:

    import sys; sys.path.insert(0, "/content/MedSymmFlow/project")
    from paired_stats import paired_tests_from_csv
    paired_tests_from_csv("/content/drive/MyDrive/MedSymmFlow_Project/results.csv")
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def paired_tests_from_csv(results_csv, alpha=0.05, baselines=("B0", "B1", "B2"),
                          synthetic=("S1", "S2", "S3"), select=None,
                          on_duplicates="error"):
    """Paired per-seed test of each synthetic arm vs the strongest baseline, with
    Benjamini-Hochberg correction across the sweep.

    Pairing by seed removes the shared subsample/initialisation variance that makes
    the independent-CI check over-conservative.

    Note: with n seeds the two-sided Wilcoxon p-value cannot go below 2/2**n
    (0.0625 at n=5), so the paired t-test is the operative statistic at 5 seeds.
    """
    df = pd.read_csv(results_csv) if isinstance(results_csv, (str, Path)) else results_csv
    if select:
        for col, val in select.items():
            if col in df.columns:
                df = df[df[col].fillna("").astype(str) == str(val)]

    # A results.csv that accumulates across runs can hold the same (arm, budget, seed)
    # under different filter configurations. pivot_table would silently average them and
    # then t-test the average, so refuse by default.
    key = [c for c in ("arm", "budget", "seed") if c in df.columns]
    dup = df[df.duplicated(key, keep=False)] if key else df.iloc[:0]
    if len(dup):
        if on_duplicates == "error":
            where = dup.groupby(key).size().head(10)
            raise ValueError(
                f"{len(dup)} duplicate (arm,budget,seed) rows -- these come from different "
                f"runs/filter configs and must not be averaged.\n{where}\n"
                "Pass select={'filter_key': ...} to disambiguate, or on_duplicates='last'.")
        if on_duplicates == "last":
            df = df.drop_duplicates(key, keep="last")
        elif on_duplicates != "mean":
            raise ValueError("on_duplicates must be 'error', 'last' or 'mean'")

    BASE, SYN = list(baselines), list(synthetic)
    rows = []
    for budget, g in df[df["budget"] > 0].groupby("budget"):
        wide = g.pivot_table(index="seed", columns="arm", values="test_auc")
        avail = [b for b in BASE if b in wide.columns]
        if not avail:
            continue
        best_base = wide[avail].mean().idxmax()
        for s in SYN:
            if s not in wide.columns:
                continue
            pair = wide[[s, best_base]].dropna()
            if len(pair) < 2:
                continue
            a, b = pair[s].values, pair[best_base].values
            d = a - b
            t_p = float(stats.ttest_rel(a, b).pvalue)
            try:
                w_p = float(stats.wilcoxon(a, b).pvalue)
            except ValueError:      # all differences zero / too few pairs
                w_p = np.nan
            sd = d.std(ddof=1)
            dz = float(d.mean() / sd) if sd > 0 else np.nan
            rows.append({"budget": budget, "arm": s, "vs": best_base,
                         "n_seeds": len(pair), "mean_diff": round(float(d.mean()), 4),
                         "wins": f"{int((d > 0).sum())}/{len(d)}",
                         "cohen_dz": None if np.isnan(dz) else round(dz, 2),
                         "paired_t_p": round(t_p, 4),
                         "wilcoxon_p": None if np.isnan(w_p) else round(w_p, 4)})

    out = pd.DataFrame(rows)
    if out.empty:
        print("No paired comparisons available (need >=2 seeds per arm).")
        return out

    # Benjamini-Hochberg step-up q-values over the whole family of tests.
    p = out["paired_t_p"].values
    m = len(p)
    order = np.argsort(p)
    q_sorted = np.minimum.accumulate((p[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    q = np.empty(m)
    q[order] = np.clip(q_sorted, 0, 1)
    out["q_value_BH"] = np.round(q, 4)
    out["significant"] = (out["q_value_BH"] < alpha) & (out["mean_diff"] > 0)

    print(f"Paired t-test vs strongest baseline, Benjamini-Hochberg at alpha={alpha} "
          f"({m} comparisons). 'significant' requires a positive effect.")
    if out["n_seeds"].max() <= 5:
        print("Note: at <=5 seeds the two-sided Wilcoxon floor is 0.0625, so it cannot "
              "reach alpha=0.05; read paired_t_p / q_value_BH, and treat 'wins' as "
              "supporting evidence.")
    return out.sort_values(["budget", "arm"]).reset_index(drop=True)
