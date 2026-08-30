"""Deterministic, cacheable filtering of the synthetic pool.

Why this module exists
----------------------
`augmentation.filter_synthetic()` used to pick its confidence-filter scorer as
`baseline_models[(max(cfg.budgets), cfg.seeds[0])]`, and filtering results were never
cached. Two sweeps with different `budgets` lists therefore filtered the *same*
generated images with a *differently trained* scorer, silently producing different
surviving subsets. Observed: the S2 arm at budget 500 moved +0.0018 -> +0.0131 between
two runs -- larger than the effect being claimed.

The fix has two halves:

1. **Scorer identity must depend only on the filter configuration**, never on the
   sweep's budget list (`resolve_scorer` below; "full" means `len(train_set)`, NOT
   `max(budgets)`).
2. **Expensive GPU work is cached as continuous scores** (embeddings, probabilities)
   while the keep/drop decision is cheap pure arithmetic (`derive_keep`). Threshold and
   mode ablations then cost zero GPU and are exactly reproducible across GPU types --
   which matters, because fp16 autocast on a T4 vs an A100 can flip a borderline
   confidence across 0.60.

Everything here is numpy/pandas only (no torch), so it can be unit-tested on CPU.
"""
import hashlib
import json

import numpy as np

# Bump on ANY semantic change to the keep/scoring maths, so stale caches are not reused.
FILTER_VERSION = 3

FILTER_MODES = ("none", "keep_confident", "keep_uncertain", "random_match")
SCORER_CHOICES = ("none", "local", "full")
MEM_REFERENCES = ("none", "local", "full")
MEM_MODES = ("quantile", "absolute")


# --------------------------------------------------------------------------- keys
def _sha1_json(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def pool_hash(image_paths, labels, gen_manifest) -> str:
    """Identity of a generated pool: its contents AND how it was sampled.

    Including the generation manifest means re-generating with a different beta,
    solver or step size automatically invalidates every downstream filter cache.
    """
    items = sorted((str(p).replace("\\", "/").split("/")[-1], int(l))
                   for p, l in zip(image_paths, labels))
    return _sha1_json({"items": items, "gen": gen_manifest})[:16]


def scorer_manifest(*, dataset, arch, pool, scorer, scorer_budget, scorer_seed,
                    pretrained, epochs, lr, batch_size, image_size, use_amp):
    return {"v": FILTER_VERSION, "dataset": dataset, "arch": arch, "pool": pool,
            "scorer": scorer, "scorer_budget": scorer_budget, "scorer_seed": scorer_seed,
            "pretrained": pretrained, "epochs": epochs, "lr": lr,
            "batch_size": batch_size, "image_size": image_size, "amp": use_amp}


def filter_manifest(scorer_man, *, mode, conf_thresh, require_correct, mem_reference,
                    mem_mode, mem_quantile, mem_thresh, embed_id, random_seed,
                    mem_budget=None, mem_seed=None):
    return dict(scorer_man, mode=mode, conf_thresh=conf_thresh,
                require_correct=require_correct, mem_reference=mem_reference,
                mem_mode=mem_mode, mem_quantile=mem_quantile, mem_thresh=mem_thresh,
                embed_id=embed_id, random_seed=random_seed,
                mem_budget=mem_budget, mem_seed=mem_seed)


def _slug(manifest) -> str:
    sc = manifest["scorer"]
    sc_part = sc if sc == "none" else f"{sc}-b{manifest['scorer_budget']}-s{manifest['scorer_seed']}"
    mem = manifest.get("mem_reference", "none")
    mem_part = f"mem{mem}" + ("" if mem == "none" else f"-q{manifest.get('mem_quantile')}")
    return (f"{manifest['dataset']}_{manifest.get('mode', 'scorer')}_{sc_part}"
            f"_c{manifest.get('conf_thresh')}_{mem_part}_{manifest['arch']}_v{FILTER_VERSION}")


def make_key(manifest) -> str:
    """Readable slug for humans + hash for correctness. The hash is what guarantees
    'cached keyed by its actual configuration'."""
    return f"{_slug(manifest)}_{_sha1_json(manifest)[:8]}"


def resolve_scorer(scorer, *, budget, seed, train_size, seeds0,
                   scorer_budget=None, scorer_seed=None):
    """Map a scorer policy to a concrete (budget, seed) -- the budget-independence fix.

    local : the arm's own model. Realistic for the "institution with `budget` labels"
            framing, and never reads the sweep's budget list.
    full  : a model trained on the whole training split. `train_size`, NOT
            `max(cfg.budgets)` -- that substitution was the original bug.
    none  : no scorer; only the memorisation screen applies.
    """
    if scorer not in SCORER_CHOICES:
        raise ValueError(f"scorer must be one of {SCORER_CHOICES}, got {scorer!r}")
    if scorer == "none":
        return None
    if scorer_budget is not None:
        return int(scorer_budget), int(scorer_seed if scorer_seed is not None else seed)
    if scorer == "local":
        return int(budget), int(seed)
    return int(train_size), int(scorer_seed if scorer_seed is not None else seeds0)


# ------------------------------------------------------------------- keep decision
def memorisation_mask(nn_dist, *, mem_mode="quantile", mem_quantile=0.015, mem_thresh=None):
    """Drop synthetic images that are near-copies of a real training image.

    Note `mem_mode="quantile"` is a fixed-FRACTION discard: it always removes
    `mem_quantile` of the pool whatever the distances look like. `"absolute"` applies a
    real distance threshold and can legitimately drop nothing.
    """
    nn_dist = np.asarray(nn_dist, dtype=float)
    if mem_mode == "absolute":
        if mem_thresh is None:
            raise ValueError("mem_mode='absolute' requires mem_thresh")
        return nn_dist > float(mem_thresh)
    if mem_mode != "quantile":
        raise ValueError(f"mem_mode must be one of {MEM_MODES}")
    return nn_dist > float(np.quantile(nn_dist, mem_quantile))


def confidence_mask(probs, labels, *, conf_thresh=0.60, require_correct=True):
    """The 'keep_confident' rule: scorer agrees with the requested label, confidently."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels).reshape(-1).astype(int)
    n, k = probs.shape
    # Binary: keep the historical `>= 0.5` rule so the exact-0.5 tie-break is preserved
    # (argmax would resolve a 0.5/0.5 tie to class 0, `>= 0.5` to class 1).
    pred = (probs[:, 1] >= 0.5).astype(int) if k == 2 else probs.argmax(1)
    p_true = probs[np.arange(n), labels]
    keep = p_true >= float(conf_thresh)
    if require_correct:
        keep &= (pred == labels)
    return keep


def derive_keep(nn_dist, probs, labels, *, mode="keep_confident", conf_thresh=0.60,
                require_correct=True, mem_mode="quantile", mem_quantile=0.015,
                mem_thresh=None, random_seed=12345):
    """Full keep-mask. Pure function of cached scores -> exactly reproducible.

    Modes form a clean partition so the ablation is interpretable:
      keep_confident  the published practice
      keep_uncertain  its EXACT complement -- boundary-adjacent samples. Decision-boundary
                      theory says these are the informative ones, i.e. the ones the
                      standard filter throws away.
      random_match    drops the same NUMBER at random. The control that separates
                      "which images survive" from "how many survive".
    """
    if mode not in FILTER_MODES:
        raise ValueError(f"mode must be one of {FILTER_MODES}, got {mode!r}")
    n = len(labels)
    keep_mem = (np.ones(n, bool) if nn_dist is None else
                memorisation_mask(nn_dist, mem_mode=mem_mode, mem_quantile=mem_quantile,
                                  mem_thresh=mem_thresh))

    if mode == "none":
        return keep_mem, keep_mem, np.ones(n, bool)
    if probs is None:
        raise ValueError(f"mode={mode!r} needs scorer probabilities (filter_scorer='none' gives none)")

    conf = confidence_mask(probs, labels, conf_thresh=conf_thresh,
                           require_correct=require_correct)
    if mode == "keep_confident":
        keep_score = conf
    elif mode == "keep_uncertain":
        keep_score = ~conf
    else:  # random_match -- same drop count as keep_confident, chosen at random
        n_drop = int((~conf).sum())
        keep_score = np.ones(n, bool)
        if n_drop:
            rng = np.random.default_rng(random_seed)
            keep_score[rng.choice(n, size=n_drop, replace=False)] = False

    return keep_mem & keep_score, keep_mem, keep_score


def summarise_keep(keep, keep_mem, keep_score, labels, n_classes=2):
    labels = np.asarray(labels).reshape(-1).astype(int)
    return {"n_pool": int(len(keep)), "n_kept": int(keep.sum()),
            "mem_dropped": int((~keep_mem).sum()),
            "score_dropped": int((~keep_score).sum()),
            "kept_per_class": [int(((labels == c) & keep).sum()) for c in range(n_classes)]}


# ------------------------------------------------------------------------ selftest
def _selftest():
    rng = np.random.default_rng(0)
    n = 400
    labels = rng.integers(0, 2, n)
    p1 = rng.random(n)
    probs = np.stack([1 - p1, p1], 1)
    nn_dist = rng.random(n) * 0.1

    keep, km, ks = derive_keep(nn_dist, probs, labels)
    conf = confidence_mask(probs, labels)
    assert (ks == conf).all()
    assert (keep == (km & conf)).all()

    # keep_uncertain is the exact complement of keep_confident, on the score axis
    _, _, ks_u = derive_keep(nn_dist, probs, labels, mode="keep_uncertain")
    assert (ks_u == ~conf).all()
    assert (ks | ks_u).all() and not (ks & ks_u).any()

    # random_match drops the same COUNT, a different SET
    _, _, ks_r = derive_keep(nn_dist, probs, labels, mode="random_match")
    assert (~ks_r).sum() == (~conf).sum()
    assert not (ks_r == conf).all()

    # mode="none" = no CONFIDENCE filter, and needs no scorer -- but the memorisation
    # screen still applies, so this is not "keep everything".
    keep_n, km_n, ks_n = derive_keep(nn_dist, None, labels, mode="none")
    assert (keep_n == km_n).all() and ks_n.all()
    assert not km_n.all(), "quantile screen should still drop the closest samples"
    # ...unless memorisation is disabled too, which is the only all-keep configuration
    keep_all, _, _ = derive_keep(None, None, labels, mode="none")
    assert keep_all.all()

    # quantile screen drops exactly the requested fraction
    km2 = memorisation_mask(nn_dist, mem_quantile=0.015)
    assert (~km2).sum() == int(round(0.015 * n)) or abs((~km2).sum() - 0.015 * n) <= 1

    # binary tie-break: p == 0.5 must count as class 1 (historical `>= 0.5`)
    tie = np.array([[0.5, 0.5]])
    assert confidence_mask(tie, [1], conf_thresh=0.5, require_correct=True)[0]
    assert not confidence_mask(tie, [0], conf_thresh=0.5, require_correct=True)[0]

    # THE BUG, tested directly: scorer identity must not depend on the budget list
    a = resolve_scorer("full", budget=500, seed=0, train_size=4708, seeds0=0)
    b = resolve_scorer("full", budget=500, seed=0, train_size=4708, seeds0=0)
    assert a == b == (4708, 0), a
    assert resolve_scorer("local", budget=500, seed=3, train_size=4708, seeds0=0) == (500, 3)
    assert resolve_scorer("none", budget=500, seed=0, train_size=4708, seeds0=0) is None

    # keys are stable and configuration-sensitive
    man = scorer_manifest(dataset="pneumoniamnist", arch="resnet18", pool="abc",
                          scorer="local", scorer_budget=500, scorer_seed=0,
                          pretrained=True, epochs=15, lr=1e-4, batch_size=64,
                          image_size=28, use_amp=True)
    f1 = filter_manifest(man, mode="keep_confident", conf_thresh=0.60, require_correct=True,
                         mem_reference="local", mem_mode="quantile", mem_quantile=0.015,
                         mem_thresh=None, embed_id="imagenet_resnet18_224", random_seed=1)
    f2 = dict(f1, conf_thresh=0.70)
    assert make_key(f1) == make_key(dict(f1))
    assert make_key(f1) != make_key(f2)
    print("filter_key:", make_key(f1))
    print("filtering.py selftest OK")


if __name__ == "__main__":
    _selftest()
