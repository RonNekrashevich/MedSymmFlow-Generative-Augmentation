"""Single source of truth for the generator/classifier disjoint split.

The PneumoniaMNIST train split (4708 images) is divided ONCE, stratified by class
and seeded independently of the experiment seeds, into a generator set and a
classifier pool. train_msf_scratch.py trains the MSF generator on the generator
set; augmentation.py restricts every classifier arm to the pool. Both import this
function, so the two sides can never overlap or drift apart.
"""
import hashlib

import numpy as np


def gen_clf_split(labels, gen_frac=0.5, split_seed=0):
    """Return (gen_idx, clf_idx): disjoint, stratified, sorted index lists."""
    labels = np.asarray(labels).reshape(-1)
    if not 0.0 < gen_frac < 1.0:
        raise ValueError(f"gen_frac must be in (0, 1), got {gen_frac}")
    rng = np.random.default_rng(split_seed)
    gen_idx = []
    for c in np.unique(labels):
        c_idx = np.where(labels == c)[0]
        take = int(round(len(c_idx) * gen_frac))
        gen_idx.extend(rng.choice(c_idx, size=take, replace=False).tolist())
    gen_idx = sorted(int(i) for i in gen_idx)
    clf_idx = sorted(set(range(len(labels))) - set(gen_idx))
    assert not set(gen_idx) & set(clf_idx)
    assert len(gen_idx) + len(clf_idx) == len(labels)
    return gen_idx, clf_idx


def split_fingerprint(idx):
    """Short hash of an index list, for manifests and cross-checks."""
    return hashlib.sha1(",".join(str(i) for i in idx).encode()).hexdigest()[:12]
