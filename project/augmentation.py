"""PneumoniaMNIST + MedSymmFlow synthetic-augmentation experiment.

Imported by notebooks/pneumoniamnist_augmentation.ipynb so that logic fixes land
via `git pull` instead of a notebook re-upload. The notebook holds only config,
narrative, and calls; all behaviour lives here.

Protocol: pneumoniamnist_augmentation_protocol v2.0.
Arms: B0/B1/B2 (baselines), S1/S2/S3 (synthetic), C1 (MSF reference).
"""
import hashlib
import io
import json
import os
import shutil
import subprocess
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset, WeightedRandomSampler
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, f1_score
from scipy import stats
from sklearn.metrics import cohen_kappa_score
import medmnist
from medmnist import PneumoniaMNIST

# Dependency-light stats module, also runnable standalone against results.csv.
from paired_stats import paired_tests_from_csv
# Deterministic filtering: pure-numpy keep-mask maths + cache keys (no torch).
import filtering as flt
# Shared generator/classifier disjoint split (also used by train_msf_scratch.py).
from data_split import gen_clf_split, split_fingerprint
from datasets_meta import dataset_meta

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class Config:
    """Experiment knobs. `quick=True` is a fast smoke test; False is the real run."""

    def __init__(self, quick=True, save_dir="/content/drive/MyDrive/MedSymmFlow_Project",
                 medsymm_root="/content/MedSymmFlow", image_size=28, gen_image_size=32,
                 use_amp=True, budgets=None, seeds=None, epochs=None, syn_per_class=None,
                 scratch_dir="/content", fig_dir=None, weights_root=None,
                 dataset="pneumoniamnist", **overrides):
        self.quick = quick
        self.dataset = dataset
        meta = dataset_meta(dataset)
        self.channels = meta["channels"]
        self.class_names = list(meta["class_names"])
        self.save_dir = Path(save_dir)
        self.medsymm_root = medsymm_root
        self.scratch_dir = scratch_dir        # ephemeral temp (Colab: /content; cluster: PVC)
        self.fig_dir = fig_dir                # if set, figures are also saved here (headless batch)
        self.image_size = image_size          # classifier / real-data resolution
        self.gen_image_size = gen_image_size  # MSF RGB_28 checkpoint trains at 32
        self.use_amp = use_amp
        # Where the 755 MB Zenodo archive is downloaded and unpacked. On a cluster this
        # must point at persistent storage, otherwise every job re-downloads it.
        self.weights_root = str(weights_root) if weights_root else medsymm_root
        self.checkpoint_path = (
            f"{self.weights_root}/models_extracted/models/SymmetricalFlowMatchingClass/"
            f"RGB_28/FM_{dataset}_beta4.0_rgb.pt"
        )
        if quick:
            self.budgets, self.seeds, self.epochs, self.syn_per_class = [500], [0], 5, 200
        else:
            self.budgets, self.seeds, self.epochs, self.syn_per_class = [500, 4708], [0, 1, 2], 15, 1000
        # explicit overrides win
        if budgets is not None: self.budgets = budgets
        if seeds is not None: self.seeds = seeds
        if epochs is not None: self.epochs = epochs
        if syn_per_class is not None: self.syn_per_class = syn_per_class
        self.gen_beta = 4.0        # paper default for PneumoniaMNIST MSF
        self.gen_step_size = 0.04  # euler, ~25 steps
        self.gen_solver = "euler"
        self.gen_chunk = 200       # images per class per subprocess call (reduce if OOM)
        # published MSF (28px) test reference for this dataset (paper Table 2)
        self.c1_auc, self.c1_acc = meta["c1_auc"], meta["c1_acc"]

        # ---- training knobs (were hardcoded; needed for arch/resolution sweeps later)
        self.arch = "resnet18"
        self.mixup_alpha = 0.2     # B3: the modern-augmentation baseline synthetic must beat
        self.pretrained = True
        self.clf_stem = "small"           # "small": 3x3/stride-1 stem at <=64px;
                                          # "standard": keep torchvision's 7x7/stride-2
                                          # stem at every size (MedMNIST protocol)
        self.heavy_aug_baseline = False   # B4: real data + TrivialAugmentWide
        self.batch_size = 64
        self.lr = 1e-4
        self.lr_finetune = 1e-5    # S1's fine-tune stage (was a literal 1e-5)
        self.n_classes = meta["n_classes"]

        # ---- filtering (see filtering.py). Defaults are the scientifically defensible
        # ones: the filter may only use labels the hypothetical institution owns, so no
        # budget-N information leaks into a budget-500 arm.
        self.filter_mode = "keep_confident"     # none|keep_confident|keep_uncertain|random_match|self_consistent
        self.self_filter_q = 0.5                # self_consistent: per-class top fraction kept
        self.fixed_total = None                 # >0: extra arm that tops the real
                                                # budget up to this many images with
                                                # synthetic ones (size-matched study)
        self.self_filter_by = "margin"          # margin | dmin | dtrue
                                                # margin: top-two gap (decisiveness)
                                                # dmin/dtrue: distance to the winning /
                                                # requested code (published convention,
                                                # small = confident)
        self.self_score_batch = None            # self_consistent: scoring batch size
                                                # (default: gen_chunk in latent mode, else 128)
        self.self_per_class = None              # self_consistent: keep top-N per class instead
                                                # (dose/balance-matched filtering from a big pool)
        self.filter_scorer = "local"            # none|local|full  (local = the arm's own model)
        self.filter_scorer_budget = None        # explicit override for ablations
        self.filter_scorer_seed = None
        self.conf_thresh = 0.60
        self.filter_require_correct = True
        self.mem_reference = "local"            # none|local|full
        self.mem_mode = "quantile"              # quantile|absolute
        self.mem_quantile = 0.015
        self.mem_thresh = None
        self.embed_id = "imagenet_resnet18_224"  # memorisation encoder; independent of arch
        self.filter_random_seed = 12345
        self.s1_pretrain_filter = "mem_only"    # S1 pretrains on the scorer-free pool so it
                                                # stays shared across budgets (see protocol)
        self.legacy_filter = False              # reproduce the old (leaky) semantics for audit
        self.fingerprint_budget = None          # None => max(budgets)

        # ---- run bookkeeping
        self.resume = True
        self.run_tag = ""

        # ---- disjoint generator/classifier data. gen_frac=None keeps the published
        # pretrained checkpoint and the full train pool (old behaviour). A fraction in
        # (0,1) means: the MSF generator is trained FROM SCRATCH (train_msf_scratch.py)
        # on that stratified share of the train split, and every classifier arm draws
        # only from the complement -- generator and ResNet share no real image.
        self.gen_frac = None
        self.split_seed = 0
        # Generator-training knobs, all keyed into the scratch checkpoint name so
        # different recipes never share a cached checkpoint (and a 20-epoch smoke
        # never masquerades as the real generator).
        self.gen_epochs = 600
        self.gen_lr = 1e-3
        self.gen_dropout = 0.0
        self.gen_balance = False          # 50/50 sampling of the 26%-normal gen half
        self.gen_pretrain_epochs = 0      # >0: ChestMNIST pretrain stage first
        self.gen_mask_code = "rgb"        # rgb | onehot | thermometer (Phase B)
        self.gen_cfg_drop = 0.0           # >0: train generator with code dropout
        self.gen_cfg_w = 0.0              # >0: classifier-free guidance at sampling
        self.gen_latent = False           # LatMSF: flow in SD-VAE latent space
                                          # (use gen_image_size=256 -> 32x32 latents)
        self.gen_model_channels = 64      # UNet width (128 ~= 36M, paper LatMSF scale)
        self.gen_t_lognorm = False        # SD3 logit-normal timestep sampling
        self.gen_vae_id = None            # latent VAE: None = published SD-VAE;
                                          # e.g. REPA-E/e2e-sdvae-hf or a local dir
        self.gen_repa_weight = 0.0        # >0: DINOv2 mid-block alignment (U-REPA)
        self.gen_repa_teacher = None      # None = dinov2_vits14; retfound:<path> etc.
        # External-corpus generators (e.g. APTOS for retina):
        self.external_checkpoint = None   # use this checkpoint AS the generator; the
                                          # generator saw zero benchmark images, so the
                                          # classifier pool is the FULL train split
        self.gen_init_checkpoint = None   # disjoint mode: warm-start the scratch
                                          # training from this checkpoint (fine-tune)

        # Any remaining keyword sets an attribute directly, so every knob above is
        # reachable as Config(..., filter_mode="keep_uncertain", gen_beta=1.0).
        for key, value in overrides.items():
            if not hasattr(self, key):
                raise TypeError(f"Config got an unexpected keyword {key!r}")
            setattr(self, key, value)

        # Applied AFTER overrides so `Config(legacy_filter=True)` actually takes effect.
        if self.legacy_filter:
            self.filter_scorer = "full"
            self.filter_scorer_budget = max(self.budgets)
            self.mem_reference = "full"

        # After overrides for the same reason: gen_frac/split_seed/gen_beta may all be
        # overridden, and the scratch checkpoint is keyed by all three.
        assert not (self.external_checkpoint and self.gen_frac), \
            "external_checkpoint replaces the generator entirely; gen_frac would be unused"
        assert not (self.gen_init_checkpoint and self.gen_pretrain_epochs), \
            "gen_init_checkpoint and gen_pretrain_epochs both warm-start; pick one"
        if self.external_checkpoint:
            self.checkpoint_path = str(self.external_checkpoint)
        elif self.gen_frac:
            self.pretrain_checkpoint_path = (
                f"{self.weights_root}/scratch/pretrain_chestmnist"
                f"_e{self.gen_pretrain_epochs}_beta{self.gen_beta}_rgb.pt")
            size_tag = "" if self.gen_image_size == 32 else f"_sz{self.gen_image_size}"
            if self.gen_latent:
                size_tag += "_lat"
            code_tag = "" if self.gen_mask_code == "rgb" else f"_mc-{self.gen_mask_code}"
            drop_tag = "" if not self.gen_cfg_drop else f"_cd{self.gen_cfg_drop}"
            if self.gen_model_channels != 64:
                drop_tag += f"_ch{self.gen_model_channels}"
            if self.gen_t_lognorm:
                drop_tag += "_ln"
            if self.gen_vae_id:
                vae_hash = hashlib.sha1(str(self.gen_vae_id).encode()).hexdigest()[:8]
                drop_tag += f"_vae{vae_hash}"
            if self.gen_repa_weight:
                drop_tag += f"_repa{self.gen_repa_weight}"
                if self.gen_repa_teacher:
                    drop_tag += "_rt" + hashlib.sha1(
                        str(self.gen_repa_teacher).encode()).hexdigest()[:6]
            init_tag = ""
            if self.gen_init_checkpoint:
                init_hash = hashlib.sha1(
                    Path(self.gen_init_checkpoint).name.encode()).hexdigest()[:8]
                init_tag = f"_init{init_hash}"
            self.checkpoint_path = (
                f"{self.weights_root}/scratch/FM_{self.dataset}_scratch"
                f"_g{self.gen_frac}_ss{self.split_seed}_e{self.gen_epochs}"
                f"_lr{self.gen_lr}_do{self.gen_dropout}"
                f"_bal{int(self.gen_balance)}_pre{self.gen_pretrain_epochs}"
                f"{size_tag}{code_tag}{drop_tag}{init_tag}_beta{self.gen_beta}_rgb.pt")

    @property
    def run_dir(self) -> Path:
        """Single root for every artefact. Kept equal to save_dir for PneumoniaMNIST so
        the existing Drive layout (and its cached synthetic images) stays valid."""
        return self.save_dir


class PathDataset(Dataset):
    def __init__(self, paths, labels, tf, mode="L"):
        self.paths, self.labels, self.tf, self.mode = list(paths), list(labels), tf, mode

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert(self.mode)), self.labels[i]


class IntLabel(Dataset):
    """Normalise labels to plain ints so a real Subset and a PathDataset concat
    without a collate-time shape clash ((1,) array vs scalar)."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, y = self.ds[i]
        return x, int(np.asarray(y).reshape(-1)[0])


class Experiment:
    def __init__(self, cfg=None, **kw):
        self.cfg = cfg or Config(**kw)
        assert torch.cuda.is_available() or os.environ.get("MSF_ALLOW_CPU"), \
            "Enable a GPU runtime (or set MSF_ALLOW_CPU=1 for local smoke tests)"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cpu":
            self.cfg.use_amp = False
        c = self.cfg
        c.save_dir.mkdir(parents=True, exist_ok=True)
        self.synthetic_dir = c.run_dir / f"synthetic_{c.image_size}"
        self.filtered_dir = c.run_dir / f"synthetic_{c.image_size}_filtered"
        if c.image_size != 28 and not self.synthetic_dir.exists() \
                and (c.run_dir / "synthetic_28").exists():
            # Legacy layout: pre-rename runs (e.g. retina-64px) stored every pool
            # under "synthetic_28" regardless of size. Reuse it -- regenerating
            # would silently put new seeds on a different pool than the old ones,
            # and its metadata.csv paths point into the legacy dir.
            print("using legacy synthetic_28 pool dir for image_size", c.image_size)
            self.synthetic_dir = c.run_dir / "synthetic_28"
            self.filtered_dir = c.run_dir / "synthetic_28_filtered"
        self.results_path = c.run_dir / "results.csv"
        self.cache_dir = c.run_dir / "cache"        # embeddings + scorer probabilities
        self.models_dir = c.run_dir / "models"      # persisted B0 / S1-pretrain weights
        self.filters_dir = c.run_dir / "filters"    # one subdir per filter_key
        self.scratch = Path(c.scratch_dir)
        self.fig_dir = Path(c.fig_dir) if c.fig_dir else None
        for d in (self.synthetic_dir, self.filtered_dir, self.cache_dir, self.models_dir,
                  self.filters_dir, self.scratch, self.fig_dir):
            if d is not None:
                d.mkdir(parents=True, exist_ok=True)

        # Grayscale(3) lifts 1-channel datasets to the ResNet's 3 channels and is a
        # no-op-shaped identity risk for RGB ones, so it is inserted only when needed.
        to3 = ([transforms.Grayscale(num_output_channels=3)] if c.channels == 1 else [])
        self.img_mode = "L" if c.channels == 1 else "RGB"   # for PathDataset/PNGs
        self.train_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(c.image_size, scale=(0.8, 1.0)),
            *to3,
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.eval_tf = transforms.Compose([
            *to3,
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        # B4 (heavy classical augmentation): TrivialAugmentWide ahead of the
        # standard pipeline -- the strong free alternative to synthetic data.
        self.heavy_tf = transforms.Compose([
            transforms.TrivialAugmentWide(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(c.image_size, scale=(0.8, 1.0)),
            *to3,
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.embed_tf = transforms.Compose([
            *to3,
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.results = []
        self.baseline_models = {}
        self.gen_idx = None           # generator-half indices (disjoint mode only)
        self.clf_pool_idx = None      # classifier pool; None = whole train split
        self.synthetic_meta = None
        self.filtered = None          # scorer-free (mem-only) pool; S1 pretrains on this
        self._filter_cache = {}       # (budget, seed) -> filtered DataFrame
        self._emb_cache = {}
        self._pool_hash = None

        # Resume: the ledger is the source of truth, so a dead Colab session costs at
        # most the cell that was running.
        self.ledger = self._load_ledger()
        print("PyTorch:", torch.__version__, "| device:",
              torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU (smoke)")
        print("quick:", c.quick, "| budgets:", c.budgets, "| seeds:", c.seeds, "| epochs:", c.epochs)
        print(f"filter: mode={c.filter_mode} scorer={c.filter_scorer} mem={c.mem_reference}"
              + ("  [LEGACY - reproduces the old leaky semantics]" if c.legacy_filter else ""))
        if c.gen_frac:
            print(f"disjoint generator: gen_frac={c.gen_frac} split_seed={c.split_seed} "
                  f"-> from-scratch checkpoint {Path(c.checkpoint_path).name}")
        if len(self.ledger):
            print(f"ledger: {len(self.ledger)} existing rows at {self.results_path}"
                  + ("  (resume ON)" if c.resume else "  (resume OFF - will re-run)"))

    # ---------------------------------------------------------------- utilities
    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def loader(self, ds, batch_size=64, shuffle=False, sampler=None):
        # A trailing batch of exactly 1 crashes BatchNorm in training mode when the
        # feature map is 1x1 (standard stem at 28px). Drop it on training loaders
        # only; it carries no usable BN statistics anyway.
        training = shuffle or sampler is not None
        drop_last = training and len(ds) % batch_size == 1
        return DataLoader(ds, batch_size=batch_size, shuffle=(shuffle and sampler is None),
                          sampler=sampler, num_workers=2, pin_memory=True,
                          drop_last=drop_last)

    def _savefig(self, name):
        # Persist the current figure to fig_dir for headless/batch runs (no-op interactively).
        if self.fig_dir is not None:
            plt.savefig(self.fig_dir / f"{name}.png", dpi=120, bbox_inches="tight")

    # ------------------------------------------------------ ledger / resume (A2)
    LEDGER_KEY = ["dataset", "arch", "arm", "budget", "seed", "filter_key", "run_tag"]

    def _load_ledger(self):
        if self.results_path.exists():
            try:
                return pd.read_csv(self.results_path)
            except Exception as e:      # a truncated write should not kill the session
                print("WARNING: could not read ledger:", e)
        return pd.DataFrame()

    def _flush_ledger(self):
        """Write via scratch + move: Drive's FUSE mount corrupts partial in-place writes."""
        tmp = self.scratch / "results.tmp.csv"
        self.ledger.to_csv(tmp, index=False)
        shutil.move(str(tmp), str(self.results_path))

    def already_done(self, arm, budget, seed, filter_key=None):
        if not self.cfg.resume or not len(self.ledger):
            return False
        key = {"dataset": self.cfg.dataset, "arch": self.cfg.arch, "arm": arm,
               "budget": budget, "seed": seed,
               "filter_key": filter_key if filter_key is not None else "",
               "run_tag": self.cfg.run_tag}
        m = pd.Series(True, index=self.ledger.index)
        for col, val in key.items():
            if col not in self.ledger.columns:
                return False
            m &= self.ledger[col].fillna("").astype(str) == str(val)
        return bool(m.any())

    def _provenance(self, filter_key="", n_syn_used=0):
        c = self.cfg
        return {"dataset": c.dataset, "arch": c.arch, "filter_key": filter_key,
                "filter_mode": c.filter_mode, "filter_scorer": c.filter_scorer,
                "filter_scorer_budget": c.filter_scorer_budget,
                "n_syn_used": n_syn_used, "pool_hash": self._pool_hash or "",
                "gen_frac": c.gen_frac or 0, "split_seed": (c.split_seed if c.gen_frac else ""),
                "run_tag": c.run_tag, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

    # ------------------------------------------------------------ selftest (A1)
    @staticmethod
    def _sha1_state_dict(sd):
        h = hashlib.sha1()
        for k in sorted(sd):
            h.update(k.encode())
            buf = io.BytesIO()
            torch.save(sd[k].detach().cpu().contiguous(), buf)
            h.update(buf.getvalue())
        return h.hexdigest()

    def selftest_repro(self, fixtures_path=None, strict=True):
        """Prove the refactor is numerically inert.

        `train_classifier` seeds and then constructs the model, and Conv2d/Linear consume
        the global torch RNG at construction -- so any reordering of module creation
        shifts every result by an amount that looks exactly like a real effect. Compare
        against fixtures recorded from the pre-refactor code by record_fixtures.py.
        """
        path = Path(fixtures_path or (self.cfg.run_dir / "fixtures_prerefactor.json"))
        self.set_seed(0)
        got = {"model_init_sha1_seed0": self._sha1_state_dict(self.build_model().state_dict())}
        if self.cfg.gen_frac:
            print("selftest: subset fixtures skipped -- disjoint mode draws from the "
                  "classifier pool, so subsets differ from the full-pool fixtures by design.")
        else:
            for n in (250, 500, 1000):
                idx, _ = self.stratified_subset(n, 0)
                got[f"subset_{n}_seed0_sha1"] = hashlib.sha1(
                    ",".join(str(int(i)) for i in idx).encode()).hexdigest()
        flt._selftest()

        if not path.exists():
            print(f"selftest: no fixtures at {path} -- nothing to compare against.")
            print("          run project/record_fixtures.py on the PRE-refactor code first.")
            return got
        want = json.loads(path.read_text())
        bad = [k for k, v in got.items() if k in want and want[k] != v]
        for k in got:
            if k in want:
                print(f"  {'OK  ' if k not in bad else 'FAIL'} {k}")
        if bad and strict:
            raise AssertionError(
                "selftest_repro FAILED for " + ", ".join(bad) +
                " -- the refactor changed initialisation or subsampling. Results from "
                "before and after are NOT comparable.")
        if not bad:
            print("selftest_repro: refactor is numerically inert.")
        return got

    # -------------------------------------------------------------- data (G1)
    def setup_data(self):
        c = self.cfg
        meta = dataset_meta(c.dataset)
        ds_cls = getattr(medmnist, meta["medmnist_class"])
        self.train_set = ds_cls(split="train", transform=self.train_tf, download=True, size=c.image_size)
        if c.heavy_aug_baseline:
            self.train_set_heavy = ds_cls(split="train", transform=self.heavy_tf,
                                          download=True, size=c.image_size)
        self.val_set = ds_cls(split="val", transform=self.eval_tf, download=True, size=c.image_size)
        self.test_set = ds_cls(split="test", transform=self.eval_tf, download=True, size=c.image_size)
        n_tr, n_va, n_te = meta["splits"]
        assert len(self.train_set) == n_tr, len(self.train_set)
        assert len(self.val_set) == n_va, len(self.val_set)
        assert len(self.test_set) == n_te, len(self.test_set)
        self.train_labels_all = np.array(self.train_set.labels).reshape(-1)
        print(f"Split sizes OK: train {n_tr} / val {n_va} / test {n_te}")

        if c.gen_frac:
            self.gen_idx, self.clf_pool_idx = gen_clf_split(
                self.train_labels_all, c.gen_frac, c.split_seed)
            over = [b for b in c.budgets if b > len(self.clf_pool_idx)]
            assert not over, (
                f"budgets {over} exceed the classifier pool ({len(self.clf_pool_idx)}); "
                f"the other {len(self.gen_idx)} train images belong to the generator")
            print(f"Disjoint split (split_seed {c.split_seed}): "
                  f"generator {len(self.gen_idx)} [{split_fingerprint(self.gen_idx)}] / "
                  f"classifier pool {len(self.clf_pool_idx)} -- no shared images")

        rows = []
        parts = [("train", None), ("val", None), ("test", None)]
        if c.gen_frac:
            parts += [("train/generator", self.gen_idx), ("train/clf_pool", self.clf_pool_idx)]
        for name, idx in parts:
            if idx is None:
                ds = {"train": self.train_set, "val": self.val_set, "test": self.test_set}[name]
                y = np.array(ds.labels).reshape(-1)
            else:
                y = self.train_labels_all[np.asarray(idx)]
            counts = np.bincount(y, minlength=c.n_classes)
            row = {"split": name}
            row.update({c.class_names[k]: int(counts[k]) for k in range(c.n_classes)})
            if c.n_classes == 2:
                row["pneumonia_frac"] = round(counts[1] / counts.sum(), 3)
            rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------- model & training
    def build_model(self, num_classes=None, pretrained=None):
        """Build the classifier with the small-image stem adaptation.

        RNG-ORDER CRITICAL. `train_classifier` calls `set_seed(seed)` and then this, and
        every `nn.Conv2d`/`nn.Linear` consumes the global torch RNG at construction. The
        resnet18 branch must keep the exact sequence resnet18 -> conv1 -> maxpool -> fc,
        with nothing RNG-consuming inserted before, between or after it, or every result
        shifts by an amount indistinguishable from a real effect (see selftest_repro).
        """
        num_classes = self.cfg.n_classes if num_classes is None else num_classes
        pretrained = self.cfg.pretrained if pretrained is None else pretrained
        arch = self.cfg.arch
        if arch != "resnet18":
            raise ValueError(f"arch {arch!r} not available yet (Stage B adds the registry)")
        model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        if self.cfg.image_size <= 64 and self.cfg.clf_stem == "small":
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 28px stem
            model.maxpool = nn.Identity()
        # else: keep the standard 7x7/stride-2 stem, matching MedMNIST's ResNet-18 (224)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model.to(self.device)

    def build_resnet18(self, num_classes=2, pretrained=True):
        """Back-compat alias; delegates without touching the construction order."""
        return self.build_model(num_classes, pretrained)

    def run_epoch(self, model, loader, criterion, optimizer=None, scaler=None,
                  mixup_alpha=0.0):
        training = optimizer is not None
        model.train(training)
        losses = []
        for images, labels in loader:
            images = images.to(self.device)
            # reshape(-1), not squeeze(): a final batch of size 1 gives shape (1,1), and
            # squeeze() collapses it to 0-dim -> CrossEntropyLoss errors. Never fired at
            # 4708/524/624 but will as soon as a filtered set has size = 1 (mod batch).
            labels = labels.reshape(-1).long().to(self.device)
            do_mix = training and mixup_alpha > 0 and images.size(0) > 1
            if do_mix:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                perm = torch.randperm(images.size(0), device=images.device)
                images = lam * images + (1.0 - lam) * images[perm]
                labels_b = labels[perm]
            with torch.set_grad_enabled(training):
                with torch.autocast("cuda", enabled=self.cfg.use_amp):
                    logits = model(images)
                    if do_mix:
                        loss = (lam * criterion(logits, labels)
                                + (1.0 - lam) * criterion(logits, labels_b))
                    else:
                        loss = criterion(logits, labels)
                if training:
                    optimizer.zero_grad()
                    if scaler is not None:
                        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                    else:
                        loss.backward(); optimizer.step()
            losses.append(loss.item())
        return {"loss": float(np.mean(losses))}

    @torch.no_grad()
    def predict_probs(self, model, loader):
        model.eval()
        ys, ps = [], []
        for images, labels in loader:
            images = images.to(self.device)
            with torch.autocast("cuda", enabled=self.cfg.use_amp):
                logits = model(images)
            ps.append(torch.softmax(logits.float(), 1)[:, 1].cpu().numpy())
            ys.append(labels.reshape(-1).long().numpy())    # see run_epoch: squeeze() breaks n=1
        return np.concatenate(ys), np.concatenate(ps)

    @torch.no_grad()
    def predict_proba(self, model, loader):
        """Full (n, K) probability matrix -- what the confidence filter caches."""
        model.eval()
        ys, ps = [], []
        for images, labels in loader:
            images = images.to(self.device)
            with torch.autocast("cuda", enabled=self.cfg.use_amp):
                logits = model(images)
            ps.append(torch.softmax(logits.float(), 1).cpu().numpy())
            ys.append(labels.reshape(-1).long().numpy())
        return np.concatenate(ys), np.concatenate(ps, axis=0)

    def _val_auc(self, model):
        """Val-set AUC: binary uses the positive-prob path (bit-compatible with the
        pre-port code); K classes use macro one-vs-rest on the full matrix."""
        if self.cfg.n_classes == 2:
            y, p = self.predict_probs(model, self.loader(self.val_set))
            return roc_auc_score(y, p)
        y, p = self.predict_proba(model, self.loader(self.val_set))
        return roc_auc_score(y, p, multi_class="ovr", average="macro")

    def best_threshold_on_val(self, model):
        if self.cfg.n_classes != 2:
            return None                     # multi-class predicts by argmax
        y, p = self.predict_probs(model, self.loader(self.val_set))
        ts = np.linspace(0.05, 0.95, 19)
        j = [balanced_accuracy_score(y, (p >= t).astype(int)) for t in ts]
        return float(ts[int(np.argmax(j))])

    def evaluate_on_test(self, model, threshold):
        if self.cfg.n_classes == 2:
            y, p = self.predict_probs(model, self.loader(self.test_set))
            pred = (p >= threshold).astype(int)
            return {
                "test_auc": roc_auc_score(y, p),
                "test_acc": accuracy_score(y, pred),
                "test_balacc": balanced_accuracy_score(y, pred),
                "test_f1": f1_score(y, pred),
            }
        y, p = self.predict_proba(model, self.loader(self.test_set))
        pred = p.argmax(1)
        return {
            "test_auc": roc_auc_score(y, p, multi_class="ovr", average="macro"),
            "test_acc": accuracy_score(y, pred),
            "test_balacc": balanced_accuracy_score(y, pred),
            "test_f1": f1_score(y, pred, average="macro"),
            "test_qwk": cohen_kappa_score(y, pred, weights="quadratic"),
        }

    def class_weights_for(self, subset_labels):
        k = self.cfg.n_classes
        counts = np.bincount(subset_labels, minlength=k)
        w = counts.sum() / (float(k) * np.maximum(counts, 1))
        return torch.tensor(w, dtype=torch.float32, device=self.device)

    def train_classifier(self, train_ds, train_labels, seed, epochs=None, lr=None,
                         init_state=None, weighted=False, sampler=None, tag="",
                         mixup_alpha=0.0):
        epochs = epochs or self.cfg.epochs
        lr = self.cfg.lr if lr is None else lr
        self.set_seed(seed)
        model = self.build_model()
        if init_state is not None:
            model.load_state_dict(init_state)
        criterion = nn.CrossEntropyLoss(
            weight=self.class_weights_for(train_labels) if weighted else None)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=self.cfg.use_amp)
        loader = self.loader(train_ds, shuffle=True, sampler=sampler)
        best_auc, best_state = -1.0, None
        for _ in range(epochs):
            self.run_epoch(model, loader, criterion, optimizer, scaler,
                           mixup_alpha=mixup_alpha)
            val_auc = self._val_auc(model)
            if val_auc > best_auc:
                best_auc = val_auc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        print(f"  [{tag} seed {seed}] best val AUC {best_auc:.4f}")
        return model, best_auc

    def stratified_subset(self, n, seed):
        """Stratified draw of n indices from the classifier pool.

        The pool is the whole train split, or its classifier half in disjoint mode
        (cfg.gen_frac). Indices are always into the full train_set. With the full
        pool this is bit-identical to the pre-disjoint implementation (same rng
        stream), so the recorded fixtures stay valid.
        """
        pool = (np.arange(len(self.train_set)) if self.clf_pool_idx is None
                else np.asarray(self.clf_pool_idx))
        pool_labels = self.train_labels_all[pool]
        if n >= len(pool):
            return pool.tolist(), pool_labels
        rng = np.random.default_rng(seed)
        idx = []
        for c in range(self.cfg.n_classes):   # (0, 1) for binary: identical rng stream
            c_idx = pool[pool_labels == c]
            take = int(round(n * (len(c_idx) / len(pool))))
            idx.extend(rng.choice(c_idx, size=take, replace=False).tolist())
        idx = sorted(idx)
        return idx, self.train_labels_all[idx]

    # ------------------------------------------------------------- results api
    def add_result(self, arm, budget, seed, metrics, val_auc, filter_key="", n_syn_used=0):
        row = {"arm": arm, "budget": budget, "seed": seed, "val_auc": val_auc, **metrics,
               **self._provenance(filter_key, n_syn_used)}
        self.results.append(row)
        # Append + flush now, so a dead session costs at most the running cell.
        self.ledger = pd.concat([self.ledger, pd.DataFrame([row])], ignore_index=True)
        self._flush_ledger()
        print(f"  -> {arm} n={budget} seed={seed}: test AUC {metrics['test_auc']:.4f} "
              f"acc {metrics['test_acc']:.4f}")

    def run_supervised(self, arm, train_ds, train_labels, budget, seed,
                       filter_key="", n_syn_used=0, **kw):
        model, val_auc = self.train_classifier(train_ds, train_labels, seed,
                                               tag=f"{arm} n={budget}", **kw)
        self.add_result(arm, budget, seed,
                        self.evaluate_on_test(model, self.best_threshold_on_val(model)),
                        val_auc, filter_key=filter_key, n_syn_used=n_syn_used)
        return model

    # ------------------------------------------------- persisted baseline models
    def _baseline_path(self, budget, seed):
        return self.models_dir / f"B0_{self.cfg.arch}_b{budget}_s{seed}.pt"

    def baseline_model(self, budget, seed, train_if_missing=True):
        """B0 for (budget, seed), from memory -> disk -> freshly trained.

        Persisting matters twice over: a session that died after run_baselines used to
        make filter_synthetic() throw KeyError, and filter_scorer='local' needs one of
        these per arm."""
        if (budget, seed) in self.baseline_models:
            return self.baseline_models[(budget, seed)]
        path = self._baseline_path(budget, seed)
        if path.exists():
            model = self.build_model()
            model.load_state_dict(torch.load(path, map_location=self.device))
            model.eval()
            self.baseline_models[(budget, seed)] = model
            return model
        if not train_if_missing:
            return None
        idx, sub_labels = self.stratified_subset(budget, seed)
        model, _ = self.train_classifier(Subset(self.train_set, idx), sub_labels, seed,
                                         weighted=False, tag=f"B0* scorer n={budget}")
        self._persist_baseline(model, budget, seed)
        return model

    def _persist_baseline(self, model, budget, seed):
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
                   self._baseline_path(budget, seed))
        self.baseline_models[(budget, seed)] = model

    def oversampler(self, sub_labels):
        class_w = 1.0 / np.bincount(sub_labels, minlength=self.cfg.n_classes).clip(min=1)
        return WeightedRandomSampler(class_w[sub_labels], num_samples=len(sub_labels), replacement=True)

    # ---------------------------------------------------------- baselines (3)
    def run_baselines(self):
        for budget in self.cfg.budgets:
            for seed in self.cfg.seeds:
                idx, sub_labels = self.stratified_subset(budget, seed)
                sub = Subset(self.train_set, idx)

                if self.already_done("B0", budget, seed):
                    print(f"  [skip] B0 n={budget} seed={seed} (in ledger)")
                    self.baseline_model(budget, seed)          # ensure weights are loaded
                else:
                    model = self.run_supervised("B0", sub, sub_labels, budget, seed, weighted=False)
                    self._persist_baseline(model, budget, seed)

                for arm, kw in (("B1", dict(weighted=True)),
                                ("B2", dict(weighted=False, sampler=self.oversampler(sub_labels))),
                                ("B3", dict(weighted=False, mixup_alpha=self.cfg.mixup_alpha))):
                    if self.already_done(arm, budget, seed):
                        print(f"  [skip] {arm} n={budget} seed={seed} (in ledger)")
                        continue
                    self.run_supervised(arm, sub, sub_labels, budget, seed, **kw)

                if self.cfg.heavy_aug_baseline:
                    # B4: real data + TrivialAugmentWide -- the strong FREE
                    # alternative to synthetic augmentation. Same subset indices,
                    # so it pairs seed-wise with every other arm.
                    if self.already_done("B4", budget, seed):
                        print(f"  [skip] B4 n={budget} seed={seed} (in ledger)")
                    else:
                        self.run_supervised("B4", Subset(self.train_set_heavy, idx),
                                            sub_labels, budget, seed, weighted=False)
        if self.cfg.dataset == "pneumoniamnist":
            print("\nB0 reproduction target (protocol): ResNet-18 @28px AUC ~= 94.4")

    # -------------------------------------------------------- generation (MSF)
    def download_weights(self):
        """Fetch and unpack the published MedSymmFlow weights, once.

        Downloads into cfg.weights_root, which on a cluster should be persistent
        storage -- otherwise each job pays the 755 MB download again. Safe to call
        repeatedly; both steps are skipped if their output already exists.
        """
        c = self.cfg
        if c.external_checkpoint:
            assert os.path.exists(c.checkpoint_path), (
                f"external generator checkpoint missing: {c.checkpoint_path}\n"
                f"train it with project/train_msf_external.py first")
            print("External generator checkpoint present:", c.checkpoint_path)
            return
        if c.gen_frac:
            # Disjoint mode never touches the published weights: the checkpoint must
            # come from train_msf_scratch.py (run_experiment.py trains it if missing).
            assert os.path.exists(c.checkpoint_path), (
                f"scratch generator checkpoint missing: {c.checkpoint_path}\n"
                f"train it first:  python project/train_msf_scratch.py "
                f"--out {c.checkpoint_path} --gen-frac {c.gen_frac} "
                f"--split-seed {c.split_seed} --beta {c.gen_beta}")
            sidecar = Path(c.checkpoint_path).with_suffix(".json")
            if sidecar.exists():
                m = json.loads(sidecar.read_text())
                assert (m["gen_frac"], m["split_seed"]) == (c.gen_frac, c.split_seed), (
                    f"checkpoint was trained with gen_frac={m['gen_frac']} "
                    f"split_seed={m['split_seed']}, config wants "
                    f"{c.gen_frac}/{c.split_seed}")
            print("Scratch generator checkpoint present:", c.checkpoint_path)
            return
        root = self.cfg.weights_root
        os.makedirs(root, exist_ok=True)
        if os.path.exists(self.cfg.checkpoint_path):
            print("Checkpoint already present:", self.cfg.checkpoint_path)
            return
        if not os.path.exists(f"{root}/models.zip"):
            print(f"downloading MedSymmFlow weights (~755 MB) -> {root}")
            subprocess.run(["wget", "-q", "-O", "models.zip",
                            "https://zenodo.org/records/16086025/files/models.zip?download=1"],
                           cwd=root, check=True)
        if not os.path.exists(f"{root}/models_extracted"):
            subprocess.run(["unzip", "-q", "models.zip", "-d", "models_extracted"], cwd=root, check=True)
        assert os.path.exists(self.cfg.checkpoint_path), self.cfg.checkpoint_path
        print("Checkpoint present:", self.cfg.checkpoint_path)

    def generate_synthetic(self, base_seed=1000):
        c = self.cfg
        per_class, out_dir = c.syn_per_class, self.synthetic_dir
        meta_path = out_dir / "metadata.csv"
        if meta_path.exists():
            existing = pd.read_csv(meta_path)
            if all((existing["label"] == k).sum() >= per_class for k in range(c.n_classes)):
                print("Synthetic set already present:", len(existing), "images")
                self.synthetic_meta = existing
                return existing

        if c.dataset != "pneumoniamnist":
            # Generic path: generate_medmnist.py handles any class count, batches
            # internally, stores at image_size, infers the UNet from the checkpoint,
            # and writes a metadata.csv in exactly the schema used below.
            cmd = ["python", "project/generate_medmnist.py",
                   "--checkpoint", c.checkpoint_path,
                   "--dataset", c.dataset, "--n_classes", str(c.n_classes),
                   "--per_class", str(per_class), "--seed", str(base_seed),
                   "--beta", str(c.gen_beta), "--image_size", str(c.gen_image_size),
                   "--store_size", str(c.image_size), "--batch", str(c.gen_chunk),
                   "--rgb_mask", "--solver", c.gen_solver,
                   "--step_size", str(c.gen_step_size),
                   "--mask_code", c.gen_mask_code,
                   *(["--cfg_w", str(c.gen_cfg_w)] if c.gen_cfg_w else []),
                   *(["--latent"] if c.gen_latent else []),
                   *(["--vae_id", str(c.gen_vae_id)] if c.gen_vae_id else []),
                   "--output_dir", str(out_dir)]
            env = dict(os.environ, PYTHONPATH=f"{c.medsymm_root}/src")
            print(f"generating {per_class}/class x {c.n_classes} classes")
            res = subprocess.run(cmd, cwd=c.medsymm_root, env=env, capture_output=True, text=True)
            if res.returncode != 0:
                print(res.stdout[-3000:]); print(res.stderr[-3000:])
                raise RuntimeError("generation failed: see output above")
            meta = pd.read_csv(meta_path)
            print("Generated", len(meta), "synthetic images ->", out_dir)
            self.synthetic_meta = meta
            return meta

        for name in ("normal", "pneumonia"):
            (out_dir / name).mkdir(parents=True, exist_ok=True)

        rows, chunk_id = [], 0
        for start in range(0, per_class, c.gen_chunk):
            n = min(c.gen_chunk, per_class - start)
            tmp = self.scratch / f"_gen_chunk_{chunk_id}"
            if tmp.exists():
                shutil.rmtree(tmp)
            cmd = [
                "python", "project/generate_pneumoniamnist.py",
                "--checkpoint", c.checkpoint_path,
                "--dataset", "pneumoniamnist", "--n_classes", "2",
                "--num_normal", str(n), "--num_pneumonia", str(n),
                "--seed", str(base_seed + chunk_id),
                "--beta", str(c.gen_beta), "--image_size", str(c.gen_image_size),
                "--rgb_mask", "--solver", "euler", "--step_size", str(c.gen_step_size),
                "--output_dir", str(tmp),
                # Architecture of the published RGB_28 checkpoint (from its state-dict shapes).
                "--model_channels", "64", "--num_res_blocks", "2",
                "--channel_mult", "1", "2", "2", "2",
                "--num_heads", "4", "--num_head_channels", "64",
                "--attention_resolutions", "2",
            ]
            env = dict(os.environ, PYTHONPATH=f"{c.medsymm_root}/src")
            print("chunk", chunk_id, "->", n, "per class")
            res = subprocess.run(cmd, cwd=c.medsymm_root, env=env, capture_output=True, text=True)
            if res.returncode != 0:
                print(res.stdout[-3000:]); print(res.stderr[-3000:])
                err = [l for l in res.stderr.strip().splitlines()
                       if l.strip() and not l.startswith((" ", "\t", "Traceback"))]
                raise RuntimeError("generation failed: " + (err[-1] if err else "see output above"))
            for cls_name, label in [("normal", 0), ("pneumonia", 1)]:
                for png in sorted((tmp / cls_name).glob("*.png")):
                    dst = out_dir / cls_name / f"{cls_name}_{chunk_id:03d}_{png.stem}.png"
                    img = Image.open(png).convert("L")
                    if img.size != (c.image_size, c.image_size):
                        img = img.resize((c.image_size, c.image_size), Image.LANCZOS)
                    img.save(dst)
                    rows.append({"image_path": str(dst), "label": label,
                                 "class_name": cls_name, "gen_seed": base_seed + chunk_id})
            shutil.rmtree(tmp)
            chunk_id += 1

        meta = pd.DataFrame(rows)
        meta.to_csv(meta_path, index=False)
        print("Generated", len(meta), "synthetic images ->", out_dir)
        self.synthetic_meta = meta
        return meta

    def visualize_samples(self, per_class=8):
        names = self.cfg.class_names
        fig, axes = plt.subplots(len(names), per_class,
                                 figsize=(2 * per_class, 2 * len(names)))
        axes = np.atleast_2d(axes)
        for r, cls in enumerate(names):
            paths = self.synthetic_meta[self.synthetic_meta.class_name == cls]["image_path"].tolist()[:per_class]
            for a, pth in zip(axes[r], paths):
                a.imshow(Image.open(pth), cmap="gray"); a.set_title(cls, fontsize=8); a.axis("off")
        plt.tight_layout(); self._savefig("synthetic_samples"); plt.show()

    # --------------------------------------------------------- filtering (Sec 7)
    # Expensive GPU work is cached as CONTINUOUS scores (embeddings, probabilities); the
    # keep/drop decision is cheap pure arithmetic in filtering.derive_keep. Mode and
    # threshold ablations therefore cost zero GPU and are exactly reproducible -- fp16
    # autocast on a T4 vs an A100 can otherwise flip a borderline confidence across 0.60.

    def _gen_manifest(self):
        c = self.cfg
        return {"checkpoint": Path(c.checkpoint_path).name, "beta": c.gen_beta,
                "mask_code": c.gen_mask_code, "cfg_w": c.gen_cfg_w,
                "step_size": c.gen_step_size, "solver": c.gen_solver,
                "gen_image_size": c.gen_image_size, "image_size": c.image_size,
                "syn_per_class": c.syn_per_class}

    def pool_hash(self):
        if self._pool_hash is None:
            self._pool_hash = flt.pool_hash(self.synthetic_meta.image_path,
                                            self.synthetic_meta.label, self._gen_manifest())
        return self._pool_hash

    def _encoder(self):
        if "enc" not in self._emb_cache:
            enc = resnet18(weights=ResNet18_Weights.DEFAULT)   # fixed, independent of cfg.arch
            enc.fc = nn.Identity()
            self._emb_cache["enc"] = enc.to(self.device).eval()
        return self._emb_cache["enc"]

    @torch.no_grad()
    def _embed_paths(self, paths):
        enc = self._encoder()
        out = []
        for x, _ in self.loader(PathDataset(paths, [0] * len(paths), self.embed_tf,
                                            mode=self.img_mode), batch_size=128):
            out.append(nn.functional.normalize(enc(x.to(self.device)), dim=1).cpu())
        return torch.cat(out).numpy()

    def _real_train_paths(self):
        real_dir = self.scratch / f"_real_train_png_{self.cfg.dataset}"
        real_dir.mkdir(parents=True, exist_ok=True)
        ds_cls = getattr(medmnist, dataset_meta(self.cfg.dataset)["medmnist_class"])
        raw = ds_cls(split="train", download=True, size=self.cfg.image_size)
        paths = []
        for i in range(len(raw)):
            p = real_dir / f"r_{i:05d}.png"
            if not p.exists():
                raw[i][0].convert(self.img_mode).save(p)
            paths.append(str(p))
        return paths

    def _embed_real(self):
        path = self.cache_dir / f"emb_real_{self.cfg.image_size}_{self.cfg.embed_id}.npy"
        if "real" not in self._emb_cache:
            if path.exists():
                self._emb_cache["real"] = np.load(path)
            else:
                emb = self._embed_paths(self._real_train_paths())
                np.save(path, emb)
                self._emb_cache["real"] = emb
        return self._emb_cache["real"]

    def _embed_syn(self):
        path = self.cache_dir / f"emb_syn_{self.pool_hash()}_{self.cfg.embed_id}.npy"
        if "syn" not in self._emb_cache:
            if path.exists():
                self._emb_cache["syn"] = np.load(path)
            else:
                emb = self._embed_paths(self.synthetic_meta.image_path.tolist())
                np.save(path, emb)
                self._emb_cache["syn"] = emb
        return self._emb_cache["syn"]

    def _mem_distances(self, reference_idx=None):
        """Nearest-real-neighbour cosine distance per synthetic image.

        `reference_idx=None` compares against the whole train split; passing a budget's
        own indices keeps full-dataset information out of a low-budget arm (the second
        leak in the original implementation).
        """
        real = self._embed_real()
        if reference_idx is not None:
            real = real[np.asarray(reference_idx)]
        return 1.0 - (self._embed_syn() @ real.T).max(axis=1)

    def _scorer_probs(self, budget, seed):
        """Cached (n, K) probabilities from the resolved scorer model."""
        key = flt.make_key(self._scorer_manifest(budget, seed))
        path = self.cache_dir / f"probs_{key}.npy"
        if path.exists():
            return np.load(path)
        model = self.baseline_model(budget, seed)
        _, probs = self.predict_proba(model, self.loader(
            PathDataset(self.synthetic_meta.image_path,
                        self.synthetic_meta.label.tolist(), self.eval_tf,
                        mode=self.img_mode)))
        np.save(path, probs)
        return probs

    def _scorer_manifest(self, budget, seed):
        c = self.cfg
        return flt.scorer_manifest(
            dataset=c.dataset, arch=c.arch, pool=self.pool_hash(),
            scorer=c.filter_scorer, scorer_budget=budget, scorer_seed=seed,
            pretrained=c.pretrained, epochs=c.epochs, lr=c.lr,
            batch_size=c.batch_size, image_size=c.image_size, use_amp=c.use_amp)

    def _filter_manifest(self, budget, seed):
        c = self.cfg
        resolved = flt.resolve_scorer(c.filter_scorer, budget=budget, seed=seed,
                                      train_size=len(self.train_set), seeds0=c.seeds[0],
                                      scorer_budget=c.filter_scorer_budget,
                                      scorer_seed=c.filter_scorer_seed)
        sb, ss = resolved if resolved else (None, None)
        if c.mem_reference == "local":
            mem_b, mem_s = ((f"gen{c.gen_frac}", c.split_seed) if c.gen_frac
                            else (budget, seed))
        else:
            mem_b, mem_s = (None, None)
        return flt.filter_manifest(
            self._scorer_manifest(sb, ss) if resolved else
            flt.scorer_manifest(dataset=c.dataset, arch=c.arch, pool=self.pool_hash(),
                                scorer="none", scorer_budget=None, scorer_seed=None,
                                pretrained=c.pretrained, epochs=c.epochs, lr=c.lr,
                                batch_size=c.batch_size, image_size=c.image_size,
                                use_amp=c.use_amp),
            mode=c.filter_mode, conf_thresh=c.conf_thresh,
            require_correct=c.filter_require_correct, mem_reference=c.mem_reference,
            mem_mode=c.mem_mode, mem_quantile=c.mem_quantile, mem_thresh=c.mem_thresh,
            embed_id=c.embed_id, random_seed=c.filter_random_seed,
            mem_budget=mem_b, mem_seed=mem_s), resolved

    def self_scores(self):
        """Generator self-consistency scores for the synthetic pool (cached CSV).
        Runs score_synthetic.py (reverse-flow classification of the pool by its
        own generator) the first time; aligned to synthetic_meta by image_path."""
        c = self.cfg
        path = self.synthetic_dir / "self_scores.csv"
        if not path.exists():
            cmd = ["python", "project/score_synthetic.py",
                   "--checkpoint", c.checkpoint_path,
                   "--synthetic-dir", str(self.synthetic_dir),
                   "--dataset", c.dataset, "--n_classes", str(c.n_classes),
                   "--image_size", str(c.gen_image_size), "--beta", str(c.gen_beta),
                   "--mask_code", c.gen_mask_code,
                   *(["--latent"] if c.gen_latent else []),
                   *(["--vae_id", str(c.gen_vae_id)] if c.gen_vae_id else []),
                   "--batch", str(c.self_score_batch or
                                  (c.gen_chunk if c.gen_latent else 128)),
                   "--solver", c.gen_solver, "--step_size", str(c.gen_step_size)]
            env = dict(os.environ, PYTHONPATH=f"{c.medsymm_root}/src")
            res = subprocess.run(cmd, cwd=c.medsymm_root, env=env,
                                 capture_output=True, text=True)
            print(res.stdout[-1200:])
            if res.returncode != 0:
                print(res.stderr[-2000:])
                raise RuntimeError("self-consistency scoring failed")
        s = pd.read_csv(path)
        keep_cols = ["image_path", "pred", "margin", "match"]
        for extra in ("dmin", "dtrue"):
            if extra in s.columns:
                keep_cols.append(extra)
        assert c.self_filter_by in ("margin",) or c.self_filter_by in s.columns, (
            f"{path} predates --self-filter-by {c.self_filter_by}; delete it to rescore")
        merged = self.synthetic_meta.merge(
            s[keep_cols], on="image_path", how="left")
        assert not merged.pred.isna().any(), "self_scores.csv does not cover the pool"
        return merged

    def _self_keep(self):
        """Per-class keep mask: round-trip match AND margin in the class's top q
        (or, with self_per_class set, the class's top-N matches by margin)."""
        c = self.cfg
        m = self.self_scores()
        keep = np.zeros(len(m), dtype=bool)
        for k in range(c.n_classes):
            cls = ((m.label == k) & (m.match == 1)).values
            if cls.sum() == 0:
                print(f"  WARNING: class {k} has zero self-consistent images")
                continue
            # margin: larger is more confident. dmin/dtrue: smaller is.
            score = m[c.self_filter_by].values
            if c.self_filter_by != "margin":
                score = -score
            if c.self_per_class:
                idx = np.flatnonzero(cls)
                if len(idx) < c.self_per_class:
                    print(f"  WARNING: class {k} has only {len(idx)} matches "
                          f"(< {c.self_per_class}); keeping all of them")
                keep[idx[np.argsort(-score[idx])][:c.self_per_class]] = True
            else:
                thr = np.quantile(score[cls], 1.0 - c.self_filter_q)
                keep |= cls & (score >= thr)
        return keep

    def filtered_for(self, budget, seed):
        """The synthetic subset this arm may train on. Deterministic and cached."""
        if (budget, seed) in self._filter_cache:
            return self._filter_cache[(budget, seed)]
        c = self.cfg
        if c.filter_mode == "self_consistent":
            # Budget-independent: the generator judges its own pool once. The mem
            # screen still applies (self-consistency cannot detect near-copies).
            manifest = {"v": 1, "mode": "self_consistent", "q": c.self_filter_q,
                        "pool": self.pool_hash(),
                        "checkpoint": Path(c.checkpoint_path).name,
                        "mem_quantile": c.mem_quantile, "embed_id": c.embed_id,
                        "dataset": c.dataset}
            if c.self_per_class:  # added conditionally so q-mode keys stay stable
                manifest["per_class"] = int(c.self_per_class)
            if c.self_filter_by != "margin":   # keeps existing margin keys unchanged
                manifest["by"] = c.self_filter_by
            mhash = hashlib.sha1(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:8]
            tag = (f"selfn{c.self_per_class}" if c.self_per_class
                   else f"selfq{c.self_filter_q}")
            if c.self_filter_by != "margin":
                tag += f"-{c.self_filter_by}"
            key = f"{c.dataset}_{tag}_{mhash}"
            out_dir = self.filters_dir / key
            meta_path = out_dir / "metadata.csv"
            if meta_path.exists():
                df = pd.read_csv(meta_path)
                self._filter_cache[(budget, seed)] = (df, key)
                return df, key
            nn_dist = self._mem_distances(None)
            labels = np.array(self.synthetic_meta.label)
            keep_mem, _, _ = flt.derive_keep(nn_dist, None, labels, mode="none",
                                             mem_mode=c.mem_mode,
                                             mem_quantile=c.mem_quantile,
                                             mem_thresh=c.mem_thresh)
            keep = keep_mem & self._self_keep()
            df = self.synthetic_meta[keep].reset_index(drop=True)
            out_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(meta_path, index=False)
            (out_dir / "manifest.json").write_text(json.dumps(
                {**manifest, "n_pool": int(len(keep)), "n_kept": int(keep.sum()),
                 "kept_per_class": [int(((labels == k) & keep).sum())
                                    for k in range(c.n_classes)]}, indent=2))
            self._filter_cache[(budget, seed)] = (df, key)
            return df, key
        manifest, resolved = self._filter_manifest(budget, seed)
        key = flt.make_key(manifest)
        out_dir = self.filters_dir / key
        meta_path = out_dir / "metadata.csv"
        if meta_path.exists():
            df = pd.read_csv(meta_path)
            self._filter_cache[(budget, seed)] = (df, key)
            return df, key

        ref = None
        if c.mem_reference == "local":
            # What can be memorised is the GENERATOR's training data. In disjoint mode
            # that is the fixed generator half (budget-independent, so nothing leaks
            # across arms); otherwise the arm's own real subset, as before.
            ref = (list(self.gen_idx) if self.gen_idx is not None
                   else self.stratified_subset(budget, seed)[0])
        nn_dist = None if c.mem_reference == "none" else self._mem_distances(ref)
        probs = self._scorer_probs(*resolved) if resolved else None
        labels = np.array(self.synthetic_meta.label)

        keep, keep_mem, keep_score = flt.derive_keep(
            nn_dist, probs, labels, mode=c.filter_mode, conf_thresh=c.conf_thresh,
            require_correct=c.filter_require_correct, mem_mode=c.mem_mode,
            mem_quantile=c.mem_quantile, mem_thresh=c.mem_thresh,
            random_seed=c.filter_random_seed)

        df = self.synthetic_meta[keep].reset_index(drop=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(meta_path, index=False)
        stats_ = flt.summarise_keep(keep, keep_mem, keep_score, labels, c.n_classes)
        (out_dir / "manifest.json").write_text(json.dumps({**manifest, **stats_,
                                                           "budget": budget, "seed": seed}, indent=2))
        self._filter_cache[(budget, seed)] = (df, key)
        return df, key

    def filter_synthetic(self, plot=True, **overrides):
        """Populate the filter caches for the current sweep; return a summary table.

        Also sets `self.filtered` to the scorer-free (memorisation-only, full-reference)
        pool. That pool is deliberately budget-independent, and is what S1 pretrains on,
        so one pretrain can be shared across budgets instead of one per (budget, seed).
        """
        for k, v in overrides.items():
            setattr(self.cfg, k, v)
        c = self.cfg

        # Scorer-free pool for S1 / D1 -- no labels beyond the requested class are used.
        nn_dist_full = self._mem_distances(None)
        labels = np.array(self.synthetic_meta.label)
        keep_mem_full, _, _ = flt.derive_keep(nn_dist_full, None, labels, mode="none",
                                              mem_mode=c.mem_mode, mem_quantile=c.mem_quantile,
                                              mem_thresh=c.mem_thresh)
        self.filtered = self.synthetic_meta[keep_mem_full].reset_index(drop=True)
        self.filtered.to_csv(self.filtered_dir / "metadata.csv", index=False)

        if plot:
            cut = np.quantile(nn_dist_full, c.mem_quantile)
            plt.hist(nn_dist_full, bins=40); plt.axvline(cut, color="r", ls="--")
            plt.xlabel("nearest-real distance"); plt.ylabel("count")
            plt.title("Memorisation screen (full-train reference)")
            self._savefig("memorisation_screen"); plt.show()
        print(f"Memorisation discard (full ref): {int((~keep_mem_full).sum())}/{len(keep_mem_full)} "
              f"({(~keep_mem_full).mean()*100:.1f}%)  -> S1/D1 pool = {len(self.filtered)}")

        rows = []
        for budget in c.budgets:
            for seed in c.seeds:
                df, key = self.filtered_for(budget, seed)
                row = {"budget": budget, "seed": seed, "mode": c.filter_mode,
                       "scorer": c.filter_scorer, "mem_ref": c.mem_reference,
                       "n_pool": len(self.synthetic_meta), "n_kept": len(df)}
                row.update({f"kept_class{k}": int((df.label == k).sum())
                            for k in range(c.n_classes)})
                row["filter_key"] = key
                rows.append(row)
        summary = pd.DataFrame(rows)
        print(summary.to_string(index=False))
        return summary

    # --------------------------------------------------- synthetic arms (S1-3)
    def _synth_ds(self, df):
        return PathDataset(df.image_path.tolist(), df.label.tolist(), self.train_tf,
                           mode=self.img_mode)

    def run_synthetic(self):
        for seed in self.cfg.seeds:
            # S1 pretrains on the scorer-free, budget-independent pool (protocol choice:
            # per-budget filtering here would multiply pretraining cost by len(budgets)).
            pre_state = None

            def pretrain_state():
                nonlocal pre_state
                if pre_state is None:
                    path = self.models_dir / f"S1pre_{self.pool_hash()}_{self.cfg.arch}_s{seed}.pt"
                    if path.exists():
                        pre_state = torch.load(path, map_location="cpu")
                    else:
                        m, _ = self.train_classifier(self._synth_ds(self.filtered),
                                                     np.array(self.filtered.label), seed,
                                                     weighted=False, tag="S1-pretrain")
                        pre_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
                        torch.save(pre_state, path)
                return pre_state

            for budget in self.cfg.budgets:
                idx, sub_labels = self.stratified_subset(budget, seed)
                real_sub = Subset(self.train_set, idx)
                syn_df, fkey = self.filtered_for(budget, seed)
                syn_labels = np.array(syn_df.label)

                if self.already_done("S1", budget, seed, fkey):
                    print(f"  [skip] S1 n={budget} seed={seed}")
                else:
                    self.run_supervised("S1", real_sub, sub_labels, budget, seed,
                                        lr=self.cfg.lr_finetune, init_state=pretrain_state(),
                                        filter_key=fkey, n_syn_used=len(self.filtered))

                if self.already_done("S2", budget, seed, fkey):
                    print(f"  [skip] S2 n={budget} seed={seed}")
                else:
                    mix_ds = ConcatDataset([IntLabel(real_sub), self._synth_ds(syn_df)])
                    self.run_supervised("S2", mix_ds, np.concatenate([sub_labels, syn_labels]),
                                        budget, seed, weighted=False,
                                        filter_key=fkey, n_syn_used=len(syn_df))

                if self.already_done("S3", budget, seed, fkey):
                    print(f"  [skip] S3 n={budget} seed={seed}")
                else:
                    # Top every class up to the majority count with its own synthetic
                    # images (capped by availability). For 2 classes this reduces to
                    # the old minority-to-parity top-up, same rows in the same order.
                    counts = np.bincount(sub_labels, minlength=self.cfg.n_classes)
                    adds = []
                    for k in range(self.cfg.n_classes):
                        need = int(counts.max() - counts[k])
                        if need > 0:
                            syn_k = syn_df[syn_df.label == k].reset_index(drop=True)
                            adds.append(syn_k.iloc[:min(need, len(syn_k))])
                    add_df = (pd.concat(adds, ignore_index=True) if adds
                              else syn_df.iloc[:0])
                    s3_ds = ConcatDataset([IntLabel(real_sub), self._synth_ds(add_df)])
                    self.run_supervised("S3", s3_ds,
                                        np.concatenate([sub_labels, np.array(add_df.label)]),
                                        budget, seed, weighted=False,
                                        filter_key=fkey, n_syn_used=len(add_df))

                if self.cfg.fixed_total and self.cfg.fixed_total > budget:
                    # Size-matched substitution: keep the TOTAL number of training
                    # images fixed and vary how many of them are real. Comparable
                    # to a classifier trained on the same total of real images.
                    if self.already_done("S5", budget, seed, fkey):
                        print(f"  [skip] S5 n={budget} seed={seed}")
                    else:
                        need = int(self.cfg.fixed_total - budget)
                        rng5 = np.random.default_rng(20000 + seed)
                        per = need // self.cfg.n_classes
                        picks = []
                        for k in range(self.cfg.n_classes):
                            syn_k = syn_df[syn_df.label == k]
                            take = min(per, len(syn_k))
                            if take:
                                picks.append(syn_k.iloc[
                                    rng5.choice(len(syn_k), size=take, replace=False)])
                        add5 = (pd.concat(picks, ignore_index=True) if picks
                                else syn_df.iloc[:0])
                        ds5 = ConcatDataset([IntLabel(real_sub), self._synth_ds(add5)])
                        self.run_supervised("S5", ds5,
                                            np.concatenate([sub_labels, np.array(add5.label)]),
                                            budget, seed, weighted=False,
                                            filter_key=fkey, n_syn_used=len(add5))

    def run_exchange_rate(self, sizes):
        """DX: synthetic-only training at matched sizes -- the synthetic-vs-real
        exchange-rate curve. Stratified subsamples of the mem-only pool; recorded
        as arm 'DX' with budget = synthetic count, so the ledger overlays directly
        on the real-budget baselines."""
        pool = self.filtered
        labels = np.array(pool.label)
        for seed in self.cfg.seeds:
            for n in sizes:
                if self.already_done("DX", n, seed):
                    print(f"  [skip] DX n={n} seed={seed}")
                    continue
                rng = np.random.default_rng(10000 + seed)
                idx = []
                for c in range(self.cfg.n_classes):
                    c_idx = np.where(labels == c)[0]
                    take = min(int(round(n * len(c_idx) / len(labels))), len(c_idx))
                    idx.extend(rng.choice(c_idx, size=take, replace=False).tolist())
                df = pool.iloc[sorted(idx)].reset_index(drop=True)
                self.run_supervised("DX", self._synth_ds(df), np.array(df.label),
                                    n, seed, weighted=False, n_syn_used=len(df))

    # ---------------------------------------------- distillation diagnostics
    def run_diagnostic_d1(self):
        """D1: train on synthetic ONLY, test on real. If D1 recovers baseline/C1-level
        AUC, the synthetic set alone carries the decision function -> distillation."""
        self.d1_models = {}
        for seed in self.cfg.seeds:
            model, val_auc = self.train_classifier(
                self._synth_ds(self.filtered), np.array(self.filtered.label), seed,
                weighted=False, tag="D1")
            self.d1_models[seed] = model
            if self.already_done("D1", 0, seed):
                print(f"  [skip-record] D1 seed={seed} already in ledger")
                continue
            self.add_result("D1", 0, seed,
                            self.evaluate_on_test(model, self.best_threshold_on_val(model)),
                            val_auc, n_syn_used=len(self.filtered))
        aucs = [r["test_auc"] for r in self.results if r["arm"] == "D1"]
        if not aucs:
            aucs = self.ledger.query("arm == 'D1'")["test_auc"].tolist()
        print(f"\nD1 (synthetic-only) mean test AUC: {np.mean(aucs):.4f}  "
              f"[C1 MSF ref {self.cfg.c1_auc}; real baselines ~0.94]")
        print("D1 near or above C1 means the synthetic set alone carries the class structure. "
              "Whether that is DISTILLATION of MSF's decision function or plain coverage of "
              "the data manifold is decided by the fingerprint in distillation_agreement(), "
              "not by this number.")

    def msf_test_predictions(self):
        """MSF's own reverse-flow classification of the real test split (subprocess), cached to Drive."""
        # Versioned filename: the old cache held only hard predictions, so a stale file
        # would be reused forever and measure_c1() could never compute an AUC.
        c = self.cfg
        out_csv = (c.run_dir /
                   f"msf_preds_{c.dataset}_test_{c.gen_image_size}_beta{c.gen_beta}"
                   f"_{c.gen_solver}{c.gen_step_size}.csv")
        if out_csv.exists():
            return pd.read_csv(out_csv)
        # classify_medmnist.py also writes soft distance-to-class scores (msf_negdist_*),
        # which is what makes a measured C1 AUC possible.
        cmd = [
            "python", "project/classify_medmnist.py",
            "--checkpoint", c.checkpoint_path, "--output_csv", str(out_csv),
            "--dataset", c.dataset, "--n_classes", str(c.n_classes),
            "--image_size", str(c.gen_image_size), "--beta", str(c.gen_beta),
            "--rgb_mask", "--mask_code", c.gen_mask_code,
            *(["--latent", "--source_size", "224"] if c.gen_latent else []),
            *(["--vae_id", str(c.gen_vae_id)] if c.gen_vae_id else []),
            "--solver", c.gen_solver, "--step_size", str(c.gen_step_size),
        ]
        env = dict(os.environ, PYTHONPATH=f"{c.medsymm_root}/src")
        res = subprocess.run(cmd, cwd=c.medsymm_root, env=env, capture_output=True, text=True)
        print(res.stdout.strip()[-800:])
        if res.returncode != 0:
            if c.dataset != "pneumoniamnist":
                print(res.stderr[-2500:])
                raise RuntimeError("MSF classification failed")
            # Fall back to the legacy script: it yields no soft scores (so no measured C1
            # AUC) but still produces the hard predictions the fingerprint needs.
            print("classify_medmnist.py failed; falling back to classify_pneumoniamnist.py")
            print(res.stderr[-1500:])
            legacy = [
                "python", "project/classify_pneumoniamnist.py",
                "--checkpoint", c.checkpoint_path, "--output_csv", str(out_csv),
                "--dataset", "pneumoniamnist", "--n_classes", "2",
                "--image_size", str(c.gen_image_size), "--beta", str(c.gen_beta),
                "--rgb_mask", "--solver", c.gen_solver, "--step_size", str(c.gen_step_size),
                "--model_channels", "64", "--num_res_blocks", "2",
                "--channel_mult", "1", "2", "2", "2",
                "--num_heads", "4", "--num_head_channels", "64", "--attention_resolutions", "2",
            ]
            res = subprocess.run(legacy, cwd=c.medsymm_root, env=env,
                                 capture_output=True, text=True)
            print(res.stdout.strip()[-800:])
            if res.returncode != 0:
                print(res.stderr[-2500:])
                raise RuntimeError("MSF classification failed")
        return pd.read_csv(out_csv)

    def distillation_agreement(self, budget=None, seed=None):
        """Fingerprint: does a synthetic-trained ResNet (D1) copy MSF's predictions --
        especially MSF's *errors* -- more than a real-trained ResNet (B0)? Copying errors
        needs copying the function, which mere data-manifold coverage cannot explain."""
        msf = self.msf_test_predictions()
        ty, msf_pred = msf["true"].values, msf["msf_pred"].values

        budget = budget if budget is not None else (
            self.cfg.fingerprint_budget or max(self.cfg.budgets))
        seeds = self.cfg.seeds if seed is None else [seed]
        err = msf_pred != ty                       # images MSF gets wrong
        n_err = int(err.sum())

        def scores(model):
            """Agreement, plus (binary only) a CALIBRATION-MATCHED variant.

            Raw agreement is confounded: D1 trains on a near-balanced synthetic set
            while B0 trains on the real class prior, so the two models sit at different
            operating points. For binary tasks, matching each model's positive rate to
            MSF's removes that prior difference; for K classes only raw agreement over
            argmax predictions is reported.
            """
            if self.cfg.n_classes == 2:
                y, p = self.predict_probs(model, self.loader(self.test_set))
                assert np.array_equal(y, ty), "test order mismatch between MSF CSV and loader"
                raw = (p >= 0.5).astype(int)
                thr = np.quantile(p, 1.0 - (msf_pred == 1).mean())  # match MSF's positive rate
                variants = (("", raw), ("_matched", (p >= thr).astype(int)))
            else:
                y, p = self.predict_proba(model, self.loader(self.test_set))
                assert np.array_equal(y, ty), "test order mismatch between MSF CSV and loader"
                variants = (("", p.argmax(1)),)
            out = {}
            for name, pred in variants:
                out["agree_with_MSF" + name] = float((pred == msf_pred).mean())
                out["agree_on_MSF_errors" + name] = (
                    float((pred[err] == msf_pred[err]).mean()) if n_err else np.nan)
            return out

        rows = []
        for s in seeds:
            real_model = self.baseline_model(budget, s, train_if_missing=False)
            syn_model = self.d1_models.get(s)
            if real_model is None or syn_model is None:
                continue
            rows.append({"seed": s, "model": "real-trained (B0)", **scores(real_model)})
            rows.append({"seed": s, "model": "synthetic-trained (D1)", **scores(syn_model)})
        if not rows:
            raise RuntimeError("no models available; run run_baselines() and run_diagnostic_d1()")

        per_seed = pd.DataFrame(rows)
        agg = (per_seed.drop(columns=["seed"]).groupby("model").agg(["mean", "std"]).round(3))
        print(f"MSF test accuracy: {(msf_pred == ty).mean():.3f}  (errors: {n_err}/{len(ty)}), "
              f"B0 budget={budget}, seeds={list(per_seed.seed.unique())}")
        print("Distillation fingerprint = HIGHER agree_on_MSF_errors for the synthetic-trained "
              "model. Read the *_matched columns: they remove the class-prior confound.")
        self._fingerprint_per_seed = per_seed
        return agg

    # ----------------------------------------------------- reference & summary
    def measure_c1(self):
        """Measure MSF's own classification on OUR test split instead of quoting the paper.

        The published 0.952 was produced on the authors' setup; our reproduction of the
        ResNet-18 baseline already lands well above the paper's (0.970 vs 0.944), so the
        published constant is not a like-for-like reference. `classify_medmnist.py` also
        writes soft distance-to-class scores, which is what makes an AUC possible here.
        """
        msf = self.msf_test_predictions()
        y, pred = msf["true"].values, msf["msf_pred"].values
        acc = float((pred == y).mean())
        if "msf_pred_mode" in msf.columns:
            mode_acc = float((msf["msf_pred_mode"].values == y).mean())
            print(f"C1 decode check: mean-distance ACC {acc:.4f}, "
                  f"per-pixel-mode ACC {mode_acc:.4f}")
        auc = np.nan
        k = self.cfg.n_classes
        dist_cols = [f"msf_negdist_{i}" for i in range(k)]
        if k > 2 and all(col in msf.columns for col in dist_cols):
            # macro one-vs-rest over softmaxed distance scores (sklearn requires
            # the multi-class score matrix to row-normalise)
            scores_ = np.exp(msf[dist_cols].values)
            scores_ = scores_ / scores_.sum(axis=1, keepdims=True)
            auc = float(roc_auc_score(y, scores_, multi_class="ovr", average="macro"))
        elif "msf_negdist_1" in msf.columns:    # binary: soft scores -> real AUC
            auc = float(roc_auc_score(y, msf["msf_negdist_1"].values))
        elif "msf_negdist_0" in msf.columns:
            auc = float(roc_auc_score(y, -msf["msf_negdist_0"].values))
        print(f"C1 measured on our test split: ACC {acc:.4f}"
              + (f", AUC {auc:.4f}" if not np.isnan(auc) else
                 "  (no soft scores in the cached CSV -- regenerate with classify_medmnist.py for AUC)")
              + f"   [published: ACC {self.cfg.c1_acc}, AUC {self.cfg.c1_auc}]")
        if not np.isnan(auc):
            self.cfg.c1_auc_measured = auc
        self.cfg.c1_acc_measured = acc
        return {"c1_acc_measured": acc, "c1_auc_measured": auc,
                "c1_acc_published": self.cfg.c1_acc, "c1_auc_published": self.cfg.c1_auc}

    def record_c1(self, use_measured=False):
        auc = getattr(self.cfg, "c1_auc_measured", None) if use_measured else None
        acc = getattr(self.cfg, "c1_acc_measured", None) if use_measured else None
        auc = self.cfg.c1_auc if auc is None else auc
        acc = self.cfg.c1_acc if acc is None else acc
        for budget in self.cfg.budgets:
            if self.already_done("C1", budget, -1):
                continue
            self.add_result("C1", budget, -1,
                            {"test_auc": auc, "test_acc": acc,
                             "test_balacc": np.nan, "test_f1": np.nan}, np.nan)
        print(f"C1 MSF reference recorded: AUC {auc}, ACC {acc}"
              + ("  (measured)" if use_measured else "  (published)"))

    @staticmethod
    def _ci95(x):
        x = x.dropna().values
        if len(x) < 2:
            return np.nan
        return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))

    def current_selection(self):
        """Rows in the ledger belonging to the current configuration."""
        df = self.ledger
        if not len(df):
            return df
        for col, val in (("dataset", self.cfg.dataset), ("arch", self.cfg.arch),
                         ("run_tag", self.cfg.run_tag)):
            if col in df.columns:
                df = df[df[col].fillna("").astype(str) == str(val)]
        return df

    def summarize(self, select=True):
        # Read the LEDGER, never overwrite it: it is now a growing record and rewriting
        # it from the in-memory list would delete every previous run.
        res = self.current_selection() if select else self.ledger
        if not len(res):
            res = pd.DataFrame(self.results)
        aggs = dict(auc_mean=("test_auc", "mean"),
                    auc_ci=("test_auc", self._ci95),
                    acc_mean=("test_acc", "mean"),
                    n_seeds=("test_auc", "count"))
        if "test_qwk" in res.columns:      # ordinal datasets (K>2) record QWK
            aggs["qwk_mean"] = ("test_qwk", "mean")
            aggs["qwk_ci"] = ("test_qwk", self._ci95)
        summary = (res.groupby(["arm", "budget"]).agg(**aggs).reset_index())

        BASE, SYN = ["B0", "B1", "B2", "B3"], ["S1", "S2", "S3"]
        # dropna=False keeps all-NaN CI columns (e.g. single-seed quick runs), so
        # `means` and `cis` stay column-aligned.
        means = summary.pivot_table(index="budget", columns="arm", values="auc_mean", dropna=False)
        cis = summary.pivot_table(index="budget", columns="arm", values="auc_ci", dropna=False)
        for col in BASE + SYN + ["C1"]:
            if col not in means.columns:
                means[col] = np.nan
            if col not in cis.columns:
                cis[col] = np.nan

        rows = []
        for b in means.index:
            if means.loc[b, BASE].isna().all():   # e.g. D1's sentinel budget 0
                continue
            best_base = means.loc[b, BASE].max()
            best_base_arm = means.loc[b, BASE].idxmax()
            for s in SYN:
                if np.isnan(means.loc[b, s]):
                    continue
                sig = ""
                if not np.isnan(cis.loc[b, s]) and not np.isnan(cis.loc[b, best_base_arm]):
                    lo_s = means.loc[b, s] - cis.loc[b, s]
                    hi_base = best_base + cis.loc[b, best_base_arm]
                    sig = "yes" if lo_s > hi_base else "no"
                rows.append({"budget": b, "arm": s, "auc": round(means.loc[b, s], 4),
                             "best_baseline": f"{best_base_arm} {best_base:.4f}",
                             "delta_vs_base": round(means.loc[b, s] - best_base, 4),
                             "CIs_separate": sig,
                             "beats_C1": "yes" if means.loc[b, s] > self.cfg.c1_auc else "no"})
        comparison = pd.DataFrame(rows)
        summary.to_csv(self.cfg.run_dir / "summary.csv", index=False)
        comparison.to_csv(self.cfg.run_dir / "comparison.csv", index=False)
        print(f"Ledger: {len(self.ledger)} rows ({len(res)} in this selection) -> {self.results_path}")
        return summary, comparison

    def paired_tests(self, alpha=0.05):
        """Paired seed-wise test vs the strongest baseline + BH correction.

        Runs on the current selection of the ledger, so rows from a different filter
        configuration cannot be silently averaged into the same (arm, budget, seed) cell.
        """
        res = self.current_selection()
        if not len(res):
            res = pd.DataFrame(self.results)
        return paired_tests_from_csv(res, alpha=alpha)

    def plot(self, summary):
        fig, ax = plt.subplots(figsize=(8, 5))
        palette = {"B0": "C0", "B1": "C1", "B2": "C2", "B3": "C7",
                   "S1": "C3", "S2": "C4", "S3": "C5"}
        for arm in ["B0", "B1", "B2", "B3", "S1", "S2", "S3"]:
            d = summary[summary.arm == arm].sort_values("budget")
            if len(d):
                ls = "--" if arm.startswith("B") else "-"
                ax.errorbar(d.budget, d.auc_mean, yerr=d.auc_ci.fillna(0), marker="o",
                            capsize=3, ls=ls, color=palette[arm], label=arm)
        ax.axhline(self.cfg.c1_auc, color="gray", ls="--", label=f"C1 MSF ({self.cfg.c1_auc})")
        ax.axhline(0.944, color="black", ls=":", label="B0 target (0.944)")
        d1 = summary[summary.arm == "D1"]
        if len(d1):
            ax.axhline(d1.auc_mean.mean(), color="C6", ls="-.",
                       label=f"D1 synth-only ({d1.auc_mean.mean():.3f})")
        ax.set_xlabel("real training images"); ax.set_ylabel("test AUC")
        ax.set_title("Gain vs data budget (baselines dashed, synthetic solid)")
        ax.legend(ncol=2, fontsize=8); ax.grid(alpha=0.3)
        plt.tight_layout(); self._savefig("gain_vs_budget"); plt.show()
