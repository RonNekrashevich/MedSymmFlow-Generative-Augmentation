"""Headless entrypoint for the PneumoniaMNIST augmentation experiment.

Runs the full pipeline (baselines B0-B2, generation, filtering, synthetic arms
S1-S3, D1 diagnostic, C1 reference, summary, distillation fingerprint) and writes
all results + figures to --out. Designed for a batch job on a GPU cluster (Run:AI)
where there is no interactive display and outputs must land on a mounted volume.

Prereqs on the node: the repo present (this file lives in it), a CUDA PyTorch, and
the light deps (medmnist torchdiffeq diffusers accelerate zuko scikit-learn scipy
loguru python-dotenv datasets). Install them in the image or via --pip.

Example (from the repo root, GPU available):
    python project/run_experiment.py --out /storage/medsymm_out --seeds 0 1 2 3 4
"""
import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: no display

HERE = Path(__file__).resolve().parent          # .../MedSymmFlow/project
REPO = HERE.parent                              # .../MedSymmFlow
sys.path.insert(0, str(HERE))

LIGHT_DEPS = ["numpy<2", "medmnist", "torchdiffeq", "diffusers", "accelerate", "zuko",
              "scikit-learn", "scipy", "loguru", "python-dotenv"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pneumoniamnist",
                    choices=["pneumoniamnist", "retinamnist", "dermamnist"])
    ap.add_argument("--out", default="/storage/medsymm_out", help="output dir on the mounted volume")
    ap.add_argument("--weights-root", default=None,
                    help="persistent dir for the 755 MB MedSymmFlow weights (default: repo dir). "
                         "Point this at the mounted volume so the download happens once.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--budgets", type=int, nargs="+", default=[250, 500, 1000])
    ap.add_argument("--quick", action="store_true", help="fast smoke test (overrides seeds/budgets)")
    ap.add_argument("--pip", action="store_true", help="pip install the light deps first")
    ap.add_argument("--filter-mode", default="keep_confident",
                    choices=["none", "keep_confident", "keep_uncertain", "random_match",
                             "self_consistent"])
    ap.add_argument("--self-q", type=float, default=0.5,
                    help="self_consistent: per-class top fraction kept by the "
                         "generator's own round-trip confidence")
    ap.add_argument("--fixed-total", type=int, default=None,
                    help="extra arm: top the real budget up to this many images with "
                         "synthetic ones, keeping the total training-set size fixed "
                         "(comparable to training on that many real images)")
    ap.add_argument("--self-filter-by", default="margin",
                    choices=["margin", "dmin", "dtrue"],
                    help="rank synthetic images by the top-two margin (ours) or by "
                         "the distance to the winning / requested class code (the "
                         "convention used in the MedSymmFlow paper)")
    ap.add_argument("--self-score-batch", type=int, default=None,
                    help="batch size when the generator scores its own pool "
                         "(default: --gen-chunk in latent mode, else 128)")
    ap.add_argument("--self-per-class", type=int, default=None,
                    help="self_consistent: keep the top-N matches per class by margin "
                         "instead of a fraction (dose- and balance-matched filtering "
                         "from an oversized pool)")
    ap.add_argument("--filter-scorer", default="local", choices=["none", "local", "full"])
    ap.add_argument("--conf-thresh", type=float, default=0.60)
    ap.add_argument("--mem-reference", default="local", choices=["none", "local", "full"])
    ap.add_argument("--beta", type=float, default=None, help="generation label-noise amplitude")
    ap.add_argument("--gen-frac", type=float, default=None,
                    help="train the MSF generator FROM SCRATCH on this stratified fraction "
                         "of the train split; classifier arms use only the complement, so "
                         "generator and ResNet share no real image")
    ap.add_argument("--split-seed", type=int, default=0,
                    help="seed of the generator/classifier split (independent of --seeds)")
    ap.add_argument("--gen-epochs", type=int, default=None,
                    help="generator training epochs (default 600, or 20 with --quick)")
    ap.add_argument("--gen-batch-size", type=int, default=128)
    ap.add_argument("--gen-lr", type=float, default=1e-3)
    ap.add_argument("--gen-dropout", type=float, default=0.0)
    ap.add_argument("--gen-balance", action="store_true",
                    help="50/50 class sampling during generator training (the train "
                         "data is 74% pneumonia)")
    ap.add_argument("--gen-pretrain-epochs", type=int, default=0,
                    help=">0: pretrain the generator on ChestMNIST (no-finding vs "
                         "pneumonia, ~43k CXRs) before fine-tuning on the gen half")
    ap.add_argument("--external-checkpoint", default=None,
                    help="use this train_msf_external.py checkpoint AS the generator "
                         "(zero benchmark images seen); classifier pool = full train split")
    ap.add_argument("--gen-mask-code", default="rgb",
                    choices=["rgb", "onehot", "thermometer"],
                    help="class-code geometry for the generator (Phase B)")
    ap.add_argument("--gen-cfg-drop", type=float, default=0.0,
                    help=">0: train the generator with class-code dropout (enables CFG)")
    ap.add_argument("--gen-cfg-w", type=float, default=0.0,
                    help=">0: classifier-free guidance weight at generation")
    ap.add_argument("--gen-init-checkpoint", default=None,
                    help="disjoint mode: warm-start scratch generator training from "
                         "this checkpoint (external-corpus fine-tune variant)")
    ap.add_argument("--gen-chunk", type=int, default=None,
                    help="images per generation subprocess batch (default 200; use "
                         "~50 for 256px latent decoding, whose VAE decoder OOMs at 200)")
    ap.add_argument("--syn-per-class", type=int, default=None,
                    help="synthetic images generated per class (default: 1000 full / 200 quick)")
    ap.add_argument("--exchange-sizes", type=int, nargs="+", default=None,
                    help="run synthetic-only arms at these matched sizes (exchange-rate curve)")
    ap.add_argument("--scratch-clf", action="store_true",
                    help="train classifiers from random init instead of ImageNet "
                         "weights (matches the MedMNIST from-scratch protocol)")
    ap.add_argument("--heavy-aug", action="store_true",
                    help="add the B4 baseline: real data + TrivialAugmentWide (the "
                         "strong free alternative to synthetic augmentation)")
    ap.add_argument("--std-stem", action="store_true",
                    help="keep torchvision's standard 7x7/stride-2 ResNet stem at "
                         "every size (with --scratch-clf: the MedMNIST/MSF-paper "
                         "baseline ResNet-18 exactly)")
    ap.add_argument("--image-size", type=int, default=28, choices=[28, 64, 128, 224],
                    help="classifier resolution (MedMNIST source size); 128/224 use "
                         "the standard ResNet-18 stem like the MedMNIST(224) baselines")
    ap.add_argument("--gen-size", type=int, default=32,
                    help="generator resolution (32 = published 28->32; 64 = native 64px; "
                         "256 = MedMNIST+ 224px source, for --gen-latent)")
    ap.add_argument("--gen-latent", action="store_true",
                    help="LatMSF: generator flows in SD-VAE latent space (use with "
                         "--gen-size 256 so latents are 32x32)")
    ap.add_argument("--gen-model-channels", type=int, default=64,
                    help="generator UNet width (64 = 9M published; 128 ~= 36M, the "
                         "paper's LatMSF capacity)")
    ap.add_argument("--gen-t-lognorm", action="store_true",
                    help="SD3 logit-normal timestep sampling for generator training")
    ap.add_argument("--gen-vae-id", type=str, default=None,
                    help="latent VAE: diffusers AutoencoderKL id or local dir "
                         "(default SD-VAE; e.g. REPA-E/e2e-sdvae-hf, or the "
                         "finetune_vae_aptos.py output dir)")
    ap.add_argument("--gen-repa-weight", type=float, default=0.0,
                    help=">0: U-REPA alignment of the generator's UNet middle block "
                         "to frozen DINOv2-S features (0.5 = REPA default)")
    ap.add_argument("--gen-repa-teacher", type=str, default=None,
                    help="alignment teacher: dinov2_vits14 (default) or "
                         "retfound:<local .pth> (RETFound fundus foundation model)")
    ap.add_argument("--run-tag", default="", help="label this run in the ledger")
    ap.add_argument("--legacy-filter", action="store_true",
                    help="reproduce the old (budget-dependent, leaky) filter semantics")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--filter-ablation", action="store_true",
                    help="after the main run, re-run the synthetic arms for every filter "
                         "mode (reuses the caches, so nearly free)")
    args = ap.parse_args()

    if args.pip:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *LIGHT_DEPS], check=True)

    from augmentation import Experiment, Config

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.gen_pretrain_epochs and args.dataset != "pneumoniamnist":
        ap.error("--gen-pretrain-epochs is only wired for pneumoniamnist "
                 "(ChestMNIST pretraining; the ablation showed it does not help anyway)")

    cfg = Config(
        quick=args.quick,
        dataset=args.dataset,
        image_size=args.image_size,
        gen_image_size=args.gen_size,
        save_dir=str(out),
        medsymm_root=str(REPO),
        weights_root=args.weights_root,
        scratch_dir=str(out / "scratch"),
        fig_dir=str(out / "figures"),
        seeds=(None if args.quick else args.seeds),
        budgets=(None if args.quick else args.budgets),
        filter_mode=args.filter_mode,
        pretrained=not args.scratch_clf,
        clf_stem=("standard" if args.std_stem else "small"),
        heavy_aug_baseline=args.heavy_aug,
        self_filter_q=args.self_q,
        self_per_class=args.self_per_class,
        self_score_batch=args.self_score_batch,
        self_filter_by=args.self_filter_by,
        fixed_total=args.fixed_total,
        filter_scorer=args.filter_scorer,
        conf_thresh=args.conf_thresh,
        mem_reference=args.mem_reference,
        legacy_filter=args.legacy_filter,
        resume=not args.no_resume,
        run_tag=args.run_tag,
        **({"gen_beta": args.beta} if args.beta is not None else {}),
        **({"syn_per_class": args.syn_per_class} if args.syn_per_class else {}),
        **({"gen_chunk": args.gen_chunk} if args.gen_chunk else {}),
        **({"gen_frac": args.gen_frac, "split_seed": args.split_seed,
            "gen_epochs": args.gen_epochs or (20 if args.quick else 600),
            "gen_lr": args.gen_lr, "gen_dropout": args.gen_dropout,
            "gen_balance": args.gen_balance,
            "gen_pretrain_epochs": args.gen_pretrain_epochs,
            "gen_init_checkpoint": args.gen_init_checkpoint,
            "gen_mask_code": args.gen_mask_code,
            "gen_cfg_drop": args.gen_cfg_drop, "gen_cfg_w": args.gen_cfg_w,
            "gen_latent": args.gen_latent,
            "gen_model_channels": args.gen_model_channels,
            "gen_t_lognorm": args.gen_t_lognorm,
            "gen_vae_id": args.gen_vae_id,
            "gen_repa_weight": args.gen_repa_weight,
            "gen_repa_teacher": args.gen_repa_teacher}
           if args.gen_frac else {}),
        **({"external_checkpoint": args.external_checkpoint}
           if args.external_checkpoint else {}),
    )
    exp = Experiment(cfg)

    print("== data =="); print(exp.setup_data().to_string(index=False))
    print("== selftest =="); exp.selftest_repro(strict=False)   # needs train_set
    print("== baselines =="); exp.run_baselines()
    if args.gen_frac and not Path(cfg.checkpoint_path).exists():
        if cfg.gen_pretrain_epochs and not Path(cfg.pretrain_checkpoint_path).exists():
            print("== generator pretraining (ChestMNIST) ==")
            subprocess.run([sys.executable, str(HERE / "pretrain_msf_chestmnist.py"),
                            "--out", cfg.pretrain_checkpoint_path,
                            "--epochs", str(cfg.gen_pretrain_epochs),
                            "--beta", str(cfg.gen_beta)],
                           check=True, cwd=str(REPO))
        print("== generator training (from scratch, disjoint split) ==")
        subprocess.run([sys.executable, str(HERE / "train_msf_scratch.py"),
                        "--out", cfg.checkpoint_path,
                        "--dataset", args.dataset,
                        "--gen-frac", str(args.gen_frac),
                        "--split-seed", str(args.split_seed),
                        "--epochs", str(cfg.gen_epochs),
                        "--batch-size", str(args.gen_batch_size),
                        "--lr", str(cfg.gen_lr),
                        "--dropout", str(cfg.gen_dropout),
                        "--mask-code", cfg.gen_mask_code,
                        "--cfg-drop", str(cfg.gen_cfg_drop),
                        "--beta", str(cfg.gen_beta),
                        "--size", str(cfg.gen_image_size),
                        *(["--latent"] if cfg.gen_latent else []),
                        "--model-channels", str(cfg.gen_model_channels),
                        *(["--t-lognorm"] if cfg.gen_t_lognorm else []),
                        *(["--vae-id", str(cfg.gen_vae_id)] if cfg.gen_vae_id else []),
                        *(["--repa-weight", str(cfg.gen_repa_weight)]
                          if cfg.gen_repa_weight else []),
                        *(["--repa-teacher", str(cfg.gen_repa_teacher)]
                          if cfg.gen_repa_teacher else []),
                        *(["--balance-classes"] if cfg.gen_balance else []),
                        *(["--init-checkpoint", cfg.pretrain_checkpoint_path]
                          if cfg.gen_pretrain_epochs else []),
                        *(["--init-checkpoint", cfg.gen_init_checkpoint]
                          if cfg.gen_init_checkpoint else [])],
                       check=True, cwd=str(REPO))
    print("== generation =="); exp.download_weights(); exp.generate_synthetic(); exp.visualize_samples()
    print("== filtering =="); exp.filter_synthetic()
    print("== synthetic arms =="); exp.run_synthetic()
    print("== D1 diagnostic =="); exp.run_diagnostic_d1()
    if args.exchange_sizes:
        print("== exchange rate (synthetic-only at matched sizes) ==")
        exp.run_exchange_rate(args.exchange_sizes)
    print("== C1 reference ==")
    if args.gen_frac:
        # The published C1 (0.952) came from a generator trained on ALL 4708 images
        # for 1000 epochs -- not a valid reference for a scratch disjoint generator.
        # Measure our generator's own reverse-flow classification and record that.
        exp.measure_c1()
        exp.record_c1(use_measured=True)
    else:
        exp.record_c1()

    print("== summary ==")
    summary, comparison = exp.summarize()
    summary.to_csv(out / "summary.csv", index=False)
    comparison.to_csv(out / "comparison.csv", index=False)
    print(summary.to_string(index=False))
    print("\n== synthetic vs strongest baseline ==")
    print(comparison.to_string(index=False))
    exp.plot(summary)

    print("\n== distillation fingerprint ==")
    try:
        fp = exp.distillation_agreement()
        fp.to_csv(out / "fingerprint.csv")
        print(fp.to_string())
        print(exp.measure_c1())
    except Exception as e:  # isolated: the rest of the run already succeeded
        print("distillation_agreement failed:", repr(e))

    if args.filter_ablation:
        # Filter modes reuse the cached embeddings and scorer probabilities, so only the
        # keep-mask and the training runs change. Tests whether keeping CONFIDENT samples
        # (the published default) is actually better than keeping the hard ones.
        for mode in ("none", "keep_uncertain", "random_match"):
            print(f"\n== filter ablation: {mode} ==")
            exp.cfg.filter_mode = mode
            exp.cfg.run_tag = f"{args.run_tag}filter={mode}"
            exp._filter_cache.clear()
            exp.filter_synthetic(plot=False)
            exp.run_synthetic()
        # Group by run_tag so each filter mode gets its own rows -- pooling modes
        # into one mean (the old behaviour) produced a meaningless average.
        led = exp.ledger
        led = led[led["arm"].isin(["S1", "S2", "S3"])]
        aggs = dict(auc_mean=("test_auc", "mean"), acc_mean=("test_acc", "mean"),
                    n_seeds=("test_auc", "count"))
        if "test_qwk" in led.columns:
            aggs["qwk_mean"] = ("test_qwk", "mean")
        abl = (led.groupby(["run_tag", "arm", "budget"]).agg(**aggs).reset_index())
        abl.to_csv(out / "filter_ablation.csv", index=False)
        print(abl.to_string(index=False))

    print("\nDONE. Outputs ->", out)


if __name__ == "__main__":
    main()
