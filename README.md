# When Are Synthetic Medical Images Worth Their Cost?

Code for the Deep Learning course project by **Ron Nekrashevich** and **Yuval Berkovich**
(Tel Aviv University, 2026).

The project evaluates generative augmentation with
[MedSymmFlow](https://github.com/caetas/MedSymmFlow), a symmetric flow-matching model that
generates class-conditional medical images and classifies by running its flow backwards.
We build a leakage-free evaluation protocol on RetinaMNIST: the training split is divided
once into a generator half and a classifier half, so no classifier is ever trained on an
image its generator saw, and any gain is attributable to the generator rather than to data
leakage. On top of that protocol we measure when generated images help, how many are worth
generating, how they compare with classical augmentation and with ImageNet pretraining,
and which changes to the model itself improve it.

## Repository layout

```text
src/medsymmflow/        the MedSymmFlow package (upstream code; our model changes are
                        confined to models/SymmFMClass.py)
project/                the evaluation system written for this project (see below)
pneumonia/              phase 1 of the project: PneumoniaMNIST experiments with the
                        published checkpoint, self-contained with its own README
requirements/           Python dependencies
setup.py, pyproject.toml
```

## What is ours and what is upstream

The generator architecture, the symmetric flow-matching loss, and the published
checkpoints come from [MedSymmFlow](https://github.com/caetas/MedSymmFlow) (MIT license),
whose UNet derives from [OpenAI guided diffusion](https://github.com/openai/guided-diffusion).
Everything under `project/` and `pneumonia/` was written for this project. Inside the
model itself we added about 250 lines to `src/medsymmflow/models/SymmFMClass.py`:
class-conditional generation and reverse-flow classification for K classes, one-hot and
thermometer class codes, an exchangeable autoencoder whose latent scale is read from its
own configuration, logit-normal timestep sampling, classifier-free guidance for the
coupled flows, and REPA/U-REPA representation alignment.

External components used unchanged: the end-to-end tuned VAE of
[REPA-E](https://github.com/End2End-Diffusion/REPA-E) (checkpoint `REPA-E/e2e-sdvae-hf`),
[DINOv2](https://github.com/facebookresearch/dinov2) and
[RETFound](https://github.com/rmaphoh/RETFound) as alignment teachers, and the alignment
losses of [REPA](https://github.com/sihyun-yu/REPA) and
[U-REPA](https://github.com/YuchuanTian/U-REPA). Published MedSymmFlow checkpoints are
from [Zenodo record 16086025](https://zenodo.org/records/16086025).

## The evaluation system (`project/`)

| File | Purpose |
| --- | --- |
| `augmentation.py` | the evaluation system: disjoint splits, training arms, results ledger, filters, statistics |
| `run_experiment.py` | command-line entry point for one full experiment |
| `data_split.py` | the disjoint generator/classifier split (stratified, seeded) |
| `datasets_meta.py` | dataset registry: classes, channels, splits |
| `filtering.py` | memorization screen and filter keys |
| `train_msf_scratch.py` | train a generator from scratch on a chosen subset |
| `train_msf_external.py` | train a generator on an external corpus (APTOS) |
| `generate_medmnist.py` | class-conditional generation for K classes |
| `classify_medmnist.py` | reverse-flow classification, ensemble readout |
| `score_synthetic.py` | the generator scores its own pool for the self-consistency filter |
| `oracle_baseline.py` | reference classifier trained directly on real data (protocol broken on purpose) |
| `pretrain_msf_chestmnist.py` | ChestMNIST pretraining for the pneumonia phase |
| `finetune_vae_aptos.py` | fine-tune the VAE on fundus images |
| `preprocess_aptos.py` | build APTOS archives at 28 and 256 px in MedMNIST format |
| `msf_arch.py` | infer the UNet architecture from a checkpoint |
| `gen_metrics.py` | FID, KID and t-SNE features of generated pools |
| `c1_metrics.py` | reverse-flow AUC and confidence under both conventions |
| `paired_stats.py` | paired t and Wilcoxon tests across seeds |
| `runai/` | cluster image, entry point and job documentation |

## Reproducing the results

1. **Install.**

   ```bash
   pip install -r requirements/requirements.txt
   pip install -e .
   ```

   MedMNIST datasets download automatically on first use. APTOS 2019 is public on
   [Kaggle](https://www.kaggle.com/c/aptos2019-blindness-detection); run
   `python project/preprocess_aptos.py` to pack it into MedMNIST format.

2. **Train a generator on the disjoint split** (~10 GPU-hours at 28 px on an RTX A5000):

   ```bash
   python project/train_msf_scratch.py --dataset retinamnist --gen-frac 0.75 --split-seed 0
   ```

3. **Run the main experiment.** One command trains every arm at every label budget and
   appends each result to a ledger keyed by setting, budget, seed, filter and run, so an
   interrupted job resumes at the cost of one cell:

   ```bash
   python project/run_experiment.py --dataset retinamnist --checkpoint <generator.pt>
   ```

4. **Cluster runs.** `project/runai/README.md` documents every job of the study as the
   exact commands submitted to a Run:AI cluster, from the smoke test to the full grid.

Splits are stratified with seed 0, label subsets use seeds 0 to 9, and each split's
fingerprint is stored in every checkpoint manifest so a run can be checked against it.

## Phase 1: PneumoniaMNIST (`pneumonia/`)

The project began on PneumoniaMNIST with the published checkpoint, before the disjoint
protocol existed. That phase, its notebooks, and its compact results are preserved
unchanged in [`pneumonia/`](pneumonia/), with its own README. Its findings, that the gain
appears only where real data is scarce and that same-data training makes the source of
the gain unattributable, are what motivated the disjoint protocol used everywhere else.
