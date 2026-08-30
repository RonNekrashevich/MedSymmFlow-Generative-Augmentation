# Can MedSymmFlow-Generated Chest X-Rays Improve Pneumonia Classification?

Focused code submission for the *Deep Learning in Medical Imaging* course project (2026), by Yuval Berkovich and Ron Nekrashevich.

This repository contains only the code changes, analysis notebooks, compact results, and dependency list needed for reviewing the project. Large datasets, generated image pools, checkpoints, and unchanged upstream MedSymmFlow source files are intentionally excluded.

## Start here

1. [src/medsymmflow/models/SymmFMClass.py](src/medsymmflow/models/SymmFMClass.py) contains the modification to `SymmFMClass.sample`. The optional `labels` argument converts requested Normal/Pneumonia labels into class-conditioning masks during sampling.
2. [project/generate_pneumoniamnist.py](project/generate_pneumoniamnist.py) is the reproducible class-conditioned PneumoniaMNIST generation script.
3. [notebooks/01_main_experiments.ipynb](notebooks/01_main_experiments.ipynb) documents the downstream ResNet-18 experiments and augmentation controls.
4. [notebooks/02_fid_kid_nearest_neighbor.ipynb](notebooks/02_fid_kid_nearest_neighbor.ipynb) contains the FID, KID, and nearest-neighbor analyses.
5. [notebooks/03_tsne_analysis.ipynb](notebooks/03_tsne_analysis.ipynb) contains the t-SNE analyses.
6. [results](results) contains the compact CSV, JSON, and figure outputs used in the report.

## Repository layout

```text
README.md
project/generate_pneumoniamnist.py
src/medsymmflow/models/SymmFMClass.py
notebooks/
  01_main_experiments.ipynb
  02_fid_kid_nearest_neighbor.ipynb
  03_tsne_analysis.ipynb
results/
  final/
  fid/
  tsne/
  figures/
requirements/submission.txt
```

## Reproduction note

This is a focused review repository rather than a standalone copy of the full framework. To rerun image generation, use the two submitted Python files with the [official MedSymmFlow implementation](https://github.com/caetas/MedSymmFlow), its released `FM_pneumoniamnist_beta4.0_rgb.pt` checkpoint, and the packages in `requirements/submission.txt`. The complete original development fork is available at https://github.com/yovalyoval10-rgb/MedSymmFlow.

The final generation configuration was: non-latent model, beta `4.0`, RGB class-conditioning mask, `32 x 32` output, Euler solver, step size `0.04`, and labels `0 = Normal`, `1 = Pneumonia`.

The notebooks can be opened directly in Google Colab. Image datasets, generated pools, and model checkpoints are not stored in this repository.

## Results map

- `results/final/`: classifier mixtures, low-data experiments, repeated-real control, and filtered-versus-random control.
- `results/fid/`: FID, KID, and nearest-neighbor summaries.
- `results/tsne/`: exported t-SNE coordinates.
- `results/figures/`: final report figures, including figures based on Ron's separately submitted experiments.
