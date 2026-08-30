# Running on Run:ai

Everything is parameterised by two variables, so you only edit them once:

| Variable | What it is | Example |
|---|---|---|
| `PVC` | your Run:ai persistent volume claim | `my-lab-pvc` |
| `DATA_ROOT` | **your own** folder inside the mounted volume — never the mount root | `/storage/<your-user>/medsymm` |

---

## 0. Find your storage (do this first)

Nothing else works until you know which volume you get and where it lands.

```bash
runai submit probe -i ubuntu --command -- bash -c "df -h; echo ---; ls -la /; echo ---; mount | grep -Ev 'proc|sys|cgroup|dev'"
runai logs probe
runai delete job probe
```

Look for a large mount that is **not** `overlay` or `tmpfs` — that is your persistent storage.
If nothing is mounted by default, list the claims you can attach:

```bash
runai list projects
kubectl get pvc            # if you have kubectl access
```

Then set, for example, `PVC=my-lab-pvc` and `DATA_ROOT=/storage/<your-user>/medsymm`.

---

## 1. Smoke test (~10 minutes, 1 GPU)

Confirms the image, the volume, the weights download and the pipeline all work.

```bash
runai submit medsymm-smoke \
  -i nvcr.io/nvidia/pytorch:24.12-py3 \
  -g 1 --large-shm \
  --pvc my-lab-pvc:/storage \
  -e DATA_ROOT=/storage/<your-user>/medsymm \
  -e RUN_NAME=smoke \
  --command -- bash -c \
  "git clone -q https://github.com/RonNekrashevich/MedSymmFlow-Generative-Augmentation.git /workspace/m && bash /workspace/m/project/runai/entrypoint.sh --quick"
```

```bash
runai logs medsymm-smoke -f
```

You want to see, in order: a GPU name, `Split sizes OK`, `selftest_repro`, the baselines,
`Checkpoint present`, the arms, and `DONE`.

**The first job downloads 755 MB of weights. Every later job reuses them** — that is the
whole point of putting `--weights-root` on the volume.

---

## 2. The real run (5 budgets x 5 seeds)

```bash
runai submit medsymm-full \
  -i nvcr.io/nvidia/pytorch:24.12-py3 \
  -g 1 --large-shm \
  --pvc my-lab-pvc:/storage \
  -e DATA_ROOT=/storage/<your-user>/medsymm \
  -e RUN_NAME=full-5seed \
  --command -- bash -c \
  "git clone -q https://github.com/RonNekrashevich/MedSymmFlow-Generative-Augmentation.git /workspace/m && bash /workspace/m/project/runai/entrypoint.sh --seeds 0 1 2 3 4 --budgets 250 500 1000 2000 4708"
```

Resume is automatic: results are appended to a ledger keyed by
`(arm, budget, seed, filter_key)`, so a job that is pre-empted or killed can simply be
re-submitted with the same `RUN_NAME` and it continues where it stopped.

---

## 2b. Disjoint generator — train MSF from scratch, no shared data

The published checkpoint saw the whole 4708-image train split, i.e. the same images
the ResNet trains on. `--gen-frac` removes that confound: a stratified, seeded split
(`data_split.py`, `split_seed` independent of `--seeds`) gives the generator its own
half, MSF is trained **from scratch** on it (`train_msf_scratch.py`, same RGB_28
architecture/beta), and every classifier arm — baselines, synthetic arms, filter
scorers — draws only from the other half.

With `--gen-frac 0.5` the classifier pool is 2354, so budgets must be ≤ 2354
(the run refuses otherwise). Generator training (~600 epochs) runs once per
`weights_root` and is cached as
`weights/scratch/FM_pneumoniamnist_scratch_g0.5_ss0_beta4.0_rgb.pt` + a `.json`
manifest recording the split.

```bash
# smoke first (~30 min: 20-epoch generator + quick pipeline)
runai submit medsymm-disjoint-smoke \
  -i nvcr.io/nvidia/pytorch:24.12-py3 -g 1 --large-shm \
  --pvc my-lab-pvc:/storage \
  -e DATA_ROOT=/storage/<your-user>/medsymm -e RUN_NAME=disjoint-smoke \
  --command -- bash -c \
  "git clone -q https://github.com/RonNekrashevich/MedSymmFlow-Generative-Augmentation.git /workspace/m && bash /workspace/m/project/runai/entrypoint.sh --quick --gen-frac 0.5"

# the real run (budgets capped at the 2354-image classifier pool)
runai submit medsymm-disjoint \
  -i nvcr.io/nvidia/pytorch:24.12-py3 -g 1 --large-shm \
  --pvc my-lab-pvc:/storage \
  -e DATA_ROOT=/storage/<your-user>/medsymm -e RUN_NAME=disjoint-full \
  --command -- bash -c \
  "git clone -q https://github.com/RonNekrashevich/MedSymmFlow-Generative-Augmentation.git /workspace/m && bash /workspace/m/project/runai/entrypoint.sh --gen-frac 0.5 --seeds 0 1 2 3 4 --budgets 250 500 1000 2354 --run-tag disjoint"
```

The smoke run's generator (20 epochs) produces poor images — it only proves the
plumbing. That's fine: the checkpoint filename is keyed by the full training
recipe (epochs, lr, dropout, balance, pretrain), so no two recipes ever share a
cached checkpoint. Keep disjoint and non-disjoint results in separate `RUN_NAME`s
and compare via the `gen_frac` column the ledger now records.

### Squeezing the most out of 2354 images

We are not bound to the paper's recipe (1000 epochs, lr 5e-4 on the FULL split) —
with half the data the goal is the best generator we can get. Extra knobs, all
keyed into the checkpoint cache:

- `--gen-epochs 1500` — epochs are ~19 steps each, so long schedules are cheap.
- `--gen-balance` — 50/50 class sampling; the data is 74% pneumonia, so the
  normal-class mask conditioning otherwise trains on 4x fewer examples.
- `--gen-dropout 0.1` — regularisation for the halved dataset (paper used 0.0
  on the full one); also reduces memorisation, which the filter screens for.
- `--gen-pretrain-epochs 60` — pretrain on ChestMNIST first (~43k NIH CXRs:
  42k no-finding + 1k pneumonia-positive, other findings excluded), then
  fine-tune on the pneumonia gen half. The pretrain checkpoint is cached once
  per `weights_root` and shared by every recipe that requests it.
  **Do not** instead mix ChestMNIST into the labeled training data: it is adult
  imaging while PneumoniaMNIST is pediatric, and enriching one class with a
  different domain teaches the model "adult film = that class".

```bash
runai submit medsymm-disjoint-best \
  -i nvcr.io/nvidia/pytorch:24.12-py3 -g 1 --large-shm \
  --pvc my-lab-pvc:/storage \
  -e DATA_ROOT=/storage/<your-user>/medsymm -e RUN_NAME=disjoint-best \
  --command -- bash -c \
  "git clone -q https://github.com/RonNekrashevich/MedSymmFlow-Generative-Augmentation.git /workspace/m && bash /workspace/m/project/runai/entrypoint.sh --gen-frac 0.5 --gen-epochs 1500 --gen-balance --gen-dropout 0.1 --gen-pretrain-epochs 60 --seeds 0 1 2 3 4 --budgets 250 500 1000 2354 --run-tag disjoint-best"
```

To find the best recipe, run 2-4 of these in parallel with different generator
knobs and separate `RUN_NAME`s/`--run-tag`s, then compare **val** AUC of the
synthetic arms across ledgers (never select the generator on test AUC).

---

## 3. Parallel sweeps — one job per configuration

This is what the cluster is actually for. Each job writes to its own `RUN_NAME`, so they
never collide; combine the ledgers afterwards.

```bash
# filter-direction ablation: is keeping CONFIDENT samples backwards?
for mode in keep_confident keep_uncertain random_match none; do
  runai submit "medsymm-filter-$mode" \
    -i nvcr.io/nvidia/pytorch:24.12-py3 -g 1 --large-shm \
    --pvc my-lab-pvc:/storage \
    -e DATA_ROOT=/storage/<your-user>/medsymm -e "RUN_NAME=filter-$mode" \
    --command -- bash -c \
    "git clone -q https://github.com/RonNekrashevich/MedSymmFlow-Generative-Augmentation.git /workspace/m && bash /workspace/m/project/runai/entrypoint.sh --seeds 0 1 2 3 4 --budgets 500 --filter-mode $mode --run-tag filter=$mode"
done

# beta sweep: does sharper class conditioning help downstream?
for b in 1 2 4 6; do
  runai submit "medsymm-beta$b" \
    -i nvcr.io/nvidia/pytorch:24.12-py3 -g 1 --large-shm \
    --pvc my-lab-pvc:/storage \
    -e DATA_ROOT=/storage/<your-user>/medsymm -e "RUN_NAME=beta-$b" \
    --command -- bash -c \
    "git clone -q https://github.com/RonNekrashevich/MedSymmFlow-Generative-Augmentation.git /workspace/m && bash /workspace/m/project/runai/entrypoint.sh --seeds 0 1 2 --budgets 500 --beta $b --run-tag beta=$b"
done
```

Each `beta` job regenerates its own synthetic pool, so give them separate `RUN_NAME`s —
they must not share a synthetic directory.

---

## 4. Collecting results

Ledgers are plain CSV. Concatenate and analyse them anywhere, including on your laptop:

```python
import pandas as pd, glob
df = pd.concat([pd.read_csv(f) for f in glob.glob("/storage/<your-user>/medsymm/runs/*/results.csv")])
df.to_csv("all_results.csv", index=False)

import sys; sys.path.insert(0, "project")
from paired_stats import paired_tests_from_csv
paired_tests_from_csv(df, select={"run_tag": ""})   # select= avoids mixing configurations
```

`paired_tests_from_csv` raises on duplicate `(arm, budget, seed)` rows rather than silently
averaging them, so pass `select=` to pick one configuration at a time.

---

## Shared-volume safety

The volume is shared with the rest of the lab, so the entrypoint refuses to start if
`DATA_ROOT` is a mount root, is a mount point, or already contains files this project did
not create — it prints the directory listing and exits rather than writing into somebody
else's folder. Point it at an empty personal folder such as `/storage/$USER/medsymm`.

Audited: the only deletions anywhere in the project are `rmtree` on
`<run>/scratch/_gen_chunk_N`, temporary directories the generation step creates itself.
Nothing removes a path it did not create, and nothing writes outside `$DATA_ROOT`.

## Building your own image (optional)

The runtime `pip install` in the entrypoint is fine **when the compute nodes have outbound
internet**. Build the image instead when they do not, or when you want jobs to start
instantly.

```bash
docker build -f project/runai/Dockerfile -t <registry>/medsymm:1 .
docker push <registry>/medsymm:1
```

Then swap the image in any submit command:

```bash
runai submit medsymm-full -i <registry>/medsymm:1 -g 1 ...
```

The repo is *not* baked into the image — the entrypoint still clones it — so a code change
never requires a rebuild. Rebuild only when dependencies change.

### OFFLINE clusters (no outbound internet on the nodes)

Three things need internet and all three have an offline route:

| Needs internet | Offline route |
|---|---|
| `pip install` | baked into the image (build it) |
| `git clone` of this repo | bake it in too: add `COPY . /opt/medsymm` and set `REPO_DIR=/opt/medsymm` |
| 755 MB Zenodo weights | download once on your laptop, copy `models.zip` into `$DATA_ROOT/weights/`; the entrypoint unpacks rather than downloads |

Test whether the nodes have internet before assuming:

```bash
runai submit nettest -i ubuntu --command -- bash -c "curl -sSI -m 10 https://pypi.org | head -1 || echo NO_INTERNET"
```

## Notes and gotchas

- **Image.** `nvcr.io/nvidia/pytorch:24.12-py3` already contains CUDA PyTorch; the entrypoint
  installs only the small extras. If your cluster blocks NGC, `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime`
  works too.
- **`--pvc` syntax** varies by Run:ai version: newer builds use
  `--existing-pvc claimname=my-lab-pvc,path=/storage`. Check `runai submit --help`.
- **One GPU is enough.** Nothing here is distributed; parallelism comes from running many
  single-GPU jobs, not from multi-GPU jobs.
- **Interactive debugging:** `runai submit -i nvcr.io/nvidia/pytorch:24.12-py3 -g 1 --large-shm --interactive --attach --command -- bash`
- **Connectivity.** The cluster API lives on a university-internal address, so you must be
  on campus or connected to the VPN. Check with `curl -k -m 5 https://<api-host>:6443/version`
  before debugging anything else; a timeout there means the network, not Run:ai.
- **The weights download is the slowest first step.** If the cluster has no outbound internet,
  copy `models.zip` onto the volume manually into `$DATA_ROOT/weights/` and the entrypoint
  will unpack it instead of downloading.
