#!/usr/bin/env bash
# Run:ai job entrypoint for the PneumoniaMNIST augmentation experiment.
#
# Everything persistent lives under $DATA_ROOT (your mounted volume):
#     $DATA_ROOT/weights   the 755 MB MedSymmFlow archive, downloaded ONCE
#     $DATA_ROOT/pipcache  pip wheel cache, so later jobs start faster
#     $DATA_ROOT/runs/$RUN_NAME   results.csv ledger, figures, models, caches
#
# The container filesystem is treated as disposable: the repo is cloned fresh each
# job, so a job always runs the current code.
#
# Usage inside a Run:ai job:
#     bash project/runai/entrypoint.sh --seeds 0 1 2 3 4 --budgets 250 500 1000
# Any arguments are passed straight through to project/run_experiment.py.
set -euo pipefail

: "${DATA_ROOT:?set DATA_ROOT to YOUR OWN folder, e.g. -e DATA_ROOT=/storage/<your-user>/medsymm}"
REPO_URL="${REPO_URL:-https://github.com/RonNekrashevich/MedSymmFlow-Generative-Augmentation.git}"
REPO_DIR="${REPO_DIR:-/workspace/MedSymmFlow}"
REPO_REF="${REPO_REF:-main}"    # branch to run
RUN_NAME="${RUN_NAME:-run}"

# ---- shared-storage safety -----------------------------------------------------
# This volume is shared with the rest of the lab. Nothing here ever deletes a path
# it did not create, but a mistyped DATA_ROOT could still scatter files across a
# shared root, so refuse anything that is not clearly a personal subfolder.
DATA_ROOT="${DATA_ROOT%/}"
depth=$(printf '%s' "$DATA_ROOT" | awk -F/ '{print NF-1}')
if [ "$depth" -lt 2 ]; then
  echo "REFUSING TO RUN: DATA_ROOT='$DATA_ROOT' is too shallow — it looks like a shared mount root." >&2
  echo "Use a personal subfolder at least two levels deep, e.g. /storage/\$USER/medsymm" >&2
  exit 1
fi
if command -v mountpoint >/dev/null 2>&1 && mountpoint -q "$DATA_ROOT"; then
  echo "REFUSING TO RUN: DATA_ROOT='$DATA_ROOT' is itself a mount point. Use a subfolder." >&2
  exit 1
fi
if [ -d "$DATA_ROOT" ] && [ -n "$(find "$DATA_ROOT" -maxdepth 1 -mindepth 1 \
      ! -name weights ! -name pipcache ! -name runs ! -name hf \
      ! -name external -print -quit 2>/dev/null)" ]; then
  echo "REFUSING TO RUN: '$DATA_ROOT' already contains files this project did not create." >&2
  echo "That usually means it is somebody else's folder. Contents:" >&2
  ls -la "$DATA_ROOT" >&2
  echo "If this really is yours, set DATA_ROOT to an empty subfolder instead." >&2
  exit 1
fi

echo "=== Run:ai job: $RUN_NAME ==="
echo "DATA_ROOT=$DATA_ROOT"
echo "this job will only ever create/write:"
echo "    $DATA_ROOT/weights/      (generator weights, downloaded once)"
echo "    $DATA_ROOT/pipcache/     (pip wheels)"
echo "    $DATA_ROOT/hf/           (huggingface caches)"
echo "    $DATA_ROOT/runs/$RUN_NAME/  (results, figures, models)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "WARNING: no GPU visible"

mkdir -p "$DATA_ROOT/weights" "$DATA_ROOT/pipcache" "$DATA_ROOT/runs/$RUN_NAME"
export PIP_CACHE_DIR="$DATA_ROOT/pipcache"
export HF_HOME="$DATA_ROOT/hf"          # keeps diffusers/datasets caches off the container
export MPLBACKEND=Agg                   # headless matplotlib

# ---- code: always current -----------------------------------------------------
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" fetch -q --all && git -C "$REPO_DIR" reset -q --hard "origin/$REPO_REF"
else
  git clone -q --branch "$REPO_REF" "$REPO_URL" "$REPO_DIR"
fi
echo "repo at $(git -C "$REPO_DIR" rev-parse --short HEAD) [$REPO_REF]"

# ---- deps: torch comes from the image, these do not ---------------------------
# "numpy<2" is load-bearing: the NGC image's torch/torchvision are compiled against
# NumPy 1.x, and without the pin the resolver upgrades to 2.x ("Numpy is not
# available" crashes in every DataLoader).
python -m pip install -q --no-input \
  "numpy<2" medmnist torchdiffeq diffusers accelerate zuko scikit-learn scipy \
  loguru python-dotenv

# ---- run ----------------------------------------------------------------------
cd "$REPO_DIR"
exec python project/run_experiment.py \
  --out "$DATA_ROOT/runs/$RUN_NAME" \
  --weights-root "$DATA_ROOT/weights" \
  "$@"
