# Cluster image for the PneumoniaMNIST augmentation experiment.
#
# Build this only if you need it -- the public NGC image plus a runtime pip install
# works fine when the cluster has outbound internet. Use this image when:
#   * the compute nodes have NO internet (pip/git/Zenodo would all fail), or
#   * you want jobs to start instantly instead of installing deps each time, or
#   * you need a pinned, reproducible environment for the write-up.
#
# The repo is deliberately NOT baked in: project/runai/entrypoint.sh clones it at
# run time, so a job always runs current code and you never rebuild for a code change.
# If your nodes have no internet, see the OFFLINE section in project/runai/README.md.

FROM nvcr.io/nvidia/pytorch:24.12-py3

# The pipeline shells out to these (weight download, unpack, repo clone).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget unzip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Only what the NGC base lacks -- torch, CUDA and cuDNN already come from the base,
# so nothing here should pull a second torch. If pip tries to, pin it and rebuild.
COPY project/runai/requirements-cluster.txt /tmp/requirements-cluster.txt
RUN pip install --no-cache-dir -r /tmp/requirements-cluster.txt \
    && python -c "import torch, medmnist, torchdiffeq, diffusers, zuko, sklearn, scipy; \
                  print('deps OK, torch', torch.__version__, 'cuda', torch.version.cuda)"

WORKDIR /workspace
ENV MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1

# No ENTRYPOINT on purpose: Run:ai supplies the command, and entrypoint.sh is invoked
# explicitly so the same image serves batch runs, sweeps and interactive debugging.
