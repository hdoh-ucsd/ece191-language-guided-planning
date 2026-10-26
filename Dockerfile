# syntax=docker/dockerfile:1.7
# NVIDIA RTX 50-series (Blackwell / sm_120) base for PyTorch + Valeo4Cast MTR.
# The Windows/WSL NVIDIA driver is supplied to the *running container* by Docker;
# do not install a Linux display/driver package in this image.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 AS gpu

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="12.0" \
    MAX_JOBS=2

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev python3-venv \
    ca-certificates build-essential ninja-build cmake git \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade 'pip<26' 'setuptools<81' wheel \
    && python3 -m pip install --no-cache-dir \
       'torch==2.7.1' 'torchvision==0.22.1' \
       --index-url https://download.pytorch.org/whl/cu128

COPY scripts/gpu_smoke.py /opt/mtr-scripts/gpu_smoke.py
WORKDIR /workspace
CMD ["python3", "/opt/mtr-scripts/gpu_smoke.py"]

# The mtr target adds Valeo4Cast and compiles its custom CUDA ops for sm_120.
# Build this only AFTER `docker compose run --rm gpu` passes.
FROM gpu AS mtr

ARG VALEO_COMMIT=dce207347c140ed5cdaeac6c217eb00ee0634e71
ENV PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PYOPENGL_PLATFORM=egl

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libegl1 libgomp1 libspatialindex-dev \
    libsm6 libxext6 libxrender1 libgeos-dev libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Pin the source revision so downstream source compatibility patches are reproducible.
RUN git init /opt/valeo4cast \
    && git -C /opt/valeo4cast remote add origin https://github.com/valeoai/valeo4cast.git \
    && git -C /opt/valeo4cast fetch --depth=1 origin "${VALEO_COMMIT}" \
    && git -C /opt/valeo4cast checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/valeo4cast rev-parse HEAD)" = "${VALEO_COMMIT}"

COPY scripts/ /opt/mtr-scripts/

# Older code calls at::Tensor.data<T>(), which is deprecated in modern PyTorch.
# The patch replaces that API without changing the operators' mathematics.
RUN python3 /opt/mtr-scripts/modernize_cuda_cpp.py /opt/valeo4cast/UniTraj/unitraj/models/mtr/ops

# Our earlier research adapter handles known discrepancies in the nuPlan ScenarioNet
# metadata and Valeo4Cast data loader; it does NOT implement a complete NuPlan port.
RUN python3 /opt/mtr-scripts/patch_valeo4cast_for_nuplan.py \
        --repo /opt/valeo4cast --apply

# Keep Blackwell CUDA wheels and NumPy 1.x while installing legacy project's packages.
RUN python3 -m pip install --no-cache-dir \
      -c /opt/mtr-scripts/constraints.txt \
      'numpy==1.24.2' 'pytorch-lightning==2.2.5' \
      'torchmetrics==1.4.3' 'hydra-core==1.3.2' 'lap' \
      'opencv-python-headless<4.11' 'metadrive-simulator==0.4.1.2' 'matplotlib'

RUN python3 -m pip install --no-cache-dir \
      -c /opt/mtr-scripts/constraints.txt \
      -r /opt/valeo4cast/UniTraj/requirements.txt \
    && python3 -m pip install --no-cache-dir --no-build-isolation \
      -c /opt/mtr-scripts/constraints.txt \
      -e /opt/valeo4cast/scenarionet

# Compiles KNN and local-attention ops. NVCC from CUDA 12.8 targets sm_120 even though
# no GPU is exposed during docker build (the GPU is supplied when containers run).
RUN cd /opt/valeo4cast/UniTraj \
    && python3 -m pip install --no-cache-dir --no-build-isolation --no-deps -e .

# UniTraj/unitraj has no __init__.py, so the PEP 660 editable finder cannot resolve
# `unitraj.models.*`. Upstream relied on `setup.py develop` putting the project root on
# sys.path; a plain .pth restores that so `unitraj` resolves as a namespace package.
RUN echo /opt/valeo4cast/UniTraj > /usr/local/lib/python3.10/dist-packages/unitraj_src.pth \
    && cd /opt/valeo4cast/UniTraj/unitraj \
    && python3 -c 'import torch, unitraj.models, unitraj.datasets; print("unitraj imports OK")'

WORKDIR /opt/valeo4cast/UniTraj/unitraj
CMD ["bash"]
