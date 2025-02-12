# syntax=docker/dockerfile:1.2

# This is a merged Jax + TensorFlow image that includes a new Triton build,
# compatible with the latest GPU architectures.


# CPU-only TensorFlow is missing packaging for new Python versions on aarch64
# So this uses TensorFlow with GPU features instead, but it has to be one
# compatible with new GPU architectures for that to work.
# Thus, this extracts the TensorFlow package from the nvcr TensorFlow image
# and turns it into a wheel that can be transferred to the nvcr Jax image.
FROM nvcr.io/nvidia/tensorflow:25.01-tf2-py3 AS tf2
WORKDIR /wheels
RUN --mount=type=bind,source=repackage.py,target=/tmp/repackage.py \
    python3 /tmp/repackage.py tensorflow --out-dir /wheels


# Alias stage, used for source builds and the final image.
FROM nvcr.io/nvidia/jax:25.01-py3 AS base


# Triton is not available on PyPI for aarch64, and a new commit is needed
# to support new GPU architectures anyway, so this is built from source.
FROM base AS builder
WORKDIR /wheels
WORKDIR /build

# The Triton build may not invoke NVCC; leaving this here just in case
ARG NVCC_APPEND_FLAGS='-gencode=arch=compute_100,code=[sm_100,compute_100] -gencode=arch=compute_100a,code=sm_100a --diag-suppress 174'

ARG TRITON_COMMIT='196a08f04b92fcf0e52015d3b1068c18e4eea5b5'
RUN python3 -m pip install -U --no-cache-dir \
      pip setuptools wheel build packaging cmake ninja pybind11 && \
    git clone --recursive --filter=blob:none https://github.com/triton-lang/triton && \
    cd triton && \
    git checkout "${TRITON_COMMIT}" && \
    python3 -m build -w -n -v -o /wheels python && \
    cd .. && \
    rm -r triton


FROM base

# Install the repackaged TensorFlow wheel
RUN --mount=type=bind,from=tf2,source=/wheels,target=/wheels \
    python3 -m pip install --no-cache-dir /wheels/*.whl

# Install the freshly-built Triton wheel
RUN --mount=type=bind,from=builder,source=/wheels,target=/wheels \
    python3 -m pip install --no-cache-dir /wheels/*.whl

# Install further dependencies
RUN --mount=type=bind,source=requirements.txt,target=/tmp/requirements.txt \
    python3 -m pip install --no-cache-dir -r /tmp/requirements.txt
