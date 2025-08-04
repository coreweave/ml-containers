# syntax=docker/dockerfile:1.2

ARG BASE_IMAGE=ghcr.io/coreweave/ml-containers/torch:es-fa3-te-update-1b2ab29-nccl-cuda12.8.1-ubuntu22.04-nccl2.27.5-1-torch2.7.1-vision0.22.1-audio2.7.1-abi1
# Dependencies requiring NVCC are built ahead of time in a separate stage
# so that the ~2 GiB dev library installations don't have to be included
# in the final finetuner image.
# gcc-10/g++-10/lld do not need to be installed here, but they improve the build.
# gfortran-10 is just for compiler_wrapper.f95.
FROM ${BASE_IMAGE} as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-nvcc-12-8 \
        cuda-nvml-dev-12-8 \
        libcurand-dev-12-8 \
        libcublas-dev-12-8 \
        libcusparse-dev-12-8 \
        libcusolver-dev-12-8 \
        cuda-profiler-api-12-8 \
        ninja-build \
        gcc g++ gfortran && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create and pre-compile wheels for DeepSpeed
RUN mkdir /wheels
WORKDIR /wheels
COPY requirements-precompilable.txt .
RUN python3 -m pip install -U --no-cache-dir wheel && \
    MAX_JOBS=16 DS_BUILD_UTILS=1 DS_BUILD_CPU_ADAM=1 \
      TORCH_CUDA_ARCH_LIST="8.0 9.0+PTX" \
      python3 -m pip wheel \
      --no-cache-dir --no-build-isolation --no-deps \
      -r requirements-precompilable.txt

# Final image stage
FROM ${BASE_IMAGE}

# Update package lists and install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    libglib2.0-0


RUN mkdir /app
WORKDIR /app

# Install the pre-compiled DeepSpeed wheel
RUN --mount=type=bind,from=builder,source=/wheels,target=/wheels \
    pip3 install --no-cache-dir /wheels/*.whl

# Install remaining Python requirements
COPY requirements.txt .
COPY requirements-precompilable.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY ds_config.json .
COPY finetuner.py .
COPY evaluator.py .
COPY inference.py .
COPY utils.py .

CMD [ "/usr/bin/python3", "finetuner.py" ]