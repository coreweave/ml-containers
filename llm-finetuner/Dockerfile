# syntax=docker/dockerfile:1.2

ARG BASE_IMAGE=ghcr.io/coreweave/ml-containers/torch:afecfe9-base-cuda11.8.0-torch2.0.0-vision0.15.1

FROM alpine/git:2.36.3 as downloader
WORKDIR /git

# Note: avoid using a branch name as the commit to prevent erroneous caching.
ARG COMMIT

# Download only the required files from the repository
RUN git clone --filter=blob:none --depth 1 --no-single-branch --no-checkout \
      https://github.com/coreweave/kubernetes-cloud && \
    cd kubernetes-cloud && \
    git sparse-checkout init && \
    git config --worktree core.sparseCheckoutCone false && \
    git config --worktree advice.detachedHead false && \
    git sparse-checkout set /finetuner-workflow/finetuner && \
    git checkout "${COMMIT}" && \
    rm -rf .git && \
    [ -d finetuner-workflow/finetuner ] && \
    ln -s $(realpath finetuner-workflow/finetuner) /src

WORKDIR /src

# Dependencies requiring NVCC are built ahead of time in a separate stage
# so that the ~2 GiB dev library installations don't have to be included
# in the final finetuner image.
FROM ${BASE_IMAGE} as builder
RUN apt-get install -y --no-install-recommends \
        cuda-nvcc-11-8 cuda-nvml-dev-11-8 libcurand-dev-11-8 \
        libcublas-dev-11-8 libcusparse-dev-11-8 \
        libcusolver-dev-11-8 cuda-nvprof-11-8 \
        cuda-profiler-api-11-8 \
        ninja-build \
        # gcc-10/g++-10/lld do not need to be installed here, but they improve the build.
        # gfortran-10 is just for compiler_wrapper.f95.
        gcc-10 g++-10 gfortran-10 lld && \
    apt-get clean && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 10 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 10 && \
    update-alternatives --install \
      /usr/bin/gfortran gfortran /usr/bin/gfortran-10 10 && \
    update-alternatives --install /usr/bin/ld ld /usr/bin/ld.lld 1

RUN mkdir /wheels /build
WORKDIR /build
COPY compiler_wrapper.f95 .
COPY --from=downloader /src/requirements-precompilable.txt .
RUN gfortran -O3 ./compiler_wrapper.f95 -o ./compiler && \
    python3 -m pip install -U --no-cache-dir \
      packaging setuptools wheel pip && \
    # Only DS_BUILT_UTILS and DS_BUILD_CPU_ADAM are necessary for the finetuner
    # See: https://www.deepspeed.ai/tutorials/advanced-install
    DS_BUILD_UTILS=1 DS_BUILD_CPU_ADAM=1 \
      # DeepSpeed forces -march=native into the compiler options,
      # making the result dependent on the processor architecture
      # used on the builder machine.
      # The compiler wrapper normalizes -march=native to -march=skylake
      # along with a couple other transformations before invoking GCC.
      CC=$(realpath -e ./compiler) \
      python3 -m pip wheel -w /wheels \
      --no-cache-dir --no-build-isolation --no-deps \
      -r requirements-precompilable.txt && \
    rm ./*

WORKDIR /wheels

FROM ${BASE_IMAGE}
RUN mkdir /app
WORKDIR /app
RUN --mount=type=bind,from=builder,source=/wheels,target=. \
    pip3 install --no-cache-dir ./*.whl
COPY --from=downloader /src/requirements.txt .
COPY --from=downloader /src/requirements-precompilable.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY --from=downloader /src/ds_config.json .
COPY --from=downloader /src/finetuner.py .
COPY --from=downloader /src/evaluator.py .
COPY --from=downloader /src/inference.py .
COPY --from=downloader /src/utils.py .
CMD [ "/usr/bin/python3", "finetuner.py" ]
