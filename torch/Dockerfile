# syntax=docker/dockerfile:1.10
ARG BUILDER_BASE_IMAGE="nvidia/cuda:13.2.0-devel-ubuntu22.04"
ARG FINAL_BASE_IMAGE="nvidia/cuda:13.2.0-base-ubuntu22.04"

ARG BUILD_TORCH_VERSION="2.11.0"
ARG BUILD_TORCH_VISION_VERSION="0.26.0"
ARG BUILD_TORCH_AUDIO_VERSION="2.11.0"
ARG BUILD_TRANSFORMERENGINE_VERSION="2.12"
ARG BUILD_FLASH_ATTN_VERSION="2.8.3"
ARG BUILD_FLASH_ATTN_3_VERSION="1ceaa984b2f348caea18b39a98458d33b4ea7a09"
ARG BUILD_FLASH_ATTN_4_VERSION="e065bf1ba1091040eeaa9328846c3bd31c718543"
ARG BUILD_TRITON_VERSION=""
ARG BUILD_TRITON="1"
ARG BUILD_TORCH_CUDA_ARCH_LIST="8.0 8.9 9.0 10.0 12.0+PTX"
ARG BUILD_TRANSFORMERENGINE_CUDA_ARCH_LIST="80;89;90;100;120"

ARG AOCL_BASE="/opt/aocl"
ARG AOCL_VER="4.2.0"
ARG AOCL_URL="https://download.amd.com/developer/eula/aocl/aocl-4-2/aocl-linux-aocc-4.2.0.tar.gz"

ARG SCCACHE_VERSION="0.14.0"

# region Downloads

FROM alpine/curl:8.17.0 AS sccache-downloader
ARG SCCACHE_VERSION
RUN ARCH=$(uname -m) && \
    curl -fsSL "https://github.com/mozilla/sccache/releases/download/v${SCCACHE_VERSION}/sccache-v${SCCACHE_VERSION}-${ARCH}-unknown-linux-musl.tar.gz" \
    | tar xz --strip-components=1 -C /opt/ "sccache-v${SCCACHE_VERSION}-${ARCH}-unknown-linux-musl/sccache" && \
    chmod 755 /opt/sccache

# Clone PyTorch repositories independently from all other build steps
# for cache-friendliness and parallelization
FROM alpine/git:2.40.1 AS downloader-base
WORKDIR /git
RUN git config --global advice.detachedHead false

COPY <<-"EOT" /git/clone.sh
	#!/bin/sh
	REPO="https://github.com/$1";
	DEST="$2";
	REF="$3";

	CLONE() { git clone -j8 --depth=1 --filter=blob:none "$@"; };

	# Try cloning REF as a tag prefixed with "v", otherwise fall back
	# to git checkout for commit hashes
	CLONE --recurse-submodules --shallow-submodules --also-filter-submodules --no-tags \
	  "$REPO" -b "v$REF" "$DEST" || { \
	    CLONE --no-single-branch --no-checkout "$REPO" "$DEST" && \
	    git -C "$DEST" checkout "$REF" && \
	    git -C "$DEST" submodule update --init --filter=blob:none --depth=1 --recursive --jobs 8; \
	};
EOT

RUN chmod 755 /git/clone.sh


FROM downloader-base AS pytorch-downloader
ARG BUILD_TORCH_VERSION
# Includes a patch for a foreach bug in PyTorch v2.5.1
RUN ./clone.sh pytorch/pytorch pytorch "${BUILD_TORCH_VERSION}" && \
    if [ "${BUILD_TORCH_VERSION}" = '2.5.1' ]; then \
      wget 'https://github.com/pytorch/pytorch/commit/1cdaf1d85f5e4b3f8952fd0737a1afeb16995d13.patch' -qO- \
      | git -C pytorch apply; \
    elif [ "${BUILD_TORCH_VERSION}" = '2.7.1' ]; then \
      git -C pytorch cherry-pick -n b74be524547f7f025066f19eda3b53a887c244ba; \
    elif [ "${BUILD_TORCH_VERSION}" = '2.9.1' ]; then \
      git -C pytorch cherry-pick -n a20afb61007a94f5c28294e9ae20043657152ef6; \
    fi && \
    rm -rf pytorch/.git

FROM downloader-base AS torchvision-downloader
ARG BUILD_TORCH_VISION_VERSION
RUN ./clone.sh pytorch/vision vision "${BUILD_TORCH_VISION_VERSION}" && \
    rm -rf vision/.git

FROM downloader-base AS torchaudio-downloader
ARG BUILD_TORCH_AUDIO_VERSION
RUN ./clone.sh pytorch/audio audio "${BUILD_TORCH_AUDIO_VERSION}"
# The torchaudio build requires that this directory remain a full git repository,
# so no rm -rf audio/.git is done for this one.

# torchaudio is broken for CUDA 12.5+ without this patch (up to and including v2.5.0)
# See https://github.com/pytorch/audio/pull/3811
# Fixed as a side effect of https://github.com/pytorch/audio/pull/3843 in versions after v2.5.0
COPY torchaudio-cu125-pr3811.patch /git/patch
RUN if grep -qF '#include <float.h>' \
      'audio/src/libtorchaudio/cuctc/src/ctc_prefix_decoder_kernel_v2.cu'; \
      then :; else git -C audio apply -v --stat --apply /git/patch; \
    fi && \
    rm /git/patch

FROM downloader-base AS transformerengine-downloader
ARG BUILD_TRANSFORMERENGINE_VERSION
RUN ./clone.sh NVIDIA/TransformerEngine TransformerEngine "${BUILD_TRANSFORMERENGINE_VERSION}"
# Make TE compatible with both the FA3 beta and the official
# flash-attn-3 3.0.0 wheels. See te-fa3-api-compat.patch for details.
COPY te-fa3-api-compat.patch /git/te-fa3-api-compat.patch
RUN git -C TransformerEngine apply -v --stat --apply /git/te-fa3-api-compat.patch

FROM downloader-base AS flash-attn-downloader
ARG BUILD_FLASH_ATTN_VERSION
# Cherry-pick a patch for FA2 to coexist with FA4, cf. flash-attention#2331
RUN ./clone.sh Dao-AILab/flash-attention flash-attention "${BUILD_FLASH_ATTN_VERSION}" && \
    git -C flash-attention cherry-pick -n 4219765dfdd8913bfe26134f748dd5ffcedd3c39

FROM downloader-base AS flash-attn-3-downloader
ARG BUILD_FLASH_ATTN_3_VERSION
RUN if [ -n "$BUILD_FLASH_ATTN_3_VERSION" ]; then \
      ./clone.sh Dao-AILab/flash-attention flash-attention "${BUILD_FLASH_ATTN_3_VERSION}" && \
      git -C flash-attention cherry-pick -n 3edf7e0daa62662cd2dd2ec8fd999dd7f254415c && \
      git -C flash-attention cherry-pick -n afc97c60f799e470886c154e3473df938f8fa93d; \
    else \
      mkdir flash-attention; \
    fi

FROM downloader-base AS triton-version
ENV TRITON_COMMIT_FILE='.ci/docker/ci_commit_pins/triton.txt'
COPY --link --from=pytorch-downloader "/git/pytorch/${TRITON_COMMIT_FILE}" /git/version.txt
ARG BUILD_TRITON_VERSION
RUN if [ -n "${BUILD_TRITON_VERSION}" ]; then \
      echo "${BUILD_TRITON_VERSION}" > /git/version.txt; \
    fi

FROM downloader-base AS triton-downloader
COPY --link --from=triton-version /git/version.txt /git/version.txt
ARG BUILD_TRITON
RUN if [ "${BUILD_TRITON}" = '1' ]; then \
      ./clone.sh openai/triton triton "$(cat /git/version.txt)"; \
    else \
      mkdir triton; \
    fi

FROM alpine/curl:8.7.1 AS aocl-downloader
WORKDIR /tmp/install

RUN apk add --no-cache bash

ARG AOCL_BASE
ARG AOCL_VER
ARG AOCL_URL

RUN curl -sSfo- "${AOCL_URL}" | tar xzf - --strip-components 1 && \
    INSTALL_LIB() { ./install.sh -l "$1" -t "${AOCL_BASE}" -i lp64; } && \
    INSTALL_LIB blis && \
    INSTALL_LIB libflame && \
    INSTALL_LIB utils && \
    . ./amd-libs.cfg && \
    rm -r "${AOCL_ROOT}/include_ILP64" && \
    rm -r "${AOCL_ROOT}/lib_ILP64" && \
    ln -s "${AOCL_ROOT}/amd-libs.cfg" "${AOCL_BASE}/amd-libs.cfg" && \
    ln -s "${AOCL_ROOT}/include" "${AOCL_BASE}/include" && \
    ln -s "${AOCL_ROOT}/lib" "${AOCL_BASE}/lib" && \
    echo "${AOCL_BASE}/lib" \
    | install -m 0644 /dev/stdin "${AOCL_BASE}/aocl.conf" && \
    rm -r ./*

# endregion Downloads

## Build PyTorch on a builder image.
FROM ${BUILDER_BASE_IMAGE} AS builder-base-shared
ENV DEBIAN_FRONTEND=noninteractive

# ninja-build and lld are optional but improve the build.
# Remove PEP 668 marker since containers are often single-purpose
# and thus do not inherently need to be pushed into venv isolation.
RUN if [ "$(. /etc/os-release && echo "$VERSION_ID")" = '22.04' ]; then \
      PACKAGES='libncurses5 python3-pip python3-distutils'; \
    else \
      PACKAGES='libncurses6 python3-dev'; \
    fi && \
    apt-get -qq update && apt-get -qq install -y \
      $PACKAGES python3 git apt-utils ssh ca-certificates \
      libomp5 libpng-dev libjpeg-dev pkg-config \
      build-essential ninja-build && \
    apt-get clean && \
    rm -f /usr/lib/python3.*/EXTERNALLY-MANAGED && \
    if ! command -v pip3 >/dev/null 2>&1; then \
      wget -qO- 'https://bootstrap.pypa.io/get-pip.py' | /usr/bin/python3; \
    fi && \
    /usr/bin/python3 -m pip install --no-cache-dir --upgrade pip 'setuptools<82' wheel packaging build && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip "$(command -v pip3)" 1 && \
    ln -s libomp.so.5 "/usr/lib/$(gcc -print-multiarch)/libomp.so" && \
    ldconfig

COPY --link --from=torch-common --chmod=755 install_cudnn.sh /tmp/install_cudnn.sh

RUN CUDA_MAJOR_VERSION=$(echo "${CUDA_VERSION}" | cut -d. -f1) && \
    CUDA_MINOR_VERSION=$(echo "${CUDA_VERSION}" | cut -d. -f2) && \
    CUDA_PACKAGE_VERSION="${CUDA_MAJOR_VERSION}-${CUDA_MINOR_VERSION}" && \
    apt-get -qq update && \
    apt-get -qq install --no-upgrade -y \
      cuda-nvtx-${CUDA_PACKAGE_VERSION} \
      cuda-nvrtc-dev-${CUDA_PACKAGE_VERSION} && \
    { if [ ! -d /opt/nccl-tests ]; then \
      NCCL_PACKAGE_VERSION="2.*+cuda${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION}" && \
      apt-get -qq install --no-upgrade -y \
        "libnccl2=${NCCL_PACKAGE_VERSION}" "libnccl-dev=${NCCL_PACKAGE_VERSION}"; fi; } && \
    /tmp/install_cudnn.sh "${CUDA_VERSION}" dev && \
    rm /tmp/install_cudnn.sh && \
    apt-get clean

# Add Kitware's apt repository to get a newer version of CMake
RUN apt-get -qq update && apt-get -qq install -y \
      software-properties-common lsb-release && \
    { wget -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc \
    | gpg --dearmor -o /etc/apt/trusted.gpg.d/kitware.gpg; } && \
    apt-add-repository -n "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" && \
    apt-get -qq update && \
    apt-get -qq install -y 'cmake=3.31.6-*' 'cmake-data=3.31.6-*' && \
    apt-get clean && \
    python3 -m pip install --no-cache-dir 'cmake==3.31.6'

# Setup for sccache and its wrappers.
# Keep the actual sccache binary off the PATH to prevent nesting shenanigans.
COPY --link --from=sccache-downloader /opt/sccache /opt/sccache
COPY --link --from=common --chmod=755 sccache*.sh /opt/
COPY --link --from=common sccache.toml /etc/sccache.toml
RUN sed -Ei 's@^(key_prefix.*)misc@\1torch@' /etc/sccache.toml
ENV SCCACHE_CONF=/etc/sccache.toml

# Build-time environment variables
ENV CC=/opt/sccache-cc.sh \
    CXX=/opt/sccache-c++.sh

# Skip nvcc's -MD dependency generation in torch extension builds,
# which sccache cannot cache. These are always built from scratch,
# so incremental build dependency tracking has no value anyway.
ENV TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES=1

# Suppress less-relevant-to-us warnings from the packages being built
ENV CFLAGS="-Wno-sign-compare -Wno-deprecated-declarations -Wno-incompatible-pointer-types -Wno-unused-parameter -Wno-unused-variable -Wno-unused-result" \
    CXXFLAGS="-Wno-sign-compare -Wno-deprecated-declarations -Wno-unused-parameter -Wno-unused-variable -Wno-unused-result"

# Update compiler (GCC) and linker (LLD) versions
RUN LLVM_VERSION='18' && \
    CODENAME="$(lsb_release -cs)" && \
    wget -qO - 'https://apt.llvm.org/llvm-snapshot.gpg.key' > /etc/apt/trusted.gpg.d/apt.llvm.org.asc && \
    apt-add-repository -n "deb https://apt.llvm.org/$CODENAME/ llvm-toolchain-$CODENAME-$LLVM_VERSION main" && \
    SETUP_TOOLCHAIN() { \
        if [ "$(. /etc/os-release && echo "$VERSION_ID")" = '22.04' ]; then \
          apt-add-repository -y ppa:ubuntu-toolchain-r/test 2>&1 \
          | sed -e '/connection timed out/{p; Q1}'; \
        fi && \
        apt-get -qq install --no-install-recommends -y \
          gcc-13 g++-13 gfortran-13 \
          "lld-$LLVM_VERSION" "libomp-$LLVM_VERSION-dev" && \
        apt-get clean; \
    } && \
    { SETUP_TOOLCHAIN || { sleep "$(shuf -i10-20 -n1)" && SETUP_TOOLCHAIN; }; } && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 13 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 13 && \
    update-alternatives --install /usr/bin/cc cc /usr/bin/gcc 1 && \
    update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 1 && \
    update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-13 13 && \
    if [ "$(uname -m)" != 'aarch64' ]; then \
      update-alternatives --install /usr/bin/ld ld "/usr/bin/ld.lld-$LLVM_VERSION" 1; \
    fi && \
    ldconfig


FROM builder-base-shared AS builder-base-arm64
ARG BUILD_TORCH_CUDA_ARCH_LIST
ENV TORCH_CUDA_ARCH_LIST="9.0${BUILD_TORCH_CUDA_ARCH_LIST#* 9.0}"
# There is currently no CPU BLAS used for ARM builds


FROM builder-base-shared AS builder-base-amd64
ARG BUILD_TORCH_CUDA_ARCH_LIST
ENV TORCH_CUDA_ARCH_LIST="${BUILD_TORCH_CUDA_ARCH_LIST}"
# Install AOCL-BLAS and AOCL-LAPACK
# See: https://www.amd.com/en/developer/aocl/dense.html
ARG AOCL_BASE
COPY --from=aocl-downloader "${AOCL_BASE}" "${AOCL_BASE}"

# `ldconfig` lets the dynamic linker access AOCL libraries
RUN install -m 0644 -t /etc/ld.so.conf.d "${AOCL_BASE}/aocl.conf" && \
    ldconfig

# These environment variables are only for the build stage,
# and register paths to build-time AOCL resources.
# This could alternatively be done by invoking `. "${AOCL_BASE}/amd-libs.cfg"`
# in every RUN compilation step, but this will make sure it is never missed.
#
# PyTorch's logic to find LAPACK during CMake configuration
# additionally requires its installed path to either be in:
# - One of:
#   - /usr/local/lib, or
#   - /usr/lib, or
#   - /usr/local/lib64, or
#   - /usr/lib64, or
#   - /usr/lib/aarch64-linux-gnu, or
# - $LD_LIBRARY_PATH
# While skipping $LIBRARY_PATH, and ld's normal configured paths,
# so it is necessary to add $LD_LIBRARY_PATH here as well.
# See: https://github.com/pytorch/pytorch/blob/v2.3.0/cmake/Modules/FindLAPACK.cmake#L56-L59
ENV C_INCLUDE_PATH="${AOCL_BASE}/include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}" \
    CPLUS_INCLUDE_PATH="${AOCL_BASE}/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}" \
    LD_LIBRARY_PATH="${AOCL_BASE}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    LIBRARY_PATH="${AOCL_BASE}/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"


FROM builder-base-${TARGETARCH} AS builder-base
RUN mkdir /build /build/dist
WORKDIR /build
COPY --link --from=torch-common --chmod=755 effective_cpu_count.sh .
COPY --link --from=torch-common --chmod=755 scale.sh .
COPY --link --from=torch-common compiler_wrapper.f95 .
ARG AMD64_NATIVE_ARCH="skylake"
ARG ARM64_NATIVE_ARCH="armv8.5-a+nopredres"
RUN if [ "$(uname -m)" = "aarch64" ]; then \
      NATIVE="WRAPPER_NATIVE=\"${ARM64_NATIVE_ARCH}\"" && \
      AVX='WRAPPER_NO_AVX'; \
    else \
      NATIVE="WRAPPER_NATIVE=\"${AMD64_NATIVE_ARCH}\"" && \
      AVX='WRAPPER_AVX="AVX256"'; \
    fi && \
    gfortran -ffree-line-length-512 -cpp -O3 "-D${NATIVE}" "-D${AVX}" ./compiler_wrapper.f95 -o ./compiler && rm ./compiler_wrapper.f95

COPY <<-"EOT" /build/version-string.sh
	#!/bin/sh
	set -e;
	VERSION="$1";

	IS_HASH() {
	  echo "$1" | grep -qxiEe '[0-9a-f]{40}';
	};

	if IS_HASH "$VERSION"; then
	  REAL_VERSION="$(cat ./version.txt)";
	  SHORT_HASH="$(echo "$VERSION" | cut -c1-7)";
	  echo "$REAL_VERSION+$SHORT_HASH";
	else
	  echo "$VERSION";
	fi;
EOT
RUN chmod 755 /build/version-string.sh

COPY <<-"EOT" /build/storage-info.sh
	#!/bin/sh
	set -e;
	TARGET="$(realpath "$1")";

	STORAGE_INFO="$(df -h '--output=fstype,used,avail,pcent,target' "$TARGET")" || exit 0;
	printf 'Storage info for %s:\n%s\n' "$TARGET" "$STORAGE_INFO";
EOT
RUN chmod 755 /build/storage-info.sh

COPY <<-"EOT" /build/build-wheel.sh
	#!/bin/bash
	set -eo pipefail
	if [ -n "$1" ]; then cd "$1"; fi
	echo "Building $(realpath -s .)"
	python3 -m build -wvxno /build/dist 2>&1 \
	| grep -Ev --line-buffered \
	  '^(ptxas (/tmp/|(info|warning)\s*:)|copying .+/|adding .+/|creating build/|Wrapped header:)|bytes spill stores'
EOT
RUN chmod 755 /build/build-wheel.sh

## Build torch
RUN --mount=type=bind,from=pytorch-downloader,source=/git/pytorch,target=pytorch/ \
    pip3 install --no-cache-dir --upgrade numpy && \
    cd pytorch && pip3 install --no-cache-dir -r requirements.txt

# Build tool & library paths, shared for all libraries to be built
ENV CMAKE_PREFIX_PATH=/usr/bin/ \
    LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64:/usr/local/lib \
    CUDA_BIN_PATH=/usr/local/cuda/bin \
    CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda/ \
    CUDNN_LIB_DIR=/usr/local/cuda/lib64

ARG BUILD_TRITON
ARG BUILD_MAX_JOBS=""
RUN --mount=type=bind,from=triton-downloader,source=/git/triton,target=triton/,rw \
    --mount=type=secret,id=s3_access_key_id,env=AWS_ACCESS_KEY_ID \
    --mount=type=secret,id=s3_secret_access_key,env=AWS_SECRET_ACCESS_KEY \
    --mount=type=tmpfs,target=/sccache \
    --mount=type=tmpfs,target=/tmp \
    . /opt/sccache-start.sh && \
    if [ "$BUILD_TRITON" = '1' ]; then \
      pip3 install --no-cache-dir pybind11 lit && \
      export MAX_JOBS="${BUILD_MAX_JOBS:-$(./scale.sh "$(./effective_cpu_count.sh)" 3 32)}" && \
      if [ -f 'triton/python/setup.py' ]; then \
        /build/build-wheel.sh triton/python; \
      else \
        /build/build-wheel.sh triton; \
      fi && \
      pip3 install --no-cache-dir dist/*.whl; \
    fi

ARG BUILD_TORCH_VERSION
ENV TORCH_VERSION=$BUILD_TORCH_VERSION
# Filter out the 10.0 & 12.0 arches on CUDA versions < 12.8
ENV TORCH_CUDA_ARCH_LIST="${CUDA_VERSION##12.8.*}||${TORCH_CUDA_ARCH_LIST/ 10.0 12.0/}||${TORCH_CUDA_ARCH_LIST}"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST#12.9.?}"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST#13.?.?}"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST#||*||}"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST%||*}"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST#*||}"

RUN printf 'Arch: %s\nTORCH_CUDA_ARCH_LIST=%s\n' "$(uname -m)" "${TORCH_CUDA_ARCH_LIST}"

ARG BUILD_NVCC_APPEND_FLAGS="-gencode=arch=compute_90a,code=sm_90a"
# Add sm_100a & sm_120a builds unless NV_CUDA_LIB_VERSION matches 12.[0-7].*
RUN FLAGS="$BUILD_NVCC_APPEND_FLAGS" && \
    case "${NV_CUDA_LIB_VERSION}" in 12.[0-7].*) ;; *) \
      FLAGS="${FLAGS}$( \
        printf -- ' -gencode=arch=compute_%s,code=sm_%s' 120a 120a 100a 100a 103a 103a \
      )" ;; \
    esac && \
    { printf '%s' \
      "-Wno-deprecated-gpu-targets" \
      " --diag-suppress=191,186,177" \
      " -Xcompiler -Wno-sign-compare" \
      ",-Wno-deprecated-declarations" \
      ",-Wno-unused-parameter" \
      ",-Wno-unused-variable" \
      ",-Wno-unused-result" \
      "${FLAGS:+ $FLAGS}" && echo; \
    } > /build/nvcc.conf

COPY --link --from=torch-common --chmod=755 nvcc-wrapper.py /build/nvcc-wrapper.py
ENV PYTORCH_NVCC='/build/nvcc-wrapper.py' \
    CMAKE_CUDA_COMPILER='/build/nvcc-wrapper.py' \
    CUDACXX='/build/nvcc-wrapper.py'
# Filter these codes because we already build for the architecture-specific
# versions of them instead.
ENV NVCC_WRAPPER_FILTER_CODES='sm_70;sm_75;sm_90;sm_100;sm_120;compute_70;compute_75;compute_90;compute_100'

# Manually replicate the sccache-friendly settings that PyTorch applies when it
# auto-detects sccache at configure time. Without sccache on the PATH during the
# CMake configure step (e.g. when sccache is set manually as a compiler
# launcher), the auto-detection doesn't fire and nvcc falls back to response
# files, which cause sccache to consider the call non-cacheable.
#
# Upstream reference:
#   https://github.com/pytorch/pytorch/blob/v2.11.0/cmake/Dependencies.cmake#L68-L74
ENV CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES=OFF \
    CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES=OFF \
    CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS=OFF

# If the directory /opt/nccl-tests exists,
# the base image is assumed to be nccl-tests,
# so it uses the system's special NCCL and UCC installations for the build.
#
# Additionally, this RUN is executed with the downloaded PyTorch repository
# mounted temporarily in "rw" mode, which allows ephemeral writes like
# OverlayFS would that do not mutate the downloaded copy.
# This means the downloaded data never needs to be duplicated in the cache in
# a layer of this build step, and temporary build files are automatically
# cleaned up at the end of the step once the directory is detached.
#
# This step is itself cacheable as long as the downloaded files (and ARCH_LIST)
# remain the same.
#
# NB: This cannot specify BLAS=FLAME directly, because PyTorch (v2.3.0)'s code
# to explicitly choose a BLAS implementation is missing that option
# (See: https://github.com/pytorch/pytorch/blob/v2.3.0/cmake/Dependencies.cmake#L195-L266),
# and using BLAS=blis makes it ignore the libflame LAPACK library, because
# that triggers its FindBLIS logic rather than FindBLAS, and FindLAPACK depends
# on a variable set only during FindBLAS (BLAS_INFO=FLAME)
# (See: https://github.com/pytorch/pytorch/blob/v2.3.0/cmake/Modules/FindLAPACK.cmake#L176-L189).
# Thus, we have to force it to use its generic FindBLAS logic,
# and narrow it down from there by specifying WITH_BLAS=FLAME
# (See: https://github.com/pytorch/pytorch/blob/v2.3.0/cmake/Modules/FindBLAS.cmake#L259-L271).
# Without WITH_BLAS, it would detect the BLAS implementation as
# BLAS_INFO=blis instead of BLAS_INFO=FLAME and wouldn't include LAPACK either.
ARG BUILD_CXX11_ABI=""
SHELL ["/bin/bash", "-eo", "pipefail", "-c"]
RUN --mount=type=bind,from=pytorch-downloader,source=/git/pytorch,target=pytorch/,rw \
    --mount=type=secret,id=s3_access_key_id,env=AWS_ACCESS_KEY_ID \
    --mount=type=secret,id=s3_secret_access_key,env=AWS_SECRET_ACCESS_KEY \
    --mount=type=tmpfs,target=/sccache \
    --mount=type=tmpfs,target=/tmp \
    . /opt/sccache-start.sh && \
    export MAX_JOBS="${BUILD_MAX_JOBS:-$(./scale.sh "$(./effective_cpu_count.sh)" 3 32)}" && \
    echo "MAX_JOBS: ${MAX_JOBS}" && \
    export NVCC_APPEND_FLAGS="$(cat /build/nvcc.conf)" && \
    echo "NVCC_APPEND_FLAGS: ${NVCC_APPEND_FLAGS}" && \
    if [ -n "${BUILD_CXX11_ABI}" ]; then \
      export _GLIBCXX_USE_CXX11_ABI="${BUILD_CXX11_ABI}"; \
    fi && \
    case "${NV_NVTX_VERSION}" in \
      12.[0-8].*) ;; \
      *) export USE_SYSTEM_NVTX=1 ;; \
    esac && \
    ./storage-info.sh . && \
    cd pytorch && \
    ../storage-info.sh . && \
    if [ "$(uname -m)" = 'aarch64' ]; then \
      export USE_PRIORITIZED_TEXT_FOR_LD=1; \
    fi && \
    { if [ -d /opt/nccl-tests ]; then \
      export \
        USE_DISTRIBUTED=1 \
        USE_NCCL=1 USE_SYSTEM_NCCL=1 \
        UCC_HOME=${HPCX_UCC_DIR} UCX_HOME=${HPCX_UCX_DIR} \
        USE_NCCL_WITH_UCC=1 \
        USE_UCC=1 USE_SYSTEM_UCC=1; fi; } && \
    USE_CUDNN=1 \
    BUILD_TORCH=ON \
    BUILD_TEST=0 \
    USE_CUDA=1 \
    USE_NNPACK=1 \
    USE_BLAS=1 \
    USE_LAPACK=1 \
    WITH_BLAS=FLAME \
    PYTORCH_BUILD_VERSION="$(../version-string.sh "$TORCH_VERSION")" \
    PYTORCH_BUILD_NUMBER=0 \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    /build/build-wheel.sh
SHELL ["/bin/sh", "-c"]
RUN pip3 install --no-cache-dir --upgrade dist/torch*.whl

FROM builder-base AS torchvision-builder
RUN rm ./dist/*

## Build torchvision
ARG BUILD_TORCH_VISION_VERSION
ENV TORCH_VISION_VERSION=$BUILD_TORCH_VISION_VERSION
RUN pip3 install --no-cache-dir --upgrade \
    matplotlib numpy typing_extensions requests pillow

RUN --mount=type=bind,from=torchvision-downloader,source=/git/vision,target=vision/,rw \
    --mount=type=secret,id=s3_access_key_id,env=AWS_ACCESS_KEY_ID \
    --mount=type=secret,id=s3_secret_access_key,env=AWS_SECRET_ACCESS_KEY \
    --mount=type=tmpfs,target=/sccache \
    --mount=type=tmpfs,target=/tmp \
    . /opt/sccache-start.sh && \
    export MAX_JOBS="${BUILD_MAX_JOBS:-$(./scale.sh "$(./effective_cpu_count.sh)" 3 32)}" && \
    echo "MAX_JOBS: ${MAX_JOBS}" && \
    export NVCC_APPEND_FLAGS="$(cat /build/nvcc.conf)" && \
    echo "NVCC_APPEND_FLAGS: ${NVCC_APPEND_FLAGS}" && \
    cd vision && \
    USE_CUDA=1 \
    FORCE_CUDA=1 \
    BUILD_VERSION="$(../version-string.sh "$TORCH_VISION_VERSION")" \
    /build/build-wheel.sh

FROM builder-base AS torchaudio-builder
RUN rm ./dist/*

## Build torchaudio
ARG BUILD_TORCH_AUDIO_VERSION
ENV TORCH_AUDIO_VERSION=$BUILD_TORCH_AUDIO_VERSION
RUN pip3 install --no-cache-dir --upgrade \
    matplotlib numpy typing_extensions requests pillow

RUN --mount=type=bind,from=torchaudio-downloader,source=/git/audio,target=audio/,rw \
    --mount=type=secret,id=s3_access_key_id,env=AWS_ACCESS_KEY_ID \
    --mount=type=secret,id=s3_secret_access_key,env=AWS_SECRET_ACCESS_KEY \
    --mount=type=tmpfs,target=/sccache \
    --mount=type=tmpfs,target=/tmp \
    . /opt/sccache-start.sh && \
    export MAX_JOBS="${BUILD_MAX_JOBS:-$(./scale.sh "$(./effective_cpu_count.sh)" 3 32)}" && \
    echo "MAX_JOBS: ${MAX_JOBS}" && \
    export NVCC_APPEND_FLAGS="$(cat /build/nvcc.conf)" && \
    echo "NVCC_APPEND_FLAGS: ${NVCC_APPEND_FLAGS}" && \
    cd audio && \
    USE_CUDA=1 \
    FORCE_CUDA=1 \
    BUILD_VERSION="$(../version-string.sh "$TORCH_AUDIO_VERSION")" \
    /build/build-wheel.sh

FROM builder-base AS transformerengine-builder
RUN rm ./dist/*

# Build TransformerEngine
ARG BUILD_TRANSFORMERENGINE_CUDA_ARCH_LIST
ENV NVTE_CUDA_ARCHS=$BUILD_TRANSFORMERENGINE_CUDA_ARCH_LIST
ENV NVTE_CMAKE_EXTRA_ARGS='-DCMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES=OFF -DCMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES=OFF -DCMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS=OFF'

RUN --mount=type=bind,from=transformerengine-downloader,source=/git/TransformerEngine,target=TransformerEngine/,rw \
    --mount=type=secret,id=s3_access_key_id,env=AWS_ACCESS_KEY_ID \
    --mount=type=secret,id=s3_secret_access_key,env=AWS_SECRET_ACCESS_KEY \
    --mount=type=tmpfs,target=/sccache \
    --mount=type=tmpfs,target=/tmp \
    . /opt/sccache-start.sh && \
    export MAX_JOBS=$(./scale.sh "$(./effective_cpu_count.sh)" 7 32) && \
    export MAX_JOBS="${BUILD_MAX_JOBS:-$MAX_JOBS}" && \
    export NVTE_BUILD_THREADS_PER_JOB=7 && \
    echo "MAX_JOBS: ${MAX_JOBS}; NVTE_BUILD_THREADS_PER_JOB: ${NVTE_BUILD_THREADS_PER_JOB}" && \
    export NVCC_APPEND_FLAGS="$(cat /build/nvcc.conf)" && \
    echo "NVCC_APPEND_FLAGS: ${NVCC_APPEND_FLAGS}" && \
    case "${CUDA_VERSION}" in 12.[0123456].*) \
      export NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS%;100*}" ;; \
    esac && \
    /build/build-wheel.sh TransformerEngine

FROM builder-base AS flash-attn-builder-base
RUN rm ./dist/*
ENV PYTHONUNBUFFERED=1
ENV FLASH_ATTENTION_FORCE_BUILD=TRUE
ARG BUILD_FLASH_ATTN_MAX_JOBS=""

FROM flash-attn-builder-base AS flash-attn-builder

# Build flash-attn
RUN --mount=type=bind,from=flash-attn-downloader,source=/git/flash-attention,target=flash-attention/,rw \
    --mount=type=secret,id=s3_access_key_id,env=AWS_ACCESS_KEY_ID \
    --mount=type=secret,id=s3_secret_access_key,env=AWS_SECRET_ACCESS_KEY \
    --mount=type=tmpfs,target=/sccache \
    --mount=type=tmpfs,target=/tmp \
    . /opt/sccache-start.sh && \
    export CC_WRAPPED=$(realpath -e ./compiler) \
      MAX_JOBS="${BUILD_FLASH_ATTN_MAX_JOBS:-$(./scale.sh "$(./effective_cpu_count.sh)" 8 12)}" && \
    echo "MAX_JOBS: ${MAX_JOBS}" && \
    export NVCC_APPEND_FLAGS="$(cat /build/nvcc.conf)" && \
    echo "NVCC_APPEND_FLAGS: ${NVCC_APPEND_FLAGS}" && \
    cd flash-attention && \
    for EXT_DIR in $(realpath -s -e \
      . \
      csrc/fused_dense_lib \
      csrc/layer_norm); \
    do /build/build-wheel.sh "$EXT_DIR" || exit 1; done

FROM flash-attn-builder-base AS flash-attn-3-builder

# Artifically sequence this build stage after the previous one
# to prevent parallelism, because these are both very resource-intensive
RUN --mount=type=bind,from=flash-attn-builder,source=/build,target=/build :
ARG BUILD_FLASH_ATTN_MAX_JOBS="${BUILD_FLASH_ATTN_MAX_JOBS:-3}"

ENV NVCC_WRAPPER_FILTER_CODES='sm_70;sm_75;compute_70;compute_75'
ENV FLASH_ATTN_CUDA_ARCHS='80;90;100;120'
# Build flash-attn v3
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN --mount=type=bind,from=flash-attn-3-downloader,source=/git/flash-attention,target=flash-attention/,rw \
    --mount=type=secret,id=s3_access_key_id,env=AWS_ACCESS_KEY_ID \
    --mount=type=secret,id=s3_secret_access_key,env=AWS_SECRET_ACCESS_KEY \
    --mount=type=tmpfs,target=/sccache \
    --mount=type=tmpfs,target=/tmp \
    . /opt/sccache-start.sh && \
    if [ ! -d flash-attention/hopper ]; then \
      echo "Not compiling flash-attn v3" && exit 0; \
    fi && \
    export CC_WRAPPED=$(realpath -e ./compiler) \
      MAX_JOBS="${BUILD_FLASH_ATTN_MAX_JOBS:-$(./scale.sh "$(./effective_cpu_count.sh)" 10 6)}" && \
    echo "MAX_JOBS: ${MAX_JOBS}" && \
    export NVCC_APPEND_FLAGS="$(cat /build/nvcc.conf)" && \
    NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:+$NVCC_APPEND_FLAGS }-Xfatbin -compress-all -Xfatbin -compress-mode=size" && \
    if [ "$(uname -m)" != "aarch64" ]; then \
      export CXXFLAGS="${CXXFLAGS:+$CXXFLAGS }-mcmodel=medium" && \
      NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:+$NVCC_APPEND_FLAGS }-Xcompiler -mcmodel=medium"; \
    else \
      export NVCC_WRAPPER_ATTEMPTS=2; \
    fi && \
    NVCC_APPEND_FLAGS="-gencode=arch=compute_103a,code=sm_103a -gencode=arch=compute_100a,code=sm_100a $(echo ${NVCC_APPEND_FLAGS} | sed -E 's@[[:space:]]?-gencode=[^[:space:]]+@@g')" && \
    echo "NVCC_APPEND_FLAGS: ${NVCC_APPEND_FLAGS}" && \
    sed -i 's@if bare_metal_version != Version("12.8"):@if bare_metal_version < Version("12.8"):@' \
      flash-attention/hopper/setup.py && \
    sed -Ei 's@(\s+cuda_post_cflags_sm80(_sm90)? = .*)$@\1 + '"['-gencode', 'arch=compute_120f,code=sm_120f']"'@' \
      flash-attention/hopper/setup.py && \
    NVCC_THREADS=4 /build/build-wheel.sh flash-attention/hopper

FROM builder-base AS builder
COPY --link --from=torchaudio-builder /build/dist/ /build/dist/
COPY --link --from=torchvision-builder /build/dist/ /build/dist/
COPY --link --from=transformerengine-builder /build/dist/ /build/dist/
COPY --link --from=flash-attn-builder /build/dist/ /build/dist/
COPY --link --from=flash-attn-3-builder /build/dist/ /build/dist/

## Build the final torch image.
FROM ${FINAL_BASE_IMAGE} as final-arm64
ARG BUILD_TORCH_CUDA_ARCH_LIST
ENV TORCH_CUDA_ARCH_LIST="9.0${BUILD_TORCH_CUDA_ARCH_LIST#* 9.0}"

FROM ${FINAL_BASE_IMAGE} as final-amd64
ARG BUILD_TORCH_CUDA_ARCH_LIST
ENV TORCH_CUDA_ARCH_LIST="${BUILD_TORCH_CUDA_ARCH_LIST}"

FROM final-${TARGETARCH}
ENV DEBIAN_FRONTEND=noninteractive

# Install core packages.
# Remove PEP 668 marker since containers are often single-purpose
# and thus do not inherently need to be pushed into venv isolation.
RUN if [ "$(. /etc/os-release && echo "$VERSION_ID")" = '22.04' ]; then \
      PACKAGES='libncurses5 python3-pip python3-distutils'; \
    else \
      PACKAGES='libncurses6 python3-dev'; \
    fi && \
    apt-get -qq update && apt-get -qq install -y \
      $PACKAGES python3 gcc \
      libomp5 libpng16-16 libjpeg-turbo8 libsodium23 \
      curl git apt-utils ssh ca-certificates tmux nano vim-tiny sudo bash \
      rsync htop wget unzip tini && \
    apt-get clean && \
    rm -f /usr/lib/python3.*/EXTERNALLY-MANAGED && \
    if ! command -v pip3 >/dev/null 2>&1; then \
      wget -qO- 'https://bootstrap.pypa.io/get-pip.py' | /usr/bin/python3; \
    fi && \
    /usr/bin/python3 -m pip install --no-cache-dir --upgrade pip 'setuptools<82' && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip "$(command -v pip3)" 1 && \
    update-alternatives --install /usr/bin/vim vim /usr/bin/vim.tiny 1 && \
    ln -s libomp.so.5 "/usr/lib/$(gcc -print-multiarch)/libomp.so" && \
    ldconfig

RUN apt-get -qq update && apt-get -qq install -y --no-install-recommends \
        software-properties-common lsb-release && \
    if [ "$(. /etc/os-release && echo "$VERSION_ID")" = '22.04' ]; then \
      SETUP_LIBSTDCXX() { \
          apt-add-repository -y ppa:ubuntu-toolchain-r/test 2>&1 \
          | sed -e '/connection timed out/{p; Q1}' && \
          apt-get -qq install -y --no-install-recommends libstdc++6 && \
          apt-get clean; \
      } && \
      { SETUP_LIBSTDCXX || { sleep "$(shuf -i10-20 -n1)" && SETUP_LIBSTDCXX; }; }; \
    else \
      apt-get -qq install -y --no-install-recommends libstdc++6 && \
      apt-get clean; \
    fi

RUN LLVM_VERSION='18' && \
    CODENAME="$(lsb_release -cs)" && \
    wget -qO - 'https://apt.llvm.org/llvm-snapshot.gpg.key' > /etc/apt/trusted.gpg.d/apt.llvm.org.asc && \
    apt-add-repository "deb https://apt.llvm.org/$CODENAME/ llvm-toolchain-$CODENAME-$LLVM_VERSION main" && \
    apt-get -qq install -y --no-install-recommends "libomp5-$LLVM_VERSION" && \
    apt-get clean

# Install AOCL-BLAS and AOCL-LAPACK
# See: https://www.amd.com/en/developer/aocl/dense.html
ARG AOCL_BASE
COPY --from=aocl-downloader "${AOCL_BASE}" "${AOCL_BASE}"

# `ldconfig` lets the dynamic linker access AOCL libraries
RUN install -m 0644 -t /etc/ld.so.conf.d "${AOCL_BASE}/aocl.conf" && \
    ldconfig

ARG BUILD_TORCH_VERSION
ARG BUILD_TORCH_VISION_VERSION
ARG BUILD_TORCH_AUDIO_VERSION
ENV TORCH_VERSION=$BUILD_TORCH_VERSION
ENV TORCH_VISION_VERSION=$BUILD_TORCH_VISION_VERSION
ENV TORCH_AUDIO_VERSION=$BUILD_TORCH_AUDIO_VERSION
# Filter out the 10.0 & 12.0 arches on CUDA versions < 12.8
ENV TORCH_CUDA_ARCH_LIST="${CUDA_VERSION##12.8.*}||${TORCH_CUDA_ARCH_LIST/ 10.0 12.0/}||${TORCH_CUDA_ARCH_LIST}"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST#12.9.?}"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST#13.?.?}"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST#||*||}"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST%||*}"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST#*||}"

COPY --link --from=torch-common --chmod=755 install_cudnn.sh /tmp/install_cudnn.sh
# - libnvjitlink-X-Y only exists for CUDA versions >= 12-0.
# - Don't mess with libnccl2 when using nccl-tests as a base,
#   checked via the existence of the directory "/opt/nccl-tests".
RUN CUDA_MAJOR_VERSION="$(echo "${CUDA_VERSION}" | cut -d. -f1)" && \
    CUDA_MINOR_VERSION="$(echo "${CUDA_VERSION}" | cut -d. -f2)" && \
    CUDA_PACKAGE_VERSION="${CUDA_MAJOR_VERSION}-${CUDA_MINOR_VERSION}" && \
    CUDART_VERSION_SPEC="${NV_CUDA_CUDART_VERSION:+=$NV_CUDA_CUDART_VERSION}" && \
    apt-get -qq update && \
    apt-get -qq install --no-upgrade -y \
      cuda-nvcc-${CUDA_PACKAGE_VERSION} \
      libcurand-${CUDA_PACKAGE_VERSION} \
      libcufft-${CUDA_PACKAGE_VERSION} \
      libcublas-${CUDA_PACKAGE_VERSION} \
      cuda-nvrtc-${CUDA_PACKAGE_VERSION} \
      cuda-cudart-dev-${CUDA_PACKAGE_VERSION}"${CUDART_VERSION_SPEC}" \
      libcusparse-${CUDA_PACKAGE_VERSION} \
      libcusolver-${CUDA_PACKAGE_VERSION} \
      libcufile-${CUDA_PACKAGE_VERSION} \
      cuda-cupti-${CUDA_PACKAGE_VERSION} \
      libnvjpeg-${CUDA_PACKAGE_VERSION} \
      libnvtoolsext1 && \
    { if [ "${CUDA_MAJOR_VERSION}" -ge 12 ]; then \
      apt-get -qq install --no-upgrade -y libnvjitlink-${CUDA_PACKAGE_VERSION}; fi; } && \
    { if [ ! -d /opt/nccl-tests ]; then \
      NCCL_PACKAGE_VERSION="2.*+cuda${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION}" && \
      apt-get -qq install --no-upgrade -y "libnccl2=${NCCL_PACKAGE_VERSION}"; fi; } && \
    /tmp/install_cudnn.sh "${CUDA_VERSION}" runtime && \
    rm /tmp/install_cudnn.sh && \
    apt-get clean && \
    ldconfig


WORKDIR /usr/src/app

# Install custom PyTorch wheels.
RUN --mount=type=bind,from=builder,source=/build/dist,target=. \
    pip3 install --no-cache-dir -U numpy packaging && \
    pip3 install --no-cache-dir -U ./*.whl

# Install FA4.
# FA4 is installed as a pure Python wheel that JIT-compiles device code
# with the CUTLASS DSL. JIT compilation requires ptxas at runtime.
ARG BUILD_FLASH_ATTN_4_VERSION
RUN if [ -z "${BUILD_FLASH_ATTN_4_VERSION}" ]; then exit 0; fi && \
    if [ "${CUDA_VERSION%%.*}" -ge 13 ]; \
      then EXTRA='[cu13]'; \
      else EXTRA=''; \
    fi && \
    FA4_PACKAGE="$( \
      printf 'flash-attn-4%s @ git+https://github.com/Dao-AILab/flash-attention.git@%s#subdirectory=flash_attn/cute' \
        "${EXTRA}" "${BUILD_FLASH_ATTN_4_VERSION}"; \
    )" && \
    pip3 install --no-cache-dir "${FA4_PACKAGE}"

# Make a symlink to flash-attn v3 where TransformerEngine expects it,
# and modify the installation record so that pip uninstall knows how to
# fully remove it.
RUN <<-"EOT" python3
	#!/bin/env python3
	from base64 import urlsafe_b64encode as b64
	from hashlib import sha256
	from importlib import metadata
	from pathlib import Path
	from py_compile import compile

	try:
	    dist = metadata.distribution("flash-attn-3")
	    record_pattern = "flash?attn?3-*.dist-info/RECORD"
	except metadata.PackageNotFoundError:
	    dist = metadata.distribution("flashattn-hopper")
	    record_pattern = "flashattn?hopper-*.dist-info/RECORD"
	p = dist.locate_file("flash_attn_interface.py")
	print("flash_attn_interface:", p)
	root = p.parent

	if not p.exists():
	    raise SystemExit("flash_attn_interface not found")
	if not p.is_file():
	    raise SystemExit("flash_attn_interface path is not a file")

	d = root / "flash_attn_3"
	d.mkdir(mode=0o755, parents=False, exist_ok=True)

	new = d / p.name
	if new.exists():
	    raise SystemExit(f'"{new}" already exists')
	new.symlink_to(p)
	print(f"Created new symlink at {new}")

	compiled = Path(compile(new))


	def record_entry(path: Path) -> str:
	    content = path.read_bytes()
	    digest = b64(sha256(content).digest()).rstrip(b"=").decode()
	    package_path = path.relative_to(root).as_posix()
	    return f"{package_path},sha256={digest},{len(content):d}\r\n"


	for f in dist.files:
	    if f.match(record_pattern):
	        with f.locate().open("a", encoding="utf-8", newline="") as record:
	            for added in (new, compiled):
	                record.write(record_entry(added))
	        break
EOT
