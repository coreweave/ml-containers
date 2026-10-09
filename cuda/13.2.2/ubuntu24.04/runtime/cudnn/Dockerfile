# SPDX-FileCopyrightText: 2019-2021 NVIDIA CORPORATION
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
#
# SPDX-License-Identifier: BSD-3-Clause

ARG IMAGE_NAME
ARG BASE_IMAGE=${IMAGE_NAME}:13.2.2-runtime-ubuntu24.04
FROM ${BASE_IMAGE} AS base

FROM base AS base-amd64

ENV NV_CUDNN_VERSION=9.27.0.42-1
ENV NV_CUDNN_PACKAGE_NAME=libcudnn9-cuda-13
ENV NV_CUDNN_PACKAGE=libcudnn9-cuda-13=${NV_CUDNN_VERSION}

FROM base AS base-arm64

ENV NV_CUDNN_VERSION=9.27.0.42-1
ENV NV_CUDNN_PACKAGE_NAME=libcudnn9-cuda-13
ENV NV_CUDNN_PACKAGE=libcudnn9-cuda-13=${NV_CUDNN_VERSION}

FROM base-${TARGETARCH}

ARG TARGETARCH

LABEL com.nvidia.cudnn.version="${NV_CUDNN_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ${NV_CUDNN_PACKAGE} \
    && apt-mark hold ${NV_CUDNN_PACKAGE_NAME} \
    && rm -rf /var/lib/apt/lists/*
