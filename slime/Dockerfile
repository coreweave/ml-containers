# syntax=docker/dockerfile:1.2
ARG BASE_IMAGE
ARG BUILDER_IMAGE="${BASE_IMAGE}"

FROM ${BUILDER_IMAGE} AS builder

ARG SGLANG_COMMIT='bbe9c7eeb520b0a67e92d133dfc137a3688dc7f2'
ARG APEX_COMMIT='10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4'
ARG SLIME_COMMIT='7014942c17625732dbe9239c0aa9d8e7d7265ccc'

WORKDIR /build
COPY build.bash patches/ /build/
RUN mkdir /wheels && \
    bash build.bash && \
    rm -rf /build/*
COPY install.bash requirements.txt patches/ /wheels/

FROM ${BASE_IMAGE}

ARG MEGATRON_COMMIT='3714d81d418c9f1bca4594fc35f9e8289f652862'
ARG SLIME_COMMIT='7014942c17625732dbe9239c0aa9d8e7d7265ccc'

RUN --mount=type=bind,from=builder,source=/wheels,target=/wheels \
    cd /wheels && \
    MEGATRON_COMMIT="${MEGATRON_COMMIT}" \
    SLIME_COMMIT="${SLIME_COMMIT}" \
    bash install.bash
