# Patch NVIDIA vLLM runtime with:
# 1. NVIDIA's rdma-core (for Data Direct DMA-BUF headers + runtime)
# 2. UCX rebuilt against those headers (so configure detects Data Direct)
#
# Without this, UCX is compiled against Ubuntu's rdma-core which lacks
# mlx5dv Data Direct support, and GPUDirect RDMA falls back to the
# SMMU/IOMMU path (~14 MB/s vs GB/s).
FROM ghcr.io/coreweave/nccl-tests:13.1.0-devel-ubuntu22.04-nccl2.29.2-1-0bf62c2 AS rdma-source
FROM nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.9.0-cuda13 AS builder
USER root
# Copy NVIDIA rdma-core libraries AND headers for UCX build
COPY --from=rdma-source /usr/lib/aarch64-linux-gnu/libibverbs.so* /usr/lib/aarch64-linux-gnu/
COPY --from=rdma-source /usr/lib/aarch64-linux-gnu/libmlx5.so* /usr/lib/aarch64-linux-gnu/
COPY --from=rdma-source /usr/lib/aarch64-linux-gnu/librdmacm.so* /usr/lib/aarch64-linux-gnu/
COPY --from=rdma-source /usr/lib/aarch64-linux-gnu/libibverbs/ /usr/lib/aarch64-linux-gnu/libibverbs/
COPY --from=rdma-source /usr/include/infiniband/ /usr/include/infiniband/
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential autoconf automake libtool pkg-config libnuma-dev \
  && rm -rf /var/lib/apt/lists/*
RUN ldconfig
# Verify Data Direct symbols present in headers and libraries
RUN grep -q "mlx5dv_get_data_direct_sysfs_path" /usr/include/infiniband/mlx5dv.h \
  && echo "Data Direct header: OK" \
  && nm -D /usr/lib/aarch64-linux-gnu/libmlx5.so | grep -q data_direct \
  && echo "Data Direct library: OK"
# Build UCX against NVIDIA rdma-core headers (detects Data Direct at configure time)
RUN git clone --depth 1 --branch v1.20.0 https://github.com/openucx/ucx.git /tmp/ucx
WORKDIR /tmp/ucx
RUN ./autogen.sh && ./configure \
  --disable-logging --disable-debug --disable-assertions --disable-params-check \
  --prefix=/usr/local/ucx --enable-shared --disable-static --disable-doxygen-doc \
  --enable-optimizations --enable-cma --enable-devel-headers \
  --with-cuda=/usr/local/cuda --with-verbs --with-dm --enable-mt \
  && grep "HAVE_DECL_MLX5DV_GET_DATA_DIRECT_SYSFS_PATH.*1" config.h \
  && echo "UCX configure: Data Direct detected" \
  && make -j$(nproc) && make install
# Final: base image + NVIDIA rdma-core + rebuilt UCX
FROM nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.9.0-cuda13
USER root
COPY --from=builder /usr/local/ucx /usr/local/ucx
COPY --from=rdma-source /usr/lib/aarch64-linux-gnu/libibverbs.so* /usr/lib/aarch64-linux-gnu/
COPY --from=rdma-source /usr/lib/aarch64-linux-gnu/libmlx5.so* /usr/lib/aarch64-linux-gnu/
COPY --from=rdma-source /usr/lib/aarch64-linux-gnu/librdmacm.so* /usr/lib/aarch64-linux-gnu/
COPY --from=rdma-source /usr/lib/aarch64-linux-gnu/libibverbs/ /usr/lib/aarch64-linux-gnu/libibverbs/
RUN ldconfig 2>/dev/null || true
USER dynamo
