# nixlbench with proper rdma-core (DOCA) + UCX for RoCE GPUDirect RDMA testing
# ARM64 only — for GB300 nodes (SUPP-23056)
#
# Builds UCX v1.20.0 from source against DOCA rdma-core (which includes
# Data Direct / mlx5dv support), then NIXL with UCX plugin, then nixlbench.

ARG BASE_IMAGE="nvcr.io/nvidia/cuda-dl-base:25.10-cuda13.0-devel-ubuntu24.04"
FROM ${BASE_IMAGE}

ARG ARCH="aarch64"

# Build dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -y install \
    ninja-build meson cmake build-essential pkg-config \
    libgflags-dev libnuma-dev autotools-dev automake libtool \
    libz-dev flex hwloc libhwloc-dev git wget && \
    rm -rf /var/lib/apt/lists/*

# Install DOCA for proper rdma-core with Data Direct support
RUN ARCH_SUFFIX=$(if [ "${ARCH}" = "aarch64" ]; then echo "arm64"; else echo "amd64"; fi) && \
    MELLANOX_OS="$(. /etc/lsb-release; echo ${DISTRIB_ID}${DISTRIB_RELEASE} | tr A-Z a-z | tr -d .)" && \
    wget --tries=3 --waitretry=5 --no-verbose \
      "https://www.mellanox.com/downloads/DOCA/DOCA_v3.2.0/host/doca-host_3.2.0-125000-25.10-${MELLANOX_OS}_${ARCH_SUFFIX}.deb" \
      -O doca-host.deb && \
    dpkg -i doca-host.deb && \
    apt-get update && apt-get upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get -y install --reinstall \
      libibverbs-dev rdma-core ibverbs-utils libibumad-dev \
      libnuma-dev librdmacm-dev ibverbs-providers && \
    rm -rf /var/lib/apt/lists/* doca-host.deb

# Verify Data Direct symbols are present
RUN grep -q "mlx5dv_get_data_direct_sysfs_path" /usr/include/infiniband/mlx5dv.h && \
    echo "Data Direct header: OK"

# Build UCX v1.20.0 from source against DOCA rdma-core
RUN git clone --depth 1 --branch v1.20.0 https://github.com/openucx/ucx.git /tmp/ucx && \
    cd /tmp/ucx && \
    ./autogen.sh && \
    ./configure \
      --prefix=/usr/local \
      --disable-logging --disable-debug --disable-assertions --disable-params-check \
      --enable-shared --disable-static --disable-doxygen-doc \
      --enable-optimizations --enable-cma --enable-devel-headers \
      --with-cuda=/usr/local/cuda --with-verbs --with-dm --enable-mt && \
    grep "HAVE_DECL_MLX5DV_GET_DATA_DIRECT_SYSFS_PATH.*1" config.h && \
    echo "UCX configure: Data Direct detected" && \
    make -j$(nproc) && make install && \
    ldconfig && \
    rm -rf /tmp/ucx

# Clone NIXL and build with UCX plugin only
RUN git clone --depth 1 https://github.com/ai-dynamo/nixl.git /workspace/nixl

WORKDIR /workspace/nixl
RUN meson setup build \
      --prefix=/usr/local/nixl \
      --buildtype=release \
      -Denable_plugins=UCX,POSIX \
      -Dbuild_tests=false \
      -Dbuild_examples=false && \
    cd build && ninja && ninja install

RUN echo "/usr/local/nixl/lib/${ARCH}-linux-gnu" > /etc/ld.so.conf.d/nixl.conf && \
    echo "/usr/local/nixl/lib/${ARCH}-linux-gnu/plugins" >> /etc/ld.so.conf.d/nixl.conf && \
    ldconfig

# Build nixlbench
WORKDIR /workspace/nixl/benchmark/nixlbench
RUN meson setup build \
      -Dnixl_path=/usr/local/nixl \
      -Dprefix=/usr/local/nixlbench \
      --buildtype=release && \
    cd build && ninja && ninja install

ENV PATH=/usr/local/nixlbench/bin:/usr/local/nixl/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/nixlbench/lib:/usr/local/nixl/lib/${ARCH}-linux-gnu:$LD_LIBRARY_PATH
