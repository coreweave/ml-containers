# nixlbench with proper rdma-core (DOCA) + UCX for RoCE GPUDirect RDMA testing
# ARM64 only — for GB300 nodes (SUPP-23056)
#
# Builds UCX v1.20.0 from source against DOCA rdma-core (which includes
# Data Direct / mlx5dv support), then NIXL with UCX plugin, then nixlbench.
# Includes etcd (server + client lib) for multi-node coordination.

ARG BASE_IMAGE="nvcr.io/nvidia/cuda-dl-base:25.10-cuda13.0-devel-ubuntu24.04"
FROM ${BASE_IMAGE}

ARG ARCH="aarch64"
ARG ABSL_TAG="lts_2025_08_14"
ARG GRPC_TAG="v1.73.0"

# Build dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -y install \
    ninja-build meson cmake build-essential pkg-config \
    libgflags-dev libnuma-dev autotools-dev automake libtool \
    libz-dev flex hwloc libhwloc-dev git wget \
    pybind11-dev python3-dev \
    libprotobuf-dev protobuf-compiler-grpc libcpprest-dev \
    etcd-server etcd-client && \
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

# Build Abseil (required by gRPC and etcd-cpp-apiv3)
WORKDIR /workspace
RUN git clone https://github.com/abseil/abseil-cpp.git && \
    cd abseil-cpp && \
    git fetch --depth 1 origin ${ABSL_TAG} && \
    git checkout ${ABSL_TAG} && \
    mkdir -p build && cd build && \
    cmake .. \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DABSL_PROPAGATE_CXX_STD=ON \
        -DABSL_ENABLE_INSTALL=ON && \
    make -j$(nproc) && make install && \
    ldconfig && \
    rm -rf /workspace/abseil-cpp

# Build gRPC (required by etcd-cpp-apiv3)
RUN git clone --recurse-submodules -b ${GRPC_TAG} --depth 1 --shallow-submodules \
      https://github.com/grpc/grpc /workspace/grpc && \
    cd /workspace/grpc && \
    mkdir -p cmake/build && cd cmake/build && \
    export CMAKE_PREFIX_PATH="/usr/local${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}" && \
    export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:/usr/local/lib/${ARCH}-linux-gnu/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}" && \
    cmake ../.. \
        -DgRPC_INSTALL=ON \
        -DgRPC_BUILD_TESTS=OFF \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_CXX_STANDARD=17 \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DCMAKE_PREFIX_PATH=/usr/local \
        -Dabsl_DIR=/usr/local/lib/cmake/absl \
        -DgRPC_SSL_PROVIDER=package \
        -DgRPC_ABSL_PROVIDER=package \
        -DgRPC_PROTOBUF_PROVIDER=module \
        -DgRPC_ZLIB_PROVIDER=package && \
    make -j$(nproc) && make install && \
    ldconfig && \
    rm -rf /workspace/grpc

# Build etcd-cpp-apiv3 (required for nixlbench multi-node coordination)
RUN git clone --depth 1 https://github.com/etcd-cpp-apiv3/etcd-cpp-apiv3.git /workspace/etcd-cpp && \
    cd /workspace/etcd-cpp && \
    sed -i '/^find_dependency(cpprestsdk)$/d' etcd-cpp-api-config.in.cmake && \
    mkdir build && cd build && \
    cmake .. -DBUILD_ETCD_CORE_ONLY=ON -DCMAKE_BUILD_TYPE=Release \
             -DETCD_CMAKE_CXX_STANDARD=17 -DCMAKE_PREFIX_PATH=/usr/local && \
    make -j$(nproc) && make install && \
    ldconfig && \
    rm -rf /workspace/etcd-cpp

ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

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
