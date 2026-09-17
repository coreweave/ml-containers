#!/bin/sh
set -eu

# Full upstream snapshot containing S3 retries, including its Python/native API changes.
streamer_ref=bc21fd4182cc06ce9475452d16697d50ce3588c4
export PACKAGE_VERSION=0.16.2.dev0+gbc21fd418
case "$1" in
  arm64) native_arch=aarch64; bazel_arch=arm64; bazel_sha=2d86d36db0c9af15747ff02a80e6db11a45d68f868ea8f62f489505c474f0099 ;;
  amd64) native_arch=x86_64; bazel_arch=x86_64; bazel_sha=ac6249d1192aea9feaf49dfee2ab50c38cee2454b00cf29bbec985a11795c025 ;;
  *) echo "Unsupported architecture: $1" >&2; exit 1 ;;
esac
curl -fsSL "https://github.com/bazelbuild/bazel/releases/download/7.6.1/bazel-7.6.1-linux-${bazel_arch}" -o /tmp/bazel
printf '%s  /tmp/bazel\n' "$bazel_sha" | sha256sum -c -
chmod +x /tmp/bazel

git init /tmp/runai-streamer
git -C /tmp/runai-streamer remote add origin https://github.com/dsx-ai-factory/model-streamer.git
git -C /tmp/runai-streamer fetch --depth=1 origin "$streamer_ref"
git -C /tmp/runai-streamer checkout --detach FETCH_HEAD
cd /tmp/runai-streamer/cpp
# Batch mode leaves no Bazel server in the BuildKit container's cgroup.
/tmp/bazel --batch build --config="$native_arch" --jobs=12 \
  streamer:libstreamer.so s3:libstreamers3.so gcs:libstreamergcs.so azure:libstreamerazure.so
/tmp/bazel --batch test --config="$native_arch" --jobs=12 \
  //streamer/impl/object_storage_worker:object_storage_retry_test \
  //streamer/impl/object_storage_worker:object_storage_worker_test \
  //streamer/impl/config:config_test //s3/client:client_test --test_output=errors
for component in runai_model_streamer runai_model_streamer_s3 runai_model_streamer_gcs runai_model_streamer_azure; do
  (
    cd "/tmp/runai-streamer/py/$component"
    python3 setup.py bdist_wheel --plat-name "manylinux2014_${native_arch}" --dist-dir /out/wheels
  )
done
