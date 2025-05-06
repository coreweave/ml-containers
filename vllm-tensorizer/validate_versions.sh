#!/bin/bash

# Makes assumptions about the structure of the output string for `nvcc --version`
CUDA_TOOLKIT_VERSION=$(nvcc --version | grep -Po 'release ([0-9]+.[0-9])' | sed 's/release //')
TORCH_CUDA_VERSION=$(python3 -c 'import torch; print(torch.version.cuda)')


assert_exists() {
  local version=$1
  local name=$2
  if [[ -z $version ]]; then
    echo "$name could not be found."
    exit 1
  fi
}

assert_eq() {
  local version_a=$1
  local version_b=$2
  if [[ "$version_a" != "$version_b" ]]; then
    echo "Version mismatch detected: $version_a != $version_b"
    exit 1
  fi
}

assert_exists "${CUDA_TOOLKIT_VERSION}" "CUDA TOOLKIT VERSION"
assert_exists "${TORCH_CUDA_VERSION}" "TORCH CUDA VERSION"

assert_eq "$CUDA_TOOLKIT_VERSION" "${TORCH_CUDA_VERSION}"


echo "CUDA toolkit and PyTorch compatible."