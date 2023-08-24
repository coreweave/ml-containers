# ml-containers

Repository for building ML images at CoreWeave


## Index

See the [list of all published images](https://github.com/orgs/coreweave/packages?repo_name=ml-containers).

### PyTorch Base Images

- [`ghcr.io/coreweave/ml-containers/torch`](https://github.com/coreweave/ml-containers/pkgs/container/ml-containers%2Ftorch)

CoreWeave provides custom builds of
[PyTorch](https://github.com/pytorch/pytorch),
[`torchvision`](https://github.com/pytorch/vision)
and [`torchaudio`](https://github.com/pytorch/audio)
tuned for our platform in a single container image, [`ml-containers/torch`](https://github.com/coreweave/ml-containers/pkgs/container/ml-containers%2Ftorch).

Versions compiled against CUDA 11.8.0, 12.0.1, and 12.1.1 are available in this repository, with two variants:

1. `base`: Tagged as `ml-containers/torch:a1b2c3d-base-...`.
   1. Built from [`nvidia/cuda:...-base-ubuntu20.04`](https://hub.docker.com/r/nvidia/cuda/tags?name=base-ubuntu20.04) as a base.
   2. Only includes essentials (CUDA, `torch`, `torchvision`, `torchaudio`),
      so it has a small image size, making it fast to launch.
2. `nccl`: Tagged as `ml-containers/torch:a1b2c3d-nccl-...`.
   1. Built from [`ghcr.io/coreweave/nccl-tests`](https://github.com/coreweave/nccl-tests/pkgs/container/nccl-tests) as a base.
   2. Ultimately inherits from [`nvidia/cuda:...-cudnn8-devel-ubuntu20.04`](https://hub.docker.com/r/nvidia/cuda/tags?name=cudnn8-devel-ubuntu20.04).
   3. Larger, but includes development libraries and build tools such as `nvcc` necessary for compiling other PyTorch extensions.
   4. These PyTorch builds are built on component libraries optimized for the CoreWeave cloud&mdash;see
      [`coreweave/nccl-tests`](https://github.com/coreweave/nccl-tests/blob/master/README.md).

### PyTorch Extras

- [`ghcr.io/coreweave/ml-containers/torch-extras`](https://github.com/coreweave/ml-containers/pkgs/container/ml-containers%2Ftorch-extras)

[`ml-containers/torch-extras`](https://github.com/coreweave/ml-containers/pkgs/container/ml-containers%2Ftorch-extras)
extends the [`ml-containers/torch`](https://github.com/coreweave/ml-containers/pkgs/container/ml-containers%2Ftorch)
images with a set of common PyTorch extensions:

1. [DeepSpeed](https://github.com/microsoft/DeepSpeed)
2. [FlashAttention](https://github.com/Dao-AILab/flash-attention)
3. [NVIDIA Apex](https://github.com/NVIDIA/apex)

Each one is compiled specially against the custom PyTorch builds in [`ml-containers/torch`](https://github.com/coreweave/ml-containers/pkgs/container/ml-containers%2Ftorch).

Both `base` and `nccl` editions are available for
[`ml-containers/torch-extras`](https://github.com/coreweave/ml-containers/pkgs/container/ml-containers%2Ftorch-extras)
matching those for
[`ml-containers/torch`](https://github.com/coreweave/ml-containers/pkgs/container/ml-containers%2Ftorch).
The `base` edition retains a small size, as a multi-stage build is used to avoid including
CUDA development libraries in it, despite those libraries being required to build
the extensions themselves.


## Organization
This repository contains multiple container image Dockerfiles, each is expected
to be within its own folder along with any other needed files for the build.


## CI Builds (Actions)
The current CI builds are set up to run when changes to files in the respective
folders are detected so that only the changed container images are built. The
actions are set up with an action per image utilizing a reusable base action
[build.yml](.github/workflows/build.yml). The reusable action accepts several inputs:

- `folder` - the folder containing the dockerfile for the image
- `image-name` - the name to use for the image
- `build-args` - arguments to pass to the docker build

Images built using the same source can utilize one action as the main reason for
the multiple actions is to handle only building the changed images. A build
matrix can be helpful for these cases
https://docs.github.com/en/actions/using-jobs/using-a-matrix-for-your-jobs.
