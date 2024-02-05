# ml-containers

Repository for building ML images at CoreWeave


## Index

See the [list of all published images](https://github.com/orgs/coreweave/packages?repo_name=ml-containers).

Special PyTorch Images:

- [PyTorch Base Images](#pytorch-base-images)
- [PyTorch Extras](#pytorch-extras)
- [PyTorch Nightly](#pytorch-nightly)

### PyTorch Base Images

- [`ghcr.io/coreweave/ml-containers/torch`](https://github.com/coreweave/ml-containers/pkgs/container/ml-containers%2Ftorch)

CoreWeave provides custom builds of
[PyTorch](https://github.com/pytorch/pytorch),
[`torchvision`](https://github.com/pytorch/vision)
and [`torchaudio`](https://github.com/pytorch/audio)
tuned for our platform in a single container image, [`ml-containers/torch`](https://github.com/coreweave/ml-containers/pkgs/container/ml-containers%2Ftorch).

Versions compiled against CUDA 11.8.0, 12.0.1, 12.1.1, and 12.2.2 are available in this repository, with two variants:

1. `base`: Tagged as `ml-containers/torch:a1b2c3d-base-...`.
   1. Built from [`nvidia/cuda:...-base-ubuntu22.04`](https://hub.docker.com/r/nvidia/cuda/tags?name=base-ubuntu22.04) as a base.
   2. Only includes essentials (CUDA, `torch`, `torchvision`, `torchaudio`),
      so it has a small image size, making it fast to launch.
2. `nccl`: Tagged as `ml-containers/torch:a1b2c3d-nccl-...`.
   1. Built from [`ghcr.io/coreweave/nccl-tests`](https://github.com/coreweave/nccl-tests/pkgs/container/nccl-tests) as a base.
   2. Ultimately inherits from [`nvidia/cuda:...-cudnn8-devel-ubuntu22.04`](https://hub.docker.com/r/nvidia/cuda/tags?name=cudnn8-devel-ubuntu22.04).
   3. Larger, but includes development libraries and build tools such as `nvcc` necessary for compiling other PyTorch extensions.
   4. These PyTorch builds are built on component libraries optimized for the CoreWeave cloud&mdash;see
      [`coreweave/nccl-tests`](https://github.com/coreweave/nccl-tests/blob/master/README.md).

> [!NOTE]
> Most `torch` images have both a variant built on Ubuntu 22.04 and a variant built on Ubuntu 20.04.
> - CUDA 11.8.0 is an exception, and is only available on Ubuntu 20.04.
> - Ubuntu 22.04 images use Python 3.10.
> - Ubuntu 20.04 images use Python 3.8.
> - The base distribution is indicated in the container image tag.

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

### PyTorch Nightly

- [`ghcr.io/coreweave/ml-containers/nightly-torch`](https://github.com/coreweave/ml-containers/pkgs/container/ml-containers%2Fnightly-torch)
- [`ghcr.io/coreweave/ml-containers/nightly-torch-extras`](https://github.com/coreweave/ml-containers/pkgs/container/ml-containers%2Fnightly-torch-extras)

[`ml-containers/nightly-torch`](https://github.com/coreweave/ml-containers/pkgs/container/ml-containers%2Fnightly-torch)
is an experimental, nightly release channel of the
[PyTorch Base Images](#pytorch-base-images) in the style of PyTorch's
own nightly preview builds, featuring the latest development versions of
`torch`, `torchvision`, and `torchaudio` pulled daily from GitHub
and compiled from source.

[`ml-containers/nightly-torch-extras`](https://github.com/coreweave/ml-containers/pkgs/container/ml-containers%2Fnightly-torch-extras)
is a version of [PyTorch Extras](#pytorch-extras) built on top of the
[`ml-containers/nightly-torch`](https://github.com/coreweave/ml-containers/pkgs/container/ml-containers%2Fnightly-torch)
container images.
These are not nightly versions of the extensions themselves, but rather match
the extension versions in the regular [PyTorch Extras](#pytorch-extras) containers.

> ⚠ The *PyTorch Nightly* containers are based on unstable, experimental preview
builds of PyTorch, and should be expected to contain bugs and other issues.
> For more stable containers use the [PyTorch Base Images](#pytorch-base-images)
> and [PyTorch Extras](#pytorch-extras) containers. 


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
