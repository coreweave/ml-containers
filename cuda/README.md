# CUDA Base Containers

This is a simplified port of NVIDIA's open source [CUDA container](https://gitlab.com/nvidia/container-images/cuda) builds.
These builds exist for filling in gaps where a CUDA release exists but doesn't have an official published container yet.

As derivatives, the `Dockerfile`s in this directory are subject to the license from NVIDIA's repository,
which is the *BSD 3-Clause "New" or "Revised" License*, available at `./LICENSE`.

The images built from these `Dockerfile`s contain NVIDIA software governed by NVIDIA's own licenses,
including the [CUDA EULA](https://docs.nvidia.com/cuda/eula/index.html)
and the [NVIDIA Deep Learning Container License](https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license).
