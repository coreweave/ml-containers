variable "REGISTRY" { default = "ghcr.io/coreweave/ml-containers" }
variable "TAG"      { default = "" }

# Pinned stack root — override via env var or workflow_dispatch input to rebase the chain
variable "TORCH_EXTRAS_IMAGE" {
  default = "ghcr.io/coreweave/ml-containers/torch-extras:17ad6db-nccl-cuda12.9.1-ubuntu22.04-nccl2.29.2-1-torch2.10.0-vision0.25.0-audio2.10.0-abi1"
}

# sglang commit pins (match Dockerfile ARG defaults)
variable "FLASHINFER_COMMIT" { default = "v0.6.3" }
variable "SGLANG_COMMIT"     { default = "v0.5.9" }
variable "DECORD_COMMIT"     { default = "d2e56190286ae394032a8141885f76d5372bd44b" }

# slime commit pins (match Dockerfile ARG defaults)
variable "SLIME_SGLANG_COMMIT" { default = "bbe9c7eeb520b0a67e92d133dfc137a3688dc7f2" }
variable "APEX_COMMIT"         { default = "10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4" }
variable "SLIME_COMMIT"        { default = "7014942c17625732dbe9239c0aa9d8e7d7265ccc" }
variable "MEGATRON_COMMIT"     { default = "3714d81d418c9f1bca4594fc35f9e8289f652862" }

group "default" {
  targets = ["sglang", "slime"]
}

target "sglang" {
  context   = "./sglang"
  platforms = ["linux/amd64", "linux/arm64"]
  contexts  = { torch-extras = "docker-image://${TORCH_EXTRAS_IMAGE}" }
  args = {
    FLASHINFER_COMMIT          = FLASHINFER_COMMIT
    SGLANG_COMMIT              = SGLANG_COMMIT
    DECORD_COMMIT              = DECORD_COMMIT
    BUILD_TORCH_CUDA_ARCH_LIST = "8.0 8.6 8.9 9.0a 10.0a 12.0+PTX"
    MAX_JOBS                   = "8"
  }
  tags       = ["${REGISTRY}/sglang:${TAG}"]
  cache-from = ["type=registry,ref=${REGISTRY}/buildcache:sglang"]
  cache-to   = ["type=registry,ref=${REGISTRY}/buildcache:sglang,mode=max"]
}

target "slime" {
  context   = "./slime"
  platforms = ["linux/amd64", "linux/arm64"]
  contexts  = { sglang = "target:sglang" }
  args = {
    SGLANG_COMMIT   = SLIME_SGLANG_COMMIT
    APEX_COMMIT     = APEX_COMMIT
    SLIME_COMMIT    = SLIME_COMMIT
    MEGATRON_COMMIT = MEGATRON_COMMIT
  }
  tags       = ["${REGISTRY}/slime:${TAG}"]
  cache-from = ["type=registry,ref=${REGISTRY}/buildcache:slime"]
  cache-to   = ["type=registry,ref=${REGISTRY}/buildcache:slime,mode=max"]
}
