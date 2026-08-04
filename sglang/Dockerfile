# syntax=docker/dockerfile:1.10
ARG BASE_IMAGE="lmsysorg/sglang:kimi-k3@sha256:6d9594a421be244f2af29d726158ebffe9c3c2b3f39b5b89affd8150a106e187"

FROM ${BASE_IMAGE}

ARG SGLANG_COMMIT
LABEL org.opencontainers.image.revision="${SGLANG_COMMIT}"

# The published K3 image still contains 0.1.4.post1, while the current K3
# recipe requires the first DeepGEMM release with K3 SiTU support.
RUN python3 -m pip install --no-cache-dir --no-deps --force-reinstall \
      'sgl-deep-gemm==0.1.5.post1'

# Keep this branch a thin, immutable republish of SGLang's complete Kimi-K3
# image. These checks fail the build if the source overlay, model integration,
# DeepEP dependency, or published FlashInfer SiTU cubin pool is absent.
RUN test "${SGLANG_COMMIT}" = "c6ad1f2602f2d485b774d305c84ea56d931fe738" && \
    test "${SGLANG_TRTLLM_GEN_MOE_CUBIN_POOL}" = "/opt/trtllm_gen_moe_cubin_pool" && \
    test "$(find "${SGLANG_TRTLLM_GEN_MOE_CUBIN_POOL}" -type f -name '*.cubin' | wc -l)" -eq 1696 && \
    python3 -c 'from importlib.metadata import version; from importlib.util import find_spec; from sglang.srt.models.kimi_k3 import KimiK3ForConditionalGeneration; expected = "0.6.15.post1"; packages = ("flashinfer-python", "flashinfer-cubin", "flashinfer-jit-cache"); actual = {package: version(package).split("+", 1)[0] for package in packages}; state = {"model": KimiK3ForConditionalGeneration.__name__, "sglang": version("sglang"), "deep_ep": find_spec("deep_ep") is not None, "deep_gemm": version("sgl-deep-gemm"), "flashinfer": actual}; print(state, flush=True); assert state["deep_ep"], state; assert all(value == expected for value in actual.values()), state; assert state["deep_gemm"] == "0.1.5.post1", state'
