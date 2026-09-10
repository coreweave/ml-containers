# DeepSeek V4.1 development build

The V4.1 candidate pins SGLang PR38798's `dsv4.1` branch at
`c36636b7da601639d4a8b19a5d008761252146a9`, including the Blackwell DSpark/MoE
follow-up PR38879. Existing released build variants remain available.

V4.1 adds model/runtime code, CUDA AOT kernels and runtime JIT kernels, so a
weight-only update is insufficient. The current build already handles its
`python/sglang/kernels/aot` package and root Rust workspace. The candidate uses
the existing CoreWeave Torch2.13/CUDA13.2.1 base, matching the preview's Torch
major/minor while retaining CoreWeave's CUDA/NCCL stack.

The resulting image records the upstream source revision in an image label and
`SGLANG_BUILD_COMMIT`. The public `lmsysorg/sglang:dev-dsv41` ARM image instead
reports an unknown commit and `sglang==0.0.0.dev0`; its preview test is separate
from validation of this source-built image.

This branch is a development candidate. Building the image is not proof of
correct GB300 inference; model startup, semantic checks and DSpark counters are
required before promotion.
