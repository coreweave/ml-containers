import argparse
import ast
from email.parser import BytesParser
import hashlib
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import zipfile


CANDIDATE_COMMIT = "96d91ef9266d2bebd8e8c09ef1f28b2d521631ff"
REQUIRED_VERSIONS = {
    "torch": "2.13.0",
    "torchvision": "0.28.0",
    "torchaudio": "2.11.0",
    "flashinfer-python": "0.6.18",
    "sglang-kernel": "0.4.6.post1",
    "apache-tvm-ffi": "0.1.11",
    "cuda-tile": "1.6.0rc5",
    "nvidia-cutlass-dsl": "4.6.2",
    "sgl-deep-ep": "0.1.2",
    "sgl-deep-gemm": "0.1.7",
    "tilelang": "0.1.12",
    "transformers": "5.12.1",
    "tokenizers": "0.22.2",
}
SOURCE_FEATURES = {
    "python/sglang/srt/configs/glm5_next.py": ("Glm5NextConfig", "Glm5NextTextConfig"),
    "python/sglang/srt/models/glm5_next.py": ("Glm5NextForConditionalGeneration",),
    "python/sglang/srt/models/glm5_next_nextn.py": ("Glm5NextForConditionalGenerationNextN",),
    "python/sglang/srt/layers/quantization/modelopt_quant.py": ("ModelOptFp4Config",),
    "python/pyproject.toml": (),
    "python/sglang/kernels/aot/pyproject.toml": (),
}


def canonical_name(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_commit(commit):
    if commit != CANDIDATE_COMMIT:
        raise ValueError("GLM53 candidate preflight requires the reviewed immutable SGLang commit")
    if any(key.startswith("SETUPTOOLS_SCM_PRETEND_") for key in os.environ):
        raise ValueError("SCM overrides are not allowed for candidate evidence")


def source_features(source_dir):
    identities = {}
    for relative, required_classes in SOURCE_FEATURES.items():
        path = source_dir / relative
        if required_classes:
            classes = {node.name for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef)}
            if not set(required_classes).issubset(classes):
                raise ValueError(f"{relative}: missing required classes {required_classes}")
        identities[relative] = sha256(path)
    return identities


def source_evidence(source_dir, commit):
    validate_commit(commit)
    observed = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source_dir, text=True
    ).strip()
    if observed != commit:
        raise ValueError(f"Checked-out SGLang commit differs: {observed}")
    subprocess.run(
        ["git", "diff", "--exit-code", "--ignore-submodules=none", "HEAD", "--"],
        cwd=source_dir,
        check=True,
    )
    submodules = subprocess.check_output(
        ["git", "submodule", "status", "--recursive"], cwd=source_dir, text=True
    ).splitlines()
    if any(line.startswith(("-", "+", "U")) for line in submodules):
        raise ValueError("Candidate submodules do not match their recorded revisions")
    return {
        "source_commit": observed,
        "submodules": submodules,
        "source_files_sha256": source_features(source_dir),
    }


def wheel_evidence(wheel_dir):
    result = {}
    for path in sorted(wheel_dir.glob("*.whl")):
        with zipfile.ZipFile(path) as wheel:
            members = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
            if len(members) != 1:
                raise ValueError(f"{path.name}: expected exactly one wheel METADATA")
            package = BytesParser().parsebytes(wheel.read(members[0]))
        name = canonical_name(package["Name"])
        if name not in ("sglang", "sglang-kernel"):
            continue
        version = package["Version"]
        if not version or version.startswith("0.0.0") or name in result:
            raise ValueError(f"{path.name}: missing/fallback version or duplicate {name} wheel")
        result[name] = {
            "filename": path.name,
            "sha256": sha256(path),
            "version": version,
            "requires_dist": package.get_all("Requires-Dist", []),
        }
    if set(result) != {"sglang", "sglang-kernel"}:
        raise ValueError("Candidate requires both built SGLang and sglang-kernel wheels")
    return result


def validate_dependencies(wheels, versions):
    for name, expected in REQUIRED_VERSIONS.items():
        observed = versions.get(name, "")
        if observed.partition("+")[0] != expected:
            raise ValueError(f"{name}: expected {expected}, observed {observed!r}")
    for name, wheel in wheels.items():
        if versions.get(name) != wheel["version"]:
            raise ValueError(f"Installed {name} differs from the built wheel version")


def runtime_features():
    module_names = (
        "torch",
        "flashinfer",
        "sgl_kernel",
        "sglang.srt.configs.glm5_next",
        "sglang.srt.models.glm5_next",
        "sglang.srt.models.glm5_next_nextn",
        "sglang.srt.layers.quantization.modelopt_quant",
    )
    modules = {name: importlib.import_module(name) for name in module_names}
    config_type = modules["sglang.srt.configs.glm5_next"].Glm5NextConfig
    target = modules["sglang.srt.models.glm5_next"].Glm5NextForConditionalGeneration
    draft = modules["sglang.srt.models.glm5_next_nextn"].Glm5NextForConditionalGenerationNextN
    quant_type = modules["sglang.srt.layers.quantization.modelopt_quant"].ModelOptFp4Config
    config = config_type(text_config={"num_hidden_layers": 45, "num_nextn_predict_layers": 1})
    if config.text_config.nextn_layer_ids != [45]:
        raise ValueError("Unexpected synthetic GLM5 NextN layer mapping")
    quant = quant_type(
        is_checkpoint_nvfp4_serialized=True,
        group_size=16,
        exclude_modules=[
            "model.language_model.embed_tokens",
            "model.language_model.layers.11.self_attn*",
            "model.language_model.layers.11.mlp.gate",
            "model.language_model.layers.11.mlp.shared_experts*",
            "model.visual*",
        ],
    )
    quant.apply_weight_name_mapper(target.hf_to_sglang_mapper)
    for name in (
        "model.embed_tokens",
        "model.layers.11.self_attn.fused_qkv_a_proj_with_mqa",
        "model.layers.11.mlp.gate",
        "model.layers.11.mlp.shared_experts",
        "visual.blocks.0.attn.qkv_proj",
    ):
        if not quant.is_layer_excluded(name):
            raise ValueError(f"Synthetic ModelOpt exclusion failed: {name}")
    if quant.is_layer_excluded("model.layers.11.mlp.experts"):
        raise ValueError("Synthetic routed experts were unexpectedly excluded")
    reason = target.shared_experts_fusion_disable_reason(config, quant)
    if not reason or "shared experts unquantized" not in reason:
        raise ValueError("Missing ModelOpt mixed-precision shared-expert fusion guard")
    mapped = draft.get_hf_to_sglang_mapper(config).apply_list(["model.layers.45.self_attn.q_b_proj.weight"])
    if mapped != ["model.decoder.self_attn.q_b_proj.weight"]:
        raise ValueError("Unexpected synthetic GLM5 draft weight-name mapping")
    return {
        "module_files": {name: module.__file__ for name, module in modules.items()},
        "torch_cuda_version": modules["torch"].version.cuda,
        "config_check": "synthetic-exclusions-fusion-and-layer45-mapping",
        "checkpoint_loading": "not-tested",
        "gpu_kernels_and_tp2_mtp": "not-tested",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("source", "wheels", "installed"))
    parser.add_argument("--commit", required=True)
    parser.add_argument("--source-dir", type=Path, default=Path("."))
    parser.add_argument("--wheel-dir", type=Path, default=Path("/wheels"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    validate_commit(args.commit)
    if args.phase in ("source", "wheels"):
        evidence = source_evidence(args.source_dir, args.commit)
        if args.phase == "source":
            observed_torch = metadata.version("torch").partition("+")[0]
            if observed_torch != REQUIRED_VERSIONS["torch"]:
                raise ValueError(f"Candidate requires Torch 2.13.0, observed {observed_torch}")
        else:
            evidence["wheels"] = wheel_evidence(args.wheel_dir)
    else:
        evidence = json.loads((args.wheel_dir / "sglang-build-evidence.json").read_text())
        if evidence["source_commit"] != args.commit:
            raise ValueError("Builder evidence has a different SGLang source commit")
        if evidence["wheels"] != wheel_evidence(args.wheel_dir):
            raise ValueError("Wheel contents changed since builder evidence was captured")
        subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
        versions = {canonical_name(dist.metadata["Name"]): dist.version for dist in metadata.distributions()}
        validate_dependencies(evidence["wheels"], versions)
        evidence["installed_distributions"] = dict(sorted(versions.items()))
        evidence["preflight"] = runtime_features()
        evidence["pip_check"] = "passed"
    evidence.update(phase=args.phase, python=sys.version, architecture=platform.machine())
    rendered = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
