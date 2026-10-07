#!/usr/bin/env python3
"""Constrain normal pip installs and record source-built SGLang/FI artifacts."""

import argparse
import ast
import base64
import csv
import email
import hashlib
import importlib
import importlib.metadata as metadata
import io
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import tomllib
import zipfile

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version


SGLANG_SOURCE = "e00930c5489053f26d86b179cee0d087f846acbb"
PROFILES = {
    "baseline": {
        "source": "69ff11fc4954396d98326656dc85debd2223f637",
        "flashinfer-python": "0.6.18",
        "nvidia-cutlass-dsl": "4.6.2",
        "quack-kernels": "0.6.4",
    },
}
FIXED = {"flash-attn-4": "4.0.0b19", "apache-tvm-ffi": "0.1.11"}
PROTOCOL_PINS = {
    "protobuf": "6.33.6",
    "grpcio": "1.84.0",
    "grpcio-health-checking": "1.81.1",
    "grpcio-reflection": "1.81.1",
}
BUILD_SELECTED = {
    "nvidia-cutlass-dsl", "quack-kernels", "apache-tvm-ffi", "cuda-tile",
}
NATIVE_PREFIXES = (
    "torch", "triton", "nvidia-", "cuda-", "nccl", "nixl", "nvshmem",
    "flash-attn", "sgl-", "sglang-kernel", "tilelang", "tokenspeed-",
    "humming-kernels", "apache-tvm-ffi", "quack-kernels",
)
LOADER_NAMES = {
    "transformers", "tokenizers", "safetensors", "huggingface-hub", "hf-xet",
    "runai-model-streamer", "runai-model-streamer-s3", "runai-model-streamer-gcs",
    "runai-model-streamer-azure", "compressed-tensors", "sentencepiece",
    "tiktoken", "blobfile",
}


def controlled_package(name):
    return (name.startswith(NATIVE_PREFIXES) or name in LOADER_NAMES or name in PROTOCOL_PINS
            or name == "sglang" or name.startswith("flashinfer-"))
KEY_MODULES = (
    "fused_moe_trtllm_sm100", "fp4_quantization_103", "fmha_gen", "trtllm_utils",
)
MAX_BUILD_INFO_BYTES = 128 * 1024
SG_FILES = (
    "python/pyproject.toml",
    "python/sglang/kernels/ops/attention/flash_attn/cute/pyproject.toml",
    "python/sglang/kernels/ops/attention/flash_attn/cute/utils.py",
    "python/sglang/srt/entrypoints/engine.py",
    "python/sglang/srt/layers/attention/flashinfer_mla_backend.py",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def distribution_identity(dist):
    name = canonicalize_name(dist.metadata["Name"])
    path = getattr(dist, "_path", None)
    require(path is not None, f"Distribution metadata location is unavailable: {name}")
    return {"name": name, "version": dist.version,
            "metadata_path": str(Path(path).resolve()),
            "location": str(Path(dist.locate_file("")).resolve())}


def installed(strict_names=(), *, evidence=None):
    strict = {canonicalize_name(name) for name in strict_names}
    groups = {}
    for dist in metadata.distributions():
        name = canonicalize_name(dist.metadata["Name"])
        groups.setdefault(name, []).append(distribution_identity(dist))
    result, duplicates, critical = {}, [], []
    for name, observations in sorted(groups.items()):
        unique = {row["metadata_path"]: row for row in observations}
        require(all(unique[row["metadata_path"]] == row for row in observations),
                f"Distribution metadata changed during enumeration: {name}")
        candidates = sorted(unique.values(), key=lambda row: row["metadata_path"])
        selected = distribution_identity(metadata.distribution(name))
        require(selected in candidates, f"Selected distribution metadata was not enumerated: {name}: {selected}")
        protected = controlled_package(name) or name in strict
        if len(candidates) > 1:
            observation = {"name": name, "selected": selected, "candidates": candidates}
            print(json.dumps({"schema": "distribution-selection/v1", "duplicate": observation,
                              "resolution": "importlib.metadata.distribution", "protected": protected,
                              "module_import_ownership_verified": False}, sort_keys=True), file=sys.stderr)
            require(not protected, "Duplicate protected distribution metadata: " + json.dumps(observation, sort_keys=True))
            duplicates.append(observation)
        if protected:
            critical.append(selected)
        result[name] = selected["version"]
    if evidence is not None:
        evidence.update(schema="distribution-selection/v1", resolution="importlib.metadata.distribution",
                        module_import_ownership_verified=False, critical_selected=critical, duplicates=duplicates)
    return result


def revision(source):
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip()


def source_requirements(source):
    document = tomllib.loads((Path(source) / "python/pyproject.toml").read_text())
    result = {}
    for value in document["project"]["dependencies"]:
        req = Requirement(value)
        if req.marker and not req.marker.evaluate():
            continue
        name = canonicalize_name(req.name)
        require(name not in result, f"Duplicate active SGLang requirement: {name}")
        result[name] = req
    return result


def exact_version(req):
    specs = list(req.specifier)
    if len(specs) == 1 and specs[0].operator == "==" and "*" not in specs[0].version:
        return specs[0].version
    return None


def source_contract(profile, sg_source, fi_source):
    policy = PROFILES[profile]
    require(revision(sg_source) == SGLANG_SOURCE, "Unreviewed SGLang source")
    require(revision(fi_source) == policy["source"], "Wrong FlashInfer source/profile")
    require((Path(fi_source) / "version.txt").read_text().strip() == policy["flashinfer-python"],
            "FlashInfer source version does not match profile")
    requirements = source_requirements(sg_source)
    for name, version in {**policy, **FIXED}.items():
        if name == "source":
            continue
        require(name in requirements, f"Missing SGLang requirement: {name}")
        if name in policy or name == "apache-tvm-ffi":
            require(exact_version(requirements[name]) == version,
                    f"SGLang source does not select {name}=={version}")
        else:
            require(requirements[name].specifier.contains(version, prereleases=True),
                    f"SGLang does not permit selected {name}=={version}")
    return requirements


def torch_abi():
    code = (
        "import json,torch; print(json.dumps({'version':torch.__version__,"
        "'cuda':torch.version.cuda,'cxx11_abi':bool(torch._C._GLIBCXX_USE_CXX11_ABI),"
        "'cuda_initialized':torch.cuda.is_initialized()}))"
    )
    result = json.loads(subprocess.check_output([sys.executable, "-c", code], text=True, timeout=60))
    require(result["cuda_initialized"] is False, "ABI observation unexpectedly initialized CUDA")
    return result


def source_evidence(sg_source):
    return {"files": {name: sha256(Path(sg_source) / name) for name in SG_FILES}}


def make_constraints(requirements, before):
    selected = {name: exact_version(req) for name, req in requirements.items()
                if exact_version(req) is not None}
    selected.update(FIXED)
    selected.update(PROTOCOL_PINS)
    selected["flashinfer-cubin"] = selected["flashinfer-python"]
    selected["flashinfer-jit-cache"] = selected["flashinfer-python"]
    optional = dsl_sibling_constraints(selected)
    preserved = {name: version for name, version in before.items()
                 if controlled_package(name) and name not in selected and name not in optional}
    for name in ("torch", "torchvision", "torchaudio", "triton"):
        require(name in before, f"Required base native package missing: {name}")
        if name in requirements:
            require(requirements[name].specifier.contains(before[name], prereleases=True),
                    f"Base {name} violates SGLang requirement")
        selected.pop(name, None)
        preserved[name] = before[name]
    pins = {**preserved, **selected, **optional}
    for name, req in requirements.items():
        if name in pins:
            require(req.specifier.contains(pins[name], prereleases=True),
                    f"Constraint for {name} conflicts with SGLang")
    return selected, preserved, pins


def dsl_sibling_constraints(selected):
    return {f"nvidia-cutlass-dsl-libs-{suffix}": selected["nvidia-cutlass-dsl"]
            for suffix in ("base", "core", "cu12", "cu13")}


def constraints(args):
    reqs = source_contract(args.profile, args.sglang_source, args.flashinfer_source)
    selection = {}
    strict = {name for name, req in reqs.items() if exact_version(req) is not None} | set(FIXED)
    before = installed(strict, evidence=selection)
    selected, preserved, pins = make_constraints(reqs, before)
    lines = [f"{name}=={version}" for name, version in sorted(pins.items())]
    Path(args.output).write_text("\n".join(lines) + "\n")
    build = [str(reqs[name]) for name in sorted(BUILD_SELECTED)]
    build.append(f"flash-attn-4=={FIXED['flash-attn-4']}")
    build.extend(str(reqs[name]) for name in sorted(LOADER_NAMES & reqs.keys()))
    sg_document = tomllib.loads((Path(args.sglang_source) / "python/pyproject.toml").read_text())
    runai = sg_document["project"]["optional-dependencies"]["runai"]
    require(len(runai) == 1 and canonicalize_name(Requirement(runai[0]).name) == "runai-model-streamer",
            "Expected source-owned RunAI loader extra")
    build.extend(runai)
    for line in (Path(args.flashinfer_source) / "requirements.txt").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            req = Requirement(line)
            if canonicalize_name(req.name) not in BUILD_SELECTED:
                build.append(str(req))
            name = canonicalize_name(req.name)
            if name in pins and (not req.marker or req.marker.evaluate()):
                require(req.specifier.contains(pins[name], prereleases=True),
                        f"Inherited/selected {name} conflicts with FlashInfer source; review the native stack")
    Path(args.requirements).write_text("\n".join(dict.fromkeys(build)) + "\n")
    write_json(args.snapshot, {
        "schema": "sglang-package-constraints/v1", "profile": args.profile,
        "sources": {"sglang": SGLANG_SOURCE, "flashinfer": PROFILES[args.profile]["source"]},
        "before": before, "selected_versions": selected, "preserved_versions": preserved,
        "distribution_selection_before": selection,
        "optional_versions": dsl_sibling_constraints(selected),
        "constraints_sha256": sha256(args.output),
        "build_requirements_sha256": sha256(args.requirements),
        "sglang_pyproject_sha256": sha256(Path(args.sglang_source) / "python/pyproject.toml"),
        "flashinfer_requirements_sha256": sha256(Path(args.flashinfer_source) / "requirements.txt"),
        "sglang_source_evidence": source_evidence(args.sglang_source),
        "torch_abi": torch_abi(),
    })


def protocol_requirements(args):
    Path(args.output).write_text("\n".join(
        f"{name}=={version}" for name, version in sorted(PROTOCOL_PINS.items())) + "\n")


def validate_protocol_metadata(versions, requirements):
    for name, expected in PROTOCOL_PINS.items():
        require(versions.get(name) == expected,
                f"Protocol dependency mismatch: {name}; expected={expected}; actual={versions.get(name, '<missing>')}")
    checked = []
    for owner in PROTOCOL_PINS:
        for value in requirements[owner]:
            req = Requirement(value)
            name = canonicalize_name(req.name)
            if name not in PROTOCOL_PINS or (req.marker and not req.marker.evaluate({"extra": ""})):
                continue
            require(req.specifier.contains(versions[name], prereleases=True),
                    f"Protocol metadata conflict: {owner} requires {req}; selected {name}=={versions[name]}")
            checked.append({"distribution": owner, "requirement": str(req)})
    return checked


def verify_protocol(args):
    selection = {}
    versions = installed(PROTOCOL_PINS, evidence=selection)
    requirements = {name: metadata.distribution(name).requires or [] for name in PROTOCOL_PINS}
    checked = validate_protocol_metadata(versions, requirements)
    import grpc
    import google.protobuf
    require(grpc.__version__ == PROTOCOL_PINS["grpcio"], "Imported grpc runtime differs from selected metadata")
    require(google.protobuf.__version__ == PROTOCOL_PINS["protobuf"],
            "Imported protobuf runtime differs from selected metadata")
    module_rows, modules = [], {}
    for distribution, prefix in (("grpcio-health-checking", "grpc_health.v1.health"),
                                 ("grpcio-reflection", "grpc_reflection.v1alpha.reflection")):
        dist = metadata.distribution(distribution)
        owned = {str(path) for path in dist.files or []}
        for suffix in ("_pb2", "_pb2_grpc"):
            name = prefix + suffix
            relative = name.replace(".", "/") + ".py"
            require(relative in owned, f"Generated protocol module is not distribution-owned: {name}")
            module = importlib.import_module(name)
            path = Path(dist.locate_file(relative)).resolve()
            require(Path(module.__file__).resolve() == path and path.is_file(),
                    f"Imported protocol module differs from selected distribution: {name}")
            modules[name] = module
            module_rows.append({"distribution": distribution, "module": name,
                                "sha256": sha256(path), "bytes": path.stat().st_size})
    health = modules["grpc_health.v1.health_pb2"]
    reflection = modules["grpc_reflection.v1alpha.reflection_pb2"]
    require(callable(modules["grpc_health.v1.health_pb2_grpc"].HealthStub)
            and callable(modules["grpc_reflection.v1alpha.reflection_pb2_grpc"].ServerReflectionStub),
            "Generated gRPC stubs are unavailable")
    messages = [health.HealthCheckRequest(service="protocol-build-check"),
                health.HealthCheckResponse(status=health.HealthCheckResponse.SERVING),
                reflection.ServerReflectionRequest(file_containing_symbol="grpc.health.v1.Health"),
                reflection.ServerReflectionResponse(list_services_response=reflection.ListServiceResponse(
                    service=[reflection.ServiceResponse(name="grpc.health.v1.Health")]))]
    roundtrips = []
    for message in messages:
        payload = message.SerializeToString(deterministic=True)
        require(0 < len(payload) <= 4096 and type(message).FromString(payload) == message,
                "Generated protocol binary roundtrip failed")
        roundtrips.append({"message": message.DESCRIPTOR.full_name, "bytes": len(payload),
                           "sha256": hashlib.sha256(payload).hexdigest()})
    result = {"schema": "sglang-protocol-check/v1", "status": "PASS", "versions": PROTOCOL_PINS,
              "checked_requirements": checked, "generated_modules": module_rows, "roundtrips": roundtrips,
              "scope": "Selected metadata, generated imports and binary serialization; no GPU or RPC execution"}
    write_json(args.output, result)
    print(json.dumps(result, sort_keys=True))

def architectures(value):
    result = []
    for arch in value.split():
        match = re.fullmatch(r"(\d{1,2})\.(\d)([af]?)", arch)
        require(match is not None, f"Unsupported exact architecture: {arch}")
        result.append("sm" + "".join(match.groups()))
    require(result and len(result) == len(set(result)), "Empty/duplicate architecture list")
    require("sm103a" in result, "The candidate must include SM103a")
    return sorted(result)


def literal_build_metadata(archive):
    candidates = [name for name in archive.namelist() if name.endswith("/_build_meta.py")]
    require(len(candidates) == 1, "Expected one FlashInfer build metadata file")
    values = {}
    for node in ast.parse(archive.read(candidates[0]).decode()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            values[node.targets[0].id] = ast.literal_eval(node.value)
    return values


def installed_path(path):
    parts = path.split("/")
    if len(parts) > 2 and parts[0].endswith(".data") and parts[1] in ("purelib", "platlib"):
        return "/".join(parts[2:])
    return path


def cubin_headers(archive):
    """Retain only the distribution-owned external header subtree."""
    marker = "/include/trtllmGen_bmm_export/"
    selected = [n for n in archive.namelist() if marker in installed_path(n) and not n.endswith("/")]
    require(0 < len(selected) <= 512, "Expected bounded cubin header subtree")
    roots = {installed_path(n).split(marker)[0] + marker.rstrip("/") for n in selected}
    require(len(roots) == 1, "Multiple cubin header subtree roots")
    root = roots.pop()
    require(not root.startswith("/") and all(p not in ("", ".", "..") for p in root.split("/")),
            "Unsafe cubin header root")
    records = [n for n in archive.namelist() if n.endswith(".dist-info/RECORD")]
    require(len(records) == 1, "One wheel RECORD required")
    record_rows = list(csv.reader(io.StringIO(archive.read(records[0]).decode())))
    require(all(len(r) == 3 for r in record_rows) and len(record_rows) == len({r[0] for r in record_rows}),
            "Invalid or duplicate wheel RECORD")
    record = {r[0]: r[1:] for r in record_rows}
    rows = []
    for name in selected:
        info = archive.getinfo(name)
        mode = info.external_attr >> 16
        require(stat.S_ISREG(mode) and stat.S_IMODE(mode) == 0o644, "Header wheel mode must be regular 0644")
        rel = installed_path(name)[len(root) + 1:]
        require(all(p not in ("", ".", "..") for p in rel.split("/")), "Unsafe header member")
        data = archive.read(name)
        checksum = hashlib.sha256(data).hexdigest()
        encoded = base64.urlsafe_b64encode(bytes.fromhex(checksum)).decode().rstrip("=")
        require(record.get(name) == ["sha256=" + encoded, str(len(data))], "Header bytes differ from wheel RECORD")
        rows.append({"path": rel, "sha256": checksum, "bytes": len(data), "mode": 0o644})
    paths = {r["path"] for r in rows}
    require(len(paths) == len(rows), "Duplicate relocated header member")
    for row in rows:
        path = row["path"]
        header = path.removesuffix(".lock")
        require(Path(header).suffix in (".h", ".hpp", ".cuh", ".inl")
                and (not path.endswith(".lock") or (row["bytes"] == 0 and header in paths)),
                "Only regular headers and their empty lock files are allowed")
    directories = sorted({str(parent) for r in rows for parent in Path(r["path"]).parents if str(parent) != "."})
    return {"path": root, "files": sorted(rows, key=lambda r: r["path"]), "directories": directories}


def inspect_wheel(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), f"Duplicate ZIP entries: {path}")
        meta = [name for name in names if name.endswith(".dist-info/METADATA")]
        require(len(meta) == 1, f"Expected one METADATA: {path}")
        value = email.message_from_bytes(archive.read(meta[0]))
        name = canonicalize_name(value["Name"])
        result = {"name": name, "version": value["Version"], "filename": Path(path).name,
                  "sha256": sha256(path), "bytes": Path(path).stat().st_size,
                  "requires": value.get_all("Requires-Dist", [])}
        if name.startswith("flashinfer-"):
            result["build_metadata"] = literal_build_metadata(archive)
        if name == "flashinfer-cubin":
            result["header_subtree"] = cubin_headers(archive)
        result["key_modules"] = {}
        for module in KEY_MODULES:
            paths = [n for n in names if n.endswith(f"/{module}/{module}.so")]
            if paths:
                require(len(paths) == 1, f"Duplicate key module: {module}")
                payload = archive.read(paths[0])
                require(payload.startswith(b"\x7fELF"), f"Key module is not ELF: {module}")
                result["key_modules"][module] = {
                    "path": installed_path(paths[0]), "wheel_path": paths[0],
                    "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
                }
        return result


def validate_wheel_set(profile, rows, source, cuda_local=None):
    require(len(rows) == len({row["name"] for row in rows}), "Duplicate wheel distributions")
    by_name = {row["name"]: row for row in rows}
    required = {"sglang", "sglang-kernel", "flashinfer-python", "flashinfer-cubin", "flashinfer-jit-cache"}
    require(required <= set(by_name), f"Missing built distributions: {required - set(by_name)}")
    fi_rows = [row for row in rows if row["name"].startswith("flashinfer-")]
    require({row["name"] for row in fi_rows} == {n for n in required if n.startswith("flashinfer-")},
            "Unexpected FlashInfer distribution in wheelhouse")
    versions = {row["version"] for row in fi_rows}
    require(len(versions) == 1, "FlashInfer package versions/local CUDA labels differ")
    version = versions.pop()
    require(Version(version).public == PROFILES[profile]["flashinfer-python"], "Wrong FI public version")
    require(re.fullmatch(r"cu13\d+", Version(version).local or ""), "Missing exact CUDA13 local label")
    if cuda_local is not None:
        require(Version(version).local == cuda_local, "FI wheel CUDA label differs from active nvcc")
    for row in fi_rows:
        build = row["build_metadata"]
        require(build.get("__version__") == version, "Wheel build metadata/version mismatch")
        require(build.get("__git_version__", build.get("__git_commit__")) == source,
                "FlashInfer wheel source mismatch")
    critical = by_name["flashinfer-jit-cache"]
    require(set(critical["key_modules"]) == set(KEY_MODULES), "Missing GLM SM103 key AOT modules")
    return critical


def inspect_baseline_device_code(path, row, cuobjdump):
    require(Path(cuobjdump).is_file(), "cuobjdump is required for baseline architecture proof")
    member = row["key_modules"]["fp4_quantization_103"]["wheel_path"]
    with zipfile.ZipFile(path) as archive, tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "fp4.so"
        target.write_bytes(archive.read(member))
        result = subprocess.run([cuobjdump, "--list-elf", str(target)], check=True, capture_output=True, text=True)
    targets = sorted(set(re.findall(r"\bsm[_-]?(\d{2,3}[af]?)\b", result.stdout)))
    require(set(targets) & {"103", "103a"}, "Baseline FP4 wrapper lacks SM103 device code")
    return {"module": "fp4_quantization_103", "cuda_architectures": ["sm" + x for x in targets]}


def newly_resolved_native(snapshot, versions):
    constrained = set(snapshot["selected_versions"]) | set(snapshot["preserved_versions"]) | set(snapshot["optional_versions"])
    for name, version in snapshot["preserved_versions"].items():
        require(versions.get(name) == version, f"Inherited native dependency changed during build: {name}")
    return {name: version for name, version in versions.items()
            if controlled_package(name) and name not in constrained}


def wheels(args):
    source_contract(args.profile, args.sglang_source, args.flashinfer_source)
    snapshot = json.loads(Path(args.snapshot).read_text())
    require(snapshot["profile"] == args.profile and snapshot["constraints_sha256"] == sha256(args.constraints),
            "Constraint snapshot does not match this build")
    targets = architectures(args.architectures)
    nvcc_version = subprocess.check_output([str(Path(args.cuobjdump).with_name("nvcc")), "--version"], text=True)
    release = re.search(r"release\s+(\d+)\.(\d+),", nvcc_version)
    require(release is not None and release[1] == "13", "Expected CUDA13 nvcc release")
    cuda_local = "cu" + release[1] + release[2]
    evidence = source_evidence(args.sglang_source)
    require(evidence == snapshot["sglang_source_evidence"], "SGLang sources changed during build")
    require(torch_abi() == snapshot["torch_abi"], "Torch ABI changed during build")
    paths = sorted(Path(args.wheel_dir).glob("*.whl"))
    rows = [inspect_wheel(path) for path in paths]
    critical = validate_wheel_set(args.profile, rows, PROFILES[args.profile]["source"], cuda_local)
    reports = {}
    path = Path(args.wheel_dir) / critical["filename"]
    reports["sm103a"] = inspect_baseline_device_code(path, critical, args.cuobjdump)
    selection = {}
    strict = set(snapshot["selected_versions"]) | set(snapshot["preserved_versions"]) | set(snapshot["optional_versions"])
    added_native = newly_resolved_native(snapshot, installed(strict, evidence=selection))
    if added_native:
        with Path(args.constraints).open("a") as stream:
            stream.write("\n# Native dependencies resolved while preparing this build.\n")
            stream.writelines(f"{name}=={version}\n" for name, version in sorted(added_native.items()))
    write_json(args.output, {
        "schema": "sglang-package-wheels/v1", "profile": args.profile,
        "sources": snapshot["sources"], "architectures": targets,
        "constraints_sha256": sha256(args.constraints), "snapshot_sha256": sha256(args.snapshot),
        "initial_constraints_sha256": snapshot["constraints_sha256"], "resolved_native_versions": added_native,
        "selected_versions": snapshot["selected_versions"], "preserved_versions": snapshot["preserved_versions"],
        "optional_versions": snapshot["optional_versions"],
        "wheels": rows, "provider_validation": reports,
        "distribution_selection_build": selection,
        "sglang_source_evidence": evidence, "torch_abi": snapshot["torch_abi"],
        "cuda_version": os.environ.get("CUDA_VERSION"),
        "nvcc_version": nvcc_version,
    })


def validate_installed(info, versions):
    for name, version in {**info["preserved_versions"], **info.get("resolved_native_versions", {})}.items():
        origin = "builder-resolved" if name in info.get("resolved_native_versions", {}) else "inherited"
        require(versions.get(name) == version,
                f"Pinned native/loader dependency mismatch: {name}; origin={origin}; "
                f"expected={version}; actual={versions.get(name, '<missing>')}")
    for name, version in info["selected_versions"].items():
        # A local CUDA version is permitted for a source-owned public version pin.
        require(name in versions and Requirement(f"{name}=={version}").specifier.contains(versions[name], prereleases=True),
                f"Selected dependency mismatch: {name}=={version}")
    for name, version in info.get("optional_versions", {}).items():
        if name in versions:
            require(versions[name] == version, f"Installed optional dependency mismatch: {name}")
    for row in info["wheels"]:
        require(versions.get(row["name"]) == row["version"], f"Built wheel was replaced: {row['name']}")


def audit_installed(info, selection=None):
    strict = set(info["selected_versions"]) | set(info["preserved_versions"]) | set(info.get("optional_versions", {}))
    strict.update(info.get("resolved_native_versions", {}))
    strict.update(row["name"] for row in info["wheels"])
    versions = installed(strict, evidence=selection)
    validate_installed(info, versions)
    require(torch_abi() == info["torch_abi"], "Installed Torch ABI differs from wheel build")
    module_rows = []
    for row in info["wheels"]:
        dist = metadata.distribution(row["name"])
        for name, item in row.get("key_modules", {}).items():
            path = Path(dist.locate_file(item["path"]))
            require(path.is_file() and path.stat().st_size == item["bytes"]
                    and sha256(path) == item["sha256"], f"Installed AOT module changed: {name}")
            module_rows.append({"distribution": row["name"], "module": name,
                                "path": str(path), "sha256": item["sha256"], "bytes": item["bytes"]})
    return versions, module_rows


def compact_build_record(info, versions, module_rows, wheel_info_sha256, selection=None):
    record = json.loads(json.dumps(info))
    if "distribution_selection_build" in record:
        observed = record.pop("distribution_selection_build")
        record["distribution_selection_build_sha256"] = hashlib.sha256(
            json.dumps(observed, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    record.update(schema="sglang-package-build/v1", installed_versions=versions,
                  key_module_files=module_rows, wheel_info_sha256=wheel_info_sha256)
    if selection is not None:
        record["distribution_selection_installed"] = selection
    require(len((json.dumps(record, indent=2, sort_keys=True) + "\n").encode()) <= MAX_BUILD_INFO_BYTES,
            "Compact build-info exceeds its explicit size bound")
    return record


def audit(args):
    info = json.loads(Path(args.wheel_info).read_text())
    require(info["schema"] == "sglang-package-wheels/v1", "Unsupported wheel record")
    require(info["snapshot_sha256"] == sha256(args.snapshot)
            and info["constraints_sha256"] == sha256(args.constraints), "Build record input changed")
    selection = {}
    versions, module_rows = audit_installed(info, selection)
    record = compact_build_record(info, versions, module_rows, sha256(args.wheel_info), selection)
    write_json(args.output, record)


def verify_installed(args):
    info = json.loads(Path(args.build_info).read_text())
    require(info["schema"] == "sglang-package-build/v1" and info["profile"] == args.profile,
            "Unexpected inherited package build/profile")
    require(info["constraints_sha256"] == sha256(args.constraints), "Inherited constraints changed")
    require(info["wheel_info_sha256"] == sha256(Path(args.build_info).with_name("wheel-info.json")),
            "Retained full wheel evidence changed")
    selection = {}
    versions, modules = audit_installed(info, selection)
    write_json(args.output, {
        "schema": "sglang-package-final/v1", "profile": args.profile,
        "build_info_sha256": sha256(args.build_info),
        "constraints_sha256": sha256(args.constraints), "sources": info["sources"],
        "installed_versions": versions, "key_module_files": modules,
        "distribution_selection_installed": selection,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for command in ("constraints", "wheels", "audit"):
        sub = subs.add_parser(command)
        sub.add_argument("--output", required=True)
        sub.add_argument("--snapshot", required=True)
        if command != "audit":
            sub.add_argument("--profile", choices=PROFILES, required=True)
            sub.add_argument("--sglang-source", required=True)
            sub.add_argument("--flashinfer-source", required=True)
        if command == "constraints":
            sub.add_argument("--requirements", required=True)
        else:
            sub.add_argument("--constraints", required=True)
        if command == "wheels":
            sub.add_argument("--wheel-dir", required=True)
            sub.add_argument("--architectures", required=True)
            sub.add_argument("--cuobjdump", default="/usr/local/cuda/bin/cuobjdump")
        if command == "audit":
            sub.add_argument("--wheel-info", required=True)
        sub.set_defaults(function=globals()[command])
    for command, function in (("protocol-requirements", protocol_requirements), ("verify-protocol", verify_protocol)):
        sub = subs.add_parser(command)
        sub.add_argument("--output", required=True)
        sub.set_defaults(function=function)
    final = subs.add_parser("verify-installed")
    final.add_argument("--profile", choices=PROFILES, required=True)
    final.add_argument("--build-info", required=True)
    final.add_argument("--constraints", required=True)
    final.add_argument("--output", required=True)
    final.set_defaults(function=verify_installed)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
