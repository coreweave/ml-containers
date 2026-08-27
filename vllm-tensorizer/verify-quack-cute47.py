import hashlib
from importlib import metadata, util
from pathlib import Path

EXPECTED_VERSIONS = {
    "quack-kernels": "0.6.4",
    "nvidia-cutlass-dsl": "4.7.0",
    "nvidia-cutlass-dsl-libs-base": "4.7.0",
    "nvidia-cutlass-dsl-libs-core": "4.7.0",
    "nvidia-cutlass-dsl-libs-cu12": "4.7.0",
    "nvidia-cutlass-dsl-libs-cu13": "4.7.0",
}
EXPECTED_HASHES = {
    "activation.py": "ac4f1db1a8f1ff4f0bd133d9729f948fc7d2ea8916e32e6788a095093ae2455c",
    "gemm_runtime/identity.py": "9794b6f315f5e78a7681ceb4cc6522c0f56b3549a3530e614f767a0816a6690f",
    "gemm_sm100.py": "a60ed28ddd2bb45ee4b054bcec9bbc50bbc7a3beeb86a12798037bfa85b13215",
    "epilogue/frontend.py": "1fc3c4a0914a23c04e69dfef03232f7efb85e03bbb8a7c6e515509d9582d6f4a",
}


versions = {name: metadata.version(name) for name in EXPECTED_VERSIONS}
assert versions == EXPECTED_VERSIONS, (versions, EXPECTED_VERSIONS)

vllm_requires = metadata.requires("vllm") or []
assert "nvidia-cutlass-dsl[cu13]==4.7.0" in vllm_requires, vllm_requires
assert not any("nvidia-cutlass-dsl[cu13]==4.6.2" in item for item in vllm_requires)

spec = util.find_spec("quack")
assert spec is not None and spec.submodule_search_locations
root = Path(next(iter(spec.submodule_search_locations)))
hashes = {
    name: hashlib.sha256((root / name).read_bytes()).hexdigest()
    for name in EXPECTED_HASHES
}
assert hashes == EXPECTED_HASHES, (hashes, EXPECTED_HASHES)
print("QuACK/CuTe 4.7 source and versions verified", versions, hashes)
