"""Portable rejection tests for package/source and AOT coverage boundaries."""

import copy
import base64
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from packaging.requirements import Requirement

import packages as p


def requirements(profile="baseline"):
    selected = p.PROFILES[profile]
    values = {
        "flashinfer-python": f"flashinfer_python[cu13]=={selected['flashinfer-python']}",
        "nvidia-cutlass-dsl": f"nvidia-cutlass-dsl[cu13]=={selected['nvidia-cutlass-dsl']}",
        "quack-kernels": f"quack-kernels=={selected['quack-kernels']}",
        "apache-tvm-ffi": "apache-tvm-ffi==0.1.11",
        "flash-attn-4": "flash-attn-4>=4.0.0b18",
        "cuda-tile": "cuda-tile==1.6.0rc5",
        "torch": "torch>=2.13.0",
        "torchaudio": "torchaudio>=2.11.0",
    }
    return {name: Requirement(value) for name, value in values.items()}


def base():
    return {"torch": "2.13.0", "torchvision": "0.28.0", "torchaudio": "2.11.0",
            "triton": "3.7.1+git5d6048aa", "nixl-cu13": "1.4.0",
            "cuda-bindings": "13.3.1", "nccl4py": "0.5.0"}


def wheel_rows(profile="upgrade"):
    version = p.PROFILES[profile]["flashinfer-python"] + "+cu132"
    names = ["sglang", "sglang-kernel", "flashinfer-python", "flashinfer-cubin", "flashinfer-jit-cache"]
    if profile == "upgrade":
        names.append("flashinfer-jit-cache-sm103a")
    rows = []
    for name in names:
        row = {"name": name, "version": version, "filename": name + ".whl",
               "bytes": 123, "sha256": "a" * 64, "requires": [], "key_modules": {}}
        if name.startswith("flashinfer-"):
            row["build_metadata"] = {"__version__": version, "__git_version__": p.PROFILES[profile]["source"]}
        if name == "flashinfer-jit-cache-sm103a":
            row["provider_manifest"] = {
                "schema_version": 1, "provider_id": "sm103a", "distribution": name,
                "version": version, "cuda_architectures": ["sm103a"], "modules": list(p.KEY_MODULES),
            }
        rows.append(row)
    critical = rows[-1] if profile == "upgrade" else rows[4]
    critical["key_modules"] = {mod: {"path": mod + ".so", "sha256": "b" * 64, "bytes": 4}
                               for mod in p.KEY_MODULES}
    if profile == "upgrade":
        rows[4]["requires"] = ["flashinfer-jit-cache-sm103a==" + version]
    return rows


class DistributionSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def distribution(self, root, name, version):
        path = self.root / root / (name.replace("-", "_") + "-" + version + ".dist-info")
        path.mkdir(parents=True)
        (path / "METADATA").write_text(f"Name: {name}\nVersion: {version}\n")
        return importlib.metadata.Distribution.at(path)

    def lookup(self, observations, search_roots, *, strict=(), evidence=None):
        with patch.object(p.metadata, "distributions", return_value=observations), \
             patch.object(sys, "path", [str(self.root / value) for value in search_roots]), \
             patch.object(sys, "stderr", new_callable=io.StringIO) as diagnostics:
            result = p.installed(strict, evidence=evidence)
        return result, diagnostics.getvalue()

    def test_unrelated_duplicate_uses_python_lookup_not_enumeration_order(self):
        selected = self.distribution("pip", "cryptography", "46.0.0")
        shadowed = self.distribution("system", "cryptography", "41.0.7")
        evidence = {}
        result, diagnostic = self.lookup([shadowed, selected], ["pip", "system"], evidence=evidence)
        self.assertEqual(result, {"cryptography": "46.0.0"})
        row = evidence["duplicates"][0]
        self.assertEqual(row["selected"]["version"], "46.0.0")
        self.assertEqual({x["version"] for x in row["candidates"]}, {"41.0.7", "46.0.0"})
        self.assertEqual(len({x["metadata_path"] for x in row["candidates"]}), 2)
        self.assertFalse(evidence["module_import_ownership_verified"])
        self.assertFalse(json.loads(diagnostic)["protected"])

    def test_same_root_two_metadata_versions_record_lookup_choice(self):
        first = self.distribution("pip", "cryptography", "46.0.0")
        second = self.distribution("pip", "cryptography", "41.0.7")
        evidence = {}
        with patch.object(sys, "path", [str(self.root / "pip")]):
            expected = p.distribution_identity(p.metadata.distribution("cryptography"))
        result, _ = self.lookup([second, first], ["pip"], evidence=evidence)
        self.assertEqual(result["cryptography"], expected["version"])
        self.assertEqual(evidence["duplicates"][0]["selected"], expected)
        self.assertEqual(len(evidence["duplicates"][0]["candidates"]), 2)

    def test_repeated_enumeration_is_not_two_installations(self):
        dist = self.distribution("pip", "torch", "2.13.0")
        evidence = {}
        result, diagnostics = self.lookup([dist, dist], ["pip"], evidence=evidence)
        self.assertEqual(result, {"torch": "2.13.0"})
        self.assertEqual(evidence["duplicates"], [])
        self.assertEqual(len(evidence["critical_selected"]), 1)
        self.assertEqual(diagnostics, "")

    def test_duplicate_native_loader_sglang_and_flashinfer_are_rejected(self):
        for index, name in enumerate(("torch", "sglang", "flashinfer-python", "flashinfer-jit-cache-sm103a",
                                      "nvidia-cutlass-dsl", "huggingface-hub")):
            with self.subTest(name=name):
                first = self.distribution(f"pip{index}", name, "1.0")
                second = self.distribution(f"system{index}", name, "1.0")
                with self.assertRaisesRegex(ValueError, "Duplicate protected") as error:
                    self.lookup([first, second], [f"pip{index}", f"system{index}"])
                self.assertIn(str(self.root / f"pip{index}"), str(error.exception))
                self.assertIn(str(self.root / f"system{index}"), str(error.exception))

    def test_explicit_selected_constraint_is_strict(self):
        first = self.distribution("pip", "cryptography", "46.0.0")
        second = self.distribution("system", "cryptography", "41.0.7")
        with self.assertRaisesRegex(ValueError, "Duplicate protected"):
            self.lookup([first, second], ["pip", "system"], strict={"cryptography"})

    def test_selected_location_must_belong_to_observed_metadata(self):
        observed = self.distribution("pip", "cryptography", "46.0.0")
        unknown = self.distribution("elsewhere", "cryptography", "46.0.0")
        with patch.object(p.metadata, "distribution", return_value=unknown), \
             self.assertRaisesRegex(ValueError, "not enumerated"):
            self.lookup([observed], ["pip"])

    def test_missing_metadata_location_is_rejected(self):
        dist = SimpleNamespace(metadata={"Name": "cryptography"}, version="46.0.0")
        with self.assertRaisesRegex(ValueError, "metadata location"):
            p.distribution_identity(dist)

class ConstraintTests(unittest.TestCase):
    def test_source_owned_selected_and_inherited_native(self):
        selected, preserved, pins = p.make_constraints("upgrade", requirements("upgrade"), base())
        self.assertEqual(pins["nvidia-cutlass-dsl"], "4.8.0")
        self.assertEqual(selected["nccl-extensions"], "0.1.0")
        self.assertEqual(preserved["nixl-cu13"], "1.4.0")
        self.assertEqual(preserved["triton"], "3.7.1+git5d6048aa")
        self.assertNotIn("torch", selected)

    def test_missing_native_base_fails(self):
        value = base(); value.pop("triton")
        with self.assertRaisesRegex(ValueError, "missing: triton"):
            p.make_constraints("baseline", requirements(), value)

    def test_wrong_torch_rejected(self):
        value = base(); value["torch"] = "2.11.0"
        with self.assertRaisesRegex(ValueError, "Base torch violates"):
            p.make_constraints("baseline", requirements(), value)

    def test_local_torch_preserved(self):
        value = base(); value["torch"] = "2.13.0+cu132"
        self.assertEqual(p.make_constraints("baseline", requirements(), value)[1]["torch"], "2.13.0+cu132")

    def test_plain_constraints_without_extras(self):
        pins = p.make_constraints("upgrade", requirements("upgrade"), base())[2]
        self.assertTrue(all("[" not in name for name in pins))

    def test_wrong_source_pair_rejected(self):
        with patch.object(p, "revision", side_effect=[p.SGLANG_SOURCE, "f" * 40]):
            with self.assertRaisesRegex(ValueError, "Wrong FlashInfer source"):
                p.source_contract("baseline", Path("sg"), Path("fi"))

    def test_unreviewed_patch_rejected(self):
        with patch.object(p, "PATCH_SHA256", "0" * 64):
            with self.assertRaisesRegex(ValueError, "Unreviewed compatibility patch"):
                p.source_evidence("upgrade", Path("unused"))

    def test_exact_version_rejects_range_and_wildcard(self):
        self.assertIsNone(p.exact_version(Requirement("x>=1")))
        self.assertIsNone(p.exact_version(Requirement("x==1.*")))


class WheelSetTests(unittest.TestCase):
    def check(self, rows, profile="upgrade"):
        return p.validate_wheel_set(profile, rows, ["sm103a"], p.PROFILES[profile]["source"])

    def test_upgrade_complete_set(self):
        self.assertEqual(self.check(wheel_rows())["name"], "flashinfer-jit-cache-sm103a")

    def test_baseline_complete_set(self):
        self.check(wheel_rows("baseline"), "baseline")

    def test_shim_without_provider_fails(self):
        with self.assertRaisesRegex(ValueError, "Missing built"):
            self.check(wheel_rows()[:-1])

    def test_empty_critical_provider_fails(self):
        rows = wheel_rows(); rows[-1]["key_modules"] = {}
        with self.assertRaisesRegex(ValueError, "Missing GLM"):
            self.check(rows)

    def test_one_missing_critical_module_fails(self):
        rows = wheel_rows(); rows[-1]["key_modules"].pop("fmha_gen")
        with self.assertRaisesRegex(ValueError, "Missing GLM"):
            self.check(rows)

    def test_wrong_local_cuda_version_fails(self):
        rows = wheel_rows(); rows[-1]["version"] = "0.7.0.post1+cu130"
        with self.assertRaisesRegex(ValueError, "versions/local CUDA"):
            self.check(rows)

    def test_uniform_wrong_local_cuda_label_rejected(self):
        rows = wheel_rows("baseline")
        for row in rows:
            if row["name"].startswith("flashinfer-"):
                row["version"] = "0.6.18+cu130"
                row["build_metadata"]["__version__"] = row["version"]
        with self.assertRaisesRegex(ValueError, "differs from active nvcc"):
            p.validate_wheel_set("baseline", rows, ["sm103a"], p.PROFILES["baseline"]["source"], "cu132")

    def test_wrong_source_fails(self):
        rows = wheel_rows(); rows[2]["build_metadata"]["__git_version__"] = "e" * 40
        with self.assertRaisesRegex(ValueError, "source mismatch"):
            self.check(rows)

    def test_unexpected_provider_fails(self):
        rows = wheel_rows(); new = copy.deepcopy(rows[-1]); new["name"] = "flashinfer-jit-cache-sm120a"; rows.append(new)
        with self.assertRaisesRegex(ValueError, "Unexpected FlashInfer"):
            self.check(rows)

    def test_shim_version_range_rejected(self):
        rows = wheel_rows(); rows[4]["requires"] = ["flashinfer-jit-cache-sm103a>=0.7"]
        with self.assertRaisesRegex(ValueError, "Shim provider requirements"):
            self.check(rows)

    def test_duplicate_distribution_rejected(self):
        rows = wheel_rows(); rows.append(copy.deepcopy(rows[0]))
        with self.assertRaisesRegex(ValueError, "Duplicate wheel"):
            self.check(rows)

    def test_wrong_manifest_target_rejected(self):
        rows = wheel_rows(); rows[-1]["provider_manifest"]["cuda_architectures"] = ["sm100a"]
        with self.assertRaisesRegex(ValueError, "manifest identity"):
            self.check(rows)


class ArtifactTests(unittest.TestCase):
    def test_header_payload_is_bound_to_wheel_record(self):
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / "headers.whl"
            root = "f.data/purelib/flashinfer_cubin/include/trtllmGen_bmm_export/"
            rows = []
            with zipfile.ZipFile(wheel, "w") as archive:
                for suffix, data in (("gen/a.h", b"header"), ("gen/a.h.lock", b"")):
                    name = root + suffix
                    info = zipfile.ZipInfo(name); info.external_attr = 0o100644 << 16
                    archive.writestr(info, data)
                    checksum = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
                    rows.append(f"{name},sha256={checksum},{len(data)}")
                archive.writestr("f.dist-info/RECORD", "\n".join(rows))
            with zipfile.ZipFile(wheel) as archive:
                value = p.cubin_headers(archive)
            self.assertEqual(value["path"], "flashinfer_cubin/include/trtllmGen_bmm_export")
            self.assertEqual(value["directories"], ["gen"])
            self.assertEqual(len(value["files"]), 2)
            with zipfile.ZipFile(wheel, "a") as archive:
                archive.writestr("f.dist-info/RECORD", "bad")
            with zipfile.ZipFile(wheel) as archive, self.assertRaises(ValueError):
                p.cubin_headers(archive)

    def test_data_purelib_paths_match_normal_pip_location(self):
        self.assertEqual(p.installed_path("f-1.data/purelib/flashinfer_jit_cache/cache/x.so"),
                         "flashinfer_jit_cache/cache/x.so")

    def test_wheel_static_inspection_and_relocation(self):
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / "fixture.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("f.dist-info/METADATA", "Name: flashinfer-jit-cache\nVersion: 0.6.18+cu132\n")
                prefix = "f.data/purelib/flashinfer_jit_cache/"
                archive.writestr(prefix + "_build_meta.py", "__version__='0.6.18+cu132'\n__git_version__='abc'\n")
                for module in p.KEY_MODULES:
                    archive.writestr(prefix + f"cached_ops/{module}/{module}.so", b"\x7fELFtest")
            row = p.inspect_wheel(wheel)
            self.assertEqual(set(row["key_modules"]), set(p.KEY_MODULES))
            self.assertTrue(all(value["path"].startswith("flashinfer_jit_cache/")
                                for value in row["key_modules"].values()))

    def test_nonliteral_metadata_is_never_executed(self):
        with tempfile.TemporaryDirectory() as directory:
            wheel = Path(directory) / "fixture.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("x/_build_meta.py", "__version__=__import__('os').getcwd()")
            with zipfile.ZipFile(wheel) as archive, self.assertRaises(ValueError):
                p.literal_build_metadata(archive)

    def test_provider_report_bound_to_wheel_and_sm103(self):
        row = wheel_rows()[-1]
        report = {
            "schema_version": 1, "provider_id": "sm103a", "version": row["version"],
            "modules": sorted(p.KEY_MODULES), "wheels": {row["name"]: {
                "filename": row["filename"], "size_bytes": row["bytes"], "sha256": row["sha256"]}},
            "module_cuda_architecture_summary": {"incompatible_targets": 0},
            "module_cuda_architectures": {"fp4_quantization_103": ["sm103a"]},
        }
        p.validate_provider_report(row, report)
        report["module_cuda_architectures"]["fp4_quantization_103"] = ["sm100a"]
        with self.assertRaisesRegex(ValueError, "actual SM103"):
            p.validate_provider_report(row, report)
        report["module_cuda_architectures"]["fp4_quantization_103"] = ["sm103a"]
        report["wheels"][row["name"]]["sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "wheel binding"):
            p.validate_provider_report(row, report)

    def test_architectures_require_sm103_and_exact_unique_targets(self):
        self.assertEqual(p.architectures("9.0a 10.0a 10.3a"), ["sm100a", "sm103a", "sm90a"])
        for value in ("9.0a", "10.3a 10.3a", "10.3+PTX"):
            with self.assertRaises(ValueError):
                p.architectures(value)


class RuntimeLoaderInstallTests(unittest.TestCase):
    def requests(self, profile, wheels):
        source = Path(__file__).with_name("install.bash").read_text()
        function = source.split("_INSTALL_WHEELS() {", 1)[1].split("\n_INSTALL_WHEELS /wheels/*.whl", 1)[0]
        script = "_INSTALL_WHEELS() {" + function + '\n_PIP_INSTALL() { printf "%s\\0" "$@"; }\n_INSTALL_WHEELS "$@"\n'
        return subprocess.run(["bash", "-ec", script, "test-install", *map(str, wheels)],
                              env=os.environ | {"SGLANG_PACKAGE_PROFILE": profile}, capture_output=True, timeout=30)

    def test_profile_requests_only_sglang_extra_and_legacy_unchanged(self):
        wheels = ["/wheels/flashinfer_python-0.6.18+cu132-py3-none-any.whl",
                  "/wheels/sglang-0.5.20-py3-none-any.whl", "/wheels/sglang_kernel-0.4.7-py3-none-any.whl"]
        for profile in ("legacy", "baseline", "upgrade"):
            with self.subTest(profile=profile):
                result = self.requests(profile, wheels)
                self.assertEqual(result.returncode, 0, result.stderr)
                expected = wheels.copy()
                if profile != "legacy":
                    expected[1] += "[runai]"
                self.assertEqual(result.stdout.decode().rstrip("\0").split("\0"), expected)

    def test_missing_or_ambiguous_sglang_rejected_before_pip(self):
        for wheels in (["/wheels/flashinfer_python-0.6.18-py3-none-any.whl"],
                       ["/wheels/sglang-0.5.20-py3-none-any.whl", "/wheels/sglang-0.5.17-py3-none-any.whl"]):
            with self.subTest(wheels=wheels):
                result = self.requests("baseline", wheels)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, b"")
                self.assertIn(b"Expected exactly one SGLang wheel", result.stderr)

    @staticmethod
    def wheel(root, name, version, metadata=""):
        stem = name.replace("-", "_") + "-" + version
        path = root / (stem + "-py3-none-any.whl")
        prefix = stem + ".dist-info/"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(prefix + "METADATA", f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n" + metadata)
            archive.writestr(prefix + "WHEEL", "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
            archive.writestr(prefix + "RECORD", "")
        return path

    def test_offline_pip_resolves_source_owned_loader_backends_under_constraints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sglang = self.wheel(root, "sglang", "0.5.20", 'Provides-Extra: runai\nRequires-Dist: runai-model-streamer[s3,gcs,azure]>=0.15.7; extra == "runai"\n')
            loader_metadata = "".join(f'Provides-Extra: {extra}\nRequires-Dist: runai-model-streamer-{extra}>=0.15.7; extra == "{extra}"\n'
                                      for extra in ("s3", "gcs", "azure"))
            expected = {"sglang": "0.5.20", "runai-model-streamer": "0.15.7"}
            self.wheel(root, "runai-model-streamer", "0.15.7", loader_metadata)
            self.wheel(root, "runai-model-streamer", "0.16.1", loader_metadata)
            for extra in ("s3", "gcs", "azure"):
                name = "runai-model-streamer-" + extra
                expected[name] = "0.15.7"
                self.wheel(root, name, "0.15.7")
                self.wheel(root, name, "0.16.1")
            constraints = root / "constraints.txt"
            constraints.write_text("".join(f"{name}=={version}\n" for name, version in sorted(expected.items())))
            for profile in ("legacy", "baseline", "upgrade"):
                with self.subTest(profile=profile):
                    requests = self.requests(profile, [sglang])
                    self.assertEqual(requests.returncode, 0, requests.stderr)
                    report = root / (profile + ".json")
                    result = subprocess.run([sys.executable, "-m", "pip", "--isolated", "--disable-pip-version-check",
                                             "install", "--dry-run", "--ignore-installed", "--no-index", "--no-cache-dir",
                                             "--find-links", str(root), "--constraint", str(constraints), "--report", str(report),
                                             *requests.stdout.decode().rstrip("\0").split("\0")],
                                            capture_output=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stderr.decode())
                    resolved = {p.canonicalize_name(item["metadata"]["name"]): item["metadata"]["version"]
                                for item in json.loads(report.read_text())["install"]}
                    self.assertEqual(resolved, {"sglang": "0.5.20"} if profile == "legacy" else expected)
            self.wheel(root, "runai-model-streamer", "0.15.6", loader_metadata)
            constraints.write_text("runai-model-streamer==0.15.6\n")
            requests = self.requests("baseline", [sglang])
            self.assertEqual(requests.returncode, 0, requests.stderr)
            incompatible = subprocess.run([sys.executable, "-m", "pip", "--isolated", "--disable-pip-version-check",
                                           "install", "--dry-run", "--ignore-installed", "--no-index", "--no-cache-dir",
                                           "--find-links", str(root), "--constraint", str(constraints),
                                           "--report", str(root / "incompatible.json"),
                                           *requests.stdout.decode().rstrip("\0").split("\0")],
                                          capture_output=True, timeout=30)
            self.assertNotEqual(incompatible.returncode, 0)
            self.assertIn(b"ResolutionImpossible", incompatible.stderr)
            self.assertFalse((root / "incompatible.json").exists())


class InstalledTests(unittest.TestCase):
    def test_compact_record_keeps_validated_provider_and_payload_identity(self):
        row = wheel_rows()[-1]
        report = {"schema_version": 1, "provider_id": "sm103a", "version": row["version"],
                  "modules": sorted(p.KEY_MODULES), "wheels": {row["name"]: {
                      "filename": row["filename"], "size_bytes": row["bytes"], "sha256": row["sha256"]}},
                  "module_cuda_architecture_summary": {"incompatible_targets": 0},
                  "module_cuda_architectures": {"fp4_quantization_103": ["sm103a"]}}
        info = {"wheels": [row], "provider_validation": {"sm103a": {"sha256": "d" * 64, "report": report}}}
        compact = p.compact_build_record(info, {"torch": "2.13.0"}, [{"module": "fp4_quantization_103"}], "e" * 64)
        self.assertNotIn("modules", compact["wheels"][0]["provider_manifest"])
        self.assertEqual(compact["wheels"][0]["provider_manifest"]["module_count"], 4)
        self.assertEqual(compact["wheels"][0]["key_modules"], row["key_modules"])
        self.assertNotIn("report", compact["provider_validation"]["sm103a"])
        self.assertEqual(compact["provider_validation"]["sm103a"]["summary"]["critical_module_cuda_architectures"],
                         {"fp4_quantization_103": ["sm103a"]})
        self.assertIn("report", info["provider_validation"]["sm103a"])
        self.assertEqual(compact["wheel_info_sha256"], "e" * 64)
        report["module_cuda_architecture_summary"]["incompatible_targets"] = 1
        with self.assertRaisesRegex(ValueError, "incompatible CUDA"):
            p.compact_build_record(info, {}, [], "e" * 64)

    def test_compact_record_explicit_size_bound(self):
        with patch.object(p, "MAX_BUILD_INFO_BYTES", 64):
            with self.assertRaisesRegex(ValueError, "size bound"):
                p.compact_build_record({"wheels": [], "provider_validation": {}}, {}, [], "e" * 64)

    def test_compact_selection_provenance_remains_bounded(self):
        selected = [{"name": f"nvidia-fixture-{n}", "version": "13.2.1",
                     "metadata_path": f"/usr/local/lib/python3.12/dist-packages/nvidia_fixture_{n}-13.2.1.dist-info",
                     "location": "/usr/local/lib/python3.12/dist-packages"} for n in range(100)]
        evidence = {"schema": "distribution-selection/v1", "resolution": "importlib.metadata.distribution",
                    "module_import_ownership_verified": False, "critical_selected": selected, "duplicates": []}
        info = {"wheels": [], "provider_validation": {}, "distribution_selection_build": evidence}
        record = p.compact_build_record(info, {f"fixture-{n}": "1.0.0" for n in range(500)}, [], "a" * 64, evidence)
        self.assertNotIn("distribution_selection_build", record)
        self.assertEqual(record["distribution_selection_installed"], evidence)
        self.assertEqual(record["distribution_selection_build_sha256"],
                         hashlib.sha256(json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
        self.assertLess(len(json.dumps(record, indent=2, sort_keys=True).encode()), p.MAX_BUILD_INFO_BYTES)

    def test_loader_families_preserved_and_new_resolution_frozen(self):
        before = base() | {"safetensors": "0.7.0", "runai-model-streamer": "0.15.7"}
        _, preserved, _ = p.make_constraints("baseline", requirements(), before)
        self.assertEqual(preserved["safetensors"], "0.7.0")
        self.assertEqual(preserved["runai-model-streamer"], "0.15.7")
        snapshot = {"selected_versions": {}, "preserved_versions": preserved, "optional_versions": {}}
        new = p.newly_resolved_native(snapshot, before | {"huggingface-hub": "1.10.0", "hf-xet": "1.5.0"})
        self.assertEqual(new, {"huggingface-hub": "1.10.0", "hf-xet": "1.5.0"})
        info = {"selected_versions": {}, "preserved_versions": preserved, "resolved_native_versions": new, "wheels": []}
        with self.assertRaises(ValueError):
            p.validate_installed(info, before | {"huggingface-hub": "1.10.1", "hf-xet": "1.5.0"})

    def test_newly_resolved_native_is_frozen_without_unrelated_python(self):
        snapshot = {"selected_versions": {"nvidia-cutlass-dsl": "4.8.0"},
                    "preserved_versions": {"torch": "2.13.0"}, "optional_versions": {}}
        resolved = p.newly_resolved_native(snapshot, {"torch": "2.13.0", "nvidia-cutlass-dsl": "4.8.0",
                                                      "nccl4py": "0.5.0", "requests": "2.0"})
        self.assertEqual(resolved, {"nccl4py": "0.5.0"})
        info = {"preserved_versions": {}, "selected_versions": {}, "wheels": [], "resolved_native_versions": resolved}
        with self.assertRaisesRegex(ValueError, "Pinned native/loader dependency mismatch"):
            p.validate_installed(info, {"nccl4py": "0.6.0"})

    def test_optional_constraints_do_not_force_unused_libraries(self):
        info = {"preserved_versions": {}, "selected_versions": {}, "wheels": [],
                "optional_versions": {"nvidia-cutlass-dsl-libs-cu12": "4.8.0"}}
        p.validate_installed(info, {})
        p.validate_installed(info, {"nvidia-cutlass-dsl-libs-cu12": "4.8.0"})
        with self.assertRaisesRegex(ValueError, "optional dependency mismatch"):
            p.validate_installed(info, {"nvidia-cutlass-dsl-libs-cu12": "4.6.2"})

    def test_preserved_native_drift_rejected(self):
        info = {"preserved_versions": base(), "selected_versions": {}, "wheels": []}
        versions = base(); versions["nixl-cu13"] = "1.3.1"
        with self.assertRaisesRegex(ValueError, "Pinned native/loader dependency mismatch: nixl-cu13"):
            p.validate_installed(info, versions)

    def test_missing_or_changed_loader_keeps_exact_audit_with_diagnostics(self):
        for origin in ("preserved_versions", "resolved_native_versions"):
            info = {"preserved_versions": {}, "resolved_native_versions": {}, "selected_versions": {}, "wheels": []}
            info[origin] = {"runai-model-streamer": "0.15.7"}
            p.validate_installed(info, {"runai-model-streamer": "0.15.7"})
            for observed in ({}, {"runai-model-streamer": "0.16.1"}):
                with self.subTest(origin=origin, observed=observed), self.assertRaises(ValueError) as raised:
                    p.validate_installed(info, observed)
                self.assertIn("expected=0.15.7", str(raised.exception))
                self.assertIn("actual=" + observed.get("runai-model-streamer", "<missing>"), str(raised.exception))
                self.assertIn("origin=" + ("inherited" if origin == "preserved_versions" else "builder-resolved"), str(raised.exception))

    def test_local_cuda_source_version_accepted(self):
        info = {"preserved_versions": {}, "selected_versions": {"flashinfer-python": "0.7.0.post1"}, "wheels": []}
        p.validate_installed(info, {"flashinfer-python": "0.7.0.post1+cu132"})

    def test_pip_cannot_replace_built_local_wheel(self):
        info = {"preserved_versions": {}, "selected_versions": {},
                "wheels": [{"name": "flashinfer-python", "version": "0.7.0.post1+cu132"}]}
        with self.assertRaisesRegex(ValueError, "Built wheel was replaced"):
            p.validate_installed(info, {"flashinfer-python": "0.7.0.post1"})

    def test_installed_payload_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); payload = root / "module.so"; payload.write_bytes(b"\x7fELFbytes")
            info = {"preserved_versions": {}, "selected_versions": {}, "torch_abi": {"cuda": "13.2"},
                    "wheels": [{"name": "flashinfer-jit-cache", "version": "0.6.18+cu132",
                                "key_modules": {"test": {"path": "module.so", "bytes": payload.stat().st_size,
                                                          "sha256": p.sha256(payload)}}}]}
            dist = SimpleNamespace(locate_file=lambda path: root / path)
            with patch.object(p, "installed", return_value={"flashinfer-jit-cache": "0.6.18+cu132"}), \
                 patch.object(p, "torch_abi", return_value=info["torch_abi"]), \
                 patch.object(p.metadata, "distribution", return_value=dist):
                p.audit_installed(info)
                payload.write_bytes(b"\x7fELFother")
                with self.assertRaisesRegex(ValueError, "Installed AOT module changed"):
                    p.audit_installed(info)

    def test_torch_abi_drift_rejected(self):
        info = {"preserved_versions": {}, "selected_versions": {}, "wheels": [], "torch_abi": {"cxx11_abi": True}}
        with patch.object(p, "installed", return_value={}), patch.object(p, "torch_abi", return_value={"cxx11_abi": False}):
            with self.assertRaisesRegex(ValueError, "Torch ABI differs"):
                p.audit_installed(info)


if __name__ == "__main__":
    unittest.main()
