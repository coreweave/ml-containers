import importlib.util
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile


ROOT = Path(__file__).resolve().parents[1]
COMMIT = "96d91ef9266d2bebd8e8c09ef1f28b2d521631ff"
CANDIDATE_BASE_IMAGE = (
    "ghcr.io/coreweave/ml-containers/torch-extras:8c5de3e-nccl-cuda13.2.1-"
    "ubuntu24.04-nccl2.30.4-1-torch2.13.0-vision0.28.0-audio2.11.0-abi1"
)
CANDIDATE_TAG = "glm53-flash-candidate-96d91ef9266d-cuda13.2.1-torch2.13.0"


def preflight():
    spec = importlib.util.spec_from_file_location("preflight", ROOT / "sglang/preflight.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_wheel(directory, name, version):
    filename = f"{name.replace('-', '_')}-{version}-py3-none-any.whl"
    path = directory / filename
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr(
            f"{name.replace('-', '_')}-{version}.dist-info/METADATA",
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\nRequires-Dist: torch==2.13.0\n",
        )
    return path


class CandidateWiringTests(unittest.TestCase):
    def test_unique_candidate_has_pinned_inputs_and_opt_in(self):
        matrix = (ROOT / ".github/configurations/sglang.yml").read_text()
        rows = re.split(r"(?m)^  - ", matrix)[1:]
        candidates = [row for row in rows if "tag: 'glm53-flash-candidate-" in row]
        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertIn(f"sglang-commit: '{COMMIT}'\n", candidate)
        self.assertIn(f"    base-image: '{CANDIDATE_BASE_IMAGE}'\n", candidate)
        self.assertIn(f"    tag: '{CANDIDATE_TAG}'\n", candidate)
        self.assertIn("    glm53-preflight: '1'\n", candidate)

    def test_opt_in_is_passed_to_both_stages(self):
        workflow = (ROOT / ".github/workflows/sglang.yml").read_text()
        dockerfile = (ROOT / "sglang/Dockerfile").read_text()
        self.assertIn("SGLANG_GLM53_PREFLIGHT=${{ matrix.glm53-preflight || '0' }}", workflow)
        self.assertEqual(dockerfile.count("ARG SGLANG_GLM53_PREFLIGHT=0"), 2)
        self.assertEqual(dockerfile.count("ARG SGLANG_COMMIT\n"), 2)
        self.assertIn("COPY build.bash preflight.py /build/", dockerfile)
        self.assertIn("COPY install.bash preflight.py /wheels/", dockerfile)
        for script in ("build.bash", "install.bash"):
            source = (ROOT / "sglang" / script).read_text()
            self.assertIn('if [ "${SGLANG_GLM53_PREFLIGHT:-0}" = 1 ]; then', source)
            self.assertIn("preflight.py", source)
        build = (ROOT / "sglang/build.bash").read_text()
        self.assertIn('if [ "${SGLANG_GLM53_PREFLIGHT:-0}" != 1 ]; then\n  TORCH_VERSION=', build)
        self.assertIn("git submodule update --init --recursive", build)

    def test_preflight_suite_is_a_build_prerequisite(self):
        workflow = (ROOT / ".github/workflows/sglang.yml").read_text()
        self.assertIn("  preflight-tests:\n", workflow)
        self.assertIn("    needs: [get-config, preflight-tests]\n", workflow)
        self.assertIn(
            "run: python3 -B -m unittest discover -s sglang -p 'test_*.py' -v",
            workflow,
        )

    def test_native_bash_syntax(self):
        major = int(subprocess.check_output(
            ["bash", "-c", 'printf "%s" "${BASH_VERSINFO[0]}"'], text=True
        ))
        if major < 4:
            self.skipTest(f"Native syntax check requires Bash >=4; found Bash {major}")
        for script in ("build.bash", "install.bash"):
            with self.subTest(script=script):
                result = subprocess.run(
                    ["bash", "-n", str(ROOT / "sglang" / script)],
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)


class PreflightTests(unittest.TestCase):
    def test_missing_flash_source_is_rejected(self):
        module = preflight()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                module.source_features(Path(directory))

    def test_wrong_commit_and_scm_override_are_rejected(self):
        module = preflight()
        with self.assertRaises(ValueError):
            module.validate_commit("0bcd822377da7b5718e674eaf9c870d349424dd1")
        with patch.dict(os.environ, {"SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG": "0.5.19"}):
            with self.assertRaises(ValueError):
                module.validate_commit(COMMIT)

    def test_source_identity_rejects_wrong_checkout_dirty_tree_and_submodules(self):
        module = preflight()
        with patch.object(module.subprocess, "check_output", return_value="wrong\n"):
            with self.assertRaises(ValueError):
                module.source_evidence(ROOT, COMMIT)
        with (
            patch.object(module.subprocess, "check_output", return_value=COMMIT),
            patch.object(module.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "git diff")),
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                module.source_evidence(ROOT, COMMIT)
        with (
            patch.object(module.subprocess, "check_output", side_effect=[COMMIT, "+wrong submodule\n"]),
            patch.object(module.subprocess, "run"),
        ):
            with self.assertRaises(ValueError):
                module.source_evidence(ROOT, COMMIT)

    def test_existing_source_without_required_class_is_rejected(self):
        module = preflight()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / next(iter(module.SOURCE_FEATURES))
            first.parent.mkdir(parents=True)
            first.write_text("class UnrelatedConfig: pass\n")
            with self.assertRaises(ValueError):
                module.source_features(root)

    def test_observed_wheel_metadata_and_hashes(self):
        module = preflight()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = make_wheel(root, "sglang", "0.5.20.dev1+gtest")
            make_wheel(root, "sglang-kernel", "0.4.6.post1")
            evidence = module.wheel_evidence(root)
            self.assertEqual(evidence["sglang"]["version"], "0.5.20.dev1+gtest")
            self.assertEqual(evidence["sglang"]["filename"], wheel.name)
            self.assertEqual(evidence["sglang"]["sha256"], module.sha256(wheel))
            self.assertEqual(evidence["sglang"]["requires_dist"], ["torch==2.13.0"])
            make_wheel(root, "sglang", "0.5.19")
            with self.assertRaises(ValueError):
                module.wheel_evidence(root)

    def test_missing_kernel_and_scm_fallback_are_rejected(self):
        module = preflight()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_wheel(root, "sglang", "0.5.20.dev1+gtest")
            with self.assertRaises(ValueError):
                module.wheel_evidence(root)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_wheel(root, "sglang", "0.0.0.dev0")
            make_wheel(root, "sglang-kernel", "0.4.6.post1")
            with self.assertRaises(ValueError):
                module.wheel_evidence(root)

    def test_dependencies_must_match_pins_and_built_wheels(self):
        module = preflight()
        wheels = {"sglang": {"version": "0.5.20.dev1+gtest"}, "sglang-kernel": {"version": "0.4.6.post1"}}
        versions = dict(module.REQUIRED_VERSIONS, sglang="0.5.20.dev1+gtest")
        versions["torch"] += "+cu132"
        module.validate_dependencies(wheels, versions)
        versions["flashinfer-python"] = "0.6.17"
        with self.assertRaises(ValueError):
            module.validate_dependencies(wheels, versions)
        versions["flashinfer-python"] = module.REQUIRED_VERSIONS["flashinfer-python"]
        versions["sglang"] = "0.5.19"
        with self.assertRaises(ValueError):
            module.validate_dependencies(wheels, versions)

    def test_runtime_import_failure_is_not_skipped(self):
        module = preflight()
        with patch.object(module.importlib, "import_module", side_effect=ModuleNotFoundError("missing Flash support")):
            with self.assertRaises(ModuleNotFoundError):
                module.runtime_features()


if __name__ == "__main__":
    unittest.main()
