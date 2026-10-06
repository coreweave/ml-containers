"""Exercise build phase control flow without running a compiler or package installer."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parent


def stages():
    text = (SOURCE / 'Dockerfile').read_text()
    rows = list(re.finditer(r'^FROM (.+)$', text, re.M))
    return {(row[1].split(' AS ')[-1] if ' AS ' in row[1] else '<final>'):
            text[row.start():rows[i + 1].start() if i + 1 < len(rows) else len(text)]
            for i, row in enumerate(rows)}


class StageTests(unittest.TestCase):
    def test_flashinfer_inputs_exclude_validation_code_and_sglang_source(self):
        stage = stages()['flashinfer-builder']
        self.assertTrue(stage.startswith('FROM builder-base AS flashinfer-builder'))
        self.assertNotIn('SGLANG_COMMIT', stage)
        for name in ('packages.py', 'test_', 'packages-before.json', '--from=sglang-source'):
            self.assertNotIn(name, stage)
        self.assertIn('COPY --link --from=package-contract /contract/constraints.txt /contract/build-requirements.txt /wheels/', stage)
        self.assertIn('COPY --link --from=flashinfer-source /build/flashinfer/ /build/flashinfer/', stage)
        for name in ('builder-base', 'flashinfer-source'):
            self.assertNotIn('SGLANG_COMMIT', stages()[name])
            self.assertNotIn('packages.py', stages()[name])

    def test_sglang_inherits_compiler_dependency_environment_and_current_audit(self):
        stage = stages()['builder']
        self.assertTrue(stage.startswith('FROM flashinfer-builder AS builder'))
        self.assertIn('COPY build.bash packages.py /build/', stage)
        self.assertIn('/contract/packages-before.json /wheels/', stage)
        self.assertIn('packages.py wheels', (SOURCE / 'build.bash').read_text())
        self.assertIn('packages.py audit', (SOURCE / 'install.bash').read_text())

    def test_all_profiles_gate_logger_and_stage_tests(self):
        stage = stages()['package-contract']
        self.assertLess(stage.index('unittest -v test_build_progress test_build_stages'),
                        stage.index('if [ "${SGLANG_PACKAGE_PROFILE}" = legacy ]'))
        self.assertIn('set -e', stage)
        self.assertIn('unittest -v test_packages', stage)

    def test_protocol_gate_precedes_compile_and_follows_final_pip_check(self):
        stage = stages()['package-contract']
        self.assertLess(stage.index('packages.py constraints'), stage.index('packages.py protocol-requirements'))
        self.assertLess(stage.index('--label protocol-install'), stage.index('--label protocol-check'))
        self.assertIn('--constraint /contract/constraints.txt', stage)
        self.assertIn('--requirement /contract/protocol-requirements.txt', stage)
        self.assertIn('timeout 120s python3 /build/packages.py verify-protocol', stage)
        source = (SOURCE / 'install.bash').read_text()
        self.assertLess(source.index('python3 -m pip check'), source.index('packages.py verify-protocol'))
        self.assertLess(source.index('packages.py verify-protocol'), source.index('packages.py audit'))

    def test_contract_protocol_failure_is_terminal_and_legacy_bypasses_it(self):
        contract = stages()['package-contract'].split("RUN <<'CONTRACT'\n", 1)[1].rsplit('\nCONTRACT', 1)[0]
        for profile in ('baseline', 'upgrade', 'legacy'):
            for failed in ('', 'protocol-install', 'protocol-check'):
                with self.subTest(profile=profile, failed=failed), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    bindir = root / 'bin'; bindir.mkdir()
                    python = bindir / 'python3'
                    python.write_text('#!' + sys.executable + '\n' +
                        'import json,os,sys\n'
                        'args=sys.argv[1:]\n'
                        'with open(os.environ["TRACE"],"a") as f:f.write(json.dumps(args)+"\\n")\n'
                        'if "--label" in args and args[args.index("--label")+1]==os.environ["FAIL_PHASE"]:sys.exit(7)\n')
                    python.chmod(0o755)
                    result = subprocess.run(['bash', '-c', contract.replace('/contract', str(root / 'contract'))],
                        env=dict(os.environ, PATH=str(bindir) + os.pathsep + os.environ['PATH'],
                                 SGLANG_PACKAGE_PROFILE=profile, TRACE=str(root / 'trace'), FAIL_PHASE=failed),
                        cwd=root, capture_output=True, timeout=10)
                    trace = [json.loads(line) for line in (root / 'trace').read_text().splitlines()]
                    labels = [row[row.index('--label')+1] for row in trace if '--label' in row]
                    if profile == 'legacy':
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertEqual(labels, [])
                    else:
                        self.assertEqual(result.returncode, 7 if failed else 0, result.stderr)
                        self.assertEqual(labels, ['package-constraints', 'protocol-install'] +
                                         ([] if failed == 'protocol-install' else ['protocol-check']))

    def test_final_image_excludes_full_logs_and_remains_default_target(self):
        rows = stages()
        self.assertEqual(list(rows)[-1], '<final>')
        self.assertIn('FROM scratch AS build-logs', rows['build-logs'])
        self.assertNotIn('/build-logs', rows['<final>'])
        self.assertIn('from=builder,source=/wheels,target=/wheels', rows['<final>'])


class PhaseTests(unittest.TestCase):
    def run_phase(self, script, profile, arch='aarch64', failed='', fi_pin='a' * 40):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('build/flashinfer/flashinfer-jit-cache-provider',
                         'build/sglang/python/sglang/kernels/aot', 'build/sglang/rust',
                         'wheels', 'bin', 'cuda/bin', 'opt'):
                (root / name).mkdir(parents=True, exist_ok=True)
            (root / 'build/flashinfer/version.txt').write_text('0.7.0.post1\n')
            (root / 'wheels/flashinfer-architectures.txt').write_text('9.0a 10.0a 10.3a\n')
            for name in ('constraints.txt', 'build-requirements.txt'):
                (root / 'wheels' / name).write_text('fixture==1\n')
            fake = root / 'bin/python3'
            fake.write_text('#!' + sys.executable + '\n' + '''import json,os,sys
from pathlib import Path
args=sys.argv[1:]
label=args[args.index('--label')+1]
command=args[args.index('--')+1:]
row={'label':label,'argv':command,'environment':{k:os.getenv(k) for k in (
 'FLASHINFER_CUDA_ARCH_LIST','FLASHINFER_JIT_CACHE_PROVIDER_ARCH',
 'FLASHINFER_JIT_CACHE_PROVIDER_ARCHS','FLASHINFER_LOCAL_VERSION',
 'FLASHINFER_BUILD_NO_PIP','BUILD_NIXL_EP','MAX_JOBS','FLASHINFER_NVCC_THREADS',
 'TORCH_CUDA_ARCH_LIST','CMAKE_BUILD_PARALLEL_LEVEL','DEBIAN_FRONTEND','PATH')}}
with open(os.environ['TRACE'],'a') as f:f.write(json.dumps(row)+'\\n')
if label==os.getenv('FAIL_PHASE'):sys.exit(7)
if label.startswith('flashinfer-provider-'):
 out=Path(command[command.index('-o')+1]);(out/'fixture.whl').write_bytes(b'fixture')
if label.startswith('flashinfer-verify-'):
 out=Path(command[command.index('--artifact-dir')+1]);(out/'provider-validation.json').write_text('{}')
''')
            fake.chmod(0o755)
            for path, text in ((root / 'bin/uname', arch), (root / 'cuda/bin/nvcc', 'Cuda compilation tools, release 13.2, V13.2.78')):
                path.write_text('#!/bin/sh\nprintf "%s\\n" ' + repr(text) + '\n')
                path.chmod(0o755)
            apt = root / 'bin/apt-get';apt.write_text('#!/bin/sh\nexit 0\n');apt.chmod(0o755)
            source = (SOURCE / script).read_text()
            for before, after in (('/build/', str(root / 'build') + '/'), ('/wheels', str(root / 'wheels')),
                                  ('/usr/local/cuda', str(root / 'cuda')), ('/opt/', str(root / 'opt') + '/')):
                source = source.replace(before, after)
            target = root / 'phase.bash';target.write_text(source)
            env = dict(os.environ, PATH=str(root / 'bin') + os.pathsep + os.environ['PATH'],
                       SGLANG_PACKAGE_PROFILE=profile, FLASHINFER_COMMIT=fi_pin,
                       TRACE=str(root / 'trace.jsonl'), FAIL_PHASE=failed)
            for name in ('MAX_JOBS', 'FLASHINFER_NVCC_THREADS'):
                env.pop(name, None)
            result = subprocess.run(['bash', str(target), '-a', '9.0 10.0+PTX'], env=env,
                                    cwd=root, capture_output=True, text=True, timeout=10)
            trace = [json.loads(line) for line in (root / 'trace.jsonl').read_text().splitlines()] if (root / 'trace.jsonl').exists() else []
            architecture = (root / 'wheels/flashinfer-architectures.txt').read_text().strip()
            return result, trace, architecture

    def test_baseline_and_upgrade_keep_complete_architectures_and_flags(self):
        for arch, targets in [('aarch64', '9.0a 10.0a 10.3a'), ('x86_64', '8.0 8.6 8.9 9.0a 10.0a 10.3a')]:
            for profile in ('baseline', 'upgrade'):
                with self.subTest(arch=arch, profile=profile):
                    result, rows, actual = self.run_phase('build-flashinfer.bash', profile, arch)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(actual, targets)
                    self.assertEqual(rows[0]['label'], 'flashinfer-dependencies')
                    self.assertEqual(rows[0]['environment']['TORCH_CUDA_ARCH_LIST'], '9.0 10.0+PTX')
                    self.assertEqual(rows[0]['environment']['DEBIAN_FRONTEND'], 'noninteractive')
                    self.assertTrue(rows[0]['environment']['PATH'].startswith('/root/.cargo/bin:'))
                    for row in rows[1:]:
                        e = row['environment']
                        self.assertEqual((e['FLASHINFER_BUILD_NO_PIP'], e['BUILD_NIXL_EP']), ('1', '0'))
                        self.assertEqual((e['MAX_JOBS'], e['FLASHINFER_NVCC_THREADS'], e['FLASHINFER_LOCAL_VERSION']), ('8', '4', 'cu132'))
                    providers = [r for r in rows if r['label'].startswith('flashinfer-provider-')]
                    self.assertEqual([r['environment']['FLASHINFER_JIT_CACHE_PROVIDER_ARCH'] for r in providers],
                                     targets.split() if profile == 'upgrade' else [])
                    self.assertEqual(rows[-1]['label'], 'flashinfer-jit-cache')

    def test_failed_flashinfer_phase_stops_later_phases(self):
        for phase in ('flashinfer-dependencies', 'flashinfer-python', 'flashinfer-cubin', 'flashinfer-provider-sm90a', 'flashinfer-verify-sm90a'):
            with self.subTest(phase=phase):
                result, rows, _ = self.run_phase('build-flashinfer.bash', 'upgrade', failed=phase)
                self.assertEqual(result.returncode, 7, result.stderr)
                self.assertEqual(rows[-1]['label'], phase)

    def test_legacy_skips_flashinfer_and_rejects_unexpected_pin(self):
        result, rows, _ = self.run_phase('build-flashinfer.bash', 'legacy', fi_pin='')
        self.assertEqual(result.returncode, 0, result.stderr);self.assertEqual(rows, [])
        result, rows, _ = self.run_phase('build-flashinfer.bash', 'legacy')
        self.assertNotEqual(result.returncode, 0);self.assertEqual(rows, [])

    def test_sglang_success_keeps_audit_and_legacy_bypass(self):
        for profile in ('baseline', 'upgrade', 'legacy'):
            with self.subTest(profile=profile):
                result, rows, _ = self.run_phase('build.bash', profile)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual([r['label'] for r in rows], ['sglang-kernel', 'sglang-python'] +
                                 ([] if profile == 'legacy' else ['package-wheel-audit']))
                self.assertEqual(rows[0]['environment']['CMAKE_BUILD_PARALLEL_LEVEL'], '20')

    def test_sglang_kernel_or_python_failure_blocks_following_audit(self):
        for phase in ('sglang-kernel', 'sglang-python'):
            result, rows, _ = self.run_phase('build.bash', 'baseline', failed=phase)
            self.assertEqual(result.returncode, 7, result.stderr)
            self.assertEqual(rows[-1]['label'], phase)


if __name__ == '__main__':
    unittest.main()
