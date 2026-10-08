import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest


SCRIPT = Path(__file__).with_name("build_progress.py")


class BuildProgressTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.log = self.root / "phase.log"

    def argv(self, command):
        return [sys.executable, str(SCRIPT), "--label", "FI-provider-sm103a", "--log-path", str(self.log),
                "--heartbeat-seconds", "0.1", "--termination-grace-seconds", "0.2", "--", *command]

    def run_child(self, source, *args, env=None):
        result = subprocess.run(self.argv([sys.executable, "-c", source, *args]), capture_output=True, env=env, timeout=10)
        return result, [json.loads(line) for line in result.stdout.splitlines()]

    def test_success_keeps_full_both_streams_without_console_flood(self):
        result, events = self.run_child("import os; os.write(1,b'O'*200000); os.write(2,b'E'*200000)")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(self.log.read_bytes(), b"O" * 200000 + b"E" * 200000)
        self.assertEqual(events[0]["event"], "build_start")
        self.assertEqual(events[-1]["event"], "build_end")
        self.assertEqual(events[-1]["exit_code"], 0)
        self.assertEqual(events[-1]["child_exit_code"], 0)
        self.assertEqual(events[-1]["log_bytes"], 400000)
        self.assertLess(len(result.stdout), 2048)

    def test_failure_exit_and_huge_single_line_are_bounded(self):
        result, events = self.run_child("import os; os.write(1,b'x'*1000000+b'ENDMARK'); raise SystemExit(37)")
        self.assertEqual(result.returncode, 37)
        self.assertEqual(events[-2]["exit_code"], 37)
        self.assertEqual(self.log.stat().st_size, 1000007)
        self.assertLessEqual(len(result.stdout.splitlines(keepends=True)[-1]), 16384)
        self.assertTrue(events[-1]["tail"].endswith("ENDMARK"))

    def test_multiline_tail_has_at_most_40_lines(self):
        result, events = self.run_child("for n in range(1000): print('line-'+str(n))\nraise SystemExit(4)")
        self.assertEqual(result.returncode, 4)
        self.assertEqual(len(self.log.read_text().splitlines()), 1000)
        self.assertEqual(events[-1]["tail"].splitlines(), ["line-" + str(n) for n in range(960, 1000)])

    def test_json_escaped_unicode_tail_obeys_actual_output_byte_bound(self):
        result, events = self.run_child("import os; os.write(1, ('😀'*5000+'\\x00'*5000+'END').encode()); raise SystemExit(9)")
        self.assertEqual(result.returncode, 9)
        self.assertLessEqual(len(result.stdout.splitlines(keepends=True)[-1]), 16384)
        self.assertTrue(events[-1]["tail"].endswith("END"))

    def test_silent_long_phase_emits_progress(self):
        result, events = self.run_child("import time; time.sleep(0.35)")
        self.assertEqual(result.returncode, 0)
        beats = [row for row in events if row["event"] == "build_heartbeat"]
        self.assertGreaterEqual(len(beats), 1)
        self.assertTrue(all(row["elapsed_seconds"] > 0 and row["log_bytes"] == 0 for row in beats))

    def test_argv_is_literal_and_never_printed(self):
        marker = self.root / "not-created"
        argument = "; touch " + str(marker)
        result, events = self.run_child("import sys; print(sys.argv[1])", argument)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(self.log.read_text().strip(), argument)
        self.assertFalse(marker.exists())
        self.assertNotIn(argument.encode(), result.stdout + result.stderr)

    def test_sensitive_tail_values_redacted_but_full_log_retained(self):
        env = dict(os.environ, BUILD_API_TOKEN="very-secret-value-123")
        result, events = self.run_child("import os; print(os.environ['BUILD_API_TOKEN']); print('password=standalone-secret'); print('https://u:p@host/path'); raise SystemExit(5)", env=env)
        self.assertEqual(result.returncode, 5)
        for value in ("very-secret-value-123", "standalone-secret", "u:p@"):
            self.assertIn(value, self.log.read_text())
            self.assertNotIn(value.encode(), result.stdout + result.stderr)
        self.assertIn("[REDACTED]", events[-1]["tail"])

    def load(self):
        spec = importlib.util.spec_from_file_location("build_progress_under_test", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_label_cut_by_tail_window_does_not_expose_its_value(self):
        module = self.load()
        filler = ("f" * 383 + "\n") * 39
        value = "s" * (module.TAIL_BYTES - len(filler) - 1)
        self.log.write_text("preamble\n" * 3 + "password=" + value + "\n" + filler)
        # The tail window starts exactly after "password=", so only the value sits inside it.
        self.assertEqual(self.log.stat().st_size - module.TAIL_BYTES, len("preamble\n" * 3 + "password="))
        tail = module.failure_tail(self.log, "label")["tail"]
        self.assertNotIn("ssss", tail)
        self.assertEqual(tail.splitlines(), ["f" * 383] * 39)

    def test_huge_record_with_label_before_window_is_redacted(self):
        module = self.load()
        self.log.write_text("password=" + "y" * 30000 + " END")
        tail = module.failure_tail(self.log, "label")["tail"]
        self.assertNotIn("yyyy", tail)
        self.assertIn("[REDACTED]", tail)
        self.assertTrue(tail.endswith("END"))

    def test_output_bound_drops_whole_records_first(self):
        module = self.load()
        lines = ["record-" + str(n) + "-" + "z" * 990 for n in range(40)]
        self.log.write_text("\n".join(lines) + "\n")
        event = module.failure_tail(self.log, "label")
        self.assertLessEqual(len(module.encoded(event).encode()), module.TAIL_BYTES)
        kept = event["tail"].splitlines()
        self.assertEqual(kept, lines[-len(kept):])

    def test_quoted_sensitive_values_are_redacted(self):
        module = self.load()
        self.assertEqual(module.redact('"password": "a b c", "x": 1'), '"password": [REDACTED], "x": 1')
        self.assertEqual(module.redact("secret='x y' tail"), "secret=[REDACTED] tail")
        self.assertEqual(module.redact("token: Bearer abc.def ghi"), "token: [REDACTED] ghi")
        self.assertEqual(module.redact("tokenizer=gpt2 password_hash=abc"), "tokenizer=gpt2 password_hash=abc")

    def test_partial_record_beyond_scan_bound_is_dropped(self):
        module = self.load()
        with self.log.open("wb") as stream:
            stream.write(b"password=" + b"y" * (module.RECORD_SCAN_BYTES + module.TAIL_BYTES) + b" END")
        self.assertEqual(module.failure_tail(self.log, "label")["tail"], "")

    def test_missing_executable_returns_127_without_echoing_it(self):
        missing = str(self.root / "private-command-name")
        result = subprocess.run(self.argv([missing]), capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 127)
        self.assertNotIn(missing.encode(), result.stdout + result.stderr)
        events = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(events[-2]["launch_error"], "FileNotFoundError")

    def test_existing_log_is_not_overwritten_or_command_executed(self):
        self.log.write_bytes(b"prior log")
        result, events = self.run_child("raise RuntimeError('should not run')")
        self.assertEqual(result.returncode, 125)
        self.assertEqual(self.log.read_bytes(), b"prior log")
        self.assertEqual(events[-1]["wrapper_error"], "FileExistsError")
        self.assertEqual(result.stderr, b"")

    def test_child_signal_is_preserved(self):
        result, events = self.run_child("import os,signal; os.kill(os.getpid(),signal.SIGTERM)")
        self.assertEqual(result.returncode, -signal.SIGTERM)
        self.assertEqual(events[-2]["exit_code"], -signal.SIGTERM)
        self.assertIsNone(events[-2]["received_signal"])

    def test_uncatchable_child_kill_is_preserved(self):
        result, events = self.run_child("import os,signal; os.kill(os.getpid(),signal.SIGKILL)")
        self.assertEqual(result.returncode, -signal.SIGKILL)
        self.assertEqual(events[-2]["exit_code"], -signal.SIGKILL)

    def interrupt(self, source):
        process = subprocess.Popen(self.argv([sys.executable, "-u", "-c", source]), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if self.log.exists() and b"READY" in self.log.read_bytes():
                    break
                if process.poll() is not None:
                    self.fail("wrapper ended before readiness")
                time.sleep(0.01)
            else:
                self.fail("child readiness deadline exceeded")
            process.send_signal(signal.SIGTERM)
            stdout, stderr = process.communicate(timeout=5)
            self.assertEqual(process.returncode, -signal.SIGTERM)
            return [json.loads(line) for line in stdout.splitlines()]
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill(); process.communicate(timeout=5)

    def test_external_cancel_reaches_child_group_and_remains_a_signal(self):
        source = """import subprocess,signal,sys,time
p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])
def stop(*_):
    print('GRANDCHILD='+str(p.wait(timeout=3)),flush=True)
    raise SystemExit(0)
signal.signal(signal.SIGTERM,stop)
print('READY',flush=True)
time.sleep(60)
"""
        events = self.interrupt(source)
        self.assertIn(b"GRANDCHILD=-15", self.log.read_bytes())
        self.assertEqual(events[-2]["exit_code"], 0)
        self.assertEqual(events[-2]["received_signal"], signal.SIGTERM)

    def test_ignoring_cancel_is_bounded_by_grace(self):
        events = self.interrupt("import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('READY',flush=True); time.sleep(60)")
        self.assertEqual(events[-2]["exit_code"], -signal.SIGKILL)
        self.assertTrue(events[-2]["escalated"])

    def test_cancel_stops_group_when_leader_exits_before_ignoring_descendant(self):
        sentinel = self.root / "descendant-ticks"
        child = ("import os,signal,time\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
                 "print('ORPHAN_PID='+str(os.getpid()),flush=True)\n"
                 "with open(" + repr(str(sentinel)) + ",'ab',buffering=0) as f:\n"
                 " while True: f.write(b'x'); time.sleep(0.02)\n")
        source = ("import subprocess,sys,time\nfrom pathlib import Path\n"
                  "subprocess.Popen([sys.executable,'-u','-c'," + repr(child) + "])\n"
                  "while not Path(" + repr(str(sentinel)) + ").exists(): time.sleep(0.01)\n"
                  "print('READY',flush=True)\ntime.sleep(60)\n")
        try:
            events = self.interrupt(source)
            self.assertTrue(events[-2]["escalated"])
            self.assertEqual(events[-2]["exit_code"], -signal.SIGTERM)
            before = sentinel.stat().st_size
            time.sleep(0.15)
            self.assertEqual(sentinel.stat().st_size, before)
        finally:
            if self.log.exists():
                for line in self.log.read_text().splitlines():
                    if line.startswith("ORPHAN_PID="):
                        try:
                            os.kill(int(line.partition("=")[2]), signal.SIGKILL)
                        except ProcessLookupError:
                            pass


if __name__ == "__main__":
    unittest.main()
