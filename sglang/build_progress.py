#!/usr/bin/env python3
"""Retain full build output while emitting bounded phase progress to CI."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

TAIL_BYTES = 16 * 1024
TAIL_LINES = 40
# How far back to look for the start of a record that the tail window cuts in half.
RECORD_SCAN_BYTES = 1024 * 1024
SECRET_NAME = re.compile(r"TOKEN|PASSWORD|PASSWD|SECRET|CREDENTIAL|ACCESS_KEY|PRIVATE_KEY|AUTHORIZATION", re.I)
SENSITIVE_FIELD = re.compile(
    r"""(?i)(\b(?:token|password|passwd|secret|credential|authorization|access[_-]?key)\b["']?\s*(?:[:=]\s*|\s+))(?:Bearer\s+)?(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[^\s,;]+)"""
)


def encoded(event):
    return json.dumps(event, sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n"


def emit(event):
    try:
        sys.stdout.write(encoded(event))
        sys.stdout.flush()
    except BrokenPipeError:
        # Loss of the CI console must not change the build command's result.
        sys.stdout = open(os.devnull, "w")


def tail_records(path):
    """Return the last complete records of the log; a record the window cuts is dropped or completed.

    Redaction below keys on labels such as "password=". A window that starts inside a record
    would hide the label while keeping the value, so the partial first record is never kept
    as is: it is dropped when a complete record follows, or completed by reading back to the
    record start (bounded by RECORD_SCAN_BYTES) when the window holds no newline at all.
    """
    size = path.stat().st_size
    start = max(0, size - TAIL_BYTES)
    with path.open("rb") as stream:
        stream.seek(start)
        raw = stream.read()
        if start > 0:
            newline = raw.find(b"\n")
            if newline >= 0:
                raw = raw[newline + 1:]
            else:
                scan_start = max(0, start - RECORD_SCAN_BYTES)
                stream.seek(scan_start)
                head = stream.read(start - scan_start)
                record_start = head.rfind(b"\n")
                if scan_start > 0 and record_start < 0:
                    raw = b""
                else:
                    raw = head[record_start + 1:] + raw
    return raw.decode("utf-8", errors="replace").splitlines()[-TAIL_LINES:]


def redact(text):
    for name, value in os.environ.items():
        if value and SECRET_NAME.search(name):
            text = text.replace(value, "[REDACTED]")
    text = SENSITIVE_FIELD.sub(lambda match: match.group(1) + "[REDACTED]", text)
    return re.sub(r"(https?://)[^/\s:@]+:[^/\s@]+@", r"\1[REDACTED]@", text)


def failure_tail(path, label):
    lines = redact("\n".join(tail_records(path))).splitlines()
    event = {"event": "build_failure_tail", "label": label, "tail": "\n".join(lines)}
    # Fit the output bound by dropping whole redacted records first; cutting inside a record
    # could otherwise separate a label from its value.
    while len(lines) > 1 and len(encoded(event).encode()) > TAIL_BYTES:
        lines.pop(0)
        event["tail"] = "\n".join(lines)
    if len(encoded(event).encode()) > TAIL_BYTES:
        text = event["tail"]
        low, high = 0, len(text)
        while low < high:
            middle = (low + high) // 2
            event["tail"] = text[middle:]
            if len(encoded(event).encode()) <= TAIL_BYTES:
                high = middle
            else:
                low = middle + 1
        event["tail"] = text[low:]
    return event


def run(label, log_path, command, heartbeat=60.0, termination_grace=30.0):
    started = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    child = None
    received = None
    signal_at = None
    escalated = False
    handled = (signal.SIGHUP, signal.SIGINT, signal.SIGTERM, signal.SIGQUIT)
    prior = {}

    def forward(signum, _frame):
        nonlocal received, signal_at
        if received is None:
            received, signal_at = signum, time.monotonic()
        if child is not None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    for signum in handled:
        prior[signum] = signal.signal(signum, forward)
    emit({"event": "build_start", "label": label, "elapsed_seconds": 0, "log_bytes": 0})
    code = 125
    launch_error = None
    log_error = None
    try:
        with os.fdopen(descriptor, "wb", buffering=0) as output:
            try:
                child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            except OSError as error:
                code = 127 if isinstance(error, FileNotFoundError) else 126
                launch_error = type(error).__name__
            if child is not None:
                if received is not None:
                    forward(received, None)
                next_heartbeat = time.monotonic() + heartbeat
                while True:
                    try:
                        code = child.wait(timeout=min(0.2, max(0.01, next_heartbeat - time.monotonic())))
                        break
                    except subprocess.TimeoutExpired:
                        current = time.monotonic()
                        if received is not None and not escalated and current - signal_at >= termination_grace:
                            try:
                                os.killpg(child.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                            escalated = True
                        if current >= next_heartbeat:
                            emit({"event": "build_heartbeat", "label": label,
                                  "elapsed_seconds": round(current - started, 3),
                                  "log_bytes": os.fstat(output.fileno()).st_size})
                            next_heartbeat = current + heartbeat
                # The leader may exit before descendants acknowledge cancellation.
                while received is not None and not escalated:
                    try:
                        os.killpg(child.pid, 0)
                    except ProcessLookupError:
                        break
                    remaining = termination_grace - (time.monotonic() - signal_at)
                    if remaining <= 0:
                        try:
                            os.killpg(child.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        escalated = True
                        break
                    time.sleep(min(0.05, remaining))
            try:
                os.fsync(output.fileno())
            except OSError as error:
                log_error = type(error).__name__
                if code == 0:
                    code = 125
    finally:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
        for signum, handler in prior.items():
            signal.signal(signum, handler)
    event = {"event": "build_end", "label": label, "elapsed_seconds": round(time.monotonic() - started, 3),
             "log_bytes": log_path.stat().st_size, "exit_code": code,
             "child_exit_code": child.returncode if child is not None else None,
             "received_signal": received, "escalated": escalated}
    if launch_error is not None:
        event["launch_error"] = launch_error
    if log_error is not None:
        event["log_error"] = log_error
    emit(event)
    if code != 0 or received is not None:
        emit(failure_tail(log_path, label))
    return code, received


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--heartbeat-seconds", type=float, default=60.0)
    parser.add_argument("--termination-grace-seconds", type=float, default=30.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", args.label):
        parser.error("label must be a short phase identifier")
    if not args.log_path.is_absolute() or not 0.1 <= args.heartbeat_seconds <= 3600 or not 0.1 <= args.termination_grace_seconds <= 300:
        parser.error("absolute log path and bounded positive intervals required")
    if len(args.command) < 2 or args.command[0] != "--":
        parser.error("one explicit command after -- is required")
    try:
        code, received = run(args.label, args.log_path, args.command[1:], args.heartbeat_seconds, args.termination_grace_seconds)
    except OSError as error:
        emit({"event": "build_end", "label": args.label, "exit_code": 125, "wrapper_error": type(error).__name__})
        return 125
    signum = received if received is not None else -code if code < 0 else None
    if signum is not None:
        if signum not in (signal.SIGKILL, signal.SIGSTOP):
            signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
        return 128 + signum
    return code


if __name__ == "__main__":
    raise SystemExit(main())
