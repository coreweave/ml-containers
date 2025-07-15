#!/bin/env python3

"""
Wraps invocations of ``nvcc``, watching for evidence of SIGKILL or SIGSEGV,
and then re-running the ``nvcc`` command a configurable number of times.

Checking for SIGKILL or SIGSEGV is implemented by checking for either:

- A subprocess return code indicating either of these signals, or
- The standard ``dash`` error messages for either signal.

``dash`` status messages are checked as NVCC utilizes ``sh``
subprocesses internally, and ``sh`` usually resolves to
the ``dash`` shell within Ubuntu container images.
"""

import asyncio
import os
import shutil
import signal
import subprocess
import sys
import typing

NVCC_PATH: typing.Final[str] = shutil.which("nvcc")
if NVCC_PATH is None:
    raise SystemExit("NVCC wrapper: fatal: nvcc binary not found")

WRAPPER_ATTEMPTS: typing.Final[int] = int(os.getenv("NVCC_WRAPPER_ATTEMPTS") or 10)
if WRAPPER_ATTEMPTS < 1:
    raise SystemExit("NVCC wrapper: fatal: invalid value for NVCC_WRAPPER_ATTEMPTS")

RETRY_RET_CODES: typing.Final[typing.FrozenSet[int]] = frozenset(
    {
        -signal.SIGSEGV,
        -signal.SIGKILL,
        128 + signal.SIGSEGV,
        128 + signal.SIGKILL,
        255,
    }
)


async def main(args) -> int:
    ret: int = 0
    for attempt in range(1, WRAPPER_ATTEMPTS + 1):
        if attempt > 1:
            print(
                "NVCC wrapper: info:"
                f" Retrying [{attempt:d}/{WRAPPER_ATTEMPTS:d}]"
                f" after exit code {ret:d}",
                file=sys.stderr,
                flush=True,
            )
            # Wait an exponentially increasing amount of time
            # before trying again, up to one minute
            await asyncio.sleep(min(60, int(1.5**attempt)))
        proc = await asyncio.create_subprocess_exec(
            NVCC_PATH, *args, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        restart_signals: tuple = await asyncio.gather(
            monitor_stream(proc.stdout, sys.stdout.buffer),
            monitor_stream(proc.stderr, sys.stderr.buffer),
        )
        ret = await proc.wait()
        del proc
        if ret == 0 or not any(restart_signals) and ret not in RETRY_RET_CODES:
            break
    else:
        print(
            "NVCC wrapper: info:"
            f" Maximum attempts reached, exiting with status {ret:d}",
            file=sys.stderr,
            flush=True,
        )
    return ret


async def monitor_stream(
    stream: asyncio.StreamReader,
    output: typing.BinaryIO,
    watch_for: typing.Iterable[bytes] = (
        b"Segmentation fault",
        b"Segmentation fault (core dumped)",
        b"Killed",
    ),
) -> bool:
    found: bool = False
    while line := await stream.readline():
        found = found or line.strip() in watch_for
        output.write(line)
        output.flush()
    return found


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main(sys.argv[1:])))
    except KeyboardInterrupt:
        sys.exit(130)
