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

This wrapper also has the ability to filter out some -gencode flags.
Gencode flags to filter out should be identified by their code parameter
in a semicolon-delimited list stored in the NVCC_WRAPPER_FILTER_CODES
environment variable.
"""

import asyncio
import os
import re
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

FILTER_CODES: typing.Final[typing.FrozenSet[str]] = frozenset(
    filter(None, os.getenv("NVCC_WRAPPER_FILTER_CODES", "").split(";"))
)
if FILTER_CODES and not all(
    re.fullmatch(r"(?:sm|compute|lto)_\d+[af]?", a) for a in FILTER_CODES
):
    raise SystemExit("NVCC wrapper: fatal: invalid value for NVCC_WRAPPER_FILTER_CODES")

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
    args = transform_args(args)
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


def transform_args(args: typing.Sequence[str]) -> typing.Sequence[str]:
    # This filters out args of the form -gencode=arch=X,code=Y
    # or -gencode arch=X,code=Y for any code in FILTER_CODES.
    # This does not filter arguments specified using the
    # --gpu-architecture and --gpu-code flags, nor codes specified
    # among others in groups, like -gencode=arch=X,code=[Y,Z].
    transformed_args = []
    partial: bool = False
    gencode: typing.Set[str] = {"-gencode", "--generate-code"}
    for arg in args:
        if not partial and arg in gencode:
            partial = True
            transformed_args.append(arg)
            continue
        if partial:
            pattern: str = r"arch=[^,]+,code=(\S+)"
        else:
            pattern: str = r"(?:-gencode|--generate-code)=arch=[^,]+,code=(\S+)"
        m = re.fullmatch(pattern, arg)
        if m:
            code = m.group(1)
            if code in FILTER_CODES:
                if partial:
                    assert transformed_args[-1] in gencode
                    del transformed_args[-1]
            else:
                transformed_args.append(arg)
        else:
            transformed_args.append(arg)
        partial = False
    return transformed_args


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main(sys.argv[1:])))
    except KeyboardInterrupt:
        sys.exit(130)
