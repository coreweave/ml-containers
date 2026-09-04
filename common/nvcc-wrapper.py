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

This also injects an ``sccache`` wrapper around ``nvcc`` where available.
``sccache`` error messages indicating unknown compiler failures or
cache corruption also trigger a retry, optionally without using the cache.

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
from typing import BinaryIO, Final, FrozenSet, Iterable, List, NamedTuple, Sequence, Set

NVCC_PATH: Final[str] = shutil.which("nvcc")
if NVCC_PATH is None:
    raise SystemExit("NVCC wrapper: fatal: nvcc binary not found")

# If sccache.sh is available, use it to wrap nvcc invocations.
# sccache.sh handles env sanitization and sccache server communication,
# or falls back to regular compilation if sccache is disabled or unavailable.
SCCACHE_SH: Final[str | None] = "/opt/sccache.sh" if os.path.isfile("/opt/sccache.sh") else None

WRAPPER_ATTEMPTS: Final[int] = int(os.getenv("NVCC_WRAPPER_ATTEMPTS") or 10)
if WRAPPER_ATTEMPTS < 1:
    raise SystemExit("NVCC wrapper: fatal: invalid value for NVCC_WRAPPER_ATTEMPTS")

FILTER_CODES: Final[FrozenSet[str]] = frozenset(
    filter(None, os.getenv("NVCC_WRAPPER_FILTER_CODES", "").split(";"))
)
if FILTER_CODES and not all(
    re.fullmatch(r"(?:sm|compute|lto)_\d+[af]?", a) for a in FILTER_CODES
):
    raise SystemExit("NVCC wrapper: fatal: invalid value for NVCC_WRAPPER_FILTER_CODES")

RETRY_RET_CODES: Final[FrozenSet[int]] = frozenset({
    -signal.SIGSEGV,
    -signal.SIGKILL,
    128 + signal.SIGSEGV,
    128 + signal.SIGKILL,
    255,
})

# When sccache cannot read an output file it wanted to put in the cache,
# it prints this line. Retrying the nvcc invocation usually works around it.
RETRY_STDERR_SUBSTRING: Final[bytes] = b"sccache: caused by: failed to open file"

# This matches gcc errors referencing any of sccache's tracked
# intermediate files. Such a file goes missing only when sccache
# extracted an incomplete cache entry, so the wrapper retries with
# SCCACHE_RECACHE=1 to bypass the bad entry.
RECACHE_STDERR_RE: Final[re.Pattern[bytes]] = re.compile(
    rb"\.(?:cudafe1\.[^:]+|cpp[14]\.ii):\d+:\d+: fatal error:.*: No such file or directory"
)


class MonitorResult(NamedTuple):
    retry: bool
    recache: bool


async def main(args) -> int:
    args = transform_args(args)
    ret: int = 0
    recaching: bool = False
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
            if attempt == WRAPPER_ATTEMPTS:
                print("NVCC wrapper: warning: Final attempt; appending --ptxas-options=--opt-level=0")
                args.append("--ptxas-options=--opt-level=0")
        cmd = (SCCACHE_SH, NVCC_PATH, *args) if SCCACHE_SH else (NVCC_PATH, *args)
        env = {**os.environ, "SCCACHE_RECACHE": "1"} if recaching else None
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        restart_signals: tuple = await asyncio.gather(
            monitor_stream(proc.stdout, sys.stdout.buffer),
            monitor_stream(proc.stderr, sys.stderr.buffer),
        )
        ret = await proc.wait()
        del proc
        retry_observed = any(r.retry for r in restart_signals)
        if ret == 0 or not retry_observed and ret not in RETRY_RET_CODES:
            break
        if not recaching and any(r.recache for r in restart_signals):
            recaching = True
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
    output: BinaryIO,
    watch_for: Iterable[bytes] = (
        b"Segmentation fault",
        b"Segmentation fault (core dumped)",
        b"Killed",
    ),
    retry_substring: bytes = RETRY_STDERR_SUBSTRING,
    recache_re: re.Pattern[bytes] = RECACHE_STDERR_RE,
) -> MonitorResult:
    found: bool = False
    recache: bool = False
    while line := await stream.readline():
        found = found or line.strip() in watch_for or retry_substring in line
        recache = recache or recache_re.search(line) is not None
        output.write(line)
        output.flush()
    return MonitorResult(retry=found or recache, recache=recache)


def transform_args(args: Sequence[str]) -> List[str]:
    # This filters out args of the form -gencode=arch=X,code=Y
    # or -gencode arch=X,code=Y for any code in FILTER_CODES.
    # This does not filter arguments specified using the
    # --gpu-architecture and --gpu-code flags, nor codes specified
    # among others in groups, like -gencode=arch=X,code=[Y,Z].
    if not FILTER_CODES:
        return args
    transformed_args = []
    partial: bool = False
    gencode: Set[str] = {"-gencode", "--generate-code"}
    for arg in args:
        if not partial and arg in gencode:
            partial = True
            transformed_args.append(arg)
            continue
        if partial:
            pattern: str = r"(arch=[^,]+,code=)(\S+)"
        else:
            pattern: str = r"((?:-gencode|--generate-code)=arch=\S+,code=)(\S+)"
        m: re.Match = re.fullmatch(pattern, arg)
        if m:
            code: str = m.group(2)
            if code in FILTER_CODES:
                if partial:
                    # There was a hanging `-gencode` arg before this, so delete it
                    assert transformed_args[-1] in gencode
                    del transformed_args[-1]
            elif re.fullmatch(r"\[\S+]", code):
                codes: List[str] = code[1:-1].split(",")
                filtered_codes: List[str] = [c for c in codes if c not in FILTER_CODES]
                if filtered_codes:
                    filtered_code: str = ",".join(filtered_codes)
                    if len(filtered_codes) > 1:
                        filtered_code = f"[{filtered_code}]"
                    transformed_args.append(m.group(1) + filtered_code)
                elif partial:
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
