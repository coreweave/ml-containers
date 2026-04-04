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
from typing import BinaryIO, Final, FrozenSet, Iterable, List, Sequence, Set

NVCC_PATH: Final[str] = shutil.which("nvcc")
if NVCC_PATH is None:
    raise SystemExit("NVCC wrapper: fatal: nvcc binary not found")

SCCACHE_PATH: Final[str | None] = shutil.which("sccache")

# When invoking sccache, pass only known-safe environment variables
# to avoid leaking anything in sccache's error output (which dumps
# the full subprocess env on fatal errors).
_SCCACHE_ENV_ALLOWLIST: Final[FrozenSet[str]] = frozenset({
    # nvcc
    "NVCC_APPEND_FLAGS", "NVCC_PREPEND_FLAGS", "NVCC_CCBIN",
    # Host compiler / linker (nvcc delegates to these)
    "CC", "CXX", "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS",
    "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH",
    "LIBRARY_PATH", "LD_LIBRARY_PATH", "LD_PRELOAD",
    "GCC_EXEC_PREFIX", "DEPENDENCIES_OUTPUT", "SUNPRO_DEPENDENCIES",
    # sccache
    "SCCACHE_CONF", "SCCACHE_DIR", "SCCACHE_CACHE_SIZE",
    "SCCACHE_LOG", "SCCACHE_ERROR_LOG",
    "SCCACHE_C_CUSTOM_CACHE_BUSTER",
    # System
    "PATH", "HOME", "TMPDIR",
    "LANG", "LC_ALL", "LC_CTYPE", "LC_MESSAGES",
})

def _init_sccache() -> dict[str, str] | None:
    """Ensure the sccache server is running with S3 credentials, then
    return a sanitized env (without credentials) for compiler calls."""
    if not SCCACHE_PATH:
        return None
    if (os.getenv("NVCC_WRAPPER_DISABLE_SCCACHE") or "0") != "0":
        return None

    clean_env = {k: v for k, v in os.environ.items() if k in _SCCACHE_ENV_ALLOWLIST}

    # Build a server env with S3 credentials from BuildKit secrets
    server_env = dict(clean_env)
    _secrets = {
        "AWS_ACCESS_KEY_ID": "/run/secrets/s3_access_key_id",
        "AWS_SECRET_ACCESS_KEY": "/run/secrets/s3_secret_access_key",
    }
    if not any(os.getenv(k) for k in _secrets):
        for _env, _path in _secrets.items():
            try:
                server_env[_env] = open(_path).read().strip()
            except FileNotFoundError:
                pass
    else:
        for k in _secrets:
            if k in os.environ:
                server_env[k] = os.environ[k]

    # Start the server with credentials, holding a file lock to
    # prevent parallel wrapper invocations from racing on startup.
    import fcntl
    with open("/dev/shm/sccache_server.lock", "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        result = subprocess.run(
            [SCCACHE_PATH, "--start-server"],
            env=server_env,
            capture_output=True,
        )
        if result.returncode != 0 and b"Address in use" not in result.stderr:
            raise SystemExit(
                "NVCC wrapper: fatal: sccache server failed to start"
            )

    return clean_env

SCCACHE_ENV: Final[dict[str, str] | None] = _init_sccache()

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
            if attempt == WRAPPER_ATTEMPTS:
                print("NVCC wrapper: warning: Final attempt; appending --ptxas-options=--opt-level=0")
                args.append("--ptxas-options=--opt-level=0")
        cmd = (SCCACHE_PATH, NVCC_PATH, *args) if SCCACHE_PATH else (NVCC_PATH, *args)
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=SCCACHE_ENV
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
    output: BinaryIO,
    watch_for: Iterable[bytes] = (
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
