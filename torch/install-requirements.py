#!/usr/bin/env python3
"""
Prints [build-system].requires from a pyproject.toml file,
skipping lines for packages that are not to be clobbered.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

SKIP_PACKAGES = {
    "torch",
    "torchvision",
    "torchaudio",
    "triton",
    "transformer-engine",
    "flash-attn",
}


def package_name(req):
    m = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", req.strip())
    return re.sub(r"[-_.]+", "-", m.group(0)).lower() if m else ""


def list_deps(pyproject_path):
    with open(pyproject_path, "rb") as f:
        for req in tomllib.load(f).get("build-system", {}).get("requires", []):
            if not package_name(req) in SKIP_PACKAGES:
                yield req


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", nargs="?", type=Path, default=Path("pyproject.toml")
    )
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    path = args.path

    if path.is_dir():
        path = path / "pyproject.toml"
    if not path.exists():
        # No pyproject.toml file, nothing to do
        return

    deps = "\n".join(list_deps(path))
    if not deps:
        return

    if not args.install:
        print(deps)
    else:
        proc = subprocess.run(
            [
                sys.executable,
                *("-m pip install --no-cache-dir --no-input -r /dev/stdin".split()),
            ],
            input=deps,
            text=True,
        )
        return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
