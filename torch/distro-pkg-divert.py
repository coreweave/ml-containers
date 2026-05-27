#!/usr/bin/env python3
"""Move apt-installed Python packages out of dist-packages so a pip install
of the same name can take over the import namespace. Each file moves to
<stash-root>/<pkg>/<original-path>, with a dpkg diversion registered so
future apt operations land at the stash instead of the original path.

When --list-removed is set, the script also writes requirements.txt-style
lines for each diverted distribution to stdout, suitable for piping into
pip:

    pip install -r <(distro-pkg-divert.py --list-removed python3-foo python3-bar)
"""

import argparse
import email
import shutil
import subprocess
from pathlib import Path

DEFAULT_DIST_PACKAGES = Path("/usr/lib/python3/dist-packages")
DEFAULT_STASH_ROOT = Path("/var/lib/dpkg-divert-stash")


def dpkg_listed_paths(pkg: str) -> list[Path]:
    out = subprocess.run(
        ["dpkg", "-L", pkg], check=True, capture_output=True, text=True
    ).stdout
    return [Path(line) for line in out.splitlines() if line]


def read_distribution_metadata(paths: list[Path]) -> dict[str, str]:
    """Read Name and Version from each .dist-info or .egg-info entry in
    ``paths``. Returns {name: version}; .dist-info wins over .egg-info for
    the same distribution."""
    found: dict[str, str] = {}
    for path in paths:
        if path.name.endswith(".dist-info"):
            meta = path / "METADATA"
        elif path.name.endswith(".egg-info"):
            # .egg-info is either a directory containing PKG-INFO or a
            # single file with PKG-INFO content.
            meta = path / "PKG-INFO" if path.is_dir() else path
        else:
            continue
        if not meta.is_file():
            raise SystemExit(f"{path}: missing distribution metadata")
        with meta.open() as fh:
            msg = email.message_from_file(fh)
        name = msg["Name"]
        version = msg["Version"]
        if not (name and version):
            raise SystemExit(f"{meta}: missing Name or Version header")
        if path.name.endswith(".dist-info") or name not in found:
            found[name] = version
    return found


def divert_package(
    pkg: str, dist_packages: Path, stash_root: Path
) -> dict[str, str]:
    paths = dpkg_listed_paths(pkg)
    distributions = read_distribution_metadata(paths)
    stash = stash_root / pkg

    files = [p for p in paths if dist_packages in p.parents and p.is_file()]
    dirs = [p for p in paths if dist_packages in p.parents and p.is_dir()]

    for path in files:
        dest = stash / path.relative_to("/")
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "dpkg-divert",
                "--local",
                "--rename",
                "--divert",
                str(dest),
                str(path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )

    # Deepest-first so children rmdir before their parents. rmdir raises on
    # non-empty directories, which surfaces any unexpected leftover.
    for path in sorted(dirs, reverse=True):
        pycache = path / "__pycache__"
        if pycache.is_dir():
            shutil.rmtree(pycache)
        path.rmdir()

    return distributions


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Divert apt-installed Python packages out of dist-packages."
        ),
    )
    parser.add_argument(
        "packages",
        nargs="+",
        metavar="PKG",
        help="apt package name (e.g. python3-cryptography)",
    )
    parser.add_argument(
        "--dist-packages",
        type=Path,
        default=DEFAULT_DIST_PACKAGES,
        help="root the packages live under (default: %(default)s)",
    )
    parser.add_argument(
        "--stash-root",
        type=Path,
        default=DEFAULT_STASH_ROOT,
        help="where diverted files land (default: %(default)s)",
    )
    parser.add_argument(
        "--list-removed",
        action="store_true",
        help=(
            'write "name==version" lines for each diverted distribution'
            " to stdout in requirements.txt format"
        ),
    )
    args = parser.parse_args()

    diverted: dict[str, str] = {}
    for pkg in args.packages:
        diverted.update(
            divert_package(pkg, args.dist_packages, args.stash_root)
        )

    if args.list_removed:
        for name, version in diverted.items():
            print(f"{name}=={version}")


if __name__ == "__main__":
    main()
