import argparse
import csv
import io
import os
import re
import sys
import zipfile
from importlib import metadata
from pathlib import Path

parser = argparse.ArgumentParser(
    "repackage an installed Python package into a wheel file"
)
parser.add_argument("package", help="installed package name")
parser.add_argument("-o", "--out-dir", type=Path, default=Path())
parser.add_argument("-c", "--compression-level", type=int, default=None)
args = parser.parse_args()

package_name = args.package
out_dir: Path = args.out_dir.resolve()
compression_level = args.compression_level

dist = metadata.distribution(package_name)
PACKAGE_DIR = dist.locate_file(package_name).resolve().parent
os.chdir(PACKAGE_DIR)

tags = []
for line in dist.read_text("WHEEL").splitlines():
    if line.startswith("Tag: "):
        if tags:
            tags.append(line.rsplit("-", 1)[1])
        else:
            tags.append(line[5:])
if not tags:
    raise RuntimeError("Could not identify tags")
tag = ".".join(tags)
print("Identified tag:", tag)

for f in dist.files:
    match = re.match(f"(.+-(.+)\\.dist-info)/WHEEL", str(f))
    if match:
        info_dir = Path(match.group(1))
        version = match.group(2)
        break
else:
    raise RuntimeError("Could not find version")

out_path: Path = out_dir / f"{package_name}-{version}-{tag}.whl"
print("Generating", out_path)


def is_top_level(path) -> bool:
    return PACKAGE_DIR in path.locate().resolve().parents


included = set()
with zipfile.ZipFile(
    out_path,
    "w",
    compression=zipfile.ZIP_DEFLATED,
    compresslevel=compression_level,
) as zf:
    zf: zipfile.ZipFile
    top_level_info = dist.read_text("top_level.txt")
    if top_level_info is None:
        for path in filter(is_top_level, dist.files):
            name = path.as_posix()
            if not (
                re.match(".+-.+\\.dist-info(?:/.*)?", name)
                or any(parent.name == "__pycache__" for parent in path.parents)
            ):
                included.add(name)
                zf.write(path, name)
    else:
        for top_level in map(
            Path, dist.read_text("top_level.txt").splitlines()
        ):
            if not top_level.exists():
                print(
                    f"Warning: skipping top-level entry {top_level} because it was not found",
                    file=sys.stderr,
                    flush=True,
                )
            elif top_level.is_file():
                zf.write(top_level, top_level.name)
            elif top_level.is_dir():
                for path in top_level.rglob("*"):
                    path: Path
                    if path.is_file():
                        name = (
                            path.relative_to(PACKAGE_DIR)
                            if path.is_absolute()
                            else path
                        ).as_posix()
                        if not any(
                            parent.name == "__pycache__"
                            for parent in path.parents
                        ):
                            included.add(name)
                            zf.write(path, name)
            else:
                raise RuntimeError(
                    f"Unknown top-level entry type for {top_level}"
                )

    metadata = (
        "WHEEL",
        "METADATA",
        "RECORD",
        "entry_points.txt",
        "top_level.txt",
    )
    metadata_paths = tuple(map(info_dir.joinpath, metadata))
    included.update(path.as_posix() for path in metadata_paths)
    for m, path in zip(metadata, metadata_paths):
        if path.is_file():
            if m == "RECORD":
                # Filter out files that are not included in the wheel
                with io.StringIO(newline="") as buffer, path.open("r") as file:
                    writer = csv.writer(buffer)
                    for entry in csv.reader(file):
                        if entry[0] in included:
                            writer.writerow(entry)
                    zf.writestr(path.as_posix(), buffer.getvalue())
            else:
                zf.write(path, path.as_posix())

print(f"Finished. Size of {out_path}: {out_path.stat().st_size} bytes")
