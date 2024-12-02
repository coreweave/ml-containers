ARG BASE_IMAGE
FROM $BASE_IMAGE

COPY requirements.txt /tmp/requirements.txt

RUN python3 -m pip install -U --no-cache-dir \
    -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

ARG COMMIT
RUN git clone https://github.com/NVIDIA/megatron-lm && \
    cd megatron-lm && \
    git checkout ${COMMIT} && \
    rm -rf .git

RUN <<"EOF"
#!/bin/env python3
from importlib.util import find_spec
from pathlib import Path

spec = find_spec("flash_attn_interface").origin
print("flash_attn_interface:", spec)

p = Path(spec)
if not p.is_file():
    raise SystemExit("flash_attn_interface path is not a file")

d = p.parent / "flashattn_hopper"
if d.exists():
    raise SystemExit(f'"{d}" already exists')

d.mkdir(mode=0o755, parents=False, exist_ok=False)
new = d / p.name
new.symlink_to(p)
print(f"Created new symlink at {new}")
EOF
