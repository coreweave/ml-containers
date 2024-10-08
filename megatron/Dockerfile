ARG BASE_IMAGE
FROM $BASE_IMAGE

COPY requirements.txt /tmp/requirements.txt

RUN python3 -m pip install -U --no-cache-dir \
    -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

ARG COMMIT
RUN git clone https://github.com/NVIDIA/megatron-lm && \
    cd megatron-lm && \
    git checkout ${COMMIT}
