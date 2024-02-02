FROM ghcr.io/coreweave/ml-containers/torch:es-22.04-3ce72cc-base-cuda12.2.2-torch2.1.2-vision0.16.2-audio2.1.2
ARG COMMIT=main

RUN mkdir /app
WORKDIR /app

RUN git clone https://github.com/coreweave/tensorizer && \
    cd tensorizer && \
    git checkout ${COMMIT} && \
    pip3 install .