FROM ghcr.io/coreweave/ml-containers/torch:afecfe9-base-cuda12.0.1-torch2.0.0-vision0.15.1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -qq update && \
    apt-get -qq install --no-install-recommends -y git curl && \
    apt-get clean

RUN mkdir /app
WORKDIR /app

ARG COMMIT=4ff78daa65b6a4a944c5f14f102e5f28770bcfcc
RUN git clone --filter=blob:none https://github.com/coreweave/kubernetes-cloud && \
    cd kubernetes-cloud && \
    git checkout ${COMMIT} && \
    cd .. && \
    cp kubernetes-cloud/online-inference/hf-llm/service/* .  && \
    cp kubernetes-cloud/online-inference/hf-llm/serializer/serialize.py . && \
    rm -rf kubernetes-cloud

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt
