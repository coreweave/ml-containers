FROM ghcr.io/coreweave/ml-containers/torch:afecfe9-base-cuda12.0.1-torch2.0.0-vision0.15.1
ENV DEBIAN_FRONTEND=noninteractive
 
RUN apt-get -qq update && \
    apt-get -qq install --no-install-recommends -y \
        python3 python3-pip git curl && \
    apt-get clean

RUN mkdir /app
WORKDIR /app

ARG COMMIT=master
RUN git clone https://github.com/coreweave/kubernetes-cloud && \
    cd kubernetes-cloud && \
    git checkout ${COMMIT} && \
    cd .. && \
    cp kubernetes-cloud/online-inference/stable-diffusion/service/* .  && \
    cp kubernetes-cloud/online-inference/stable-diffusion/serializer/serialize.py . && \
    rm -rf kubernetes-cloud

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt
