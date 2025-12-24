FROM ghcr.io/coreweave/ml-containers/torch:8a60b2d-nccl-cuda12.9.1-ubuntu22.04-nccl2.28.3-1-torch2.8.0-vision0.23.0-audio2.8.0-abi1
ENV DEBIAN_FRONTEND=noninteractive
 
RUN apt-get -qq update && \
    apt-get -qq install --no-install-recommends -y git curl && \
    apt-get clean

RUN mkdir /app
WORKDIR /app

ARG COMMIT=master
RUN git clone --filter=tree:0 --sparse --no-checkout https://github.com/coreweave/kubernetes-cloud && \
    git -C kubernetes-cloud sparse-checkout set online-inference/stable-diffusion && \
    git -C kubernetes-cloud checkout "${COMMIT}" && \
    cp kubernetes-cloud/online-inference/stable-diffusion/service/* .  && \
    cp kubernetes-cloud/online-inference/stable-diffusion/serializer/serialize.py . && \
    rm -rf kubernetes-cloud

RUN sed -i \
      -e '/^torch==/d' \
      -e 's/^tensorizer==2\.3\..*$/tensorizer==2.12.0/' \
      -e 's/^diffusers==0\.20\..*$/diffusers==0.36.0/' \
      -e 's/^transformers==4\.33\..*$/transformers==4.57.3/' \
      requirements.txt

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt
