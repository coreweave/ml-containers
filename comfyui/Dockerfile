FROM ghcr.io/coreweave/ml-containers/torch-extras:es-22.04-3ce72cc-nccl-cuda12.2.2-nccl2.19.3-1-torch2.1.2-vision0.16.2-audio2.1.2-flash_attn2.3.6

RUN mkdir -p /usr/src/app

# this should already by the WORKDIR
WORKDIR /usr/src/app
RUN git clone --depth=1 https://github.com/comfyanonymous/ComfyUI
RUN pip install --upgrade pip && pip install -r ComfyUI/requirements.txt

WORKDIR /usr/src/app/ComfyUI/custom_nodes
RUN git clone --depth=1 https://github.com/ltdrdata/ComfyUI-Manager
RUN git clone --depth=1 https://github.com/pythongosssss/ComfyUI-Custom-Scripts

# # Put additional extensions you want installed in repos.txt and they'll be installed here
# # TODO: modularize this so user can pass in a URL to a text file of repos to install instead of repos.txt, e.g. a gh gist
# COPY install_script.sh /install_script.sh
# COPY repos.txt /repos.txt
# RUN chmod +x /install_script.sh
# RUN /install_script.sh

WORKDIR /usr/src/app/ComfyUI
#ENTRYPOINT ["python", "main.py", "--port", "8000", "--listen", "0.0.0.0"]
