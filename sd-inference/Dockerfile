ARG COMMIT=master
FROM gooseai/torch-base:6cfdc11

ENV tenzorizer_commit=35381e3812ba342991d30b71ce257503622ae828

RUN apt-get install -y git

RUN mkdir /app
WORKDIR /app

RUN git clone https://github.com/coreweave/kubernetes-cloud && \
    cd kubernetes-cloud && \
    git checkout ${COMMIT} && \
    cd .. && \
    cp kubernetes-cloud/online-inference/stable-diffusion/service/* .

RUN git clone https://github.com/coreweave/tensorizer && \
    cd tensorizer && \
    git checkout ${tenzorizer_commit} && \
    cd .. && \
    mv tensorizer/tensorizer.py . && \
    rm -rf tensorizer

RUN pip3 install --no-cache-dir -r requirements.txt

CMD [ "/usr/bin/python3", "service.py" ]
