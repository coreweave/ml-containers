ARG COMMIT=master
FROM python:3.9

RUN mkdir /app
WORKDIR /app

ENV tenzorizer_commit=35381e3812ba342991d30b71ce257503622ae828

RUN git clone https://github.com/coreweave/kubernetes-cloud && \
    cd kubernetes-cloud && \
    git checkout ${COMMIT} && \
    cd .. && \
    cp kubernetes-cloud/online-inference/stable-diffusion/serializer/* . && \
    pip3 install --no-cache-dir -r requirements.txt

RUN git clone https://github.com/coreweave/tensorizer && \
    cd tensorizer && \
    git checkout ${tenzorizer_commit} && \
    cd .. && \
    mv tensorizer/tensorizer.py . && \
    rm -rf tensorizer

CMD ["python3", "/app/serialize.py"]