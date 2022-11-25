FROM gooseai/torch-base:6cfdc11

RUN apt-get install -y cuda-nvcc-11-3 cuda-nvml-dev-11-3 libcurand-dev-11-3 \
                       libcublas-dev-11-3 libcusparse-dev-11-3 \
                       libcusolver-dev-11-3 cuda-nvprof-11-3 \
                       ninja-build git && \
    apt-get clean

RUN mkdir /app
WORKDIR /app

RUN git clone https://github.com/coreweave/kubernetes-cloud
RUN cp kubernetes-cloud/sd-finetuner-workflow/sd-finetuner/* .
RUN pip3 install --no-cache-dir -r requirements.txt

CMD [ "/usr/bin/python3", "finetuner.py" ]
