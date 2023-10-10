FROM python:3.12.0rc2-slim
ARG COMMIT=main

RUN mkdir /app
WORKDIR /app

RUN git clone https://github.com/coreweave/tensorizer && \
    cd tensorizer && \
    git checkout ${COMMIT} && \
    pip3 install . && \
    cd ..

CMD ["python3", "-m", "tensorizer.tensorizer"]