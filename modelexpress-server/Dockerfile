FROM rust:1.90 AS builder

RUN apt-get update && \
    apt-get install -y protobuf-compiler && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

ARG MODELEXPRESS_COMMIT

RUN  git clone https://github.com/ai-dynamo/modelexpress.git && \
  cd modelexpress && git checkout ${MODELEXPRESS_COMMIT} && \
  cargo build --release --bin modelexpress-server

FROM ubuntu:24.04 AS runtime

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates libssl-dev && \
    apt-get install -y redis-server && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/modelexpress/target/release/modelexpress-server .
COPY --chmod=+x entrypoint.sh .
ARG MODELEXPRESS_PORT
EXPOSE ${MODELEXPRESS_PORT}
ENV MODELEXPRESS_PORT=${MODELEXPRESS_PORT}
ENV REDIS_URL=redis://localhost:6379
ENV MX_METADATA_BACKEND=redis

CMD ["./entrypoint.sh"]
