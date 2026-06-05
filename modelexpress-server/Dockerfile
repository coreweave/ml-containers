FROM rust:1.90 AS builder

RUN apt-get update && \
    apt-get install -y protobuf-compiler && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN  git clone https://github.com/ai-dynamo/modelexpress.git && \
  cd modelexpress && git checkout 6cf440700e9341ce17718a51b981dbccf81b6c57 && \
  cargo build --release --bin modelexpress-server

FROM ubuntu:24.04 AS runtime

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates libssl-dev && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/modelexpress-server .

EXPOSE 8001

CMD ["./modelexpress-server", "--port", "8001"]
