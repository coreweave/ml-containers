FROM golang:1.23.2-bookworm AS builder
ARG TARGETOS
ARG TARGETARCH

WORKDIR /app

COPY . .

RUN go build -o /s3-perf-test main.go

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y ca-certificates

RUN update-ca-certificates

COPY --from=builder /s3-perf-test .

ENTRYPOINT [ "/s3-perf-test" ]
