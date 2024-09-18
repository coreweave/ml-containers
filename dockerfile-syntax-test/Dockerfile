# syntax=docker/dockerfile:1.4

FROM scratch as base-stage

COPY <<-"EOT" /hello-world.sh
    #!/bin/sh
    echo 'Hello world!'
EOT

FROM alpine:3.20.3

RUN --mount=type=bind,from=base-stage,source=/,target=/base-stage/ \
    ls /base-stage && \
    stat /base-stage/hello-world.sh && \
    sh /base-stage/hello-world.sh
