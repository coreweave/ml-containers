# syntax=docker/dockerfile:1.4

# Canary build to validate that build.yml can build and push an image.

FROM alpine:3
COPY --link example.txt /example.txt
RUN stat /example.txt
