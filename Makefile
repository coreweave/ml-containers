SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
.DEFAULT_GOAL := start
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

ifeq ($(origin .RECIPEPREFIX), undefined)
  $(error This Make does not support .RECIPEPREFIX. Please use GNU Make 4.0 or later)
endif
.RECIPEPREFIX = >

PROJECT:=ml-containers
IMAGES:=torch torch-extras

.PHONY: help
help: ## displays this help screen.
> @grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort | awk '\
		BEGIN {FS = ":.*?## "}; \
		{printf "\033[36m%-30s\033[0m %s\n", $$1, $$2};\
		'

.PHONY: lint
lint: lint-dockerfile

.PHONY: lint-dockerfile
lint-dockerfile:
> @echo "Checking Dockerfile syntax..."
> @for image in ${IMAGES}; do \
> 	docker run --rm hadolint/hadolint hadolint - < $$image/Dockerfile; \
> done

.PHONY: buildx-create
buildx-create: ## create buildx builder zstd-builder
>- @docker buildx create \
> 	--name zstd-builder \
>		--driver docker-container \
>		--driver-opt image=moby/buildkit:v0.12.2 2>/dev/null || true

.PHONY: buildx-use ## use buildx builder zstd-builder
buildx-use:
>- @docker buildx use zstd-builder

.PHONY: buildx-rm ## remove buildx builder zstd-builder
buildx-rm:
> @docker buildx rm zstd-builder

.PHONY: build
build: buildx-create buildx-use ## build all images
> @for image in ${IMAGES}; do \
> 	pushd $$image
> 	docker buildx build \
> 		--file Dockerfile \
> 	  --output "type=image,name=$$image:zstd-latest,compression=zstd,compression-level=3,force-compression=true,push=true" \
> 		.
>		popd;
>	done
