#!/bin/bash
# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

set -e

# set default ip to 0.0.0.0
if [[ "${NOTEBOOK_ARGS} $*" != *"--ip="* ]]; then
    NOTEBOOK_ARGS="--ip=0.0.0.0 --port 8888 --allow-root --NotebookApp.token='$INSTANCE_TOKEN' ${NOTEBOOK_ARGS}"
fi
echo "NOTEBOOK_ARGS=${NOTEBOOK_ARGS}"

# shellcheck disable=SC1091,SC2086
. /usr/local/bin/start.sh jupyterhub-singleuser ${NOTEBOOK_ARGS} "$@"