#!/bin/sh
# Start an sccache server with S3 credentials on a random port.
# Source this script to set SCCACHE_SERVER_PORT and register a
# cleanup trap:
#
#   . /build/sccache-start.sh
#
# The server will be stopped automatically when the shell exits.

SCCACHE_SERVER_PORT="$(
  port_in_use() {
    bash -c 'echo >"/dev/tcp/127.0.0.1/$1"' bash "$1" 2>/dev/null
  }

  for i in $(seq 20); do
    candidate=$(shuf -i 10000-60000 -n 1) || exit 1
    if ! port_in_use "${candidate}"; then
      echo "${candidate}"
      exit 0
    fi
  done
  echo 'sccache-start: fatal: could not find a free port' >&2
  exit 1
)" || return 1
export SCCACHE_SERVER_PORT

# Auto-shutdown after 30 minutes if idle, to prevent leaking
# daemon processes onto the builder host
SCCACHE_IDLE_TIMEOUT=1800
export SCCACHE_IDLE_TIMEOUT

# Start the server in a subshell with credentials so they
# keep out of the outer shell's environment.
(
  if [ -f /run/secrets/s3_access_key_id ] && [ -f /run/secrets/s3_secret_access_key ]; then
    AWS_ACCESS_KEY_ID=$(cat /run/secrets/s3_access_key_id) && \
    export AWS_ACCESS_KEY_ID && \
    AWS_SECRET_ACCESS_KEY=$(cat /run/secrets/s3_secret_access_key) && \
    export AWS_SECRET_ACCESS_KEY
  fi && \
  /opt/sccache --start-server
) || { echo 'sccache-start: fatal: server failed to start' >&2; return 1; }

# Stop the server when the step finishes, showing stats on success
_sccache_cleanup() {
  _rc="$?"
  if [ "${_rc}" -eq 0 ]; then
    echo 'sccache stats:'
    /opt/sccache --show-stats | sed 's@^@  @'
  fi
  /opt/sccache --stop-server || echo 'sccache-start: warning: failed to stop server' >&2
  exit "${_rc}"
}
trap _sccache_cleanup EXIT

printf 'sccache server started on port %d\n' "${SCCACHE_SERVER_PORT}"
