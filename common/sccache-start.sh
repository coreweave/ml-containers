#!/bin/sh
# Start an sccache server with S3 credentials on a random port.
# Source this script to set SCCACHE_SERVER_PORT and register a
# cleanup trap. The server will be stopped when the shell exits.

# Auto-shutdown after 30 minutes if idle, to mitigate any
# daemon processes leaked onto the builder host.
export SCCACHE_IDLE_TIMEOUT=1800

# Pick a free port, start the server, and retry up to 5 times on
# port conflicts. These can occur from a TOCTOU race between the
# port check and bind, or from TIME_WAIT sockets left by a recently
# stopped server (sccache does not set SO_REUSEADDR).
_sccache_started=false
for _sccache_attempt in $(seq 5); do
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

  : "${AWS_ACCESS_KEY_ID:?} ${AWS_SECRET_ACCESS_KEY:?}"

  # Start the server with credentials from the environment,
  # then unset them so nothing later in the step can read them.
  # Capture stderr to check for port conflicts while still emitting it.
  _sccache_stderr="$(
    /opt/sccache --start-server 3>&2 2>&1 1>&3 3>&-
  )" 2>&1  # Golly, these file descriptors are really dancing out here.
  _sccache_rc="$?"
  unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY

  # Shouldn't be necessary, but just to make sure that BuildKit isn't weird.
  if [ -n "${AWS_ACCESS_KEY_ID}${AWS_SECRET_ACCESS_KEY}" ]; then
    echo 'sccache-start: failed to drop credentials' >&2
    return 1
  fi

  if [ -n "${_sccache_stderr}" ]; then
    printf '%s\n' "${_sccache_stderr}" >&2
  fi

  if [ "${_sccache_rc}" -eq 0 ]; then
    _sccache_started=true
    break
  fi

  if printf '%s' "${_sccache_stderr}" | grep -q 'Address in use'; then
    printf 'sccache-start: port %d conflict, retrying (%d/5)\n' \
      "${SCCACHE_SERVER_PORT}" "${_sccache_attempt}" >&2
    continue
  fi

  echo 'sccache-start: fatal: server failed to start' >&2
  unset _sccache_started _sccache_attempt _sccache_stderr _sccache_rc
  return 1
done

if [ "${_sccache_started}" != true ]; then
  echo 'sccache-start: fatal: server failed to start after 5 port conflict retries' >&2
  unset _sccache_started _sccache_attempt _sccache_stderr _sccache_rc
  return 1
fi
unset _sccache_started _sccache_attempt _sccache_stderr _sccache_rc

# Stop the server when the step finishes.
# This will also show cache stats implicitly.
_sccache_cleanup() {
  _rc="$?"
  echo 'sccache info:'
  {
    /opt/sccache --stop-server || \
    echo '  sccache-start: warning: failed to stop server' >&2
  } | sed 's@^@  @'
  exit "${_rc}"
}
trap _sccache_cleanup EXIT

printf 'sccache server started on port %d\n' "${SCCACHE_SERVER_PORT}"
