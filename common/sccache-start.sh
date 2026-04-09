#!/bin/sh
# Start an sccache server with S3 credentials on a Unix domain socket.
# Source this script to set SCCACHE_SERVER_UDS and register a
# cleanup trap. The server will be stopped when the shell exits.

# sccache's idle timeout can be wonky. Set it long enough for any build,
# end-to-end, but short enough that leaked daemons don't persist
# indefinitely on builder nodes.
export SCCACHE_IDLE_TIMEOUT=64800

# Use a Unix domain socket in /sccache, which is a per-step tmpfs mount.
# Compared to using TCP on a loopback address, this gives us the benefits
# of having no port conflicts, no TIME_WAIT interference, and isolation
# from all other build steps.
export SCCACHE_SERVER_UDS='/sccache/server.sock'

if [ ! -e '/sccache' ]; then
  echo 'sccache-start: /sccache: directory not found' >&2
  return 1
fi

if [ -e "${SCCACHE_SERVER_UDS}" ]; then
  echo 'sccache-start: socket already exists, mysteriously' >&2
  return 1
fi

: "${AWS_ACCESS_KEY_ID:?} ${AWS_SECRET_ACCESS_KEY:?}"

# Start the server with credentials from the environment,
# then unset them so nothing later in the step can read them.
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

if [ "${_sccache_rc}" -ne 0 ]; then
  echo 'sccache-start: fatal: server failed to start' >&2
  unset _sccache_stderr _sccache_rc
  return 1
fi
unset _sccache_stderr _sccache_rc

# Stop the server when the step finishes.
# This will also show cache stats implicitly.
_sccache_cleanup() {
  _rc="$?"
  echo 'sccache info:'
  {
    /opt/sccache --stop-server || \
    echo '  sccache-start: warning: failed to stop server' >&2
  } | sed 's@^@  @'
  rm -f "${SCCACHE_SERVER_UDS}"
  exit "${_rc}"
}
trap _sccache_cleanup EXIT

printf 'sccache server started on %s\n' "${SCCACHE_SERVER_UDS}"
