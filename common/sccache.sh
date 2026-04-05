#!/bin/bash
# Wraps a compiler invocation through sccache with a sanitized environment.
# Usage: sccache.sh <compiler> [args...]
#
# Only whitelisted env vars are passed to the sccache subprocess, preventing
# sccache's error output from dumping unrelated environment variables.
# Falls back to running the compiler directly if sccache is unavailable,
# DISABLE_SCCACHE is set, or no server port is configured.

if [[ ! -x /opt/sccache ]] || \
  [[ ! "${DISABLE_SCCACHE:-0}" = 0 ]] || \
  [[ -z "${SCCACHE_SERVER_PORT-}" ]] || \
  [[ -n "${SCCACHE_GUARD-}" ]]; then
  exec -- "$@"
fi

# Guard against nesting.
export SCCACHE_GUARD=1

SCCACHE_ENV_ALLOWLIST=(
  # nvcc
  'NVCC_APPEND_FLAGS' 'NVCC_PREPEND_FLAGS' 'NVCC_CCBIN'
  # Host compiler / linker
  'CC' 'CXX' 'CFLAGS' 'CXXFLAGS' 'CPPFLAGS' 'LDFLAGS'
  'CPATH' 'C_INCLUDE_PATH' 'CPLUS_INCLUDE_PATH'
  'COMPILER_PATH' 'GCC_EXEC_PREFIX'
  'LIBRARY_PATH' 'LD_LIBRARY_PATH' 'LD_RUN_PATH' 'LD_PRELOAD'
  'DEPENDENCIES_OUTPUT' 'SUNPRO_DEPENDENCIES'
  'SOURCE_DATE_EPOCH'
  # sccache
  'SCCACHE_CONF' 'SCCACHE_DIR' 'SCCACHE_CACHE_SIZE'
  'SCCACHE_LOG' 'SCCACHE_ERROR_LOG'
  'SCCACHE_C_CUSTOM_CACHE_BUSTER' 'SCCACHE_RECACHE'
  'SCCACHE_SERVER_PORT' 'SCCACHE_SERVER_UDS' 'SCCACHE_IDLE_TIMEOUT'
  # State for our own sccache wrapper scripts
  'SCCACHE_GUARD' 'CC_WRAPPED' 'CXX_WRAPPED'
  # System
  'PATH' 'HOME' 'TMPDIR' 'TMP' 'TEMP'
  'LANG' 'LC_ALL' 'LC_CTYPE' 'LC_MESSAGES'
)

SCCACHE_ENV=()
for VAR in "${SCCACHE_ENV_ALLOWLIST[@]}"; do
  if [[ -v "${VAR}" ]]; then
    SCCACHE_ENV+=("${VAR}=${!VAR}")
  fi
done

exec -- env -i -- "${SCCACHE_ENV[@]}" SCCACHE_START_SERVER=0 /opt/sccache -- "$@"
