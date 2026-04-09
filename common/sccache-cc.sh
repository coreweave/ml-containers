#!/bin/sh
# Set the CC environment variable to this script to auto-apply the
# sccache.sh compiler launcher. Using this instead of sccache.sh directly
# is required in some cases to handle quirks with word-splitting and nesting.
# Make sure to set up cc or $CC_WRAPPED before using this so that it can
# pick the correct underlying compiler.
exec /opt/sccache.sh "${CC_WRAPPED:-cc}" "$@"
