#!/bin/sh

PATTERN="";
for DEP in "$@"; do {
  PATTERN="${PATTERN:+$PATTERN|}${DEP}";
}; done;
PATTERN="^(${PATTERN})\b";

python3 -m pip list --format freeze --disable-pip-version-check \
  | { grep -iE "${PATTERN}" || :; };
