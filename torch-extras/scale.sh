#!/bin/sh

set -e;

VAL="$1";
DIVISOR="$2";
MAXIMUM="$3";

[ -n "$VAL" ];

if [ -n "$DIVISOR" ];
then VAL="$((( $VAL + $DIVISOR - 1 ) / $DIVISOR))";
fi;

if [ -n "$MAXIMUM" ];
then VAL="$((VAL > MAXIMUM ? MAXIMUM : VAL))";
fi;

echo "$VAL";
