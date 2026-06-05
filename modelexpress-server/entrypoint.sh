#!/bin/bash
set -euo pipefail

redis-server --port 6379 --bind 127.0.0.1 --save "" --appendonly no --daemonize yes

until redis-cli -h 127.0.0.1 ping | grep -q PONG; do
  sleep 0.1
done

exec ./modelexpress-server --port 8001