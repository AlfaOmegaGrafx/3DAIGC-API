#!/usr/bin/env bash
# Ensure Redis is reachable for 3DAIGC-API (job queue). Starts 3daigc-redis if needed.
#
# Usage:
#   bash scripts/ensure_redis.sh
#
# Env: P3D_REDIS_URL (default redis://localhost:6379)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REDIS_URL="${P3D_REDIS_URL:-redis://localhost:6379}"

redis_ping() {
  if command -v redis-cli &>/dev/null; then
    redis-cli -u "$REDIS_URL" ping &>/dev/null
    return $?
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx '3daigc-redis'; then
    docker exec 3daigc-redis redis-cli ping 2>/dev/null | grep -qx PONG
    return $?
  fi
  return 1
}

start_redis() {
  if ! command -v docker &>/dev/null; then
    echo "ERROR: docker not found — cannot start Redis for $REDIS_URL"
    exit 1
  fi
  cd "$ROOT"
  docker start 3daigc-redis 2>/dev/null \
    || docker compose up -d redis 2>/dev/null \
    || docker run -d -p 6379:6379 --restart unless-stopped --name 3daigc-redis \
      redis:7-alpine redis-server --appendonly yes --maxmemory 2gb --maxmemory-policy allkeys-lru
  sleep 2
}

if redis_ping; then
  echo "Redis OK ($REDIS_URL)"
  exit 0
fi

echo "Redis not reachable — starting 3daigc-redis..."
start_redis

if redis_ping; then
  echo "Redis OK ($REDIS_URL)"
  exit 0
fi

echo "ERROR: Redis still unreachable at $REDIS_URL"
exit 1
