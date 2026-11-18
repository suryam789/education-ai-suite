#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export SC_CONFIG_PATH="${SC_CONFIG_PATH:-$ROOT/config.yaml}"
python3 -m utils.service_config >&2
REGISTRY="$(python3 -m utils.registry_mode)"

ENV_FILES=()
if [[ -f "$ROOT/.env" ]]; then
  ENV_FILES+=(--env-file "$ROOT/.env")
fi
ENV_FILES+=(--env-file "$ROOT/.generated/compose.env")
while IFS='=' read -r key value; do
  unset "$key"
done < "$ROOT/.generated/compose.env"

COMPOSE_FILES=(-f docker/docker-compose.yml -f docker/docker-compose.gpu.yml)
if [[ "$REGISTRY" == "false" ]]; then
  COMPOSE_FILES+=(-f docker/docker-compose.local-build.yml)
fi
exec docker compose "${ENV_FILES[@]}" "${COMPOSE_FILES[@]}" "$@"