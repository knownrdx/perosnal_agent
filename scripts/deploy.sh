#!/usr/bin/env bash
# One-shot VPS deployment helper for Ubuntu 24.04.
#
#   ./scripts/deploy.sh          build + start + wait for health
#   ./scripts/deploy.sh --pull   also pull the LLM model
#
set -euo pipefail

cd "$(dirname "$0")/.."

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info() { echo -e "${GREEN}==>${NC} $*"; }
warn() { echo -e "${YELLOW}!!${NC} $*"; }
fail() { echo -e "${RED}xx${NC} $*" >&2; exit 1; }

command -v docker >/dev/null || fail "docker is not installed: curl -fsSL https://get.docker.com | sh"
docker compose version >/dev/null 2>&1 || fail "docker compose v2 is required"

[ -f .env ] || fail "no .env file. Run: cp .env.example .env && nano .env"

# --- validate required settings -------------------------------------------
missing=()
check_var() {
  local value
  value="$(grep -E "^$1=" .env | head -1 | cut -d= -f2- || true)"
  if [ -z "$value" ] || [ "$value" = "change-me-to-a-long-random-string" ] || [ "$value" = "change-me-too" ]; then
    missing+=("$1")
  fi
}
check_var TELEGRAM_BOT_TOKEN
check_var TELEGRAM_ALLOWED_USER_IDS
check_var API_TOKEN
check_var POSTGRES_PASSWORD

if [ ${#missing[@]} -gt 0 ]; then
  fail "these .env values are missing or still at their placeholder: ${missing[*]}
  API_TOKEN=\$(openssl rand -hex 32)
  POSTGRES_PASSWORD=\$(openssl rand -hex 16)   (also update DATABASE_URL)"
fi

mkdir -p data/{downloads,uploads,tasks,temp,output}

# The container runs as uid 10001 (non-root); a bind mount keeps the host's
# ownership, so without this the agent cannot write a single file into its own
# workspace - downloads, output and temp all fail with EACCES.
chown -R 10001:10001 data 2>/dev/null || \
  info "could not chown data/ (not root?) - the agent may fail to write files"

info "building and starting containers"
docker compose up -d --build

info "waiting for the agent to become healthy"
for i in $(seq 1 60); do
  if curl -fsS localhost:"${API_PORT:-8080}"/health/ready >/dev/null 2>&1; then
    info "agent is ready"
    break
  fi
  [ "$i" -eq 60 ] && { docker compose logs --tail 60 agent; fail "agent did not become ready"; }
  sleep 2
done

if [ "${1:-}" = "--pull" ]; then
  model="$(grep -E '^LLM_MODEL=' .env | cut -d= -f2- || echo 'qwen2.5-coder:7b-instruct-q4_K_M')"
  info "pulling model: $model (this takes a few minutes)"
  docker compose exec -T ollama ollama pull "$model"
fi

echo
curl -s localhost:"${API_PORT:-8080}"/health | python3 -m json.tool || true
echo
info "done. Message your bot on Telegram: /start"
warn "logs: docker compose logs -f agent"
