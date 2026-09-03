#!/usr/bin/env bash
# Generate a production .env on the VPS.
#   ./scripts/make_env.sh <TELEGRAM_BOT_TOKEN> <TELEGRAM_USER_ID>
# Existing secrets are preserved if .env already exists.
set -euo pipefail

cd "$(dirname "$0")/.."

TOKEN="${1:-}"
USER_ID="${2:-}"

if [ -z "$TOKEN" ] || [ -z "$USER_ID" ]; then
  echo "usage: $0 <telegram_bot_token> <telegram_user_id>" >&2
  exit 1
fi

keep() {  # keep an existing value if .env already has one
  local key="$1" fallback="$2"
  if [ -f .env ]; then
    local existing
    existing="$(grep -E "^${key}=" .env | head -1 | cut -d= -f2- || true)"
    if [ -n "$existing" ]; then echo "$existing"; return; fi
  fi
  echo "$fallback"
}

rand() { openssl rand -hex "${1:-32}"; }

API_TOKEN="$(keep API_TOKEN "$(rand 32)")"
BRIDGE_TOKEN="$(keep BRIDGE_TOKEN "$(rand 32)")"
PG_PASS="$(keep POSTGRES_PASSWORD "$(rand 16)")"

cat > .env <<EOF
APP_ENV=prod
LOG_LEVEL=INFO

API_HOST=0.0.0.0
API_PORT=8080
API_TOKEN=${API_TOKEN}

POSTGRES_USER=agent
POSTGRES_PASSWORD=${PG_PASS}
POSTGRES_DB=agent
DATABASE_URL=postgresql+asyncpg://agent:${PG_PASS}@postgres:5432/agent

TELEGRAM_BOT_TOKEN=${TOKEN}
TELEGRAM_ALLOWED_USER_IDS=${USER_ID}
TELEGRAM_OWNER_CHAT_ID=${USER_ID}

# --- LLM: OmniRoute gateway is the default (needs no API key) --------------
LLM_PROVIDER=omniroute
LLM_TIMEOUT_S=300
LLM_NUM_CTX=8192
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=4096
LLM_FALLBACK_ENABLED=true

OMNIROUTE_ENABLED=true
OMNIROUTE_BASE_URL=http://omniroute:20128/v1
OMNIROUTE_API_KEY=
OMNIROUTE_MODEL=auto
OMNIROUTE_PORT=20128
OMNIROUTE_MEMORY_MB=2048

OLLAMA_BASE_URL=http://ollama:11434
LLM_MODEL=qwen2.5-coder:7b-instruct-q4_K_M

# Add your own keys any time, then: /provider claude   or   /provider chatgpt
OPENAI_API_KEY=$(keep OPENAI_API_KEY "")
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini

ANTHROPIC_API_KEY=$(keep ANTHROPIC_API_KEY "")
ANTHROPIC_BASE_URL=https://api.anthropic.com
ANTHROPIC_MODEL=claude-sonnet-4-5

CUSTOM_LLM_NAME=
CUSTOM_LLM_API_KEY=
CUSTOM_LLM_BASE_URL=
CUSTOM_LLM_MODEL=

# --- Messaging bridges ------------------------------------------------------
BRIDGE_TOKEN=${BRIDGE_TOKEN}
BRIDGE_TIMEOUT_S=120

WHATSAPP_ENABLED=$(keep WHATSAPP_ENABLED "true")
WHATSAPP_BRIDGE_URL=http://whatsapp-bridge:8081
WA_DOWNLOAD_MEDIA=true

TEAMS_ENABLED=$(keep TEAMS_ENABLED "true")
TEAMS_BRIDGE_URL=http://teams-bridge:8082
TEAMS_TENANT_ID=$(keep TEAMS_TENANT_ID "")
TEAMS_CLIENT_ID=$(keep TEAMS_CLIENT_ID "")
TEAMS_CLIENT_SECRET=$(keep TEAMS_CLIENT_SECRET "")
TEAMS_DEFAULT_CHAT=$(keep TEAMS_DEFAULT_CHAT "")

INBOUND_AUTO_TASK=false
INBOUND_ALLOWED_SENDERS=

# --- Workspace / engine -----------------------------------------------------
WORKSPACE_DIR=/data
MAX_FILE_MB=45

MAX_TASK_STEPS=14
TASK_TIMEOUT_S=3600
WORKER_CONCURRENCY=2
WORKER_POLL_INTERVAL_S=2
SCHEDULER_POLL_INTERVAL_S=10
MAX_TASK_RETRIES=2

# --- Safety -----------------------------------------------------------------
REQUIRE_APPROVAL_HIGH_RISK=true
SHELL_ALLOWLIST=ls,pwd,cat,head,tail,wc,grep,find,df,du,file,sha256sum,stat,python,python3,git
ENABLE_BROWSER_TOOLS=false
ENABLE_PYTHON_TOOL=true
ENABLE_SHELL_TOOL=true
EOF

chmod 600 .env
mkdir -p data/{downloads,uploads,tasks,temp,output}

echo "wrote .env"
echo "  telegram user : ${USER_ID}"
echo "  api token     : ${API_TOKEN}"
echo "  bridge token  : ${BRIDGE_TOKEN}"
