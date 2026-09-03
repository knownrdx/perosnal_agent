# Private 24/7 Personal AI Agent

A small, private, self-hosted agent that runs on your own VPS and is controlled
through Telegram. It understands an instruction, plans, calls tools, runs the
work in the background, survives restarts, and messages you the result — while
you are asleep or offline.

Built to the spec in `personal_ai_agent_master_prompt.txt`.
Reliability > complexity. Security > autonomy. Correctness > speed.


## What it actually does

    You (Telegram)  ->  bot  ->  task in PostgreSQL  ->  background worker
                                                              |
                             LLM decides one step at a time   |
                                                              v
                        tools (files / shell / python / browser / telegram)
                                                              |
                        verified result + file  ->  sent back to your Telegram

Send: "Download https://example.com/report.pdf, check it, and send it to me."
You get a task id immediately, and a `✅ Task completed` message with the file
when it is genuinely done and verified.

**It plans, then checks its own work.** Anything non-trivial gets a short plan
before the first tool call, and before the agent is allowed to report success a
second pass compares its claim against the real execution trace. A claim the
trace does not support is sent back with what is still missing, once. Honest
failure ("I could not reach the server") passes; a fabricated success does not.

Long tasks do not forget how they started: past steps are summarised rather
than truncated, so step 20 still knows what step 1 downloaded.

**It asks less over time.** Routine, reversible, or explicitly-requested actions
run without interrupting you; irreversible ones still stop and ask. Every
decision you make is remembered, so after you approve the same kind of action
three times it stops asking - and one rejection revokes that trust immediately.
Credentials and task state are never auto-approved at any setting.

    /autonomy            see the level and what I have learned
    /autonomy high       act without confirmation
    /autonomy paranoid   confirm everything
    /forgetrule <sig>    make me ask about something again

It is a conversation, not a ticket queue. Every message is routed first:

    "hi" / "what can you do"        -> CHAT       answered inline, no job
    "download X and send it"        -> TASK       background job created
    "and also make it a PDF"        -> FOLLOW_UP  attaches to that same job
    "how is it going"               -> CONTROL    answered from stored state

Follow-ups never spawn a duplicate job: while the work is running they are fed
into it, and once it has finished they continue it as a linked child task that
carries the previous result. Cheap regex rules settle the obvious cases so
chatting stays fast; the model is only asked when a message is ambiguous, and
if the model is unreachable an ambiguous message is treated as a task so real
work is never silently dropped.

    /new       start a fresh thread (tasks and memory are untouched)
    /mode      auto (default) | chat (never start jobs) | task (always)
    /session   what this conversation is currently about


## Architecture

    app/
      config.py            all environment configuration (single source)
      logging_conf.py      structured JSON logs + secret redaction
      monitoring.py        health snapshot (db, llm, workers, cpu/ram/disk)
      main.py              process entrypoint, wiring, graceful shutdown
      api.py               FastAPI: /health + token-protected control API

      security/
        permissions.py     READ / WRITE / HIGH_RISK + approval gate
        paths.py           workspace sandbox (no traversal, ever)

      db/
        models.py          tasks, tool_calls, events, approvals, memory,
                           scheduled_jobs, operations, conversation
        repo.py            every state transition lives here
        base.py            async engine/session (PostgreSQL or SQLite)

      llm/
        base.py            LLMClient interface + robust JSON extraction
        manager.py         active provider, live switching, fallback chain
        anthropic_client.py  Claude (Messages API)
        openai_client.py     ChatGPT + any OpenAI-compatible endpoint
        ollama_client.py     local models (retry + model pull)
        echo_client.py       deterministic offline stub (tests / dry runs)
        prompts.py         system prompt + context rendering

      integrations/
        bridge_client.py   authenticated HTTP clients for the Go bridges
        inbound.py         inbound WhatsApp/Teams -> store, notify, auto-task

      tools/
        base.py            schema, permission, timeout, retry, verify hook
        registry.py        the only list of what the LLM may call
        file_tools.py      write/read/exists/list/copy/move/delete/download
        exec_tools.py      python_execute, safe_shell_execute (allowlisted)
        telegram_tools.py  send message/file, verified by message_id
        task_tools.py      task_create/status/cancel, scheduler_*
        memory_tools.py    memory_store/search (refuses secrets)
        messaging_tools.py whatsapp_* and teams_* tools
        browser_tools.py   Playwright, disabled by default

      agent/
        router.py          chat vs task vs follow-up vs control
        conversation.py    sessions, inline replies, follow-up attachment
        planner.py         up-front plan + completion self-check
        context.py         summarise old steps instead of dropping them
        executor.py        permission -> approval -> idempotency -> run -> audit
        engine.py          the step loop; all state in the database
        learning.py        reflection, failure rules, memory retrieval

      workers/
        task_worker.py     leased claim loop, crash recovery, heartbeat
        scheduler_worker.py  persistent once/interval/cron jobs

      telegram/
        bot.py             aiogram, allowlist, /status /tasks /approve ...
        notifier.py        lifecycle notifications, dedupe-guarded

      bridges/
        whatsapp/main.go   Go + whatsmeow: own WhatsApp session, QR pairing,
                           send text/file, receive + media download, webhook
        teams/main.go      Go + Microsoft Graph (client credentials):
                           send/read messages, download hosted files

    tests/                 243 tests, fully offline
    scripts/smoke.py       boots the real process and verifies it end to end


## Safety properties (enforced, and tested)

| Guarantee | Where | Test |
|---|---|---|
| No path escapes the workspace | `security/paths.py` | `test_security.py` |
| READ tasks cannot write | `agent/executor.py` | `test_engine.py` |
| HIGH_RISK needs your approval before it runs | `agent/executor.py` | `test_engine.py` |
| A rejected action never executes | `agent/executor.py` | `test_engine.py` |
| Shell is allowlisted, no `;` `|` `>` `` ` `` | `tools/exec_tools.py` | `test_tools.py` |
| Temporary failures retry, permanent ones don't | `tools/base.py` | `test_tools.py` |
| A send is only "sent" with a real message_id | `tools/telegram_tools.py` | `test_end_to_end.py` |
| Restart never re-sends the same file | `db.operations` ledger | `test_end_to_end.py` |
| Crashed tasks are reclaimed, not lost | lease + `recover_stale_running` | `test_tasks_scheduler.py` |
| Secrets never reach logs or memory | `logging_conf.py`, `memory_tools.py` | `test_security.py` |
| Only allowlisted Telegram IDs can control it | `telegram/bot.py` | `test_telegram.py` |
| API keys / sessions encrypted at rest | `security/vault.py` | `test_vault_setup.py` |
| Credential messages deleted from the chat | `telegram/setup_commands.py` | manual |
| The agent never learns a credential | `agent/learning.py` | `test_learning.py` |
| Bridge calls need the shared BRIDGE_TOKEN | `integrations/bridge_client.py` | `test_integrations.py` |
| Unknown WhatsApp/Teams senders cannot trigger tasks | `integrations/inbound.py` | `test_integrations.py` |
| Duplicate inbound messages are ignored | `inbound_messages` unique key | `test_integrations.py` |
| A dead LLM provider falls back instead of failing | `llm/manager.py` | `test_llm_providers.py` |
| Chatting never creates background jobs | `agent/router.py` | `test_conversation.py` |
| A follow-up never duplicates a job | `agent/conversation.py` | `test_conversation.py` |
| An unroutable message becomes a task, not silence | `agent/router.py` | `test_conversation.py` |
| A success claim the trace does not support is rejected | `agent/planner.py` | `test_planning.py` |
| A 20-step task still knows what step 1 did | `agent/context.py` | `test_planning.py` |
| Routine actions stop interrupting; secrets never do | `security/autonomy.py` | `test_autonomy.py` |
| One rejection revokes learned trust | `security/autonomy.py` | `test_autonomy.py` |
| A finished task always reaches the owner | `workers/task_worker.py` | `test_notification_reliability.py` |


## Install (Ubuntu 24.04 VPS, Docker)

    # 1. docker
    curl -fsSL https://get.docker.com | sh

    # 2. configure
    cp .env.example .env
    nano .env
    #   TELEGRAM_BOT_TOKEN=...            from @BotFather
    #   TELEGRAM_ALLOWED_USER_IDS=...     your numeric id, from @userinfobot
    #   API_TOKEN=$(openssl rand -hex 32)
    #   POSTGRES_PASSWORD=$(openssl rand -hex 16)
    #   DATABASE_URL must use that same password

    # 3. start
    docker compose up -d --build

    # 4. pull the model (once, ~5 GB, a few minutes)
    docker compose exec ollama ollama pull qwen2.5-coder:7b-instruct-q4_K_M

    # 5. verify
    curl -s localhost:8080/health | python3 -m json.tool
    docker compose logs -f agent

Then message your bot on Telegram: `/start`.

Enable browser automation only if you need it:

    ENABLE_BROWSER_TOOLS=true INSTALL_BROWSER=true docker compose up -d --build


## Run locally (no Docker, no GPU, no model)

    uv venv .venv --python 3.12
    uv pip install --python .venv/bin/python -r requirements.txt pytest pytest-asyncio

    # tests: 243 offline tests, ~25s
    .venv/bin/python -m pytest -q

    # smoke test: boots the real process, runs a real task through the worker
    .venv/bin/python scripts/smoke.py

On Windows use `.venv/Scripts/python.exe` instead of `.venv/bin/python`.

Expected output:

    243 passed
    10/10 checks passed
    SMOKE TEST OK - the agent boots, serves, and completes a real task.


## Telegram commands

    /start              is the agent alive
    /help               command list
    /autonomy           how much I do without asking
    /forgetrule <sig>   revoke one learned permission
    /new                fresh conversation thread
    /mode               auto | chat | task
    /session            what this chat is about
    /status             db, llm, workers, cpu/ram/disk, task counts
    /tasks              recent tasks
    /task <id>          full detail incl. every tool call
    /cancel <id>        stop a task
    /reply <id> <text>  answer an agent question
    /approve <id>       allow a HIGH_RISK action
    /reject <id>        deny it
    /jobs               scheduled jobs
    /memory <query>     search long-term memory

    AI model (switch live, no restart, choice survives reboot):
    /models             list providers + models, shows which is active
    /provider claude    switch to Claude
    /provider chatgpt   switch to ChatGPT
    /provider local     switch to the local Ollama model
    /model gpt-4o       change model on the active provider

    Accounts (connect everything from chat, nothing on the VPS):
    /connect            what is connected right now
    /setup              how to connect each thing
    /setkey claude sk-ant-...     save an API key (deleted from chat after)
    /addllm <name> <url> <model> [key]   add any OpenAI-compatible LLM

    Your Telegram account (lets me act as you + manage your bots):
    /tglogin <api_id> <api_hash> <phone>   from my.telegram.org
    /tgcode <code>      the code Telegram sends you
    /tg2fa <password>   only if you use 2FA
    /tgstatus  /tglogout
    /bots               your bots, via BotFather

    Messaging:
    /wa connect | status | logout | send <number> <text>
    /teams login | login <client_id> | token <access_token>
    /teams connect <tenant> <client> <secret> | status | send <chat> <text>
    /inbox              recent WhatsApp/Teams messages

    Memory (it learns by itself; these are just controls):
    /learned            what it has picked up
    /teach <key> <fact> tell it something directly
    /forget <key>       drop a memory

Anything that is not a command becomes a background task.


## LLM providers

**You do not need any API key to start.** The stack ships with OmniRoute, a
self-hosted MIT gateway that speaks the OpenAI protocol and fans out to many
providers (including free tiers). It is the default whenever you have no
personal key configured:

    LLM_PROVIDER=omniroute      ->  /provider omniroute   (no key needed)

Connecting another model is a two-message job - no .env editing:

    /llm                 list every provider (12 presets built in)
    /llm claude          get the exact link + steps
    /paste sk-ant-...    paste what you copied; it verifies and switches

Built-in presets (base URL and models already known):

    omniroute   free, no key, already running       ollama      local, private
    anthropic   Claude                              openai      ChatGPT
    openrouter  300+ models behind one key         groq        fastest, free tier
    mwapi       Claude opus/sonnet via gateway
    gemini      free tier                           deepseek    very cheap
    github      free with a GitHub token            mistral     free tier
    together    open models                         xai         Grok

Anything not listed still works via `/addllm <name> <base_url> <model> [key]`.

Manage gateway accounts on its dashboard (loopback only):

    ssh -L 20128:127.0.0.1:20128 root@your-vps    # then open localhost:20128

Anything speaking the OpenAI schema also works (OpenRouter, Groq, DeepSeek,
Together, xAI, vLLM, LM Studio, llama.cpp server):

    CUSTOM_LLM_NAME=openrouter
    CUSTOM_LLM_API_KEY=sk-or-...
    CUSTOM_LLM_BASE_URL=https://openrouter.ai/api/v1
    CUSTOM_LLM_MODEL=anthropic/claude-sonnet-4.5

With `LLM_FALLBACK_ENABLED=true` a failure on the active provider is retried
on the next configured one, so an API outage does not kill an overnight task.
The order prefers your paid keys, then the gateway, then the local model:

    anthropic -> openai -> omniroute -> ollama

The selection is persisted in the database and restored on restart.

Running cloud-only? Skip the 7 GB local model entirely:

    docker compose up -d --scale ollama=0


## It learns on its own

You should not have to explain the same thing twice, so after **every** task
the agent reflects on what happened and writes durable lessons to memory:

    task finishes  ->  reflection pass  ->  memory
                       (preferences, workflows, gotchas)

Three mechanisms, all automatic:

| What | How | Example it stores |
|---|---|---|
| Reflection | the model reads the finished task's tool trace | "owner wants reports as PDF" |
| Failure rules | no LLM: 3+ identical tool errors become a rule | "file_download keeps failing with DNS errors - check the URL" |
| Retrieval | memories are relevance-scored per request and injected | only invoice memories load for an invoice task |

Guards: credentials are never learned (the same secret filter as the memory
tool), lessons are capped at 3 per task, trivial one-step tasks teach nothing,
and a failed reflection can never break the task itself.

Inspect and correct it any time with `/learned`, `/teach`, `/forget`.


## Your Telegram account (userbot)

A bot token cannot read your chats or manage your other bots. Linking your own
account (Telethon) gives the agent tools to do your Telegram work:

    tg_send_message / tg_send_file    message anyone as you
    tg_read_messages / tg_list_chats  read your chats
    tg_search_messages                search your history
    tg_bot_admin                      drive BotFather (HIGH_RISK -> needs /approve)

Link it from chat:

    /tglogin 1234567 <api_hash> +8801712345678
    /tgcode 12345

The session string is stored **encrypted** in the credential vault, so a
restart never asks you to log in again. `/tglogout` revokes it.


## WhatsApp and Teams

Both are separate Go services (small, single-purpose, restartable) that the
Python agent calls over an authenticated local HTTP API.

WhatsApp - `bridges/whatsapp`, uses whatsmeow (the WhatsApp Web multi-device
protocol) and links your own account, exactly like WhatsApp Web:

    WHATSAPP_ENABLED=true
    BRIDGE_TOKEN=$(openssl rand -hex 32)

    docker compose up -d --build whatsapp-bridge
    # then on Telegram:
    /walogin      -> scan the QR with WhatsApp > Linked devices

The session is stored in the `wa_session` volume, so a restart does not need a
new scan. Incoming media is saved to `/data/downloads/whatsapp/`.

Teams - `bridges/teams`, official Microsoft Graph API (no browser automation).
Two ways to connect:

**Teams needs a work or school Microsoft account.** Personal accounts
(outlook.com, hotmail, live.com) cannot use the Teams API at all - Microsoft
does not expose Teams resources to consumer accounts, so no client id or
token will help. Sign-in fails with *"You can't sign in here with a personal
account"* / `AADSTS50020`. The bridge detects this and says so plainly instead
of echoing the raw Azure error.

**1. Device-code sign-in (recommended).** Nothing to set up in Azure:

    /teams login
    -> "Open microsoft.com/devicelogin and enter code H7QW2XK9"
    -> sign in with your normal Microsoft account (MFA works fine)
    -> the bot tells you when it is connected

The agent then acts *as you* (delegated permissions), so it can read and send
your Teams messages without admin consent. A refresh token is stored, so you
sign in once. `/teams status` shows the account; `/teams disconnect` revokes it.

It uses the Microsoft Graph CLI public client
(`14d82eec-204b-4c2f-b7e8-296a70dab67e`), which is preauthorized for Graph.
The Azure CLI client id is **not**, and fails with `AADSTS65002`.

If your tenant blocks that client, register your own app (5 minutes) and pass
its id - the flow is otherwise identical:

    portal.azure.com -> App registrations -> New
      Accounts: "Any organizational directory"
      Authentication -> Add platform -> Mobile and desktop
      Allow public client flows: Yes

    /teams login <your_client_id>

Arguments are order-independent: a GUID is read as a client id, anything with a
dot as a tenant domain, and `common` / `work` / `personal` as tenant shortcuts.

    /teams login contoso.onmicrosoft.com      pin the tenant
    /teams login <client_id> common           both

**2. Paste a Graph access token** when you just need it working now:

    /teams token <access_token>

Get one from https://developer.microsoft.com/graph/graph-explorer (sign in,
"Access token" tab). The bridge verifies it against `/me` before accepting it.
These expire in about an hour and have no refresh, so it is a stopgap.

A note on browser cookies: a `claude.ai`/Teams web session cookie is not a
Graph API token, so it cannot be used to call Graph. Scraping one would also
break the terms of service and expire within the hour. The three flows above
are the supported paths.

**App-only (advanced),** if you want the agent to act as its own identity:

    /teams connect <tenant_id> <client_id> <client_secret> [default_chat]

That needs an Azure app registration with application permissions
(`Chat.ReadWrite.All`, `ChannelMessage.Send`, `ChannelMessage.Read.All`) and
admin consent. Note Microsoft treats the Teams message APIs as protected in
app-only mode, so production use requires their approval - device-code sign-in
avoids that entirely.

Incoming messages: by default they are stored and forwarded to you on
Telegram. To let the agent act on them autonomously:

    INBOUND_AUTO_TASK=true
    INBOUND_ALLOWED_SENDERS=8801712345678,karim@corp.com

An empty allowlist means nobody can trigger a task - a stranger messaging your
WhatsApp can never make the agent run anything.


## HTTP API (localhost only)

    curl -H "X-API-Token: $API_TOKEN" localhost:8080/api/tasks
    curl -H "X-API-Token: $API_TOKEN" -H 'Content-Type: application/json' \
         -d '{"instruction":"check disk usage and report"}' \
         localhost:8080/api/tasks
    curl -H "X-API-Token: $API_TOKEN" localhost:8080/api/tasks/<id>
    curl -H "X-API-Token: $API_TOKEN" localhost:8080/api/tools

`/health`, `/health/live`, `/health/ready` need no token.


## Resource usage on the target VPS (8 vCPU / 32 GB, CPU only)

    ollama + qwen2.5-coder 7B q4_K_M   ~6-7 GB RAM while generating
    postgres                           ~200-300 MB
    agent (python)                     ~250-400 MB
    ---------------------------------------------------------------
    total                              ~8 GB of 32 GB

7B q4 on 8 CPU cores gives roughly 5-12 tokens/sec — fine for an agent that
makes a handful of decisions per task. If you want more headroom later:
`qwen2.5-coder:14b-instruct-q4_K_M` (~10 GB). Do not run 32B on CPU.


## Model choice

`qwen2.5-coder:7b-instruct-q4_K_M` — strong instruction following and JSON
output for its size, good at code/tooling, comfortable on CPU with 32 GB.
The agent only needs it to emit one small JSON decision per step, which small
models handle reliably.

To switch models, change `LLM_MODEL` and restart. To switch *engines*, add one
file under `app/llm/` implementing `LLMClient` and register it in
`app/llm/__init__.py` — nothing else in the codebase changes.


## Operations

    docker compose logs -f agent          structured JSON logs
    docker compose restart agent          safe: tasks resume from the database
    docker compose down                   stop
    docker compose exec postgres pg_dump -U agent agent > backup.sql

Backup `./data` (files) and the `pgdata` volume (state).

After a restart the worker requeues anything left `RUNNING`, replays the tool
history into the LLM context, and the idempotency ledger stops repeated side
effects. That combination is what makes "it kept working while I slept" true
rather than aspirational.


## Deliberately NOT in V1

Multi-user SaaS, Kubernetes, multi-server, fine-tuning, vector database, big
RAG, multiple LLM providers, unrestricted shell/browser, CAPTCHA or anti-bot
bypass, financial automation, web dashboard. Interfaces are clean enough to add
what you actually need later — Teams and WhatsApp slot in as their own modules
under `app/tools/` using their official APIs.
