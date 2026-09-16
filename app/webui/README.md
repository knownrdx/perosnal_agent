# Web UI

This directory contains a single self-contained dashboard file, `index.html`, for operating the
personal AI agent from a browser: tasks, approvals, contacts, skills memory, semantic memory,
scheduled jobs, LLM provider switching, bridge status (WhatsApp/Teams/Telegram), and autonomy
settings. There is no build step, no bundler, and no external JS/CSS dependencies — everything is
inline in the one HTML file so it keeps working offline or behind Cloudflare without needing to
allow extra origins.

FastAPI is expected to serve this directory as static files mounted at the path `/ui` (e.g. via
`StaticFiles(directory="app/webui")` mounted at `/ui` in `app/api.py`). That mount is being wired
up separately; this directory only needs to contain the static assets themselves. Once mounted,
visiting `/ui` in a browser loads `index.html`, which talks to the existing JSON API under `/api/...`
using same-origin `fetch` calls with `credentials: 'include'` so the session cookie is sent
automatically.

Authentication is entirely cookie-based from the browser's perspective — the page never holds or
sends an API token. For login to work at all, the `WEB_UI_PASSWORD` environment variable must be
set on the server; if it is unset, `POST /api/auth/login` is expected to respond with HTTP 503, and
the dashboard will show a static "web UI login is not configured on the server" message instead of
the password form.
