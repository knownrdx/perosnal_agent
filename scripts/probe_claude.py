"""Prove the agent's own LLM stack can talk to Claude.

Uses the real AnthropicClient the agent runs on - not a bespoke HTTP call - so a
pass here means the agent itself can use Claude.

Usage:
    python scripts/probe_claude.py <base_url> <api_key> [model]
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.llm import Message
from app.llm.anthropic_client import AnthropicClient


async def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    base_url, api_key = sys.argv[1], sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else "claude-opus-4-6"

    client = AnthropicClient(api_key=api_key, base_url=base_url, model=model)

    print(f"provider : {client.name}")
    print(f"base_url : {client.base_url}")
    print(f"model    : {client.model}")
    print(f"key      : {api_key[:8]}...{api_key[-4:]}\n")

    print("--- health() ---")
    health = await client.health()
    print(f"  {health}\n")

    print("--- chat() ---")
    response = await client.chat([Message("user", "Reply with exactly: agent-connected")])
    reply = response.content
    print(f"  reply: {reply.strip()[:100]!r}")
    print(f"  model: {response.model}\n")

    print("--- chat_json() (the agent loop's decision format) ---")
    decision = await client.chat_json(
        [
            Message(
                "system",
                'Respond with ONE JSON object: {"action": "final", '
                '"final_answer": "<text>"}. No prose, no code fences.',
            ),
            Message("user", "Say that the agent's Claude connection works."),
        ]
    )
    print(f"  action      : {decision.get('action')}")
    print(f"  final_answer: {str(decision.get('final_answer'))[:120]!r}\n")

    ok = bool(reply.strip()) and decision.get("action") == "final"
    print("CLAUDE OK" if ok else "CLAUDE FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
