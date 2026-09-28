#!/usr/bin/env python3
"""Compare the local models on the two prompts the agent actually runs:
router classification (JSON) and a short Banglish chat reply.

Run on the VPS host; talks to the ollama container over its published port.
"""
import json
import subprocess
import sys
import time

MODELS = [
    "qwen2.5:1.5b-instruct-q4_K_M",
    "qwen2.5:3b-instruct-q4_K_M",
    "qwen2.5-coder:7b-instruct-q4_K_M",
    "qwen2.5:7b-instruct-q4_K_M",
]

ROUTER_SYSTEM = (
    "You classify one message sent to a personal AI agent.\n"
    'Reply with ONE JSON object: {"intent": "CHAT"|"TASK"|"FOLLOW_UP"|"CONTROL",'
    ' "reason": "<5 words>"}\n'
    "The owner writes Banglish (Bengali in Latin script).\n"
    '"banao"/"pathao"/"koro" -> TASK. "ki obostha" -> CONTROL.\n'
    '"ar ekta"/"oitao" -> FOLLOW_UP. "kemon acho" -> CHAT.'
)

ROUTER_CASES = [
    ("amar jonno ekta report banao", "TASK"),
    ("ki obostha, koto dur hoyeche?", "CONTROL"),
    ("ar ekta oirokom banao", "FOLLOW_UP"),
    ("tumi ki ki korte paro?", "CHAT"),
]


def call(model, messages, json_mode):
    body = {
        "model": model,
        "stream": False,
        "keep_alive": "2h",
        "messages": messages,
        "options": {"temperature": 0.2, "num_ctx": 4096},
    }
    if json_mode:
        body["format"] = "json"
    started = time.time()
    proc = subprocess.run(
        [
            "docker", "run", "--rm", "-i",
            "--network", "container:personal-ai-agent-ollama-1",
            "curlimages/curl:latest",
            "-s", "-m", "180",
            "http://localhost:11434/api/chat",
            "-H", "Content-Type: application/json",
            "-d", "@-",
        ],
        input=json.dumps(body).encode(),
        capture_output=True,
    )
    elapsed = time.time() - started
    if proc.returncode != 0:
        return None, elapsed, proc.stderr.decode()[:200]
    try:
        data = json.loads(proc.stdout.decode())
    except Exception as exc:
        return None, elapsed, f"unparseable: {exc}"
    return (data.get("message") or {}).get("content", ""), elapsed, ""


for model in MODELS:
    print(f"\n=== {model} ===", flush=True)
    # Warm it up so the first case does not pay the load time.
    call(model, [{"role": "user", "content": "hi"}], False)

    correct = 0
    valid_json = 0
    times = []
    for text, want in ROUTER_CASES:
        content, elapsed, err = call(
            model,
            [
                {"role": "system", "content": ROUTER_SYSTEM},
                {"role": "user", "content": f"MESSAGE:\n{text}"},
            ],
            json_mode=True,
        )
        times.append(elapsed)
        if err or content is None:
            print(f"  {text[:34]:36} ERROR {err}")
            continue
        try:
            got = json.loads(content).get("intent", "")
            valid_json += 1
        except Exception:
            got = f"NOT-JSON:{content[:40]!r}"
        hit = got == want
        correct += hit
        print(f"  {text[:34]:36} -> {str(got):12} want {want:10} {elapsed:5.1f}s"
              f" {'OK' if hit else 'MISS'}")

    reply, elapsed, err = call(
        model,
        [
            {"role": "system", "content": "You are a helpful assistant. The owner writes Banglish; reply in Banglish, max 2 sentences."},
            {"role": "user", "content": "bot ta ki ekhon cholche? short kore bolo."},
        ],
        json_mode=False,
    )
    print(f"  router: {correct}/{len(ROUTER_CASES)} correct, "
          f"{valid_json}/{len(ROUTER_CASES)} valid JSON, "
          f"avg {sum(times)/len(times):.1f}s")
    print(f"  chat ({elapsed:.1f}s): {(reply or err)[:160]!r}")
