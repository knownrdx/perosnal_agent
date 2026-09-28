"""Live verification of this round's changes. Run inside the agent container."""
import asyncio
import time

from app.agent.router import classify
from app.automation import otp_bot, otp_schedule, phone_countries
from app.automation.otp_caption import parse_caption
from app.llm import get_llm
from app.llm.base import Message


async def main() -> None:
    print("=== 1. country detection (245 regions) ===")
    print("   library:", phone_countries._HAS_PN,
          "| countries:", len(phone_countries.known_country_names()))
    for number, want in (
        ("+8801712345678", "Bangladesh"),
        ("+37061234567", "Lithuania"),      # was "+370" before
        ("+18091234567", "Dominican Republic"),
        ("+14165551234", "Canada"),
        ("+77012345678", "Kazakhstan"),     # +7 shared with Russia
    ):
        got = phone_countries.country_of(number)
        print(f"   {'OK ' if got == want else 'BAD'} {number} -> {got}")

    print("\n=== 2. caption answers the questions ===")
    for caption in ("bangladesh whatsapp 20h", "shuru 21:00 bondho 06:00",
                    "nigeria telegram kono stop nai", "just some numbers"):
        print(f"   {caption!r:34} -> {parse_caption(caption)}")

    print("\n=== 3. router settles Banglish without the model ===")
    for text, active in (("amar jonno ekta report banao", None),
                         ("ki obostha, koto dur?", None),
                         ("ar ekta banao", "task-1"),
                         ("kemon acho", None)):
        started = time.time()
        decision = await classify(text, active_task_id=active)
        print(f"   {text[:30]:32} -> {decision.intent.value:10}"
              f" {time.time() - started:.3f}s  ({decision.reason})")

    print("\n=== 4. run-length + schedule config ===")
    cfg = await otp_bot.get_config()
    print("   default run:", otp_bot.format_run_minutes(cfg["default_run_minutes"]))
    print("   start_at present:", "start_at" in cfg, "| ask_run_time:", cfg["ask_run_time"])
    print("   parse '20h'/'2 din'/'limit nai':",
          [otp_bot.parse_run_minutes(t) for t in ("20h", "2 din", "limit nai")])
    print("   clock:", otp_schedule.clock_now()["dubai_full"])

    print("\n=== 5. LLM: JSON is constrained, dead provider backs off ===")
    manager = get_llm()
    print("   active:", manager.active_key())
    for index in range(3):
        started = time.time()
        try:
            data = await manager.chat_json([
                Message("system", 'Reply ONE JSON object: {"intent":"CHAT","reason":"x"}'),
                Message("user", "hello"),
            ])
            print(f"   call {index + 1}: {time.time() - started:5.1f}s -> {data}")
        except Exception as exc:  # noqa: BLE001
            print(f"   call {index + 1}: {time.time() - started:5.1f}s -> FAILED {str(exc)[:70]}")
    now = time.monotonic()
    print("   cooldowns:", {k: round(v - now) for k, v in manager._cooldowns.items()})
    print("   strikes:  ", dict(manager._failures))


asyncio.run(main())


def check_languages() -> None:
    """The owner's own language reaches the deterministic paths."""
    import asyncio

    from app.agent import language
    from app.agent.router import classify
    from app.automation import otp_bot
    from app.automation.otp_caption import parse_caption

    print("\n--- language ---")
    for text in ("বট টা কি চলছে", "bot ta ki cholche", "قم بإنشاء ملف",
                 "是否在运行", "is it running"):
        print(f"  detect {text[:22]:24} -> {language.detect(text)}")

    print("  triggers:",
          "stop=", otp_bot.is_stop_trigger("বন্ধ করো"),
          "start=", otp_bot.is_start_trigger("শুরু করো"),
          "status=", otp_bot.is_status_trigger("স্ট্যাটাস"))

    print("  caption:", parse_caption("বাংলাদেশ হোয়াটসঅ্যাপ ২০ ঘন্টা"))
    print("  caption:", parse_caption("সকাল ৬টা থেকে রাত ১১টা পর্যন্ত"))

    async def routes() -> None:
        for text, want in (("বট টা কি এখন চলছে?", "control"),
                           ("একটা রিপোর্ট বানাও", "task"),
                           ("কেমন আছো", "chat")):
            started = time.time()
            decision = await classify(text)
            mark = "OK  " if decision.intent.value == want else "DIFF"
            print(f"  {mark} {text[:20]:22} -> {decision.intent.value:9}"
                  f" {time.time() - started:5.3f}s ({decision.reason})")

    asyncio.run(routes())


check_languages()
