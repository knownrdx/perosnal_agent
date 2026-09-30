"""What the live box actually has, before and after this change.

Read-only: prints the stored config, the per-country settings and the run
state so the effect of the new defaults can be seen rather than assumed.
"""
import asyncio
import json


async def main() -> None:
    from app.automation import otp_bot, otp_schedule

    cfg = await otp_bot.get_config()
    print("stored start_at:", repr(cfg.get("start_at")))
    print("default_run_minutes:", cfg.get("default_run_minutes"))
    print("enabled:", cfg.get("enabled"))

    per_country = await otp_schedule.get_all_country_settings()
    print("\nper-country settings:")
    for key, value in per_country.items():
        keep = {k: v for k, v in value.items() if k != "display_name"}
        print(f"  {value.get('display_name', key)}: {json.dumps(keep)}")

    state = await otp_schedule._get_run_state()
    print("\nrun state:")
    for key, value in state.items():
        print(f"  {value.get('display_name', key)}: gate={value.get('gate')!r}"
              f" armed={value.get('armed_at')} refills={value.get('refills')}")

    active = await otp_bot.get_active_files()
    print("\nactive files:")
    for entry in active:
        print(f"  {entry.get('country')}: tag={entry.get('tag')}"
              f" waiting={entry.get('waiting_start')!r} count={entry.get('count')}")

    print("\nhas_started per active country:")
    for entry in active:
        country = entry.get("country") or entry.get("name")
        started = await otp_schedule.has_started(country, cfg)
        print(f"  {country}: {started}")


asyncio.run(main())
