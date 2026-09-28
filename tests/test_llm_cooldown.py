"""A failed provider must not be re-tried on every single message.

The symptom this prevents: with a rate-limited gateway as the active
provider, every message paid its ~16s failure before the working provider
was even attempted, so the bot felt slow no matter how fast the fallback was.
"""

from __future__ import annotations

import time

import pytest

from app.llm.base import LLMError, LLMResponse, Message
from app.llm.manager import LLMManager

asyncio_test = pytest.mark.asyncio


class _Boom:
    """A provider that always fails, counting how often it was asked."""

    def __init__(self, error: str = "boom") -> None:
        self.calls = 0
        self.error = error

    async def chat(self, messages, *, temperature=None):
        self.calls += 1
        raise LLMError(self.error)


class _Fine:
    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, messages, *, temperature=None):
        self.calls += 1
        return LLMResponse(content="ok", model="fine")


def _manager(order: list[str], clients: dict[str, object]) -> LLMManager:
    manager = LLMManager()
    manager.fallback_order = lambda: list(order)  # type: ignore[assignment]
    manager.client = lambda key=None: clients[key or order[0]]  # type: ignore[assignment]
    return manager


MSGS = [Message("user", "hi")]


@asyncio_test
async def test_a_dead_provider_is_skipped_on_the_next_message():
    bad, good = _Boom(), _Fine()
    manager = _manager(["bad", "good"], {"bad": bad, "good": good})

    assert (await manager.chat(MSGS)).content == "ok"
    assert bad.calls == 1

    # Second message: the dead provider must not be tried again.
    assert (await manager.chat(MSGS)).content == "ok"
    assert bad.calls == 1, "dead provider was retried during its cooldown"
    assert good.calls == 2


@asyncio_test
async def test_a_rate_limit_backs_off_longer_than_a_normal_error():
    """429s state their own window, so they deserve a longer pause."""
    rate_limited = _Boom("[429]: Rate limit exceeded")
    plain = _Boom("connection reset")
    good = _Fine()

    m1 = _manager(["rl", "good"], {"rl": rate_limited, "good": good})
    await m1.chat(MSGS)
    rl_cooldown = m1._cooldowns["rl"]

    m2 = _manager(["plain", "good"], {"plain": plain, "good": good})
    await m2.chat(MSGS)
    plain_cooldown = m2._cooldowns["plain"]

    assert rl_cooldown > plain_cooldown


@asyncio_test
async def test_a_recovered_provider_is_used_again():
    bad, good = _Boom(), _Fine()
    manager = _manager(["bad", "good"], {"bad": bad, "good": good})
    await manager.chat(MSGS)
    assert "bad" in manager._cooldowns

    # Time passes.
    manager._cooldowns["bad"] = 0.0
    assert manager._in_cooldown("bad") is False


@asyncio_test
async def test_success_clears_an_earlier_cooldown():
    good = _Fine()
    manager = _manager(["good"], {"good": good})
    manager._cooldowns["good"] = 0.0  # already expired

    await manager.chat(MSGS)
    assert "good" not in manager._cooldowns


@asyncio_test
async def test_everything_cooling_down_still_tries_rather_than_refusing():
    """Cooldowns are a latency optimisation, not a ban: if every provider is
    cooling down the request must still be attempted, not rejected outright.
    """
    good = _Fine()
    manager = _manager(["a", "good"], {"a": _Boom(), "good": good})
    import time as _time

    manager._cooldowns["a"] = _time.monotonic() + 999
    manager._cooldowns["good"] = _time.monotonic() + 999

    assert (await manager.chat(MSGS)).content == "ok"
    assert good.calls == 1


@asyncio_test
async def test_all_providers_failing_still_raises():
    manager = _manager(["a", "b"], {"a": _Boom("one"), "b": _Boom("two")})
    with pytest.raises(LLMError) as exc:
        await manager.chat(MSGS)
    assert "one" in str(exc.value) and "two" in str(exc.value)


def test_health_summary_reports_what_is_usable():
    manager = _manager(["a", "b"], {})
    import time as _time

    manager._cooldowns["a"] = _time.monotonic() + 60
    rows = {r["provider"]: r for r in manager.provider_health_summary()}

    assert rows["a"]["available"] is False
    assert rows["a"]["cooldown_s"] > 0
    assert rows["b"]["available"] is True


async def test_a_bad_key_is_not_retried_every_two_minutes():
    """An auth failure cannot fix itself, unlike a rate limit or an outage.

    A dead OmniRoute key cost 83s on the first message of every cooldown
    window; backing off for an hour removes that entirely without making it
    permanent - the owner can paste a new key and /provider clears it.
    """
    from app.llm.manager import LLMManager

    manager = LLMManager()
    manager._mark_failed("omniroute", Exception("omniroute: authentication failed (401)"))
    auth_until = manager._cooldowns["omniroute"]

    manager._mark_failed("openai", Exception("rate limited (retry after 30s)"))
    rate_until = manager._cooldowns["openai"]

    manager._mark_failed("ollama", Exception("connection refused"))
    plain_until = manager._cooldowns["ollama"]

    assert auth_until > rate_until > plain_until
    # And a working key still clears it immediately.
    manager._clear_cooldown("omniroute")
    assert manager._in_cooldown("omniroute") is False


async def test_a_provider_that_keeps_failing_is_probed_less_often():
    """A provider failing again after its cooldown is down, not blipping.

    Measured live: a dead gateway costs ~30s per attempt (its client retries
    three times) before the working provider is tried, and that was paid once
    per cooldown window forever. Escalating turns it into a few probes a day.
    """
    from app.llm.manager import LLMManager

    manager = LLMManager()
    waits = []
    for _ in range(4):
        manager._mark_failed("omniroute", Exception("omniroute: server error 502"))
        waits.append(manager._cooldowns["omniroute"] - time.monotonic())

    assert waits[0] < waits[1] < waits[2] < waits[3]
    assert waits[-1] <= LLMManager._MAX_COOLDOWN_S + 1

    # One success and it is fully trusted again - the next failure starts
    # from the short cooldown, not from the escalated one.
    manager._clear_cooldown("omniroute")
    manager._mark_failed("omniroute", Exception("omniroute: server error 502"))
    assert (manager._cooldowns["omniroute"] - time.monotonic()) == pytest.approx(
        waits[0], rel=0.1
    )
