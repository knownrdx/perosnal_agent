"""Entrypoint: wires database, LLM, tools, workers, Telegram and the HTTP API
into one supervised process.

    python -m app.main
"""

from __future__ import annotations

import asyncio
import contextlib
import signal

import uvicorn

from app.agent.engine import AgentEngine
from app.api import create_app
from app.config import get_settings
from app.db.base import create_all, dispose_engine, init_engine
from app.llm import close_llm, get_llm
from app.logging_conf import get_logger, setup_logging
from app.monitoring import RUNTIME
from app.telegram.notifier import Notifier
from app.tools import registry  # noqa: F401 - importing registers all tools
from app.workers import SchedulerRunner, TaskWorker

log = get_logger(__name__)


class Application:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.notifier = Notifier()
        self.worker: TaskWorker | None = None
        self.scheduler: SchedulerRunner | None = None
        self.bot = None
        self.server: uvicorn.Server | None = None
        self._stop = asyncio.Event()

    async def _restore_runtime_settings(self) -> None:
        """Re-apply choices the owner made from Telegram.

        These live in the database rather than the environment so they can be
        changed while running; without this they would silently revert to the
        .env defaults on every restart.
        """
        from app.db import repo
        from app.db.base import session_scope

        async with session_scope() as session:
            level = await repo.get_setting(session, "autonomy_level")
            briefing = await repo.get_setting(session, "briefing_enabled")
            cron = await repo.get_setting(session, "briefing_cron")

        if level in {"balanced", "high", "paranoid"}:
            self.settings.autonomy_level = level
            log.info("autonomy_restored", extra={"level": level})
        if briefing in {"true", "false"}:
            self.settings.briefing_enabled = briefing == "true"
        if cron:
            self.settings.briefing_cron = cron

    def _check_workspace_writable(self) -> None:
        """Fail loudly at boot if the workspace is not writable.

        A bind-mounted volume keeps the host's ownership, so a container running
        as a non-root user can end up unable to write anything. Every download,
        report and temp file then fails at the moment of use, which looks like a
        mysterious tool bug hours later. Better to say so on the first line of
        the log.
        """
        probe = self.settings.workspace / ".write_probe"
        try:
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            log.error(
                "workspace_not_writable",
                extra={
                    "workspace": str(self.settings.workspace),
                    "error": str(exc)[:200],
                    "fix": "chown -R 10001:10001 data (on the host)",
                },
            )

    async def startup(self) -> None:
        settings = self.settings
        settings.ensure_workspace()
        self._check_workspace_writable()
        init_engine()
        await create_all()
        await self._restore_runtime_settings()
        log.info(
            "startup",
            extra={
                "env": settings.app_env,
                "llm": f"{settings.llm_provider}:{settings.llm_model}",
                "tools": len(registry.names()),
                "workspace": str(settings.workspace),
                "whatsapp": settings.whatsapp_enabled,
                "teams": settings.teams_enabled,
            },
        )

        # Decrypt runtime credentials, then restore the LLM selection.
        from app.llm import get_manager
        from app.security.vault import get_vault

        await get_vault().load()
        await get_manager().load()

        # Best-effort model warm-up; the agent still starts if Ollama is cold.
        if get_manager().active_key() == "ollama":
            client = get_manager().client("ollama")
            info = await client.health()
            if info.get("ok") and not info.get("model_available"):
                log.warning("llm_model_missing", extra={"model": settings.llm_model})
                asyncio.create_task(client.ensure_model())
            elif not info.get("ok"):
                log.warning("llm_unreachable", extra={"error": str(info.get("error"))[:200]})

        engine = AgentEngine(notifier=self.notifier)

        if settings.start_worker:
            self.worker = TaskWorker(engine=engine, notifier=self.notifier)
            await self.worker.start()
            RUNTIME["worker"] = self.worker

        if settings.start_scheduler:
            self.scheduler = SchedulerRunner(notifier=self.notifier)
            await self.scheduler.start()
            RUNTIME["scheduler"] = self.scheduler

        if settings.start_telegram_bot and settings.telegram_enabled:
            from app.telegram.bot import AgentBot

            self.bot = AgentBot(notifier=self.notifier)
            await self.bot.start()
            RUNTIME["bot"] = self.bot
        else:
            log.warning("telegram_not_started",
                        extra={"configured": settings.telegram_enabled})

    async def shutdown(self) -> None:
        log.info("shutdown_start")
        if self.bot is not None:
            await self.bot.stop()
        if self.scheduler is not None:
            await self.scheduler.stop()
        if self.worker is not None:
            await self.worker.stop()
        with contextlib.suppress(Exception):
            from app.tools.browser_tools import shutdown_browser

            await shutdown_browser()
        with contextlib.suppress(Exception):
            from app.tools.telegram_tools import close_api

            await close_api()
        with contextlib.suppress(Exception):
            from app.integrations import close_bridges, close_userbot

            await close_bridges()
            await close_userbot()
        await close_llm()
        await dispose_engine()
        log.info("shutdown_complete")

    async def serve(self) -> None:
        settings = self.settings
        config = uvicorn.Config(
            create_app(),
            host=settings.api_host,
            port=settings.api_port,
            log_config=None,
            access_log=False,
            lifespan="off",
        )
        self.server = uvicorn.Server(config)
        await self.server.serve()

    async def run(self) -> None:
        await self.startup()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError, AttributeError):
                loop.add_signal_handler(sig, self._stop.set)
        try:
            server_task = asyncio.create_task(self.serve())
            stop_task = asyncio.create_task(self._stop.wait())
            done, pending = await asyncio.wait(
                {server_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if stop_task in done and self.server is not None:
                self.server.should_exit = True
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await server_task
            for task in pending:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        finally:
            await self.shutdown()


def main() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    application = Application()
    try:
        asyncio.run(application.run())
    except KeyboardInterrupt:  # pragma: no cover
        log.info("interrupted")


if __name__ == "__main__":
    main()
