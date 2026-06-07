"""Standalone worker entrypoint.

Run as: ``python -m apps.worker``.
"""
import asyncio
import signal
from typing import Any

from src.database import engine
from apps.api.dependencies import (
    check_service_health,
    get_scheduler,
    init_dependencies,
    init_vector_store_scheduler,
    rebuild_bm25,
)
from src.observability.logging import get_logger, setup_logging

setup_logging()
logger = get_logger()


async def main() -> None:
    init_vector_store_scheduler()
    scheduler = get_scheduler()
    scheduler.start()
    try:
        init_dependencies()
        await rebuild_bm25()
        await check_service_health()
    except Exception as e:
        logger.warning("dependency_init_failed", error=str(e))

    stop_event = asyncio.Event()

    def _on_signal(signum: int, _frame: Any) -> None:
        logger.info("worker_signal_received", signum=signum)
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _on_signal, sig, None)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler for SIGTERM
            signal.signal(sig, _on_signal)

    try:
        await stop_event.wait()
    finally:
        await scheduler.stop()
        await engine.dispose()
        logger.info("worker_shutdown_complete")


if __name__ == "__main__":
    asyncio.run(main())
