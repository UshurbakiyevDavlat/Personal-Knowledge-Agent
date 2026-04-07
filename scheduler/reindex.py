"""
Scheduler — периодическая переиндексация Notion.
Запускается как отдельный процесс, работает фоном.

Расписание:
  - Каждые 2 часа — инкрементальная переиндексация Notion
  - Только изменённые страницы (force=False)
"""
import logging
import sys

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("knowledge-agent-scheduler")

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

from indexer.notion_indexer import run_full_index


def periodic_reindex() -> None:
    """Инкрементальная переиндексация Notion."""
    logger.info("🔄 Starting periodic Notion reindex...")
    try:
        stats = run_full_index(force=False)
        logger.info(
            f"✅ Reindex done: "
            f"{stats['indexed']} indexed, {stats['skipped']} skipped, {stats['failed']} failed"
        )
    except Exception as e:
        logger.error(f"Reindex failed: {e}", exc_info=True)


if __name__ == "__main__":
    scheduler = BlockingScheduler(timezone="Asia/Almaty")

    # Каждые 2 часа
    scheduler.add_job(
        periodic_reindex,
        trigger=IntervalTrigger(hours=2),
        id="periodic_notion_reindex",
        name="Periodic Notion Reindex",
        misfire_grace_time=3600,  # если пропустили — запустить в течение часа
        replace_existing=True,
    )

    logger.info("Scheduler started. Notion reindex every 2 hours.")
    logger.info("Press Ctrl+C to stop.")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")
