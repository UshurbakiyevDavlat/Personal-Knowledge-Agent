"""
Scheduler — ночная переиндексация Notion.
Запускается как отдельный процесс, работает фоном.

Расписание:
  - Каждую ночь в 03:00 (по Almaty UTC+5) — инкрементальная переиндексация Notion
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
from apscheduler.triggers.cron import CronTrigger

from indexer.notion_indexer import run_full_index


def nightly_reindex() -> None:
    """Инкрементальная переиндексация Notion."""
    logger.info("🔄 Starting nightly Notion reindex...")
    try:
        stats = run_full_index(force=False)
        logger.info(
            f"✅ Nightly reindex done: "
            f"{stats['indexed']} indexed, {stats['skipped']} skipped, {stats['failed']} failed"
        )
    except Exception as e:
        logger.error(f"Nightly reindex failed: {e}", exc_info=True)


if __name__ == "__main__":
    scheduler = BlockingScheduler(timezone="Asia/Almaty")

    # Каждую ночь в 03:00 по Almaty
    scheduler.add_job(
        nightly_reindex,
        trigger=CronTrigger(hour=3, minute=0, timezone="Asia/Almaty"),
        id="nightly_notion_reindex",
        name="Nightly Notion Reindex",
        misfire_grace_time=3600,  # если пропустили — запустить в течение часа
        replace_existing=True,
    )

    logger.info("Scheduler started. Notion reindex at 03:00 Almaty time.")
    logger.info("Press Ctrl+C to stop.")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")
