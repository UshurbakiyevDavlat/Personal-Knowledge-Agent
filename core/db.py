"""
Database connection pool и базовые операции.
Используем psycopg2 + pgvector.
"""
import logging
from contextlib import contextmanager
from typing import Generator

import psycopg2
import psycopg2.pool
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector

from config import config

logger = logging.getLogger(__name__)

# Пул соединений (минимум 2, максимум 10)
_pool: psycopg2.pool.ThreadedConnectionPool | None = None


def get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        _pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=2,
            maxconn=10,
            dsn=config.DATABASE_URL,
        )
        # Регистрируем pgvector тип для всех соединений
        with _pool.getconn() as conn:
            register_vector(conn)
            _pool.putconn(conn)
        logger.info("Database connection pool created")
    return _pool


@contextmanager
def get_conn() -> Generator:
    """Context manager для безопасного получения соединения из пула."""
    pool = get_pool()
    conn = pool.getconn()
    try:
        register_vector(conn)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


@contextmanager
def get_cursor(conn=None):
    """Context manager для курсора. Принимает существующее соединение или создаёт новое."""
    if conn is not None:
        yield conn.cursor(cursor_factory=RealDictCursor)
    else:
        with get_conn() as c:
            yield c.cursor(cursor_factory=RealDictCursor)


def health_check() -> bool:
    """Проверить доступность БД."""
    try:
        with get_conn() as conn:
            with get_cursor(conn) as cur:
                cur.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


def close_pool() -> None:
    """Закрыть пул соединений (при выключении приложения)."""
    global _pool
    if _pool:
        _pool.closeall()
        _pool = None
        logger.info("Database connection pool closed")
