"""
Конфигурация Knowledge Agent.
Все параметры берутся из .env файла.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Database
    DATABASE_URL: str = os.environ["DATABASE_URL"]

    # Voyage AI (embeddings)
    VOYAGE_API_KEY: str = os.environ["VOYAGE_API_KEY"]
    EMBEDDING_MODEL: str = "voyage-3"
    EMBEDDING_DIM: int = 1024   # voyage-3 → 1024 dimensions
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))

    # Notion
    NOTION_API_KEY: str = os.environ["NOTION_API_KEY"]
    NOTION_ROOT_PAGE_IDS: list[str] = [
        p.strip()
        for p in os.getenv("NOTION_ROOT_PAGE_IDS", "").split(",")
        if p.strip()
    ]

    # Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # Search
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "5"))


config = Config()
