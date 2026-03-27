import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from token0.api.v1.chat import router as chat_router
from token0.api.v1.usage import router as usage_router
from token0.config import settings
from token0.storage.postgres import close_db, init_db
from token0.storage.redis import close_redis, init_redis

logger = logging.getLogger("token0")


@asynccontextmanager
async def lifespan(app: FastAPI):
    mode = settings.storage_mode
    logger.info(f"Starting Token0 in {mode} mode")
    if settings.is_lite:
        logger.info(f"  Database: SQLite ({settings.sqlite_path})")
        logger.info("  Cache: in-memory")
        logger.info("  Storage: local filesystem")
        logger.info("  Tip: Set STORAGE_MODE=full for production (Postgres + Redis + S3)")
    else:
        logger.info(f"  Database: {settings.database_url}")
        logger.info(f"  Cache: {settings.redis_url}")
        logger.info(f"  Storage: {settings.s3_endpoint}")
    await init_db()
    await init_redis()
    yield
    await close_db()
    await close_redis()


app = FastAPI(
    title="Token0",
    description="Open-source API proxy that makes vision LLM calls 5-10x cheaper",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(chat_router, prefix="/v1")
app.include_router(usage_router, prefix="/v1")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "token0",
        "storage_mode": settings.storage_mode,
    }
