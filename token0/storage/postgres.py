from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from token0.config import settings
from token0.models.db import Base

engine = create_async_engine(settings.effective_database_url, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db():
    await engine.dispose()


async def get_session() -> AsyncSession:
    async with async_session() as session:
        yield session
