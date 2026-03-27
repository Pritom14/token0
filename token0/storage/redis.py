"""Cache layer — Redis in full mode, in-memory dict in lite mode."""

from token0.config import settings

pool = None
_memory_cache: dict[str, str] = {}


class MemoryCache:
    """Simple in-memory cache that mimics the redis async interface we use."""

    async def get(self, key: str) -> str | None:
        return _memory_cache.get(key)

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        _memory_cache[key] = value

    async def delete(self, key: str) -> None:
        _memory_cache.pop(key, None)

    async def incr(self, key: str) -> int:
        val = int(_memory_cache.get(key, "0")) + 1
        _memory_cache[key] = str(val)
        return val

    async def expire(self, key: str, seconds: int) -> None:
        pass  # no expiry in lite mode

    async def close(self) -> None:
        _memory_cache.clear()


async def init_redis():
    global pool
    if settings.is_lite:
        pool = MemoryCache()
    else:
        import redis.asyncio as redis

        pool = redis.from_url(settings.redis_url, decode_responses=True)


async def close_redis():
    global pool
    if pool:
        await pool.close()


def get_redis():
    if pool is None:
        raise RuntimeError("Cache not initialized")
    return pool
