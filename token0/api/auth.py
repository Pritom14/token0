"""API key authentication middleware."""

import hashlib

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from token0.config import settings

api_key_header = APIKeyHeader(name="X-Token0-Key", auto_error=False)


def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


async def verify_api_key(api_key: str | None = Security(api_key_header)) -> str | None:
    """Verify API key if provided. For open-source mode, auth is optional.

    Returns the hashed key if valid, None if no key provided.
    In cloud mode, this would be required.
    """
    if api_key is None:
        # Open-source mode — no auth required
        return None

    # For now, validate against master key
    if api_key == settings.token0_master_key:
        return hash_api_key(api_key)

    # TODO: Look up in database for multi-tenant cloud mode
    raise HTTPException(status_code=401, detail="Invalid API key")
