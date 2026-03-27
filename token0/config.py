from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Storage mode: "lite" (SQLite + in-memory) or "full" (Postgres + Redis + S3)
    # Use "lite" for local dev/testing, "full" for production
    storage_mode: str = "lite"

    # Database — only needed in full mode
    database_url: str = "postgresql+asyncpg://token0:token0@localhost:5432/token0"

    # SQLite path — used in lite mode
    sqlite_path: str = "token0.db"

    # Redis — only needed in full mode
    redis_url: str = "redis://localhost:6379/0"

    # Object Storage — only needed in full mode
    s3_endpoint: str = "http://localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket: str = "token0-images"

    # LLM Provider Keys (optional defaults — users can pass their own per-request)
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    # Auth
    token0_master_key: str = "change-me-in-production"

    # Optimization defaults
    max_image_dimension: int = 1568  # Claude's max before auto-downscale
    jpeg_quality: int = 85
    text_density_threshold: float = 0.52  # Above this → OCR route instead of vision

    @property
    def is_lite(self) -> bool:
        return self.storage_mode == "lite"

    @property
    def effective_database_url(self) -> str:
        if self.is_lite:
            return f"sqlite+aiosqlite:///{self.sqlite_path}"
        return self.database_url

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
