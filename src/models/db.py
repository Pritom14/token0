import uuid
from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255))
    api_key_hash: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    # Customers can pass their own provider keys, stored encrypted
    provider_keys: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    is_active: Mapped[bool] = mapped_column(default=True)


class Request(Base):
    __tablename__ = "requests"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    customer_id: Mapped[str] = mapped_column(String(36), index=True)
    provider: Mapped[str] = mapped_column(String(50))  # openai, anthropic, google
    model: Mapped[str] = mapped_column(String(100))  # gpt-4o, claude-sonnet-4-6, etc.

    # Image metadata
    original_width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    original_height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    original_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    image_count: Mapped[int] = mapped_column(Integer, default=0)

    # Optimization applied
    optimization_type: Mapped[str] = mapped_column(
        String(50)
    )  # resize, ocr_route, detail_mode, none
    optimized_width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    optimized_height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    detail_mode: Mapped[str | None] = mapped_column(String(20), nullable=True)  # low, high, auto

    # Token accounting
    tokens_original_estimate: Mapped[int] = mapped_column(Integer)  # what it would have cost
    tokens_actual: Mapped[int] = mapped_column(Integer)  # what it actually cost
    tokens_saved: Mapped[int] = mapped_column(Integer)

    # Cost accounting (USD)
    cost_original_estimate: Mapped[float] = mapped_column(Float)
    cost_actual: Mapped[float] = mapped_column(Float)
    cost_saved: Mapped[float] = mapped_column(Float)

    # Prompt/response metadata
    prompt_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    response_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Full optimization decision log
    optimization_details: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class OptimizationProfile(Base):
    """Learned optimization profiles — Month 3 feature.
    Stores what works best per customer + content type + task type.
    """

    __tablename__ = "optimization_profiles"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    customer_id: Mapped[str] = mapped_column(String(36), index=True)
    content_type: Mapped[str] = mapped_column(String(100))  # invoice, receipt, screenshot, photo
    task_type: Mapped[str] = mapped_column(String(100))  # classify, extract, describe, ocr
    recommended_optimization: Mapped[str] = mapped_column(String(50))
    recommended_detail_mode: Mapped[str | None] = mapped_column(String(20), nullable=True)
    recommended_max_dimension: Mapped[int | None] = mapped_column(Integer, nullable=True)
    avg_quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    sample_count: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
