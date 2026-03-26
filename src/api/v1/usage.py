"""Usage and savings dashboard endpoints."""

from fastapi import APIRouter
from sqlalchemy import func, select

from src.models.db import Request
from src.models.request import UsageSummary
from src.storage.postgres import async_session

router = APIRouter()


@router.get("/usage", response_model=UsageSummary)
async def get_usage():
    """Get aggregate usage and savings stats."""
    async with async_session() as session:
        result = await session.execute(
            select(
                func.count(Request.id).label("total_requests"),
                func.coalesce(func.sum(Request.tokens_saved), 0).label("total_tokens_saved"),
                func.coalesce(func.sum(Request.cost_saved), 0.0).label("total_cost_saved"),
                func.coalesce(func.avg(Request.tokens_original_estimate), 0).label("avg_original"),
                func.coalesce(func.avg(Request.tokens_actual), 0).label("avg_actual"),
            )
        )
        row = result.one()

        # Get optimization type breakdown
        breakdown_result = await session.execute(
            select(Request.optimization_type, func.count(Request.id)).group_by(
                Request.optimization_type
            )
        )
        breakdown = {opt_type: count for opt_type, count in breakdown_result.all()}

        avg_original = float(row.avg_original) if row.avg_original else 0
        avg_actual = float(row.avg_actual) if row.avg_actual else 0
        compression_ratio = (avg_original / avg_actual) if avg_actual > 0 else 0

        return UsageSummary(
            total_requests=row.total_requests,
            total_tokens_saved=int(row.total_tokens_saved),
            total_cost_saved_usd=round(float(row.total_cost_saved), 4),
            avg_compression_ratio=round(compression_ratio, 2),
            optimization_breakdown=breakdown,
        )
