from pydantic import BaseModel


class ImageUrl(BaseModel):
    url: str
    detail: str | None = None  # "low", "high", "auto"


class ContentPart(BaseModel):
    type: str  # "text" or "image_url"
    text: str | None = None
    image_url: ImageUrl | None = None


class Message(BaseModel):
    role: str
    content: str | list[ContentPart]


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool = False
    # Token0-specific options
    token0_optimize: bool = True  # set to False to passthrough without optimization
    token0_detail_override: str | None = None  # force "low" or "high"
    token0_enable_cache: bool = True  # semantic response caching
    token0_enable_cascade: bool = True  # auto-route to cheaper models for simple tasks


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Token0Usage(BaseModel):
    original_prompt_tokens_estimate: int
    optimized_prompt_tokens: int
    tokens_saved: int
    cost_saved_usd: float
    optimizations_applied: list[str]
    cache_hit: bool = False
    model_cascaded_to: str | None = None


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str | None = None


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    model: str
    choices: list[Choice]
    usage: UsageInfo
    token0: Token0Usage


class UsageSummary(BaseModel):
    total_requests: int
    total_tokens_saved: int
    total_cost_saved_usd: float
    avg_compression_ratio: float
    optimization_breakdown: dict[str, int]
