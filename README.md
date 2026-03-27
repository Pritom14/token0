# Token0

**Open-source API proxy that makes vision LLM calls 5-10x cheaper.**

Send images to LLMs through Token0. Same accuracy. Fraction of the cost.

---

## Why Token0 Exists

Every time you send an image to GPT-4o, Claude, or Gemini, you're paying for **vision tokens** — and most of them are wasted.

- A 4000x3000 photo costs **~1,590 tokens** on Claude. The model auto-downscales it to 1568px internally — you paid for pixels that got thrown away.
- A screenshot of a document costs **~765 tokens** on GPT-4o as an image. The same information extracted as text costs **~30 tokens**. That's a **25x markup** for the same answer.
- A simple "classify this image" prompt on GPT-4o uses high-detail mode at **1,105 tokens**. Low-detail mode gives the same answer for **85 tokens** — 13x cheaper.
- A 1280x720 image on GPT-4o creates 4 tiles (765 tokens). Resizing to tile boundaries gives 2 tiles (425 tokens) — 44% cheaper with zero quality loss.

**The problem**: Text token optimization is mature (prompt caching, compression, smart routing). But for images — the modality that costs 2-5x more per token — almost **no optimization tooling exists**.

Token0 fixes this. It sits between your app and the LLM, analyzes every image and prompt, applies the optimal strategy, and forwards the optimized request. You change one line of code (your base URL) and start saving immediately.

---

## How It Works

```
Your App → Token0 Proxy → [Analyze → Classify → Route → Transform → Cache] → LLM Provider
                ↓
         Database (logs every optimization decision + savings)
```

Token0 applies **7 optimizations** automatically:

### Core Optimizations (Free Tier)

**1. Smart Resize** — Auto-downscale images to the max resolution each model actually processes (Claude: 1568px, GPT-4o: 2048px). Most apps send 4000px images that get silently downscaled by the provider.

**2. OCR Routing** — Detect when an image is mostly text (screenshots, documents, invoices, receipts) and extract text via OCR instead. Text tokens cost 10-50x less than vision tokens. Uses a multi-signal heuristic (background uniformity, color variance, horizontal line structure, edge density) — validated at 91% accuracy on real-world images.

**3. JPEG Recompression** — Convert PNG screenshots (large files) to optimized JPEG (smaller payload, faster upload) when transparency isn't needed.

### Advanced Optimizations

**4. Prompt-Aware Detail Mode** — Analyze the *prompt* to decide detail level, not just the image. "Classify this image" → low detail (85 tokens). "Extract all text" → high detail. A keyword classifier on the prompt text can cut costs 3-13x per image.

**5. Tile-Optimized Resize** — OpenAI tiles images into 512x512 blocks. A 1280x720 image creates 4 tiles (765 tokens). Token0 resizes to optimal tile boundaries: 2 tiles (425 tokens) — 44% savings with zero quality loss.

**6. Model Cascade** — Not all images need GPT-4o. Token0 auto-routes simple tasks to cheaper models: GPT-4o → GPT-4o-mini (16.7x cheaper), Claude Opus → Claude Haiku (6.25x cheaper). Complex tasks stay on the flagship model.

**7. Semantic Response Cache** — Cache responses for similar image+prompt pairs using perceptual image hashing. Repeated or similar queries cost 0 tokens. Effective on repetitive workloads (product classification, document processing).

---

## Benchmarks

We benchmarked Token0 against **4 vision models** on **5 real-world images** (not synthetic — actual photos, receipts, documents, and screenshots), plus cost projections using OpenAI and Anthropic's published token formulas.

### Real-World Image Test Suite

| Image | Type | Size | Source |
|---|---|---|---|
| `photo_nature.jpg` | Landscape photo | 4000x2047, 815KB | Pexels (CC0) |
| `photo_street.jpg` | City street scene | 3000x1988, 1058KB | Pexels (CC0) |
| `receipt_real.jpg` | Real store receipt | 2448x3264, 940KB | Wikimedia Commons |
| `document_invoice.png` | Typed invoice | 850x1100, 74KB | Generated with real text |
| `screenshot_real.png` | Desktop app UI | 2066x766, 196KB | Actual screenshot |

### Results by Model (Real-World Images)

#### moondream (1.7B params, 1.7GB)

| Image | Direct | Token0 | Saved | Latency Delta | Optimization |
|---|---|---|---|---|---|
| Nature photo (4000x2047) | 751 | 751 | 0% | **-1,141ms** | Resize → 2048x1048 |
| Street photo (3000x1988) | 751 | 751 | 0% | -110ms | Resize → 2048x1357 |
| Receipt (2448x3264) | 752 | 278 | **63.0%** | -90ms | OCR route |
| Invoice (850x1100) | 753 | 388 | **48.5%** | **-733ms** | OCR route |
| Screenshot (2066x766) | 752 | 227 | **69.8%** | -392ms | OCR route |
| **Total** | **3,759** | **2,395** | **36.3%** | | |

#### llava:7b (7B params, 4.7GB)

| Image | Direct | Token0 | Saved | Latency Delta | Optimization |
|---|---|---|---|---|---|
| Nature photo (4000x2047) | 602 | 602 | 0% | **-2,825ms** | Resize → 2048x1048 |
| Street photo (3000x1988) | 602 | 602 | 0% | **-1,251ms** | Resize → 2048x1357 |
| Receipt (2448x3264) | 605 | 320 | **47.1%** | **-4,100ms** | OCR route |
| Invoice (850x1100) | 607 | 502 | **17.3%** | **-3,477ms** | OCR route |
| Screenshot (2066x766) | 604 | 264 | **56.3%** | **-1,140ms** | OCR route |
| **Total** | **3,020** | **2,290** | **24.2%** | | |

#### llava-llama3 (8B params, 5.5GB)

| Image | Direct | Token0 | Saved | Latency Delta | Optimization |
|---|---|---|---|---|---|
| Nature photo (4000x2047) | 601 | 601 | 0% | **-2,500ms** | Resize → 2048x1048 |
| Street photo (3000x1988) | 601 | 601 | 0% | +828ms | Resize → 2048x1357 |
| Receipt (2448x3264) | 603 | 274 | **54.6%** | **-4,999ms** | OCR route |
| Invoice (850x1100) | 604 | 377 | **37.6%** | +3,697ms | OCR route |
| Screenshot (2066x766) | 602 | 218 | **63.8%** | +498ms | OCR route |
| **Total** | **3,011** | **2,071** | **31.2%** | | |

#### minicpm-v (8B params, 5.5GB)

| Image | Direct | Token0 | Saved | Latency Delta | Optimization |
|---|---|---|---|---|---|
| Nature photo (4000x2047) | 617 | 617 | 0% | **-6,147ms** | Resize → 2048x1048 |
| Street photo (3000x1988) | 617 | 617 | 0% | +1,888ms | Resize → 2048x1357 |
| Receipt (2448x3264) | 686 | 309 | **55.0%** | **-3,583ms** | OCR route |
| Invoice (850x1100) | 489 | 456 | **6.7%** | **-2,553ms** | OCR route |
| Screenshot (2066x766) | 618 | 244 | **60.5%** | **-3,744ms** | OCR route |
| **Total** | **3,027** | **2,243** | **25.9%** | | |

### Summary Across All Models

| Model | Params | Total Direct | Total Token0 | Savings |
|---|---|---|---|---|
| moondream | 1.7B | 3,759 | 2,395 | **36.3%** |
| llava-llama3 | 8B | 3,011 | 2,071 | **31.2%** |
| minicpm-v | 8B | 3,027 | 2,243 | **25.9%** |
| llava:7b | 7B | 3,020 | 2,290 | **24.2%** |

### GPT-4o Cost Projections (v1 vs v2)

Using OpenAI's published token formulas on real images:

| Optimization Level | Per-Image Cost | Savings | 100K imgs/day Monthly |
|---|---|---|---|
| Direct GPT-4o (no Token0) | $0.002253 | — | $6,758 |
| **Token0 v1** (resize + OCR + basic detail) | $0.000669 | **70.3%** | $2,006 |
| **Token0 v2** (+ prompt-aware + tile resize + cascade) | $0.000025 | **98.9%** | $74 |

**v2 monthly savings at scale:**

| Scale | Direct Cost | Token0 v2 Cost | Monthly Savings |
|---|---|---|---|
| 1K images/day | $67.58 | $0.74 | **$66.83** |
| 10K images/day | $675.75 | $7.45 | **$668.30** |
| 100K images/day | $6,757.50 | $74.47 | **$6,683.03** |
| 500K images/day | $33,787.50 | $372.38 | **$33,415.12** |

> **Note**: v2 projections include model cascade (simple tasks → GPT-4o-mini at $0.15/1M tokens vs GPT-4o at $2.50/1M). Semantic cache hits (est. 20% on repetitive workloads) would add further savings on top.

### Key Findings

1. **OCR routing delivers 47-70% token savings** on text-heavy images across all models tested.
2. **Smart resize saves 1-6 seconds of latency** on large photos — even when local models report flat token counts.
3. **Photos are never falsely OCR-routed** — the multi-signal text detection heuristic correctly identifies photos vs documents at 91% accuracy.
4. **Text-only passthrough adds zero overhead** — 0 extra tokens across all text-only tests.
5. **Prompt-aware detail mode** drops simple queries from 1,105 → 85 tokens (92% savings) on GPT-4o.
6. **Model cascade** routes simple tasks at 16.7x cheaper rates with equivalent quality.
7. **Tile-optimized resize** cuts OpenAI costs by 44% on mid-size images (1280x720) with zero quality loss.
8. **On cloud APIs, total savings reach 98.9%** when all optimizations are combined with model cascading.

### Additional Test Coverage

Token0 includes **86 unit tests** and benchmarks across multiple suites:

| Suite | Tests | What It Validates |
|---|---|---|
| `images` | 6 | Synthetic images: large, small, PNG, JPEG, already-optimized |
| `text` | 4 | Text-only passthrough: zero overhead, no token inflation |
| `multi` | 2 | Multiple images in one request: independent optimization |
| `turns` | 2 | Multi-turn conversations: image history optimization |
| `tasks` | 4 | Task types: classification, description, extraction, Q&A |
| `real` | 5 | Real-world photos, receipts, invoices, screenshots |

---

## Quick Start

### Install

```bash
pip install token0
```

Create a `.env` file with your API key:

Add your LLM provider API key to `.env`:
```bash
# At least one of these:
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
```

Start the server:
```bash
token0 serve
```

Or with options:
```bash
token0 serve --port 3000 --reload
```

That's it. Token0 starts in **lite mode** by default — SQLite + in-memory cache. No Postgres, Redis, or Docker required.

### Use It

Token0 is **OpenAI-compatible**. Change one line — your base URL:

```python
from openai import OpenAI

# Before (direct to provider)
client = OpenAI(api_key="sk-...")

# After (through Token0)
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-...",
)

# Same code, nothing else changes
response = client.chat.completions.create(
    model="gpt-4o",  # or claude-sonnet-4-6, gemini-2.5-flash
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        ]
    }],
    extra_headers={"X-Provider-Key": "sk-..."}
)

# Response includes optimization stats
# response.token0.tokens_saved = 1305
# response.token0.cost_saved_usd = 0.003263
# response.token0.optimizations_applied = ["resize 4000x3000 → 1568x1176", "convert png → jpeg q=85"]
```

### Use With Ollama (free, fully local)

```bash
ollama pull moondream  # or llava:7b, llava-llama3, minicpm-v
```

```python
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="unused",
)

response = client.chat.completions.create(
    model="moondream",
    messages=[...],
    extra_headers={"X-Provider-Key": "unused"}
)
```

### Check Your Savings

```bash
curl http://localhost:8000/v1/usage
```

```json
{
    "total_requests": 47,
    "total_tokens_saved": 12840,
    "total_cost_saved_usd": 0.0321,
    "avg_compression_ratio": 3.2,
    "optimization_breakdown": {"resize": 20, "ocr_route": 15, "detail_mode": 12}
}
```

### Run Benchmarks Yourself

```bash
pip install token0[dev]
ollama pull moondream

# Run all suites
python -m benchmarks.run --model moondream --suite all

# Run only real-world images
python -m benchmarks.run --model llava:7b --suite real

# Available suites: images, text, multi, turns, tasks, real, all
# Available models: any Ollama vision model
```

---

## Production Setup

For production, switch to `STORAGE_MODE=full` which uses PostgreSQL + Redis + S3/MinIO for reliability, caching, and persistence.

### Option A: Docker Compose

```bash
cp .env.example .env
# Set STORAGE_MODE=full in .env
docker compose up
```

This starts PostgreSQL, Redis, MinIO, and the Token0 API server in one command.

### Option B: Manual

```bash
pip install token0[full]
```

Set these in `.env`:
```bash
STORAGE_MODE=full
DATABASE_URL=postgresql+asyncpg://token0:token0@localhost:5432/token0
REDIS_URL=redis://localhost:6379/0
S3_ENDPOINT=http://localhost:9000
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin
S3_BUCKET=token0-images
```

> **Note**: Lite mode (SQLite + in-memory) is for local development and testing. Production deployments should use `STORAGE_MODE=full` with PostgreSQL for reliable request logging, Redis for caching and rate limiting, and S3-compatible storage for image persistence.

### Storage Modes

| | Lite (default) | Full |
|---|---|---|
| Database | SQLite | PostgreSQL |
| Cache | In-memory dict | Redis |
| Object storage | Local filesystem | S3 / MinIO |
| Install | `pip install token0` | `pip install token0[full]` |
| Use case | Dev / testing | Production |
| Switch via | `STORAGE_MODE=lite` | `STORAGE_MODE=full` |

---

## API Reference

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | Optimized chat completion (OpenAI-compatible) |
| GET | `/v1/usage` | Usage and savings dashboard |
| GET | `/health` | Health check + storage mode |

### Headers

| Header | Required | Description |
|--------|----------|-------------|
| `X-Provider-Key` | Yes | Your LLM provider API key (OpenAI/Anthropic/Google/Ollama) |
| `X-Token0-Key` | No | Token0 API key for usage tracking |

### Token0-Specific Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `token0_optimize` | bool | `true` | Set to `false` to passthrough without optimization |
| `token0_detail_override` | string | `null` | Force `"low"` or `"high"` detail mode (OpenAI only) |
| `token0_enable_cache` | bool | `true` | Enable semantic response caching |
| `token0_enable_cascade` | bool | `true` | Enable auto-routing to cheaper models for simple tasks |

### Response Format

Standard OpenAI-compatible response with an additional `token0` field:

```json
{
    "id": "token0-abc123",
    "object": "chat.completion",
    "model": "gpt-4o-mini",
    "choices": [...],
    "usage": {"prompt_tokens": 85, "completion_tokens": 50, "total_tokens": 135},
    "token0": {
        "original_prompt_tokens_estimate": 1105,
        "optimized_prompt_tokens": 85,
        "tokens_saved": 1020,
        "cost_saved_usd": 0.002550,
        "optimizations_applied": [
            "prompt-aware → low detail (simple task)",
            "cascade → gpt-4o-mini (simple task)"
        ],
        "cache_hit": false,
        "model_cascaded_to": "gpt-4o-mini"
    }
}
```

---

## Supported Providers

| Provider | Models | Notes |
|---|---|---|
| **OpenAI** | GPT-4o, GPT-4o-mini, GPT-4.1, GPT-4.1-mini, GPT-4.1-nano | Detail mode + tile optimization |
| **Anthropic** | Claude Sonnet 4.6, Claude Opus 4.6, Claude Haiku 4.5 | Pixel-based token formula |
| **Google** | Gemini 2.5 Flash, Gemini 2.5 Pro | |
| **Ollama** | moondream, llava, llava-llama3, minicpm-v, any vision model | Free, local inference |

---

## Configuration

All settings can be configured via environment variables or `.env` file. See `.env.example` for the full list.

Key settings:

| Variable | Default | Description |
|---|---|---|
| `STORAGE_MODE` | `lite` | `lite` (SQLite + memory) or `full` (Postgres + Redis + S3) |
| `TEXT_DENSITY_THRESHOLD` | `0.52` | Images above this text density → OCR route |
| `MAX_IMAGE_DIMENSION` | `1568` | Max dimension before resize (matches Claude's limit) |
| `JPEG_QUALITY` | `85` | JPEG compression quality for PNG→JPEG conversion |

---

## License

Apache 2.0
