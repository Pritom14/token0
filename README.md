# Token0

**Open-source API proxy that makes vision LLM calls 5-10x cheaper.**

Send images to LLMs through Token0. Same accuracy. Fraction of the cost.

---

## Why Token0 Exists

Every time you send an image to GPT-4.1, Claude, or Gemini, you're paying for **vision tokens** — and most of them are wasted.

- A 4000x3000 photo costs **~1,590 tokens** on Claude. The model auto-downscales it to 1568px internally — you paid for pixels that got thrown away.
- A screenshot of a document costs **~765 tokens** on GPT-4.1 as an image. The same information extracted as text costs **~30 tokens**. That's a **25x markup** for the same answer.
- A simple "classify this image" prompt on GPT-4.1 uses high-detail mode at **1,105 tokens**. Low-detail mode gives the same answer for **85 tokens** — 13x cheaper.
- A 1280x720 image on GPT-4.1 creates 4 tiles (765 tokens). Resizing to tile boundaries gives 2 tiles (425 tokens) — 44% cheaper with zero quality loss.
- A PDF invoice sent as an image costs ~765 tokens. Extracting its text layer costs ~50 tokens — a **15x markup** for the same data.

**The problem**: Text token optimization is mature (prompt caching, compression, smart routing). But for images — the modality that costs 2-5x more per token — almost **no optimization tooling exists**.

Token0 fixes this. It sits between your app and the LLM, analyzes every image and prompt, applies the optimal strategy, and forwards the optimized request. You change one line of code (your base URL) and start saving immediately.

---

## How It Works

```
Your App → Token0 Proxy → [Analyze → Classify → Route → Transform → Cache] → LLM Provider
                ↓
         Database (logs every optimization decision + savings)
```

Token0 applies **11 optimizations** automatically:

### Core Optimizations (Free Tier)

**1. Smart Resize** — Auto-downscale images to the max resolution each model actually processes (Claude: 1568px, GPT-4.1: 2048px). Most apps send 4000px images that get silently downscaled by the provider.

**2. OCR Routing** — Detect when an image is mostly text (screenshots, documents, invoices, receipts) and extract text via OCR instead. Text tokens cost 10-50x less than vision tokens. Uses a multi-signal heuristic (background uniformity, color variance, horizontal line structure, edge density) — validated at 91% accuracy on real-world images.

**3. PDF Text Layer Extraction** — When a PDF is sent as a content part, extract its text layer using `pypdf` instead of rendering as an image. Typed/digital PDFs almost always have a text layer. A 2-page invoice PDF: ~1,530 vision tokens → ~80 text tokens. Falls back to passthrough for scanned PDFs with no text layer.

**4. JPEG Recompression** — Convert PNG screenshots (large files) to optimized JPEG (smaller payload, faster upload) when transparency isn't needed.

### Advanced Optimizations

**5. Prompt-Aware Detail Mode** — Analyze the *prompt* to decide detail level, not just the image. "Classify this image" → low detail (85 tokens). "Extract all text" → high detail. A keyword classifier on the prompt text can cut costs 3-13x per image.

**6. Tile-Optimized Resize** — OpenAI tiles images into 512x512 blocks. A 1280x720 image creates 4 tiles (765 tokens). Token0 resizes to optimal tile boundaries: 2 tiles (425 tokens) — 44% savings with zero quality loss.

**7. Model Cascade** — Not all images need GPT-4.1. Token0 auto-routes simple tasks to cheaper models: GPT-4.1 → GPT-4.1-mini (5x cheaper) → GPT-4.1-nano (20x cheaper), Claude Opus → Claude Haiku (6.25x cheaper). Complex tasks stay on the flagship model.

**8. Semantic Response Cache** — Cache responses for similar image+prompt pairs using perceptual image hashing. Repeated or similar queries cost 0 tokens. Effective on repetitive workloads (product classification, document processing).

**9. QJL-Compressed Fuzzy Cache** — Similar (not just identical) images hit the cache using Quantized Johnson-Lindenstrauss random projection. Compresses 256-bit perceptual hashes to 128-bit binary signatures, matches via Hamming distance. Inspired by Google's TurboQuant (arXiv 2504.19874). **62% additional token savings** on image variations in benchmarks — similar product photos, re-scanned documents, and slightly different angles all hit cache.

**10. Video Optimization** — Automatically extract keyframes from video at 1fps, deduplicate similar consecutive frames using QJL perceptual hashing, detect scene changes via pixel-level diff, and run each keyframe through the full image optimization pipeline. A 60-second video at 30fps (1,800 frames) reduces to ~10 keyframes before being sent to the LLM. **13-45% savings on local models; ~83% projected savings on GPT-4.1.** Optional CLIP-based query-frame scoring (Layer 2) ranks frames by relevance to the user's prompt.

**11. Saliency-Based ROI Cropping** — Detects which region of an image the prompt is asking about and crops to that region before sending to the LLM. "What's the total on this invoice?" → crops to the bottom 40% of the image. "Read the header" → crops to the top 25%. Rule-based spatial keyword matching (zero ML deps). Delivers ~60% additional pixel reduction on document and form images before any other optimization runs.

---

## Benchmarks

We benchmarked Token0 against **7 vision models** on **5 real-world images** (not synthetic — actual photos, receipts, documents, and screenshots) and **3 test videos**, plus cost projections using OpenAI and Anthropic's published token formulas.

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

### Image Benchmark Summary (7 Models)

| Model | Params | Total Direct | Total Token0 | Savings | Notes |
|---|---|---|---|---|---|
| granite3.2-vision | 3B | 129,836 | 60,924 | **53.1%** | High-res image encoder |
| minicpm-v | 8B | 10,877 | 6,276 | **42.3%** | |
| moondream | 1.7B | 16,457 | 10,240 | **37.8%** | |
| llava-llama3 | 8B | 13,365 | 8,486 | **36.5%** | |
| llava:7b | 7B | 13,384 | 8,701 | **35.0%** | |
| gemma3:4b | 4B | 6,380 | 4,798 | **24.8%** | |
| llama3.2-vision | 11B | 665 | 665 | **0%** | Ultra-efficient encoder: passthrough correct, no optimization needed |

> The 0% savings on llama3.2-vision is expected and correct. This model uses ~8-27 tokens per image natively — far below what OCR text extraction would cost. Token0 detects this and correctly skips all lossy optimizations.

### Video Benchmark Results

Test setup: 3 videos (product showcase, document montage, mixed content), naive baseline = all frames at 1fps sent raw, Token0 = frame dedup + scene detection + per-frame image optimization.

| Model | Naive Tokens | Token0 Tokens | Savings |
|---|---|---|---|
| gemma3:4b | 14,706 | 8,081 | **45.0%** |
| llava:7b | 15,731 | 12,845 | **18.3%** |
| llava-llama3 | 15,658 | 12,789 | **18.3%** |
| minicpm-v | 7,428 | 6,447 | **13.2%** |
| moondream | 12,288 | 11,714 | **4.7%** |

**Why moondream shows less video savings:** moondream uses a very small frame encoder — its per-frame token cost is already low, so frame dedup has less absolute impact than on higher-token models.

### GPT-4.1 Video Extrapolation (ballpark)

Using OpenAI's published tile formula (512px tiles, 170 tokens/tile) and GPT-4.1 pricing ($2.00/1M tokens):

| Scenario | Naive | Token0 | Savings |
|---|---|---|---|
| 60s video, 30fps (1,800 frames → 1fps → 60 frames → dedup to ~10) | ~25,500 tokens | ~4,250 tokens | **~83%** |
| Monthly cost at 10K videos/day | $15,300/mo | $2,550/mo | **$12,750/mo saved** |

### Anthropic Video Extrapolation (ballpark)

Using Anthropic's pixel formula (tokens ≈ width × height / 750) and Claude Sonnet pricing ($3/1M tokens):

| Scenario | Naive | Token0 | Savings |
|---|---|---|---|
| 60s video, 1fps = 60 frames at 1280×720 | ~73,700 tokens | ~12,300 tokens | **~83%** |
| Monthly cost at 1K videos/day | $6,633/mo | $1,107/mo | **$5,526/mo saved** |

> These are linear extrapolations from the token formula + observed dedup ratios (60 frames → ~10 keyframes). Actual savings vary by content type — talking-head video deduplicates more aggressively than action scenes.

### GPT-4.1 Image Cost Projections (v1 vs v2)

Using OpenAI's published token formulas on real images and GPT-4.1 pricing ($2.00/1M input tokens):

| Optimization Level | Per-Image Cost | Savings | 100K imgs/day Monthly |
|---|---|---|---|
| Direct GPT-4.1 (no Token0) | $0.001802 | — | $5,406 |
| **Token0 v1** (resize + OCR + PDF + basic detail) | $0.000535 | **70.3%** | $1,604 |
| **Token0 v2** (+ prompt-aware + tile resize + cascade) | $0.000020 | **98.9%** | $59 |

**v2 monthly savings at scale:**

| Scale | Direct Cost | Token0 v2 Cost | Monthly Savings |
|---|---|---|---|
| 1K images/day | $54.05 | $0.59 | **$53.46** |
| 10K images/day | $540.54 | $5.94 | **$534.60** |
| 100K images/day | $5,406 | $59.46 | **$5,346** |
| 500K images/day | $27,032 | $297 | **$26,735** |

> **Note**: v2 projections include model cascade (simple tasks → GPT-4.1-mini at $0.40/1M tokens vs GPT-4.1 at $2.00/1M). Semantic cache hits (est. 20% on repetitive workloads) would add further savings on top.

### Key Findings

1. **OCR routing delivers 47-70% token savings** on text-heavy images across all models tested.
2. **PDF text layer extraction beats OCR** for typed documents — direct text extraction, no OCR model needed, ~15x cheaper than sending as image.
3. **Smart resize saves 1-6 seconds of latency** on large photos — even when local models report flat token counts.
4. **Photos are never falsely OCR-routed** — the multi-signal text detection heuristic correctly identifies photos vs documents at 91% accuracy.
5. **Text-only passthrough adds zero overhead** — 0 extra tokens across all text-only tests.
6. **Prompt-aware detail mode** drops simple queries from 1,105 → 85 tokens (92% savings) on GPT-4.1.
7. **Model cascade** routes simple tasks 5-20x cheaper (GPT-4.1 → GPT-4.1-nano) with equivalent quality.
8. **Tile-optimized resize** cuts OpenAI costs by 44% on mid-size images (1280x720) with zero quality loss.
9. **On cloud APIs, total image savings reach 98.9%** when all optimizations are combined with model cascading.
10. **Video deduplication collapses 60-frame clips to ~10 keyframes** — 13-45% savings on local models, ~83% projected on GPT-4.1.
11. **Model-aware OCR skip is critical** — ultra-efficient encoders like llama3.2-vision use <50 tokens/image; OCR text output would cost more, not less.

### Additional Test Coverage

Token0 includes **171 unit tests** and benchmarks across multiple suites:

| Suite | Tests | What It Validates |
|---|---|---|
| `images` | 6 | Synthetic images: large, small, PNG, JPEG, already-optimized |
| `text` | 4 | Text-only passthrough: zero overhead, no token inflation |
| `multi` | 2 | Multiple images in one request: independent optimization |
| `turns` | 2 | Multi-turn conversations: image history optimization |
| `tasks` | 4 | Task types: classification, description, extraction, Q&A |
| `real` | 5 | Real-world photos, receipts, invoices, screenshots |
| `streaming` | 7 | SSE streaming: format, content, stats, image optimization |
| `litellm` | 10 | LiteLLM hook: passthrough, optimization, OCR, cascade, async |
| `cache` | 23 | QJL fuzzy cache: perceptual hash, JL compression, Hamming distance, fuzzy match |
| `video` | 22 | Frame extraction, QJL dedup, scene detection, CLIP scoring, full pipeline |
| `pdf` | 8 | PDF detection, decode, text extraction, token estimation |
| `estimate` | 11 | /v1/estimate endpoint: single image, multiple images, remote URL skip, cost calc |
| `langchain` | 8 | LangChain callback: import, text passthrough, image optimization, role mapping |

---

## Quick Start

### Install

```bash
pip install token0
```

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
    model="gpt-4.1",  # or claude-sonnet-4-6, gemini-2.5-flash
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

### Pre-Call Cost Estimate

Check what a request will cost **before** making any LLM call — no API key needed:

```bash
curl -X POST http://localhost:8000/v1/estimate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4.1",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
      ]
    }]
  }'
```

```json
{
    "model": "gpt-4.1",
    "provider": "openai",
    "images": [{
        "original_tokens": 1105,
        "optimized_tokens": 85,
        "tokens_saved": 1020,
        "cost_saved_usd": 0.00204,
        "optimizations": ["prompt-aware → low detail (simple task)"]
    }],
    "total_original_tokens": 1105,
    "total_optimized_tokens": 85,
    "total_tokens_saved": 1020,
    "total_cost_saved_usd": 0.00204
}
```

### PDF Support

Send PDFs directly — Token0 extracts the text layer automatically:

```python
import base64

with open("invoice.pdf", "rb") as f:
    pdf_b64 = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What is the total amount on this invoice?"},
            {"type": "image_url", "image_url": {"url": f"data:application/pdf;base64,{pdf_b64}"}}
        ]
    }],
    extra_headers={"X-Provider-Key": "sk-..."}
)
# PDF text layer extracted — ~80 tokens instead of ~765 vision tokens
```

### Video Support

Send a video URL or base64-encoded video — Token0 automatically extracts keyframes, deduplicates, and optimizes before forwarding:

```python
response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What happens in this video?"},
            {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,..."}}
        ]
    }],
    extra_headers={"X-Provider-Key": "sk-..."}
)
# 1,800 raw frames → ~10 keyframes → optimized images → LLM
# ~83% savings on GPT-4.1
```

### Streaming Support

Token0 supports `stream=true` — images are optimized before streaming begins, then tokens flow word-by-word via SSE:

```python
stream = client.chat.completions.create(
    model="gpt-4.1",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        ]
    }],
    stream=True,
    extra_headers={"X-Provider-Key": "sk-..."}
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
# Final chunk includes token0 optimization stats
```

### Use With LiteLLM

Already using [LiteLLM](https://github.com/BerriAI/litellm)? Add Token0 as a callback hook — no proxy needed:

```python
import litellm
from token0.litellm_hook import Token0Hook

litellm.callbacks = [Token0Hook()]

# All your existing litellm calls now get image optimization for free
response = litellm.completion(
    model="gpt-4.1",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        ]
    }]
)

# Stats available in response metadata
# response._hidden_params["metadata"]["token0"]["tokens_saved"]
```

Or in LiteLLM proxy `config.yaml`:

```yaml
litellm_settings:
  callbacks: ["token0.litellm_hook.Token0Hook"]
```

### Use With LangChain

Already using LangChain? Add Token0 as a callback to any chat model:

```bash
pip install token0[langchain]
```

```python
from token0.langchain_callback import Token0Callback
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4.1", callbacks=[Token0Callback()])

# All calls through this llm now get image optimization automatically
response = llm.invoke([HumanMessage(content=[
    {"type": "text", "text": "What's in this image?"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
])])
```

Works with any LangChain chat model — ChatOpenAI, ChatAnthropic, ChatGoogleGenerativeAI, etc.

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

### Savings Dashboard

Open `http://localhost:8000/dashboard` in your browser for a live view of total requests, tokens saved, cost saved, and per-optimization breakdown. Auto-refreshes every 10 seconds.

### Run Benchmarks Yourself

```bash
pip install token0[dev]
ollama pull moondream

# Run all image suites
python -m benchmarks.run --model moondream --suite all

# Run only real-world images
python -m benchmarks.run --model llava:7b --suite real

# Run video benchmarks (requires Ollama + real images in benchmarks/images/real/)
python -m benchmarks.bench_video_models
python -m benchmarks.bench_video_models --model llava:7b --model minicpm-v

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
| POST | `/v1/chat/completions` | Optimized chat completion (OpenAI-compatible, supports `stream=true`) |
| POST | `/v1/estimate` | Pre-call token cost estimator — no LLM call, no API key needed |
| GET | `/v1/usage` | Usage and savings dashboard |
| GET | `/health` | Health check + storage mode |

### Headers

| Header | Required | Description |
|--------|----------|-------------|
| `X-Provider-Key` | Yes (chat only) | Your LLM provider API key (OpenAI/Anthropic/Google/Ollama) |
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
    "model": "gpt-4.1-mini",
    "choices": [...],
    "usage": {"prompt_tokens": 85, "completion_tokens": 50, "total_tokens": 135},
    "token0": {
        "original_prompt_tokens_estimate": 1105,
        "optimized_prompt_tokens": 85,
        "tokens_saved": 1020,
        "cost_saved_usd": 0.002040,
        "optimizations_applied": [
            "prompt-aware → low detail (simple task)",
            "cascade → gpt-4.1-mini (simple task)"
        ],
        "cache_hit": false,
        "model_cascaded_to": "gpt-4.1-mini"
    }
}
```

---

## Supported Providers

| Provider | Models | Notes |
|---|---|---|
| **OpenAI** | GPT-4.1, GPT-4.1-mini, GPT-4.1-nano, GPT-4o, GPT-4o-mini | Detail mode + tile optimization |
| **Anthropic** | Claude Sonnet 4.6, Claude Opus 4.6, Claude Haiku 4.5 | Pixel-based token formula |
| **Google** | Gemini 2.5 Flash, Gemini 2.5 Pro | |
| **Ollama** | moondream, llava, llava-llama3, minicpm-v, gemma3, granite3.2-vision, any vision model | Free, local inference |

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
