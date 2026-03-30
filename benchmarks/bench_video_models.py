"""Benchmark: Video optimization against real Ollama vision models.

Creates test videos from real images, then compares:
  - Naive: send all frames at 1fps directly to the model
  - Token0: extract keyframes, dedup, optimize, then send

Usage:
    python -m benchmarks.bench_video_models
    python -m benchmarks.bench_video_models --model moondream
    python -m benchmarks.bench_video_models --model llava:7b --model minicpm-v
"""

import argparse
import asyncio
import base64
import io
import os
import tempfile
import time

import cv2
import numpy as np
from PIL import Image

from token0.optimization.analyzer import analyze_image
from token0.optimization.router import plan_optimization
from token0.optimization.transformer import transform_image
from token0.optimization.video import (
    deduplicate_frames,
    detect_scene_changes,
    extract_frames,
    process_video,
)
from token0.providers.ollama import OllamaProvider

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BENCHMARK_DIR, "images")
REAL_DIR = os.path.join(IMAGES_DIR, "real")

DEFAULT_MODELS = ["moondream", "llava:7b", "llava-llama3", "minicpm-v"]


def _pil_to_data_uri(img: Image.Image, quality: int = 85) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def _create_video_from_images(
    image_paths: list[str],
    frames_per_image: int = 30,
    fps: float = 30.0,
    noise_level: int = 8,
) -> str:
    """Create a video by cycling through real images with slight variation.

    Each image becomes a 'scene' lasting frames_per_image/fps seconds.
    Slight noise is added per frame to simulate camera jitter.
    """
    images = []
    for path in image_paths:
        img = Image.open(path).convert("RGB").resize((640, 480))
        images.append(np.array(img))

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp.name, fourcc, fps, (640, 480))

    for scene_idx, base_pixels in enumerate(images):
        # Convert RGB to BGR for OpenCV
        base_bgr = cv2.cvtColor(base_pixels, cv2.COLOR_RGB2BGR)
        for frame_idx in range(frames_per_image):
            rng = np.random.RandomState(seed=scene_idx * 10000 + frame_idx)
            noise = rng.randint(-noise_level, noise_level + 1, base_bgr.shape, dtype=np.int16)
            frame = np.clip(base_bgr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            writer.write(frame)

    writer.release()
    total_frames = len(images) * frames_per_image
    duration = total_frames / fps
    print(f"  Created video: {len(images)} scenes, {total_frames} frames, {duration:.1f}s")
    return tmp.name


def _optimize_frame(pil_image: Image.Image, model: str) -> tuple[dict, list[str], int, int]:
    """Run token0 optimization on a single frame. Returns (message_part, optimizations, before, after)."""
    data_uri = _pil_to_data_uri(pil_image)
    analysis, raw_bytes, pil_img = analyze_image(data_uri)
    plan = plan_optimization(analysis, model)

    if plan.use_ocr_route:
        result = transform_image(plan, analysis, raw_bytes, pil_img)
        part = {"type": "text", "text": f"[Extracted text from video frame]:\n{result['content']}"}
    elif any([plan.resize, plan.recompress_jpeg, plan.force_detail_low]):
        result = transform_image(plan, analysis, raw_bytes, pil_img)
        detail = "low" if plan.force_detail_low else "auto"
        part = {
            "type": "image_url",
            "image_url": {
                "url": f"data:{result['media_type']};base64,{result['base64']}",
                "detail": detail,
            },
        }
    else:
        part = {"type": "image_url", "image_url": {"url": data_uri, "detail": "auto"}}

    return part, plan.reasons, plan.estimated_tokens_before, plan.estimated_tokens_after


async def run_video_benchmark(
    model: str, provider: OllamaProvider, video_path: str, video_name: str, prompt: str
):
    """Run a single video benchmark: naive vs token0 optimized."""
    print(f'\n  [{video_name}] — "{prompt}"')

    # --- Naive approach: extract at 1fps, send all frames ---
    naive_frames = extract_frames(video_path, fps=1.0, max_frames=60)
    print(f"    Naive: {len(naive_frames)} frames at 1fps")

    # Build naive message (all frames as images)
    naive_parts = [{"type": "text", "text": prompt}]
    for _, frame in naive_frames:
        naive_parts.append(
            {
                "type": "image_url",
                "image_url": {"url": _pil_to_data_uri(frame), "detail": "auto"},
            }
        )

    naive_messages = [{"role": "user", "content": naive_parts}]

    print("    Sending naive to model...", end="", flush=True)
    start = time.time()
    try:
        naive_resp = await provider.chat_completion(
            model=model, messages=naive_messages, max_tokens=200
        )
        naive_latency = int((time.time() - start) * 1000)
        print(f" {naive_latency}ms, {naive_resp.prompt_tokens} prompt tokens")
    except Exception as e:
        print(f" ERROR: {e}")
        return None

    # --- Token0 approach: process_video + per-frame optimization ---
    optimized_images, video_stats = process_video(video_path, prompt=prompt, max_frames=32)
    print(
        f"    Token0: {video_stats['total_video_frames']} total → "
        f"{video_stats['frames_extracted_at_1fps']} extracted → "
        f"{video_stats['frames_after_dedup']} deduped → "
        f"{video_stats['frames_selected']} selected"
    )

    # Optimize each selected frame through image pipeline
    opt_parts = [{"type": "text", "text": prompt}]
    all_optimizations = [
        f"video: {video_stats['total_video_frames']} → {video_stats['frames_selected']} keyframes"
    ]
    total_before = 0
    total_after = 0

    for frame_img in optimized_images:
        part, reasons, before, after = _optimize_frame(frame_img, model)
        opt_parts.append(part)
        all_optimizations.extend(reasons)
        total_before += before
        total_after += after

    opt_messages = [{"role": "user", "content": opt_parts}]

    print("    Sending optimized to model...", end="", flush=True)
    start = time.time()
    try:
        opt_resp = await provider.chat_completion(
            model=model, messages=opt_messages, max_tokens=200
        )
        opt_latency = int((time.time() - start) * 1000)
        print(f" {opt_latency}ms, {opt_resp.prompt_tokens} prompt tokens")
    except Exception as e:
        print(f" ERROR: {e}")
        return None

    # Calculate savings
    saved = naive_resp.prompt_tokens - opt_resp.prompt_tokens
    pct = (saved / naive_resp.prompt_tokens * 100) if naive_resp.prompt_tokens > 0 else 0

    return {
        "video": video_name,
        "prompt": prompt,
        "naive_frames": len(naive_frames),
        "token0_frames": len(optimized_images),
        "naive_prompt_tokens": naive_resp.prompt_tokens,
        "token0_prompt_tokens": opt_resp.prompt_tokens,
        "tokens_saved": saved,
        "savings_pct": round(pct, 1),
        "naive_latency_ms": naive_latency,
        "token0_latency_ms": opt_latency,
        "latency_delta_ms": opt_latency - naive_latency,
        "optimizations": all_optimizations,
        "video_stats": video_stats,
    }


async def run_all_benchmarks(models: list[str]):
    provider = OllamaProvider(base_url="http://localhost:11434/v1")

    # --- Create test videos from real images ---
    print("=" * 80)
    print("  Video Optimization Benchmark — Real Models")
    print("=" * 80)

    # Check which real images exist
    real_images = []
    for name in [
        "photo_nature.jpg",
        "photo_street.jpg",
        "receipt_real.jpg",
        "document_invoice.png",
        "screenshot_real.png",
    ]:
        path = os.path.join(REAL_DIR, name)
        if os.path.exists(path):
            real_images.append(path)

    if not real_images:
        print("  ERROR: No real images found in benchmarks/images/real/")
        return

    print(f"\n  Found {len(real_images)} real images for video creation")

    # Video 1: Product showcase (nature + street photos, ~4s)
    video1_path = _create_video_from_images(
        [p for p in real_images if "photo" in p][:2],
        frames_per_image=60,
        fps=30.0,
    )

    # Video 2: Document scanning (receipt + invoice + screenshot, ~3s)
    doc_images = [p for p in real_images if "receipt" in p or "document" in p or "screenshot" in p][
        :3
    ]
    video2_path = _create_video_from_images(
        doc_images,
        frames_per_image=30,
        fps=30.0,
    )

    # Video 3: Mixed content (all images, ~5s)
    video3_path = _create_video_from_images(
        real_images[:5],
        frames_per_image=30,
        fps=30.0,
    )

    test_cases = [
        (video1_path, "photos_4s", "Describe what you see in this video"),
        (video2_path, "documents_3s", "Extract any text or numbers visible in this video"),
        (video3_path, "mixed_5s", "Summarize the content shown in this video"),
    ]

    # --- Also check for real video files in benchmarks/videos/ ---
    videos_dir = os.path.join(BENCHMARK_DIR, "videos")
    if os.path.isdir(videos_dir):
        for fname in sorted(os.listdir(videos_dir)):
            if fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                vpath = os.path.join(videos_dir, fname)
                vname = os.path.splitext(fname)[0]
                test_cases.append((vpath, f"real_{vname}", "Describe what happens in this video"))
                print(f"  Found real video: {fname}")


    all_results = {}

    for model in models:
        print(f"\n{'=' * 80}")
        print(f"  Model: {model}")
        print(f"{'=' * 80}")

        model_results = []
        for video_path, video_name, prompt in test_cases:
            result = await run_video_benchmark(model, provider, video_path, video_name, prompt)
            if result:
                model_results.append(result)

        all_results[model] = model_results

        # Print model summary
        if model_results:
            total_naive = sum(r["naive_prompt_tokens"] for r in model_results)
            total_token0 = sum(r["token0_prompt_tokens"] for r in model_results)
            total_saved = total_naive - total_token0
            total_pct = (total_saved / total_naive * 100) if total_naive > 0 else 0

            print(f"\n  --- {model} Summary ---")
            print(
                f"  {'Video':<20s} {'Naive Frames':>12s} {'T0 Frames':>10s} {'Naive Tokens':>13s} {'T0 Tokens':>10s} {'Saved':>8s}"
            )
            print(f"  {'-' * 20} {'-' * 12} {'-' * 10} {'-' * 13} {'-' * 10} {'-' * 8}")
            for r in model_results:
                print(
                    f"  {r['video']:<20s} {r['naive_frames']:>12d} {r['token0_frames']:>10d} "
                    f"{r['naive_prompt_tokens']:>13,} {r['token0_prompt_tokens']:>10,} {r['savings_pct']:>7.1f}%"
                )
            print(
                f"  {'TOTAL':<20s} {'':>12s} {'':>10s} {total_naive:>13,} {total_token0:>10,} {total_pct:>7.1f}%"
            )

    # --- Grand summary across all models ---
    print(f"\n{'=' * 80}")
    print("  GRAND SUMMARY — All Models")
    print(f"{'=' * 80}")
    print(
        f"\n  {'Model':<20s} {'Naive Tokens':>13s} {'T0 Tokens':>13s} {'Saved':>13s} {'Savings':>8s}"
    )
    print(f"  {'-' * 20} {'-' * 13} {'-' * 13} {'-' * 13} {'-' * 8}")

    for model, results in all_results.items():
        if results:
            total_naive = sum(r["naive_prompt_tokens"] for r in results)
            total_token0 = sum(r["token0_prompt_tokens"] for r in results)
            total_saved = total_naive - total_token0
            pct = (total_saved / total_naive * 100) if total_naive > 0 else 0
            print(
                f"  {model:<20s} {total_naive:>13,} {total_token0:>13,} {total_saved:>13,} {pct:>7.1f}%"
            )

    print(f"\n  {'=' * 80}")


def main():
    parser = argparse.ArgumentParser(
        description="Video optimization benchmark against Ollama models"
    )
    parser.add_argument(
        "--model", action="append", help="Ollama model(s) to test (can specify multiple)"
    )
    args = parser.parse_args()

    models = args.model or DEFAULT_MODELS
    asyncio.run(run_all_benchmarks(models))


if __name__ == "__main__":
    main()
