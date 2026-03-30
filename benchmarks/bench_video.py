"""Benchmark: Video optimization pipeline.

Demonstrates token savings from video frame extraction, deduplication,
and scene detection vs naive frame-by-frame approach.

Usage:
    python -m benchmarks.bench_video
"""

import time

import cv2
import numpy as np

from token0.optimization.video import (
    deduplicate_frames,
    detect_scene_changes,
    extract_frames,
    process_video,
)

# GPT-4o token costs
TOKENS_PER_FRAME_HIGH = 765  # high detail, typical 720p frame
TOKENS_PER_FRAME_LOW = 85  # low detail
COST_PER_TOKEN = 2.50 / 1_000_000


def _create_benchmark_video(
    duration_seconds: int = 30,
    fps: float = 30.0,
    width: int = 640,
    height: int = 480,
    num_scenes: int = 5,
) -> str:
    """Create a realistic benchmark video with multiple scenes."""
    import tempfile

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp.name, fourcc, fps, (width, height))

    total_frames = int(duration_seconds * fps)
    frames_per_scene = total_frames // num_scenes

    for scene_idx in range(num_scenes):
        # Each scene has a unique base pattern
        rng = np.random.RandomState(seed=scene_idx * 42)
        base = rng.randint(0, 256, (height, width, 3), dtype=np.uint8)

        for frame_idx in range(frames_per_scene):
            # Add slight per-frame variation (simulates camera movement, lighting)
            noise_rng = np.random.RandomState(seed=scene_idx * 10000 + frame_idx)
            noise = noise_rng.randint(-5, 6, base.shape, dtype=np.int16)
            frame = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            writer.write(frame)

    writer.release()
    return tmp.name


def run_benchmark():
    print("=" * 80)
    print("  Video Optimization Benchmark")
    print("=" * 80)

    configs = [
        {"duration": 10, "scenes": 3, "label": "10s video, 3 scenes"},
        {"duration": 30, "scenes": 5, "label": "30s video, 5 scenes"},
        {"duration": 60, "scenes": 8, "label": "60s video, 8 scenes"},
    ]

    for config in configs:
        print(f"\n  --- {config['label']} ---\n")

        # Create video
        path = _create_benchmark_video(
            duration_seconds=config["duration"],
            num_scenes=config["scenes"],
        )

        total_frames = int(config["duration"] * 30)

        # Naive approach: send every frame at 1fps
        start = time.time()
        naive_frames = extract_frames(path, fps=1.0, max_frames=1000)
        naive_time = time.time() - start
        naive_tokens = len(naive_frames) * TOKENS_PER_FRAME_HIGH

        # Token0 approach: full pipeline
        start = time.time()
        optimized_images, stats = process_video(
            path,
            prompt="describe what happens in this video",
            max_frames=32,
        )
        token0_time = time.time() - start
        token0_tokens = len(optimized_images) * TOKENS_PER_FRAME_HIGH

        # Full naive (every frame)
        full_naive_tokens = total_frames * TOKENS_PER_FRAME_HIGH

        print(f"  Total video frames (30fps):     {total_frames}")
        print(f"  Naive 1fps frames:              {len(naive_frames)}")
        print(f"  Token0 optimized frames:        {stats['frames_selected']}")
        print(f"    - After extraction (1fps):    {stats['frames_extracted_at_1fps']}")
        print(f"    - After dedup:                {stats['frames_after_dedup']}")
        print(f"    - After scene detection:      {stats['frames_after_scene_detection']}")
        print()

        # Token comparison
        print(f"  {'Method':<35s} {'Frames':>8s} {'Tokens':>10s} {'Cost':>10s}")
        print(f"  {'-'*35} {'-'*8} {'-'*10} {'-'*10}")
        print(
            f"  {'All frames (30fps)':<35s} {total_frames:>8d} "
            f"{full_naive_tokens:>10,} ${full_naive_tokens * COST_PER_TOKEN:>9.4f}"
        )
        print(
            f"  {'Naive 1fps':<35s} {len(naive_frames):>8d} "
            f"{naive_tokens:>10,} ${naive_tokens * COST_PER_TOKEN:>9.4f}"
        )
        print(
            f"  {'Token0 (dedup + scene detect)':<35s} {len(optimized_images):>8d} "
            f"{token0_tokens:>10,} ${token0_tokens * COST_PER_TOKEN:>9.4f}"
        )

        # Savings
        savings_vs_30fps = (1 - token0_tokens / full_naive_tokens) * 100
        savings_vs_1fps = (1 - token0_tokens / naive_tokens) * 100 if naive_tokens > 0 else 0

        print()
        print(f"  Savings vs 30fps:               {savings_vs_30fps:.1f}%")
        print(f"  Savings vs naive 1fps:          {savings_vs_1fps:.1f}%")
        print(f"  Processing time:                {token0_time * 1000:.0f}ms")

    # Scale projections
    print(f"\n  {'='*70}")
    print("  SCALE PROJECTIONS (GPT-4o, 60s videos)")
    print(f"  {'='*70}")
    print()

    # Use 60s video stats
    path = _create_benchmark_video(duration_seconds=60, num_scenes=8)
    _, stats = process_video(path, prompt="describe this video", max_frames=32)

    frames_per_video_naive = 60  # 1fps
    frames_per_video_token0 = stats["frames_selected"]

    for daily_videos in [100, 1_000, 10_000]:
        naive_daily_tokens = daily_videos * frames_per_video_naive * TOKENS_PER_FRAME_HIGH
        token0_daily_tokens = daily_videos * frames_per_video_token0 * TOKENS_PER_FRAME_HIGH
        naive_monthly = naive_daily_tokens * 30 * COST_PER_TOKEN
        token0_monthly = token0_daily_tokens * 30 * COST_PER_TOKEN
        savings = naive_monthly - token0_monthly

        print(f"  {daily_videos:,} videos/day:")
        print(f"    Naive 1fps:   ${naive_monthly:>10,.2f}/month")
        print(f"    Token0:       ${token0_monthly:>10,.2f}/month")
        print(f"    Savings:      ${savings:>10,.2f}/month")
        print()

    print(f"  Token0 frames per 60s video: {frames_per_video_token0}")
    print(f"  Reduction: {frames_per_video_naive} → {frames_per_video_token0} frames")
    print(f"  {'='*70}")


if __name__ == "__main__":
    run_benchmark()
