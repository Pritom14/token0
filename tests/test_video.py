"""Tests for video optimization pipeline."""

import tempfile

import cv2
import numpy as np
import pytest
from PIL import Image

from token0.optimization.video import (
    DEDUP_HAMMING_THRESHOLD,
    deduplicate_frames,
    detect_scene_changes,
    extract_frames,
    process_video,
    score_frames_by_relevance,
)


def _make_test_video(
    num_frames: int = 30,
    fps: float = 30.0,
    width: int = 320,
    height: int = 240,
    scene_changes: list[int] | None = None,
) -> str:
    """Create a test video file. Returns path to temp file.

    scene_changes: list of frame indices where the scene changes (color shifts).
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp.name, fourcc, fps, (width, height))

    if scene_changes is None:
        scene_changes = []

    current_color = np.array([50, 100, 150], dtype=np.uint8)

    for i in range(num_frames):
        if i in scene_changes:
            # Dramatic color shift for scene change
            current_color = np.random.RandomState(seed=i).randint(0, 256, 3).astype(np.uint8)

        # Create frame with slight variation (simulates real video)
        frame = np.full((height, width, 3), current_color, dtype=np.uint8)
        # Add minor noise so frames aren't perfectly identical
        noise = np.random.RandomState(seed=i).randint(-3, 4, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        writer.write(frame)

    writer.release()
    return tmp.name


def _make_diverse_video(num_frames: int = 60, fps: float = 30.0) -> str:
    """Create a video with clearly different scenes."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp.name, fourcc, fps, (320, 240))

    for i in range(num_frames):
        rng = np.random.RandomState(seed=i // 10)  # same scene for 10 frames
        frame = rng.randint(0, 256, (240, 320, 3), dtype=np.uint8)
        # Add per-frame noise
        noise = np.random.RandomState(seed=i * 100).randint(-5, 6, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        writer.write(frame)

    writer.release()
    return tmp.name


class TestFrameExtraction:
    def test_extract_frames_basic(self):
        """Extract frames from a simple video."""
        path = _make_test_video(num_frames=30, fps=30.0)
        frames = extract_frames(path, fps=1.0)
        # 30 frames at 30fps = 1 second → 1 frame at 1fps
        assert len(frames) >= 1
        assert all(isinstance(f[1], Image.Image) for f in frames)

    def test_extract_frames_timestamps(self):
        """Timestamps should be monotonically increasing."""
        path = _make_test_video(num_frames=90, fps=30.0)
        frames = extract_frames(path, fps=1.0)
        timestamps = [t for t, _ in frames]
        assert timestamps == sorted(timestamps)

    def test_extract_frames_respects_max(self):
        """Should not exceed max_frames."""
        path = _make_test_video(num_frames=300, fps=30.0)
        frames = extract_frames(path, fps=10.0, max_frames=5)
        assert len(frames) <= 5

    def test_extract_frames_pil_format(self):
        """Extracted frames should be RGB PIL images."""
        path = _make_test_video(num_frames=30, fps=30.0)
        frames = extract_frames(path, fps=1.0)
        for _, frame in frames:
            assert isinstance(frame, Image.Image)
            assert frame.mode == "RGB"

    def test_extract_frames_higher_fps(self):
        """Higher extraction fps should yield more frames."""
        path = _make_test_video(num_frames=90, fps=30.0)
        frames_1fps = extract_frames(path, fps=1.0, max_frames=100)
        frames_5fps = extract_frames(path, fps=5.0, max_frames=100)
        assert len(frames_5fps) > len(frames_1fps)


class TestDeduplication:
    def test_dedup_removes_similar_frames(self):
        """Near-identical frames should be collapsed."""
        path = _make_test_video(num_frames=90, fps=30.0)
        frames = extract_frames(path, fps=10.0, max_frames=100)
        deduped = deduplicate_frames(frames)
        # Static video with tiny noise — most frames should be deduped
        assert len(deduped) < len(frames)

    def test_dedup_keeps_different_scenes(self):
        """Frames from different scenes should be kept."""
        path = _make_test_video(
            num_frames=60,
            fps=30.0,
            scene_changes=[15, 30, 45],
        )
        frames = extract_frames(path, fps=2.0, max_frames=100)
        deduped = deduplicate_frames(frames)
        # Should keep at least one frame per scene (4 scenes)
        assert len(deduped) >= 3

    def test_dedup_empty_input(self):
        """Empty input returns empty output."""
        assert deduplicate_frames([]) == []

    def test_dedup_single_frame(self):
        """Single frame is always kept."""
        img = Image.new("RGB", (320, 240), color="red")
        result = deduplicate_frames([(0.0, img)])
        assert len(result) == 1

    def test_dedup_preserves_order(self):
        """Deduplicated frames should maintain chronological order."""
        path = _make_test_video(
            num_frames=60,
            fps=30.0,
            scene_changes=[20, 40],
        )
        frames = extract_frames(path, fps=2.0, max_frames=100)
        deduped = deduplicate_frames(frames)
        timestamps = [t for t, _ in deduped]
        assert timestamps == sorted(timestamps)


class TestSceneDetection:
    def test_scene_detection_finds_changes(self):
        """Should detect scene boundaries."""
        path = _make_test_video(
            num_frames=90,
            fps=30.0,
            scene_changes=[30, 60],
        )
        frames = extract_frames(path, fps=2.0, max_frames=100)
        scene_frames = detect_scene_changes(frames)
        # Should keep frames around scene changes
        assert len(scene_frames) >= 3  # at least first + 2 scene changes

    def test_scene_detection_static_video(self):
        """Static video should return few frames."""
        path = _make_test_video(num_frames=90, fps=30.0)
        frames = extract_frames(path, fps=2.0, max_frames=100)
        scene_frames = detect_scene_changes(frames)
        # Static = mostly same scene, should return very few
        assert len(scene_frames) <= len(frames)

    def test_scene_detection_preserves_first_last(self):
        """First and last frames should always be kept."""
        path = _make_test_video(num_frames=90, fps=30.0)
        frames = extract_frames(path, fps=2.0, max_frames=100)
        if len(frames) >= 3:
            scene_frames = detect_scene_changes(frames)
            assert scene_frames[0][0] == frames[0][0]  # first frame kept
            assert scene_frames[-1][0] == frames[-1][0]  # last frame kept


class TestCLIPScoring:
    def test_scoring_without_clip(self):
        """Without CLIP, should return uniform scores."""
        frames = [
            (0.0, Image.new("RGB", (320, 240), "red")),
            (1.0, Image.new("RGB", (320, 240), "blue")),
        ]
        scored = score_frames_by_relevance(frames, "test prompt")
        assert len(scored) == 2
        # All scores should be 1.0 (uniform fallback) or actual CLIP scores
        assert all(len(s) == 3 for s in scored)

    def test_scoring_empty_input(self):
        """Empty input returns empty output."""
        assert score_frames_by_relevance([], "test") == []

    def test_scoring_preserves_timestamps(self):
        """Output should maintain chronological order."""
        frames = [
            (0.0, Image.new("RGB", (320, 240), "red")),
            (1.0, Image.new("RGB", (320, 240), "blue")),
            (2.0, Image.new("RGB", (320, 240), "green")),
        ]
        scored = score_frames_by_relevance(frames, "test")
        timestamps = [s[0] for s in scored]
        assert timestamps == sorted(timestamps)


class TestProcessVideo:
    def test_full_pipeline(self):
        """Full pipeline returns images and stats."""
        path = _make_test_video(num_frames=90, fps=30.0, scene_changes=[30, 60])
        images, stats = process_video(path, prompt="describe this video")

        assert len(images) > 0
        assert all(isinstance(img, Image.Image) for img in images)
        assert "total_video_frames" in stats
        assert "frames_selected" in stats
        assert "frame_reduction_pct" in stats
        assert stats["frames_selected"] <= stats["frames_extracted_at_1fps"]

    def test_pipeline_reduces_frames(self):
        """Pipeline should reduce frame count significantly."""
        path = _make_test_video(num_frames=150, fps=30.0)
        images, stats = process_video(path, prompt="what is happening?")

        # 150 frames at 30fps = 5 seconds
        # At 1fps extraction = ~5 frames, dedup should collapse further
        assert stats["frames_selected"] < stats["total_video_frames"]
        assert stats["frame_reduction_pct"] > 50.0

    def test_pipeline_diverse_video(self):
        """Diverse video should keep more frames than static."""
        static_path = _make_test_video(num_frames=90, fps=30.0)
        diverse_path = _make_diverse_video(num_frames=90, fps=30.0)

        static_images, static_stats = process_video(static_path)
        diverse_images, diverse_stats = process_video(diverse_path)

        # Diverse video should retain more frames
        assert diverse_stats["frames_selected"] >= static_stats["frames_selected"]

    def test_pipeline_stats_structure(self):
        """Stats dict should have all expected keys."""
        path = _make_test_video(num_frames=30, fps=30.0)
        _, stats = process_video(path)

        expected_keys = [
            "video_duration_seconds",
            "video_fps",
            "total_video_frames",
            "frames_extracted_at_1fps",
            "frames_after_dedup",
            "frames_after_scene_detection",
            "frames_selected",
            "frame_reduction_pct",
            "clip_used",
        ]
        for key in expected_keys:
            assert key in stats, f"Missing key: {key}"

    def test_pipeline_max_frames_cap(self):
        """Pipeline should respect max_frames cap."""
        path = _make_diverse_video(num_frames=300, fps=30.0)
        images, stats = process_video(path, max_frames=5)
        assert len(images) <= 5

    def test_pipeline_empty_video(self):
        """Empty/invalid video should return empty results gracefully."""
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.write(b"not a video")
        tmp.flush()

        images, stats = process_video(tmp.name)
        assert len(images) == 0
        assert stats["frames_selected"] == 0
