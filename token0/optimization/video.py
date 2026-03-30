"""Video optimization — extract keyframes, deduplicate, and optimize.

Layer 1: Frame sampling + perceptual hash dedup (no ML dependencies)
Layer 2: CLIP-based query-frame relevance scoring (optional, needs sentence-transformers)

Accepts video as base64 data URI or file path. Extracts keyframes,
deduplicates similar frames using QJL fuzzy matching, optionally scores
frames against the user's prompt, and returns optimized PIL images.
"""

import base64
import logging
import tempfile

import cv2
import numpy as np
from PIL import Image

from token0.optimization.cache import (
    _hamming_distance,
    _image_hash,
    _jl_compress,
)

logger = logging.getLogger("token0.video")

# Try to import CLIP for Layer 2 (optional)
_clip_model = None
_clip_preprocess = None
_clip_available = False

try:
    import clip
    import torch

    _clip_available = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_FPS = 1  # extract 1 frame per second
MAX_FRAMES = 32  # hard cap on frames sent to LLM
DEDUP_HAMMING_THRESHOLD = 12  # tighter than cache (more aggressive dedup for consecutive frames)
MIN_SCENE_CHANGE_THRESHOLD = 15.0  # minimum pixel difference for scene change


def _decode_video_input(video_input: str) -> str:
    """Decode video input to a temp file path. Accepts base64 data URI or file path."""
    if video_input.startswith("data:"):
        # data:video/mp4;base64,...
        _, b64_data = video_input.split(",", 1)
        raw_bytes = base64.b64decode(b64_data)
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.write(raw_bytes)
        tmp.flush()
        return tmp.name
    elif video_input.startswith(("http://", "https://")):
        # URL — download first
        import httpx

        resp = httpx.get(video_input, timeout=30)
        resp.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.write(resp.content)
        tmp.flush()
        return tmp.name
    else:
        # Assume file path
        return video_input


def extract_frames(
    video_path: str,
    fps: float = DEFAULT_FPS,
    max_frames: int = MAX_FRAMES,
) -> list[tuple[float, Image.Image]]:
    """Extract frames from video at given fps.

    Returns list of (timestamp_seconds, PIL.Image) tuples.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("Cannot open video: %s", video_path)
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0

    # Calculate frame interval
    frame_interval = max(1, int(video_fps / fps))

    frames = []
    frame_idx = 0

    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            # Convert BGR (OpenCV) to RGB (PIL)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            timestamp = frame_idx / video_fps
            frames.append((timestamp, pil_image))

        frame_idx += 1

    cap.release()

    logger.info(
        "extracted %d frames from %.1fs video (%.1f fps, interval=%d)",
        len(frames),
        duration,
        video_fps,
        frame_interval,
    )
    return frames


def deduplicate_frames(
    frames: list[tuple[float, Image.Image]],
    hamming_threshold: int = DEDUP_HAMMING_THRESHOLD,
) -> list[tuple[float, Image.Image]]:
    """Remove near-duplicate frames using QJL perceptual hash signatures.

    Keeps the first frame from each group of similar consecutive frames.
    """
    if not frames:
        return []

    kept = [frames[0]]
    prev_sig = _jl_compress(_image_hash(frames[0][1]))

    for timestamp, frame in frames[1:]:
        sig = _jl_compress(_image_hash(frame))
        dist = _hamming_distance(prev_sig, sig)

        if dist > hamming_threshold:
            kept.append((timestamp, frame))
            prev_sig = sig
        else:
            logger.debug(
                "dedup: dropped frame at %.1fs (hamming=%d, threshold=%d)",
                timestamp,
                dist,
                hamming_threshold,
            )

    logger.info(
        "dedup: %d → %d frames (removed %d duplicates)",
        len(frames),
        len(kept),
        len(frames) - len(kept),
    )
    return kept


def detect_scene_changes(
    frames: list[tuple[float, Image.Image]],
    threshold: float = MIN_SCENE_CHANGE_THRESHOLD,
) -> list[tuple[float, Image.Image]]:
    """Detect scene changes by pixel-level difference between consecutive frames.

    Returns frames at scene boundaries + first and last frame.
    """
    if len(frames) <= 2:
        return frames

    kept = [frames[0]]  # always keep first frame

    for i in range(1, len(frames)):
        prev_arr = np.array(frames[i - 1][1].resize((160, 120))).astype(np.float32)
        curr_arr = np.array(frames[i][1].resize((160, 120))).astype(np.float32)
        diff = np.mean(np.abs(curr_arr - prev_arr))

        if diff > threshold:
            kept.append(frames[i])

    # Always keep last frame
    if kept[-1][0] != frames[-1][0]:
        kept.append(frames[-1])

    logger.info(
        "scene detection: %d → %d frames",
        len(frames),
        len(kept),
    )
    return kept


def _load_clip_model():
    """Lazy-load CLIP model on first use."""
    global _clip_model, _clip_preprocess
    if _clip_model is None and _clip_available:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=device)
        logger.info("CLIP model loaded on %s", device)


def score_frames_by_relevance(
    frames: list[tuple[float, Image.Image]],
    prompt: str,
    top_k: int | None = None,
) -> list[tuple[float, Image.Image, float]]:
    """Score frames by relevance to the prompt using CLIP.

    Returns list of (timestamp, frame, score) sorted by score descending.
    Falls back to uniform scoring if CLIP is not available.
    """
    if not frames:
        return []

    if not _clip_available:
        # Fallback: uniform scores, keep all
        logger.debug("CLIP not available, using uniform frame scores")
        return [(t, f, 1.0) for t, f in frames]

    _load_clip_model()
    device = next(_clip_model.parameters()).device

    # Encode prompt
    text_tokens = clip.tokenize([prompt]).to(device)
    with torch.no_grad():
        text_features = _clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Encode frames
    scored = []
    for timestamp, frame in frames:
        image_input = _clip_preprocess(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = _clip_model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        similarity = (text_features @ image_features.T).item()
        scored.append((timestamp, frame, similarity))

    # Sort by score descending
    scored.sort(key=lambda x: x[2], reverse=True)

    if top_k is not None:
        scored = scored[:top_k]

    # Re-sort by timestamp for chronological order
    scored.sort(key=lambda x: x[0])

    return scored


def process_video(
    video_input: str,
    prompt: str = "",
    fps: float = DEFAULT_FPS,
    max_frames: int = MAX_FRAMES,
    deduplicate: bool = True,
    use_scene_detection: bool = True,
    use_clip_scoring: bool = True,
) -> tuple[list[Image.Image], dict]:
    """Full video optimization pipeline.

    Returns (list of optimized PIL images, stats dict).

    Pipeline:
    1. Decode video input (base64/URL/path) → temp file
    2. Extract frames at configured fps
    3. Deduplicate similar frames (QJL perceptual hash)
    4. Detect scene changes
    5. Score by prompt relevance (CLIP, if available)
    6. Cap at max_frames
    """
    # Step 1: Decode input
    video_path = _decode_video_input(video_input)

    # Step 2: Extract frames
    all_frames = extract_frames(video_path, fps=fps, max_frames=max_frames * 4)
    total_extracted = len(all_frames)

    if not all_frames:
        return [], {
            "total_video_frames": 0,
            "frames_extracted": 0,
            "frames_after_dedup": 0,
            "frames_after_scene_detection": 0,
            "frames_selected": 0,
            "frame_reduction_pct": 0.0,
            "clip_used": False,
        }

    # Get video metadata
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_video_frames / video_fps if video_fps > 0 else 0
    cap.release()

    # Step 3: Deduplicate
    if deduplicate:
        frames = deduplicate_frames(all_frames)
    else:
        frames = all_frames
    frames_after_dedup = len(frames)

    # Step 4: Scene detection
    if use_scene_detection and len(frames) > max_frames:
        frames = detect_scene_changes(frames)
    frames_after_scene = len(frames)

    # Step 5: CLIP scoring (trim to max_frames by relevance)
    clip_used = False
    if use_clip_scoring and _clip_available and prompt and len(frames) > max_frames:
        scored = score_frames_by_relevance(frames, prompt, top_k=max_frames)
        frames = [(t, f) for t, f, s in scored]
        clip_used = True
    elif len(frames) > max_frames:
        # Uniform sampling to max_frames
        indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = [frames[i] for i in indices]

    # Extract just the PIL images (drop timestamps)
    images = [f for _, f in frames]

    stats = {
        "video_duration_seconds": round(duration, 1),
        "video_fps": round(video_fps, 1),
        "total_video_frames": total_video_frames,
        "frames_extracted_at_1fps": total_extracted,
        "frames_after_dedup": frames_after_dedup,
        "frames_after_scene_detection": frames_after_scene,
        "frames_selected": len(images),
        "frame_reduction_pct": round((1 - len(images) / max(total_video_frames, 1)) * 100, 1),
        "clip_used": clip_used,
    }

    logger.info(
        "video pipeline: %d total → %d extracted → %d deduped → %d selected (%.1f%% reduction)",
        total_video_frames,
        total_extracted,
        frames_after_dedup,
        len(images),
        stats["frame_reduction_pct"],
    )

    return images, stats
