"""
mask_pipeline.py
----------------
Reads the persisted bounding-box data for a given show_id from scene_vault.json,
extracts every frame from the source video, draws a green bounding-box rectangle
with a label around the tracked object, and encodes the annotated result as AVI.

Key design decisions:
  - Gemini returns sparse *keyframes* (typically 1/second).  We build a
    timestamp-sorted list and linearly interpolate the bbox for every video
    frame in between.
  - Gemini sometimes returns coordinates in its native [0, 1000] normalised
    space rather than actual pixel values.  We auto-detect this and
    denormalize before rendering.

Output contract:
  - Original scene is preserved (full color)
  - A green rectangle + label is drawn around the tracked object
  - Same resolution & FPS as the source video
  - Saved to  assets/masks/<show_id>_mask.avi
"""

import cv2
import json
import math
import os
from typing import Optional

VAULT_PATH = "app/db/data/scene_vault.json"
MASKS_DIR = "assets/masks"
os.makedirs(MASKS_DIR, exist_ok=True)

# Annotation style
BOX_COLOR        = (0, 255, 0)   # Lime green (BGR)
BOX_THICKNESS    = 4
FONT             = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE       = 0.8
FONT_THICKNESS   = 2
LABEL_BG_COLOR   = (0, 200, 0)
LABEL_TEXT_COLOR  = (0, 0, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_scene(show_id: str) -> Optional[dict]:
    """Return the first scene entry matching *show_id* from scene_vault.json."""
    if not os.path.exists(VAULT_PATH):
        raise FileNotFoundError(f"scene_vault.json not found at {VAULT_PATH}")

    with open(VAULT_PATH, "r") as f:
        vault = json.load(f)

    for scene in vault.get("scenes", []):
        if scene.get("show_id") == show_id:
            return scene

    return None


def _is_valid_bbox(bb) -> bool:
    """True if bb is a 4-element list with no None values."""
    return (
        bb is not None
        and isinstance(bb, list)
        and len(bb) == 4
        and all(v is not None for v in bb)
    )


def _detect_normalized(frame_bboxes: list, width: int, height: int) -> bool:
    """
    Heuristic: if ALL valid bounding-box coordinates fall within [0, 1000]
    and the video is larger than 1000px on either axis, assume the coords
    are in Gemini's normalised [0, 1000] space.
    """
    max_dim = max(width, height)
    if max_dim <= 1000:
        return False          # video fits inside 1000px — can't tell

    for entry in frame_bboxes:
        bb = entry.get("bounding_box")
        if not _is_valid_bbox(bb):
            continue
        if any(v > 1000 for v in bb):
            return False      # at least one coord exceeds 1000 → pixel space
    return True               # everything ≤ 1000 on a bigger video → normalised


def _denormalize(bb: list, width: int, height: int) -> list:
    """Convert [0, 1000] normalised coords to actual pixel coords."""
    x_min, y_min, x_max, y_max = bb
    return [
        x_min / 1000.0 * width,
        y_min / 1000.0 * height,
        x_max / 1000.0 * width,
        y_max / 1000.0 * height,
    ]


def _build_keyframes(frame_bboxes: list, width: int, height: int) -> list:
    """
    Build a sorted list of (timestamp_ms, [x_min, y_min, x_max, y_max])
    keyframes from the vault data.  Invalid / null entries are skipped.
    Coordinates are denormalized if needed.
    """
    normalised = _detect_normalized(frame_bboxes, width, height)
    if normalised:
        print("[MASK]   ↳ Detected normalised [0-1000] coords — converting to pixels")

    keyframes = []
    for entry in frame_bboxes:
        bb = entry.get("bounding_box")
        if not _is_valid_bbox(bb):
            continue
        ts = entry.get("timestamp_ms", 0)
        coords = _denormalize(bb, width, height) if normalised else list(bb)
        keyframes.append((ts, coords))

    keyframes.sort(key=lambda k: k[0])
    return keyframes


def _lerp_bbox(bb_a: list, bb_b: list, t: float) -> list:
    """Linearly interpolate between two bounding boxes.  t ∈ [0, 1]."""
    return [
        bb_a[i] + (bb_b[i] - bb_a[i]) * t
        for i in range(4)
    ]


def _bbox_at_timestamp(keyframes: list, ts_ms: float) -> Optional[list]:
    """
    Given sorted keyframes and a timestamp, return an interpolated bbox.
    - Before the first keyframe  → use the first keyframe's bbox.
    - After the last keyframe   → use the last keyframe's bbox.
    - Between two keyframes     → linear interpolation.
    """
    if not keyframes:
        return None

    # Before or at first keyframe
    if ts_ms <= keyframes[0][0]:
        return list(keyframes[0][1])

    # After or at last keyframe
    if ts_ms >= keyframes[-1][0]:
        return list(keyframes[-1][1])

    # Find the two surrounding keyframes
    for i in range(len(keyframes) - 1):
        ts_a, bb_a = keyframes[i]
        ts_b, bb_b = keyframes[i + 1]
        if ts_a <= ts_ms <= ts_b:
            span = ts_b - ts_a
            if span == 0:
                return list(bb_a)
            t = (ts_ms - ts_a) / span
            return _lerp_bbox(bb_a, bb_b, t)

    return list(keyframes[-1][1])


def _draw_annotation(frame, x_min, y_min, x_max, y_max, label: str):
    """Draw a green bounding box + filled label chip above it."""
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), BOX_COLOR, BOX_THICKNESS)

    (text_w, text_h), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
    chip_y1 = max(0, y_min - text_h - baseline - 8)
    chip_y2 = max(text_h + baseline, y_min)
    cv2.rectangle(frame, (x_min, chip_y1), (x_min + text_w + 8, chip_y2), LABEL_BG_COLOR, -1)

    cv2.putText(
        frame, label,
        (x_min + 4, chip_y2 - baseline - 2),
        FONT, FONT_SCALE, LABEL_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_mask_video(show_id: str) -> str:
    """
    Reads scene_vault.json for show_id, draws per-frame green bounding boxes
    on the original video using timestamp-based interpolation, and saves the
    annotated output.

    Returns the absolute path to the written file.
    """
    # ── 1. Load scene metadata ────────────────────────────────────────────────
    scene = _load_scene(show_id)
    if scene is None:
        raise ValueError(
            f"No scene with show_id='{show_id}' found in scene_vault.json. "
            "Ingest the video first via POST /ingest/video."
        )

    video_path = scene.get("filepath")
    if not video_path or not os.path.exists(video_path):
        raise FileNotFoundError(
            f"Source video '{video_path}' for show_id='{show_id}' not found on disk."
        )

    frame_bboxes = scene.get("frame_bounding_boxes", [])
    if not frame_bboxes:
        raise ValueError(f"No frame_bounding_boxes for show_id='{show_id}'.")

    target_object = scene.get("target_object", "object").upper()

    # ── 2. Open source video ──────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ── 3. Build timestamp-indexed keyframes ──────────────────────────────────
    keyframes = _build_keyframes(frame_bboxes, width, height)
    print(
        f"[MASK] {show_id} | {total_frames} frames | {width}x{height} "
        f"@ {fps:.1f} fps | {len(keyframes)} keyframes | object: {target_object}"
    )

    if not keyframes:
        cap.release()
        raise ValueError(
            f"All bounding boxes for show_id='{show_id}' are null — nothing to annotate."
        )

    # ── 4. Set up output writer (XVID + AVI — most reliable on Windows) ───────
    output_path = os.path.join(MASKS_DIR, f"{show_id}_mask.avi")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"OpenCV VideoWriter failed for: {output_path}")

    # ── 5. Frame-by-frame annotation via timestamp interpolation ──────────────
    frame_idx      = 0
    rendered_count = 0
    skipped_count  = 0
    ms_per_frame   = 1000.0 / fps

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ts_ms = frame_idx * ms_per_frame
        bbox  = _bbox_at_timestamp(keyframes, ts_ms)

        if bbox is not None:
            x_min = max(0, int(round(bbox[0])))
            y_min = max(0, int(round(bbox[1])))
            x_max = min(width  - 1, int(round(bbox[2])))
            y_max = min(height - 1, int(round(bbox[3])))

            if x_max > x_min and y_max > y_min:
                _draw_annotation(frame, x_min, y_min, x_max, y_max, target_object)
                rendered_count += 1
            else:
                skipped_count += 1
        else:
            skipped_count += 1

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    print(f"[MASK] Done -> {output_path} | annotated={rendered_count} | blank={skipped_count}")
    return os.path.abspath(output_path)
