import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    from sam3 import Sam3Processor
except ImportError:  # pragma: no cover
    Sam3Processor = None

DATA_DIR = "app/db/data"
MASKS_DIR = "assets/masks"
SAM3_MODEL_PATH = os.getenv("SAM3_MODEL_PATH", "meta/sam3-optimized")


def load_scene_metadata(show_id: str) -> Dict:
    """Load scene data from app/db/data/{show_id}.json"""
    filepath = os.path.join(DATA_DIR, f"{show_id}.json")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Scene metadata not found at {filepath}")

    with open(filepath, "r", encoding="utf-8") as handle:
        scene = json.load(handle)
        
    return scene


def initialize_sam_processor() -> Optional[object]:
    if Sam3Processor is None:
        return None

    return Sam3Processor.from_pretrained(SAM3_MODEL_PATH)


def ensure_output_dirs(show_id: str) -> str:
    alpha_dir = os.path.join(MASKS_DIR, show_id, "alpha")
    os.makedirs(alpha_dir, exist_ok=True)
    return alpha_dir


# Target processing width.
# 720px gives a good balance of speed and detail for masking.
# Masks are upscaled back to original resolution before saving.
PROCESS_WIDTH = 720


def read_video_frames(video_path: str, target_width: int = PROCESS_WIDTH):
    """Read all frames from video, auto-downscaling if the source is wider than target_width.
    
    Returns:
        frames       - list of BGR numpy arrays at processing resolution
        fps          - original frame rate
        orig_width   - original pixel width  (for output mask upscale)
        orig_height  - original pixel height (for output mask upscale)
        proc_width   - width actually used for processing
        proc_height  - height actually used for processing
        scale_x      - orig_width / proc_width  (for bbox remapping)
        scale_y      - orig_height / proc_height
    """
    if cv2 is None:
        raise RuntimeError("opencv-python is required for video frame processing. Please install opencv-python.")

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise IOError(f"Unable to open video file: {video_path}")

    fps         = capture.get(cv2.CAP_PROP_FPS) or 30.0
    orig_width  = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Compute downscale dimensions while preserving aspect ratio
    if orig_width > target_width:
        scale        = target_width / orig_width
        proc_width   = target_width
        proc_height  = int(orig_height * scale)
        print(f"[MASK] Downscaling from {orig_width}x{orig_height} to {proc_width}x{proc_height} for processing.")
    else:
        proc_width   = orig_width
        proc_height  = orig_height
        print(f"[MASK] Source is {orig_width}x{orig_height}; no downscale needed.")

    scale_x = orig_width  / proc_width
    scale_y = orig_height / proc_height

    frames = []
    while True:
        success, frame = capture.read()
        if not success:
            break
        if proc_width != orig_width:
            frame = cv2.resize(frame, (proc_width, proc_height), interpolation=cv2.INTER_AREA)
        frames.append(frame)

    capture.release()
    return frames, fps, orig_width, orig_height, proc_width, proc_height, scale_x, scale_y


def normalize_anchor_mask(mask: np.ndarray, frame_height: int, frame_width: int) -> np.ndarray:
    if mask.shape != (frame_height, frame_width):
        mask_image = Image.fromarray(mask)
        mask_image = mask_image.resize((frame_width, frame_height), Image.NEAREST)
        mask = np.array(mask_image)

    normalized = np.where(mask > 127, 255, 0).astype(np.uint8)
    return normalized


def load_manual_mask(mask_path: str, frame_height: int, frame_width: int) -> np.ndarray:
    mask_image = Image.open(mask_path).convert("L")
    mask_array = np.array(mask_image)
    return normalize_anchor_mask(mask_array, frame_height, frame_width)


def bbox_to_mask(bbox, frame_height: int, frame_width: int) -> np.ndarray:
    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    if bbox and len(bbox) == 4 and all(v is not None for v in bbox) and all(isinstance(v, (int, float)) for v in bbox):
        x0, y0, x1, y1 = [int(v) for v in bbox]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(frame_width, x1), min(frame_height, y1)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 255
    return mask


def _remap_bbox_to_processing_resolution(bbox: list, proc_width: int, proc_height: int, scale_x: float, scale_y: float) -> Optional[list]:
    if not bbox or len(bbox) != 4 or any(v is None for v in bbox):
        return None
    x0, y0, x1, y1 = bbox
    mapped = [
        max(0, min(int(x0 / scale_x), proc_width - 1)),
        max(0, min(int(y0 / scale_y), proc_height - 1)),
        max(0, min(int(x1 / scale_x), proc_width - 1)),
        max(0, min(int(y1 / scale_y), proc_height - 1)),
    ]
    if mapped[2] <= mapped[0] or mapped[3] <= mapped[1]:
        return None
    return mapped


def _interpolate_bbox(left: list, right: list, ratio: float) -> list:
    return [int(round(left[i] + (right[i] - left[i]) * ratio)) for i in range(4)]


def _detection_bboxes_by_frame(scene: Dict, frame_count: int, proc_width: int, proc_height: int, scale_x: float, scale_y: float) -> Optional[List[Optional[list]]]:
    detection_data = scene.get("second_by_second_detection") or {}
    detections = detection_data.get("detections") or []
    keyframes = []

    for item in detections:
        if not item.get("found") or not item.get("bbox"):
            continue
        bbox = _remap_bbox_to_processing_resolution(item["bbox"], proc_width, proc_height, scale_x, scale_y)
        if bbox is None:
            continue
        frame_index = int(item.get("frame_index", 0))
        if 0 <= frame_index < frame_count:
            keyframes.append((frame_index, bbox))

    if not keyframes:
        return None

    keyframes.sort(key=lambda item: item[0])
    frame_bboxes: List[Optional[list]] = [None] * frame_count

    for idx, (frame_index, bbox) in enumerate(keyframes):
        frame_bboxes[frame_index] = bbox

        if idx == 0:
            continue

        prev_frame, prev_bbox = keyframes[idx - 1]
        gap = frame_index - prev_frame
        if gap <= 1:
            continue

        for frame in range(prev_frame + 1, frame_index):
            ratio = (frame - prev_frame) / float(gap)
            frame_bboxes[frame] = _interpolate_bbox(prev_bbox, bbox, ratio)

    first_frame, first_bbox = keyframes[0]
    for frame in range(first_frame, frame_count):
        if frame_bboxes[frame] is None:
            frame_bboxes[frame] = first_bbox if frame == first_frame else frame_bboxes[frame - 1]

    return frame_bboxes


def _render_detection_bbox_masks(
    show_id: str,
    video_path: str,
    frames: List[np.ndarray],
    fps: float,
    orig_width: int,
    orig_height: int,
    proc_width: int,
    proc_height: int,
    alpha_dir: str,
    mask_video_path: str,
    preview_video_path: str,
    frame_bboxes: List[Optional[list]],
) -> Dict[str, object]:
    output_masks = []
    for bbox in frame_bboxes:
        if bbox is None:
            mask = np.zeros((proc_height, proc_width), dtype=np.uint8)
        else:
            mask = bbox_to_mask(bbox, proc_height, proc_width)
        if orig_width != proc_width:
            mask = cv2.resize(mask, (orig_width, orig_height), interpolation=cv2.INTER_NEAREST)
        output_masks.append(mask)

    frame_urls: List[str] = []
    for idx, mask in enumerate(output_masks):
        out_path = os.path.join(alpha_dir, f"frame_{idx:04d}.png")
        save_alpha_mask(mask, out_path)
        frame_urls.append(f"/masks/{show_id}/alpha/frame_{idx:04d}.png")

    if orig_width != proc_width:
        cap = cv2.VideoCapture(video_path)
        orig_frames = []
        while True:
            ok, frm = cap.read()
            if not ok:
                break
            orig_frames.append(frm)
        cap.release()
    else:
        orig_frames = frames

    render_mask_video(alpha_dir, mask_video_path, fps, orig_width, orig_height)
    render_overlay_video(orig_frames, output_masks, preview_video_path, fps, orig_width, orig_height)
    print(f"[MASK] Detection-driven whole-car preview -> {preview_video_path}")

    return {
        "show_id": show_id,
        "frame_count": len(output_masks),
        "mask_directory": alpha_dir,
        "mask_video": mask_video_path,
        "frame_urls": frame_urls,
        "preview_url": f"/masks/{show_id}_preview.mp4",
    }


# ── Thresholds (tune here) ────────────────────────────────────────────────────
MIN_MASK_PIXELS     = 80    # mask with fewer active pixels is treated as "lost"
CUT_THRESHOLD       = 0.50  # histogram correlation below this → hard scene cut
APPEARANCE_THRESH   = 0.15  # colour similarity below this → mask drifted to wrong object (loosened for orange/red products)
BLACK_ROW_THRESH    = 12    # rows/cols with mean brightness below this are letterbox bars


# ── Letterbox helpers ─────────────────────────────────────────────────────────

def detect_content_bounds(frame: np.ndarray) -> tuple:
    """
    Return (top, bottom, left, right) — the pixel bounds of actual video content,
    excluding solid-black letterbox / pillarbox bars on all four sides.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    row_means = gray.mean(axis=1)
    col_means = gray.mean(axis=0)

    content_rows = np.where(row_means > BLACK_ROW_THRESH)[0]
    content_cols = np.where(col_means > BLACK_ROW_THRESH)[0]

    top    = int(content_rows[0])  if len(content_rows) else 0
    bottom = int(content_rows[-1]) if len(content_rows) else frame.shape[0] - 1
    left   = int(content_cols[0])  if len(content_cols) else 0
    right  = int(content_cols[-1]) if len(content_cols) else frame.shape[1] - 1
    return top, bottom, left, right


def apply_letterbox_exclusion(mask: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """
    Zero out mask pixels that fall in any black bar (top/bottom/left/right).
    """
    top, bottom, left, right = detect_content_bounds(frame)
    cleaned = mask.copy()
    cleaned[:top, :]       = 0   # top bar
    cleaned[bottom + 1:, :] = 0  # bottom bar
    cleaned[:, :left]       = 0  # left bar
    cleaned[:, right + 1:]  = 0  # right bar
    return cleaned


# ── Shot-cut detection ────────────────────────────────────────────────────────

def _frame_histogram(frame: np.ndarray) -> np.ndarray:
    """Return a normalised HSV hue histogram for the frame."""
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [64], [0, 180])
    cv2.normalize(hist, hist)
    return hist


def is_shot_cut(prev_frame: np.ndarray, next_frame: np.ndarray) -> bool:
    """
    Return True if there is a hard scene cut between prev_frame and next_frame.
    Uses HSV hue histogram correlation — drops sharply on a cut.
    """
    h1 = _frame_histogram(prev_frame)
    h2 = _frame_histogram(next_frame)
    correlation = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
    return bool(correlation < CUT_THRESHOLD)


# ── Appearance-match validation ───────────────────────────────────────────────

def _masked_histogram(frame: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    """Return a normalised HSV hue histogram for the masked region only."""
    if mask is None or int(mask.sum() / 255) < MIN_MASK_PIXELS:
        return None
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], mask, [64], [0, 180])
    cv2.normalize(hist, hist)
    return hist


def is_same_object(
    current_frame: np.ndarray,
    current_mask: np.ndarray,
    anchor_hist: np.ndarray,
) -> bool:
    """
    Compare the colour appearance of the current masked region against the
    anchor frame's product histogram.  Returns False if the mask has drifted
    onto a person, furniture, or other unrelated object.
    """
    if anchor_hist is None:
        return True  # no reference → can't reject
    candidate_hist = _masked_histogram(current_frame, current_mask)
    if candidate_hist is None:
        return False
    similarity = cv2.compareHist(anchor_hist, candidate_hist, cv2.HISTCMP_CORREL)
    return bool(similarity >= APPEARANCE_THRESH)


# ── Basic mask validity ───────────────────────────────────────────────────────

def is_mask_valid(mask: np.ndarray, frame: np.ndarray) -> bool:
    """
    Quick sanity checks before the heavier appearance comparison:
    1. Enough active pixels.
    2. Region is not solid black (letterbox or cut-to-black).
    """
    active_pixels = int(mask.sum() / 255)
    if active_pixels < MIN_MASK_PIXELS:
        return False

    gray        = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    region_mean = float(gray[mask > 0].mean())
    return region_mean >= BLACK_ROW_THRESH


def warp_mask(prev_mask: np.ndarray, prev_frame: np.ndarray, next_frame: np.ndarray) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("opencv-python is required for mask propagation.")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        next_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    h, w = prev_mask.shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow[..., 0]).astype(np.float32)
    map_y = (grid_y + flow[..., 1]).astype(np.float32)
    warped = cv2.remap(prev_mask, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return normalize_anchor_mask(warped, h, w)


def sam_mask_from_prompt(sam_processor, frame: np.ndarray, prompt: str, bbox: list = None, prev_mask: np.ndarray = None) -> Optional[np.ndarray]:
    if sam_processor is None:
        return None
    try:
        candidate = sam_processor.generate_mask(frame, prompt=prompt, bbox=bbox, prev_mask=prev_mask)
        if candidate is None:
            return None

        if isinstance(candidate, np.ndarray):
            return normalize_anchor_mask(candidate, frame.shape[0], frame.shape[1])

        if isinstance(candidate, Image.Image):
            candidate_arr = np.array(candidate.convert("L"))
            return normalize_anchor_mask(candidate_arr, frame.shape[0], frame.shape[1])

        if hasattr(candidate, "mask"):
            candidate_arr = np.array(candidate.mask)
            return normalize_anchor_mask(candidate_arr, frame.shape[0], frame.shape[1])

    except Exception:
        return None

    return None


def refine_mask(frame: np.ndarray, prompt: str, warped_mask: np.ndarray, sam_processor) -> np.ndarray:
    if warped_mask is None:
        return np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

    sam_candidate = sam_mask_from_prompt(sam_processor, frame, prompt, prev_mask=warped_mask)
    if sam_candidate is None:
        return warped_mask

    combined = np.where(sam_candidate > 0, 255, warped_mask)
    return normalize_anchor_mask(combined, frame.shape[0], frame.shape[1])


# ── GrabCut product masking ────────────────────────────────────────────────────

def grabcut_product_mask(frame: np.ndarray, bbox: list) -> Optional[np.ndarray]:
    """
    Use OpenCV GrabCut with the product bounding box to produce a tight pixel
    mask that isolates the object and excludes the background (including people).
    Much more accurate than a simple rectangle fill.
    """
    if cv2 is None or bbox is None:
        return None
    if not (len(bbox) == 4 and all(v is not None and isinstance(v, (int, float)) for v in bbox)):
        return None

    x0, y0, x1, y1 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w - 1, x1), min(h - 1, y1)
    if x1 <= x0 or y1 <= y0:
        return None

    try:
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)
        rect      = (x0, y0, x1 - x0, y1 - y0)
        gc_mask   = np.zeros((h, w), dtype=np.uint8)

        cv2.grabCut(frame, gc_mask, rect, bgd_model, fgd_model, iterCount=5, mode=cv2.GC_INIT_WITH_RECT)

        # Pixels marked probable or definite foreground are kept
        fg_mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

        # Morphological clean-up: close small holes and remove stray specks
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel, iterations=1)

        return fg_mask if int(fg_mask.sum() / 255) >= MIN_MASK_PIXELS else None
    except Exception as e:
        print(f"[MASK] GrabCut failed: {e}")
        return None


# ── Skin-tone rejection ──────────────────────────────────────────────────────────

def remove_skin_tones(mask: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """
    Subtract human skin-tone pixels from the mask.
    If more than 60% of the remaining mask is skin, the whole mask is cleared
    (the tracker has locked onto a person, not a product).

    Skin range in HSV (tuned for broad human skin tones across ethnicities):
        Hue:  0–22 and 160–180
        Sat:  40–170
        Val:  80–255
    """
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Lower skin range
    lower1 = np.array([0,  40,  80], dtype=np.uint8)
    upper1 = np.array([22, 170, 255], dtype=np.uint8)
    # Upper skin range (wraps around hue wheel)
    lower2 = np.array([160, 40,  80], dtype=np.uint8)
    upper2 = np.array([180, 170, 255], dtype=np.uint8)

    skin_mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    # How much of the current product mask overlaps with skin?
    mask_pixels = int(mask.sum() / 255)
    if mask_pixels == 0:
        return mask

    overlap = int((mask & skin_mask).sum() / 255)
    skin_fraction = overlap / mask_pixels

    if skin_fraction > 0.85:
        # More than 85% of what we're tracking is skin — it's a person, not a product
        # (threshold loosened from 60% because red/orange products overlap with skin HSV range)
        return np.zeros_like(mask)

    # Otherwise just remove the overlapping skin pixels
    return mask & ~skin_mask


def save_alpha_mask(mask: np.ndarray, output_path: str) -> None:
    Image.fromarray(mask).save(output_path)


def render_mask_video(alpha_dir: str, output_path: str, fps: float, width: int, height: int) -> None:
    if cv2 is None:
        raise RuntimeError("opencv-python is required for mask video rendering.")

    frame_files = sorted(Path(alpha_dir).glob("frame_*.png"))
    if not frame_files:
        raise FileNotFoundError("No alpha masks found to render into video.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

    if not writer.isOpened():
        raise IOError(f"Unable to open VideoWriter for {output_path}")

    for frame_file in frame_files:
        mask_img = cv2.imread(str(frame_file), cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            raise IOError(f"Unable to read mask frame: {frame_file}")
        # Ensure frame matches writer dimensions exactly
        if mask_img.shape[:2] != (height, width):
            mask_img = cv2.resize(mask_img, (width, height), interpolation=cv2.INTER_NEAREST)
        writer.write(mask_img)

    writer.release()


def render_overlay_video(frames: List[np.ndarray], masks: List[np.ndarray], output_path: str, fps: float, width: int, height: int) -> None:
    if cv2 is None:
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)

    if not writer.isOpened():
        raise IOError(f"Unable to open VideoWriter for {output_path}")

    for frame, mask in zip(frames, masks):
        # Ensure frame and mask match writer dimensions
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))
        if mask.shape[1] != width or mask.shape[0] != height:
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        overlay = frame.copy()
        # Apply a semi-transparent green overlay where the mask is active
        overlay[mask == 255] = [0, 255, 0] # BGR green
        blended = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
        writer.write(blended)

    writer.release()


def grabcut_product_mask(frame: np.ndarray, bbox: list) -> np.ndarray:
    """
    Use OpenCV GrabCut to extract a pixel-perfect mask of the object inside the bounding box.
    bbox is [x0, y0, x1, y1]
    """
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return np.zeros(frame.shape[:2], dtype=np.uint8)

    mask = np.zeros(frame.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (x0, y0, w, h)

    try:
        cv2.grabCut(frame, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
        return mask2
    except Exception as e:
        print(f"[MASK] GrabCut failed: {e}")
        return np.zeros(frame.shape[:2], dtype=np.uint8)


def _detect_bbox_with_groq(frame: np.ndarray, product_name: str, frame_width: int, frame_height: int) -> Optional[list]:
    """
    Call Groq Vision on the given frame to get a fresh bounding box for the product.
    Returns [x_min, y_min, x_max, y_max] in pixel coords at (frame_width x frame_height),
    or None if detection fails.
    """
    try:
        import base64, json, re
        from app.core.config import settings

        # Downscale frame for the API call (save tokens but keep enough detail for small objects)
        h, w = frame.shape[:2]
        max_px = 1024
        if max(h, w) > max_px:
            sc = max_px / max(h, w)
            small = cv2.resize(frame, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
        else:
            small = frame
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return None
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

        prompt = (
            f"You are a precision object detector.\n\n"
            f"Find the '{product_name}' in this image.\n\n"
            f"Rules:\n"
            f"- Use NORMALIZED coordinates from 0 to 1000 (NOT pixels).\n"
            f"- (0,0) is top-left, (1000,1000) is bottom-right.\n"
            f"- x_max MUST be greater than x_min. y_max MUST be greater than y_min.\n"
            f"- Set found to true if the object is visible, false if not.\n\n"
            f'Respond with ONLY valid json: {{"found": bool, "x_min": int, "y_min": int, "x_max": int, "y_max": int}}'
        )

        response = settings.groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )

        raw = response.choices[0].message.content or ""
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)

        if not data.get("found", True):
            return None

        xn0, yn0, xn1, yn1 = data.get("x_min"), data.get("y_min"), data.get("x_max"), data.get("y_max")
        if any(v is None for v in [xn0, yn0, xn1, yn1]):
            return None
        if xn0 == 0 and yn0 == 0 and xn1 == 0 and yn1 == 0:
            return None

        # Convert normalized 0-1000 -> pixel coords and clamp
        x0 = max(0, min(int(xn0 * frame_width  / 1000.0), frame_width  - 1))
        y0 = max(0, min(int(yn0 * frame_height / 1000.0), frame_height - 1))
        x1 = max(0, min(int(xn1 * frame_width  / 1000.0), frame_width  - 1))
        y1 = max(0, min(int(yn1 * frame_height / 1000.0), frame_height - 1))

        if x1 <= x0 or y1 <= y0:
            print(f"[MASK] Groq bbox is degenerate ({x0},{y0},{x1},{y1}) -- ignoring.")
            return None

        return [x0, y0, x1, y1]

    except Exception as e:
        print(f"[MASK] Groq live bbox detection failed: {e}")
        return None


def _detect_keyframe_bboxes(
    frames: list,
    fps: float,
    product_name: str,
    frame_width: int,
    frame_height: int,
    fallback_bbox: list,
) -> dict:
    """
    Detect the product bbox on one frame per second using Groq Vision.
    Returns a dict mapping frame_index -> [x0, y0, x1, y1] at processing resolution.
    Frames where detection fails fall back to the provided fallback_bbox.
    """
    import time
    total = len(frames)
    keyframe_interval = max(1, int(fps))  # 1 keyframe per second
    keyframe_indices = list(range(0, total, keyframe_interval))

    print(f"[MASK] Detecting product on {len(keyframe_indices)} keyframes (every {keyframe_interval} frames)...")

    # Always trust frame 0 from ingestion as the ground truth start point
    keyframe_bboxes = {0: fallback_bbox}
    print(f"[MASK]   Keyframe 0: {fallback_bbox} (from ingestion anchor)")

    for i, kf_idx in enumerate(keyframe_indices):
        if kf_idx == 0:
            continue
            
        bbox = _detect_bbox_with_groq(frames[kf_idx], product_name, frame_width, frame_height)
        if bbox:
            keyframe_bboxes[kf_idx] = bbox
            print(f"[MASK]   Keyframe {kf_idx}: {bbox}")
        else:
            print(f"[MASK]   Keyframe {kf_idx}: product not found, skipping (will interpolate across gap)")

        # Small delay between API calls to avoid rate limits
        if i < len(keyframe_indices) - 1:
            time.sleep(0.5)

    return keyframe_bboxes


def generate_temporal_alpha_masks(
    show_id: str,
    manual_mask_path: Optional[str] = None,
    anchor_frame_index: int = 0,
    prompt: Optional[str] = None,
) -> Dict[str, object]:
    scene = load_scene_metadata(show_id)
    video_path = scene.get("filepath")
    if not video_path:
        raise ValueError("Scene metadata is missing a valid video filepath.")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video path does not exist: {video_path}")

    frames, fps, orig_width, orig_height, proc_width, proc_height, scale_x, scale_y = read_video_frames(video_path)
    if not frames:
        raise ValueError("No frames were extracted from the video.")

    if anchor_frame_index < 0 or anchor_frame_index >= len(frames):
        raise ValueError("anchor_frame_index must be within the video frame count.")

    alpha_dir          = ensure_output_dirs(show_id)
    mask_video_path    = os.path.join(MASKS_DIR, f"{show_id}_mask.mp4")
    preview_video_path = os.path.join(MASKS_DIR, f"{show_id}_preview.mp4")

    prompt_text   = prompt or scene.get("target_object") or "the object marked by the provided mask"
    sam_processor = initialize_sam_processor()

    detection_frame_bboxes = _detection_bboxes_by_frame(
        scene,
        len(frames),
        proc_width,
        proc_height,
        scale_x,
        scale_y,
    )
    if detection_frame_bboxes:
        print("[MASK] Using stored second-by-second whole-car detections for preview masks.")
        return _render_detection_bbox_masks(
            show_id=show_id,
            video_path=video_path,
            frames=frames,
            fps=fps,
            orig_width=orig_width,
            orig_height=orig_height,
            proc_width=proc_width,
            proc_height=proc_height,
            alpha_dir=alpha_dir,
            mask_video_path=mask_video_path,
            preview_video_path=preview_video_path,
            frame_bboxes=detection_frame_bboxes,
        )

    # ── Step 1: Resolve the initial bounding box ─────────────────────────────
    # Use the stored bbox from ingestion (already validated and clamped to frame bounds).
    # Re-map from original resolution → processing resolution.
    stored_bbox = scene.get("initial_bounding_box")
    if not stored_bbox or len(stored_bbox) != 4 or any(v is None for v in stored_bbox):
        raise ValueError("Valid initial_bounding_box is required in scene metadata. Re-ingest the video.")

    sx0, sy0, sx1, sy1 = stored_bbox
    # Remap to processing resolution and clamp
    bx0 = max(0, min(int(sx0 / scale_x), proc_width  - 1))
    by0 = max(0, min(int(sy0 / scale_y), proc_height - 1))
    bx1 = max(0, min(int(sx1 / scale_x), proc_width  - 1))
    by1 = max(0, min(int(sy1 / scale_y), proc_height - 1))

    if bx1 <= bx0 or by1 <= by0:
        # Stored bbox is out-of-bounds for this processing resolution — use raw clamped
        bx0 = max(0, min(int(sx0), proc_width  - 1))
        by0 = max(0, min(int(sy0), proc_height - 1))
        bx1 = max(0, min(int(sx1), proc_width  - 1))
        by1 = max(0, min(int(sy1), proc_height - 1))
        print(f"[MASK] Stored bbox remapped degenerate, using clamped raw: [{bx0},{by0},{bx1},{by1}]")

    if bx1 <= bx0 or by1 <= by0:
        raise ValueError(f"Bounding box [{sx0},{sy0},{sx1},{sy1}] is completely out of range for this video. Re-ingest.")

    proc_bbox = [bx0, by0, bx1, by1]  # [x0, y0, x1, y1] at processing resolution
    print(f"[MASK] Using product bbox (proc-res): {proc_bbox}")

    # ── Step 2: Track with Lucas-Kanade sparse optical flow ──────────────────
    # LK tracks feature points (corners) detected INSIDE the bbox on frame 0.
    # The median displacement of all living points updates the bbox each frame.
    # 
    # Anti-drift measures:
    #   1. Scene cut detection: reset to original bbox when the camera angle jumps
    #   2. Periodic feature refresh (every 10 frames): re-detect features inside
    #      the current bbox to prevent accumulated LK error
    #   3. Keep bbox size constant (only translate, never shrink/grow)

    REFRESH_INTERVAL = 10  # re-detect features every N frames to fight drift

    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    bbox_w = bx1 - bx0  # lock the width/height — only translate the box
    bbox_h = by1 - by0

    def _detect_features(gray_frame, x0, y0, x1, y1):
        """Detect good features to track inside a bbox region."""
        roi_mask = np.zeros(gray_frame.shape[:2], dtype=np.uint8)
        roi_mask[max(0,y0):min(gray_frame.shape[0],y1),
                 max(0,x0):min(gray_frame.shape[1],x1)] = 255
        pts = cv2.goodFeaturesToTrack(
            gray_frame, maxCorners=80, qualityLevel=0.03, minDistance=4, mask=roi_mask
        )
        if pts is None or len(pts) == 0:
            # Grid fallback
            xs = np.linspace(x0 + 2, x1 - 2, 7, dtype=np.float32)
            ys = np.linspace(y0 + 2, y1 - 2, 7, dtype=np.float32)
            gx, gy = np.meshgrid(xs, ys)
            pts = np.stack([gx.ravel(), gy.ravel()], axis=1).reshape(-1, 1, 2)
        return pts

    anchor_frame = frames[anchor_frame_index]
    anchor_gray = cv2.cvtColor(anchor_frame, cv2.COLOR_BGR2GRAY)
    pts       = _detect_features(anchor_gray, bx0, by0, bx1, by1)
    prev_gray = anchor_gray.copy()
    cur_cx    = (bx0 + bx1) / 2.0  # track the CENTER of the bbox (float for sub-pixel)
    cur_cy    = (by0 + by1) / 2.0

    masks = [None] * len(frames)
    
    # GrabCut the anchor mask frame
    anchor_gc_mask = grabcut_product_mask(anchor_frame, [bx0, by0, bx1, by1])
    if anchor_gc_mask.sum() > 0:
        masks[anchor_frame_index] = anchor_gc_mask
    else:
        masks[anchor_frame_index] = bbox_to_mask([bx0, by0, bx1, by1], proc_height, proc_width)

    total_frames = len(frames)
    frames_since_refresh = 0

    for idx in range(anchor_frame_index + 1, total_frames):
        if idx % 10 == 0:
            print(f"[MASK] Tracking and applying GrabCut {idx}/{total_frames}...")

        curr_gray = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2GRAY)

        # ── Scene cut detection ──────────────────────────────────────────
        if is_shot_cut(frames[idx - 1], frames[idx]):
            print(f"[MASK] Scene cut at frame {idx} -- resetting tracker to original bbox.")
            # Reset bbox to original position (product reappears in a new shot)
            cur_cx = (bx0 + bx1) / 2.0
            cur_cy = (by0 + by1) / 2.0
            pts = _detect_features(curr_gray, bx0, by0, bx1, by1)
            frames_since_refresh = 0
            
            rx0 = max(0, int(cur_cx - bbox_w / 2))
            ry0 = max(0, int(cur_cy - bbox_h / 2))
            rx1 = min(proc_width,  rx0 + bbox_w)
            ry1 = min(proc_height, ry0 + bbox_h)
            
            gc_mask = grabcut_product_mask(frames[idx], [rx0, ry0, rx1, ry1])
            if gc_mask.sum() > 0:
                masks[idx] = gc_mask
            else:
                masks[idx] = bbox_to_mask([rx0, ry0, rx1, ry1], proc_height, proc_width)
            
            prev_gray = curr_gray
            continue

        # ── Periodic feature refresh to combat drift ─────────────────────
        frames_since_refresh += 1
        if frames_since_refresh >= REFRESH_INTERVAL and pts is not None:
            rx0 = max(0, int(cur_cx - bbox_w / 2))
            ry0 = max(0, int(cur_cy - bbox_h / 2))
            rx1 = min(proc_width,  rx0 + bbox_w)
            ry1 = min(proc_height, ry0 + bbox_h)
            pts = _detect_features(curr_gray, rx0, ry0, rx1, ry1)
            frames_since_refresh = 0

        # ── LK optical flow tracking ─────────────────────────────────────
        if pts is not None and len(pts) >= 3:
            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, pts, None, **lk_params
            )
            good_new = new_pts[status.ravel() == 1]
            good_old = pts[status.ravel() == 1]

            if len(good_new) >= 3:
                # Median displacement (robust to outliers from people moving in front)
                dx = float(np.median(good_new[:, 0, 0] - good_old[:, 0, 0]))
                dy = float(np.median(good_new[:, 0, 1] - good_old[:, 0, 1]))

                cur_cx += dx
                cur_cy += dy

                pts = good_new.reshape(-1, 1, 2)
            else:
                # Too few points survived this frame — hold position, try to re-detect
                print(f"[MASK] LK lost tracking at frame {idx} -- re-detecting features.")
                rx0 = max(0, int(cur_cx - bbox_w / 2))
                ry0 = max(0, int(cur_cy - bbox_h / 2))
                rx1 = min(proc_width,  rx0 + bbox_w)
                ry1 = min(proc_height, ry0 + bbox_h)
                pts = _detect_features(curr_gray, rx0, ry0, rx1, ry1)
                frames_since_refresh = 0

        # ── Convert center + fixed size -> GrabCut mask ──────────────────
        rx0 = max(0, int(cur_cx - bbox_w / 2))
        ry0 = max(0, int(cur_cy - bbox_h / 2))
        rx1 = min(proc_width,  rx0 + bbox_w)
        ry1 = min(proc_height, ry0 + bbox_h)

        if rx1 > rx0 and ry1 > ry0:
            gc_mask = grabcut_product_mask(frames[idx], [rx0, ry0, rx1, ry1])
            if gc_mask.sum() > 0:
                masks[idx] = gc_mask
            else:
                masks[idx] = bbox_to_mask([rx0, ry0, rx1, ry1], proc_height, proc_width)
        else:
            masks[idx] = np.zeros((proc_height, proc_width), dtype=np.uint8)

        prev_gray = curr_gray

    # Upscale masks from processing resolution back to original resolution
    output_masks = []
    for mask in masks:
        if mask is None:
            mask = np.zeros((proc_height, proc_width), dtype=np.uint8)
        if orig_width != proc_width:
            mask = cv2.resize(mask, (orig_width, orig_height), interpolation=cv2.INTER_NEAREST)
        output_masks.append(mask)

    frame_urls: List[str] = []
    for idx, mask in enumerate(output_masks):
        out_path = os.path.join(alpha_dir, f"frame_{idx:04d}.png")
        save_alpha_mask(mask, out_path)
        frame_urls.append(f"/masks/{show_id}/alpha/frame_{idx:04d}.png")

    # Render preview at original resolution
    if orig_width != proc_width:
        print(f"[MASK] Re-reading original frames for preview at {orig_width}x{orig_height}...")
        cap = cv2.VideoCapture(video_path)
        orig_frames = []
        while True:
            ok, frm = cap.read()
            if not ok:
                break
            orig_frames.append(frm)
        cap.release()
    else:
        orig_frames = frames

    render_mask_video(alpha_dir, mask_video_path, fps, orig_width, orig_height)
    render_overlay_video(orig_frames, output_masks, preview_video_path, fps, orig_width, orig_height)
    print(f"[MASK] Preview video -> {preview_video_path}")


    return {
        "show_id": show_id,
        "frame_count": len(output_masks),
        "mask_directory": alpha_dir,
        "mask_video": mask_video_path,
        "frame_urls": frame_urls,
        "preview_url": f"/masks/{show_id}_preview.mp4",
    }


def generate_mask_video(show_id: str) -> str:
    """
    Wrapper function to maintain backward compatibility with ingestion.py
    Calls the new SAM3 pipeline and returns the absolute path to the mask video.
    """
    result = generate_temporal_alpha_masks(show_id)
    return os.path.abspath(result["mask_video"])


def mask_status(show_id: str) -> Dict[str, object]:
    alpha_dir = os.path.join(MASKS_DIR, show_id, "alpha")
    if not os.path.exists(alpha_dir):
        return {"ready": False, "frame_count": 0, "mask_directory": alpha_dir}

    files = sorted([p for p in os.listdir(alpha_dir) if p.endswith(".png")])
    return {
        "ready": True,
        "frame_count": len(files),
        "mask_directory": alpha_dir,
        "preview_url": f"/masks/{show_id}_preview.mp4" if files else None,
    }
