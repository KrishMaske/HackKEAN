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
        print(f"[MASK] Downscaling from {orig_width}x{orig_height} → {proc_width}x{proc_height} for processing.")
    else:
        proc_width   = orig_width
        proc_height  = orig_height
        print(f"[MASK] Source is {orig_width}x{orig_height} — no downscale needed.")

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


# ── Thresholds (tune here) ────────────────────────────────────────────────────
MIN_MASK_PIXELS     = 80    # mask with fewer active pixels is treated as "lost"
CUT_THRESHOLD       = 0.50  # histogram correlation below this → hard scene cut
APPEARANCE_THRESH   = 0.35  # colour similarity below this → mask drifted to wrong object
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

    if skin_fraction > 0.60:
        # More than 60% of what we're tracking is skin — it's a person, not a product
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
        overlay = frame.copy()
        # Apply a semi-transparent green overlay where the mask is active
        overlay[mask == 255] = [0, 255, 0] # BGR green
        blended = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
        writer.write(blended)

    writer.release()


def _detect_bbox_with_groq(frame: np.ndarray, product_name: str, frame_width: int, frame_height: int) -> Optional[list]:
    """
    Call Groq Vision on the given frame to get a fresh bounding box for the product.
    Returns [x_min, y_min, x_max, y_max] in pixel coords at (frame_width x frame_height),
    or None if detection fails.
    """
    try:
        import base64
        from app.core.config import settings

        # Encode the frame as JPEG for the API call
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return None
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

        prompt = (
            f"You are a precision object detector. This image is {frame_width}x{frame_height} pixels.\n\n"
            f"Find the '{product_name}' in this image and return its EXACT bounding box in pixel coordinates.\n\n"
            f"Rules:\n"
            f"- x_min and x_max must be between 0 and {frame_width}\n"
            f"- y_min and y_max must be between 0 and {frame_height}\n"
            f"- The box must TIGHTLY enclose the product only, not the background\n"
            f"- If the product is not visible, return null for all values\n\n"
            f'Respond ONLY with: {{"x_min": int, "y_min": int, "x_max": int, "y_max": int}}'
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

        import json, re
        raw = response.choices[0].message.content or ""
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)

        x0 = data.get("x_min")
        y0 = data.get("y_min")
        x1 = data.get("x_max")
        y1 = data.get("y_max")

        if any(v is None for v in [x0, y0, x1, y1]):
            print("[MASK] Groq returned null bbox — product not found in anchor frame.")
            return None

        # Clamp to frame boundaries
        x0 = max(0, min(int(x0), frame_width  - 1))
        y0 = max(0, min(int(y0), frame_height - 1))
        x1 = max(0, min(int(x1), frame_width  - 1))
        y1 = max(0, min(int(y1), frame_height - 1))

        if x1 <= x0 or y1 <= y0:
            print(f"[MASK] Groq bbox is degenerate ({x0},{y0},{x1},{y1}) — ignoring.")
            return None

        return [x0, y0, x1, y1]

    except Exception as e:
        print(f"[MASK] Groq live bbox detection failed: {e}")
        return None


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

    alpha_dir         = ensure_output_dirs(show_id)
    mask_video_path   = os.path.join(MASKS_DIR, f"{show_id}_mask.mp4")
    preview_video_path = os.path.join(MASKS_DIR, f"{show_id}_preview.mp4")

    prompt_text  = prompt or scene.get("target_object") or "the object marked by the provided mask"
    sam_processor = initialize_sam_processor()

    anchor_frame = frames[anchor_frame_index]
    if manual_mask_path and os.path.exists(manual_mask_path):
        anchor_mask = load_manual_mask(manual_mask_path, proc_height, proc_width)
    else:
        # ── Step 1: get bounding box ─────────────────────────────────────────
        # Always re-detect fresh from the anchor frame using Groq Vision.
        # The stored initial_bounding_box from ingestion is often stale or has
        # wrong coordinates, so we re-query every time masking runs.
        stored_bbox = scene.get("initial_bounding_box")
        fresh_bbox  = _detect_bbox_with_groq(anchor_frame, prompt_text, proc_width, proc_height)

        if fresh_bbox:
            initial_bbox = fresh_bbox
            print(f"[MASK] Fresh Groq bbox: {initial_bbox}")
        elif stored_bbox and all(v is not None for v in stored_bbox):
            # Remap stored bbox from original → processing resolution
            x0, y0, x1, y1 = stored_bbox
            initial_bbox = [
                int(x0 / scale_x), int(y0 / scale_y),
                int(x1 / scale_x), int(y1 / scale_y),
            ]
            print(f"[MASK] Falling back to stored bbox (remapped): {initial_bbox}")
        else:
            initial_bbox = None
            print("[MASK] No bounding box available.")

        # ── Step 2: build anchor mask ────────────────────────────────────────
        print(f"[MASK] Building anchor mask for product: '{prompt_text}'")
        anchor_mask = None

        # 1st attempt: GrabCut — tight pixel-accurate product mask
        if initial_bbox:
            gc_mask = grabcut_product_mask(anchor_frame, initial_bbox)
            if gc_mask is not None:
                anchor_mask = gc_mask
                print("[MASK] Anchor mask built via GrabCut.")

        # 2nd attempt: SAM3 (if available)
        if anchor_mask is None:
            sam_mask = sam_mask_from_prompt(sam_processor, anchor_frame, prompt_text, bbox=initial_bbox)
            if sam_mask is not None and sam_mask.sum() > 0:
                anchor_mask = sam_mask
                print("[MASK] Anchor mask built via SAM3.")

        # Last resort: simple bounding box rectangle
        if anchor_mask is None and initial_bbox:
            print("[MASK] Falling back to bounding box rectangle.")
            anchor_mask = bbox_to_mask(initial_bbox, proc_height, proc_width)

        if anchor_mask is None or anchor_mask.sum() == 0:
            raise ValueError("Unable to build anchor mask — check bounding box coordinates.")


    # Strip skin from anchor mask right away (in case GrabCut included hands, etc.)
    anchor_mask = remove_skin_tones(anchor_mask, anchor_frame)
    anchor_mask = apply_letterbox_exclusion(anchor_mask, anchor_frame)

    # Pre-compute the anchor product's colour fingerprint for appearance matching
    anchor_hist = _masked_histogram(anchor_frame, anchor_mask)

    masks = [None] * len(frames)
    masks[anchor_frame_index] = anchor_mask

    def _process_frame(idx: int, prev_mask: np.ndarray, prev_frame: np.ndarray, curr_frame: np.ndarray) -> np.ndarray:
        """Run all six guards and return the final mask for curr_frame."""
        blank = np.zeros((curr_frame.shape[0], curr_frame.shape[1]), dtype=np.uint8)

        # Guard 1: hard scene cut → blank
        if is_shot_cut(prev_frame, curr_frame):
            print(f"[MASK] Shot cut at frame {idx} — mask reset.")
            return blank

        warped = warp_mask(prev_mask, prev_frame, curr_frame)

        # Guard 2: strip all letterbox bars
        warped = apply_letterbox_exclusion(warped, curr_frame)

        # Guard 3: skin-tone rejection (removes people from mask)
        warped = remove_skin_tones(warped, curr_frame)

        # Guard 4: GrabCut refinement every 3rd frame
        if idx % 3 == 0:
            candidate = refine_mask(curr_frame, prompt_text, warped, sam_processor)
        else:
            candidate = warped

        # Guard 5: basic validity (pixel count + region brightness)
        if not is_mask_valid(candidate, curr_frame):
            return blank

        # Guard 6: colour appearance vs anchor product fingerprint
        if not is_same_object(curr_frame, candidate, anchor_hist):
            print(f"[MASK] Appearance mismatch at frame {idx} — mask cleared.")
            return blank

        return candidate

    total_frames = len(frames)
    for idx in range(anchor_frame_index + 1, total_frames):
        if idx % 10 == 0:
            print(f"[MASK] Processing frame {idx}/{total_frames}...")
        masks[idx] = _process_frame(idx, masks[idx - 1], frames[idx - 1], frames[idx])

    for idx in range(anchor_frame_index - 1, -1, -1):
        if idx % 10 == 0:
            print(f"[MASK] Processing frame {idx}/{total_frames}...")
        masks[idx] = _process_frame(idx, masks[idx + 1], frames[idx + 1], frames[idx])

    # Upscale masks from processing resolution back to original resolution
    output_masks = []
    for mask in masks:
        if orig_width != proc_width and mask is not None:
            mask = cv2.resize(mask, (orig_width, orig_height), interpolation=cv2.INTER_NEAREST)
        output_masks.append(mask if mask is not None else np.zeros((orig_height, orig_width), dtype=np.uint8))

    frame_urls: List[str] = []
    for idx, mask in enumerate(output_masks):
        output_path = os.path.join(alpha_dir, f"frame_{idx:04d}.png")
        save_alpha_mask(mask, output_path)
        frame_urls.append(f"/masks/{show_id}/alpha/frame_{idx:04d}.png")


    # Render videos at original resolution using original frames
    # For preview, re-read original frames so the overlay is at full quality
    if orig_width != proc_width:
        print(f"[MASK] Re-reading original frames for preview video at {orig_width}x{orig_height}...")
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

    render_mask_video(alpha_dir,    mask_video_path,    fps, orig_width, orig_height)
    render_overlay_video(orig_frames, output_masks, preview_video_path, fps, orig_width, orig_height)
    print(f"[MASK] Generated Preview Video -> {preview_video_path}")


    return {
        "show_id": show_id,
        "frame_count": len(output_masks),
        "mask_directory": alpha_dir,
        "mask_video": mask_video_path,
        "frame_urls": frame_urls,
        "preview_url": frame_urls[0] if frame_urls else None,
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
        "preview_url": f"/masks/{show_id}/alpha/{files[0]}" if files else None,
    }
