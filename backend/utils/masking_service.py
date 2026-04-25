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

MASKS_DIR = "assets/masks"
SCENE_VAULT_PATH = "db/scene_vault.json"
SAM3_MODEL_PATH = os.getenv("SAM3_MODEL_PATH", "meta/sam3-optimized")


def load_scene_metadata(show_id: str) -> Dict:
    if not os.path.exists(SCENE_VAULT_PATH):
        raise FileNotFoundError(f"Scene vault not found at {SCENE_VAULT_PATH}")

    with open(SCENE_VAULT_PATH, "r", encoding="utf-8") as handle:
        vault = json.load(handle)

    scenes = vault.get("scenes", [])
    for scene in scenes:
        if scene.get("show_id") == show_id:
            return scene

    raise ValueError(f"No scene metadata found for show_id={show_id}")


def initialize_sam_processor() -> Optional[object]:
    if Sam3Processor is None:
        return None

    return Sam3Processor.from_pretrained(SAM3_MODEL_PATH)


def ensure_output_dirs(show_id: str) -> str:
    alpha_dir = os.path.join(MASKS_DIR, show_id, "alpha")
    os.makedirs(alpha_dir, exist_ok=True)
    return alpha_dir


def read_video_frames(video_path: str):
    if cv2 is None:
        raise RuntimeError("opencv-python is required for video frame processing. Please install opencv-python.")

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise IOError(f"Unable to open video file: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []

    while True:
        success, frame = capture.read()
        if not success:
            break
        frames.append(frame)

    capture.release()
    return frames, fps, width, height


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
    if bbox and len(bbox) == 4 and all(isinstance(v, int) for v in bbox):
        x0, y0, x1, y1 = bbox
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(frame_width, x1), min(frame_height, y1)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 255
    return mask


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


def sam_mask_from_prompt(sam_processor, frame: np.ndarray, prompt: str) -> Optional[np.ndarray]:
    if sam_processor is None:
        return None

    if not prompt:
        return None

    try:
        candidate = sam_processor.generate_mask(frame, prompt=prompt)
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

    sam_candidate = sam_mask_from_prompt(sam_processor, frame, prompt)
    if sam_candidate is None:
        return warped_mask

    combined = np.where(sam_candidate > 0, 255, warped_mask)
    return normalize_anchor_mask(combined, frame.shape[0], frame.shape[1])


def save_alpha_mask(mask: np.ndarray, output_path: str) -> None:
    Image.fromarray(mask).save(output_path)


def render_mask_video(alpha_dir: str, output_path: str, fps: float, width: int, height: int) -> None:
    if cv2 is None:
        raise RuntimeError("opencv-python is required for mask video rendering.")

    frame_files = sorted(Path(alpha_dir).glob("frame_*.png"))
    if not frame_files:
        raise FileNotFoundError("No alpha masks found to render into video.")

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

    if not writer.isOpened():
        raise IOError(f"Unable to open VideoWriter for {output_path}")

    for frame_file in frame_files:
        mask_img = cv2.imread(str(frame_file), cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            raise IOError(f"Unable to read mask frame: {frame_file}")
        writer.write(mask_img)

    writer.release()


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

    frames, fps, width, height = read_video_frames(video_path)
    if not frames:
        raise ValueError("No frames were extracted from the video.")

    if anchor_frame_index < 0 or anchor_frame_index >= len(frames):
        raise ValueError("anchor_frame_index must be within the video frame count.")

    alpha_dir = ensure_output_dirs(show_id)
    mask_video_path = os.path.join(MASKS_DIR, f"{show_id}_mask.avi")

    prompt_text = prompt or scene.get("target_object") or "the object marked by the provided mask"
    sam_processor = initialize_sam_processor()

    anchor_frame = frames[anchor_frame_index]
    if manual_mask_path and os.path.exists(manual_mask_path):
        anchor_mask = load_manual_mask(manual_mask_path, height, width)
    else:
        anchor_mask = bbox_to_mask(scene.get("initial_bounding_box"), height, width)

    if anchor_mask.sum() == 0:
        raise ValueError("Unable to build an anchor mask from the manual input or scene metadata.")

    masks = [None] * len(frames)
    masks[anchor_frame_index] = anchor_mask

    for idx in range(anchor_frame_index + 1, len(frames)):
        warped = warp_mask(masks[idx - 1], frames[idx - 1], frames[idx])
        masks[idx] = refine_mask(frames[idx], prompt_text, warped, sam_processor)

    for idx in range(anchor_frame_index - 1, -1, -1):
        warped = warp_mask(masks[idx + 1], frames[idx + 1], frames[idx])
        masks[idx] = refine_mask(frames[idx], prompt_text, warped, sam_processor)

    frame_urls: List[str] = []
    for idx, mask in enumerate(masks):
        output_path = os.path.join(alpha_dir, f"frame_{idx:04d}.png")
        save_alpha_mask(mask, output_path)
        frame_urls.append(f"/masks/{show_id}/alpha/frame_{idx:04d}.png")

    render_mask_video(alpha_dir, mask_video_path, fps, width, height)

    return {
        "show_id": show_id,
        "frame_count": len(masks),
        "mask_directory": alpha_dir,
        "mask_video": mask_video_path,
        "frame_urls": frame_urls,
        "preview_url": frame_urls[0] if frame_urls else None,
    }


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
