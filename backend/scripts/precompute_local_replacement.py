import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.masking import generate_temporal_alpha_masks, load_scene_metadata


MASKS_DIR = Path("assets/masks")
OUTPUT_DIR = Path("assets/output")
DEFAULT_MODEL = "runwayml/stable-diffusion-inpainting"
DEFAULT_ASSET = "ferrari.png"
MASK_MIN_PIXELS = 100
SEED = 42
FERRARI_RED_BGR = np.array([32, 32, 210], dtype=np.float32)


def load_diffusers_pipeline(model_id: str):
    import torch
    from diffusers import StableDiffusionInpaintPipeline

    cuda = torch.cuda.is_available()
    dtype = torch.float16 if cuda else torch.float32
    device = "cuda" if cuda else "cpu"

    print(f"[LOCAL-REPLACE] Loading {model_id} on {device} ({dtype})")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)

    if cuda:
        pipe.enable_attention_slicing()
    return pipe, torch, device


def read_frames(video_path: str) -> tuple[List[np.ndarray], float, int, int]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)

    capture.release()
    return frames, fps, width, height


def frame_to_image(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def load_mask(mask_path: Path, width: int, height: int) -> Optional[np.ndarray]:
    if not mask_path.exists():
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return np.where(mask > 127, 255, 0).astype(np.uint8)


def mask_to_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(mask, mode="L")


def soft_mask(mask: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(mask, (7, 7), 0)
    alpha = blurred.astype(np.float32) / 255.0
    return alpha[:, :, None]


def composite(original_frame: np.ndarray, generated: Image.Image, mask: np.ndarray) -> np.ndarray:
    height, width = original_frame.shape[:2]
    generated = generated.resize((width, height), Image.Resampling.LANCZOS)
    generated_bgr = cv2.cvtColor(np.array(generated), cv2.COLOR_RGB2BGR).astype(np.float32)
    original = original_frame.astype(np.float32)
    alpha = soft_mask(mask)
    result = generated_bgr * alpha + original * (1.0 - alpha)
    return np.clip(result, 0, 255).astype(np.uint8)


def orange_paint_mask(frame: np.ndarray, replacement_region: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([4, 55, 45], dtype=np.uint8)
    upper_orange = np.array([35, 255, 255], dtype=np.uint8)
    lower_red_orange = np.array([0, 65, 45], dtype=np.uint8)
    upper_red_orange = np.array([8, 255, 255], dtype=np.uint8)
    paint = cv2.inRange(hsv, lower_orange, upper_orange) | cv2.inRange(hsv, lower_red_orange, upper_red_orange)
    paint = cv2.bitwise_and(paint, np.where(replacement_region > 127, 255, 0).astype(np.uint8))

    ys, xs = np.where(replacement_region > 127)
    if len(xs) and len(ys):
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        top_cut = y0 + int((y1 - y0) * 0.28)
        geometry_gate = np.zeros_like(paint)
        geometry_gate[top_cut:y1 + 1, x0:x1 + 1] = 255
        paint = cv2.bitwise_and(paint, geometry_gate)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    paint = cv2.morphologyEx(paint, cv2.MORPH_CLOSE, kernel, iterations=2)
    paint = cv2.morphologyEx(paint, cv2.MORPH_OPEN, kernel, iterations=1)

    components, labels, stats, _ = cv2.connectedComponentsWithStats(paint, connectivity=8)
    filtered = np.zeros_like(paint)
    for component_id in range(1, components):
        area = stats[component_id, cv2.CC_STAT_AREA]
        width = stats[component_id, cv2.CC_STAT_WIDTH]
        height = stats[component_id, cv2.CC_STAT_HEIGHT]
        if area >= 80 and width >= 12 and height >= 6:
            filtered[labels == component_id] = 255
    paint = filtered
    return paint


def recolor_to_red_ferrari(original_frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    paint = orange_paint_mask(original_frame, mask)
    if int(cv2.countNonZero(paint)) <= MASK_MIN_PIXELS:
        return original_frame

    hsv = cv2.cvtColor(original_frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    red_hue = np.zeros_like(hsv[:, :, 0])
    hsv[:, :, 0] = np.where(paint > 0, red_hue, hsv[:, :, 0])
    hsv[:, :, 1] = np.where(paint > 0, np.maximum(hsv[:, :, 1], 185), hsv[:, :, 1])
    hsv[:, :, 2] = np.where(paint > 0, np.clip(hsv[:, :, 2] * 0.95 + 18, 0, 255), hsv[:, :, 2])
    red_frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Push the paint further toward Ferrari red while preserving highlights and shadows.
    paint_float = paint.astype(np.float32) / 255.0
    red_mix = red_frame.astype(np.float32) * 0.65 + FERRARI_RED_BGR * 0.35
    red_frame = np.where(paint_float[:, :, None] > 0, red_mix, red_frame.astype(np.float32))

    alpha = soft_mask(paint)
    result = red_frame * alpha + original_frame.astype(np.float32) * (1.0 - alpha)
    return np.clip(result, 0, 255).astype(np.uint8)


def bbox_from_mask(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 127)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def build_smoothed_bboxes(alpha_dir: Path, width: int, height: int, frame_count: int) -> Dict[int, tuple[int, int, int, int]]:
    raw: Dict[int, tuple[int, int, int, int]] = {}
    for idx in range(frame_count):
        mask = load_mask(alpha_dir / f"frame_{idx:04d}.png", width, height)
        if mask is None or int(cv2.countNonZero(mask)) <= MASK_MIN_PIXELS:
            continue
        bbox = bbox_from_mask(mask)
        if bbox is not None:
            raw[idx] = bbox

    if not raw:
        return {}

    smoothed: Dict[int, tuple[int, int, int, int]] = {}
    previous = None
    alpha = 0.28
    for idx in range(frame_count):
        current = raw.get(idx)
        if current is None:
            if previous is not None:
                smoothed[idx] = tuple(int(round(v)) for v in previous)
            continue

        values = np.array(current, dtype=np.float32)
        if previous is None:
            previous = values
        else:
            previous = previous * (1.0 - alpha) + values * alpha
        smoothed[idx] = tuple(int(round(v)) for v in previous)
    return smoothed


def make_ferrari_sprite(width: int, height: int) -> np.ndarray:
    width = max(width, 40)
    height = max(height, 24)
    sprite = np.zeros((height, width, 4), dtype=np.uint8)

    def pt(x: float, y: float) -> tuple[int, int]:
        return int(round(x * width)), int(round(y * height))

    red = (20, 20, 222, 255)
    dark_red = (12, 10, 125, 255)
    highlight = (80, 90, 255, 230)
    black = (8, 8, 10, 255)
    glass = (38, 45, 52, 245)
    tire = (4, 4, 5, 255)
    rim = (210, 210, 205, 255)

    shadow = np.array([pt(0.08, 0.80), pt(0.92, 0.80), pt(0.98, 0.92), pt(0.02, 0.92)], dtype=np.int32)
    cv2.fillPoly(sprite, [shadow], (0, 0, 0, 85))

    body = np.array([
        pt(0.04, 0.62), pt(0.12, 0.48), pt(0.28, 0.43), pt(0.40, 0.30),
        pt(0.62, 0.28), pt(0.78, 0.42), pt(0.93, 0.48), pt(0.98, 0.62),
        pt(0.91, 0.74), pt(0.15, 0.76),
    ], dtype=np.int32)
    cv2.fillPoly(sprite, [body], red)

    lower = np.array([pt(0.08, 0.64), pt(0.94, 0.64), pt(0.90, 0.76), pt(0.13, 0.76)], dtype=np.int32)
    cv2.fillPoly(sprite, [lower], dark_red)

    cabin = np.array([pt(0.34, 0.43), pt(0.44, 0.31), pt(0.61, 0.31), pt(0.73, 0.45), pt(0.63, 0.49), pt(0.39, 0.48)], dtype=np.int32)
    cv2.fillPoly(sprite, [cabin], glass)

    windshield = np.array([pt(0.37, 0.44), pt(0.45, 0.34), pt(0.51, 0.34), pt(0.50, 0.47)], dtype=np.int32)
    side_window = np.array([pt(0.53, 0.34), pt(0.61, 0.34), pt(0.70, 0.45), pt(0.54, 0.47)], dtype=np.int32)
    cv2.fillPoly(sprite, [windshield, side_window], (70, 82, 92, 245))

    highlight_poly = np.array([pt(0.16, 0.51), pt(0.52, 0.43), pt(0.86, 0.50), pt(0.80, 0.56), pt(0.18, 0.58)], dtype=np.int32)
    cv2.fillPoly(sprite, [highlight_poly], highlight)

    intake = np.array([pt(0.62, 0.60), pt(0.85, 0.59), pt(0.80, 0.70), pt(0.60, 0.70)], dtype=np.int32)
    cv2.fillPoly(sprite, [intake], black)

    cv2.rectangle(sprite, pt(0.88, 0.50), pt(0.96, 0.55), (230, 220, 185, 255), thickness=-1)
    cv2.rectangle(sprite, pt(0.08, 0.58), pt(0.15, 0.62), (245, 240, 215, 255), thickness=-1)

    for cx in (0.27, 0.75):
        center = pt(cx, 0.73)
        radius = max(4, int(height * 0.16))
        cv2.circle(sprite, center, radius, tire, thickness=-1)
        cv2.circle(sprite, center, max(2, int(radius * 0.58)), rim, thickness=-1)
        cv2.circle(sprite, center, max(1, int(radius * 0.25)), (45, 45, 48, 255), thickness=-1)

    sprite[:, :, 3] = cv2.GaussianBlur(sprite[:, :, 3].copy(), (3, 3), 0)
    return sprite


def load_ferrari_asset(asset_path: str, width: int, height: int) -> Optional[np.ndarray]:
    path = Path(asset_path)
    if not path.exists():
        return None

    image = Image.open(path).convert("RGBA")
    asset = np.array(image)
    asset = cv2.cvtColor(asset, cv2.COLOR_RGBA2BGRA)

    alpha = asset[:, :, 3]
    if int(alpha.max()) == 255 and int(alpha.min()) == 255:
        bgr = asset[:, :, :3]
        dark_bg = np.all(bgr < 18, axis=2)
        alpha = np.where(dark_bg, 0, 255).astype(np.uint8)
        alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
        asset[:, :, 3] = alpha

    ys, xs = np.where(asset[:, :, 3] > 8)
    if len(xs) and len(ys):
        asset = asset[int(ys.min()):int(ys.max()) + 1, int(xs.min()):int(xs.max()) + 1]

    resized = cv2.resize(asset, (max(1, width), max(1, height)), interpolation=cv2.INTER_AREA)
    resized[:, :, 3] = cv2.GaussianBlur(resized[:, :, 3].copy(), (3, 3), 0)
    return resized


def place_asset_on_quad(frame: np.ndarray, asset: np.ndarray, quad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = frame.shape[:2]
    ah, aw = asset.shape[:2]
    src = np.array([[0, 0], [aw - 1, 0], [aw - 1, ah - 1], [0, ah - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    warped = cv2.warpPerspective(asset, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    alpha = warped[:, :, 3].astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
    result = warped[:, :, :3].astype(np.float32) * alpha[:, :, None] + frame.astype(np.float32) * (1.0 - alpha[:, :, None])
    return np.clip(result, 0, 255).astype(np.uint8), (alpha * 255).astype(np.uint8)


def estimate_car_quad(x0: int, y0: int, x1: int, y1: int, frame_width: int) -> np.ndarray:
    box_w = x1 - x0
    box_h = y1 - y0
    center_x = (x0 + x1) / 2.0
    view = (center_x / max(frame_width, 1)) - 0.5
    skew = int(box_w * 0.08 * view)

    return np.array([
        [x0 + int(box_w * 0.02) + skew, y0 + int(box_h * 0.08)],
        [x1 - int(box_w * 0.02) + skew, y0 + int(box_h * 0.18)],
        [x1 - int(box_w * 0.02), y1 - int(box_h * 0.03)],
        [x0 + int(box_w * 0.02), y1 - int(box_h * 0.02)],
    ], dtype=np.float32)


def foreground_occluder_mask(frame: np.ndarray, replacement_region: np.ndarray, sprite_alpha: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    region = replacement_region > 127

    orange = ((h <= 35) & (s >= 45) & (v >= 45)) | ((h >= 170) & (s >= 45) & (v >= 45))
    dark_clothes = (v < 85) & (s < 120)
    skin = (((h <= 22) | (h >= 160)) & (s >= 30) & (s <= 180) & (v >= 70))
    light_objects = (s < 45) & (v > 120)
    candidate = region & (~orange) & (dark_clothes | skin | light_objects) & (sprite_alpha > 8)

    ys, xs = np.where(region)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros(frame.shape[:2], dtype=np.uint8)

    y0, y1 = int(ys.min()), int(ys.max())
    min_top = y0 + int((y1 - y0) * 0.52)
    candidate_u8 = np.where(candidate, 255, 0).astype(np.uint8)
    components, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_u8, connectivity=8)

    keep = np.zeros_like(candidate_u8)
    for component_id in range(1, components):
        area = stats[component_id, cv2.CC_STAT_AREA]
        top = stats[component_id, cv2.CC_STAT_TOP]
        height = stats[component_id, cv2.CC_STAT_HEIGHT]
        width = stats[component_id, cv2.CC_STAT_WIDTH]
        if area >= 120 and width >= 6 and height >= 10 and top <= min_top:
            keep[labels == component_id] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cv2.GaussianBlur(keep, (5, 5), 0)


def add_contact_shadow(frame: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    shadow = np.zeros(frame.shape[:2], dtype=np.uint8)
    box_w = x1 - x0
    box_h = y1 - y0
    center = (int((x0 + x1) / 2), y1 - int(box_h * 0.04))
    axes = (max(8, int(box_w * 0.42)), max(4, int(box_h * 0.08)))
    cv2.ellipse(shadow, center, axes, 0, 0, 360, 120, -1)
    shadow = cv2.GaussianBlur(shadow, (31, 31), 0)
    alpha = (shadow.astype(np.float32) / 255.0)[:, :, None] * 0.45
    return np.clip(frame.astype(np.float32) * (1.0 - alpha), 0, 255).astype(np.uint8)


def match_asset_lighting(asset: np.ndarray, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return asset
    scene_v = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:, :, 2]
    asset_rgb = asset[:, :, :3]
    asset_alpha = asset[:, :, 3] > 8
    if not np.any(asset_alpha):
        return asset
    asset_v = cv2.cvtColor(asset_rgb, cv2.COLOR_BGR2HSV)[:, :, 2]
    scene_mean = float(np.mean(scene_v))
    asset_mean = float(np.mean(asset_v[asset_alpha]))
    gain = np.clip(scene_mean / max(asset_mean, 1.0), 0.65, 1.25)
    adjusted = asset.copy()
    adjusted[:, :, :3] = np.clip(adjusted[:, :, :3].astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return adjusted


def replace_with_ferrari_sprite(
    original_frame: np.ndarray,
    mask: np.ndarray,
    asset_path: str,
    tracked_bbox: Optional[tuple[int, int, int, int]] = None,
) -> np.ndarray:
    bbox = tracked_bbox or bbox_from_mask(mask)
    if bbox is None:
        return original_frame

    x0, y0, x1, y1 = bbox
    raw_bbox = bbox_from_mask(mask)
    if raw_bbox is not None:
        rx0, ry0, rx1, ry1 = raw_bbox
        x0 = int(round(x0 * 0.70 + rx0 * 0.30))
        y0 = int(round(y0 * 0.70 + ry0 * 0.30))
        x1 = int(round(x1 * 0.70 + rx1 * 0.30))
        y1 = int(round(y1 * 0.70 + ry1 * 0.30))

    x0 = max(0, min(original_frame.shape[1] - 2, x0))
    y0 = max(0, min(original_frame.shape[0] - 2, y0))
    x1 = max(x0 + 2, min(original_frame.shape[1] - 1, x1))
    y1 = max(y0 + 2, min(original_frame.shape[0] - 1, y1))
    box_w = x1 - x0
    box_h = y1 - y0
    if box_w < 20 or box_h < 20:
        return original_frame

    result = original_frame.copy()
    sprite_w = max(40, int(box_w * 1.06))
    sprite_h = max(24, int(box_h * 0.82))
    sprite_x0 = max(0, x0 - int((sprite_w - box_w) / 2))
    sprite_y0 = min(original_frame.shape[0] - sprite_h, y0 + int(box_h * 0.12))

    # Neutralize the original car region before compositing the replacement.
    underlay_alpha = np.zeros(result.shape[:2], dtype=np.uint8)
    body = np.array([
        [x0 + int(box_w * 0.04), y0 + int(box_h * 0.50)],
        [x0 + int(box_w * 0.24), y0 + int(box_h * 0.33)],
        [x0 + int(box_w * 0.72), y0 + int(box_h * 0.34)],
        [x0 + int(box_w * 0.98), y0 + int(box_h * 0.52)],
        [x0 + int(box_w * 0.92), y0 + int(box_h * 0.86)],
        [x0 + int(box_w * 0.10), y0 + int(box_h * 0.86)],
    ], dtype=np.int32)
    cv2.fillPoly(underlay_alpha, [body], 245)
    erase_mask = cv2.bitwise_or(underlay_alpha, cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))))
    result = cv2.inpaint(result, erase_mask, 7, cv2.INPAINT_TELEA)

    sprite = load_ferrari_asset(asset_path, sprite_w, sprite_h)
    if sprite is None:
        sprite = make_ferrari_sprite(sprite_w, sprite_h)

    sprite = match_asset_lighting(sprite, original_frame, bbox)
    result = add_contact_shadow(result, x0, y0, x1, y1)

    quad = estimate_car_quad(
        sprite_x0,
        sprite_y0,
        min(original_frame.shape[1] - 1, sprite_x0 + sprite_w),
        min(original_frame.shape[0] - 1, sprite_y0 + sprite_h),
        original_frame.shape[1],
    )
    result, sprite_alpha = place_asset_on_quad(result, sprite, quad)

    # Restore people/objects that are in front of the car, so the replacement sits in the scene.
    occluders = foreground_occluder_mask(original_frame, mask, sprite_alpha)
    occluder_alpha = (occluders.astype(np.float32) / 255.0)[:, :, None]
    result = np.clip(
        original_frame.astype(np.float32) * occluder_alpha + result.astype(np.float32) * (1.0 - occluder_alpha),
        0,
        255,
    ).astype(np.uint8)
    return result


def write_video(frame_paths: List[Path], output_video_path: Path, fps: float, width: int, height: int) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height), isColor=True)
    if not writer.isOpened():
        raise IOError(f"Could not open VideoWriter: {output_video_path}")

    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise IOError(f"Could not read output frame: {frame_path}")
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        writer.write(frame)

    writer.release()


def replace_locally(
    show_id: str,
    prompt: str,
    model_id: str,
    start_frame: int,
    max_frames: Optional[int],
    strength: float,
    guidance_scale: float,
    steps: int,
    mode: str,
    asset_path: str,
) -> Path:
    scene = load_scene_metadata(show_id)
    video_path = scene.get("filepath")
    if not video_path or not os.path.exists(video_path):
        raise FileNotFoundError(f"Video path does not exist: {video_path}")

    alpha_dir = MASKS_DIR / show_id / "alpha"
    if not alpha_dir.exists() or not list(alpha_dir.glob("frame_*.png")):
        generate_temporal_alpha_masks(show_id)

    frames, fps, width, height = read_frames(video_path)
    tracked_bboxes = build_smoothed_bboxes(alpha_dir, width, height, len(frames)) if mode == "sprite" else {}
    output_dir = OUTPUT_DIR / show_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = OUTPUT_DIR / f"{show_id}_local_replaced.mp4"

    pipe = None
    torch = None
    generator = None
    if mode == "diffusers":
        pipe, torch, device = load_diffusers_pipeline(model_id)
        generator = torch.Generator(device=device).manual_seed(SEED)

    frame_paths: List[Path] = []
    end_frame = len(frames) if max_frames is None else min(len(frames), start_frame + max_frames)

    for idx, original_frame in enumerate(frames[:end_frame]):
        output_path = output_dir / f"frame_{idx:04d}.jpg"

        if idx < start_frame:
            cv2.imwrite(str(output_path), original_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            frame_paths.append(output_path)
            continue

        mask = load_mask(alpha_dir / f"frame_{idx:04d}.png", width, height)
        if mask is None or int(cv2.countNonZero(mask)) <= MASK_MIN_PIXELS:
            cv2.imwrite(str(output_path), original_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            frame_paths.append(output_path)
            continue

        if mode == "recolor":
            print(f"[LOCAL-REPLACE] Recoloring frame {idx}/{end_frame - 1}")
            result = recolor_to_red_ferrari(original_frame, mask)
        elif mode == "sprite":
            print(f"[LOCAL-REPLACE] Replacing frame {idx}/{end_frame - 1} with Ferrari sprite")
            result = replace_with_ferrari_sprite(original_frame, mask, asset_path, tracked_bboxes.get(idx))
        else:
            print(f"[LOCAL-REPLACE] Inpainting frame {idx}/{end_frame - 1}")
            generated = pipe(
                prompt=prompt,
                image=frame_to_image(original_frame),
                mask_image=mask_to_image(mask),
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                generator=generator,
            ).images[0]
            result = composite(original_frame, generated, mask)
        cv2.imwrite(str(output_path), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frame_paths.append(output_path)

    write_video(frame_paths, output_video_path, fps, width, height)
    return output_video_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Precompute replacement frames locally with diffusers.")
    parser.add_argument("--show-id", default="orange_car")
    parser.add_argument("--prompt", default="a red Ferrari sports car, realistic, matching the scene lighting and perspective")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--strength", type=float, default=0.85)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--asset", default=DEFAULT_ASSET, help="Transparent Ferrari image used by sprite mode.")
    parser.add_argument(
        "--mode",
        choices=["sprite", "recolor", "diffusers"],
        default="sprite",
        help="sprite fully covers the old car with a generated red sports car; recolor is color-only; diffusers runs local inpainting.",
    )
    args = parser.parse_args()

    output = replace_locally(
        show_id=args.show_id,
        prompt=args.prompt,
        model_id=args.model,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        steps=args.steps,
        mode=args.mode,
        asset_path=args.asset,
    )
    print(f"[LOCAL-REPLACE] Output video: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
