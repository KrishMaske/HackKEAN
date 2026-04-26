import base64
import os
import time
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import requests
from PIL import Image

from app.core.config import settings
from app.services.masking import generate_temporal_alpha_masks, load_scene_metadata


MASKS_DIR = "assets/masks"
OUTPUT_DIR = "assets/output"
REPLICATE_MODEL = "stability-ai/stable-diffusion-inpainting"
REPLICATE_VERSION = "95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3"
REPLICATE_PREDICTIONS_URL = f"https://api.replicate.com/v1/models/{REPLICATE_MODEL}/predictions"
REPLICATE_VERSIONED_PREDICTIONS_URL = "https://api.replicate.com/v1/predictions"
MASK_MIN_PIXELS = 100
SEED = 42


def _encode_image_data_url(image: Image.Image, image_format: str) -> str:
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    mime = "image/jpeg" if image_format.upper() in {"JPEG", "JPG"} else "image/png"
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def _frame_to_pil(frame: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _mask_to_pil(mask: np.ndarray, width: int, height: int) -> Image.Image:
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    mask = np.where(mask > 127, 255, 0).astype(np.uint8)
    return Image.fromarray(mask, mode="L")


def _download_replicate_output(output: object) -> Image.Image:
    if isinstance(output, list):
        if not output:
            raise RuntimeError("Replicate returned an empty output list.")
        output = output[0]

    if isinstance(output, str):
        response = requests.get(output, timeout=120)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")

    raise RuntimeError(f"Unsupported Replicate output type: {type(output).__name__}")


def _wait_for_prediction(prediction_url: str, headers: dict) -> object:
    while True:
        response = requests.get(prediction_url, headers=headers, timeout=120)
        response.raise_for_status()
        data = response.json()
        status = data.get("status")

        if status == "succeeded":
            return data.get("output")
        if status in {"failed", "canceled"}:
            raise RuntimeError(f"Replicate prediction {status}: {data.get('error')}")

        time.sleep(1.0)


def _run_replicate_inpaint(frame: np.ndarray, mask: np.ndarray, prompt: str) -> Image.Image:
    if not settings.replicate_api_token:
        raise ValueError("REPLICATE_API_TOKEN is not set")

    height, width = frame.shape[:2]
    frame_image = _frame_to_pil(frame)
    mask_image = _mask_to_pil(mask, width, height)

    headers = {
        "Authorization": f"Bearer {settings.replicate_api_token}",
        "Content-Type": "application/json",
        "Prefer": "wait=60",
    }
    payload = {
        "input": {
            "prompt": prompt,
            "image": _encode_image_data_url(frame_image, "JPEG"),
            "mask": _encode_image_data_url(mask_image, "PNG"),
            "seed": SEED,
        }
    }

    response = requests.post(REPLICATE_PREDICTIONS_URL, headers=headers, json=payload, timeout=120)
    if response.status_code == 404:
        versioned_payload = {"version": REPLICATE_VERSION, **payload}
        response = requests.post(REPLICATE_VERSIONED_PREDICTIONS_URL, headers=headers, json=versioned_payload, timeout=120)
    if response.status_code == 402:
        try:
            detail = response.json().get("detail") or response.text
        except ValueError:
            detail = response.text
        raise RuntimeError(f"Replicate billing/credit error: {detail}")
    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        detail = f"Replicate rate limit reached. Retry later"
        if retry_after:
            detail += f" after {retry_after} seconds"
        raise RuntimeError(detail)
    response.raise_for_status()
    data = response.json()

    output = data.get("output")
    if output is None:
        prediction_url = data.get("urls", {}).get("get")
        if not prediction_url:
            raise RuntimeError("Replicate response did not include output or polling URL.")
        output = _wait_for_prediction(prediction_url, headers)

    return _download_replicate_output(output)


def _soft_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    mask = np.where(mask > 127, 255, 0).astype(np.uint8)
    blurred = cv2.GaussianBlur(mask, (7, 7), 0)
    alpha = blurred.astype(np.float32) / 255.0
    return alpha[:, :, None]


def _composite(original_frame: np.ndarray, inpainted_image: Image.Image, mask: np.ndarray) -> np.ndarray:
    height, width = original_frame.shape[:2]
    inpainted = np.array(inpainted_image.resize((width, height), Image.Resampling.LANCZOS))
    inpainted = cv2.cvtColor(inpainted, cv2.COLOR_RGB2BGR).astype(np.float32)
    original = original_frame.astype(np.float32)
    alpha = _soft_mask(mask, width, height)
    result = (inpainted * alpha) + (original * (1.0 - alpha))
    return np.clip(result, 0, 255).astype(np.uint8)


def _read_original_frames(video_path: str) -> tuple[List[np.ndarray], float, int, int]:
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


def _write_video(frame_paths: List[Path], output_video_path: str, fps: float, width: int, height: int) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=True)
    if not writer.isOpened():
        raise IOError(f"Unable to open VideoWriter for {output_video_path}")

    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise IOError(f"Unable to read output frame: {frame_path}")
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        writer.write(frame)

    writer.release()


def replace_product(show_id: str, replacement_prompt: str) -> str:
    """Returns path to the output video file."""
    scene = load_scene_metadata(show_id)
    video_path = scene.get("filepath")
    if not video_path:
        raise ValueError("Scene metadata is missing a valid video filepath.")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video path does not exist: {video_path}")

    alpha_dir = Path(MASKS_DIR) / show_id / "alpha"
    if not alpha_dir.exists() or not list(alpha_dir.glob("frame_*.png")):
        generate_temporal_alpha_masks(show_id)

    frames, fps, width, height = _read_original_frames(video_path)
    show_output_dir = Path(OUTPUT_DIR) / show_id
    show_output_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = str(Path(OUTPUT_DIR) / f"{show_id}_replaced.mp4")
    output_frame_paths: List[Path] = []

    max_frames: Optional[int] = None
    if os.getenv("REPLACEMENT_MAX_FRAMES"):
        max_frames = max(0, int(os.getenv("REPLACEMENT_MAX_FRAMES", "0")))

    for idx, original_frame in enumerate(frames):
        if max_frames is not None and idx >= max_frames:
            break

        output_frame_path = show_output_dir / f"frame_{idx:04d}.jpg"
        mask_path = alpha_dir / f"frame_{idx:04d}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None

        if mask is None or int(cv2.countNonZero(mask)) <= MASK_MIN_PIXELS:
            cv2.imwrite(str(output_frame_path), original_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            output_frame_paths.append(output_frame_path)
            continue

        print(f"[REPLACE] Inpainting frame {idx}/{len(frames)}")
        inpainted = _run_replicate_inpaint(original_frame, mask, replacement_prompt)
        composited = _composite(original_frame, inpainted, mask)
        cv2.imwrite(str(output_frame_path), composited, [cv2.IMWRITE_JPEG_QUALITY, 95])
        output_frame_paths.append(output_frame_path)

    if not output_frame_paths:
        raise RuntimeError("No frames were written for replacement output.")

    _write_video(output_frame_paths, output_video_path, fps, width, height)
    return output_video_path
