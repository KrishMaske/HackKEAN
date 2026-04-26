import base64
import json
import math
import os
import re
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from app.core.config import settings


DEFAULT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


class DetectorRateLimitError(RuntimeError):
    pass


def _parse_json_response(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def clamp_bbox(bbox: Optional[List[Any]], width: int, height: int) -> Optional[List[int]]:
    if not bbox or len(bbox) != 4 or any(v is None for v in bbox):
        return None

    x0, y0, x1, y1 = [int(round(float(v))) for v in bbox]
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def _encode_frame(frame: np.ndarray, max_size: int = 1024, jpeg_quality: int = 88) -> str:
    h, w = frame.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise RuntimeError("Failed to encode frame as JPEG.")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def _normalized_to_pixel_bbox(data: Dict[str, Any], width: int, height: int) -> Optional[List[int]]:
    values = [data.get("x_min"), data.get("y_min"), data.get("x_max"), data.get("y_max")]
    if any(v is None for v in values):
        return None
    x0, y0, x1, y1 = [float(v) for v in values]
    return clamp_bbox(
        [
            int(round(x0 * width / 1000.0)),
            int(round(y0 * height / 1000.0)),
            int(round(x1 * width / 1000.0)),
            int(round(y1 * height / 1000.0)),
        ],
        width,
        height,
    )


def detect_with_groq(frame: np.ndarray, target: str, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    height, width = frame.shape[:2]
    b64 = _encode_frame(frame)
    prompt = f"""You are a precision video object detector.

Task: detect the entire visible "{target}" in this video frame. This is the tracked product/object.

Image size: {width}x{height} pixels.

Rules:
- Only detect "{target}". Do not return a different product, person, background object, light, reflection, or nearby prop.
- Return the tight visible bounding box around the WHOLE visible product/object, not just one colored or branded region.
- If the product/object is partially visible, box the whole visible part.
- If the product/object is not visible, set found to false and all coordinates to null.
- Use NORMALIZED coordinates from 0 to 1000, not pixels.

Respond with ONLY valid JSON:
{{"found": bool, "confidence": float, "x_min": int|null, "y_min": int|null, "x_max": int|null, "y_max": int|null, "notes": "short reason"}}
"""
    try:
        response = settings.groq_client.chat.completions.create(
            model=model,
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
            temperature=0.0,
        )
    except Exception as exc:
        message = str(exc).lower()
        if "429" in message or "rate limit" in message or "rate_limit" in message:
            raise DetectorRateLimitError(str(exc)) from exc
        raise

    data = _parse_json_response(response.choices[0].message.content)
    bbox = _normalized_to_pixel_bbox(data, width, height) if data.get("found") else None
    confidence = max(0.0, min(float(data.get("confidence") or 0.0), 1.0))
    return {
        "found": bool(data.get("found") and bbox),
        "confidence": confidence,
        "bbox": bbox,
        "notes": str(data.get("notes") or ""),
    }


def analyze_product_seconds(
    video_path: str,
    target: str,
    mode: str = "groq",
    sleep_seconds: float = 0.5,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps else 0.0
    seconds = list(range(0, int(math.ceil(duration))))

    detections = []
    target = (target or "prominent product").strip()

    for idx, second in enumerate(seconds, start=1):
        frame_index = min(int(round(second * fps)), max(0, frame_count - 1))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            continue

        try:
            result = detect_with_groq(frame, target=target, model=model)
            detector = "groq"
        except Exception as exc:
            print(f"[DETECT] Groq failed at {second}s: {exc}")
            result = {"found": False, "confidence": 0.0, "bbox": None, "notes": f"Error: {str(exc)}"}
            detector = "error"

        bbox = clamp_bbox(result.get("bbox"), width, height)
        detection = {
            "second": second,
            "timestamp_ms": int(second * 1000),
            "frame_index": frame_index,
            "detector": detector,
            "found": bool(result.get("found") and bbox),
            "status": "ok" if result.get("found") and bbox else "missing",
            "confidence": float(result.get("confidence") or 0.0),
            "bbox": bbox,
            "notes": result.get("notes", ""),
        }
        detections.append(detection)

        if sleep_seconds > 0 and idx < len(seconds):
            time.sleep(sleep_seconds)

    capture.release()

    first_found = next((item for item in detections if item.get("bbox")), None)
    return {
        "target": target,
        "sampling": "one frame per second",
        "mode": mode,
        "model": model,
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": duration,
        "first_detected_second": first_found.get("second") if first_found else None,
        "first_detected_frame": first_found.get("frame_index") if first_found else None,
        "initial_bounding_box": first_found.get("bbox") if first_found else [None, None, None, None],
        "detections": detections,
    }

