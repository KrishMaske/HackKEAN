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


def expand_orange_anchor_to_car_bbox(anchor_bbox: List[int], width: int, height: int) -> List[int]:
    x0, y0, x1, y1 = anchor_bbox
    anchor_w = x1 - x0
    anchor_h = y1 - y0
    expanded = [
        x0 - int(anchor_w * 0.08),
        y0 - int(anchor_h * 0.70),
        x1 + int(anchor_w * 0.08),
        y1 + int(anchor_h * 0.30),
    ]
    return clamp_bbox(expanded, width, height) or anchor_bbox


def orange_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([4, 70, 70], dtype=np.uint8)
    upper_orange = np.array([28, 255, 255], dtype=np.uint8)
    lower_red_orange = np.array([0, 80, 60], dtype=np.uint8)
    upper_red_orange = np.array([8, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_orange, upper_orange) | cv2.inRange(hsv, lower_red_orange, upper_red_orange)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)


def orange_ratio(frame: np.ndarray, bbox: Optional[List[int]]) -> float:
    if not bbox:
        return 0.0
    x0, y0, x1, y1 = bbox
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0
    mask = orange_mask(roi)
    return float(cv2.countNonZero(mask)) / float(mask.shape[0] * mask.shape[1])


def detect_by_color(frame: np.ndarray, min_area_ratio: float = 0.001) -> Dict[str, Any]:
    height, width = frame.shape[:2]
    mask = orange_mask(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = width * height * min_area_ratio
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 5 or h <= 5:
            continue
        aspect = w / float(h)
        if aspect < 1.15 or aspect > 5.5:
            continue
        orange_anchor = [x, y, x + w, y + h]
        candidates.append((area, orange_anchor))

    if not candidates:
        return {"found": False, "confidence": 0.0, "bbox": None, "notes": "no orange car-sized region found"}

    main_area, main_anchor = max(candidates, key=lambda item: item[0])
    mx0, my0, mx1, my1 = main_anchor
    main_cy = (my0 + my1) / 2.0
    main_h = my1 - my0
    merged = [mx0, my0, mx1, my1]
    merged_area = main_area

    for area, anchor in candidates:
        if anchor == main_anchor:
            continue
        x0, y0, x1, y1 = anchor
        cy = (y0 + y1) / 2.0
        h = y1 - y0
        vertical_match = abs(cy - main_cy) <= max(main_h, h) * 2.5
        horizontal_reach = min(abs(x0 - mx1), abs(mx0 - x1), abs(x0 - mx0)) <= width * 0.65
        if vertical_match and horizontal_reach:
            merged = [min(merged[0], x0), min(merged[1], y0), max(merged[2], x1), max(merged[3], y1)]
            merged_area += area

    bbox = expand_orange_anchor_to_car_bbox(merged, width, height)
    score = min(1.0, max(0.05, merged_area / (width * height * 0.08)))
    return {
        "found": True,
        "confidence": round(score, 3),
        "bbox": bbox,
        "orange_anchor_bbox": merged,
        "notes": "local whole-car bbox expanded from orange body anchor",
    }


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


def detect_with_groq(frame: np.ndarray, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    height, width = frame.shape[:2]
    b64 = _encode_frame(frame)
    prompt = f"""You are a precision video object detector.

Task: detect THE ENTIRE ORANGE CAR in this video frame. The whole visible car is the replacement target.

Image size: {width}x{height} pixels.

Rules:
- Only detect the orange car. Do not return a different car, a person, a background object, flames, lights, or road markings.
- Return the tight visible bounding box around the WHOLE visible car, not just the orange paint.
- Include the full visible silhouette: hood, roof, windshield/windows, wheels/tires, bumper, spoiler, and attached car shadow.
- If the orange car is partially visible, box the whole visible part of the car.
- If the orange car is not visible, set found to false and all coordinates to null.
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


def analyze_orange_car_seconds(
    video_path: str,
    mode: str = "hybrid",
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
    groq_disabled = mode == "color"

    for idx, second in enumerate(seconds, start=1):
        frame_index = min(int(round(second * fps)), max(0, frame_count - 1))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            continue

        detector = "color"
        if mode in {"groq", "hybrid"} and not groq_disabled:
            try:
                result = detect_with_groq(frame, model=model)
                detector = "groq"
            except DetectorRateLimitError as exc:
                if mode == "groq":
                    raise
                print(f"[DETECT] Groq rate-limited at {second}s; using local detector for remaining seconds: {exc}")
                groq_disabled = True
                result = detect_by_color(frame)
                detector = "color-after-rate-limit"
            except Exception as exc:
                if mode == "groq":
                    raise
                print(f"[DETECT] Groq failed at {second}s; using local detector: {exc}")
                result = detect_by_color(frame)
                detector = "color-fallback"
        else:
            result = detect_by_color(frame)
            detector = "color" if mode == "color" else "color-after-rate-limit"

        bbox = clamp_bbox(result.get("bbox"), width, height)
        orange_anchor_bbox = clamp_bbox(result.get("orange_anchor_bbox"), width, height)
        detection = {
            "second": second,
            "timestamp_ms": int(second * 1000),
            "frame_index": frame_index,
            "detector": detector,
            "found": bool(result.get("found") and bbox),
            "status": "ok" if detector == "groq" and result.get("found") and bbox else ("review" if bbox else "missing"),
            "confidence": float(result.get("confidence") or 0.0),
            "bbox": bbox,
            "orange_anchor_bbox": orange_anchor_bbox,
            "orange_anchor_ratio": round(orange_ratio(frame, bbox), 4),
            "notes": result.get("notes", ""),
        }
        detections.append(detection)

        if sleep_seconds > 0 and idx < len(seconds) and detector == "groq":
            time.sleep(sleep_seconds)

    capture.release()

    first_found = next((item for item in detections if item.get("bbox")), None)
    return {
        "target": "whole visible orange car",
        "sampling": "one frame per second",
        "mode": mode,
        "model": model if mode in {"groq", "hybrid"} else None,
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
