import argparse
import base64
import csv
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from dotenv import load_dotenv


DEFAULT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
DEFAULT_VIDEO = "assets/input/fastnfurious.mp4"
DEFAULT_OUT_DIR = "assets/detections/orange_car"


class DetectorRateLimitError(RuntimeError):
    pass


def parse_json_response(text: str) -> Dict[str, Any]:
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


def encode_frame_for_api(frame: np.ndarray, max_size: int, jpeg_quality: int) -> str:
    h, w = frame.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise RuntimeError("Failed to encode frame as JPEG.")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def clamp_bbox(
    bbox: Optional[List[Any]],
    width: int,
    height: int,
) -> Optional[List[int]]:
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
    """Expand an orange paint/body anchor to the full visible car silhouette."""
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


def normalized_to_pixel_bbox(data: Dict[str, Any], width: int, height: int) -> Optional[List[int]]:
    values = [data.get("x_min"), data.get("y_min"), data.get("x_max"), data.get("y_max")]
    if any(v is None for v in values):
        return None

    x0, y0, x1, y1 = [float(v) for v in values]
    bbox = [
        int(round(x0 * width / 1000.0)),
        int(round(y0 * height / 1000.0)),
        int(round(x1 * width / 1000.0)),
        int(round(y1 * height / 1000.0)),
    ]
    return clamp_bbox(bbox, width, height)


def orange_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Broad orange/red-orange range. Kept conservative enough to avoid skin and tail lights.
    lower_orange = np.array([4, 70, 70], dtype=np.uint8)
    upper_orange = np.array([28, 255, 255], dtype=np.uint8)
    lower_red_orange = np.array([0, 80, 60], dtype=np.uint8)
    upper_red_orange = np.array([8, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_orange, upper_orange) | cv2.inRange(hsv, lower_red_orange, upper_red_orange)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


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


def detect_with_groq(
    client: Any,
    frame: np.ndarray,
    model: str,
    max_image_size: int,
    jpeg_quality: int,
) -> Dict[str, Any]:
    height, width = frame.shape[:2]
    b64 = encode_frame_for_api(frame, max_image_size, jpeg_quality)
    prompt = f"""You are a precision video object detector.

Task: detect THE ENTIRE ORANGE CAR in this video frame. The whole visible car is the replacement target.

Image size: {width}x{height} pixels.

Rules:
- Only detect the orange car. Do not return a different car, a person, a background object, flames, lights, or road markings.
- Return the tight visible bounding box around the WHOLE visible car, not just the orange paint.
- Include the full visible silhouette: hood, roof, windshield/windows, wheels/tires, bumper, spoiler, and shadows attached to the car.
- If the orange car is partially visible, box the whole visible part of the car.
- If the orange car is not visible, set found to false and all coordinates to null.
- Use NORMALIZED coordinates from 0 to 1000, not pixels.
- Include a confidence from 0.0 to 1.0.

Respond with ONLY valid JSON:
{{"found": bool, "confidence": float, "x_min": int|null, "y_min": int|null, "x_max": int|null, "y_max": int|null, "notes": "short reason"}}
"""

    try:
        response = client.chat.completions.create(
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

    data = parse_json_response(response.choices[0].message.content)
    found = bool(data.get("found"))
    bbox = normalized_to_pixel_bbox(data, width, height) if found else None
    confidence = float(data.get("confidence") or 0.0)

    return {
        "found": found and bbox is not None,
        "confidence": max(0.0, min(confidence, 1.0)),
        "bbox": bbox,
        "notes": str(data.get("notes") or ""),
        "raw": data,
    }


def draw_detection(frame: np.ndarray, detection: Dict[str, Any]) -> np.ndarray:
    annotated = frame.copy()
    bbox = detection.get("bbox")
    status = detection.get("status", "missing")
    second = detection["second"]
    confidence = detection.get("confidence", 0.0)
    ratio = detection.get("orange_anchor_ratio", 0.0)

    if bbox:
        if status == "ok":
            color = (0, 220, 0)
        elif status == "review":
            color = (0, 180, 255)
        else:
            color = (0, 0, 255)

        x0, y0, x1, y1 = bbox
        cv2.rectangle(annotated, (x0, y0), (x1, y1), color, 3)
    else:
        color = (0, 0, 255)

    label = f"{second:04d}s {status} conf={confidence:.2f} anchor={ratio:.2f}"
    cv2.rectangle(annotated, (8, 8), (min(760, annotated.shape[1] - 8), 44), (0, 0, 0), -1)
    cv2.putText(annotated, label, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)
    return annotated


def make_contact_sheet(image_paths: List[Path], output_path: Path, thumb_width: int = 320) -> None:
    thumbs = []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = thumb_width / float(w)
        thumbs.append(cv2.resize(img, (thumb_width, int(h * scale)), interpolation=cv2.INTER_AREA))

    if not thumbs:
        return

    cols = min(4, len(thumbs))
    rows = int(math.ceil(len(thumbs) / cols))
    thumb_h = max(img.shape[0] for img in thumbs)
    sheet = np.zeros((rows * thumb_h, cols * thumb_width, 3), dtype=np.uint8)

    for idx, thumb in enumerate(thumbs):
        row = idx // cols
        col = idx % cols
        y = row * thumb_h
        x = col * thumb_width
        sheet[y : y + thumb.shape[0], x : x + thumb.shape[1]] = thumb

    cv2.imwrite(str(output_path), sheet)


def extract_second_frame(capture: cv2.VideoCapture, frame_index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame index {frame_index}.")
    return frame


def classify_detection(detection: Dict[str, Any], min_confidence: float, min_orange_anchor_ratio: float) -> str:
    if not detection.get("found") or not detection.get("bbox"):
        return "missing"
    if detection.get("detector") != "groq":
        return "review"
    if detection.get("confidence", 0.0) < min_confidence:
        return "review"
    if detection.get("orange_anchor_ratio", 0.0) < min_orange_anchor_ratio:
        return "review"
    return "ok"


def run(args: argparse.Namespace) -> int:
    load_dotenv()
    load_dotenv(".env")

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    out_dir = Path(args.out)
    frames_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps else 0.0
    seconds = list(range(0, int(math.ceil(duration))))
    if args.limit is not None:
        seconds = seconds[: args.limit]

    client = None
    if args.mode in {"groq", "hybrid"}:
        from groq import Groq

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is required for groq/hybrid mode.")
        client = Groq(api_key=api_key)

    detections: List[Dict[str, Any]] = []
    annotated_paths: List[Path] = []

    print(f"[DETECT] Video: {video_path}")
    print(f"[DETECT] {width}x{height} @ {fps:.3f} fps, {frame_count} frames, {duration:.2f}s")
    print(f"[DETECT] Sampling {len(seconds)} frames, exactly one frame per second.")

    groq_disabled = False

    for idx, second in enumerate(seconds, start=1):
        frame_index = min(int(round(second * fps)), max(0, frame_count - 1))
        frame = extract_second_frame(capture, frame_index)

        if args.mode == "color" or groq_disabled:
            result = detect_by_color(frame)
            detector = "color" if args.mode == "color" else "color-after-rate-limit"
        else:
            try:
                result = detect_with_groq(client, frame, args.model, args.max_image_size, args.jpeg_quality)
                detector = "groq"
            except DetectorRateLimitError as exc:
                if args.mode != "hybrid":
                    raise
                print(f"[DETECT] {second:04d}s Groq rate-limited; using color detector for this and remaining seconds.")
                print(f"[DETECT] Rate-limit detail: {exc}")
                groq_disabled = True
                result = detect_by_color(frame)
                detector = "color-after-rate-limit"
            except Exception as exc:
                if args.mode != "hybrid":
                    raise
                print(f"[DETECT] {second:04d}s Groq failed, falling back to color detector: {exc}")
                result = detect_by_color(frame)
                detector = "color-fallback"

        bbox = clamp_bbox(result.get("bbox"), width, height)
        orange_anchor_bbox = clamp_bbox(result.get("orange_anchor_bbox"), width, height)
        ratio = orange_ratio(frame, bbox)
        detection = {
            "second": second,
            "timestamp_ms": int(second * 1000),
            "frame_index": frame_index,
            "detector": detector,
            "found": bool(result.get("found") and bbox),
            "confidence": float(result.get("confidence") or 0.0),
            "bbox": bbox,
            "orange_anchor_bbox": orange_anchor_bbox,
            "orange_anchor_ratio": round(ratio, 4),
            "notes": result.get("notes", ""),
        }
        detection["status"] = classify_detection(detection, args.min_confidence, args.min_orange_anchor_ratio)

        annotated = draw_detection(frame, detection)
        annotated_path = frames_dir / f"second_{second:04d}.jpg"
        cv2.imwrite(str(annotated_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
        detection["annotated_frame"] = str(annotated_path)

        detections.append(detection)
        annotated_paths.append(annotated_path)

        bbox_text = detection["bbox"] if detection["bbox"] else "none"
        print(
            f"[DETECT] {idx:03d}/{len(seconds):03d} "
            f"{second:04d}s status={detection['status']} conf={detection['confidence']:.2f} "
            f"anchor={detection['orange_anchor_ratio']:.2f} bbox={bbox_text}"
        )

        if args.sleep > 0 and idx < len(seconds):
            time.sleep(args.sleep)

    capture.release()

    summary = {
        "video": str(video_path),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": duration,
        "sampling": "one frame per second",
        "target": "whole visible orange car",
        "mode": args.mode,
        "model": args.model if args.mode in {"groq", "hybrid"} else None,
        "counts": {
            "total_seconds": len(detections),
            "ok": sum(1 for item in detections if item["status"] == "ok"),
            "review": sum(1 for item in detections if item["status"] == "review"),
            "missing": sum(1 for item in detections if item["status"] == "missing"),
        },
        "detections": detections,
    }

    json_path = out_dir / "orange_car_second_by_second.json"
    csv_path = out_dir / "orange_car_second_by_second.csv"
    sheet_path = out_dir / "orange_car_contact_sheet.jpg"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "second",
                "timestamp_ms",
                "frame_index",
                "detector",
                "found",
                "status",
                "confidence",
                "orange_anchor_ratio",
                "orange_anchor_bbox",
                "bbox",
                "notes",
                "annotated_frame",
            ],
        )
        writer.writeheader()
        for row in detections:
            writer.writerow(row)

    make_contact_sheet(annotated_paths, sheet_path)

    print(f"[DETECT] Wrote JSON: {json_path}")
    print(f"[DETECT] Wrote CSV: {csv_path}")
    print(f"[DETECT] Wrote annotated frames: {frames_dir}")
    print(f"[DETECT] Wrote contact sheet: {sheet_path}")
    print(f"[DETECT] Counts: {summary['counts']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect the whole visible orange car once per second across an entire video.",
    )
    parser.add_argument("--video", default=DEFAULT_VIDEO, help="Path to the input video.")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR, help="Directory for detections and annotated frames.")
    parser.add_argument("--mode", choices=["hybrid", "groq", "color"], default="hybrid")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Groq vision model for groq/hybrid mode.")
    parser.add_argument("--max-image-size", type=int, default=1024, help="Max long edge sent to the vision model.")
    parser.add_argument("--jpeg-quality", type=int, default=88, help="JPEG quality sent to the vision model.")
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--min-orange-anchor-ratio", type=float, default=0.015)
    parser.add_argument("--sleep", type=float, default=0.5, help="Delay between vision API calls to reduce rate-limit risk.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of seconds to process for quick checks.")
    return parser


if __name__ == "__main__":
    try:
        raise SystemExit(run(build_parser().parse_args()))
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"[DETECT] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
