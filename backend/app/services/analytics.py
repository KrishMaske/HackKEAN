import json
import math
import os
from pathlib import Path
from typing import Any, Optional


DATA_DIR = Path("app/db/data")


def _load_scene(show_id: str) -> dict[str, Any]:
    path = DATA_DIR / f"{show_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Scene metadata not found for show_id='{show_id}'")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _bbox_area(bbox: Optional[list[int]]) -> int:
    if not bbox or len(bbox) != 4:
        return 0
    x0, y0, x1, y1 = bbox
    return max(0, x1 - x0) * max(0, y1 - y0)


def _bbox_center(bbox: list[int]) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def _position_label(cx: float, cy: float, width: int, height: int) -> str:
    horizontal = "left" if cx < width * 0.33 else "right" if cx > width * 0.66 else "center"
    vertical = "upper" if cy < height * 0.33 else "lower" if cy > height * 0.66 else "middle"
    return f"{vertical} {horizontal}"


def _segments(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    active_start = None
    active_end = None

    for item in detections:
        second = int(item.get("second", 0))
        if item.get("found"):
            if active_start is None:
                active_start = second
            active_end = second
        elif active_start is not None:
            segments.append(
                {
                    "start_second": active_start,
                    "end_second": active_end,
                    "duration_seconds": max(1, active_end - active_start + 1),
                }
            )
            active_start = None
            active_end = None

    if active_start is not None:
        segments.append(
            {
                "start_second": active_start,
                "end_second": active_end,
                "duration_seconds": max(1, active_end - active_start + 1),
            }
        )
    return segments


def _timeline(detections: list[dict[str, Any]], width: int, height: int) -> list[dict[str, Any]]:
    frame_area = max(1, width * height)
    timeline = []
    previous_center = None
    for item in detections:
        bbox = item.get("bbox")
        area = _bbox_area(bbox)
        coverage = area / frame_area
        center = _bbox_center(bbox) if bbox else None
        movement = 0.0
        if center and previous_center:
            movement = math.dist(center, previous_center)
        if center:
            previous_center = center

        timeline.append(
            {
                "second": int(item.get("second", 0)),
                "timestamp_ms": int(item.get("timestamp_ms", 0)),
                "frame_index": int(item.get("frame_index", 0)),
                "found": bool(item.get("found")),
                "confidence": float(item.get("confidence") or 0.0),
                "bbox": bbox,
                "screen_coverage": round(coverage, 4),
                "position": _position_label(center[0], center[1], width, height) if center else "off screen",
                "movement_from_previous_px": round(movement, 2),
                "status": item.get("status", "unknown"),
            }
        )
    return timeline


def _key_moments(timeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    visible = [item for item in timeline if item["found"]]
    if not visible:
        return [{"second": 0, "type": "not_detected", "label": "Product was not detected in this clip."}]

    peak = max(visible, key=lambda item: item["screen_coverage"])
    most_confident = max(visible, key=lambda item: item["confidence"])
    moments = [
        {
            "second": visible[0]["second"],
            "type": "entry",
            "label": f"Product first appears around {visible[0]['second']}s.",
        },
        {
            "second": peak["second"],
            "type": "peak_screen_time",
            "label": f"Largest on-screen presence at {peak['second']}s ({peak['screen_coverage'] * 100:.1f}% of frame).",
        },
    ]
    if most_confident["second"] != peak["second"]:
        moments.append(
            {
                "second": most_confident["second"],
                "type": "cleanest_detection",
                "label": f"Cleanest detection at {most_confident['second']}s.",
            }
        )
    moments.append(
        {
            "second": visible[-1]["second"],
            "type": "exit",
            "label": f"Last detected around {visible[-1]['second']}s.",
        }
    )
    return moments


def _interaction_insights(timeline: list[dict[str, Any]], width: int) -> list[dict[str, Any]]:
    visible = [item for item in timeline if item["found"]]
    if not visible:
        return [{"label": "No product interaction can be inferred because the product was not detected.", "confidence": "low"}]

    peak = max(visible, key=lambda item: item["screen_coverage"])
    high_motion = [item for item in visible if item["movement_from_previous_px"] > width * 0.18]
    low_conf = [item for item in visible if item["confidence"] < 0.12]
    insights = [
        {
            "label": "The product is treated as a dominant scene object when it expands to a large portion of the frame.",
            "evidence": f"Peak coverage is {peak['screen_coverage'] * 100:.1f}% at {peak['second']}s.",
            "confidence": "medium",
        }
    ]
    if high_motion:
        insights.append(
            {
                "label": "Camera motion or scene movement changes the product viewpoint significantly.",
                "evidence": f"Largest second-to-second shift occurs near {max(high_motion, key=lambda item: item['movement_from_previous_px'])['second']}s.",
                "confidence": "medium",
            }
        )
    if low_conf:
        insights.append(
            {
                "label": "Some moments likely contain occlusion, partial visibility, or motion blur.",
                "evidence": f"{len(low_conf)} visible samples have low detector confidence.",
                "confidence": "medium",
            }
        )
    insights.append(
        {
            "label": "Character-level interaction analysis needs person tracking layered on top of this product track.",
            "evidence": "Current analytics measure product visibility, position, size, and motion; person-product proximity is the next detector layer.",
            "confidence": "high",
        }
    )
    return insights


async def build_product_analytics(show_id: str) -> dict[str, Any]:
    scene = _load_scene(show_id)
    detection_block = scene.get("second_by_second_detection") or {}
    detections = detection_block.get("detections") or []
    
    # If rendering isn't done yet, return a partial state
    if not detections:
        video_name = os.path.basename(scene.get("filepath", ""))
        return {
            "show_id": show_id,
            "status": "processing",
            "product": scene.get("target_object") or "tracked product",
            "video": {
                "path": scene.get("filepath"),
                "url": f"/uploads/{video_name}" if video_name else None,
                "width": scene.get("video_width", 1),
                "height": scene.get("video_height", 1),
                "duration_seconds": scene.get("total_frames", 0) / scene.get("video_fps", 30.0) if scene.get("video_fps") else 0
            },
            "summary": {"detected_seconds": 0, "visibility_rate": 0},
            "scene_understanding": {"headline": "Analysis in progress...", "key_moments": [], "interaction_insights": []},
            "marketing": {"insights": [], "optimizations": []},
            "timeline": []
        }

    width = int(scene.get("video_width") or detection_block.get("width") or 1)
    height = int(scene.get("video_height") or detection_block.get("height") or 1)
    fps = float(scene.get("video_fps") or detection_block.get("fps") or 30.0)
    frame_count = int(scene.get("total_frames") or detection_block.get("frame_count") or 0)
    duration = float(detection_block.get("duration_seconds") or (frame_count / fps if fps else 0))

    timeline = _timeline(detections, width, height)
    visible = [item for item in timeline if item["found"]]
    segments = _segments(detections)
    detected_seconds = len(visible)
    sample_count = len(timeline)
    avg_coverage = sum(item["screen_coverage"] for item in visible) / max(1, len(visible))
    max_coverage = max((item["screen_coverage"] for item in visible), default=0.0)
    avg_confidence = sum(item["confidence"] for item in visible) / max(1, len(visible))

    summary = {
        "detected_seconds": detected_seconds,
        "sampled_seconds": sample_count,
        "visibility_rate": round(detected_seconds / max(1, sample_count), 3),
        "first_seen_second": visible[0]["second"] if visible else None,
        "last_seen_second": visible[-1]["second"] if visible else None,
        "continuous_segments": segments,
        "average_screen_coverage": round(avg_coverage, 4),
        "max_screen_coverage": round(max_coverage, 4),
        "average_confidence": round(avg_confidence, 3),
        "total_airtime_ms": int(detected_seconds * 1000),
    }

    # ── Marketing Agents Integration (READ ONLY) ──────────────────────────────
    # Analysis is now triggered by the masking service only after rendering is done.
    marketing_results = scene.get("marketing") or scene.get("marketing_analysis") or {}

    video_name = os.path.basename(scene.get("filepath", ""))
    return {
        "show_id": show_id,
        "status": "ready" if marketing_results else "rendering",
        "product": scene.get("target_object") or detection_block.get("target") or "tracked product",
        "video": {
            "path": scene.get("filepath"),
            "url": f"/uploads/{video_name}" if video_name else None,

            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration_seconds": round(duration, 3),
        },
        "summary": summary,
        "scene_understanding": {
            "headline": _headline(visible, duration),
            "key_moments": _key_moments(timeline),
            "interaction_insights": marketing_results.get("interactions") or marketing_results.get("interaction_log", []),
        },
        "marketing": {
            "insights": marketing_results.get("insights") or marketing_results.get("market_insights", []),
            "optimizations": marketing_results.get("optimizations") or marketing_results.get("optimization_ideas", []),
        },
        "timeline": timeline,
    }



def _headline(visible: list[dict[str, Any]], duration: float) -> str:
    if not visible:
        return "The product is not visible in the sampled clip."
    first = visible[0]["second"]
    last = visible[-1]["second"]
    peak = max(visible, key=lambda item: item["screen_coverage"])
    return (
        f"The product is visible from about {first}s to {last}s in a {duration:.1f}s clip, "
        f"peaking at {peak['second']}s when it takes up {peak['screen_coverage'] * 100:.1f}% of the frame."
    )
