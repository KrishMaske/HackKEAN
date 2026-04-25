import os
import re
import time
import json
import cv2
from app.core.config import settings
from app.services.masking import generate_mask_video
from google.genai import types

DATA_DIR = "app/db/data"
os.makedirs(DATA_DIR, exist_ok=True)

# The user explicitly requested to switch back to Gemini
# because Groq was detecting the wrong object.
MODEL_NAME = "gemini-2.5-flash"


def _parse_json_response(text: str) -> dict:
    """
    Robustly extract a JSON object from a response.
    Handles cases where the model wraps output in markdown code fences.
    """
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])
        raise


def _extract_keyframes(video_path: str, num_frames: int = 1) -> list:
    """
    Extract evenly-spaced frames from the video.
    We now only need 1 frame because SAM3 will track it for the rest of the video!
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames <= 0:
        cap.release()
        raise ValueError("Could not read frame count from video.")

    num_frames = min(num_frames, total_frames)
    step = max(1, total_frames // num_frames)

    frames_dir = os.path.join("assets", "temp_frames")
    os.makedirs(frames_dir, exist_ok=True)

    extracted = []
    for i in range(num_frames):
        frame_idx = i * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        ts_ms = int((frame_idx / fps) * 1000)
        path = os.path.join(frames_dir, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(path, frame)
        extracted.append((ts_ms, frame_idx, path))

    cap.release()
    return extracted, width, height, fps, total_frames


def _detect_object_in_first_frame(client, first_frame_path: str, width: int, height: int) -> str:
    """
    Send the first frame to Gemini and ask it to identify the most prominent
    trackable inanimate object. Returns the object name.
    """
    frame_file = client.files.upload(file=first_frame_path)

    prompt = f"""You are an object detection system. This image is {width}x{height} pixels.

Identify the ONE most prominent, inanimate, trackable object in this scene.
Pick something that is clearly visible and likely to remain in frame across multiple shots
(e.g. a cup, lamp, phone, bucket, poster, furniture piece, etc).

Respond with ONLY the object name in lowercase, nothing else.
Example: "red coffee mug"
"""
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[frame_file, prompt],
                config=types.GenerateContentConfig(temperature=0.1)
            )
            return response.text.strip().strip('"').strip("'")
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print(f"[WARN] Identification rate limited. Waiting 60s to reset...")
                time.sleep(60)
            else:
                print(f"[WARN] Identification failed: {e}")
                return "object"
    
    # 5-second delay to stay under the 5 RPM limit
    time.sleep(5)
    return "object"


def _detect_bbox_batch(client, frame_paths: list, target_object: str,
                        width: int, height: int) -> list:
    """
    Send frames to Gemini and get bounding boxes for the target object.
    Since we optimized, this will only run on 1 frame to serve as the SAM3 anchor.
    """
    results = []

    for ts_ms, frame_idx, frame_path in frame_paths:
        frame_file = client.files.upload(file=frame_path)

        prompt = f"""You are a precision object detector. This image is {width}x{height} pixels.

Find the object "{target_object}" in this image and return its TIGHT bounding box
in PIXEL coordinates for this {width}x{height} image.

Coordinate system:
- x increases left to right (0 = left edge, {width} = right edge)
- y increases top to bottom (0 = top edge, {height} = bottom edge)

CRITICAL RULES:
- The bounding box must TIGHTLY enclose the object — not the general area.
- Use ACTUAL pixel coordinates for {width}x{height}, NOT normalized 0-1000 values.
- If the object is NOT visible, return null values.

Respond with ONLY a JSON object:
{{"x_min": int, "y_min": int, "x_max": int, "y_max": int}}

If the object is not visible:
{{"x_min": null, "y_min": null, "x_max": null, "y_max": null}}
"""
        bbox = [None, None, None, None]
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=[frame_file, prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1
                    )
                )

                if response.text:
                    bbox_data = _parse_json_response(response.text)
                    bbox = [
                        bbox_data.get("x_min"),
                        bbox_data.get("y_min"),
                        bbox_data.get("x_max"),
                        bbox_data.get("y_max"),
                    ]
                    break
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    print(f"[WARN] Frame {frame_idx} rate limited. Waiting 60s...")
                    time.sleep(60)
                else:
                    print(f"[WARN] BBox detection failed for frame {frame_idx}: {e}")
                    break

        results.append({
            "frame_index": frame_idx,
            "timestamp_ms": ts_ms,
            "bounding_box": bbox
        })

        print(f"  [DETECT] frame={frame_idx} ts={ts_ms}ms bbox={bbox}")
        # 5-second delay to stay under the 5 RPM limit
        time.sleep(5)

    return results


def analyze_data(path, show_id):
    """
    Main ingestion pipeline using Google Gemini.
    """
    client = settings.google_client

    # ── 1. Extract keyframes from the video ──────────────────────────────────
    print(f"[INGEST] Extracting keyframes from: {path}")
    # We only extract 1 frame now because SAM3 tracks the rest of the video automatically!
    frames, vid_width, vid_height, vid_fps, total_frames = _extract_keyframes(path, num_frames=1)
    vid_duration_s = total_frames / vid_fps
    print(f"[INGEST] Video: {vid_width}x{vid_height} @ {vid_fps:.1f}fps, "
          f"{total_frames} frames, {vid_duration_s:.1f}s, extracted {len(frames)} keyframes")

    if not frames:
        raise Exception("Failed to extract any frames from the video.")

    # ── 2. Detect the target object from the first frame ─────────────────────
    print("[INGEST] Identifying target object (via Gemini)...")
    target_object = _detect_object_in_first_frame(
        client, frames[0][2], vid_width, vid_height
    )
    print(f"[INGEST] Target object: {target_object}")

    # ── 3. Get the initial bounding box for the anchor frame ─────────────────
    print(f"[INGEST] Detecting bounding box anchor...")
    frame_bboxes = _detect_bbox_batch(
        client, frames, target_object, vid_width, vid_height
    )

    # ── 4. Extract metadata ──────────────────────────────────────────────────
    print("[INGEST] Extracting metadata (via Gemini)...")
    first_frame_file = client.files.upload(file=frames[0][2])
    meta_prompt = """Look at this video frame. Determine:
1. The approximate production year of this scene (based on visual style, technology visible, etc.)
2. The visual setting/location (e.g. "Living Room", "Office", "Kitchen", etc.)

Respond with ONLY a JSON object:
{"production_year": int, "location": "string"}
"""
    meta = {"production_year": 0, "location": "Unknown"}
    for attempt in range(3):
        try:
            meta_response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[first_frame_file, meta_prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2
                )
            )
            meta = _parse_json_response(meta_response.text)
            break
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print(f"[WARN] Metadata rate limited. Waiting 60s...")
                time.sleep(60)
            else:
                print(f"[WARN] Metadata extraction failed: {e}")
                break

    # ── 5. Build the scene data ──────────────────────────────────────────────
    scene_data = {
        "show_id": show_id,
        "filepath": path,
        "production_year": meta.get("production_year", 0),
        "location": meta.get("location", "Unknown"),
        "target_object": target_object,
        "video_width": vid_width,
        "video_height": vid_height,
        "video_fps": vid_fps,
        "total_frames": total_frames,
        "frame_bounding_boxes": frame_bboxes,
    }

    if frame_bboxes:
        scene_data["initial_bounding_box"] = frame_bboxes[0]["bounding_box"]

    # ── 6. Save to per-show JSON ─────────────────────────────────────────────
    save_scene(show_id, scene_data)

    # ── 7. Clean up temp frames ──────────────────────────────────────────────
    _cleanup_temp_frames(frames)

    # ── 8. Auto-trigger mask generation ──────────────────────────────────────
    try:
        print(f"[AUTO] Triggering mask generation for: {show_id}...")
        mask_path = generate_mask_video(show_id)
        print(f"[AUTO] Mask generation complete -> {mask_path}")
    except Exception as e:
        print(f"[ERROR] Auto-mask generation failed: {e}")


def save_scene(show_id: str, scene_data: dict):
    """Save scene data to its own JSON file: app/db/data/{show_id}.json"""
    filepath = os.path.join(DATA_DIR, f"{show_id}.json")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(scene_data, f, indent=4)
    print(f"[INGEST] Saved scene data -> {filepath}")


def load_scene(show_id: str) -> dict:
    """Load scene data from app/db/data/{show_id}.json"""
    filepath = os.path.join(DATA_DIR, f"{show_id}.json")
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r") as f:
        return json.load(f)


def _cleanup_temp_frames(frames: list):
    """Remove temporary extracted frame images."""
    for _, _, path in frames:
        try:
            os.remove(path)
        except OSError:
            pass
    # Try to remove the temp dir if empty
    temp_dir = os.path.join("assets", "temp_frames")
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass