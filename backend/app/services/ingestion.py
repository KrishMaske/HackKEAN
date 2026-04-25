import os
import re
import time
import json
import cv2
import base64
from app.core.config import settings
from app.services.masking import generate_mask_video

DATA_DIR = "app/db/data"
os.makedirs(DATA_DIR, exist_ok=True)

GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_TEXT_MODEL = "llama-3.3-70b-versatile"


def _parse_json_response(text: str) -> dict:
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


def _image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def _extract_keyframes(video_path: str, num_frames: int = 5) -> list:
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames <= 0:
        cap.release()
        raise ValueError("Could not read frame count from video.")

    # If num_frames is 0, calculate 1 keyframe per second
    if num_frames == 0:
        num_frames = max(1, int(total_frames / fps))

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
    base64_image = _image_to_base64(first_frame_path)

    prompt = f"""You are an object detection system. This image is {width}x{height} pixels.

Identify the ONE most prominent, inanimate, trackable PRODUCT or HANDHELD item in this scene.
Pick something that is clearly visible and likely to be the subject of a product placement 
(e.g., a branded cup, food bucket, soda can, phone, etc). 

CRITICAL: DO NOT select large furniture (like tables, chairs, couches) or background elements.

Respond with ONLY the object name in lowercase, nothing else.
Example: "kfc bucket" or "coca cola can"
"""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=GROQ_VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.1,
            )
            return response.choices[0].message.content.strip().strip('"').strip("'")
        except Exception as e:
            if "429" in str(e):
                print(f"[WARN] Identification rate limited. Waiting 60s to reset...")
                time.sleep(60)
            else:
                print(f"[WARN] Identification failed: {e}")
                return "object"
    
    time.sleep(1)
    return "object"


def _detect_initial_bbox(client, target_object: str, frame_path: str, width: int, height: int) -> list:
    """
    Uses Groq to get the anchor bounding box on Frame 0.
    """
    base64_image = _image_to_base64(frame_path)

    prompt = f"""You are a precision object detector.

Find the object "{target_object}" in this image and return its TIGHT bounding box.

CRITICAL RULES:
- The bounding box must TIGHTLY enclose the object — not the general area.
- Use NORMALIZED coordinates from 0 to 1000.
- (0,0) is top-left, (1000, 1000) is bottom-right.
- If the object is NOT visible, return null values.

Respond with ONLY a JSON object:
{{"x_min": int, "y_min": int, "x_max": int, "y_max": int}}
"""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=GROQ_VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            if content:
                bbox_data = _parse_json_response(content)
                if bbox_data.get("x_min") is not None and bbox_data.get("x_max") is not None:
                    bbox = [
                        int(bbox_data["x_min"] * width / 1000.0),
                        int(bbox_data["y_min"] * height / 1000.0),
                        int(bbox_data["x_max"] * width / 1000.0),
                        int(bbox_data["y_max"] * height / 1000.0),
                    ]
                    print(f"[INGEST] Initial Anchor BBox from Groq: {bbox}")
                    return bbox
        except Exception as e:
            if "429" in str(e):
                print(f"[WARN] Initial bbox detection rate limited. Waiting 60s...")
                time.sleep(60)
            else:
                print(f"[WARN] Initial bbox detection failed: {e}")
                break
        
    return [None, None, None, None]


def analyze_data(path, show_id):
    client = settings.groq_client

    print(f"[INGEST] Extracting keyframes from: {path}")
    # Extract just the first frame since Optical Flow handles the rest
    frames, vid_width, vid_height, vid_fps, total_frames = _extract_keyframes(path, num_frames=1)
    vid_duration_s = total_frames / vid_fps
    print(f"[INGEST] Video: {vid_width}x{vid_height} @ {vid_fps:.1f}fps, "
          f"{total_frames} frames, {vid_duration_s:.1f}s, extracted {len(frames)} keyframes")

    if not frames:
        raise Exception("Failed to extract any frames from the video.")

    print("[INGEST] Identifying target object (via Groq)...")
    target_object = _detect_object_in_first_frame(
        client, frames[0][2], vid_width, vid_height
    )
    print(f"[INGEST] Target object: {target_object}")

    print(f"[INGEST] Detecting tight anchor bounding box on Frame 0 (via Groq)...")
    initial_bbox = _detect_initial_bbox(
        client, target_object, frames[0][2], vid_width, vid_height
    )

    print("[INGEST] Extracting metadata (via Groq)...")
    base64_first = _image_to_base64(frames[0][2])
    
    desc_response = client.chat.completions.create(
        model=GROQ_VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this video frame in detail, focusing on visual style, technology, and setting."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_first}"
                        }
                    }
                ]
            }
        ]
    )
    description = desc_response.choices[0].message.content

    meta_prompt = f"""Based on this visual description of a video frame, determine:
1. The approximate production year of this scene.
2. The visual setting/location (e.g. "Living Room", "Office", "Kitchen", etc.)

Visual Description: {description}

Respond with ONLY a JSON object:
{{"production_year": int, "location": "string"}}
"""
    meta = {"production_year": 0, "location": "Unknown"}
    try:
        meta_response = client.chat.completions.create(
            model=GROQ_TEXT_MODEL,
            messages=[{"role": "user", "content": meta_prompt}],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        meta = _parse_json_response(meta_response.choices[0].message.content)
    except Exception as e:
        print(f"[WARN] Metadata extraction failed: {e}")

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
        "initial_bounding_box": initial_bbox,
    }

    save_scene(show_id, scene_data)

    _cleanup_temp_frames(frames)

    try:
        print(f"[AUTO] Triggering mask generation for: {show_id}...")
        mask_path = generate_mask_video(show_id)
        print(f"[AUTO] Mask generation complete -> {mask_path}")
    except Exception as e:
        print(f"[ERROR] Auto-mask generation failed: {e}")


def save_scene(show_id: str, scene_data: dict):
    filepath = os.path.join(DATA_DIR, f"{show_id}.json")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(scene_data, f, indent=4)
    print(f"[INGEST] Saved scene data -> {filepath}")


def load_scene(show_id: str) -> dict:
    filepath = os.path.join(DATA_DIR, f"{show_id}.json")
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r") as f:
        return json.load(f)


def _cleanup_temp_frames(frames: list):
    for _, _, path in frames:
        try:
            os.remove(path)
        except OSError:
            pass
    temp_dir = os.path.join("assets", "temp_frames")
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass