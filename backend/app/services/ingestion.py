import os
import cv2
import json
import time
import base64
import re
from typing import List, Tuple, Any
from app.core.config import settings
from app.services.product_detection import analyze_product_seconds
from app.services.masking import generate_mask_video

DATA_DIR = "app/db/data"
os.makedirs(DATA_DIR, exist_ok=True)

# ── Groq Models ───────────────────────────────────────────────────────────────
GROQ_VISION_MODEL = "llama-3.2-11b-vision-preview"
GROQ_TEXT_MODEL = "llama-3.3-70b-versatile"

def _parse_json_response(text: str) -> dict:
    """Robustly extract a JSON object from a response."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return {}

def _extract_keyframes(video_path: str, num_frames: int = 5) -> list:
    """Extract evenly-spaced frames from the video."""
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

def _image_to_base64(image_path: str, max_size: int = 512) -> str:
    """Reads an image and returns it as a base64 string."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buffer).decode("utf-8")

def _detect_object_in_first_frame(client, image_path: str, width: int, height: int) -> str:
    """Identify the most prominent commercial product in the scene."""
    b64 = _image_to_base64(image_path)
    prompt = "Look at this image. Identify the most prominent commercial product in the scene. Return ONLY the name of the product (e.g. 'orange car', 'coca cola can', 'nike shoes')."
    try:
        response = client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]
            }]
        )
        return response.choices[0].message.content.strip().lower()
    except Exception:
        return "product"

def analyze_data(path: str, show_id: str, target_object: str = None):
    """
    Ingest a video, detect the product, and extract scene metadata.
    """
    print(f"[INGEST] Starting analysis for show: {show_id}, path: {path}")
    client = settings.groq_client

    # ── 1. Extract frames ───────────────────────────────────────────────────
    frames, vid_width, vid_height, vid_fps, total_frames = _extract_keyframes(path, num_frames=5)

    # ── 2. Detect the product target second-by-second ───────────────────────
    if not target_object:
        print("[INGEST] No target provided; auto-selecting prominent product from first frame...")
        target_object = _detect_object_in_first_frame(client, frames[0][2], vid_width, vid_height)
    
    print(f"[INGEST] Detecting target once per second: {target_object}")
    product_detection = analyze_product_seconds(path, target=target_object, mode="groq")
    initial_bbox = product_detection["initial_bounding_box"]
    anchor_frame_index = product_detection["first_detected_frame"] or 0

    # ── 3. Extract metadata ──────────────────────────────────────────────────
    print("[INGEST] Extracting metadata (via Groq)...")
    base64_first = _image_to_base64(frames[0][2])
    
    # Stage 4a: Describe the frame visually
    description = ""
    try:
        desc_response = client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this video frame in detail. Focus on: 1. What is happening in the scene? 2. How are people interacting with objects? 3. What is the visual mood and lighting?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_first}"}}
                ]
            }]
        )
        description = desc_response.choices[0].message.content
    except Exception as e:
        print(f"[WARN] Frame description failed: {e}")

    # Stage 4b: Extract structured metadata
    meta_prompt = f"""Based on this visual description, determine production year and location. 
    Respond with ONLY JSON: {{"production_year": int, "location": "string"}}
    Description: {description}"""
    
    meta = {"production_year": 0, "location": "Unknown"}
    try:
        meta_response = client.chat.completions.create(
            model=GROQ_TEXT_MODEL,
            messages=[{"role": "user", "content": meta_prompt}],
            response_format={"type": "json_object"}
        )
        meta = _parse_json_response(meta_response.choices[0].message.content)
    except Exception as e:
        print(f"[WARN] Metadata extraction failed: {e}")

    # ── 4. Build and Save ───────────────────────────────────────────────────
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
        "anchor_frame_index": anchor_frame_index,
        "second_by_second_detection": product_detection,
        "scene_description": description,
    }
    save_scene(show_id, scene_data)
    _cleanup_temp_frames(frames)

    # ── 5. Trigger Masking ──────────────────────────────────────────────────
    try:
        print(f"[INGEST] Triggering mask generation for: {show_id}...")
        generate_mask_video(show_id)
    except Exception as e:
        print(f"[ERROR] Mask generation failed: {e}")

def save_scene(show_id: str, scene_data: dict):
    filepath = os.path.join(DATA_DIR, f"{show_id}.json")
    with open(filepath, "w") as f:
        json.dump(scene_data, f, indent=4)

def load_scene(show_id: str) -> dict:
    filepath = os.path.join(DATA_DIR, f"{show_id}.json")
    if not os.path.exists(filepath): return None
    with open(filepath, "r") as f:
        return json.load(f)

def _cleanup_temp_frames(frames: list):
    for _, _, path in frames:
        try: os.remove(path)
        except: pass
