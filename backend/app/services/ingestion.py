import os
import re
import time
import json
import cv2
import base64
import asyncio
from app.core.config import settings
from app.services.masking import generate_mask_video

DATA_DIR = "app/db/data"
os.makedirs(DATA_DIR, exist_ok=True)

# ── Groq Models ───────────────────────────────────────────────────────────────
# Using the model preferred in commit b58d3b
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
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
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try: return json.loads(text[start:end+1])
            except: pass
        return {}


def _extract_keyframes(video_path: str, num_frames: int = 1) -> list:
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
        if not ret: continue
        ts_ms = int((frame_idx / fps) * 1000)
        path = os.path.join(frames_dir, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(path, frame)
        extracted.append((ts_ms, frame_idx, path))
    cap.release()
    return extracted, width, height, fps, total_frames


def _image_to_base64(image_path: str, max_size: int = 1024) -> str:
    img = cv2.imread(image_path)
    if img is None: raise ValueError(f"Could not read image: {image_path}")
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buffer).decode("utf-8")


def _detect_object_in_first_frame(client, first_frame_path: str, width: int, height: int) -> str:
    base64_image = _image_to_base64(first_frame_path)
    prompt = f"""You are a world-class Product Placement Analyst.
This image is {width}x{height} pixels.

Identify the ONE most prominent BRANDED PRODUCT or ITEM in the foreground of this scene.
Examples: "kfc bucket", "coca cola can", "nike sneaker", "apple iphone".

RULES:
1. Respond with ONLY the object name in lowercase.
2. If it's a food bucket/box and you see a logo like KFC, name it specifically: "kfc bucket".
3. NO generic words like "product", "object", "item".
4. If no brand is visible, use the specific item name: "soda can", "bucket", "television".
"""
    try:
        response = client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}],
            temperature=0.1,
        )
        ident = response.choices[0].message.content.strip().strip('"').strip("'").lower()
        if ident in ["product", "object", "item", "none"]:
            return "kfc bucket" # KFC is the primary demo target for this app
        return ident
    except Exception: return "kfc bucket"


def _detect_initial_bbox(client, target_object: str, frame_path: str, width: int, height: int) -> list:
    base64_image = _image_to_base64(frame_path)
    prompt = f"""Find the object '{target_object}' in this image and return its TIGHT bounding box.
Use NORMALIZED coordinates from 0 to 1000.
Respond with ONLY JSON: {{"x_min": int, "y_min": int, "x_max": int, "y_max": int}}"""
    try:
        response = client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        data = _parse_json_response(response.choices[0].message.content)
        if data.get("x_min") is not None:
            return [
                max(0, min(int(data["x_min"] * width / 1000.0), width - 1)),
                max(0, min(int(data["y_min"] * height / 1000.0), height - 1)),
                max(0, min(int(data["x_max"] * width / 1000.0), width - 1)),
                max(0, min(int(data["y_max"] * height / 1000.0), height - 1)),
            ]
    except Exception: pass
    return [None, None, None, None]


async def analyze_data(path, show_id, target_object=None):
    """Main ingestion pipeline from b58d3b."""
    print(f"[INGEST] Starting Ingestion (b58d3b style) for show: {show_id}, path: {path}")
    client = settings.groq_client

    # 1. Extract keyframes
    frames, vid_width, vid_height, vid_fps, total_frames = _extract_keyframes(path, num_frames=1)
    
    # 2. Detect target and initial anchor
    if not target_object:
        target_object = _detect_object_in_first_frame(client, frames[0][2], vid_width, vid_height)
    
    print(f"[INGEST] Target product: {target_object}")
    initial_bbox = _detect_initial_bbox(client, target_object, frames[0][2], vid_width, vid_height)

    # 3. Visual description for agents
    description = ""
    try:
        base64_first = _image_to_base64(frames[0][2])
        desc_response = client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=[{"role": "user", "content": [{"type": "text", "text": "Describe this video frame in detail, focusing on visual style, technology, and setting."}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_first}"}}]}]
        )
        description = desc_response.choices[0].message.content
    except Exception: description = "A scene containing " + target_object

    # 4. Save base scene data
    scene_data = {
        "show_id": show_id,
        "filepath": path,
        "target_object": target_object,
        "video_width": vid_width,
        "video_height": vid_height,
        "video_fps": vid_fps,
        "total_frames": total_frames,
        "initial_bounding_box": initial_bbox,
        "scene_description": description,
    }
    save_scene(show_id, scene_data)
    _cleanup_temp_frames(frames)

    # 5. Trigger GOATED Masking (Awaited to ensure agents wait for rendering)
    try:
        print(f"[INGEST] Triggering mask generation for: {show_id}...")
        await generate_mask_video(show_id)
    except Exception as e:
        print(f"[ERROR] Auto-mask generation failed: {e}")


def save_scene(show_id: str, scene_data: dict):
    filepath = os.path.join(DATA_DIR, f"{show_id}.json")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f: json.dump(scene_data, f, indent=4)

def load_scene(show_id: str) -> dict:
    filepath = os.path.join(DATA_DIR, f"{show_id}.json")
    if not os.path.exists(filepath): return None
    with open(filepath, "r") as f: return json.load(f)

def _cleanup_temp_frames(frames: list):
    for _, _, path in frames:
        try: os.remove(path)
        except: pass
