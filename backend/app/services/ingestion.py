import os
import re
import time
import json
from app.core.config import settings
from google.genai import types
from google.genai import errors as genai_errors
from app.services.masking import generate_mask_video


def _parse_json_response(text: str) -> dict:
    """
    Robustly extract a JSON object from a Gemini response.
    Handles cases where the model wraps output in markdown code fences.
    """
    # Strip leading/trailing whitespace
    text = text.strip()

    # Remove markdown code fences if present: ```json ... ``` or ``` ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    return json.loads(text)

def analyze_data(path, show_id):
    client = settings.google_client
    file = client.files.upload(file=path)
    
    while file.state == types.FileState.PROCESSING:
        print(".", end="", flush=True)
        time.sleep(5)
        file = client.files.get(name=file.name)
        
    if file.state == types.FileState.FAILED:
        error_message = file.error.message if file.error else "Unknown file processing error"
        raise Exception(f"File processing failed: {error_message}")
    
    # First, get the video dimensions so we can tell Gemini the resolution
    import cv2 as _cv2
    _cap = _cv2.VideoCapture(path)
    vid_width  = int(_cap.get(_cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(_cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))
    vid_fps    = _cap.get(_cv2.CAP_PROP_FPS) or 30.0
    vid_frames = int(_cap.get(_cv2.CAP_PROP_FRAME_COUNT))
    vid_duration_s = vid_frames / vid_fps
    _cap.release()

    num_samples = max(5, int(vid_duration_s))  # ~1 keyframe per second

    prompt = f"""
    You are an automated spatial mapping engine for a video editing pipeline.
    The video resolution is {vid_width}x{vid_height} pixels ({vid_width} wide, {vid_height} tall).
    The video is approximately {vid_duration_s:.1f} seconds long.

    Your task:
    1. Determine the production year and the visual setting (location) of this scene.
    2. Identify ONE prominent, inanimate object (e.g. a cup, phone, poster, bucket, etc.)
       that can be tracked visually throughout the video.
    3. For approximately {num_samples} evenly-spaced moments in the video (one per second),
       estimate the bounding box of that object in ACTUAL PIXEL COORDINATES.

    IMPORTANT — Coordinate rules:
    - x_min and x_max are horizontal pixel positions (0 = left edge, {vid_width} = right edge).
    - y_min and y_max are vertical pixel positions (0 = top edge, {vid_height} = bottom edge).
    - Do NOT use normalised [0-1000] coordinates.  Use real pixel values for this {vid_width}x{vid_height} frame.
    - The bounding box should tightly enclose the object — not be too large or too small.

    Output ONLY a JSON object matching this exact schema:
    {{
      "production_year": int,
      "location": str,
      "target_object": str,
      "frame_bounding_boxes": [
        {{
          "frame_index": int,
          "timestamp_ms": int,
          "bounding_box": [x_min, y_min, x_max, y_max]
        }}
      ]
    }}

    Rules:
    - Include {num_samples} entries, evenly spaced from timestamp_ms 0 to the end.
    - frame_index should be a sequential counter starting at 0.
    - timestamp_ms should be the actual millisecond position in the video.
    - If the object is NOT visible at a timestamp, set bounding_box to [null, null, null, null].
    - All visible coordinates must be integers in pixel space.
    - Output ONLY valid JSON, no markdown fences or explanation.
    """
    MAX_RETRIES = 3
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[file, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2
            )
        )

        if not response.text:
            last_error = Exception("Model returned an empty response.")
            print(f"[WARN] Attempt {attempt}/{MAX_RETRIES}: empty response, retrying...")
            time.sleep(2)
            continue

        try:
            scene_data = _parse_json_response(response.text)
            break  # Success — exit the retry loop
        except json.JSONDecodeError as e:
            last_error = e
            print(f"[WARN] Attempt {attempt}/{MAX_RETRIES}: JSON parse failed ({e}), retrying...")
            time.sleep(2)
    else:
        raise Exception(
            f"Gemini returned unparseable JSON after {MAX_RETRIES} attempts. Last error: {last_error}"
        )

    scene_data["show_id"] = show_id
    scene_data["filepath"] = path
    if "frame_bounding_boxes" in scene_data and scene_data["frame_bounding_boxes"]:
        scene_data["initial_bounding_box"] = scene_data["frame_bounding_boxes"][0]["bounding_box"]
    
    save_to_vault(scene_data)
    print(f"Successfully ingested {file.name} into db/scene_vault.json!")

    # Auto-trigger the mask/bounding-box generation
    try:
        print(f"[AUTO] Triggering mask generation for: {show_id}...")
        mask_path = generate_mask_video(show_id)
        print(f"[AUTO] Mask generation complete -> {mask_path}")
    except Exception as e:
        print(f"[ERROR] Auto-mask generation failed: {e}")

def save_to_vault(scene_data):
    vault_path = "app/db/data/scene_vault.json"
    
    if not os.path.exists(vault_path):
        os.makedirs(os.path.dirname(vault_path), exist_ok=True)
        with open(vault_path, "w") as f:
            json.dump({"scenes": []}, f)
    
    try:
        with open(vault_path, "r") as f:
            content = f.read().strip()
            vault = json.loads(content) if content else {"scenes": []}
    except json.JSONDecodeError:
        vault = {"scenes": []}
        
    if "scenes" not in vault or not isinstance(vault["scenes"], list):
        vault["scenes"] = []

    vault["scenes"].append(scene_data)
    
    with open(vault_path, "w") as f:
        json.dump(vault, f, indent=4)