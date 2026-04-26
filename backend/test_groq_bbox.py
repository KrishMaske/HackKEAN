import os
import cv2
import base64
import json
import re
import numpy as np
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# -- Config: change these to test different videos/objects ---------------------
VIDEO_PATH    = "assets/uploads/STRANGER_THINGS_CLIP.mp4"
TARGET_OBJECT = "kfc bucket"
SEND_MAX_PX   = 512  # same downscale as ingestion.py
# -----------------------------------------------------------------------------


def _image_to_base64(img: np.ndarray, max_size: int = SEND_MAX_PX) -> str:
    """Downscale to max_size on longest edge, then base64-encode as JPEG."""
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode("utf-8")


def _parse_json_response(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        return json.loads(text[start:end + 1])


def test_groq_bounding_box():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("[ERROR] GROQ_API_KEY not found in .env")
        return

    client = Groq(api_key=api_key)

    # -- Extract frame 0 -------------------------------------------------------
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {VIDEO_PATH}")
        return

    orig_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"[ERROR] Failed to read frame 0 from {VIDEO_PATH}")
        return

    print(f"[INFO] Video: {orig_width}x{orig_height}")
    print(f"[INFO] Target object: '{TARGET_OBJECT}'")
    cv2.imwrite("groq_original_frame.jpg", frame)
    print("[INFO] Saved groq_original_frame.jpg")

    # -- Ask Groq for normalized bbox (same prompt as ingestion.py) ------------
    b64 = _image_to_base64(frame)

    # Save the downscaled image so we can see exactly what Groq sees
    h, w = frame.shape[:2]
    if max(h, w) > SEND_MAX_PX:
        scale = SEND_MAX_PX / max(h, w)
        frame_sent = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        frame_sent = frame.copy()
    cv2.imwrite("groq_sent_image.jpg", frame_sent)
    print(f"[INFO] Saved groq_sent_image.jpg  ({frame_sent.shape[1]}x{frame_sent.shape[0]}) -- this is exactly what Groq sees")

    prompt = f"""You are a precision object detector.

Find the object "{TARGET_OBJECT}" in this image.

CRITICAL RULES:
- Use NORMALIZED coordinates from 0 to 1000 (NOT pixels).
- (0,0) is top-left, (1000, 1000) is bottom-right.
- x_max MUST be greater than x_min. y_max MUST be greater than y_min.
- Set "found" to true if the object is visible, false if it is not.
- If not found, still return 0 for all coordinates.

Respond with ONLY valid JSON:
{{"found": bool, "x_min": int, "y_min": int, "x_max": int, "y_max": int}}
"""

    try:
        response = client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                    ]
                }
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
    except Exception as e:
        print(f"[ERROR] Groq API call failed: {e}")
        return

    raw = response.choices[0].message.content
    print(f"[GROQ] Raw response: {raw}")

    bbox_data = _parse_json_response(raw)
    x_min_n = bbox_data.get("x_min")
    y_min_n = bbox_data.get("y_min")
    x_max_n = bbox_data.get("x_max")
    y_max_n = bbox_data.get("y_max")
    found   = bbox_data.get("found", True)

    if not found:
        print(f"[WARN] Groq says the object was NOT found in frame 0.")
        return

    if any(v is None for v in [x_min_n, y_min_n, x_max_n, y_max_n]):
        print(f"[WARN] Groq returned null coordinates -- object not found.")
        return

    if x_min_n == 0 and y_min_n == 0 and x_max_n == 0 and y_max_n == 0:
        print(f"[WARN] Groq returned all-zeros -- treating as 'not found'. Check groq_sent_image.jpg to verify the object is visible.")
        return

    # -- Convert normalized -> pixel and clamp ---------------------------------
    x0 = max(0, min(int(x_min_n * orig_width  / 1000.0), orig_width  - 1))
    y0 = max(0, min(int(y_min_n * orig_height / 1000.0), orig_height - 1))
    x1 = max(0, min(int(x_max_n * orig_width  / 1000.0), orig_width  - 1))
    y1 = max(0, min(int(y_max_n * orig_height / 1000.0), orig_height - 1))

    print(f"[BBOX] Normalized:  x_min={x_min_n}, y_min={y_min_n}, x_max={x_max_n}, y_max={y_max_n}")
    print(f"[BBOX] Pixel coords: [{x0}, {y0}, {x1}, {y1}]  (orig {orig_width}x{orig_height})")

    if x1 <= x0 or y1 <= y0:
        print(f"[ERROR] Degenerate bbox after conversion -- check Groq output.")
        return

    # -- Draw on frame ---------------------------------------------------------
    vis = frame.copy()

    # Groq bbox -- RED
    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 2)
    label = f"Groq: {TARGET_OBJECT}"
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    ly = max(y0 - 8, lh + 4)
    cv2.rectangle(vis, (x0, ly - lh - 4), (x0 + lw + 4, ly + 2), (0, 0, 255), -1)
    cv2.putText(vis, label, (x0 + 2, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # Crosshair on bbox centre
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    cv2.drawMarker(vis, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

    cv2.imwrite("groq_bbox_test_result.jpg", vis)
    print("[SUCCESS] Saved groq_bbox_test_result.jpg  <- red box = Groq detection")

    # -- Also run GrabCut to show mask quality ---------------------------------
    gc_mask  = np.zeros(frame.shape[:2], np.uint8)
    bgd      = np.zeros((1, 65), np.float64)
    fgd      = np.zeros((1, 65), np.float64)
    rect     = (x0, y0, x1 - x0, y1 - y0)
    cv2.grabCut(frame, gc_mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    binary   = np.where((gc_mask == 1) | (gc_mask == 3), 255, 0).astype(np.uint8)
    cv2.imwrite("groq_grabcut_mask.jpg", binary)
    print("[SUCCESS] Saved groq_grabcut_mask.jpg  <- white = foreground pixels")


if __name__ == "__main__":
    test_groq_bounding_box()
