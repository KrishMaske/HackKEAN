import os
import cv2
import base64
import json
import re
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
VIDEO_PATH = "assets/input/STRANGER_THINGS_CLIP.mp4"
TARGET_OBJECT = "kfc bucket"

def _parse_json_response(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        return json.loads(text[start:end+1])

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_groq_bounding_box():
    api_key = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=api_key)

    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"[ERROR] Failed to extract frame from {VIDEO_PATH}")
        return

    temp_path = "temp_test_frame.jpg"
    cv2.imwrite(temp_path, frame)
    print(f"Extracted frame. Resolution: {width}x{height}")

    base64_img = image_to_base64(temp_path)
    prompt = f"""You are a precision object detector.

Find the object "{TARGET_OBJECT}" in this image and return its TIGHT bounding box.

CRITICAL RULES:
- The bounding box must TIGHTLY enclose the object — not the general area.
- Use NORMALIZED coordinates from 0 to 1000.
- (0,0) is top-left, (1000, 1000) is bottom-right.
- If the object is NOT visible, return null values.

Respond with ONLY a JSON object:
{{"x_min": int, "y_min": int, "x_max": int, "y_max": int}}
"""
    
    response = client.chat.completions.create(
        model=GROQ_VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                ]
            }
        ],
        temperature=0.1,
        response_format={"type": "json_object"}
    )
    
    content = response.choices[0].message.content
    print(f"Groq Raw Output: {content}")
    
    bbox_data = _parse_json_response(content)
    if bbox_data.get("x_min") is not None:
        x1 = int(bbox_data["x_min"] * width / 1000.0)
        y1 = int(bbox_data["y_min"] * height / 1000.0)
        x2 = int(bbox_data["x_max"] * width / 1000.0)
        y2 = int(bbox_data["y_max"] * height / 1000.0)
        print(f"Scaled Pixel Coordinates: [{x1}, {y1}, {x2}, {y2}]")
        
        # Save original frame first
        cv2.imwrite("groq_original_frame.jpg", frame)
        
        # --- PADDING LOGIC ---
        pad_x = int((x2 - x1) * 0.25)
        pad_y = int((y2 - y1) * 0.25)
        
        px1 = max(0, x1 - pad_x)
        py1 = max(0, y1 - pad_y)
        px2 = min(width, x2 + pad_x)
        py2 = min(height, y2 + pad_y)
        
        print(f"Padded Coordinates for GrabCut: [{px1}, {py1}, {px2}, {py2}]")
        
        # Draw Original Groq Box (RED)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, "Groq (Too Tight)", (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw Padded Box (GREEN)
        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 3)
        cv2.putText(frame, "Expanded (For GrabCut)", (px1, max(py1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imwrite("groq_bbox_test_result.jpg", frame)
        print(f"[SUCCESS] Saved image with bounding boxes to: groq_bbox_test_result.jpg")
        
        # --- RUN GRABCUT TO PROVE IT WORKS ---
        import numpy as np
        mask = np.zeros(frame.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        
        rect = (px1, py1, px2 - px1, py2 - py1)
        cv2.grabCut(cv2.imread("groq_original_frame.jpg"), mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        
        binary_mask = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
        cv2.imwrite("groq_grabcut_mask.jpg", binary_mask)
        print(f"[SUCCESS] Saved final pixel-perfect mask to: groq_grabcut_mask.jpg")

if __name__ == "__main__":
    test_groq_bounding_box()
