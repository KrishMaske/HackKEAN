import json
import os
import time
import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Optional
from app.core.config import settings

DATA_DIR = "app/db/data"
MASKS_DIR = "assets/masks"
PROCESS_WIDTH = 720
MIN_MASK_PIXELS = 100
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

def load_scene_metadata(show_id: str) -> Dict:
    filepath = os.path.join(DATA_DIR, f"{show_id}.json")
    if not os.path.exists(filepath): raise FileNotFoundError(f"Scene metadata not found at {filepath}")
    with open(filepath, "r", encoding="utf-8") as handle: return json.load(handle)

def save_alpha_mask(mask: np.ndarray, path: str):
    Image.fromarray(mask).save(path)

def bbox_to_mask(bbox, frame_height: int, frame_width: int) -> np.ndarray:
    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    if bbox and len(bbox) == 4 and all(v is not None for v in bbox):
        x0, y0, x1, y1 = [int(v) for v in bbox]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(frame_width, x1), min(frame_height, y1)
        if x1 > x0 and y1 > y0: mask[y0:y1, x0:x1] = 255
    return mask

def render_mask_video(alpha_dir: str, output_path: str, fps: float, width: int, height: int):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
    files = sorted([p for p in os.listdir(alpha_dir) if p.endswith(".png")])
    for f in files:
        img = cv2.imread(os.path.join(alpha_dir, f), cv2.IMREAD_GRAYSCALE)
        if img.shape[:2] != (height, width): img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
        writer.write(img)
    writer.release()

def render_overlay_video(frames: List[np.ndarray], masks: List[np.ndarray], output_path: str, fps: float, width: int, height: int):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)
    for frame, mask in zip(frames, masks):
        if frame.shape[1] != width or frame.shape[0] != height: frame = cv2.resize(frame, (width, height))
        if mask.shape[1] != width or mask.shape[0] != height: mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        overlay = frame.copy()
        overlay[mask == 255] = [0, 255, 0]
        blended = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
        writer.write(blended)
    writer.release()

def _detect_bbox_with_groq(frame: np.ndarray, product_name: str, width: int, height: int) -> Optional[list]:
    import base64, json, re
    # Downscale for API
    sc = 512 / max(frame.shape[:2])
    small = cv2.resize(frame, (int(frame.shape[1] * sc), int(frame.shape[0] * sc)))
    _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
    b64 = base64.b64encode(buf).decode("utf-8")
    
    prompt = f'Find the "{product_name}" and return tight normalized coordinates (0-1000). Respond with ONLY valid JSON: {{"found": bool, "x_min": int, "y_min": int, "x_max": int, "y_max": int}}'
    try:
        response = settings.groq_client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        data = json.loads(response.choices[0].message.content)
        if not data.get("found"): return None
        return [max(0, min(int(data["x_min"] * width / 1000.0), width-1)), max(0, min(int(data["y_min"] * height / 1000.0), height-1)), max(0, min(int(data["x_max"] * width / 1000.0), width-1)), max(0, min(int(data["y_max"] * height / 1000.0), height-1))]
    except: return None

async def generate_temporal_alpha_masks(show_id: str) -> Dict[str, object]:
    scene = load_scene_metadata(show_id)
    video_path = scene["filepath"]
    
    cap = cv2.VideoCapture(video_path)
    fps, orig_w, orig_h = cap.get(cv2.CAP_PROP_FPS) or 30.0, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    proc_w = 720
    proc_h = int(orig_h * (proc_w / orig_w))
    
    frames = []
    while True:
        ok, f = cap.read()
        if not ok: break
        frames.append(cv2.resize(f, (proc_w, proc_h)))
    cap.release()
    
    # Keyframe detection (1 per second)
    interval = int(fps)
    kf_indices = list(range(0, len(frames), interval))
    if (len(frames)-1) not in kf_indices: kf_indices.append(len(frames)-1)
    
    target = scene.get("target_object", "product")
    initial_bbox = scene.get("initial_bounding_box")
    # Remap initial bbox to proc resolution
    p_sx, p_sy = proc_w / orig_w, proc_h / orig_h
    proc_initial = [int(initial_bbox[0]*p_sx), int(initial_bbox[1]*p_sy), int(initial_bbox[2]*p_sx), int(initial_bbox[3]*p_sy)] if initial_bbox else [0,0,0,0]

    keyframe_bboxes = {}
    print(f"[MASK] Analyzing {len(kf_indices)} keyframes for {target}...")
    for idx in kf_indices:
        bbox = _detect_bbox_with_groq(frames[idx], target, proc_w, proc_h)
        keyframe_bboxes[idx] = bbox or proc_initial
        time.sleep(0.5)

    # Interpolation
    masks = []
    kf_sorted = sorted(keyframe_bboxes.keys())
    for i in range(len(frames)):
        # Find segment
        prev_kf = kf_sorted[0]; next_kf = kf_sorted[-1]
        for k in kf_sorted:
            if k >= i: next_kf = k; break
            prev_kf = k
        
        if prev_kf == next_kf: bbox = keyframe_bboxes[prev_kf]
        else:
            t = (i - prev_kf) / (next_kf - prev_kf)
            b0, b1 = keyframe_bboxes[prev_kf], keyframe_bboxes[next_kf]
            bbox = [int(b0[j] + (b1[j]-b0[j])*t) for j in range(4)]
        
        masks.append(bbox_to_mask(bbox, proc_h, proc_w))

    # Save to disk
    alpha_dir = os.path.join(MASKS_DIR, show_id, "alpha")
    os.makedirs(alpha_dir, exist_ok=True)
    output_masks = []
    for idx, m in enumerate(masks):
        m_full = cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        save_alpha_mask(m_full, os.path.join(alpha_dir, f"frame_{idx:04d}.png"))
        output_masks.append(m_full)

    mask_v, prev_v = os.path.join(MASKS_DIR, f"{show_id}_mask.mp4"), os.path.join(MASKS_DIR, f"{show_id}_preview.mp4")
    render_mask_video(alpha_dir, mask_v, fps, orig_w, orig_h)
    # Re-read orig frames for preview
    cap = cv2.VideoCapture(video_path); orig_frames = []
    while True:
        ok, f = cap.read()
        if not ok: break
        orig_frames.append(f)
    cap.release()
    render_overlay_video(orig_frames, output_masks, prev_v, fps, orig_w, orig_h)

    # Hifi ROI Analysis
    final_detections = []
    for s in range(int(len(frames)/fps) + 1):
        idx = min(int(round(s * fps)), len(masks) - 1)
        m = masks[idx]
        active = int(m.sum() / 255)
        found = active >= MIN_MASK_PIXELS
        bbox = None
        if found:
            coords = np.argwhere(m > 0); y0, x0 = coords.min(axis=0); y1, x1 = coords.max(axis=0)
            bbox = [int(x0 * orig_w/proc_w), int(y0 * orig_h/proc_h), int(x1 * orig_w/proc_w), int(y1 * orig_h/proc_h)]
        final_detections.append({"second": s, "timestamp_ms": int(s * 1000), "frame_index": idx, "found": found, "status": "ok" if found else "missing", "confidence": 1.0 if found else 0.0, "bbox": bbox, "screen_coverage": round(active/(proc_w*proc_h), 4)})

    scene["second_by_second_detection"] = {"target": target, "sampling": "Keyframe Interpolation (Llama-4)", "width": orig_w, "height": orig_h, "fps": fps, "duration_seconds": len(frames)/fps, "detections": final_detections}
    from app.services.ingestion import save_scene
    save_scene(show_id, scene)

    # FINAL AGENT TRIGGER
    try:
        from app.services.orchestrator import run_marketing_workflow
        print(f"[MASK] Rendering complete. Triggering Marketing Agents...")
        await run_marketing_workflow(show_id)
    except Exception as e: print(f"[ERROR] Marketing trigger failed: {e}")

    return {"show_id": show_id, "mask_video": mask_v}

async def generate_mask_video(show_id: str) -> str:
    result = await generate_temporal_alpha_masks(show_id)
    return os.path.abspath(result["mask_video"])


def mask_status(show_id: str) -> Dict[str, object]:
    """Check if the neural mask rendering is complete."""
    alpha_dir = os.path.join(MASKS_DIR, show_id, "alpha")
    if not os.path.exists(alpha_dir):
        return {"ready": False, "frame_count": 0, "mask_directory": alpha_dir}

    files = sorted([p for p in os.listdir(alpha_dir) if p.endswith(".png")])
    return {
        "ready": True,
        "frame_count": len(files),
        "mask_directory": alpha_dir,
        "preview_url": f"/masks/{show_id}_preview.mp4" if files else None,
    }
