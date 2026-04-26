#!/usr/bin/env python3
"""
Offline preprocessing script for SceneShift.

Replaces the live ingestion.py flow. Extracts keyframes from a video,
sends them to Groq Vision for product detection and bounding box extraction,
and saves structured scene metadata to app/db/data/{show_id}.json.

Usage:
    python -m scripts.preprocess \
        --video_path assets/input/STRANGER_THINGS_CLIP.mp4 \
        --show_id stranger_things_83 \
        --target_object "kfc bucket"
"""

import argparse
import json
import os
import sys

# Add parent directory to path so we can import from app/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cv2
except ImportError:
    print("ERROR: opencv-python is required. Install with: pip install opencv-python")
    sys.exit(1)

import base64
import re
import time

DATA_DIR = "app/db/data"
os.makedirs(DATA_DIR, exist_ok=True)

GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_TEXT_MODEL = "llama-3.3-70b-versatile"


def parse_json_response(text: str) -> dict:
    """Robustly extract a JSON object from a model response."""
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
            return json.loads(text[start : end + 1])
        raise


def extract_keyframes(video_path: str, interval_seconds: float = 2.0) -> list:
    """Extract keyframes at regular intervals from the video."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames <= 0:
        cap.release()
        raise ValueError("Could not read frame count from video.")

    duration = total_frames / fps
    frame_interval = int(fps * interval_seconds)

    frames = []
    frame_idx = 0
    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(buffer).decode("utf-8")

        frames.append(
            {
                "frame_idx": frame_idx,
                "timestamp": round(timestamp, 3),
                "base64": b64,
            }
        )

        frame_idx += frame_interval

    cap.release()

    print(f"  Extracted {len(frames)} keyframes from {duration:.1f}s video ({width}x{height} @ {fps:.1f}fps)")
    return frames, {
        "total_frames": total_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "duration": round(duration, 3),
    }


def analyze_frames_with_groq(frames: list, target_object: str, groq_client, video_meta: dict) -> list:
    """Send frames to Groq Vision for product detection."""
    results = []
    batch_size = 5  # Process in small batches to stay within rate limits

    for batch_start in range(0, len(frames), batch_size):
        batch = frames[batch_start : batch_start + batch_size]

        for frame_data in batch:
            prompt = f"""Analyze this video frame. I'm looking for a "{target_object}" or similar product.

Return a JSON object with exactly these fields:
{{
  "product_detected": true/false,
  "product_name": "exact product name or null",
  "confidence": 0.0 to 1.0,
  "bbox": [x1, y1, x2, y2] as ABSOLUTE PIXEL coordinates (e.g. [120, 200, 450, 380]) or null if not detected,
  "description": "brief visual description of what you see"
}}

Be strict: only report the product if you actually see it in the frame."""

            try:
                response = groq_client.chat.completions.create(
                    model=GROQ_VISION_MODEL,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{frame_data['base64']}"
                                    },
                                },
                            ],
                        }
                    ],
                    max_tokens=300,
                    temperature=0.1,
                )

                result_text = response.choices[0].message.content
                parsed = parse_json_response(result_text)

                # Fix bbox if Groq hallucinated normalized floats
                if parsed.get("bbox"):
                    b = parsed["bbox"]
                    if len(b) == 4 and all(isinstance(v, (int, float)) for v in b):
                        # If values are < 2, they are likely normalized floats
                        if max(b) <= 2.0:
                            width = video_meta.get("width", 1280)
                            height = video_meta.get("height", 720)
                            parsed["bbox"] = [
                                int(b[0] * width),
                                int(b[1] * height),
                                int(b[2] * width),
                                int(b[3] * height)
                            ]
                
                parsed["frame_idx"] = frame_data["frame_idx"]
                parsed["timestamp"] = frame_data["timestamp"]
                results.append(parsed)

                status = "✅" if parsed.get("product_detected") else "  "
                print(
                    f"  {status} Frame {frame_data['frame_idx']:5d} @ {frame_data['timestamp']:6.2f}s — "
                    f"conf={parsed.get('confidence', 0):.2f} — {parsed.get('product_name', 'nothing')}"
                )

            except Exception as e:
                print(f"  ⚠️  Frame {frame_data['frame_idx']} failed: {e}")
                results.append(
                    {
                        "frame_idx": frame_data["frame_idx"],
                        "timestamp": frame_data["timestamp"],
                        "product_detected": False,
                        "error": str(e),
                    }
                )

            # Rate limiting: small delay between calls
            time.sleep(0.5)

    return results


def build_segments(frame_results: list, video_meta: dict) -> list:
    """Build continuous segments from per-frame detection results."""
    segments = []
    current_segment = None

    for result in frame_results:
        if result.get("product_detected"):
            if current_segment is None:
                current_segment = {
                    "start_frame": result["frame_idx"],
                    "start_time": result["timestamp"],
                    "end_frame": result["frame_idx"],
                    "end_time": result["timestamp"],
                    "product_name": result.get("product_name", "Unknown"),
                    "confidence": result.get("confidence", 0),
                    "bbox": result.get("bbox"),
                }
            else:
                current_segment["end_frame"] = result["frame_idx"]
                current_segment["end_time"] = result["timestamp"]
                current_segment["confidence"] = max(
                    current_segment["confidence"], result.get("confidence", 0)
                )
                # Keep the best bbox (highest confidence)
                if result.get("bbox") and result.get("confidence", 0) >= current_segment["confidence"]:
                    current_segment["bbox"] = result["bbox"]
        else:
            if current_segment is not None:
                segments.append(current_segment)
                current_segment = None

    if current_segment is not None:
        segments.append(current_segment)

    return segments


def main():
    parser = argparse.ArgumentParser(description="SceneShift Offline Preprocessor")
    parser.add_argument("--video_path", required=True, help="Path to input video")
    parser.add_argument("--show_id", required=True, help="Show identifier (e.g. stranger_things_83)")
    parser.add_argument("--target_object", required=True, help="Product to detect (e.g. 'kfc bucket')")
    parser.add_argument("--interval", type=float, default=2.0, help="Keyframe interval in seconds")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  SceneShift Offline Preprocessor")
    print(f"  Video:  {args.video_path}")
    print(f"  Show:   {args.show_id}")
    print(f"  Target: {args.target_object}")
    print(f"{'='*60}\n")

    # Initialize Groq client
    try:
        from app.core.config import settings
        groq_client = settings.groq_client
        print("✅ Groq client initialized\n")
    except Exception as e:
        print(f"❌ Could not initialize Groq client: {e}")
        print("   Make sure GROQ_API_KEY is set in backend/.env")
        sys.exit(1)

    # Step 1: Extract keyframes
    print("[1/3] Extracting keyframes...")
    frames, video_meta = extract_keyframes(args.video_path, args.interval)

    # Step 2: Analyze with Groq Vision
    print(f"\n[2/3] Analyzing {len(frames)} frames with Groq Vision...")
    frame_results = analyze_frames_with_groq(frames, args.target_object, groq_client, video_meta)

    # Step 3: Build segments and save
    print("\n[3/3] Building segments...")
    segments = build_segments(frame_results, video_meta)
    print(f"  Found {len(segments)} product segment(s)")

    # Build output
    output = {
        "show_id": args.show_id,
        "target_object": args.target_object,
        "video_filepath": args.video_path,
        "video_meta": video_meta,
        "segments": segments,
        "frame_results": [
            {k: v for k, v in r.items() if k != "base64"}
            for r in frame_results
        ],
    }

    output_path = os.path.join(DATA_DIR, f"{args.show_id}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Saved scene metadata to {output_path}")
    print(f"   {len(segments)} segment(s), {len(frame_results)} frame(s) analyzed")
    print(f"\n   Next step: python -m scripts.run_masking --show_id {args.show_id}\n")


if __name__ == "__main__":
    main()
