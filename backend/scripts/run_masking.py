#!/usr/bin/env python3
"""
Offline masking script for SceneShift.

Reads pre-computed scene metadata from Phase 1 (preprocess.py) and runs
the existing masking pipeline to generate mask and preview videos.

Usage:
    python -m scripts.run_masking --show_id stranger_things_83
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = "app/db/data"
MASKS_DIR = "assets/masks"
INPUT_DIR = "assets/input"


def main():
    parser = argparse.ArgumentParser(description="SceneShift Offline Masking Pipeline")
    parser.add_argument("--show_id", required=True, help="Show identifier (e.g. stranger_things_83)")
    parser.add_argument("--force", action="store_true", help="Re-generate even if mask video already exists")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  SceneShift Offline Masking Pipeline")
    print(f"  Show: {args.show_id}")
    print(f"{'='*60}\n")

    # Check for pre-computed scene metadata
    scene_path = os.path.join(DATA_DIR, f"{args.show_id}.json")
    if not os.path.exists(scene_path):
        print(f"❌ Scene metadata not found at {scene_path}")
        print(f"   Run preprocessing first: python -m scripts.preprocess --show_id {args.show_id} ...")
        sys.exit(1)

    with open(scene_path, "r") as f:
        scene_data = json.load(f)

    print(f"✅ Loaded scene metadata from {scene_path}")
    print(f"   Video: {scene_data.get('video_path', 'unknown')}")
    print(f"   Segments: {len(scene_data.get('segments', []))}")

    # Check for existing mask video
    mask_path = os.path.join(MASKS_DIR, f"{args.show_id}_mask.mp4")
    preview_path = os.path.join(MASKS_DIR, f"{args.show_id}_preview.mp4")

    if os.path.exists(mask_path) and not args.force:
        print(f"\n⚠️  Mask video already exists at {mask_path}")
        print(f"   Use --force to regenerate")
        return

    os.makedirs(MASKS_DIR, exist_ok=True)

    # Get the video path and bbox from scene data
    video_path = scene_data.get("video_path")
    if not video_path or not os.path.exists(video_path):
        print(f"❌ Video file not found at {video_path}")
        sys.exit(1)

    segments = scene_data.get("segments", [])
    if not segments:
        print("⚠️  No product segments found in scene data — nothing to mask")
        return

    # Use the first segment's bbox for masking
    best_segment = max(segments, key=lambda s: s.get("confidence", 0))
    bbox = best_segment.get("bbox")

    if not bbox:
        print("⚠️  No bounding box available — cannot generate mask")
        return

    print(f"\n[1/2] Generating temporal mask video...")
    print(f"  Bounding box: {bbox}")
    print(f"  Segment: {best_segment.get('start_time', 0):.1f}s - {best_segment.get('end_time', 0):.1f}s")
    print(f"  Product: {best_segment.get('product_name', 'unknown')}")

    try:
        from app.services.masking import generate_mask_video
        generate_mask_video(
            show_id=args.show_id,
            video_path=video_path,
            coordinates=bbox,
        )
        print(f"\n✅ Mask video saved to {mask_path}")
    except ImportError as e:
        print(f"\n⚠️  Could not import masking module: {e}")
        print("   This is expected if OpenCV or SAM3 dependencies aren't installed.")
        print("   The mask video should be generated manually or pre-existing.")
    except Exception as e:
        print(f"\n❌ Masking failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n   Next step: python -m scripts.run_agents\n")


if __name__ == "__main__":
    main()
