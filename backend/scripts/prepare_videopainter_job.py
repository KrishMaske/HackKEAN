import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.masking import generate_temporal_alpha_masks, load_scene_metadata


MASK_ID = 1
DEFAULT_JOB_ID = "sceneshift_orange_car000"


def read_video_info(video_path: str) -> tuple[int, int, float, int]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return width, height, fps, frame_count


def build_all_masks(alpha_dir: Path, width: int, height: int, frame_count: int) -> np.ndarray:
    masks = np.zeros((frame_count, height, width), dtype=np.uint8)
    for idx in range(frame_count):
        mask_path = alpha_dir / f"frame_{idx:04d}.png"
        if not mask_path.exists():
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        masks[idx] = np.where(mask > 127, MASK_ID, 0).astype(np.uint8)
    return masks


def prepare_job(show_id: str, videopainter_root: str, caption: str, job_id: str) -> dict:
    if len(job_id) <= 3:
        raise ValueError("job_id must be longer than 3 characters because VideoPainter derives the parent folder from it.")

    scene = load_scene_metadata(show_id)
    video_path = scene.get("filepath")
    if not video_path or not os.path.exists(video_path):
        raise FileNotFoundError(f"Video path does not exist: {video_path}")

    generate_temporal_alpha_masks(show_id)
    alpha_dir = Path("assets/masks") / show_id / "alpha"
    if not alpha_dir.exists():
        raise FileNotFoundError(f"Mask alpha directory does not exist: {alpha_dir}")

    width, height, fps, frame_count = read_video_info(video_path)
    vp_root = Path(videopainter_root).resolve()
    data_root = vp_root / "data"
    video_parent_id = job_id[:-3]
    video_filename = f"{job_id}.0.mp4"
    raw_video_dir = data_root / "videovo" / "raw_video" / video_parent_id
    mask_dir = data_root / "video_inpainting" / "videovo" / job_id
    csv_path = data_root / "sceneshift_videopainter.csv"

    raw_video_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(video_path, raw_video_dir / video_filename)

    all_masks = build_all_masks(alpha_dir, width, height, frame_count)
    np.savez_compressed(mask_dir / "all_masks.npz", all_masks)

    data_root.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "fps", "mask_id", "start_frame", "end_frame", "caption"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "path": f"{job_id}.0.mp4",
                "fps": int(round(fps)),
                "mask_id": MASK_ID,
                "start_frame": 0,
                "end_frame": frame_count,
                "caption": caption,
            }
        )

    return {
        "video_path": str(raw_video_dir / video_filename),
        "mask_path": str(mask_dir / "all_masks.npz"),
        "csv_path": str(csv_path),
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export SceneShift video/masks to VideoPainter dataset format.")
    parser.add_argument("--show-id", default="orange_car")
    parser.add_argument("--videopainter-root", required=True)
    parser.add_argument("--job-id", default=DEFAULT_JOB_ID)
    parser.add_argument(
        "--caption",
        default="A red Ferrari sports car is parked on a street while people walk around it.",
    )
    args = parser.parse_args()

    result = prepare_job(
        show_id=args.show_id,
        videopainter_root=args.videopainter_root,
        caption=args.caption,
        job_id=args.job_id,
    )
    print("[VIDEOPAINTER] Prepared job:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
