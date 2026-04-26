import argparse
import os
import shutil
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.masking import load_scene_metadata


def original_fps(show_id: str) -> float:
    scene = load_scene_metadata(show_id)
    video_path = scene.get("filepath")
    if not video_path or not os.path.exists(video_path):
        return 30.0

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return 30.0
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    capture.release()
    return fps


def import_videopainter_output(show_id: str, source_video: str, crop_debug_grid: bool = True) -> str:
    source_path = Path(source_video)
    if not source_path.exists():
        raise FileNotFoundError(f"VideoPainter output does not exist: {source_path}")

    output_dir = Path("assets/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{show_id}_videopainter.mp4"

    if not crop_debug_grid:
        shutil.copy2(source_path, output_path)
        return str(output_path)

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open VideoPainter output: {source_path}")

    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    generated_width = frame_width // 4
    if generated_width <= 0:
        capture.release()
        raise RuntimeError(f"Unexpected VideoPainter frame width: {frame_width}")

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        original_fps(show_id),
        (generated_width, frame_height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not write output video: {output_path}")

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        generated = frame[:, frame_width - generated_width : frame_width]
        writer.write(generated)

    capture.release()
    writer.release()
    return str(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Import a VideoPainter result into SceneShift output assets.")
    parser.add_argument("--show-id", default="orange_car")
    parser.add_argument("--source-video", required=True)
    parser.add_argument(
        "--no-crop-debug-grid",
        action="store_true",
        help="Use this only if your VideoPainter output is already the clean generated video.",
    )
    args = parser.parse_args()

    output_path = import_videopainter_output(
        show_id=args.show_id,
        source_video=args.source_video,
        crop_debug_grid=not args.no_crop_debug_grid,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
