import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.car_detection import analyze_product_seconds


def slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return cleaned or "product"


def write_csv(path: Path, detections: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "second",
        "timestamp_ms",
        "frame_index",
        "detector",
        "found",
        "status",
        "confidence",
        "bbox",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in detections:
            row = {field: item.get(field) for field in fields}
            row["bbox"] = json.dumps(row["bbox"])
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect any target product/object once per second in a video.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--target", required=True, help='Example: "coca cola can", "laptop", "whole visible orange car"')
    parser.add_argument("--show-id", default=None)
    parser.add_argument("--mode", choices=["groq", "hybrid", "color"], default="hybrid")
    parser.add_argument("--out", default=None)
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args()

    show_slug = args.show_id or slug(args.target)
    out_dir = Path(args.out or f"assets/detections/{show_slug}")
    result = analyze_product_seconds(
        video_path=args.video,
        target=args.target,
        mode=args.mode,
        sleep_seconds=args.sleep,
    )

    json_path = out_dir / "product_second_by_second.json"
    csv_path = out_dir / "product_second_by_second.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csv(csv_path, result["detections"])
    print(f"[DETECT] JSON: {json_path}")
    print(f"[DETECT] CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
