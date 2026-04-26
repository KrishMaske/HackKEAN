import argparse
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_videopainter_job import DEFAULT_JOB_ID, prepare_job


def add_path(zip_file: zipfile.ZipFile, source: Path, arcname: Path) -> None:
    if source.is_dir():
        for child in source.rglob("*"):
            if child.is_file():
                zip_file.write(child, arcname / child.relative_to(source))
    elif source.is_file():
        zip_file.write(source, arcname)


def package_job(show_id: str, caption: str, job_id: str, output_zip: str) -> str:
    package_root = Path("assets/videopainter_package")
    if package_root.exists():
        shutil.rmtree(package_root)

    vp_root = package_root / "VideoPainter"
    prepare_job(show_id=show_id, videopainter_root=str(vp_root), caption=caption, job_id=job_id)

    scripts_dir = package_root / "sceneshift_scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2("scripts/import_videopainter_output.py", scripts_dir / "import_videopainter_output.py")
    shutil.copy2("scripts/run_videopainter_job.ps1", scripts_dir / "run_videopainter_job.ps1")

    readme = package_root / "README_GPU_RUN.txt"
    readme.write_text(
        "\n".join(
            [
                "SceneShift VideoPainter job package",
                "",
                "This package contains only the prepared video, mask npz, CSV, and helper scripts.",
                "It does not contain the VideoPainter repo or model checkpoints.",
                "",
                "On a CUDA GPU machine:",
                "1. git clone https://github.com/TencentARC/VideoPainter.git",
                "2. Follow VideoPainter's README to create Python 3.10 env, install requirements, install ./diffusers, and download ckpt.",
                "3. Copy this package's VideoPainter/data folder into the cloned VideoPainter/data folder.",
                "4. From the cloned VideoPainter/infer folder, run inpaint.py using data/sceneshift_videopainter.csv.",
                "",
                "Expected generated debug video:",
                f"VideoPainter/infer/sceneshift_output/{job_id}_fps8.mp4",
                "",
                "After rendering, bring that mp4 back to backend and run:",
                f"python scripts/import_videopainter_output.py --show-id {show_id} --source-video <rendered_mp4>",
            ]
        ),
        encoding="utf-8",
    )

    output_path = Path(output_zip)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        add_path(zip_file, package_root / "VideoPainter" / "data", Path("VideoPainter/data"))
        add_path(zip_file, scripts_dir, Path("sceneshift_scripts"))
        zip_file.write(readme, "README_GPU_RUN.txt")

    return str(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Package a prepared VideoPainter job for a GPU machine.")
    parser.add_argument("--show-id", default="orange_car")
    parser.add_argument("--job-id", default=DEFAULT_JOB_ID)
    parser.add_argument(
        "--caption",
        default="A red Ferrari sports car replaces the masked car, realistic lighting, same camera perspective, same street scene.",
    )
    parser.add_argument("--output", default="assets/output/videopainter_orange_car_job.zip")
    args = parser.parse_args()

    output = package_job(
        show_id=args.show_id,
        caption=args.caption,
        job_id=args.job_id,
        output_zip=args.output,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
