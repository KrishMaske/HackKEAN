import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def run(command: list[str], cwd: str | None = None) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def patch_videopainter_for_direct_prompt(videopainter_root: Path) -> None:
    inpaint_path = videopainter_root / "infer" / "inpaint.py"
    text = inpaint_path.read_text(encoding="utf-8")
    old = "prompt, image_inpainting_prompt = video_editing_prompt(prompt, llm_model, masked_image=masked_image)"
    new = "image_inpainting_prompt = prompt"
    if old in text and new not in text:
        inpaint_path.write_text(text.replace(old, new), encoding="utf-8")


def extract_job(job_zip: Path, videopainter_root: Path) -> None:
    if not job_zip.exists():
        raise FileNotFoundError(f"Job zip not found: {job_zip}")

    with zipfile.ZipFile(job_zip, "r") as archive:
        archive.extractall("/content/sceneshift_job")

    source_data = Path("/content/sceneshift_job/VideoPainter/data")
    target_data = videopainter_root / "data"
    if target_data.exists():
        shutil.rmtree(target_data)
    shutil.copytree(source_data, target_data)


def ensure_openai_dummy_key() -> None:
    # VideoPainter creates an OpenAI client at import time even when llm_model is unused.
    os.environ.setdefault("OPENAI_API_KEY", "unused-videopainter-local-run")


def run_videopainter(
    videopainter_root: Path,
    job_id: str,
    prompt: str,
    frames: int,
    fps: int,
    steps: int,
    guidance: float,
    dilation: int,
    preserve_gt: bool,
    use_mask_add: bool,
    img_inpainting_model: str | None,
) -> Path:
    ensure_openai_dummy_key()
    patch_videopainter_for_direct_prompt(videopainter_root)
    output_dir = videopainter_root / "infer" / "sceneshift_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "inpaint.py",
        "--prompt",
        prompt,
        "--model_path",
        "../ckpt/CogVideoX-5b-I2V",
        "--inpainting_branch",
        "../ckpt/VideoPainter/checkpoints/branch",
        "--output_path",
        f"./sceneshift_output/{job_id}.mp4",
        "--num_inference_steps",
        str(steps),
        "--guidance_scale",
        str(guidance),
        "--num_videos_per_prompt",
        "1",
        "--dtype",
        "bfloat16",
        "--generate_type",
        "i2v_inpainting",
        "--inpainting_mask_meta",
        "../data/sceneshift_videopainter.csv",
        "--inpainting_sample_id",
        "0",
        "--inpainting_frames",
        str(frames),
        "--image_or_video_path",
        "../data/videovo/raw_video",
        "--down_sample_fps",
        str(fps),
        "--dilate_size",
        str(dilation),
        "--long_video",
    ]
    if preserve_gt:
        command.extend(["--first_frame_gt", "--replace_gt"])
    if use_mask_add:
        command.append("--mask_add")
    if img_inpainting_model:
        command.extend(["--img_inpainting_model", img_inpainting_model])

    run(command, cwd=str(videopainter_root / "infer"))
    return output_dir / f"{job_id}_fps{fps}.mp4"


def copy_result(rendered_video: Path, output_path: Path) -> None:
    if not rendered_video.exists():
        raise FileNotFoundError(f"Expected VideoPainter output not found: {rendered_video}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(rendered_video, output_path)
    print(f"Copied result to: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a packaged SceneShift VideoPainter job on Google Colab.")
    parser.add_argument("--job-zip", default="/content/videopainter_orange_car_job.zip")
    parser.add_argument("--videopainter-root", default="/content/VideoPainter")
    parser.add_argument("--job-id", default="sceneshift_orange_car000")
    parser.add_argument("--frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--steps", type=int, default=70)
    parser.add_argument("--guidance", type=float, default=9.0)
    parser.add_argument("--dilation", type=int, default=48)
    parser.add_argument("--preserve-gt", action="store_true")
    parser.add_argument("--mask-add", action="store_true")
    parser.add_argument("--img-inpainting-model", default=None)
    parser.add_argument(
        "--prompt",
        default=(
            "A bright red Ferrari sports car in the masked region, realistic glossy paint, "
            "same size as the original car, matching street lighting, camera perspective, "
            "shadows, reflections, and occlusion. Remove the orange car completely."
        ),
    )
    parser.add_argument("--output", default="/content/orange_car_videopainter_render.mp4")
    args = parser.parse_args()

    videopainter_root = Path(args.videopainter_root)
    extract_job(Path(args.job_zip), videopainter_root)
    rendered_video = run_videopainter(
        videopainter_root=videopainter_root,
        job_id=args.job_id,
        prompt=args.prompt,
        frames=args.frames,
        fps=args.fps,
        steps=args.steps,
        guidance=args.guidance,
        dilation=args.dilation,
        preserve_gt=args.preserve_gt,
        use_mask_add=args.mask_add,
        img_inpainting_model=args.img_inpainting_model,
    )
    copy_result(rendered_video, Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
