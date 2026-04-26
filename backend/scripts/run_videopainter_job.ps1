param(
    [string]$VideoPainterRoot = "..\VideoPainter",
    [string]$ShowId = "orange_car",
    [string]$JobId = "sceneshift_orange_car000",
    [string]$Prompt = "A red Ferrari sports car replaces the masked car, realistic lighting, same camera perspective, same street scene.",
    [int]$Frames = 49,
    [int]$DownSampleFps = 8,
    [int]$Steps = 50,
    [double]$Guidance = 6.0,
    [int]$Dilation = 32,
    [switch]$SkipPreflight
)

$ErrorActionPreference = "Stop"

$BackendRoot = Split-Path -Parent $PSScriptRoot
$VideoPainterFullPath = Resolve-Path -Path (Join-Path $BackendRoot $VideoPainterRoot)
$env:HF_HOME = Join-Path $VideoPainterFullPath ".hf_cache"
$env:HUGGINGFACE_HUB_CACHE = Join-Path $env:HF_HOME "hub"
if (-not $env:OPENAI_API_KEY) {
    $env:OPENAI_API_KEY = "unused-videopainter-local-run"
}
New-Item -ItemType Directory -Force -Path $env:HUGGINGFACE_HUB_CACHE | Out-Null

function Invoke-Checked {
    param([scriptblock]$Command)
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE"
    }
}

if (-not $SkipPreflight) {
    python -c "import sys, torch; print('python', sys.version.split()[0]); print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); raise SystemExit(0 if sys.version_info[:2] == (3, 10) and torch.cuda.is_available() else 1)"
    if ($LASTEXITCODE -ne 0) {
        throw "VideoPainter requires Python 3.10 with CUDA-enabled PyTorch. This machine is not ready for local VideoPainter inference."
    }

    $RequiredPaths = @(
        "ckpt\CogVideoX-5b-I2V",
        "ckpt\VideoPainter\checkpoints\branch"
    )
    foreach ($Path in $RequiredPaths) {
        $FullPath = Join-Path $VideoPainterFullPath $Path
        if (-not (Test-Path -Path $FullPath)) {
            throw "Missing VideoPainter checkpoint path: $FullPath"
        }
    }
}

Push-Location $BackendRoot
Invoke-Checked {
    python scripts\prepare_videopainter_job.py `
        --show-id $ShowId `
        --videopainter-root $VideoPainterFullPath `
        --job-id $JobId `
        --caption $Prompt
}
Pop-Location

$OutputDir = Join-Path $VideoPainterFullPath "infer\sceneshift_output"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

Push-Location (Join-Path $VideoPainterFullPath "infer")
Invoke-Checked {
    python inpaint.py `
        --prompt $Prompt `
        --model_path "..\ckpt\CogVideoX-5b-I2V" `
        --inpainting_branch "..\ckpt\VideoPainter\checkpoints\branch" `
        --output_path ".\sceneshift_output\$JobId.mp4" `
        --num_inference_steps $Steps `
        --guidance_scale $Guidance `
        --num_videos_per_prompt 1 `
        --dtype "bfloat16" `
        --generate_type "i2v_inpainting" `
        --inpainting_mask_meta "..\data\sceneshift_videopainter.csv" `
        --inpainting_sample_id 0 `
        --inpainting_frames $Frames `
        --image_or_video_path "..\data\videovo\raw_video" `
        --first_frame_gt `
        --replace_gt `
        --mask_add `
        --down_sample_fps $DownSampleFps `
        --dilate_size $Dilation `
        --long_video
}
Pop-Location

$GeneratedPath = Join-Path $OutputDir "$($JobId)_fps$DownSampleFps.mp4"
Push-Location $BackendRoot
Invoke-Checked {
    python scripts\import_videopainter_output.py --show-id $ShowId --source-video $GeneratedPath
}
Pop-Location
