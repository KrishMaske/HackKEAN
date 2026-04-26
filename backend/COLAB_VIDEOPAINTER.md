# Google Colab VideoPainter Run

Use this when local machines do not have an NVIDIA CUDA GPU.

## 1. Upload The Job Zip

Upload this file to Colab:

```text
backend/assets/output/videopainter_orange_car_job.zip
```

In Colab, enable GPU:

```text
Runtime > Change runtime type > T4 GPU
```

## 2. Setup VideoPainter

Run these cells in Colab.

```bash
!nvidia-smi
%cd /content
!git clone https://github.com/TencentARC/VideoPainter.git
%cd /content/VideoPainter
!pip install -r requirements.txt
%cd /content/VideoPainter/diffusers
!pip install -e .
```

If `requirements.txt` fails on `deepspeed`, continue with:

```bash
%cd /content/VideoPainter
!pip install torch torchvision transformers==4.42.2 accelerate opencv-python pillow imageio imageio-ffmpeg moviepy decord kornia scikit-image openai peft sentencepiece scipy einops rotary_embedding_torch pytorch_lightning easydict pandas matplotlib safetensors huggingface_hub av gdown omegaconf beartype
%cd /content/VideoPainter/diffusers
!pip install -e .
```

## 3. Download Checkpoints

This is large. Colab may take a while.

```bash
%cd /content/VideoPainter
!git lfs install
!mkdir -p ckpt
%cd /content/VideoPainter/ckpt
!git clone https://huggingface.co/TencentARC/VideoPainter
!git clone https://huggingface.co/THUDM/CogVideoX-5b-I2V
```

If Hugging Face asks for auth, run:

```bash
!huggingface-cli login
```

Then rerun the failed clone.

## 4. Upload Runner Script

Upload this repo file to Colab:

```text
backend/scripts/colab_videopainter_runner.py
```

Put it at:

```text
/content/colab_videopainter_runner.py
```

Also make sure the zip is at:

```text
/content/videopainter_orange_car_job.zip
```

## 5. Render

```bash
%cd /content
!python colab_videopainter_runner.py \
  --job-zip /content/videopainter_orange_car_job.zip \
  --videopainter-root /content/VideoPainter \
  --output /content/orange_car_videopainter_render.mp4
```

Download:

```text
/content/orange_car_videopainter_render.mp4
```

## 6. Import Back Into The App

Put the downloaded MP4 anywhere on your local machine, then run from `backend`:

```powershell
python scripts\import_videopainter_output.py --show-id orange_car --source-video <path-to-downloaded-mp4>
```

Then call:

```json
{
  "show_id": "orange_car",
  "prompt": "red Ferrari sports car",
  "provider": "videopainter"
}
```

