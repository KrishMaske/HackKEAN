"""
mask_routes.py
──────────────
FastAPI routes for mask generation services.
"""

import os
import shutil
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

# We import both functionalities to support both Krish's and Vedant's tasks
from app.services.masking import generate_temporal_alpha_masks, mask_status, generate_mask_video

router = APIRouter(prefix="/generate", tags=["mask"])
MASKS_DIR = "assets/masks"
os.makedirs(MASKS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# POST /generate/mask/temporal/{show_id} — generate temporal alpha mask sequence
# ─────────────────────────────────────────────────────────────────────────────
@router.post("/mask/temporal/{show_id}")
async def generate_temporal_mask(
    show_id: str,
    manual_mask: UploadFile = File(None),
    anchor_frame: int = Form(0),
    prompt: str = Form(None),
):
    """Generate a temporal alpha mask sequence using a manual anchor mask and SAM 3 propagation."""
    temp_mask_path = None
    try:
        if manual_mask:
            if manual_mask.content_type not in {"image/png", "image/jpeg", "image/jpg", "image/bmp"}:
                raise HTTPException(status_code=400, detail="Manual mask must be an image file.")

            temp_mask_path = os.path.join(MASKS_DIR, f"{show_id}_manual_mask.png")
            with open(temp_mask_path, "wb") as buffer:
                shutil.copyfileobj(manual_mask.file, buffer)

        result = generate_temporal_alpha_masks(
            show_id=show_id,
            manual_mask_path=temp_mask_path,
            anchor_frame_index=anchor_frame,
            prompt=prompt,
        )

        return {
            "success": True,
            "show_id": show_id,
            "frame_count": result["frame_count"],
            "mask_directory": result["mask_directory"],
            "mask_video": result["mask_video"],
            "preview_url": result["preview_url"],
            "frame_urls": result["frame_urls"],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if temp_mask_path and os.path.exists(temp_mask_path):
            os.remove(temp_mask_path)


# ─────────────────────────────────────────────────────────────────────────────
# POST /generate/mask/bw/{show_id}   — generate (or regenerate) the B&W mask video
# ─────────────────────────────────────────────────────────────────────────────
@router.post("/mask/bw/{show_id}")
async def generate_bw_mask(show_id: str):
    """
    Reads the bounding-box data stored in scene_vault.json for *show_id*,
    renders a per-frame black-and-white mask video, and returns its location.
    """
    try:
        output_path = generate_mask_video(show_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    filename = os.path.basename(output_path)

    return {
        "show_id":       show_id,
        "status":        "success",
        "mask_filename": filename,
        "mask_path":     output_path,
        "preview_url":   f"/masks/{show_id}",
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /masks/{show_id}/download   — stream the mask MP4
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/mask/{show_id}/download", tags=["mask"])
async def download_mask(show_id: str):
    """Stream the pre-generated mask MP4 as a binary response."""
    mask_path = os.path.join(MASKS_DIR, f"{show_id}_mask.mp4")

    if not os.path.exists(mask_path):
        raise HTTPException(
            status_code=404,
            detail=(
                f"No mask found for show_id='{show_id}'. "
                "Call POST /generate/mask/bw/{show_id} or /mask/temporal/{show_id} first."
            ),
        )

    return FileResponse(
        path=mask_path,
        media_type="video/mp4",
        filename=f"{show_id}_mask.mp4",
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /generate/mask/{show_id}/status   — non-destructive existence check
# ─────────────────────────────────────────────────────────────────────────────
@router.get("/mask/{show_id}/status", tags=["mask"])
async def get_mask_status(show_id: str):
    """
    Returns whether a mask video for *show_id* already exists on disk,
    along with its file size (bytes) if present.
    """
    try:
        # Krish's original status call
        status_msg = mask_status(show_id)
    except Exception:
        status_msg = "Unknown"

    mask_path = os.path.join(MASKS_DIR, f"{show_id}_mask.mp4")
    exists    = os.path.exists(mask_path)

    return {
        "success": True,
        "status_message": status_msg,
        "show_id":  show_id,
        "exists":   exists,
        "path":     mask_path if exists else None,
        "size_bytes": os.path.getsize(mask_path) if exists else None,
    }
