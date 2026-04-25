<<<<<<< HEAD:backend/app/api/routes/mask_routes.py
"""
mask_routes.py
──────────────
FastAPI routes for the B&W mask generation service.

Endpoints
---------
POST /generate/mask/{show_id}
    Trigger mask generation for an already-ingested scene.
    Runs synchronously (mask gen is fast; no upload involved).
    Returns JSON with the output path and a preview URL.

GET  /masks/{show_id}
    Stream the generated mask MP4 back to the client.

GET  /generate/mask/{show_id}/status
    Check whether a mask file already exists without regenerating it.
"""

import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from app.services import masking as mask_pipeline

router = APIRouter(prefix="/generate", tags=["mask"])

MASKS_DIR = "assets/masks"


# ─────────────────────────────────────────────────────────────────────────────
# POST /generate/mask/{show_id}   — generate (or regenerate) the mask video
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/mask/{show_id}")
async def generate_mask(show_id: str):
    """
    Reads the bounding-box data stored in scene_vault.json for *show_id*,
    renders a per-frame black-and-white mask video, and returns its location.

    • White pixels  → tracked object region
    • Black pixels  → everything else

    The output MP4 preserves the original video's resolution and frame-rate so
    it can be fed directly into Vedant's SAM 3 temporal propagation service.
    """
    try:
        output_path = mask_pipeline.generate_mask_video(show_id)
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
        # Convenience URL the frontend / Vedant's service can hit directly
        "preview_url":   f"/masks/{show_id}",
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /masks/{show_id}   — stream the mask MP4
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/mask/{show_id}/download", tags=["mask"])
async def download_mask(show_id: str):
    """Stream the pre-generated mask MP4 as a binary response."""
    mask_path = os.path.join(MASKS_DIR, f"{show_id}_mask.avi")

    if not os.path.exists(mask_path):
        raise HTTPException(
            status_code=404,
            detail=(
                f"No mask found for show_id='{show_id}'. "
                "Call POST /generate/mask/{show_id} first."
            ),
        )

    return FileResponse(
        path=mask_path,
        media_type="video/x-msvideo",
        filename=f"{show_id}_mask.avi",
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /generate/mask/{show_id}/status   — non-destructive existence check
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/mask/{show_id}/status", tags=["mask"])
async def mask_status(show_id: str):
    """
    Returns whether a mask video for *show_id* already exists on disk,
    along with its file size (bytes) if present.
    """
    mask_path = os.path.join(MASKS_DIR, f"{show_id}_mask.avi")
    exists    = os.path.exists(mask_path)

    return {
        "show_id":  show_id,
        "exists":   exists,
        "path":     mask_path if exists else None,
        "size_bytes": os.path.getsize(mask_path) if exists else None,
    }
=======
import os
import shutil
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from utils.masking_service import generate_temporal_alpha_masks, mask_status

router = APIRouter(prefix="/generate", tags=["mask"])
MASKS_DIR = "assets/masks"
os.makedirs(MASKS_DIR, exist_ok=True)


@router.post("/mask/{show_id}")
async def generate_mask(
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


@router.get("/mask/{show_id}/status", tags=["mask"])
async def mask_generation_status(show_id: str):
    status = mask_status(show_id)
    return {"success": True, "status": status}
>>>>>>> f909f3a (b):backend/routes/mask_routes.py
