import os
import shutil
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from app.services.masking import generate_temporal_alpha_masks, mask_status

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
