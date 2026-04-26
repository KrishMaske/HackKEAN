from fastapi import APIRouter, HTTPException

from app.services.analytics import build_product_analytics


router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/{show_id}")
async def get_product_analytics(show_id: str):
    try:
        return await build_product_analytics(show_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
