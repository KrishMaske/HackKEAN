import os
import shutil
from fastapi import APIRouter, BackgroundTasks, Form, HTTPException, UploadFile, File
from app.services import ingestion as ingestion_pipeline

router = APIRouter()

UPLOAD_DIR = "assets/uploads/"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/ingest/video")
async def ingest_video(
    background_tasks: BackgroundTasks,
    show_id: str = Form(...),
    file: UploadFile = File(...),
    target_object: str = Form(None)
):
    file_location = os.path.join(UPLOAD_DIR, file.filename)

    if file.filename not in os.listdir(UPLOAD_DIR):
        try:
            with open(file_location, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not save file: {e}")
        finally:
            file.file.close()
    else:
        print(f"File {file.filename} already exists. Skipping upload.")

    background_tasks.add_task(ingestion_pipeline.analyze_data, file_location, show_id, target_object)

    return {"filename": file.filename, "content_type": file.content_type, "target_object": target_object}