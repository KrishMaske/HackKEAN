import os
import shutil
from fastapi import APIRouter, BackgroundTasks, Form, HTTPException, UploadFile, File
from utils import ingestion_pipeline

router = APIRouter()

UPLOAD_DIR = "assets/uploads/"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/ingest/video")
async def ingest_video(background_tasks: BackgroundTasks, show_id: str = Form(...), file: UploadFile = File(...)):
    if not file.filename in os.listdir(UPLOAD_DIR):

        file_location = os.path.join(UPLOAD_DIR, file.filename)
        
        try:
            with open(file_location, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not save file: {e}")
        finally:
            file.file.close()        
    else:
        print(f"File {file.filename} already exists. Skipping upload.")
    
    
    background_tasks.add_task(ingestion_pipeline.analyze_data, file_location, show_id)
    
    return {"filename": file.filename, "content_type": file.content_type}