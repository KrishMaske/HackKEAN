from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from schemas import AgentState
from orchestrator import execute_sceneshift
from config import settings
from database import seed_db
from contextlib import asynccontextmanager

class GenerateSceneRequest(BaseModel):
    user_interest: str
    scene_id: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic: Seed the DB
    await seed_db()
    yield
    # Shutdown logic (if any) can go here

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "SceneShift Orchestrator Active"}

@app.post("/chat")
async def chat(message: str):
    client = settings.google_client
    response = client.models.generate_content(
        model="gemma-4-26b-a4b-it",
        contents=message
    )
    return {"response": response.text}

@app.post("/generate-scene")
async def generate_scene(request: GenerateSceneRequest, guardrails: bool = True):
    try:
        result = await execute_sceneshift(request.user_interest, request.scene_id, guardrails)        
        return {
            "success": True,
            "final_selection": result.get("final_selection"),
            "reasoning_log": result.get("reasoning_log", [])
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)