import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager

# Project Imports
from schemas import AgentState
from orchestrator import execute_sceneshift
from config import settings
from database import seed_db
from routes import ingestion_routes
# Note: Ensure mask_routes.py exists in your routes/ folder
try:
    from routes import mask_routes
except ImportError:
    mask_routes = None

# ── Request Models ───────────────────────────────────────────────────────────
class GenerateSceneRequest(BaseModel):
    user_interest: str
    scene_id: str

# ── Lifespan Management ──────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup tasks like seeding the MongoDB Scene Vault."""
    print("🚀 Starting SceneShift API...")
    await seed_db()
    yield
    print("🛑 Shutting down SceneShift API...")

# ── App Initialization ───────────────────────────────────────────────────────
app = FastAPI(
    title="SceneShift API", 
    version="0.1.0",
    lifespan=lifespan
)

# ── Routers & Static Files ───────────────────────────────────────────────────
app.include_router(ingestion_routes.router)
if mask_routes:
    app.include_router(mask_routes.router)

# Ensure the masks directory exists for Krish's video assets
MASKS_DIR = "assets/masks"
os.makedirs(MASKS_DIR, exist_ok=True)
app.mount("/masks", StaticFiles(directory=MASKS_DIR), name="masks")

# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"message": "SceneShift Orchestrator Active"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

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
    """
    The main entry point for the Agentic Workflow.
    Combines user interest with historical context to suggest an object.
    """
    try:
        result = await execute_sceneshift(request.user_interest, request.scene_id, guardrails)        
        return {
            "success": True,
            "final_selection": result.get("final_selection"),
            "reasoning_log": result.get("reasoning_log", [])
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# ── Server Entry ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    # Using 'main:app' allows for the 'reload' feature during development
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)