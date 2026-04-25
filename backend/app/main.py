import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from app.core.config import settings
from contextlib import asynccontextmanager
from app.api.routes import ingestion_routes, mask_routes

try:
    from app.db.schemas import AgentState
    from app.services.orchestrator import execute_sceneshift
    from app.db.database import seed_db
    _HAS_ORCHESTRATOR = True
except ImportError:
    _HAS_ORCHESTRATOR = False

class GenerateSceneRequest(BaseModel):
    user_interest: str
    scene_id: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup tasks like seeding the MongoDB Scene Vault."""
    print("[STARTUP] Starting SceneShift API...")
    if _HAS_ORCHESTRATOR:
        await seed_db()
    yield
    print("[SHUTDOWN] Shutting down SceneShift API...")

app = FastAPI(
    title="SceneShift API",
    version="0.2.0",
    lifespan=lifespan
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(ingestion_routes.router)
app.include_router(mask_routes.router)

# ── Static file serving (mask videos) ────────────────────────────────────────
# Relative to the root of the project
MASKS_DIR = "assets/masks"
os.makedirs(MASKS_DIR, exist_ok=True)
app.mount("/masks", StaticFiles(directory=MASKS_DIR), name="masks")

@app.get("/")
async def root():
    return {"message": "SceneShift Orchestrator Active", "status": "Ready"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/chat")
async def chat(message: str):
    client = settings.groq_client
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": message}]
    )
    return {"response": response.choices[0].message.content}

@app.post("/generate-scene")
async def generate_scene(request: GenerateSceneRequest, guardrails: bool = True):
    if not _HAS_ORCHESTRATOR:
        return {"success": False, "error": "orchestrator components unavailable"}
    try:
        result = await execute_sceneshift(request.user_interest, request.scene_id, guardrails)        
        return {
            "success": True,
            "final_selection": result.get("final_selection"),
            "reasoning_log": result.get("reasoning_log", []),
            "audio_specs": result.get("audio_specs", {})
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
