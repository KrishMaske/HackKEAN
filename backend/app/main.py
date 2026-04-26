import os
import json
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
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

# ── Pre-computed data directory ──────────────────────────────────────────────
DATA_DIR = "app/db/data"
os.makedirs(DATA_DIR, exist_ok=True)

class GenerateSceneRequest(BaseModel):
    user_interest: str
    scene_id: str
    thread_id: str = "default"  # Optional session key for checkpointing

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup tasks like seeding the MongoDB Scene Vault."""
    print("[STARTUP] Starting SceneShift API...")
    if _HAS_ORCHESTRATOR:
        await seed_db()
    yield
    print("[SHUTDOWN] Shutting down SceneShift API...")

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="SceneShift API",
    version="0.2.0",
    lifespan=lifespan
)

# ── CORS Middleware ──────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(ingestion_routes.router)
app.include_router(mask_routes.router)

# ── Static file serving (videos and masks) ───────────────────────────────────
# Relative to the root of the project
MASKS_DIR = "assets/masks"
INPUT_DIR = "assets/input"
os.makedirs(MASKS_DIR, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True)
app.mount("/masks", StaticFiles(directory=MASKS_DIR), name="masks")
app.mount("/input", StaticFiles(directory=INPUT_DIR), name="input")

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
    # ── Cache-First: check for pre-computed agent results ────────────────────
    cache_path = os.path.join(DATA_DIR, "agent_cache.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                cached_results = json.load(f)
            for result in cached_results:
                if (result.get("show_id") == request.scene_id and
                    result.get("persona") == request.user_interest and
                    result.get("success")):
                    return {
                        "success": True,
                        "final_selection": result.get("final_selection"),
                        "reasoning_log": result.get("reasoning_log", []),
                        "audio_specs": result.get("audio_specs", {}),
                        "visual_specs": result.get("visual_specs"),
                        "source": "pre-computed",
                    }
        except (json.JSONDecodeError, KeyError):
            pass  # Fall through to live execution

    # ── Live Fallback ────────────────────────────────────────────────────────
    if not _HAS_ORCHESTRATOR:
        return {"success": False, "error": "orchestrator components unavailable"}
    try:
        result = await execute_sceneshift(
            request.user_interest,
            request.scene_id,
            guardrails,
            thread_id=request.thread_id,
        )        
        
        # Grab the original scene metadata for the coordinates
        from app.db.database import scene_vault
        scene = await scene_vault.find_one({"scene_id": request.scene_id})
        mask_area = None
        if scene and "coordinate_map" in scene:
            coords = scene["coordinate_map"]
            if len(coords) == 4:
                mask_area = {
                    "x": coords[0],
                    "y": coords[1],
                    "w": coords[2] - coords[0],
                    "h": coords[3] - coords[1]
                }
                
        return {
            "success": True,
            "final_selection": result.get("final_selection"),
            "reasoning_log": result.get("reasoning_log", []),
            "audio_specs": result.get("audio_specs", {}),
            "visual_specs": result.get("visual_specs"),
            "mask_area": mask_area,
            "source": "live",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ── Pre-Computed Data API ─────────────────────────────────────────────────────

@app.get("/api/scene/{show_id}")
async def get_scene(show_id: str):
    """Return pre-computed scene metadata for a show."""
    scene_path = os.path.join(DATA_DIR, f"{show_id}.json")
    if not os.path.exists(scene_path):
        return JSONResponse(
            status_code=404,
            content={"error": f"No pre-computed scene data for '{show_id}'"},
        )
    with open(scene_path, "r") as f:
        return json.load(f)


@app.get("/api/agents/{show_id}/{user_interest}")
async def get_agent_result(show_id: str, user_interest: str):
    """Return pre-computed agent reasoning for a (show, persona) combination."""
    cache_path = os.path.join(DATA_DIR, "agent_cache.json")
    if not os.path.exists(cache_path):
        return JSONResponse(
            status_code=404,
            content={"error": "No pre-computed agent results available"},
        )
    with open(cache_path, "r") as f:
        results = json.load(f)

    for result in results:
        if (result.get("show_id") == show_id and
            result.get("persona") == user_interest):
            return result

    return JSONResponse(
        status_code=404,
        content={"error": f"No results for {show_id} × {user_interest}"},
    )


@app.get("/api/shows")
async def get_shows():
    """Return the list of available shows with pre-computed data."""
    shows = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".json") and not filename.startswith("agent"):
            show_id = filename.replace(".json", "")
            shows.append({"show_id": show_id, "data_file": filename})
    return {"shows": shows}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
