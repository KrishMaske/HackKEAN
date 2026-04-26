import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import ingestion_routes, mask_routes, analytics_routes
from app.core.config import settings

# ── Pre-computed data directory ──────────────────────────────────────────────
DATA_DIR = "app/db/data"
os.makedirs(DATA_DIR, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup tasks."""
    print("[STARTUP] Starting ProductIntel API...")
    yield
    print("[SHUTDOWN] Shutting down ProductIntel API...")

app = FastAPI(
    title="ProductIntel API",
    version="0.3.0",
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
app.include_router(analytics_routes.router)

# ── Static file serving ──────────────────────────────────────────────────────
MASKS_DIR = "assets/masks"
INPUT_DIR = "assets/input"
UPLOADS_DIR = "assets/uploads"
os.makedirs(MASKS_DIR, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

app.mount("/masks", StaticFiles(directory=MASKS_DIR), name="masks")
app.mount("/input", StaticFiles(directory=INPUT_DIR), name="input")
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")

@app.get("/")
async def root():
    return {"message": "ProductIntel Orchestrator Active", "status": "Ready"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
