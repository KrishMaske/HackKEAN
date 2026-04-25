from pydantic import BaseModel, Field
from typing import List, Optional

class SceneMetadata(BaseModel):
    scene_id: str
    show_name: str
    year: int
    location: str
    vibe: str  # e.g., "Gritty", "Corporate", "Nostalgic"
    lighting_context: str # e.g., "Fluorescent", "Warm Lamp", "Natural Sunlight"
    forbidden_tech: List[str] # Tech that definitely didn't exist yet
    ambient_objects: List[str] # The items we are allowed to swap