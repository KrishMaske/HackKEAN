from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings

# MongoDB Connection
client = AsyncIOMotorClient(settings.mongodb_uri)
db = client.sceneshift_db
scene_vault = db.scene_vault

# Task 1.2: The "Big Three" Scene Metadata
DEMO_SCENES = [
    {
        "scene_id": "stranger_things_83",
        "show": "Stranger Things",
        "year": 1983,
        "location": "Hawkins, IN",
        "forbidden_tech": ["Cellphones", "Internet", "Laptops", "CDs"],
        "vibe": "Nostalgic/Gritty",
        "lighting": "Warm incandescent",
        "video_path": "assets/input/STRANGER_THINGS_CLIP.mp4",
        "coordinate_map": [100, 100, 300, 300]
    },
    {
        "scene_id": "the_office_05",
        "show": "The Office",
        "year": 2005,
        "location": "Scranton, PA",
        "forbidden_tech": ["iPhone", "Streaming", "USB-C"],
        "vibe": "Corporate/Flat",
        "lighting": "Cool fluorescent",
        "video_path": "assets/input/OFFICE_CLIP.mp4",
        "coordinate_map": [150, 150, 250, 250]
    },
    {
        "scene_id": "succession_20",
        "show": "Succession",
        "year": 2020,
        "location": "NYC",
        "forbidden_tech": ["Generative AI", "Vision Pro"],
        "vibe": "Ultra-Luxury/Cold",
        "lighting": "Natural window light",
        "video_path": "",
        "coordinate_map": [0, 0, 0, 0]
    }
]

async def seed_db():
    """Seed the vault with default demo scenes when it is empty."""
    if await scene_vault.count_documents({}) > 0:
        return

    await scene_vault.insert_many(DEMO_SCENES)
    print("Database seeded successfully.")

async def get_historical_context(scene_id: str) -> dict:
    """
    Query the scene vault to get historical context for a given scene_id.
    """
    scene = await scene_vault.find_one({"scene_id": scene_id})
    
    if not scene:
        return {}
    
    return {
        "year": scene.get("year"),
        "forbidden_tech": scene.get("forbidden_tech", []),
        "vibe": scene.get("vibe"),
        "location": scene.get("location"),
        "show_name": scene.get("show"),
        "lighting_context": scene.get("lighting")
    }
