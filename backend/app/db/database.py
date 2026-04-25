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
        "lighting": "Warm incandescent"
    },
    {
        "scene_id": "the_office_05",
        "show": "The Office",
        "year": 2005,
        "location": "Scranton, PA",
        "forbidden_tech": ["iPhone", "Streaming", "USB-C"],
        "vibe": "Corporate/Flat",
        "lighting": "Cool fluorescent"
    },
    {
        "scene_id": "succession_20",
        "show": "Succession",
        "year": 2020,
        "location": "NYC",
        "forbidden_tech": ["Generative AI", "Vision Pro"],
        "vibe": "Ultra-Luxury/Cold",
        "lighting": "Natural window light"
    }
]

async def seed_db():
    """Wipes and re-seeds the vault for clean hackathon testing."""
    await scene_vault.delete_many({})
    await scene_vault.insert_many(DEMO_SCENES)
    print("Database Seeded Successfully!")

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
