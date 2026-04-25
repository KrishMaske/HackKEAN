import httpx
import logging

VISION_SERVER_URL = "http://localhost:8001/inpaint"

async def trigger_visual_generation(object_name: str, lighting_specs: dict, scene_id: str):
    """
    Sends the final object selection and physical specs to Vedant's Vision Pipeline.
    """
    payload = {
        "scene_id": scene_id,
        "target_object": object_name,
        "physics_specs": lighting_specs
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(VISION_SERVER_URL, json=payload, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            
            return {
                "status": "triggered",
                "job_id": data.get("job_id"),
                "render_url": data.get("render_url")
            }
    except Exception as e:
        logging.error(f"Vision Tool Error: {e}")
        return {"status": "error", "message": str(e)}