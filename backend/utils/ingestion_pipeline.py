import os
import time
import json
from config import settings
from google.genai import types
from google.genai import errors as genai_errors

def analyze_data(path, show_id):
    client = settings.google_client
    file = client.files.upload(file=path)
    
    while file.state == types.FileState.PROCESSING:
        print(".", end="", flush=True)
        time.sleep(5)
        file = client.files.get(name=file.name)
        
    if file.state == types.FileState.FAILED:
        error_message = file.error.message if file.error else "Unknown file processing error"
        raise Exception(f"File processing failed: {error_message}")
    
    prompt = """
    Watch this video clip. You are an automated spatial mapping engine for a video editing pipeline.
        1. Determine the production year and the visual setting of this scene.
        2. Identify ONE prominent, inanimate background object (like a cup, phone, or poster) that can be tracked throughout the video.
        3. For EVERY frame in the video, estimate the rough [x_min, y_min, x_max, y_max] pixel coordinates of that same object.
    
    Output strictly as a JSON object matching this exact schema:
    { 
      "production_year": int, 
      "location": str, 
      "target_object": str, 
            "frame_bounding_boxes": [
                {
                    "frame_index": int,
                    "timestamp_ms": int,
                    "bounding_box": [int, int, int, int]
                }
            ]
    }

        Rules:
        - Include one entry for every frame in the video, in order.
        - Use frame_index starting at 0.
        - If the object is not visible in a frame, set bounding_box to [null, null, null, null].
        - Keep coordinates as integers when visible.
    """
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[file, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2  # Low temperature for factual, consistent extraction
        )
    )

    if not response.text:
        raise Exception("Model returned an empty response while extracting scene metadata.")

    scene_data = json.loads(response.text)
    scene_data["show_id"] = show_id
    scene_data["filepath"] = path
    if "frame_bounding_boxes" in scene_data and scene_data["frame_bounding_boxes"]:
        scene_data["initial_bounding_box"] = scene_data["frame_bounding_boxes"][0]["bounding_box"]
    
    save_to_vault(scene_data)
    print(f"💾 Successfully ingested {file.name} into db/scene_vault.json!")

def save_to_vault(scene_data):
    vault_path = "db/scene_vault.json"
    
    if not os.path.exists(vault_path):
        os.makedirs(os.path.dirname(vault_path), exist_ok=True)
        with open(vault_path, "w") as f:
            json.dump({"scenes": []}, f)
    
    try:
        with open(vault_path, "r") as f:
            content = f.read().strip()
            vault = json.loads(content) if content else {"scenes": []}
    except json.JSONDecodeError:
        vault = {"scenes": []}
        
    if "scenes" not in vault or not isinstance(vault["scenes"], list):
        vault["scenes"] = []

    vault["scenes"].append(scene_data)
    
    with open(vault_path, "w") as f:
        json.dump(vault, f, indent=4)