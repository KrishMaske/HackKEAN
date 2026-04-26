import os
import PIL.Image
from fastapi import FastAPI, Request
from omni_python_sdk import OmniAPI

app = FastAPI()
painter = OmniAPI(api_key=settings.OMNIPAINTER_KEY)

class PhysicsMapper:
    MATERIAL_MAP = {
        "translucent_plastic": "frosted poly-carbonate, subsurface scattering, soft diffused light",
        "brushed_metal": "anisotropic highlights, industrial steel, cold reflections",
        "matte_paper": "uncoated paper texture, slight yellowing, ink bleed, non-reflective",
        "lacquered_wood": "high gloss finish, deep grain depth, warm specular highlights"
    }

    @classmethod
    def build_final_prompt(cls, base_object, material, temp, intensity):
        mat_specs = cls.MATERIAL_MAP.get(material, "photorealistic texture")
        color_tag = "warm tungsten glow" if temp < 4000 else "cool daylight"
        intensity_tag = "harsh shadows" if intensity > 0.8 else "soft ambient lighting"
        return f"A {base_object}, {mat_specs}, {color_tag}, {intensity_tag}, high quality render."

@app.post("/inpaint")
async def standalone_inpaint(request: Request):
    payload = await request.json()
    
    scene_id = payload.get("scene_id", "test_scene")
    target = payload.get("target_object", "object")
    physics = payload.get("physics_specs", {})

    final_prompt = PhysicsMapper.build_final_prompt(
        target, 
        physics.get("material_type"),
        physics.get("color_temperature", 4500),
        physics.get("lighting_intensity", 0.5)
    )

    image_path = f"assets/scenes/{scene_id}_bg.png"
    mask_path = f"assets/masks/{scene_id}_alpha.png"

   
    if not os.path.exists(image_path):
        return {"status": "error", "message": f"MISSING ASSET: Place image at {image_path}"}
    if not os.path.exists(mask_path):
        return {"status": "error", "message": f"MISSING MASK: Place mask at {mask_path}"}

    image = PIL.Image.open(image_path)
    mask = PIL.Image.open(mask_path).convert("L")

    result = painter.inpaint(image=image, mask=mask, prompt=final_prompt)
    
    output_url = f"assets/processed_frames/{scene_id}_final.png"
    result.save(output_url)

    return {"status": "success", "render_url": output_url, "applied_prompt": final_prompt}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
