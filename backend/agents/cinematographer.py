# backend/agents/cinematographer.py
from schemas import AgentState
from config import settings
from tools.vision_tools import trigger_visual_generation

async def cinematographer_agent(state: AgentState) -> AgentState:
    """
    Analyzes scene lighting and object material to ensure visual harmony.
    """
    final_selection = state.get("final_selection")
    historical_context = state.get("historical_context", {})
    lighting_context = historical_context.get("lighting_context", "natural")

    cinematography_prompt = f"""You are a Master Cinematographer. 
    Analyze how a '{final_selection}' should look in a scene with '{lighting_context}' lighting.
    
    Return a JSON object with:
    - "lighting_intensity": (float 0.0 to 1.0)
    - "material_type": (string, e.g., 'matte', 'metallic', 'glass')
    - "shadow_direction": (string, e.g., 'top-down', 'diffuse', 'left-to-right')
    - "color_temperature": (string, e.g., 'warm', 'cool')

    Respond ONLY with the JSON."""

    client = settings.google_client
    response = client.models.generate_content(
        model="gemma-4-26b-a4b-it",
        contents=cinematography_prompt
    )

    import json
    try:
        # Clean response text in case Gemma adds markdown backticks
        clean_json = response.text.strip().replace("```json", "").replace("```", "")
        specs = json.loads(clean_json)
        state["visual_specs"] = specs
        state["reasoning_log"].append(f"Cinematographer Agent: Applied {specs['material_type']} physics for {lighting_context} environment.")
    except:
        state["visual_specs"] = {"material_type": "matte", "lighting_intensity": 0.5}
        state["reasoning_log"].append("Cinematographer Agent: Failed to parse specs, used matte fallback.")

    # Trigger the physical world generation
    vision_result = await trigger_visual_generation(
        object_name=state["final_selection"],
        lighting_specs=state["visual_specs"],
        scene_id=state["scene_id"]
    )
    
    state["reasoning_log"].append(f"Vision Pipeline: {vision_result['status']} (Job: {vision_result.get('job_id')})")
    return state