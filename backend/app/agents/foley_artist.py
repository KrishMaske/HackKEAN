# backend/agents/foley_artist.py
from app.db.schemas import AgentState
from app.core.config import settings
import json

async def foley_artist_agent(state: AgentState) -> AgentState:
    """
    Foley Artist Agent - Generates an ElevenLabs sound effects prompt 
    based on the final object selection and historical context.
    """
    final_selection = state.get("final_selection")
    historical_context = state.get("historical_context", {})
    year = historical_context.get("year", "Unknown")
    vibe = historical_context.get("vibe", "neutral")
    location = historical_context.get("location", "unspecified")

    foley_prompt = f"""You are a Master Foley Artist for Hollywood.
    Your job is to design a sound effect for the object '{final_selection}' 
    placed in a scene set in the year {year}. The scene's vibe is '{vibe}' and location is '{location}'.
    
    Return a JSON object with:
    - "sound_effect_prompt": A highly descriptive prompt for the ElevenLabs Sound Effects API (e.g., "A heavy metallic clank echoing in a quiet room").
    - "duration": The estimated duration in seconds (e.g., "2.5").
    - "ambient_layer": What background noise should be mixed in (e.g., "1980s analog office hum").

    Respond ONLY with the JSON."""

    client = settings.google_client
    response = client.models.generate_content(
        model="gemma-4-26b-a4b-it",
        contents=foley_prompt
    )

    try:
        # Clean response text in case Gemma adds markdown backticks
        clean_json = response.text.strip().replace("```json", "").replace("```", "")
        specs = json.loads(clean_json)
        state["audio_specs"] = specs
        state["reasoning_log"].append({
            "agent": "Foley Artist", 
            "action": "Audio Design", 
            "message": f"Generated sound profile for {final_selection} in {year} environment."
        })
    except Exception as e:
        state["audio_specs"] = {
            "sound_effect_prompt": f"Generic handling sound of a {final_selection}", 
            "duration": "2.0",
            "ambient_layer": "Room tone"
        }
        state["reasoning_log"].append({
            "agent": "Foley Artist", 
            "action": "Fallback", 
            "message": "Failed to parse audio specs, used generic fallback."
        })

    return state
