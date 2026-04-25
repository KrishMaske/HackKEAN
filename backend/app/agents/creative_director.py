
from typing import List, Optional
from app.core.config import settings
from app.db.schemas import AgentState

# Task 1.5: Create the Creative Director Agent Node
def creative_director_agent(state: AgentState) -> AgentState:
    """
    Elite Hollywood Production Designer Agent - Suggests ambient objects.
    
    Uses Gemma 4 to suggest three ambient objects based on user_interest and
    the vibe/location from historical_context.
    """
    user_interest = state.get("user_interest", "")
    historical_context = state.get("historical_context", {})
    correction_note = state.get("correction_note")  # Set by Historian on loop-back

    # Extract vibe and location from historical_context
    vibe = historical_context.get("vibe", "neutral")
    location = historical_context.get("location", "unspecified")
    show_name = historical_context.get("show_name", "the show")
    
    if not user_interest:
        state["reasoning_log"].append({"agent": "Creative Director", "action": "Error", "message": "No user interest provided"})
        return state

    # Increment retry counter (starts at 0 in initial state)
    state["retry_count"] = state.get("retry_count", 0) + 1

    # Build the correction block that gets appended to the prompt on loop-back
    correction_block = ""
    if correction_note:
        correction_block = f"""
⚠️ HISTORIAN CORRECTION NOTE (Attempt #{state['retry_count']}):
{correction_note}

You MUST take this feedback into account and suggest entirely different objects this time.
"""

    # Build the creative prompt for Gemma 4
    creative_prompt = f"""You are an elite Hollywood Production Designer. Your goal is to dress the set with objects that reflect the viewer's personality while staying invisible to the plot.

User Interest: {user_interest}
Show: {show_name}
Vibe: {vibe}
Location: {location}
{correction_block}
Suggest exactly three distinct ambient objects that would appear naturally in this scene. 
These objects should reflect the viewer's personality ({user_interest}) while being historically appropriate for the setting.

Respond with exactly three items, one per line, in this format:
OBJECT: [item name]

Do not include any other text or explanation."""

    # Use Gemma 4 model via settings.google_client
    client = settings.google_client
    response = client.models.generate_content(
        model="gemma-4-26b-a4b-it",
        contents=creative_prompt
    )
    
    gemma_response = response.text
    
    # Parse the response to extract three objects
    proposed_objects = []
    for line in gemma_response.split("\n"):
        line = line.strip()
        if line.startswith("OBJECT:"):
            obj = line.replace("OBJECT:", "").strip()
            if obj:
                proposed_objects.append(obj)
    
    # Ensure we have exactly three objects
    if len(proposed_objects) < 3:
        # Fallback objects if Gemma didn't return enough
        fallback_objects = [
            f"Vintage {user_interest} magazine",
            f"Period-appropriate furniture",
            f"Thematic decor item"
        ]
        while len(proposed_objects) < 3:
            for fb in fallback_objects:
                if len(proposed_objects) >= 3:
                    break
                if fb not in proposed_objects:
                    proposed_objects.append(fb)
    
    # Trim to exactly three
    proposed_objects = proposed_objects[:3]
    
    # Update state
    state["proposed_objects"] = proposed_objects
    
    # Append reasoning to reasoning_log
    reasoning = f"Selected '{proposed_objects[0]}', '{proposed_objects[1]}', and '{proposed_objects[2]}' to bridge viewer demographic ({user_interest}) with {show_name}'s {vibe} aesthetic at {location}"
    state["reasoning_log"].append({"agent": "Creative Director", "action": "Proposal", "message": reasoning})
    state["reasoning_log"].append({"agent": "Creative Director", "action": "LLM Response", "message": f"Gemma 4 response: {gemma_response}"})
    
    return state
