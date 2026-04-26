from app.db.schemas import AgentState
from app.core.config import settings

async def scene_analyst_agent(state: AgentState) -> AgentState:
    """
    Analyzes scene context and character-product interactions.
    """
    print("[AGENT] Scene Analyst examining interactions...")
    
    product_data = state.get("product_data", {})
    scene_description = state.get("scene_description", "No visual description available.")
    
    client = settings.groq_client
    prompt = f"""You are a Scene Analyst specializing in product-character interaction.
Scene Context: {scene_description}
Tracked Product: {product_data.get('product')}

Analyze the scene and describe:
1. What exactly is going on in the scene?
2. How are the characters interacting with or reacting to the product?
3. What is the emotional tone of the scene?

Respond with ONLY valid JSON:
{{"scene_summary": "string", "interactions": ["interaction 1", "interaction 2"], "tone": "string"}}
"""
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        import json
        data = json.loads(response.choices[0].message.content)
        
        summary = data.get("scene_summary", "")
        interactions = data.get("interactions", [])
        tone = data.get("tone", "")
        
        state["interaction_log"] = [
            f"Scene: {summary}",
            f"Tone: {tone}",
            *interactions
        ]
        
        state["reasoning_log"].append({
            "agent": "Scene Analyst",
            "action": "Interaction Analysis",
            "message": f"Identified {len(interactions)} character-product interactions."
        })
    except Exception as e:
        print(f"[WARN] Scene Analyst failed: {e}")
        state["interaction_log"] = ["Interaction analysis unavailable."]

    return state
