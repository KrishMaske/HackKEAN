from app.db.schemas import AgentState
from app.core.config import settings

async def ad_strategist_agent(state: AgentState) -> AgentState:
    """
    Generates optimization ideas for better product placement.
    """
    print("[AGENT] Ad Strategist brainstorming optimizations...")
    
    product_data = state.get("product_data", {})
    scene_description = state.get("scene_description", "Unknown scene context.")
    
    client = settings.groq_client
    prompt = f"""You are a brilliant Ad Strategist. 
Based on the product "{product_data.get('product')}" and this scene description:
"{scene_description}"

Generate 3 creative 'Optimization Ideas' to increase the product's business impact. 
Think about:
1. Better placement within the scene.
2. Contextual alignment with character actions.
3. Interactive ad triggers.

Respond with ONLY valid JSON:
{{"optimizations": ["idea 1", "idea 2", "idea 3"]}}
"""
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        import json
        data = json.loads(response.choices[0].message.content)
        state["optimization_ideas"] = data.get("optimizations", [])
        
        state["reasoning_log"].append({
            "agent": "Ad Strategist",
            "action": "Optimization Generation",
            "message": f"Generated {len(state['optimization_ideas'])} optimization ideas based on scene context."
        })
    except Exception as e:
        print(f"[WARN] Ad Strategist failed: {e}")
        state["optimization_ideas"] = ["Could not generate optimization ideas."]

    return state
