from typing import Dict, List
from app.db.schemas import AgentState
from app.core.config import settings

async def market_analyst_agent(state: AgentState) -> AgentState:
    """
    Analyzes product exposure and calculates estimated marketing value.
    """
    print("[AGENT] Market Analyst analyzing exposure...")
    
    product_data = state.get("product_data", {})
    summary = product_data.get("summary", {})
    
    visibility_rate = summary.get("visibility_rate", 0)
    detected_seconds = summary.get("detected_seconds", 0)
    max_coverage = summary.get("max_screen_coverage", 0)
    
    # CPM Calculation (Simplified)
    # Assume $20 CPM for prime placement
    # Visibility and coverage boost the value
    base_value = detected_seconds * 0.5 # $0.50 per second of exposure
    coverage_multiplier = 1.0 + (max_coverage * 5.0)
    estimated_value = base_value * coverage_multiplier
    
    client = settings.groq_client
    prompt = f"""You are a senior Market Analyst. Analyze the following product exposure data:
Product: {product_data.get('product')}
Detected Seconds: {detected_seconds}s
Visibility Rate: {visibility_rate * 100:.1f}%
Peak Screen Coverage: {max_coverage * 100:.1f}%

Calculate the 'Market Impact Score' (0-100) and provide 3 key insights about the product's market value in this scene.

Respond with ONLY valid JSON:
{{"impact_score": int, "insights": ["insight 1", "insight 2", "insight 3"]}}
"""
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        import json
        data = json.loads(response.choices[0].message.content)
        
        impact_score = data.get("impact_score", 0)
        insights = data.get("insights", [])
        
        state["market_insights"] = [
            f"Market Impact Score: {impact_score}/100",
            f"Estimated Media Value: ${estimated_value:.2f}",
            *insights
        ]
        
        state["reasoning_log"].append({
            "agent": "Market Analyst",
            "action": "Exposure Analysis",
            "message": f"Calculated market value of ${estimated_value:.2f} with impact score {impact_score}."
        })
    except Exception as e:
        print(f"[WARN] Market Analyst failed: {e}")
        state["market_insights"] = [f"Exposure Analysis failed: {str(e)}"]

    return state
