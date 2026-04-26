from typing import TypedDict, List, Optional

class AgentState(TypedDict):
    show_id: str
    product_data: dict
    scene_description: str
    market_insights: List[str]
    optimization_ideas: List[str]
    interaction_log: List[str]
    reasoning_log: List[dict]