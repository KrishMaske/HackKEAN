# backend/schemas.py
from typing import TypedDict, List, Optional

class AgentState(TypedDict):
    user_interest: str
    scene_id: str
    historical_context: dict
    proposed_objects: List[str]
    final_selection: Optional[str]
    selected_object: str          # Added to match orchestrator initialization
    visual_specs: dict
    reasoning_log: List[str]
    guardrails_enabled: bool      # Added for Task 5.1