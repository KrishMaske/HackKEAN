from langgraph.graph import StateGraph, END
from app.db.schemas import AgentState
from app.agents.creative_director import creative_director_agent
from app.agents.historian import historian_agent
from app.agents.cinematographer import cinematographer_agent
from app.agents.foley_artist import foley_artist_agent
from app.db.database import get_historical_context
from app.tools.vision_tools import trigger_visual_generation

def create_scene_shift_workflow():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("creative_director", creative_director_agent)
    workflow.add_node("historian", historian_agent)
    workflow.add_node("cinematographer", cinematographer_agent) # Add Node
    workflow.add_node("foley_artist", foley_artist_agent)
    
    workflow.set_entry_point("creative_director")
    workflow.add_edge("creative_director", "historian")
    workflow.add_edge("historian", "cinematographer") # New Link
    workflow.add_edge("cinematographer", "foley_artist")
    workflow.add_edge("foley_artist", END)
    
    return workflow.compile()

scene_shift_workflow = create_scene_shift_workflow()

async def execute_sceneshift(user_interest: str, scene_id: str, guardrails: bool = True) -> dict:
    historical_context = await get_historical_context(scene_id)
    initial_state: AgentState = {
        "user_interest": user_interest,
        "scene_id": scene_id,
        "historical_context": historical_context,
        "proposed_objects": [],
        "final_selection": None,
        "selected_object": "",
        "visual_specs": {},
        "audio_specs": {},
        "reasoning_log": [],
        "guardrails_enabled": guardrails # Pass the flag
    }
    return await scene_shift_workflow.ainvoke(initial_state)