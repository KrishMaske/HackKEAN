from langgraph.graph import StateGraph, END
from schemas import AgentState
from agents.creative_director import creative_director_agent
from agents.historian import historian_agent
from agents.cinematographer import cinematographer_agent
from database import get_historical_context
from tools.vision_tools import trigger_visual_generation

def create_scene_shift_workflow():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("creative_director", creative_director_agent)
    workflow.add_node("historian", historian_agent)
    workflow.add_node("cinematographer", cinematographer_agent) # Add Node
    
    workflow.set_entry_point("creative_director")
    workflow.add_edge("creative_director", "historian")
    workflow.add_edge("historian", "cinematographer") # New Link
    workflow.add_edge("cinematographer", END) # End after Cinematographer
    
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
        "reasoning_log": [],
        "guardrails_enabled": guardrails # Pass the flag
    }
    return await scene_shift_workflow.ainvoke(initial_state)