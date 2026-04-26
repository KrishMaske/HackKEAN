from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.db.schemas import AgentState
from app.agents.scene_analyst import scene_analyst_agent
from app.agents.market_analyst import market_analyst_agent
from app.agents.ad_strategist import ad_strategist_agent

# ── Persistent in-memory checkpointer ────────────────────────────────────────
memory = MemorySaver()

def create_marketing_workflow():
    workflow = StateGraph(AgentState)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    workflow.add_node("scene_analyst", scene_analyst_agent)
    workflow.add_node("market_analyst", market_analyst_agent)
    workflow.add_node("ad_strategist", ad_strategist_agent)

    # ── Entry Point ───────────────────────────────────────────────────────────
    workflow.set_entry_point("scene_analyst")

    # ── Edges ─────────────────────────────────────────────────────────────────
    workflow.add_edge("scene_analyst",  "market_analyst")
    workflow.add_edge("market_analyst", "ad_strategist")
    workflow.add_edge("ad_strategist",  END)

    return workflow.compile(checkpointer=memory)

marketing_workflow = create_marketing_workflow()

async def execute_marketing_analysis(
    show_id: str,
    product_data: dict,
    scene_description: str,
    thread_id: str = "default",
) -> dict:
    """
    Execute the Marketing Analysis workflow for a given product/scene.
    """
    initial_state: AgentState = {
        "show_id": show_id,
        "product_data": product_data,
        "scene_description": scene_description,
        "market_insights": [],
        "optimization_ideas": [],
        "interaction_log": [],
        "reasoning_log": [],
    }

    config = {"configurable": {"thread_id": thread_id}}

    return await marketing_workflow.ainvoke(initial_state, config=config)

async def run_marketing_workflow(show_id: str):
    """
    Wrapper for the ingestion pipeline to trigger marketing analysis.
    Loads scene data and runs the workflow.
    """
    from app.services.ingestion import load_scene, save_scene
    
    scene_data = load_scene(show_id)
    if not scene_data:
        print(f"[ERROR] Cannot run marketing workflow: {show_id} not found.")
        return

    result = await execute_marketing_analysis(
        show_id=show_id,
        product_data={"product": scene_data.get("target_object"), "summary": scene_data.get("second_by_second_detection", {}).get("summary", {})},
        scene_description=scene_data.get("scene_description", "")
    )
    
    # Merge results back into scene data
    scene_data["marketing"] = {
        "insights": result.get("market_insights", []),
        "optimizations": result.get("optimization_ideas", []),
        "interactions": result.get("interaction_log", [])
    }
    
    save_scene(show_id, scene_data)
    print(f"[ORCHESTRATOR] Marketing Analysis complete for {show_id}.")