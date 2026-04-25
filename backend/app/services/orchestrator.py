from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.db.schemas import AgentState
from app.agents.creative_director import creative_director_agent
from app.agents.historian import historian_agent
from app.agents.cinematographer import cinematographer_agent
from app.agents.foley_artist import foley_artist_agent
from app.db.database import get_historical_context

MAX_RETRIES = 2  # Safety cap — graph will never loop more than twice

# ── Persistent in-memory checkpointer ────────────────────────────────────────
# MemorySaver stores the full state snapshot after every node execution.
# Keyed by thread_id, this lets us replay or resume any user session.
memory = MemorySaver()


def should_continue(state: AgentState) -> str:
    """
    Conditional edge: routes the graph after the Historian node.

    Returns:
        "retry"           → final_selection is None and we still have retry budget
                            → back to Creative Director with a correction note
        "cinematographer" → we have a valid object, proceed to the visual pipeline
    """
    has_selection = bool(state.get("final_selection"))
    retry_count   = state.get("retry_count", 0)

    if not has_selection and retry_count < MAX_RETRIES:
        return "retry"

    # If we've exhausted retries and still have no selection, pick the first
    # proposed object as a graceful fallback so the pipeline never hangs.
    if not has_selection:
        fallback = (state.get("proposed_objects") or ["era-appropriate object"])[0]
        state["final_selection"] = fallback
        state["reasoning_log"].append({
            "agent": "Orchestrator",
            "action": "Fallback",
            "message": f"Max retries ({MAX_RETRIES}) reached. Using fallback: '{fallback}'."
        })

    return "cinematographer"


def create_scene_shift_workflow():
    workflow = StateGraph(AgentState)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    workflow.add_node("creative_director", creative_director_agent)
    workflow.add_node("historian",         historian_agent)
    workflow.add_node("cinematographer",   cinematographer_agent)
    workflow.add_node("foley_artist",      foley_artist_agent)

    # ── Entry Point ───────────────────────────────────────────────────────────
    workflow.set_entry_point("creative_director")

    # ── Edges ─────────────────────────────────────────────────────────────────
    workflow.add_edge("creative_director", "historian")

    # Self-correcting conditional edge: Historian → loop back OR proceed
    workflow.add_conditional_edges(
        "historian",
        should_continue,
        {
            "retry":           "creative_director",  # loop back with correction note
            "cinematographer": "cinematographer",    # proceed to visual pipeline
        }
    )

    workflow.add_edge("cinematographer", "foley_artist")
    workflow.add_edge("foley_artist",    END)

    # ── Compile with checkpointer ─────────────────────────────────────────────
    # Passing `checkpointer=memory` enables thread-scoped state persistence.
    return workflow.compile(checkpointer=memory)


scene_shift_workflow = create_scene_shift_workflow()


async def execute_sceneshift(
    user_interest: str,
    scene_id: str,
    guardrails: bool = True,
    thread_id: str = "default",
) -> dict:
    """
    Execute the SceneShift workflow for a given user/scene.

    Args:
        user_interest: The viewer's personality / interest string (e.g. "Gym Bro").
        scene_id:      The show scene identifier (e.g. "stranger_things_83").
        guardrails:    Whether the Historian's historical validation is active.
        thread_id:     Unique key for this user session. Using the same thread_id
                       across multiple calls lets the graph remember prior context
                       (e.g. switching from Stranger Things → The Office without
                       losing the user's interest profile).
    """
    historical_context = await get_historical_context(scene_id)

    initial_state: AgentState = {
        "user_interest":      user_interest,
        "scene_id":           scene_id,
        "historical_context": historical_context,
        "proposed_objects":   [],
        "final_selection":    None,
        "selected_object":    "",
        "visual_specs":       {},
        "audio_specs":        {},
        "reasoning_log":      [],
        "guardrails_enabled": guardrails,
        "correction_note":    None,
        "retry_count":        0,
    }

    # The config dict is LangGraph's mechanism for selecting the checkpoint thread.
    # Every invocation with the same thread_id shares a persistent state store.
    config = {"configurable": {"thread_id": thread_id}}

    return await scene_shift_workflow.ainvoke(initial_state, config=config)