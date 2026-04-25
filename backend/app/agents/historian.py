from typing import List, Optional
from app.core.config import settings
from app.db.schemas import AgentState

# Task 1.4: Create the Historian Agent Node
def historian_agent(state: AgentState) -> AgentState:
    """
    Narrative Historian Agent - Validates proposed objects for chronological accuracy.
    
    Retrieves year and forbidden_tech from historical_context, uses Gemma 4 to validate
    proposed_objects, and updates final_selection with reasoning.
    """
    historical_context = state.get("historical_context", {})
    proposed_objects = state.get("proposed_objects", [])
    
    # Retrieve year and forbidden_tech from historical_context
    year = historical_context.get("year")
    forbidden_tech = historical_context.get("forbidden_tech", [])
    
    if not year:
        state["reasoning_log"].append({"agent": "Historian", "action": "Error", "message": "No year found in historical context"})
        return state
    
    if not proposed_objects:
        state["reasoning_log"].append({"agent": "Historian", "action": "Error", "message": "No proposed objects to validate"})
        return state
    
    # Task 5.1: Check if guardrails are bypassed
    if not state.get("guardrails_enabled", True):
        final_selection = state["proposed_objects"][0]
        state["final_selection"] = final_selection
        state["reasoning_log"].append({"agent": "Historian", "action": "Bypass", "message": f"GUARDRAILS DISABLED. Letting '{final_selection}' pass through without validation."})
        return state

    # Build the validation prompt for Gemma 4
    validation_prompt = f"""You are a Hollywood Historical Consultant. Cross-reference the proposed object with the year {year}. If it did not exist or is anachronistic, suggest the most culturally relevant substitute from that era.

Proposed objects: {proposed_objects}
Year: {year}
Forbidden technology: {forbidden_tech}

For each proposed object, respond with either:
- "VALID: [object]" if it existed in {year}
- "SUBSTITUTE: [original object] -> [suggested substitute]" if it's anachronistic

Provide your analysis for each object. For any substitutes, explicitly state your reasoning on a new line in this exact format:
THINKING: I thought about [Original], but the scene is {year}, so I chose [Substitute]."""

    # Use Gemma 4 model via settings.google_client
    client = settings.google_client
    response = client.models.generate_content(
        model="gemma-4-26b-a4b-it",
        contents=validation_prompt
    )
    
    gemma_response = response.text
    
    # Parse the response to determine final selection
    valid_objects = []
    substitutes = []
    thinking_logs = []
    
    for line in gemma_response.split("\n"):
        line = line.strip()
        if line.startswith("VALID:"):
            obj = line.replace("VALID:", "").strip()
            valid_objects.append(obj)
        elif line.startswith("SUBSTITUTE:"):
            # Extract the substitute suggestion
            parts = line.replace("SUBSTITUTE:", "").strip()
            if "->" in parts:
                original, substitute = parts.split("->")
                substitutes.append(substitute.strip())
        elif line.startswith("THINKING:"):
            thinking_logs.append(line.replace("THINKING:", "").strip())
    
    # Determine final selection
    # If any validated/substituted objects exist, pick the best one and proceed.
    # If NOTHING was returned (LLM parse failure), trigger a retry loop.
    if valid_objects:
        final_selection = valid_objects[0]
        reasoning = f"'{final_selection}' validated as historically accurate for {year}"
        state["final_selection"] = final_selection
        state["correction_note"] = None  # Clear any previous correction
        state["reasoning_log"].append({"agent": "Historian", "action": "Validation", "message": reasoning})
        state["reasoning_log"].append({"agent": "Historian", "action": "LLM Response", "message": f"Gemma 4 response: {gemma_response}"})
    elif substitutes:
        final_selection = substitutes[0]
        # Sponsor Hook Logic: prioritize the explicit 'THINKING' log if available
        if thinking_logs:
            reasoning = thinking_logs[0]
        else:
            reasoning = f"Original object anachronistic, substituted with '{final_selection}' from {year}"
        state["final_selection"] = final_selection
        state["correction_note"] = None  # Substitution resolved it — no loop needed
        state["reasoning_log"].append({"agent": "Historian", "action": "Substitution", "message": reasoning})
        state["reasoning_log"].append({"agent": "Historian", "action": "LLM Response", "message": f"Gemma 4 response: {gemma_response}"})
    else:
        # Complete parse failure — build a rich correction note and send the graph back
        rejected = ", ".join(f"'{o}'" for o in proposed_objects)
        correction_note = (
            f"Your suggestions ({rejected}) could not be validated for the year {year}. "
            f"The following technologies are strictly forbidden: {forbidden_tech}. "
            f"Please propose three completely different, era-appropriate objects that a '{state.get('user_interest', 'viewer')}' "
            f"would recognise from {year}. Avoid anything invented after {year}."
        )
        state["final_selection"] = None
        state["correction_note"] = correction_note
        state["reasoning_log"].append({
            "agent": "Historian",
            "action": "Correction Required",
            "message": f"Loop-back triggered. Note sent to Creative Director: {correction_note}"
        })

    return state
