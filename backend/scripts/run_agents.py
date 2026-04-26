#!/usr/bin/env python3
"""
Offline agent reasoning script for SceneShift.

Pre-runs the LangGraph multi-agent pipeline for all (show_id × user_interest)
combinations and caches the results to JSON. This means the frontend can
display agent reasoning instantly with zero API calls.

Usage:
    python -m scripts.run_agents
    python -m scripts.run_agents --show_id stranger_things_83 --persona "Gym Bro"
"""

import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = "app/db/data"
os.makedirs(DATA_DIR, exist_ok=True)

# The full demo matrix: all shows × all personas
DEMO_SHOWS = [
    "stranger_things_83",
    "the_office_05",
    "succession_20",
]

DEMO_PERSONAS = [
    "Gym Bro",
    "Foodie",
    "Tech Nerd",
]


async def run_single(show_id: str, persona: str, thread_id: str) -> dict:
    """Run the agent pipeline for a single (show, persona) pair."""
    from app.services.orchestrator import execute_sceneshift

    print(f"  🤖 Running agents for {show_id} × {persona}...")

    try:
        result = await execute_sceneshift(
            user_interest=persona,
            scene_id=show_id,
            guardrails=True,
            thread_id=thread_id,
        )

        print(f"     ✅ Result: {result.get('final_selection', 'unknown')}")
        return {
            "show_id": show_id,
            "persona": persona,
            "success": True,
            "final_selection": result.get("final_selection"),
            "reasoning_log": result.get("reasoning_log", []),
            "visual_specs": result.get("visual_specs", {}),
            "audio_specs": result.get("audio_specs", {}),
        }

    except Exception as e:
        print(f"     ❌ Failed: {e}")
        return {
            "show_id": show_id,
            "persona": persona,
            "success": False,
            "error": str(e),
        }


async def run_all(shows: list, personas: list):
    """Run all (show × persona) combinations sequentially."""
    all_results = []
    total = len(shows) * len(personas)

    print(f"\n  Running {total} agent pipelines ({len(shows)} shows × {len(personas)} personas)\n")

    for show_id in shows:
        for persona in personas:
            thread_id = f"precompute_{show_id}_{persona.lower().replace(' ', '_')}"
            result = await run_single(show_id, persona, thread_id)
            all_results.append(result)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="SceneShift Offline Agent Runner")
    parser.add_argument("--show_id", help="Run for a single show only")
    parser.add_argument("--persona", help="Run for a single persona only")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  SceneShift Offline Agent Pipeline")
    print(f"{'='*60}\n")

    shows = [args.show_id] if args.show_id else DEMO_SHOWS
    personas = [args.persona] if args.persona else DEMO_PERSONAS

    # Run the async pipeline
    results = asyncio.run(run_all(shows, personas))

    # Save results
    output_path = os.path.join(DATA_DIR, "agent_cache.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    succeeded = sum(1 for r in results if r.get("success"))
    failed = sum(1 for r in results if not r.get("success"))

    print(f"\n{'='*60}")
    print(f"  ✅ {succeeded} succeeded, ❌ {failed} failed")
    print(f"  Saved to {output_path}")
    print(f"{'='*60}\n")

    if failed > 0:
        print("  Failed combinations:")
        for r in results:
            if not r.get("success"):
                print(f"    - {r['show_id']} × {r['persona']}: {r.get('error', 'unknown')}")
        print()


if __name__ == "__main__":
    main()
