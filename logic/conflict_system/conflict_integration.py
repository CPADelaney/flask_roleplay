# logic/conflict_system/conflict_integration.py

"""
Story Routes Integration for Conflict System

This module provides the necessary adjustments to integrate the conflict system
with the existing story_routes.py file.
"""

import logging
from typing import Dict, Any, List, Optional

from logic.conflict_system.conflict_integration import ConflictSystemIntegration

logger = logging.getLogger(__name__)

async def enrich_storybeat_with_conflicts(
    user_id: int, 
    conversation_id: int, 
    response_data: Dict[str, Any],
    user_input: str
) -> Dict[str, Any]:
    """
    Enrich the next_storybeat response with conflict information.
    
    Args:
        user_id: The user ID
        conversation_id: The conversation ID
        response_data: The response data to enrich
        user_input: The original user input text
        
    Returns:
        The enriched response data
    """
    try:
        # Create conflict integration instance
        conflict_integration = ConflictSystemIntegration(user_id, conversation_id)
        
        # Get active conflicts
        active_conflicts = await conflict_integration.get_active_conflicts()
        
        # Add active conflicts summary to response
        response_data["conflict_system"] = {
            "active_conflicts": len(active_conflicts),
            "has_major_conflict": any(c.get("conflict_type") == "major" for c in active_conflicts),
            "has_minor_conflict": any(c.get("conflict_type") == "minor" for c in active_conflicts)
        }
        
        # Add brief summaries of the top 3 active conflicts
        if active_conflicts:
            top_conflicts = active_conflicts[:3]
            conflict_summaries = []
            for conflict in top_conflicts:
                conflict_id = conflict.get("conflict_id")
                if conflict_id:
                    # Get full conflict details
                    conflict_details = await conflict_integration.get_conflict_details(conflict_id)
                    if conflict_details:
                        # Create a summary
                        summary = {
                            "conflict_id": conflict_id,
                            "conflict_name": conflict_details.get("conflict_name"),
                            "conflict_type": conflict_details.get("conflict_type"),
                            "phase": conflict_details.get("phase"),
                            "progress": conflict_details.get("progress"),
                            "factions": {
                                "a": conflict_details.get("faction_a_name"),
                                "b": conflict_details.get("faction_b_name")
                            }
                        }
                        conflict_summaries.append(summary)
            
            response_data["conflict_system"]["top_conflicts"] = conflict_summaries
        
        # Get player vitals
        vitals = await conflict_integration.get_player_vitals()
        if not isinstance(vitals, dict) or "error" in vitals:
            vitals = {"energy": 100, "hunger": 100}
        
        response_data["player_vitals"] = {
            "energy": vitals.get("energy", 100),
            "hunger": vitals.get("hunger", 100)
        }
        
        # Check if user input might trigger a conflict
        if active_conflicts and len(active_conflicts) < 3:  # Only consider adding if we have room
            conflict_result = await conflict_integration.add_conflict_to_narrative(user_input)
            if conflict_result.get("conflict_generated", False):
                response_data["conflict_system"]["new_conflict"] = {
                    "message": conflict_result.get("message"),
                    "conflict": conflict_result.get("conflict")
                }
        
        return response_data
    except Exception as e:
        logger.error(f"Error enriching storybeat with conflicts: {e}")
        # Return original data if there's an error
        return response_data

# Add this to story_bp.route("/next_storybeat") after the AI response is generated
# and before the final response is returned
async def process_conflicts_in_storybeat(
    user_id: int,
    conversation_id: int,
    user_input: str,
    ai_response: str
) -> Dict[str, Any]:
    """
    Process conflicts in the storybeat based on the AI response and user input.
    
    Args:
        user_id: The user ID
        conversation_id: The conversation ID
        user_input: The user's input text
        ai_response: The AI's response text
        
    Returns:
        A dict with conflict-related results
    """
    try:
        # Create conflict integration instance
        conflict_integration = ConflictSystemIntegration(user_id, conversation_id)
        
        # Check for conflict-related keywords in AI response
        response_lower = ai_response.lower()
        conflict_keywords = {
            "brewing": 5,
            "tension": 5,
            "argument": 10,
            "disagreement": 10,
            "challenge": 15,
            "conflict": 20,
            "fight": 15,
            "battle": 15,
            "confrontation": 20,
            "opposition": 10,
            "struggle": 10,
            "strife": 15,
            "crisis": 20,
            "dilemma": 15,
            "problem": 5,
            "obstacle": 5,
            "hurdle": 5,
            "difficulty": 5,
            "complication": 10,
            "sabotage": 20,
            "betrayal": 25,
            "conspiracy": 25
        }
        
        # Calculate conflict intensity based on keywords
        conflict_intensity = 0
        for keyword, weight in conflict_keywords.items():
            if keyword in response_lower:
                conflict_intensity += weight
                
        # Normalize conflict intensity (0-100)
        conflict_intensity = min(100, conflict_intensity)
        
        # Get active conflicts
        active_conflicts = await conflict_integration.get_active_conflicts()
        
        results = {
            "conflict_intensity": conflict_intensity,
            "active_conflicts": len(active_conflicts)
        }
        
        # If there are active conflicts, update their progress based on intensity
        if active_conflicts:
            for conflict in active_conflicts:
                conflict_id = conflict.get("conflict_id")
                if conflict_id:
                    # Calculate progress increment based on intensity and conflict type
                    progress_increment = 0
                    
                    if conflict.get("conflict_type") == "major":
                        # Major conflicts progress more slowly
                        progress_increment = conflict_intensity / 20
                    elif conflict.get("conflict_type") == "minor":
                        # Minor conflicts progress more quickly
                        progress_increment = conflict_intensity / 10
                    elif conflict.get("conflict_type") == "standard":
                        # Standard conflicts progress at medium pace
                        progress_increment = conflict_intensity / 15
                    elif conflict.get("conflict_type") == "catastrophic":
                        # Catastrophic conflicts progress very slowly
                        progress_increment = conflict_intensity / 25
                    
                    # Only update if there's a meaningful increment
                    if progress_increment >= 1:
                        await conflict_integration.update_progress(conflict_id, progress_increment)
        
        # Generate a new conflict if intensity is high enough and we don't have too many
        if conflict_intensity >= 50 and len(active_conflicts) < 3:
            # Determine conflict type based on intensity
            conflict_type = "standard"
            if conflict_intensity >= 90:
                conflict_type = "catastrophic"
            elif conflict_intensity >= 70:
                conflict_type = "major"
            elif conflict_intensity >= 50:
                conflict_type = "minor"
            
            # 30% chance to generate a conflict when conditions are met
            import random
            if random.random() < 0.3:
                new_conflict = await conflict_integration.generate_new_conflict(conflict_type)
                if not isinstance(new_conflict, dict) or "error" not in new_conflict:
                    results["new_conflict"] = {
                        "message": f"New {conflict_type} conflict generated",
                        "conflict": new_conflict
                    }
        
        # Update player vitals based on conflict intensity
        activity_type = "standard"
        if conflict_intensity >= 70:
            activity_type = "intense"
        elif conflict_intensity <= 20:
            activity_type = "restful"
        
        await conflict_integration.update_player_vitals(activity_type)
        
        return results
    except Exception as e:
        logger.error(f"Error processing conflicts in storybeat: {e}")
        return {"error": str(e)}

# Add this function to update the next_storybeat route
def update_next_storybeat_function():
    """
    This function contains the code changes needed in the next_storybeat route
    to integrate with the conflict system.
    
    Add this code at the appropriate places in the next_storybeat route.
    """
    # After getting the AI response but before returning the final JSON
    # Add these lines:
    
    """
    # Process conflicts based on AI response
    tracker.start_phase("process_conflicts")
    conflict_results = await process_conflicts_in_storybeat(
        user_id, conv_id, user_input, final_response
    )
    tracker.end_phase()
    
    # Enrich response with conflict information
    tracker.start_phase("enrich_conflicts")
    response = await enrich_storybeat_with_conflicts(
        user_id, conv_id, response, user_input
    )
    tracker.end_phase()
    
    # Add conflict results to response
    if "conflict_system" not in response:
        response["conflict_system"] = {}
    response["conflict_system"].update(conflict_results)
    """
