# story_agent/conflict_narrative_api.py

"""
API Routes for the Story Director Agent

This module defines the Flask routes for integrating the Story Director Agent
with your existing game API.
"""

import logging
import json
from flask import Blueprint, request, jsonify, session
from utils.performance import timed_function

from story_director_agent import (
    initialize_story_director,
    get_current_story_state,
    process_narrative_input,
    advance_story
)

story_director_bp = Blueprint("story_director_bp", __name__)
logger = logging.getLogger(__name__)

@story_director_bp.route("/story/state", methods=["GET"])
@timed_function(name="get_story_state")
async def get_story_state():
    """
    Get the current state of the story, including narrative stage, 
    active conflicts, and potential narrative events.
    """
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        
        conversation_id = request.args.get("conversation_id")
        if not conversation_id:
            return jsonify({"error": "Missing conversation_id parameter"}), 400
        
        # Initialize the Story Director Agent
        agent, context = await initialize_story_director(
            user_id, int(conversation_id)
        )
        
        # Get current story state
        result = await get_current_story_state(agent, context)
        
        # Extract relevant information from the result
        response_text = result.final_output
        
        # Look for any structured data in the response
        state_data = {}
        try:
            # Try to extract JSON if present
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                state_data = json.loads(json_str)
        except:
            # If JSON extraction fails, use the full text
            state_data = {"state_description": response_text}
        
        return jsonify({
            "story_state": state_data,
            "narrative_analysis": response_text
        })
    
    except Exception as e:
        logger.exception("Error getting story state")
        return jsonify({"error": str(e)}), 500

@story_director_bp.route("/story/process", methods=["POST"])
@timed_function(name="process_narrative")
async def process_narrative():
    """
    Process narrative text to determine if it should trigger conflicts or narrative events.
    """
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        
        data = request.get_json() or {}
        conversation_id = data.get("conversation_id")
        if not conversation_id:
            return jsonify({"error": "Missing conversation_id parameter"}), 400
        
        narrative_text = data.get("narrative_text")
        if not narrative_text:
            return jsonify({"error": "Missing narrative_text parameter"}), 400
        
        # Initialize the Story Director Agent
        agent, context = await initialize_story_director(
            user_id, int(conversation_id)
        )
        
        # Process narrative input
        result = await process_narrative_input(agent, context, narrative_text)
        
        # Extract relevant information from the result
        response_text = result.final_output
        
        # Try to extract any structured data about generated conflicts
        conflict_data = {}
        try:
            # Look for conflict/event info in the response
            if "conflict generated" in response_text.lower():
                # Try to find any JSON data
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    conflict_data = json.loads(json_str)
        except:
            # If JSON extraction fails, continue with empty data
            pass
        
        return jsonify({
            "analysis": response_text,
            "conflict_generated": "conflict generated" in response_text.lower(),
            "narrative_event_generated": "narrative event" in response_text.lower(),
            "conflict_data": conflict_data
        })
    
    except Exception as e:
        logger.exception("Error processing narrative")
        return jsonify({"error": str(e)}), 500

@story_director_bp.route("/story/advance", methods=["POST"])
@timed_function(name="advance_story")
async def api_advance_story():
    """
    Advance the story based on player actions.
    """
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        
        data = request.get_json() or {}
        conversation_id = data.get("conversation_id")
        if not conversation_id:
            return jsonify({"error": "Missing conversation_id parameter"}), 400
        
        player_actions = data.get("player_actions")
        if not player_actions:
            return jsonify({"error": "Missing player_actions parameter"}), 400
        
        # Initialize the Story Director Agent
        agent, context = await initialize_story_director(
            user_id, int(conversation_id)
        )
        
        # Advance the story
        result = await advance_story(agent, context, player_actions)
        
        # Extract relevant information from the result
        response_text = result.final_output
        
        # Try to extract structured data about story advancement
        advancement_data = {}
        try:
            # Look for JSON data in the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                advancement_data = json.loads(json_str)
        except:
            # If JSON extraction fails, continue with empty data
            pass
        
        # Determine what occurred during story advancement
        conflicts_advanced = "conflict" in response_text.lower() and (
            "progress" in response_text.lower() or 
            "advance" in response_text.lower() or
            "update" in response_text.lower()
        )
        
        conflicts_resolved = "conflict" in response_text.lower() and (
            "resolve" in response_text.lower() or
            "resolution" in response_text.lower() or
            "concluded" in response_text.lower()
        )
        
        narrative_events = (
            "revelation" in response_text.lower() or
            "narrative moment" in response_text.lower() or
            "dream sequence" in response_text.lower() or
            "moment of clarity" in response_text.lower()
        )
        
        return jsonify({
            "story_advancement": response_text,
            "conflicts_advanced": conflicts_advanced,
            "conflicts_resolved": conflicts_resolved,
            "narrative_events": narrative_events,
            "advancement_data": advancement_data
        })
    
    except Exception as e:
        logger.exception("Error advancing story")
        return jsonify({"error": str(e)}), 500

@story_director_bp.route("/story/generate-conflict", methods=["POST"])
@timed_function(name="generate_conflict")
async def generate_conflict_api():
    """
    Generate a new conflict via the Story Director Agent.
    """
    try:
        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"error": "Not logged in"}), 401
        
        data = request.get_json() or {}
        conversation_id = data.get("conversation_id")
        if not conversation_id:
            return jsonify({"error": "Missing conversation_id parameter"}), 400
        
        conflict_type = data.get("conflict_type")  # Optional
        
        # Initialize the Story Director Agent
        agent, context = await initialize_story_director(
            user_id, int(conversation_id)
        )
        
        # Craft a prompt to generate a conflict
        prompt = f"Generate a new {conflict_type if conflict_type else ''} conflict based on the current narrative stage and story state."
        
        # Run the agent with this prompt
        result = await Runner.run(agent, prompt, context=context)
        
        # Extract relevant information from the result
        response_text = result.final_output
        
        # Try to extract any structured data about the generated conflict
        conflict_data = {}
        try:
            # Look for conflict info in the response
            if "conflict generated" in response_text.lower():
                # Try to find any JSON data
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    conflict_data = json.loads(json_str)
        except:
            # If JSON extraction fails, continue with empty data
            pass
        
        return jsonify({
            "conflict_generation": response_text,
            "conflict_generated": "conflict generated" in response_text.lower(),
            "conflict_data": conflict_data
        })
    
    except Exception as e:
        logger.exception("Error generating conflict")
        return jsonify({"error": str(e)}), 500
