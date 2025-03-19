# nyx/scene_manager_sdk.py

import logging
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import asyncpg

from agents import Agent, handoff, function_tool, Runner
from agents import ModelSettings, RunConfig
from pydantic import BaseModel, Field

from db.connection import get_db_connection
from utils.caching import NPC_CACHE

logger = logging.getLogger(__name__)

# ===== Pydantic Models for Structured Outputs =====

class NPCResponse(BaseModel):
    """Structured output for NPC responses"""
    dialogue: str = Field(..., description="NPC's dialogue")
    emotions: List[str] = Field(default_factory=list, description="Emotions displayed by the NPC")
    actions: List[str] = Field(default_factory=list, description="Physical actions taken by the NPC")
    internal_thoughts: Optional[str] = Field(None, description="NPC's internal thoughts (not shared with player)")

class SceneUpdate(BaseModel):
    """Structured output for scene updates"""
    narrative: str = Field(..., description="The main narrative description")
    environment_changes: Optional[Dict[str, Any]] = Field(None, description="Changes to the environment")
    npc_positions: Optional[Dict[str, str]] = Field(None, description="Positions of NPCs in the scene")
    mood: str = Field("neutral", description="Overall mood of the scene")
    tension_level: int = Field(0, description="Scene tension level (0-10)") 
    suggested_soundtrack: Optional[str] = Field(None, description="Suggested soundtrack/music for the scene")

class SceneTrigger(BaseModel):
    """Structured output for identifying scene triggers"""
    should_trigger: bool = Field(..., description="Whether a scene should be triggered")
    scene_type: Optional[str] = Field(None, description="Type of scene to trigger")
    trigger_reason: Optional[str] = Field(None, description="Reason for triggering the scene")
    suggested_npcs: List[str] = Field(default_factory=list, description="NPCs that should be involved")
    suggested_location: Optional[str] = Field(None, description="Suggested location for the scene")

# ===== Function Tools =====

@function_tool
async def spawn_npc(
    ctx,
    environment_description: str,
    archetype: Optional[str] = None,
    relationship_to_player: Optional[str] = None
) -> str:
    """
    Spawn a new NPC in the current environment using the comprehensive NPCCreationHandler.
    
    Args:
        environment_description: Description of the current environment
        archetype: Optional archetype for the NPC (e.g., "mentor", "rival", "servant")
        relationship_to_player: Optional relationship to player
    """
    # Import here to avoid circular imports
    from npcs.new_npc_creation import NPCCreationHandler
    
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    
    # Create NPC creation handler
    npc_handler = NPCCreationHandler()
    
    # Convert single archetype to list if provided
    archetype_names = [archetype] if archetype else None
    
    # Define specific traits if relationship specified
    specific_traits = {}
    if relationship_to_player:
        specific_traits["relationship_to_player"] = relationship_to_player
    
    try:
        # Create the NPC using the comprehensive handler
        npc_result = await npc_handler.create_npc_with_context(
            environment_desc=environment_description,
            archetype_names=archetype_names,
            specific_traits=specific_traits if specific_traits else None,
            user_id=user_id,
            conversation_id=conversation_id
        )
        
        # Format the NPC data in a way that's compatible with existing code
        npc_data = {
            "npc_id": npc_result.npc_id,
            "name": npc_result.npc_name,
            "archetype": ', '.join(npc_result.archetypes.archetype_names) if npc_result.archetypes.archetype_names else "Generic",
            "physical_description": npc_result.physical_description,
            "personality": {
                "traits": npc_result.personality.personality_traits,
                "likes": npc_result.personality.likes,
                "dislikes": npc_result.personality.dislikes,
                "hobbies": npc_result.personality.hobbies
            },
            "stats": {
                "dominance": npc_result.stats.dominance,
                "cruelty": npc_result.stats.cruelty,
                "closeness": npc_result.stats.closeness,
                "trust": npc_result.stats.trust,
                "respect": npc_result.stats.respect,
                "intensity": npc_result.stats.intensity
            },
            "current_location": npc_result.current_location
        }
        
        # Cache the NPC data for faster access
        cache_key = f"npc:{user_id}:{conversation_id}:{npc_result.npc_name}"
        NPC_CACHE.set(cache_key, npc_data, 300)  # 5 minute TTL
        
        return f"NPC created: {npc_data['name']} - {npc_data['archetype']}"
    except Exception as e:
        logger.error(f"Error spawning NPC: {e}")
        return f"Failed to create NPC: {str(e)}"

@function_tool
async def update_npc_state(
    ctx,
    npc_name: str,
    current_emotion: str,
    relationship_value: Optional[int] = None,
    loyalty_value: Optional[int] = None
) -> str:
    """
    Update the state of an NPC.
    
    Args:
        npc_name: Name of the NPC
        current_emotion: Current emotional state of the NPC
        relationship_value: Optional relationship value adjustment (-100 to 100)
        loyalty_value: Optional loyalty value adjustment (0 to 100)
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    
    async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
        async with pool.acquire() as conn:
            updates = []
            params = [user_id, conversation_id, npc_name]
            
            updates.append("current_emotion = $4")
            params.append(current_emotion)
            
            query_base = "UPDATE NPCStats SET "
            
            if relationship_value is not None:
                updates.append("relationship_to_player = relationship_to_player + $5")
                params.append(relationship_value)
                
            if loyalty_value is not None:
                updates.append("loyalty_value = loyalty_value + $6")
                params.append(loyalty_value)
                
            query = query_base + ", ".join(updates) + " WHERE user_id = $1 AND conversation_id = $2 AND npc_name = $3"
            await conn.execute(query, *params)
            
            # Invalidate cache
            cache_key = f"npc:{user_id}:{conversation_id}:{npc_name}"
            NPC_CACHE.remove(cache_key)
    
    return f"Updated NPC {npc_name}: emotion={current_emotion}, relationship adjustment={relationship_value}, loyalty adjustment={loyalty_value}"

# ===== Scene Manager Agents =====

class SceneContext:
    """Context object for scene manager agents"""
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.scene_data = {}

# NPC Agent - for individual NPC responses
npc_agent = Agent[SceneContext](
    name="NPC Agent",
    instructions="""You roleplay as an NPC in a femdom-themed game.
    
Given information about the NPC (personality, history, etc.) and the current context,
generate an authentic response that:
1. Maintains character consistency in dialogue and actions
2. Reflects the NPC's relationship with the player
3. Expresses appropriate emotions based on the situation
4. Takes believable actions within the scene
5. Has realistic internal thoughts that inform behavior

Keep responses concise but characteristic, showing the NPC's unique personality.""",
    output_type=NPCResponse
)

# Scene Description Agent
scene_description_agent = Agent[SceneContext](
    name="Scene Description Agent",
    instructions="""You create vivid, atmospheric scene descriptions for a femdom roleplay game.
    
Your descriptions should:
1. Establish the mood and atmosphere of the location
2. Highlight important environmental details that could affect interactions
3. Note the positions and activities of NPCs present
4. Subtly incorporate sensory details (sights, sounds, smells, etc.)
5. Maintain a tone appropriate for adult themes without being explicitly sexual

Keep descriptions concise but evocative, focusing on elements that enhance the narrative.""",
    output_type=SceneUpdate,
    tools=[
        get_location_description,
        get_npcs_in_scene
    ]
)

# Scene Trigger Agent - determines when scene transitions should occur
scene_trigger_agent = Agent[SceneContext](
    name="Scene Trigger Agent",
    instructions="""You analyze game state to determine when narrative scenes should trigger.
    
Your role is to:
1. Identify appropriate moments for scene transitions based on context
2. Suggest scene types that would enhance the narrative flow
3. Recommend which NPCs should be involved based on relationships
4. Consider the pacing of the overall experience
5. Ensure triggered scenes align with the femdom themes of the game

Be selective about triggering scenes - they should feel natural and meaningful.""",
    output_type=SceneTrigger
)

# Scene Manager Agent - main orchestrator
scene_manager_agent = Agent[SceneContext](
    name="Scene Manager",
    instructions="""You manage the overall scene flow in a femdom roleplay game.
    
Your responsibilities are to:
1. Coordinate interactions between the player and NPCs
2. Determine when to update scene descriptions
3. Manage scene transitions based on player actions
4. Ensure narrative coherence and emotional impact
5. Balance between player agency and guided storytelling

Use your specialized sub-agents to handle specific aspects of scene management.""",
    handoffs=[
        handoff(npc_agent, tool_name_override="generate_npc_response"),
        handoff(scene_description_agent, tool_name_override="generate_scene_description"),
        handoff(scene_trigger_agent, tool_name_override="check_for_scene_trigger")
    ],
    tools=[
        spawn_npc,
        get_npc_info,
        update_npc_state
    ],
    model_settings=ModelSettings(
        temperature=0.7
    )
)

# ===== Main Functions =====

async def process_scene_input(
    user_id: int,
    conversation_id: int,
    user_input: str,
    context_data: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Process user input through the scene manager system
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        user_input: User's input text
        context_data: Additional context data
        
    Returns:
        Complete scene update with narrative and metadata
    """
    # Create scene context
    scene_context = SceneContext(user_id, conversation_id)
    
    # Add any additional context
    if context_data:
        scene_context.scene_data = context_data
    
    # Run the scene manager agent
    result = await Runner.run(
        scene_manager_agent,
        user_input,
        context=scene_context,
        run_config=RunConfig(
            workflow_name="Scene Management",
            trace_id=f"scene-{conversation_id}-{int(datetime.now().timestamp())}",
            group_id=f"user-{user_id}"
        )
    )
    
    # Process all the new items generated
    scene_result = {}
    
    for item in result.new_items:
        if item.type == "message_output_item":
            # Main scene manager response
            scene_result["narrative"] = item.raw_item.content
        elif item.type == "handoff_output_item":
            # Responses from sub-agents
            if "generate_npc_response" in str(item.raw_item):
                npc_response = json.loads(item.raw_item.content)
                scene_result["npc_response"] = npc_response
            elif "generate_scene_description" in str(item.raw_item):
                scene_update = json.loads(item.raw_item.content)
                scene_result["scene_update"] = scene_update
            elif "check_for_scene_trigger" in str(item.raw_item):
                scene_trigger = json.loads(item.raw_item.content)
                scene_result["scene_trigger"] = scene_trigger
    
    return scene_result

async def generate_npc_response(
    user_id: int,
    conversation_id: int,
    npc_name: str,
    user_input: str,
    context_data: Dict[str, Any] = None
) -> NPCResponse:
    """
    Generate a response from a specific NPC
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        npc_name: Name of the NPC
        user_input: User's input directed at the NPC
        context_data: Additional context data
        
    Returns:
        NPC's response
    """
    # Create scene context
    scene_context = SceneContext(user_id, conversation_id)
    
    # Add any additional context
    if context_data:
        scene_context.scene_data = context_data
    
    # Get NPC information
    npc_info = await get_npc_info(scene_context, npc_name)
    
    # Create prompt for the NPC agent
    prompt = f"""
As {npc_name}, respond to the following from the player:

"{user_input}"

NPC Information:
{npc_info}

Current context:
{json.dumps(context_data)}
"""
    
    # Run the NPC agent
    result = await Runner.run(
        npc_agent,
        prompt,
        context=scene_context
    )
    
    # Get structured output
    npc_response = result.final_output_as(NPCResponse)
    
    return npc_response

async def check_for_scene_trigger(
    user_id: int,
    conversation_id: int,
    recent_events: List[Dict[str, Any]],
    context_data: Dict[str, Any] = None
) -> SceneTrigger:
    """
    Check if a scene should be triggered based on recent events
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
        recent_events: List of recent events in the game
        context_data: Additional context data
        
    Returns:
        Scene trigger information
    """
    # Create scene context
    scene_context = SceneContext(user_id, conversation_id)
    
    # Add any additional context
    if context_data:
        scene_context.scene_data = context_data
    
    # Create prompt for the scene trigger agent
    prompt = f"""
Analyze these recent events and determine if a scene should be triggered:

{json.dumps(recent_events)}

Current context:
{json.dumps(context_data)}
"""
    
    # Run the scene trigger agent
    result = await Runner.run(
        scene_trigger_agent,
        prompt,
        context=scene_context
    )
    
    # Get structured output
    scene_trigger = result.final_output_as(SceneTrigger)
    
    return scene_trigger

    # Create NPC with your existing function
    npc_data = await spawn_npc(
        user_id, 
        conversation_id, 
        environment_description, 
        archetype=archetype,
        relationship=relationship_to_player
    )
    
    if npc_data:
        return f"NPC created: {npc_data['name']} - {npc_data['archetype']}"
    else:
        return "Failed to create NPC"

@function_tool
async def get_npc_info(ctx, npc_name: str) -> str:
    """
    Get information about an NPC.
    
    Args:
        npc_name: Name of the NPC
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    
    # Check cache first
    cache_key = f"npc:{user_id}:{conversation_id}:{npc_name}"
    cached_npc = NPC_CACHE.get(cache_key)
    
    if cached_npc:
        return json.dumps(cached_npc)
    
    # Fetch from database
    async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
        async with pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM NPCStats 
                WHERE user_id = $1 AND conversation_id = $2 AND npc_name = $3
            """, user_id, conversation_id, npc_name)
    
    if row:
        npc_data = dict(row)
        # Cache for future use
        NPC_CACHE.set(cache_key, npc_data, 300)  # 5 minute TTL
        return json.dumps(npc_data)
    else:
        return f"NPC {npc_name} not found"

@function_tool
async def get_npcs_in_scene(ctx) -> str:
    """
    Get a list of NPCs in the current scene.
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    
    async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
        async with pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT npc_name, archetype, current_location, relationship_to_player 
                FROM NPCStats 
                WHERE user_id = $1 AND conversation_id = $2 AND 
                      is_active = TRUE AND introduced = TRUE
            """, user_id, conversation_id)
    
    if rows:
        npcs = [dict(row) for row in rows]
        return json.dumps(npcs)
    else:
        return "No NPCs in the current scene"

@function_tool
async def get_location_description(ctx) -> str:
    """
    Get the description of the current location.
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id

# Add to the appropriate SDK module for narrative management

@function_tool
async def update_narrative_arcs_for_interaction(
    ctx,
    user_input: str,
    ai_response: str
) -> str:
    """
    Update narrative arcs based on the player interaction.
    This helps Nyx maintain coherent storylines.
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    
    async with asyncpg.create_pool(dsn=DB_DSN) as pool:
        async with pool.acquire() as conn:
            # Get current narrative arcs
            row = await conn.fetchrow("""
                SELECT value FROM CurrentRoleplay 
                WHERE user_id = $1 AND conversation_id = $2 AND key = 'NyxNarrativeArcs'
            """, user_id, conversation_id)
            
            if not row or not row["value"]:
                return "No narrative arcs defined"
            
            narrative_arcs = json.loads(row["value"])
            
            # Check for progression in active arcs
            for arc in narrative_arcs.get("active_arcs", []):
                # Simple keyword matching to detect progression
                arc_keywords = arc.get("keywords", [])
                progression_detected = False
                
                # Check user input and AI response for keywords
                combined_text = f"{user_input} {ai_response}".lower()
                for keyword in arc_keywords:
                    if keyword.lower() in combined_text:
                        progression_detected = True
                        break
                
                if progression_detected:
                    # Update arc progress
                    if "progress" not in arc:
                        arc["progress"] = 0
                    
                    # Increment progress (small increment for keyword matches)
                    arc["progress"] = min(100, arc["progress"] + 5)
                    
                    # Record the interaction
                    if "interactions" not in arc:
                        arc["interactions"] = []
                    
                    arc["interactions"].append({
                        "timestamp": datetime.now().isoformat(),
                        "progression_amount": 5,
                        "notes": f"Keyword match in interaction"
                    })
                    
                    # Check for completion
                    if arc["progress"] >= 100 and arc.get("status") != "completed":
                        arc["status"] = "completed"
                        arc["completion_date"] = datetime.now().isoformat()
                        
                        # Move from active to completed
                        if arc in narrative_arcs["active_arcs"]:
                            narrative_arcs["active_arcs"].remove(arc)
                            narrative_arcs["completed_arcs"].append(arc)
                        
                        # Add record of completion
                        if "narrative_adaption_history" not in narrative_arcs:
                            narrative_arcs["narrative_adaption_history"] = []
                        
                        narrative_arcs["narrative_adaption_history"].append({
                            "event": f"Arc completed: {arc.get('name', 'Unnamed Arc')}",
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        # Activate a new arc if available
                        if narrative_arcs.get("planned_arcs", []):
                            new_arc = narrative_arcs["planned_arcs"].pop(0)
                            new_arc["status"] = "active"
                            new_arc["start_date"] = datetime.now().isoformat()
                            narrative_arcs["active_arcs"].append(new_arc)
                            
                            narrative_arcs["narrative_adaption_history"].append({
                                "event": f"New arc activated: {new_arc.get('name', 'Unnamed Arc')}",
                                "timestamp": datetime.now().isoformat()
                            })
            
            # Save updated narrative arcs
            await conn.execute("""
                UPDATE CurrentRoleplay
                SET value = $1
                WHERE user_id = $2 AND conversation_id = $3 AND key = 'NyxNarrativeArcs'
            """, json.dumps(narrative_arcs), user_id, conversation_id)
            
            # Return a summary of changes
            active_arcs = len(narrative_arcs.get("active_arcs", []))
            completed_arcs = len(narrative_arcs.get("completed_arcs", []))
            planned_arcs = len(narrative_arcs.get("planned_arcs", []))
            
            return json.dumps({
                "active_arcs": active_arcs,
                "completed_arcs": completed_arcs,
                "planned_arcs": planned_arcs,
                "updated": True
            })

@function_tool
async def evaluate_narrative_impact(
    ctx,
    memory_text: str,
    significance: int,
    tags: List[str]
) -> str:
    """
    Evaluate how a new memory impacts ongoing narrative arcs.
    Updates Nyx's narrative plans based on player actions.
    """
    user_id = ctx.context.user_id
    conversation_id = ctx.context.conversation_id
    
    # Only process significant memories
    if significance < 5:
        return json.dumps({"impact": "none", "reason": "Memory not significant enough"})
    
    async with asyncpg.create_pool(dsn=DB_DSN) as pool:
        async with pool.acquire() as conn:
            # Get current narrative arcs
            row = await conn.fetchrow("""
                SELECT value FROM CurrentRoleplay 
                WHERE user_id = $1 AND conversation_id = $2 AND key = 'NyxNarrativeArcs'
            """, user_id, conversation_id)
            
            if not row or not row["value"]:
                return json.dumps({"impact": "none", "reason": "No narrative arcs defined"})
            
            narrative_arcs = json.loads(row["value"])
            impact_detected = False
            impacted_arcs = []
            
            # Check each active arc for impact
            for arc in narrative_arcs.get("active_arcs", []):
                arc_keywords = arc.get("keywords", [])
                arc_npcs = arc.get("involved_npcs", [])
                
                # Simple impact detection - keyword overlap
                impact_score = 0
                for keyword in arc_keywords:
                    if keyword.lower() in memory_text.lower():
                        impact_score += 1
                
                # Tag overlap
                for tag in tags:
                    if tag in arc_keywords:
                        impact_score += 1
                
                # If significant impact detected
                if impact_score >= 2:
                    impact_detected = True
                    # Update arc progress
                    if "progress" not in arc:
                        arc["progress"] = 0
                    arc["progress"] += min(25, significance * 5)  # Cap at 25% progress per event
                    
                    # Record impact
                    if "key_events" not in arc:
                        arc["key_events"] = []
                    arc["key_events"].append({
                        "memory_text": memory_text,
                        "impact_score": impact_score,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Check for arc completion
                    if arc["progress"] >= 100:
                        arc["status"] = "completed"
                        arc["completion_date"] = datetime.now().isoformat()
                        narrative_arcs["completed_arcs"].append(arc)
                        narrative_arcs["active_arcs"].remove(arc)
                        
                        # Add record of adaptation
                        if "narrative_adaption_history" not in narrative_arcs:
                            narrative_arcs["narrative_adaption_history"] = []
                        
                        narrative_arcs["narrative_adaption_history"].append({
                            "event": f"Arc completed: {arc['name']}",
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        # Activate a planned arc if available
                        if narrative_arcs["planned_arcs"]:
                            new_arc = narrative_arcs["planned_arcs"].pop(0)
                            new_arc["status"] = "active"
                            new_arc["start_date"] = datetime.now().isoformat()
                            narrative_arcs["active_arcs"].append(new_arc)
                            
                            narrative_arcs["narrative_adaption_history"].append({
                                "event": f"New arc activated: {new_arc['name']}",
                                "timestamp": datetime.now().isoformat()
                            })
                    
                    impacted_arcs.append(arc["name"])
            
            # Save updated narrative arcs if changes were made
            if impact_detected:
                await conn.execute("""
                    UPDATE CurrentRoleplay
                    SET value = $1
                    WHERE user_id = $2 AND conversation_id = $3 AND key = 'NyxNarrativeArcs'
                """, json.dumps(narrative_arcs), user_id, conversation_id)
                
                return json.dumps({
                    "impact": "significant",
                    "impacted_arcs": impacted_arcs
                })
            else:
                return json.dumps({
                    "impact": "minimal",
                    "reason": "No significant impact on current narrative arcs"
                })
