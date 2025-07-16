# npcs/npc_routes.py

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import logging
import asyncio

from .creation import NPCCreationHandler
from .agent import NPCAgent
from .coordinator import NPCAgentCoordinator
from .system import NPCAgentSystem
from .relationship import NPCRelationshipManager
from .handler import NPCInteractionHandler

# Change the import to use the async version
from db.connection import get_db_connection_context

router = APIRouter(prefix="/npcs", tags=["NPCs"])

# Input models
class NPCCreationRequest(BaseModel):
    user_id: int
    conversation_id: int
    environment_desc: Optional[str] = None
    archetype_names: Optional[List[str]] = None
    specific_traits: Optional[Dict[str, Any]] = None

class MultipleNPCRequest(BaseModel):
    user_id: int
    conversation_id: int
    count: int = 3
    environment_desc: Optional[str] = None

class NPCUpdateRequest(BaseModel):
    user_id: int
    conversation_id: int
    npc_id: int
    updates: Dict[str, Any]
    
class NPCInteractionRequest(BaseModel):
    user_id: int
    conversation_id: int
    npc_id: int
    player_input: str
    interaction_type: str = "standard_interaction"
    context: Dict[str, Any] = Field(default_factory=dict)

class NPCRelationshipRequest(BaseModel):
    user_id: int
    conversation_id: int
    npc1_id: int
    npc2_id: Optional[int] = None
    entity2_type: str = "player"
    entity2_id: int = 0
    relationship_type: Optional[str] = None
    level_change: int = 0

class NPCScheduleRequest(BaseModel):
    user_id: int
    conversation_id: int
    npc_id: int
    day_names: Optional[List[str]] = None

class NPCActivityRequest(BaseModel):
    user_id: int
    conversation_id: int
    location: Optional[str] = None

# Response models
class NPCResponse(BaseModel):
    npc_id: int
    npc_name: str
    physical_description: str
    personality_traits: List[str] = Field(default_factory=list)
    archetypes: List[str] = Field(default_factory=list)
    current_location: Optional[str] = None
    message: Optional[str] = None

class MultipleNPCResponse(BaseModel):
    npc_ids: List[int]
    message: str

class NPCInteractionResponse(BaseModel):
    npc_id: int
    npc_name: str
    response: str
    stat_changes: Dict[str, int] = Field(default_factory=dict)
    memory_created: bool = False

class NPCRelationshipResponse(BaseModel):
    npc1_id: int
    entity2_type: str
    entity2_id: int
    relationship_type: str
    relationship_level: int
    message: str
    
class NPCScheduleResponse(BaseModel):
    npc_id: int
    schedule: Dict[str, Dict[str, str]]
    message: str

class NPCActivityResponse(BaseModel):
    results: List[Dict[str, Any]]
    message: str

# Dependencies for the various NPC services
def get_npc_creation_handler():
    return NPCCreationHandler()

def get_npc_interaction_handler():
    return NPCInteractionHandler()

def get_npc_coordinator(user_id: int, conversation_id: int):
    return NPCAgentCoordinator(user_id, conversation_id)

async def get_npc_system(user_id: int, conversation_id: int):
    # Updated to use async connection
    return NPCAgentSystem(user_id, conversation_id, None)

async def create_npc_background(
    creation_handler: NPCCreationHandler, 
    user_id: int, 
    conversation_id: int, 
    environment_desc: Optional[str] = None,
    archetype_names: Optional[List[str]] = None,
    specific_traits: Optional[Dict[str, Any]] = None
):
    """Background task to create an NPC"""
    try:
        await creation_handler.create_npc(
            environment_desc=environment_desc,
            archetype_names=archetype_names,
            specific_traits=specific_traits,
            user_id=user_id,
            conversation_id=conversation_id
        )
    except Exception as e:
        logging.error(f"Error in background NPC creation: {e}")

@router.post("/create", response_model=NPCResponse)
async def create_npc(
    request: NPCCreationRequest,
    background_tasks: BackgroundTasks,
    creation_handler: NPCCreationHandler = Depends(get_npc_creation_handler)
):
    """
    Create a new NPC with the agent.
    
    Returns immediately with basic info while full creation continues in background.
    """
    try:
        # Start creation process and get initial result
        result = await creation_handler.create_npc(
            environment_desc=request.environment_desc,
            archetype_names=request.archetype_names,
            specific_traits=request.specific_traits,
            user_id=request.user_id,
            conversation_id=request.conversation_id
        )
        
        # Process result
        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Extract archetypes
        archetypes = []
        if "archetypes" in result and isinstance(result["archetypes"], list):
            for arch in result["archetypes"]:
                if isinstance(arch, dict) and "name" in arch:
                    archetypes.append(arch["name"])
        
        # Extract personality traits
        personality_traits = []
        if "personality_traits" in result and isinstance(result["personality_traits"], list):
            personality_traits = result["personality_traits"]
        
        return NPCResponse(
            npc_id=result["npc_id"],
            npc_name=result["npc_name"],
            physical_description=result.get("physical_description", ""),
            personality_traits=personality_traits,
            archetypes=archetypes,
            current_location=result.get("current_location", ""),
            message="NPC created successfully"
        )
    
    except Exception as e:
        logging.error(f"Error creating NPC: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/spawn-multiple", response_model=MultipleNPCResponse)
async def spawn_multiple_npcs(
    request: MultipleNPCRequest,
    creation_handler: NPCCreationHandler = Depends(get_npc_creation_handler)
):
    """
    Spawn multiple NPCs at once.
    """
    try:
        # Spawn multiple NPCs
        npc_ids = await creation_handler.spawn_multiple_npcs(
            count=request.count,
            environment_desc=request.environment_desc,
            user_id=request.user_id,
            conversation_id=request.conversation_id
        )
        
        return MultipleNPCResponse(
            npc_ids=npc_ids,
            message=f"Successfully spawned {len(npc_ids)} NPCs"
        )
    
    except Exception as e:
        logging.error(f"Error spawning multiple NPCs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{npc_id}", response_model=NPCResponse)
async def update_npc(
    npc_id: int,
    request: NPCUpdateRequest,
    creation_handler: NPCCreationHandler = Depends(get_npc_creation_handler)
):
    """
    Update an existing NPC.
    """
    try:
        # Update the NPC
        result = await creation_handler.update_npc(
            npc_id=npc_id,
            updates=request.updates,
            user_id=request.user_id,
            conversation_id=request.conversation_id
        )
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        
        # Extract archetypes
        archetypes = []
        if "archetypes" in result and isinstance(result["archetypes"], list):
            for arch in result["archetypes"]:
                if isinstance(arch, dict) and "name" in arch:
                    archetypes.append(arch["name"])
        
        return NPCResponse(
            npc_id=result["npc_id"],
            npc_name=result["npc_name"],
            physical_description=result.get("physical_description", ""),
            personality_traits=result.get("personality_traits", []),
            archetypes=archetypes,
            current_location=result.get("current_location", ""),
            message="NPC updated successfully"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error updating NPC: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{npc_id}", response_model=NPCResponse)
async def get_npc(
    npc_id: int,
    user_id: int,
    conversation_id: int,
    creation_handler: NPCCreationHandler = Depends(get_npc_creation_handler)
):
    """
    Get details for a specific NPC.
    """
    try:
        # Get NPC details
        npc_details = await creation_handler.get_npc_details(
            npc_id=npc_id,
            user_id=user_id,
            conversation_id=conversation_id
        )
        
        if "error" in npc_details:
            raise HTTPException(status_code=404, detail="NPC not found")
        
        # Extract archetypes
        archetypes = []
        if "archetypes" in npc_details and isinstance(npc_details["archetypes"], list):
            for arch in npc_details["archetypes"]:
                if isinstance(arch, dict) and "name" in arch:
                    archetypes.append(arch["name"])
        
        return NPCResponse(
            npc_id=npc_details["npc_id"],
            npc_name=npc_details["npc_name"],
            physical_description=npc_details.get("physical_description", ""),
            personality_traits=npc_details.get("personality_traits", []),
            archetypes=archetypes,
            current_location=npc_details.get("current_location", "")
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting NPC details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{npc_id}/interact", response_model=NPCInteractionResponse)
async def interact_with_npc(
    npc_id: int,
    request: NPCInteractionRequest
):
    """Interact with an NPC using the proper handler."""
    try:
        # Create handler
        handler = NPCHandler(request.user_id, request.conversation_id)
        
        # Process interaction
        response = await handler.handle_interaction(
            npc_id=npc_id,
            interaction_type=request.interaction_type,
            player_input=request.player_input,
            context=request.context
        )
        
        if "error" in response:
            raise HTTPException(status_code=404, detail=response["error"])
        
        return NPCInteractionResponse(
            npc_id=response["npc_id"],
            npc_name=response["npc_name"],
            response=response["response"],
            stat_changes=response.get("stat_changes", {}),
            memory_created=response.get("memory_created", False)
        )
    
    except Exception as e:
        logging.error(f"Error interacting with NPC: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{npc_id}/memories")
async def get_npc_memories(
    npc_id: int,
    user_id: int,
    conversation_id: int,
    query: Optional[str] = None,
    limit: int = 10,
    memory_types: Optional[List[str]] = None,
    femdom_focus: bool = False
):
    """Get memories for an NPC."""
    try:
        # Create agent
        agent = NPCAgent(npc_id, user_id, conversation_id)
        await agent.initialize()
        
        # Get memory manager
        memory_manager = await agent.get_memory_manager()
        
        # Retrieve memories
        result = await memory_manager.retrieve_memories(
            query=query or "",
            limit=limit,
            memory_types=memory_types,
            femdom_focus=femdom_focus
        )
        
        return result
    
    except Exception as e:
        logging.error(f"Error getting NPC memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{npc_id}/memories")
async def add_npc_memory(
    npc_id: int,
    user_id: int,
    conversation_id: int,
    memory_text: str,
    memory_type: str = "observation",
    significance: int = 5,
    feminine_context: bool = False
):
    """Add a memory for an NPC."""
    try:
        # Create agent
        agent = NPCAgent(npc_id, user_id, conversation_id)
        await agent.initialize()
        
        # Get memory manager
        memory_manager = await agent.get_memory_manager()
        
        # Add memory
        memory_id = await memory_manager.add_memory(
            memory_text=memory_text,
            memory_type=memory_type,
            significance=significance,
            feminine_context=feminine_context
        )
        
        return {"memory_id": memory_id, "success": True}
    
    except Exception as e:
        logging.error(f"Error adding NPC memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/group-interaction")
async def handle_group_interaction(
    user_id: int,
    conversation_id: int,
    npc_ids: List[int],
    player_action: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None
):
    """Handle a group interaction with multiple NPCs."""
    try:
        # Get coordinator
        coordinator = NPCAgentCoordinator(user_id, conversation_id)
        await coordinator.load_agents(npc_ids)
        
        # Process group interaction
        result = await coordinator.handle_player_action(
            player_action=player_action,
            context=context or {},
            npc_ids=npc_ids
        )
        
        return result
    
    except Exception as e:
        logging.error(f"Error in group interaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{npc_id}/relationship", response_model=NPCRelationshipResponse)
async def manage_npc_relationship(
    npc_id: int,
    request: NPCRelationshipRequest
):
    """
    Manage or update a relationship between NPCs or between an NPC and the player.
    """
    try:
        # Create relationship manager
        relationship_manager = NPCRelationshipManager(
            npc_id, request.user_id, request.conversation_id
        )
        
        # Process the relationship update
        entity2_id = request.npc2_id if request.npc2_id is not None else request.entity2_id
        entity2_type = "npc" if request.npc2_id is not None else request.entity2_type
        
        result = await relationship_manager.update_relationship_from_interaction(
            entity_type=entity2_type,
            entity_id=entity2_id,
            player_action={"type": "relationship_update"},
            npc_action={"type": "relationship_response"},
            context={"relationship_type": request.relationship_type, "level_change": request.level_change}
        )
        
        if not result.get("success", False):
            raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
        
        return NPCRelationshipResponse(
            npc1_id=npc_id,
            entity2_type=entity2_type,
            entity2_id=entity2_id,
            relationship_type=result.get("new_type", result.get("old_type", "unknown")),
            relationship_level=result.get("new_level", result.get("old_level", 0)),
            message="Relationship updated successfully"
        )
    
    except Exception as e:
        logging.error(f"Error managing NPC relationship: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{npc_id}/schedule", response_model=NPCScheduleResponse)
async def create_npc_schedule(
    npc_id: int,
    request: NPCScheduleRequest,
    creation_handler: NPCCreationHandler = Depends(get_npc_creation_handler)
):
    """
    Create a detailed schedule for an NPC.
    """
    try:
        # Create the schedule
        result = await creation_handler.create_npc_schedule(
            npc_id=npc_id,
            day_names=request.day_names,
            user_id=request.user_id,
            conversation_id=request.conversation_id
        )
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        
        return NPCScheduleResponse(
            npc_id=npc_id,
            schedule=result["schedule"],
            message="Schedule created successfully"
        )
    
    except Exception as e:
        logging.error(f"Error creating NPC schedule: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/daily-activities", response_model=NPCActivityResponse)
async def process_daily_activities(
    request: NPCActivityRequest
):
    """
    Process daily activities for all NPCs.
    """
    try:
        # Get NPC system
        system = await get_npc_system(request.user_id, request.conversation_id)
        
        # Process activities
        result = await system.process_daily_npc_activities()
        
        return NPCActivityResponse(
            results=result.get("results", []),
            message=f"Processed daily activities for time: {result.get('time_of_day', 'unknown')}"
        )
    
    except Exception as e:
        logging.error(f"Error processing daily activities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/nearby", response_model=List[NPCResponse])
async def get_nearby_npcs(
    user_id: int,
    conversation_id: int,
    location: Optional[str] = None,
    limit: int = 5
):
    """
    Get NPCs that are at a specific location.
    """
    try:
        # Get NPC system
        system = await get_npc_system(user_id, conversation_id)
        
        # Get nearby NPCs
        npcs = await system.get_nearby_npcs(location=location)
        
        # Format response
        response_npcs = []
        for npc in npcs[:limit]:
            response_npcs.append(
                NPCResponse(
                    npc_id=npc["npc_id"],
                    npc_name=npc["npc_name"],
                    physical_description="",  # This endpoint returns minimal info
                    personality_traits=[],
                    archetypes=[],
                    current_location=npc["current_location"],
                    message=None
                )
            )
        
        return response_npcs
    
    except Exception as e:
        logging.error(f"Error getting nearby NPCs: {e}")
        raise HTTPException(status_code=500, detail=str(e))
