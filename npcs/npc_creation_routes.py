# routes/npc_creation_routes.py

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import logging
import asyncio

from npcs.npc_creation_agent import NPCCreationAgent, NPCCreationResult

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

# Dependency for the NPC creation agent
def get_npc_agent():
    return NPCCreationAgent()

async def create_npc_background(
    agent: NPCCreationAgent, 
    user_id: int, 
    conversation_id: int, 
    environment_desc: Optional[str] = None,
    archetype_names: Optional[List[str]] = None,
    specific_traits: Optional[Dict[str, Any]] = None
):
    """Background task to create an NPC"""
    try:
        await agent.create_npc(
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
    agent: NPCCreationAgent = Depends(get_npc_agent)
):
    """
    Create a new NPC with the agent.
    
    Returns immediately with basic info while full creation continues in background.
    """
    try:
        # Start creation process and get initial result
        result = await agent.create_npc(
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
        if hasattr(result, "archetypes") and hasattr(result.archetypes, "archetype_names"):
            archetypes = result.archetypes.archetype_names
        
        # Extract personality traits
        personality_traits = []
        if hasattr(result, "personality") and hasattr(result.personality, "personality_traits"):
            personality_traits = result.personality.personality_traits
        
        return NPCResponse(
            npc_id=result.npc_id,
            npc_name=result.npc_name,
            physical_description=result.physical_description,
            personality_traits=personality_traits,
            archetypes=archetypes,
            current_location=result.current_location,
            message="NPC created successfully"
        )
    
    except Exception as e:
        logging.error(f"Error creating NPC: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/spawn-multiple", response_model=MultipleNPCResponse)
async def spawn_multiple_npcs(
    request: MultipleNPCRequest,
    agent: NPCCreationAgent = Depends(get_npc_agent)
):
    """
    Spawn multiple NPCs at once.
    """
    try:
        # Spawn multiple NPCs
        npc_ids = await agent.spawn_multiple_npcs(
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

@router.get("/{npc_id}", response_model=NPCResponse)
async def get_npc(
    npc_id: int,
    user_id: int,
    conversation_id: int,
    agent: NPCCreationAgent = Depends(get_npc_agent)
):
    """
    Get details for a specific NPC.
    """
    try:
        # Create a dummy context object
        ctx = type("Context", (), {"context": {"user_id": user_id, "conversation_id": conversation_id}})
        
        # Get NPC details
        npc_details = await agent.get_npc_details(ctx, npc_id=npc_id)
        
        if "error" in npc_details:
            raise HTTPException(status_code=404, detail="NPC not found")
        
        # Extract archetypes
        archetypes = []
        if "archetypes" in npc_details:
            for arch in npc_details["archetypes"]:
                if isinstance(arch, dict) and "name" in arch:
                    archetypes.append(arch["name"])
        
        return NPCResponse(
            npc_id=npc_details["npc_id"],
            npc_name=npc_details["npc_name"],
            physical_description=npc_details["physical_description"],
            personality_traits=npc_details["personality_traits"],
            archetypes=archetypes,
            current_location=npc_details.get("current_location", "")
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting NPC details: {e}")
        raise HTTPException(status_code=500, detail=str(e))
