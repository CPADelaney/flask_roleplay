# routes/npc_creation_routes.py

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import logging
import asyncio

from npcs.new_npc_creation import NPCCreationHandler

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

def get_npc_handler():
    return NPCCreationHandler()

async def create_npc_background(
    handler: NPCCreationHandler, 
    user_id: int, 
    conversation_id: int, 
    environment_desc: Optional[str] = None,
    archetype_names: Optional[List[str]] = None,
    specific_traits: Optional[Dict[str, Any]] = None
):
    """Background task to create an NPC"""
    try:
        await handler.create_npc_with_context(
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
    handler: NPCCreationHandler = Depends(get_npc_handler)
):
    """
    Create a new NPC with the handler.
    
    Returns immediately with basic info while full creation continues in background.
    """
    try:
        # Use the handler's API method
        result = await handler.create_npc_api(request)
        
        # Check for errors
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return NPCResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error creating NPC: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/spawn-multiple", response_model=MultipleNPCResponse)
async def spawn_multiple_npcs(
    request: MultipleNPCRequest,
    handler: NPCCreationHandler = Depends(get_npc_handler)
):
    """
    Spawn multiple NPCs at once.
    """
    try:
        # Use the handler's API method
        result = await handler.spawn_multiple_npcs_api(request)
        
        # Check for errors
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return MultipleNPCResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error spawning multiple NPCs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{npc_id}", response_model=NPCResponse)
async def get_npc(
    npc_id: int,
    user_id: int,
    conversation_id: int,
    handler: NPCCreationHandler = Depends(get_npc_handler)
):
    """
    Get details for a specific NPC.
    """
    try:
        # Use the handler's API method
        result = await handler.get_npc_api(npc_id, user_id, conversation_id)
        
        # Check for errors
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        
        return NPCResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting NPC details: {e}")
        raise HTTPException(status_code=500, detail=str(e))
