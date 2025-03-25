# server.py

import os
import asyncio
import logging
import json
import uvicorn
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from nyx.api import nyx_api
from nyx.session_factory import NyxSessionFactory
from nyx.resource_monitor import ResourceMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("nyx-server")

# Initialize FastAPI app
app = FastAPI(
    title="Nyx Roleplay API",
    description="API for Nyx roleplay with session-based architecture",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class MessageRequest(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None

class ConversationRequest(BaseModel):
    user_id: int
    conversation_id: int

class MessageWithConversation(BaseModel):
    user_id: int
    conversation_id: int
    message: str
    context: Optional[Dict[str, Any]] = None

class PriorityRequest(BaseModel):
    user_id: int
    priority: str

class CleanupRequest(BaseModel):
    max_idle_time: Optional[int] = 3600

class StatusResponse(BaseModel):
    success: bool
    status: str
    session_stats: Optional[Dict[str, Any]] = None
    health_metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# API Routes

@app.post("/message")
async def process_message(request: MessageWithConversation):
    """Process a message in a conversation."""
    result = await nyx_api.process_message(
        request.user_id,
        request.conversation_id,
        request.message,
        request.context
    )
    
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
    
    return result

@app.post("/conversation/close")
async def close_conversation(request: ConversationRequest):
    """Close a conversation."""
    result = await nyx_api.close_conversation(
        request.user_id,
        request.conversation_id
    )
    
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
    
    return result

@app.get("/status")
async def get_system_status():
    """Get system status."""
    result = await nyx_api.get_system_status()
    
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
    
    return result

@app.get("/conversation/{user_id}/{conversation_id}/insights")
async def get_conversation_insights(user_id: int, conversation_id: int):
    """Get insights for a conversation."""
    result = await nyx_api.get_conversation_insights(user_id, conversation_id)
    
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
    
    return result

@app.get("/insights/cross-user")
async def get_cross_user_insights():
    """Get cross-user insights."""
    result = await nyx_api.get_cross_user_insights()
    
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
    
    return result

@app.post("/user/priority")
async def set_user_priority(request: PriorityRequest):
    """Set user priority."""
    result = await nyx_api.set_user_priority(
        request.user_id,
        request.priority
    )
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Invalid priority"))
    
    return result

@app.post("/maintenance/cleanup")
async def cleanup_inactive_sessions(request: CleanupRequest):
    """Cleanup inactive sessions."""
    result = await nyx_api.cleanup_inactive_sessions(
        request.max_idle_time
    )
    
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
    
    return result

# Background tasks

async def run_periodic_maintenance():
    """Run periodic maintenance tasks."""
    while True:
        try:
            # Clean up inactive sessions
            await nyx_api.cleanup_inactive_sessions()
            
            # Wait before next run
            await asyncio.sleep(3600)  # Run every hour
        except Exception as e:
            logger.error(f"Error in periodic maintenance: {e}")
            await asyncio.sleep(60)  # Retry after a minute

@app.on_event("startup")
async def startup_event():
    """Run when the server starts."""
    logger.info("Starting Nyx API server")
    
    # Start the resource monitor
    resource_monitor = ResourceMonitor.get_instance()
    resource_monitor.start_monitoring()
    
    # Start periodic maintenance
    asyncio.create_task(run_periodic_maintenance())

@app.on_event("shutdown")
async def shutdown_event():
    """Run when the server shuts down."""
    logger.info("Shutting down Nyx API server")
    
    # Stop the resource monitor
    resource_monitor = ResourceMonitor.get_instance()
    await resource_monitor.stop_monitoring()
    
    # Close all sessions
    session_factory = NyxSessionFactory.get_instance()
    
    # Get session keys
    session_keys = list(session_factory.sessions.keys())
    
    # Close each session
    for session_key in session_keys:
        try:
            user_id, conversation_id = map(int, session_key.split('_'))
            await session_factory.close_session(user_id, conversation_id)
        except Exception as e:
            logger.error(f"Error closing session {session_key}: {e}")

# Run the server
if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 8000))
    
    # Run with uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
