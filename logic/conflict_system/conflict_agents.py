# logic/conflict_system/conflict_agents.py
"""
Conflict System Agents - OpenAI Assistants Implementation
Combines best practices from both approaches with enhanced error handling and maintainability.

Key features:
- Robust error handling with local fallback
- External instruction files with caching
- Concurrent assistant creation
- Proper metadata persistence
- Safe handling of tool call outputs
- Production-ready timeout handling
- Integration with your existing OpenAI configuration

IMPORTANT: Ensure OPENAI_API_KEY is set in your environment variables.

Environment Variables:
- OPENAI_API_KEY: Your OpenAI API key (required)
- CONFLICT_AGENTS_MODEL: Model to use (default: gpt-4.1-nano)
- CONFLICT_AGENTS_TEMPERATURE: Temperature setting (default: 0.7)
- CONFLICT_AGENTS_RETRY_MAX: Max retry attempts (default: 5)
- CONFLICT_AGENTS_RETRY_DELAY: Initial retry delay in seconds (default: 1)
- CONFLICT_AGENTS_RETRY_BACKOFF: Backoff multiplier (default: 2)

Usage:
    from logic.conflict_system.conflict_agents import initialize_conflict_assistants, routed_conflict_query, ConflictContext
    
    # Initialize assistants
    assistants = await initialize_conflict_assistants()
    
    # Create context
    context = ConflictContext(user_id=123, conversation_id=456)
    
    # Route a query
    result = await routed_conflict_query(
        "Lady Nyx wants to manipulate the merchant guild",
        context,
        assistants
    )
"""
from __future__ import annotations

import logging
import json
import asyncio
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from functools import lru_cache

from openai.types.beta.assistant import Assistant
from pydantic import BaseModel, Field

# ── Local imports (unchanged) ────────────────────────────────────────────────
from db.connection import get_db_connection_context
from logic.stats_logic import apply_stat_change
from logic.resource_management import ResourceManager
from npcs.npc_relationship import NPCRelationshipManager
from logic.relationship_integration import RelationshipIntegration

# Import OpenAI client from your existing integration
# This provides:
# - get_async_openai_client(): Returns AsyncOpenAI client with your API key
from logic.chatgpt_integration import get_async_openai_client

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────
INSTRUCTION_DIR = Path(__file__).parent.parent.parent / "docs" / "conflict_instructions"

# Model configuration - can be overridden via environment variable
DEFAULT_MODEL = os.getenv("CONFLICT_AGENTS_MODEL", "gpt-4.1-nano")
DEFAULT_TEMPERATURE = float(os.getenv("CONFLICT_AGENTS_TEMPERATURE", "0.7"))

# Retry configuration - adjust these if needed
# Note: We implement async retry logic inline rather than using decorators
# to maintain compatibility with async functions
RETRY_MAX_ATTEMPTS = int(os.getenv("CONFLICT_AGENTS_RETRY_MAX", "5"))
RETRY_INITIAL_DELAY = float(os.getenv("CONFLICT_AGENTS_RETRY_DELAY", "1"))
RETRY_BACKOFF_FACTOR = float(os.getenv("CONFLICT_AGENTS_RETRY_BACKOFF", "2"))

# ── Global state ─────────────────────────────────────────────────────────────
_ASSISTANT_CACHE: Dict[str, Assistant] = {}

# Lazy initialization of client
_client = None

def get_client():
    """Get or create the OpenAI client using your configuration."""
    global _client
    if _client is None:
        _client = get_async_openai_client()
    return _client


# ╭─────────────────────── Context Management ──────────────────────────────╮
@dataclass
class ConflictContext:
    """Context object that travels with requests through the assistant system."""
    user_id: int
    conversation_id: int
    resource_manager: Optional[ResourceManager] = None
    cached_data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.resource_manager is None:
            self.resource_manager = ResourceManager(self.user_id, self.conversation_id)
        if self.cached_data is None:
            self.cached_data = {}

    def stringify_metadata(metadata: Dict[str, Any]) -> Dict[str, str]:
        """
        Convert all metadata values to strings for OpenAI API compatibility.
        OpenAI requires all metadata values to be strings.
        """
        if not metadata:
            return {}
        
        result = {}
        for key, value in metadata.items():
            if isinstance(value, (dict, list)):
                # For complex types, JSON serialize them
                result[key] = json.dumps(value)
            elif value is None:
                result[key] = "null"
            elif isinstance(value, bool):
                result[key] = str(value).lower()
            else:
                result[key] = str(value)
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for metadata storage."""
        return {
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "cached_data": self.cached_data
        }
# ╰──────────────────────────────────────────────────────────────────────────╯


# ╭─────────────────── Instruction Management ──────────────────────────────╮
@lru_cache(maxsize=32)
def load_instructions(role: str, fallback: str = "") -> str:
    """
    Load instructions from external file with fallback to inline text.
    Cached for performance.
    """
    instruction_file = INSTRUCTION_DIR / f"{role}_instructions.md"
    
    if instruction_file.exists():
        try:
            return instruction_file.read_text().strip()
        except Exception as e:
            logger.warning(f"Failed to load instructions for {role}: {e}")
    
    return fallback.strip() if fallback else f"Assistant for {role} operations."


# Default instruction fallbacks
INSTRUCTION_FALLBACKS = {
    "triage": """
You are the Conflict Triage Assistant for a femdom RPG.
Classify each incoming request into one of these categories:
- generate_conflict OR conflict_generation: Creating new conflicts
- manage_stakeholders OR stakeholder_management: Managing NPC motivations and relationships  
- manipulation: Handling manipulation attempts
- resolution_tracking: Tracking conflict progress and beats
- resolve_conflict OR conflict_resolution: Resolving conflicts
- personality OR stakeholder_personality: Creating stakeholder personalities
- negotiation OR alliance_negotiation: Managing alliances
- revelation OR secret_revelation: Revealing secrets strategically
- evolution OR conflict_evolution: Evolving existing conflicts

Return JSON: {"target": "<category>", "query": "<refined_query>", "confidence": 0.0-1.0}
Use the first form (generate_conflict, manage_stakeholders, etc.) for consistency.
""",
    
    "conflict_generation": """
Conflict Generation Assistant. Create rich, multi-layered conflicts with:
- Multiple stakeholders with competing interests
- Several resolution paths with different outcomes
- Femdom-themed manipulation opportunities
- Hidden secrets and revelations
- Power dynamic considerations

Return structured JSON matching the ConflictDetails schema.
""",
    
    "stakeholder_management": """
Stakeholder Management Assistant. Manage:
- NPC motivations and hidden agendas
- Relationship dynamics and power struggles
- Alliance formations and betrayals
- Influence networks
- All through a femdom narrative lens

Focus on complex interpersonal dynamics and power exchanges.
""",
    
    "manipulation": """
Manipulation Assistant. Generate and resolve:
- Domination attempts and power plays
- Blackmail and leverage scenarios
- Seduction and control tactics
- Psychological manipulation
- Submission dynamics

Return JSON with ManipulationAttempt structure including success factors.
""",
    
    "resolution_tracking": """
Resolution Tracking Assistant. Monitor:
- Conflict progression and story beats
- Character development through conflict
- Power dynamic shifts
- Outcome branches and consequences
- Narrative coherence

Return StoryBeatResult JSON for each update.
""",
}
# ╰──────────────────────────────────────────────────────────────────────────╯


# ╭────────────────── Assistant Factory with Fallback ──────────────────────╮
async def create_or_get_assistant(
    role: str,
    *,
    instructions: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    tools: Optional[List[Dict]] = None,
    response_format: Optional[Dict] = None,
    metadata: Optional[Dict] = None,
) -> Union[Assistant, 'LocalAssistantStub']:
    """
    Create or retrieve a cached Assistant with robust error handling.
    Falls back to local stub on failure to maintain development flow.
    Uses your OpenAI configuration with retry logic.
    """
    cache_key = f"conflict:{role}"
    
    if cache_key in _ASSISTANT_CACHE:
        logger.debug(f"Returning cached assistant for {role}")
        return _ASSISTANT_CACHE[cache_key]
    
    # Load instructions with fallback
    if instructions is None:
        instructions = load_instructions(
            role, 
            INSTRUCTION_FALLBACKS.get(role, f"Assistant for {role} operations")
        )
    
    # Retry logic for assistant creation
    delay = RETRY_INITIAL_DELAY
    last_error = None
    
    for attempt in range(RETRY_MAX_ATTEMPTS):
        try:
            # Get client using your configuration
            client = get_client()
            
            # Prepare assistant configuration
            config = {
                "name": role.replace("_", " ").title() + " Assistant",
                "model": model,
                "instructions": instructions,
                "temperature": temperature,
                "tools": tools or [],
                "metadata": stringify_metadata(metadata or {}),  # Use the helper
            }
            
            # Add response format if specified
            if response_format:
                config["response_format"] = response_format
            else:
                config["response_format"] = {"type": "json_object"}
            
            # Create the assistant
            assistant = await client.beta.assistants.create(**config)
            logger.info(f"Created assistant '{assistant.name}' (ID: {assistant.id})")
            
            _ASSISTANT_CACHE[cache_key] = assistant
            return assistant
            
        except Exception as e:
            last_error = e
            if "rate" in str(e).lower() and attempt < RETRY_MAX_ATTEMPTS - 1:
                logger.warning(f"Rate limit hit on attempt {attempt+1}/{RETRY_MAX_ATTEMPTS}: {e}. Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
                delay *= RETRY_BACKOFF_FACTOR
            else:
                break
    
    # If we get here, all retries failed
    logger.error(f"Failed to create assistant for {role} after {RETRY_MAX_ATTEMPTS} attempts: {last_error}")
    logger.warning(f"Using local stub for {role} to maintain functionality")
    
    # Create local fallback
    stub = LocalAssistantStub(
        role=role,
        instructions=instructions,
        temperature=temperature,
        metadata=metadata or {}
    )
    
    _ASSISTANT_CACHE[cache_key] = stub
    return stub


class LocalAssistantStub:
    """Local fallback when Assistant creation fails."""
    def __init__(self, role: str, instructions: str, temperature: float, metadata: Dict):
        self.id = f"local-stub-{role}"
        self.name = f"{role} (Local Stub)"
        self.instructions = instructions
        self.temperature = temperature
        self.metadata = metadata
        self.model = "local"
        self.tools = []  # For compatibility with Assistant interface
        
    async def process(self, message: str, context: Optional[ConflictContext] = None) -> Dict:
        """Minimal processing for development continuity."""
        logger.warning(f"Using local stub for {self.name}")
        return {
            "status": "fallback",
            "message": f"Local processing for: {message[:100]}...",
            "role": self.name
        }
# ╰──────────────────────────────────────────────────────────────────────────╯


# ╭────────────────── Configuration Validation ─────────────────────────────╮
async def verify_openai_configuration() -> bool:
    """
    Verify that OpenAI configuration is working properly.
    Returns True if successful, False otherwise.
    """
    try:
        client = get_client()
        # Try a simple API call to verify credentials
        # Note: Some endpoints might require different permissions
        # If models.list fails, try a simple chat completion instead
        try:
            models = await client.models.list()
            logger.info(f"OpenAI configuration verified. Models accessible.")
            return True
        except Exception:
            # Fallback: try creating a simple chat completion
            response = await client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            logger.info("OpenAI configuration verified via chat completion test.")
            return True
    except Exception as e:
        logger.error(f"OpenAI configuration error: {e}")
        logger.error("Please ensure OPENAI_API_KEY is set in your environment")
        return False
# ╰──────────────────────────────────────────────────────────────────────────╯


# ╭────────────────── Assistant Initialization ───────────────────────────╮
async def initialize_conflict_assistants(verify_config: bool = True) -> Dict[str, Assistant]:
    """
    Initialize all conflict system assistants with proper configuration.
    Returns a comprehensive mapping of role -> Assistant.
    
    Args:
        verify_config: If True, verify OpenAI configuration before proceeding
    """
    if verify_config:
        if not await verify_openai_configuration():
            logger.error("Failed to verify OpenAI configuration. Using local stubs.")
    
    logger.info("Initializing conflict system assistants...")
    
    # Core assistants with specific configurations
    assistant_configs = [
        # Triage - lower temperature for consistent routing, with metadata placeholder
        ("triage", {
            "temperature": 0.3, 
            "model": DEFAULT_MODEL,  # Use configured model
            "metadata": {"handoff_table": {}}  # Initialize with empty table
        }),
        
        # Primary conflict assistants
        ("conflict_generation", {"temperature": 0.8}),
        ("stakeholder_management", {"temperature": 0.7}),
        ("manipulation", {"temperature": 0.8}),
        ("resolution_tracking", {"temperature": 0.6}),
        ("conflict_resolution", {"temperature": 0.7}),
        
        # Specialized assistants
        ("stakeholder_personality", {"temperature": 0.9}),
        ("alliance_negotiation", {"temperature": 0.7}),
        ("secret_revelation", {"temperature": 0.8}),
        ("conflict_manager", {"temperature": 0.6}),
        ("conflict_evolution", {"temperature": 0.8}),
        ("conflict_seed", {"temperature": 0.85}),
        ("world_state_interpreter", {"temperature": 0.5}),
    ]
    
    assistants = {}
    
    # Create assistants concurrently for better performance
    tasks = []
    for role, config in assistant_configs:
        task = create_or_get_assistant(role, **config)
        tasks.append((role, task))
    
    # Wait for all assistants to be created concurrently
    roles, coros = zip(*tasks)
    created_assistants = await asyncio.gather(*coros)
    
    for role, assistant in zip(roles, created_assistants):
        assistants[role] = assistant
        
        # Add legacy mappings for backward compatibility
        if role == "conflict_generation":
            assistants["conflict_generation_agent"] = assistant
        elif role == "stakeholder_management":
            assistants["stakeholder_agent"] = assistant
        elif role == "manipulation":
            assistants["manipulation_agent"] = assistant
        elif role == "resolution_tracking":
            assistants["resolution_agent"] = assistant
    
    # Set up handoff table with both normalized and legacy keys
    handoff_mapping = {
        # Normalized keys (what triage returns)
        "generate_conflict": assistants["conflict_generation"],
        "manage_stakeholders": assistants["stakeholder_management"],
        "manipulation": assistants["manipulation"],
        "resolution_tracking": assistants["resolution_tracking"],
        "resolve_conflict": assistants["conflict_resolution"],
        "personality": assistants["stakeholder_personality"],
        "negotiation": assistants["alliance_negotiation"],
        "revelation": assistants["secret_revelation"],
        "evolution": assistants["conflict_evolution"],
        
        # Direct mapping keys (for flexibility)
        "conflict_generation": assistants["conflict_generation"],
        "stakeholder_management": assistants["stakeholder_management"],
        "conflict_resolution": assistants["conflict_resolution"],
        "stakeholder_personality": assistants["stakeholder_personality"],
        "alliance_negotiation": assistants["alliance_negotiation"],
        "secret_revelation": assistants["secret_revelation"],
        "conflict_manager": assistants["conflict_manager"],
        "conflict_evolution": assistants["conflict_evolution"],
        "conflict_seed": assistants["conflict_seed"],
        "world_state_interpreter": assistants["world_state_interpreter"],
    }
    
    # Update triage metadata both locally and on server
    triage = assistants["triage"]
    # Store the actual Assistant objects locally for runtime use
    triage_metadata = (triage.metadata or {}) | {"handoff_table": handoff_mapping}
    triage.metadata = triage_metadata  # Fix: was using undefined 'handoff_table'
    
    api_metadata = stringify_metadata({
        "role": "triage",
        "updated_at": datetime.utcnow().isoformat()
        # Don't include handoff_table here as it contains non-serializable Assistant objects
    })
    
    try:
        client = get_client()
        await client.beta.assistants.update(
            triage.id,
            metadata=api_metadata
        )
        logger.debug("Successfully updated triage assistant metadata on server")
    except Exception as e:
        logger.warning(f"Could not sync triage metadata to server: {e}")
    
    logger.info(f"Initialized {len(assistants)} conflict assistants")
    return assistants
# ╰──────────────────────────────────────────────────────────────────────────╯


# ╭────────────────────── Runtime Helpers ──────────────────────────────────╮
async def ask_assistant(
    assistant: Union[Assistant, LocalAssistantStub],
    message: str,
    context: Optional[ConflictContext] = None,
    parse_json: bool = True,
    timeout: int = 30,
) -> Union[Dict[str, Any], str]:
    """
    Send a message to an assistant and get the response.
    Handles both real Assistants and local stubs.
    Uses your OpenAI configuration with retry logic.
    """
    # Handle local stub
    if isinstance(assistant, LocalAssistantStub):
        return await assistant.process(message, context)
    
    # Retry logic for API calls
    delay = RETRY_INITIAL_DELAY
    last_error = None
    
    for attempt in range(RETRY_MAX_ATTEMPTS):
        try:
            # Get client using your configuration
            client = get_client()
            
            # Create thread with context metadata
            thread_metadata = stringify_metadata(context.to_dict() if context else {})
            thread = await client.beta.threads.create(metadata=thread_metadata)
            
            # Add user message with context in message metadata for privacy
            await client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=message,
                metadata=thread_metadata,  # Already stringified
            )
            
            # Create and poll run with timeout
            run = await asyncio.wait_for(
                client.beta.threads.runs.create_and_poll(
                    thread_id=thread.id,
                    assistant_id=assistant.id,
                ),
                timeout=timeout
            )
            
            # Check run status
            if run.status != "completed":
                logger.error(f"Run failed with status: {run.status}")
                return {"error": f"Run failed: {run.status}", "assistant": assistant.name}
            
            # Get the response within the same timeout window
            messages = await client.beta.threads.messages.list(
                thread_id=thread.id,
                order="desc",
                limit=1
            )
            
            if not messages.data:
                raise ValueError("No response from assistant")
            
            # Handle different content types (text vs tool calls)
            response_text = None
            content_parts = messages.data[0].content
            
            for part in content_parts:
                if hasattr(part, 'type') and part.type == "text":
                    response_text = part.text.value
                    break
            
            if response_text is None:
                # No text content found, might be tool calls
                logger.warning(f"No text response from {assistant.name}, content: {content_parts}")
                return {"error": "No text response", "content_type": "non-text"}
            
            # Parse JSON if requested
            if parse_json:
                try:
                    return json.loads(response_text)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON from {assistant.name}: {response_text[:100]}")
                    return {"error": "Invalid JSON response", "raw": response_text}
            
            return response_text
            
        except asyncio.TimeoutError:
            logger.error(f"Timeout waiting for {assistant.name}")
            return {"error": "Assistant timeout", "assistant": assistant.name}
        except Exception as e:
            last_error = e
            if "rate" in str(e).lower() and attempt < RETRY_MAX_ATTEMPTS - 1:
                logger.warning(f"Rate limit hit on attempt {attempt+1}/{RETRY_MAX_ATTEMPTS}: {e}. Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
                delay *= RETRY_BACKOFF_FACTOR
            else:
                break
    
    # If we get here, all retries failed
    logger.error(f"Error calling {assistant.name} after {RETRY_MAX_ATTEMPTS} attempts: {last_error}")
    return {"error": str(last_error), "assistant": assistant.name}


async def routed_conflict_query(
    message: str,
    context: ConflictContext,
    assistants: Optional[Dict[str, Assistant]] = None,
) -> Dict[str, Any]:
    """
    Route a conflict query through triage to the appropriate specialist.
    """
    # Initialize assistants if not provided
    if assistants is None:
        assistants = await initialize_conflict_assistants()
    
    triage = assistants["triage"]
    
    # Get routing decision from triage
    triage_response = await ask_assistant(triage, message, context)
    
    if "error" in triage_response:
        return triage_response
    
    # Extract routing information
    target = triage_response.get("target", "conflict_resolution")
    refined_query = triage_response.get("query", message)
    confidence = triage_response.get("confidence", 0.5)
    
    logger.info(f"Routing to {target} (confidence: {confidence})")
    
    # Get specialist from handoff table (safely handle None metadata)
    triage_metadata = triage.metadata or {}
    handoff_table = triage_metadata.get("handoff_table", {})
    specialist = handoff_table.get(target)
    
    if not specialist:
        # Try direct assistant lookup as fallback
        specialist = assistants.get(target)
        
    if not specialist:
        logger.warning(f"No specialist found for {target}, using resolution")
        specialist = assistants.get("conflict_resolution", assistants.get("resolution_tracking"))
    
    # Get response from specialist
    response = await ask_assistant(specialist, refined_query, context)
    
    # Add routing metadata
    if isinstance(response, dict):
        response["_routing"] = {
            "target": target,
            "confidence": confidence,
            "original_query": message
        }
    
    return response
# ╰──────────────────────────────────────────────────────────────────────────╯


# ╭──────────────── Specialized Helper Functions ───────────────────────────╮
async def generate_conflict(
    context: ConflictContext,
    conflict_type: str,
    participants: List[Dict],
    stakes: str,
    assistants: Optional[Dict[str, Assistant]] = None,
) -> Dict[str, Any]:
    """Generate a new conflict with rich details."""
    if assistants is None:
        assistants = await initialize_conflict_assistants()
    
    generator = assistants["conflict_generation"]
    
    prompt = f"""
    Generate a {conflict_type} conflict:
    Participants: {json.dumps(participants)}
    Stakes: {stakes}
    Context: User {context.user_id}, Conversation {context.conversation_id}
    
    Include multiple resolution paths, manipulation opportunities, and hidden agendas.
    """
    
    return await ask_assistant(generator, prompt, context)


async def process_manipulation_attempt(
    context: ConflictContext,
    manipulator: Dict,
    target: Dict,
    technique: str,
    leverage: Optional[Dict] = None,
    assistants: Optional[Dict[str, Assistant]] = None,
) -> Dict[str, Any]:
    """Process a manipulation attempt with full context."""
    if assistants is None:
        assistants = await initialize_conflict_assistants()
    
    manipulator_assistant = assistants["manipulation"]
    
    # Get relationship leverage if not provided
    if leverage is None and "id" in manipulator and "id" in target:
        leverage = await get_manipulation_leverage(
            context.user_id,
            context.conversation_id,
            manipulator["id"],
            target["id"]
        )
    
    prompt = f"""
    Process manipulation attempt:
    Manipulator: {json.dumps(manipulator)}
    Target: {json.dumps(target)}
    Technique: {technique}
    Leverage: {json.dumps(leverage or {})}
    
    Calculate success probability and generate outcome narrative.
    """
    
    return await ask_assistant(manipulator_assistant, prompt, context)
# ╰──────────────────────────────────────────────────────────────────────────╯


# ╭───────────── Compatibility Functions (unchanged) ───────────────────────╮
async def get_relationship_status(user_id, conversation_id,
                                  entity1_type, entity1_id,
                                  entity2_type, entity2_id):
    """Get relationship status between two entities."""
    if entity1_type == "npc":
        mgr = NPCRelationshipManager(entity1_id, user_id, conversation_id)
        return await mgr.get_relationship_details(entity2_type, entity2_id)
    
    integrator = RelationshipIntegration(user_id, conversation_id)
    return await integrator.get_relationship(entity1_type, entity1_id,
                                             entity2_type, entity2_id)


async def get_manipulation_leverage(user_id, conversation_id,
                                    manipulator_id, target_id):
    """Calculate manipulation leverage based on relationship."""
    mgr = NPCRelationshipManager(manipulator_id, user_id, conversation_id)
    rel = await mgr.get_relationship_details("npc", target_id)
    
    leverage = 0.0
    lvl = rel.get("link_level", 0)
    
    if lvl > 75: 
        leverage = 0.8
    elif lvl > 50: 
        leverage = 0.5
    elif lvl > 25: 
        leverage = 0.3
    
    # Add control dynamics
    leverage += rel.get("dynamics", {}).get("control", 0) / 100 * 0.2
    
    return {
        "leverage_score": min(1.0, leverage),
        "relationship_level": lvl,
        "relationship_type": rel.get("link_type", "neutral"),
        "control_factor": rel.get("dynamics", {}).get("control", 0),
    }
# ╰──────────────────────────────────────────────────────────────────────────╯


# ╭────────────────────── Main Entry Point ─────────────────────────────────╮
async def initialize_agents(verify_config: bool = True):
    """Legacy compatibility wrapper."""
    return await initialize_conflict_assistants(verify_config)


# Module exports
__all__ = [
    "ConflictContext",
    "initialize_conflict_assistants",
    "initialize_agents",  # legacy
    "ask_assistant",
    "routed_conflict_query",
    "generate_conflict",
    "process_manipulation_attempt",
    "get_relationship_status",
    "get_manipulation_leverage",
]
# ╰──────────────────────────────────────────────────────────────────────────╯
