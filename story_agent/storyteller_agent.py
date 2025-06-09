# story_agent/storyteller_agent.py

"""
Nyx-Integrated Storyteller Agent with Enhanced Context Management

This module implements the Storyteller Agent with enhanced context management,
fully integrated with the context system and Nyx central governance system.

Key Enhancements:
1. Consistent use of comprehensive context retrieval with vector search
2. Progressive summarization for memory management
3. Performance monitoring and token budget management
4. Context version tracking for efficient delta updates
5. Memory system integration for narrative coherence
"""

import logging
import json
import asyncio
import time
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from math import floor

from agents import Agent, Runner, function_tool, handoff
from pydantic import BaseModel, Field

# Database connection
from db.connection import get_db_connection_context

# Comprehensive Nyx governance integration
from nyx.governance_helpers import with_governance, with_governance_permission, with_action_reporting
from nyx.directive_handler import DirectiveHandler
from nyx.nyx_governance import AgentType, DirectiveType, DirectivePriority
from nyx.integrate import (
    get_central_governance,
    process_story_beat_with_governance,
    broadcast_event_with_governance, 
    remember_with_governance,
    recall_with_governance
)

# Enhanced context system integration
from context.context_service import get_context_service, get_comprehensive_context
from context.context_config import get_config
from context.memory_manager import get_memory_manager, Memory
from context.vector_service import get_vector_service
from context.context_manager import get_context_manager, ContextDiff
from context.context_performance import PerformanceMonitor, track_performance
from context.unified_cache import context_cache

# Import existing components
from logic.aggregator_sdk import get_aggregated_roleplay_context, format_context_for_compatibility
from logic.addiction_emergence import analyze_and_apply_emergent_addictions
from routes.story_routes import build_aggregator_text
from logic.universal_updater_agent import apply_universal_updates_async
from logic.gpt_image_decision import should_generate_image_for_response
from routes.ai_image_generator import generate_roleplay_image_from_gpt
from logic.time_cycle import get_current_time, should_advance_time, nightly_maintenance
from utils.performance import PerformanceTracker
from logic.conflict_system.conflict_integration import ConflictSystemIntegration

# Progressive summarization integration
from story_agent.progressive_summarization import (
    ProgressiveNarrativeSummarizer, 
    SummaryLevel,
    RPGNarrativeManager
)

from logic.conflict_system.hooks import (
    on_player_major_action,
    get_player_conflicts,
    get_conflict_summary,
    advance_conflict_story
)
from logic.conflict_system.conflict_integration import ConflictSystemIntegration

# Define DB_DSN
DB_DSN = os.getenv("DB_DSN")

# Configure logging
logger = logging.getLogger(__name__)

# Models for input/output
class NarrativeContext(BaseModel):
    user_id: int
    conversation_id: int
    user_input: str
    current_location: str = "Unknown"
    time_of_day: str = "Morning"
    aggregator_data: dict = Field(default_factory=dict)
    npc_responses: list = Field(default_factory=list)
    time_result: dict = Field(default_factory=dict)

class NarrativeResponse(BaseModel):
    message: str
    generate_image: bool = False
    image_prompt: str = ""
    tension_level: int = 0
    stat_changes: dict = Field(default_factory=dict)
    environment_update: dict = Field(default_factory=dict)

class NPCResponse(BaseModel):
    npc_id: int
    npc_name: str
    response: str
    stat_changes: dict = Field(default_factory=dict)

class TimeAdvancement(BaseModel):
    time_advanced: bool = False
    would_advance: bool = False
    periods: int = 0
    current_time: str = ""
    confirm_needed: bool = False
    new_time: dict = Field(default_factory=dict)

class UniversalUpdateInput(BaseModel):
    user_id: int
    conversation_id: int
    narrative: str
    roleplay_updates: dict = Field(default_factory=dict)
    ChaseSchedule: dict = Field(default_factory=dict)
    MainQuest: str = None
    PlayerRole: str = None
    npc_creations: list = Field(default_factory=list)
    npc_updates: list = Field(default_factory=list)
    character_stat_updates: dict = Field(default_factory=dict)
    relationship_updates: list = Field(default_factory=list)
    npc_introductions: list = Field(default_factory=list)
    location_creations: list = Field(default_factory=list)
    event_list_updates: list = Field(default_factory=list)
    inventory_updates: dict = Field(default_factory=dict)
    quest_updates: list = Field(default_factory=list)
    social_links: list = Field(default_factory=list)
    perk_unlocks: list = Field(default_factory=list)
    activity_updates: list = Field(default_factory=list)
    journal_updates: list = Field(default_factory=list)
    image_generation: dict = Field(default_factory=dict)

class StorytellerAgent:
    """
    Enhanced Storyteller Agent with integrated context management
    
    This agent orchestrates storytelling and player interactions with complete integration 
    into both Nyx's central governance system and the comprehensive context system.
    """
    
    def __init__(self):
        """Initialize the Storyteller Agent with enhanced context integration"""
        # Each sub-agent acknowledges Nyx governance in their instructions
        self.npc_handler = Agent(
            name="NPCHandler",
            instructions="""
            You manage NPC interactions and responses in a roleplaying game with subtle femdom elements,
            under the governance of Nyx's central system.
            
            Generate realistic, character-appropriate reactions to player actions that maintain their personalities.
            Include subtle hints of control and dominance in their responses without being overt.
            
            Follow all directives issued by Nyx and operate within the governance framework.
            """,
            output_type=list[NPCResponse],
            tools=[
                function_tool(self.get_nearby_npcs)
            ]
        )
        
        self.time_manager = Agent(
            name="TimeManager",
            instructions="""
            You manage time advancement in a roleplaying game under the governance of Nyx.
            
            Determine if player actions should advance time, and by how much.
            Consider the type and duration of activities when making this determination.
            
            Follow all directives issued by Nyx and operate within the governance framework.
            """,
            output_type=TimeAdvancement,
            tools=[
                function_tool(self.get_current_game_time)
            ]
        )
        
        self.narrator = Agent(
            name="Narrator",
            instructions="""
            You are Nyx, the mysterious narrator for a roleplaying game with subtle femdom undertones.
            Your voice should blend velvet darkness and subtle dominance—intimate yet commanding.
            
            Create immersive, atmospheric responses to player actions that:
            1. Acknowledge and respond to the player's input
            2. Incorporate relevant NPC responses
            3. Advance the narrative in a way that subtly guides the player
            4. Maintain the balance between mundane daily life and subtle control dynamics
            
            Use a writing style that is:
            - Rich with sensory details and atmosphere
            - Subtly leading and suggestive
            - Intimate and personal, using "you" to address the player
            - Maintaining an undercurrent of control beneath a friendly facade
            
            You operate within your own governance framework, ensuring all narration aligns with your central directives.
            """,
            output_type=NarrativeResponse,
            tools=[
                function_tool(self.check_for_addiction_status),
                function_tool(self.check_relationship_crossroads)
            ]
        )
        
        self.universal_updater = Agent(
            name="UniversalUpdater",
            instructions="""
            You process narrative text and generate appropriate updates to the game state,
            under the governance of Nyx's central system.
            
            Extract meaningful changes to NPCs, locations, player stats, and other game elements.
            Focus on creating structured outputs that accurately reflect narrative developments.
            
            Follow all directives issued by Nyx and operate within the governance framework.
            """,
            output_type=UniversalUpdateInput,
            tools=[
                function_tool(self.get_comprehensive_context)
            ]
        )
        
        # Main coordinating agent with integrated governance tools
        self.agent = Agent(
            name="NyxStoryManager",
            instructions="""
            You are the centralized Story Manager, fully integrated with and operating under
            the governance of Nyx - the central oversight intelligence.
            
            Process player input, generate appropriate responses, and manage game state updates,
            all in accordance with Nyx's directives and governance policies.
            
            Your job is to:
            1. Handle player input and determine the appropriate processing while adhering to Nyx's oversight
            2. Coordinate NPC responses to player actions with explicit governance permission
            3. Manage time advancement based on player activities in compliance with Nyx's temporal policies
            4. Generate narrative responses through the Narrator under strict governance constraints
            5. Update the game state through the Universal Updater with Nyx's approval
            6. Ensure all actions comply with Nyx's governance directives and restrictions
            7. Report all significant actions and outcomes back to Nyx's central governance
            8. Process and implement directives issued by Nyx, adapting operations accordingly
            
            Maintain a balance between player agency and subtle narrative guidance,
            always prioritizing compliance with Nyx's governance framework.
            """,
            tools=[
                function_tool(self.get_comprehensive_context),
                function_tool(self.get_narratively_relevant_content),
                function_tool(self.get_summarized_memories),
                function_tool(self.process_npc_responses),
                function_tool(self.process_time_advancement),
                function_tool(self.generate_narrative_response),
                function_tool(self.apply_universal_updates),
                function_tool(self.generate_image_if_needed),
                function_tool(self.store_message),
                function_tool(self.check_governance_permission),
                function_tool(self.report_action_to_governance),
                function_tool(self.process_governance_directive),
                function_tool(self.create_memory_for_nyx),
                function_tool(self.retrieve_memories_for_nyx)
            ]
        )
        
        # Directive handler for processing Nyx directives
        self.directive_handler = None
        
        # Context version tracking
        self.last_context_version = None
        
        # Narrative manager for progressive summarization
        self.narrative_manager = None
        
        # Performance monitor
        self.performance_monitor = None
        
        # Store previous inputs for continuity
        self.previous_inputs = []
        
        # Cache for contextually relevant data
        self.context_cache = {}
        
        # Last retrieval timestamp to know when to invalidate cache
        self.last_retrieval_time = 0
    
    async def initialize_directive_handler(self, user_id: int, conversation_id: int):
        """Initialize directive handler with comprehensive directive handling"""
        governance = await get_central_governance(user_id, conversation_id)
        self.directive_handler = DirectiveHandler(
            user_id=user_id,
            conversation_id=conversation_id,
            agent_type=AgentType.STORY_DIRECTOR,  # Using STORY_DIRECTOR for storyteller
            agent_id=f"storyteller_{conversation_id}",
            governance=governance  # pass the object here
        )
        
        # Register handlers for all directive types
        self.directive_handler.register_handler(
            DirectiveType.ACTION, 
            self.handle_action_directive
        )
        self.directive_handler.register_handler(
            DirectiveType.OVERRIDE,
            self.handle_override_directive
        )
        self.directive_handler.register_handler(
            DirectiveType.PROHIBITION,
            self.handle_prohibition_directive
        )
        self.directive_handler.register_handler(
            DirectiveType.SCENE,
            self.handle_scene_directive
        )
        
        # Start background processing of directives
        await self.directive_handler.start_background_processing()
        
        # Initialize narrative manager for progressive summarization
        self.narrative_manager = RPGNarrativeManager(
            user_id=user_id,
            conversation_id=conversation_id,
            db_connection_string=DB_DSN
        )
        await self.narrative_manager.initialize()
        
        # Initialize performance monitor
        self.performance_monitor = PerformanceMonitor.get_instance(user_id, conversation_id)
        
        # Subscribe to context changes
        context_manager = get_context_manager()
        context_manager.subscribe_to_changes("/narrative_stage", self.handle_narrative_stage_change)
        context_manager.subscribe_to_changes("/npcs", self.handle_npc_changes)
        
        # Initialize context service and memory manager for later use
        self.context_service = await get_context_service(user_id, conversation_id)
        self.memory_manager = await get_memory_manager(user_id, conversation_id)
        self.vector_service = await get_vector_service(user_id, conversation_id)
    
    async def handle_narrative_stage_change(self, changes: List[ContextDiff]):
        """Handle changes to the narrative stage"""
        logger.info(f"Narrative stage changed: {changes}")
        for change in changes:
            if change.operation in ("add", "replace"):
                stage_info = change.value
                if isinstance(stage_info, dict) and "name" in stage_info:
                    # Create a memory about the stage change
                    ctx = type('obj', (object,), {'context': {
                        'user_id': self.directive_handler.user_id, 
                        'conversation_id': self.directive_handler.conversation_id
                    }})
                    
                    await self.create_memory_for_nyx(
                        ctx,
                        f"Narrative stage progressed to {stage_info['name']}",
                        "narrative_progression",
                        0.9  # Very high importance for stage changes
                    )
                    
                    # Add to narrative manager for progressive summarization
                    if self.narrative_manager:
                        await self.narrative_manager.add_revelation(
                            content=f"The narrative has progressed to the {stage_info['name']} stage: {stage_info['description']}",
                            revelation_type="stage_progression",
                            importance=0.9,
                            tags=["narrative_stage", "progression", stage_info['name'].lower().replace(" ", "_")]
                        )
                    
                    # Report to governance
                    governance = await get_central_governance(
                        self.directive_handler.user_id, 
                        self.directive_handler.conversation_id
                    )
                    await governance.process_agent_action_report(
                        agent_type=AgentType.STORY_DIRECTOR,
                        agent_id="director",
                        action={"type": "narrative_stage_change"},
                        result={"new_stage": stage_info["name"]}
                    )
                    
                    # Invalidate all context caches to ensure fresh data with new stage
                    self.context_cache = {}
                    self.last_retrieval_time = 0
                    context_cache.invalidate()
    
    async def handle_npc_changes(self, changes: List[ContextDiff]):
        """Handle changes to NPCs"""
        logger.info(f"NPC data changed: {changes}")
        
        significant_changes = False
        npc_names = []
        details = []
        
        for change in changes:
            if isinstance(change.value, dict) and "npc_name" in change.value:
                npc_names.append(change.value["npc_name"])
                
                # Detect significant changes that should create memories
                if change.operation == "add":
                    details.append(f"NPC {change.value['npc_name']} was added")
                    significant_changes = True
                elif change.operation == "replace":
                    # Check for significant stat changes
                    if (change.old_value and isinstance(change.old_value, dict) and 
                        isinstance(change.value, dict)):
                        for key in ["dominance", "cruelty", "closeness", "trust", "respect", "intensity"]:
                            if (key in change.value and key in change.old_value and 
                                abs(change.value.get(key, 0) - change.old_value.get(key, 0)) > 10):
                                details.append(f"{change.value['npc_name']}'s {key} changed significantly")
                                significant_changes = True
                                break
                        
                        # Location change
                        if (change.value.get("current_location") != change.old_value.get("current_location") and
                            change.value.get("current_location") and change.old_value.get("current_location")):
                            details.append(f"{change.value['npc_name']} moved from {change.old_value.get('current_location')} to {change.value.get('current_location')}")
                            significant_changes = True
        
        # Create a memory for significant changes
        if significant_changes and npc_names:
            ctx = type('obj', (object,), {'context': {
                'user_id': self.directive_handler.user_id, 
                'conversation_id': self.directive_handler.conversation_id
            }})
            
            memory_content = f"NPC changes detected: {'; '.join(details)}"
            
            await self.create_memory_for_nyx(
                ctx,
                memory_content,
                "npc_change",
                0.7
            )
            
            # Add to narrative manager for progressive summarization
            if self.narrative_manager:
                await self.narrative_manager.add_interaction(
                    content=memory_content,
                    importance=0.6,
                    tags=["npc_change"] + [name.lower().replace(" ", "_") for name in npc_names]
                )
    
    async def handle_action_directive(self, directive: dict) -> dict:
        """Handle an action directive from Nyx"""
        instruction = directive.get("instruction", "")
        logging.info(f"[StoryTeller] Processing action directive: {instruction}")
        
        # Handle different instructions
        if "generate response" in instruction.lower():
            # Extract parameters
            params = directive.get("parameters", {})
            user_input = params.get("user_input", "")
            if not user_input:
                return {"result": "error", "reason": "No user input provided"}
            
            # Create context
            ctx = type('obj', (object,), {'context': {'user_id': self.directive_handler.user_id, 'conversation_id': self.directive_handler.conversation_id, 'user_input': user_input}})
            
            # Get comprehensive context with vector search
            context_data = await self.get_comprehensive_context(ctx, user_input)
            
            # Generate narrative response
            narrative_response = await self.generate_narrative_response(
                ctx, user_input, context_data, [], None
            )
            
            return {
                "result": "response_generated", 
                "response": narrative_response.dict() if hasattr(narrative_response, "dict") else narrative_response
            }
        
        elif "advance time" in instruction.lower():
            # Extract parameters
            params = directive.get("parameters", {})
            activity_type = params.get("activity_type", "directive")
            confirm_advance = params.get("confirm_advance", True)
            
            # Create context
            ctx = type('obj', (object,), {'context': {'user_id': self.directive_handler.user_id, 'conversation_id': self.directive_handler.conversation_id}})
            
            # Process time advancement
            time_result = await self.process_time_advancement(
                ctx, activity_type, confirm_advance
            )
            
            return {
                "result": "time_advanced",
                "time_result": time_result.dict() if hasattr(time_result, "dict") else time_result
            }
        
        return {"result": "action_not_recognized"}
    
    async def handle_override_directive(self, directive: dict) -> dict:
        """Handle an override directive from Nyx"""
        logging.info(f"[StoryTeller] Processing override directive")
        
        # Extract override details
        override_action = directive.get("override_action", {})
        
        # Store override for future operations
        if not hasattr(self, "current_overrides"):
            self.current_overrides = {}
        
        directive_id = directive.get("id")
        if directive_id:
            self.current_overrides[directive_id] = override_action
        
        return {"result": "override_stored"}
    
    async def handle_prohibition_directive(self, directive: dict) -> dict:
        """Handle a prohibition directive from Nyx"""
        logging.info(f"[StoryTeller] Processing prohibition directive")
        
        # Extract prohibition details
        prohibited_actions = directive.get("prohibited_actions", [])
        reason = directive.get("reason", "No reason provided")
        
        # Store prohibition for future operations
        if not hasattr(self, "current_prohibitions"):
            self.current_prohibitions = {}
        
        directive_id = directive.get("id")
        if directive_id:
            self.current_prohibitions[directive_id] = {
                "prohibited_actions": prohibited_actions,
                "reason": reason
            }
        
        return {"result": "prohibition_stored"}
    
    async def handle_scene_directive(self, directive: dict) -> dict:
        """Handle a scene directive from Nyx"""
        logging.info(f"[StoryTeller] Processing scene directive")
        
        # Extract scene details
        location = directive.get("location")
        context = directive.get("context", {})
        
        if location:
            # Create context
            ctx = type('obj', (object,), {'context': {'user_id': self.directive_handler.user_id, 'conversation_id': self.directive_handler.conversation_id}})
            
            # Get NPCs for this location
            npcs = await self.get_nearby_npcs(ctx, location)
            
            # Record the scene setup
            await self.create_memory_for_nyx(
                ctx,
                f"Scene directive processed for location: {location}",
                "observation",
                6
            )
            
            return {
                "result": "scene_processed",
                "location": location,
                "npcs": npcs
            }
        
        return {"result": "invalid_scene_directive", "reason": "No location provided"}
    
    async def process_governance_directive(self, ctx, directive_type: str = None) -> dict:
        """
        Process any pending directives from Nyx governance.
        
        Args:
            directive_type: Optional specific directive type to process
            
        Returns:
            Results of directive processing
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        # Initialize directive handler if needed
        if not self.directive_handler:
            await self.initialize_directive_handler(user_id, conversation_id)
        
        # Process directives
        result = await self.directive_handler.process_directives(force_check=True)
        
        return result
    
    async def check_governance_permission(
        self,
        ctx,
        action_type: str,
        action_details: dict,
        agent_type: str = "storyteller"
    ) -> dict:
        """
        Check if an action is permitted by the governance system.
        
        Args:
            action_type: Type of action
            action_details: Details of the action
            agent_type: Type of agent (defaults to "storyteller")
            
        Returns:
            Dictionary with permission check results
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        try:
            # Get the governance system
            governance = await get_central_governance(user_id, conversation_id)
            
            # Check permission
            result = await governance.check_action_permission(
                agent_type=agent_type,
                agent_id=f"storyteller_{conversation_id}",
                action_type=action_type,
                action_details=action_details
            )
            
            return result
        except Exception as e:
            logger.error(f"Error checking governance permission: {e}")
            # Default to approved if there's an error
            return {
                "approved": True,
                "directive_applied": False,
                "reasoning": f"Error checking permission: {e}"
            }
    
    async def report_action_to_governance(
        self,
        ctx,
        action_type: str,
        action_description: str,
        result: dict,
        agent_type: str = "storyteller"
    ) -> dict:
        """
        Report an action and its result to the governance system.
        
        Args:
            action_type: Type of action
            action_description: Description of the action
            result: Result of the action
            agent_type: Type of agent (defaults to "storyteller")
            
        Returns:
            Dictionary with reporting results
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        try:
            # Get the governance system
            governance = await get_central_governance(user_id, conversation_id)
            
            # Report action
            report_result = await governance.process_agent_action_report(
                agent_type=agent_type,
                agent_id=f"storyteller_{conversation_id}",
                action={
                    "type": action_type,
                    "description": action_description
                },
                result=result
            )
            
            return report_result
        except Exception as e:
            logger.error(f"Error reporting action to governance: {e}")
            # Return basic success if there's an error
            return {
                "reported": True,
                "error": str(e)
            }
    
    async def create_memory_for_nyx(
        self,
        ctx,
        memory_text: str,
        memory_type: str = "observation",
        significance: int = 5
    ) -> dict:
        """
        Create a memory in Nyx's memory system.
        
        Args:
            memory_text: Text of the memory
            memory_type: Type of memory (observation, reflection, etc.)
            significance: Significance level (1-10)
            
        Returns:
            Result of memory creation
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        try:
            # Use both systems for memory storage - Nyx and context system
            # First, use Nyx's memory system
            nyx_result = await remember_with_governance(
                user_id=user_id,
                conversation_id=conversation_id,
                entity_type="storyteller",
                entity_id=f"storyteller_{conversation_id}",
                memory_text=memory_text,
                importance="medium" if significance <= 5 else "high",
                emotional=True,
                tags=[memory_type, "storyteller_memory"]
            )
            
            # Also add to context memory system
            memory_manager = await get_memory_manager(user_id, conversation_id)
            importance = significance / 10.0  # Convert 1-10 to 0.0-1.0
            
            memory_id = await memory_manager.add_memory(
                content=memory_text,
                memory_type=memory_type,
                importance=importance,
                tags=[memory_type, "storyteller_memory"],
                metadata={"source": "storyteller", "significance": significance}
            )
            
            # Also add to narrative summarizer for progressive summarization
            if self.narrative_manager:
                await self.narrative_manager.add_interaction(
                    content=memory_text,
                    importance=importance,
                    tags=[memory_type, "storyteller_memory"]
                )
            
            # Since we're creating a new memory, we should invalidate any cached contexts
            # that might need this new information
            self.context_cache = {}
            self.last_retrieval_time = 0
            
            return {
                "nyx_result": nyx_result,
                "memory_id": memory_id,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error creating memory: {e}")
            return {
                "error": str(e),
                "success": False
            }
    
    async def retrieve_memories_for_nyx(
        self,
        ctx,
        query: str = None,
        context_text: str = None,
        limit: int = 5
    ) -> dict:
        """
        Retrieve memories from Nyx's memory system.
        
        Args:
            query: Optional search query
            context_text: Optional context for retrieval
            limit: Maximum number of memories to retrieve
            
        Returns:
            Retrieved memories
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        try:
            # Try context system first for semantic search
            memory_manager = await get_memory_manager(user_id, conversation_id)
            context_memories = await memory_manager.search_memories(
                query_text=query or context_text or "",
                limit=limit,
                use_vector=True
            )
            
            # If we got good results from context system, use them
            if context_memories and len(context_memories) > 0:
                memory_dicts = []
                for memory in context_memories:
                    if hasattr(memory, 'to_dict'):
                        memory_dicts.append(memory.to_dict())
                    else:
                        memory_dicts.append(memory)
                        
                return {
                    "memories": memory_dicts,
                    "source": "context_system"
                }
            
            # Fallback to Nyx's memory system
            result = await recall_with_governance(
                user_id=user_id,
                conversation_id=conversation_id,
                entity_type="storyteller",
                entity_id=f"storyteller_{conversation_id}",
                query=query,
                context=context_text,
                limit=limit
            )
            
            return result
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return {
                "error": str(e),
                "memories": []
            }
    
    # Enhanced context retrieval methods
    
    @with_governance_permission(AgentType.STORY_DIRECTOR, "get_comprehensive_context")
    async def get_comprehensive_context(self, ctx, user_input: str = "") -> Dict[str, Any]:
        """
        Get comprehensive game context using the context system.
        
        Args:
            user_input: Current user input for relevance scoring
            
        Returns:
            Dictionary with comprehensive game context
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        # Check if we have a recent cached context to avoid redundant retrievals
        cache_key = f"comprehensive_context:{user_input[:50]}"
        current_time = time.time()
        
        # Cache context for 10 seconds to prevent redundant retrievals in the same turn
        if (cache_key in self.context_cache and 
            current_time - self.last_retrieval_time < 10):
            return self.context_cache[cache_key]
        
        # Start timer for performance tracking
        timer_id = None
        if self.performance_monitor:
            timer_id = self.performance_monitor.start_timer("get_comprehensive_context")
        
        try:
            # Get context service
            if not hasattr(self, 'context_service') or not self.context_service:
                self.context_service = await get_context_service(user_id, conversation_id)
            
            # Get configuration
            config = get_config()
            context_budget = config.get_token_budget("default")
            use_vector_search = config.is_enabled("use_vector_search")
            use_delta = config.is_enabled("use_delta_updates")
            
            # Get comprehensive context with optional delta tracking
            if self.last_context_version is not None and use_delta:
                context_result = await self.context_service.get_context(
                    input_text=user_input,
                    context_budget=context_budget,
                    use_vector_search=use_vector_search,
                    use_delta=True,
                    source_version=self.last_context_version
                )
                
                # Update version
                if "version" in context_result:
                    self.last_context_version = context_result["version"]
            else:
                # First retrieval - get full context
                context_result = await self.context_service.get_context(
                    input_text=user_input,
                    context_budget=context_budget,
                    use_vector_search=use_vector_search,
                    use_delta=False
                )
                
                # Store version for delta updates
                if "version" in context_result:
                    self.last_context_version = context_result["version"]
            
            # Format for compatibility with existing code
            formatted_context = format_context_for_compatibility(context_result)
            
            # Build aggregator text for easy use
            formatted_context["aggregator_text"] = build_aggregator_text(formatted_context)
            
            # Cache the context
            self.context_cache[cache_key] = formatted_context
            self.last_retrieval_time = current_time
            
            return formatted_context
        finally:
            # Stop timer
            if timer_id and self.performance_monitor:
                elapsed = self.performance_monitor.stop_timer(timer_id)
                logger.info(f"Comprehensive context retrieval took {elapsed:.3f}s")
    
    @with_governance_permission(AgentType.STORY_DIRECTOR, "get_aggregated_context")
    async def get_aggregated_context(self, ctx, conversation_id=None, player_name="Chase"):
        """
        Get the aggregated game context using the legacy system.
        
        Args:
            conversation_id: Optional conversation ID (defaults to context)
            player_name: Player name (defaults to "Chase")
            
        Returns:
            Dictionary with aggregated game context
        """
        user_id = ctx.context["user_id"]
        conv_id = conversation_id or ctx.context["conversation_id"]
        
        # Get current input from context if available for better relevance
        current_input = ctx.context.get("user_input", "")
        
        # Try to use the more comprehensive context system first
        try:
            context = await self.get_comprehensive_context(ctx, current_input)
            return context
        except Exception as e:
            logger.warning(f"Error getting comprehensive context, falling back to legacy system: {e}")
            
            # Fallback to legacy system
            aggregator_data = await get_aggregated_roleplay_context(
                user_id, 
                conv_id, 
                player_name,
                current_input=current_input
            )
            
            return aggregator_data
    
    @with_governance_permission(AgentType.STORY_DIRECTOR, "get_narratively_relevant_content")
    async def get_narratively_relevant_content(
        self, 
        ctx, 
        query_text: str, 
        narrative_stage: str = None
    ) -> List[Dict[str, Any]]:
        """
        Get content that's narratively relevant to the current situation using vector search.
        
        Args:
            query_text: Text to use for relevance matching
            narrative_stage: Optional narrative stage name for context
            
        Returns:
            List of relevant content items
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        # Cache key for this particular query
        cache_key = f"vector_search:{query_text[:50]}:{narrative_stage}"
        current_time = time.time()
        
        # Use cached results if recent (last 30 seconds)
        if (cache_key in self.context_cache and 
            current_time - self.last_retrieval_time < 30):
            return self.context_cache[cache_key]
        
        # Get vector service
        if not hasattr(self, 'vector_service') or not self.vector_service:
            self.vector_service = await get_vector_service(user_id, conversation_id)
        
        # If no vector search, return empty
        if not self.vector_service.enabled:
            return []
        
        # Enhance query with narrative stage if provided
        enhanced_query = query_text
        if narrative_stage:
            enhanced_query = f"{query_text} [Stage: {narrative_stage}]"
        
        # Get relevant items across multiple entity types
        results = await self.vector_service.search_entities(
            query_text=enhanced_query,
            entity_types=["npc", "memory", "narrative", "location"],
            top_k=5,
            hybrid_ranking=True
        )
        
        # Cache the results
        self.context_cache[cache_key] = results
        
        return results
    
    @with_governance_permission(AgentType.STORY_DIRECTOR, "get_summarized_memories")
    async def get_summarized_memories(
        self, 
        ctx,
        query_text: str = None,
        summary_level: int = None,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """
        Get automatically summarized memories based on query relevance.
        
        Args:
            query_text: Text to search for relevant memories
            summary_level: Summary level (0-3, default is auto-determined by age)
            max_tokens: Maximum tokens for all memories
            
        Returns:
            Dictionary with summarized memories
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        # Cache key for this query
        cache_key = f"summarized_memories:{query_text[:50]}:{summary_level}:{max_tokens}"
        current_time = time.time()
        
        # Use cached results if recent (last 20 seconds)
        if (cache_key in self.context_cache and 
            current_time - self.last_retrieval_time < 20):
            return self.context_cache[cache_key]
        
        # Use narrative manager's optimized context
        if not hasattr(self, 'narrative_manager') or not self.narrative_manager:
            self.narrative_manager = RPGNarrativeManager(
                user_id=user_id,
                conversation_id=conversation_id,
                db_connection_string=DB_DSN
            )
            await self.narrative_manager.initialize()
        
        if self.narrative_manager:
            try:
                optimal_context = await self.narrative_manager.get_optimal_narrative_context(
                    query=query_text or "",
                    max_tokens=max_tokens
                )
                
                # Cache the results
                self.context_cache[cache_key] = optimal_context
                
                return optimal_context
            except Exception as e:
                logger.warning(f"Error getting optimal narrative context: {e}")
        
        # Fallback to memory manager if narrative manager unavailable
        if not hasattr(self, 'memory_manager') or not self.memory_manager:
            self.memory_manager = await get_memory_manager(user_id, conversation_id)
        
        # Search for relevant memories
        memories = await self.memory_manager.search_memories(
            query_text=query_text or "",
            limit=10,
            use_vector=True
        )
        
        # Format for return
        memory_list = []
        for memory in memories:
            memory_dict = memory.to_dict() if hasattr(memory, "to_dict") else memory
            
            # Apply summarization based on age if level not specified
            if summary_level is None:
                # Determine summary level based on age
                age_days = (datetime.now() - memory.created_at).days
                
                if age_days < 3:  # Very recent
                    summary_level = SummaryLevel.DETAILED
                elif age_days < 7:  # Recent
                    summary_level = SummaryLevel.CONDENSED
                elif age_days < 30:  # Older
                    summary_level = SummaryLevel.SUMMARY
                else:  # Very old
                    summary_level = SummaryLevel.HEADLINE
            
            # Add summary level info
            memory_dict["summary_level"] = summary_level
            
            memory_list.append(memory_dict)
        
        result = {
            "memories": memory_list,
            "query": query_text,
            "total_memories": len(memory_list)
        }
        
        # Cache the results
        self.context_cache[cache_key] = result
        
        return result
    
    @with_governance_permission(AgentType.STORY_DIRECTOR, "get_nearby_npcs")
    async def get_nearby_npcs(self, ctx, location=None):
        """
        Get NPCs that are at the specified location.
        
        Args:
            location: Location to filter by (optional)
            
        Returns:
            List of nearby NPCs
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        # Try to get from comprehensive context first for better integration
        try:
            comprehensive_context = await self.get_comprehensive_context(ctx)
            npcs = comprehensive_context.get("introduced_npcs", [])
            
            # Filter by location if specified
            if location:
                nearby_npcs = [
                    npc for npc in npcs 
                    if npc.get("current_location") == location
                ]
            else:
                nearby_npcs = npcs[:5]  # Limit to 5 if no location filter
                
            # Format appropriately
            result = []
            for npc in nearby_npcs:
                formatted_npc = {
                    "npc_id": npc.get("npc_id"),
                    "npc_name": npc.get("npc_name"),
                    "current_location": npc.get("current_location"),
                    "dominance": npc.get("dominance"),
                    "cruelty": npc.get("cruelty"),
                }
                
                # Add archetypes and memories if available
                if "archetypes" in npc:
                    formatted_npc["archetypes"] = npc["archetypes"]
                else:
                    formatted_npc["archetypes"] = []
                
                if "memory" in npc:
                    formatted_npc["recent_memories"] = npc["memory"][:3] if isinstance(npc["memory"], list) else []
                else:
                    formatted_npc["recent_memories"] = []
                
                result.append(formatted_npc)
                
            # If we found NPCs, return them
            if result:
                return result
                
        except Exception as e:
            logger.warning(f"Error getting NPCs from context system: {e}, falling back to database")
        
        # Fallback to direct database query using the new async context manager
        async with get_db_connection_context() as conn:
            if location:
                rows = await conn.fetch("""
                    SELECT npc_id, npc_name, current_location, dominance, cruelty, 
                           archetypes, memory
                    FROM NPCStats
                    WHERE user_id=$1 AND conversation_id=$2 
                    AND current_location=$3
                    ORDER BY introduced DESC
                    LIMIT 5
                """, user_id, conversation_id, location)
            else:
                rows = await conn.fetch("""
                    SELECT npc_id, npc_name, current_location, dominance, cruelty,
                           archetypes, memory
                    FROM NPCStats
                    WHERE user_id=$1 AND conversation_id=$2
                    ORDER BY introduced DESC
                    LIMIT 5
                """, user_id, conversation_id)
            
            nearby_npcs = []
            for row in rows:
                try:
                    archetypes = json.loads(row["archetypes"]) if isinstance(row["archetypes"], str) else row["archetypes"] or []
                except (json.JSONDecodeError, TypeError):
                    archetypes = []
                    
                try:
                    memories = json.loads(row["memory"]) if isinstance(row["memory"], str) else row["memory"] or []
                except (json.JSONDecodeError, TypeError):
                    memories = []
                
                nearby_npcs.append({
                    "npc_id": row["npc_id"],
                    "npc_name": row["npc_name"],
                    "current_location": row["current_location"],
                    "dominance": row["dominance"],
                    "cruelty": row["cruelty"],
                    "archetypes": archetypes,
                    "recent_memories": memories[:3] if memories else []
                })
            
            return nearby_npcs
    
    @with_governance_permission(AgentType.STORY_DIRECTOR, "get_current_game_time")
    async def get_current_game_time(self, ctx):
        """
        Get the current game time.
        
        Returns:
            Tuple of (year, month, day, time_of_day)
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        # Try to get from context system first
        try:
            comprehensive_context = await self.get_comprehensive_context(ctx)
            
            # Extract time info from context
            time_info = comprehensive_context.get("time_info", {})
            current_roleplay = comprehensive_context.get("current_roleplay", {})
            
            if time_info:
                year = int(time_info.get("year", "1")) if time_info.get("year", "").isdigit() else 1
                month = int(time_info.get("month", "1")) if time_info.get("month", "").isdigit() else 1
                day = int(time_info.get("day", "1")) if time_info.get("day", "").isdigit() else 1
                time_of_day = time_info.get("time_of_day", "Morning")
                
                return year, month, day, time_of_day
                
        except Exception as e:
            logger.warning(f"Error getting time from context system: {e}, falling back to database")
        
        # Fallback to direct database query using the new async context manager
        async with get_db_connection_context() as conn:
            year, month, day, time_of_day = 1, 1, 1, "Morning"
            
            for key in ["CurrentYear", "CurrentMonth", "CurrentDay", "TimeOfDay"]:
                row = await conn.fetchrow("""
                    SELECT value
                    FROM CurrentRoleplay
                    WHERE user_id=$1 AND conversation_id=$2 AND key=$3
                """, user_id, conversation_id, key)
                
                if row:
                    if key == "CurrentYear":
                        year = int(row["value"]) if row["value"].isdigit() else 1
                    elif key == "CurrentMonth":
                        month = int(row["value"]) if row["value"].isdigit() else 1
                    elif key == "CurrentDay":
                        day = int(row["value"]) if row["value"].isdigit() else 1
                    elif key == "TimeOfDay":
                        time_of_day = row["value"]
            
            return year, month, day, time_of_day
    
    @with_governance(
        agent_type=AgentType.STORY_DIRECTOR,
        action_type="process_npc_responses",
        action_description="Processed NPC responses to player input"
    )
    async def process_npc_responses(self, ctx, user_input, activity_type="conversation", location=None):
        """
        Process NPC responses to player input with governance oversight.
        
        Args:
            user_input: Player's input text
            activity_type: Type of activity the player is performing
            location: Current location (optional)
            
        Returns:
            List of NPC responses
        """
        # Start timer for performance tracking
        timer_id = None
        if self.performance_monitor:
            timer_id = self.performance_monitor.start_timer("process_npc_responses")
        
        try:
            # If there's an override action from a directive, apply it
            if hasattr(self, "current_overrides"):
                for directive_id, override in self.current_overrides.items():
                    if "npc_responses" in override:
                        # Check if the override applies to this action
                        if override.get("for_input") == user_input or not override.get("for_input"):
                            logging.info(f"Applying NPC response override from directive {directive_id}")
                            return override["npc_responses"]
            
            # Get nearby NPCs
            nearby_npcs = await self.get_nearby_npcs(ctx, location)
            
            if not nearby_npcs:
                return []
            
            # Get context using comprehensive system
            context_data = await self.get_comprehensive_context(ctx, user_input)
            
            # Get relevant memories for each NPC
            for i, npc in enumerate(nearby_npcs):
                # Get memories for this NPC using vector search
                if not hasattr(self, 'memory_manager') or not self.memory_manager:
                    self.memory_manager = await get_memory_manager(
                        ctx.context["user_id"], 
                        ctx.context["conversation_id"]
                    )
                
                # Try to get memories related to this NPC with vector search
                npc_name = npc["npc_name"]
                npc_query = f"{npc_name} interaction with player"
                memories = await self.memory_manager.search_memories(
                    query_text=npc_query,
                    tags=[npc_name.lower().replace(" ", "_")],
                    limit=3,
                    use_vector=True
                )
                
                # Format memories
                memory_content = []
                for memory in memories:
                    if hasattr(memory, 'content'):
                        memory_content.append(memory.content)
                    elif isinstance(memory, dict) and 'content' in memory:
                        memory_content.append(memory['content'])
                
                # Add memories to NPC data
                nearby_npcs[i]["interaction_history"] = memory_content
            
            # Create prompt for the NPC handler
            prompt = f"""
            The player has input: "{user_input}"
            Activity type: {activity_type}
            Current location: {location or 'Unknown'}
            
            Generate appropriate responses for these NPCs:
            {json.dumps(nearby_npcs, indent=2)}
            
            Each response should:
            - Match the NPC's personality and stats (dominance, cruelty)
            - Relate to the player's input or activity
            - Include subtle hints of control where appropriate
            - Consider the NPC's recent memories and interaction history
            """
            
            # Run the NPC handler
            result = await Runner.run(
                self.npc_handler,
                prompt,
                context=ctx.context
            )
            
            npc_responses = result.final_output
            
            # Create a memory about this interaction
            if npc_responses:
                npc_names = ", ".join([npc.npc_name for npc in npc_responses])
                await self.create_memory_for_nyx(
                    ctx,
                    f"Player interaction with NPCs: {npc_names}. Input: '{user_input[:50]}...'",
                    "observation",
                    4
                )
            
            return npc_responses
        finally:
            # Stop timer
            if timer_id and self.performance_monitor:
                elapsed = self.performance_monitor.stop_timer(timer_id)
                logger.info(f"NPC response processing took {elapsed:.3f}s")
    
    @with_governance(
        agent_type=AgentType.STORY_DIRECTOR,
        action_type="process_time_advancement",
        action_description="Processed time advancement based on player activity"
    )
    async def process_time_advancement(self, ctx, activity_type="conversation", confirm_advance=False):
        """
        Process time advancement with governance oversight.
        
        Args:
            activity_type: Type of activity the player is performing
            confirm_advance: Whether to confirm time advancement
            
        Returns:
            Time advancement results
        """
        # Start timer for performance tracking
        timer_id = None
        if self.performance_monitor:
            timer_id = self.performance_monitor.start_timer("process_time_advancement")
                    if "time_advancement" in override:
                        logging.info(f"Applying time advancement override from directive {directive_id}")
                        return override["time_advancement"]
            
            user_id = ctx.context["user_id"]
            conversation_id = ctx.context["conversation_id"]
            
            # Get current time
            year, month, day, time_of_day = await self.get_current_game_time(ctx)
            
            # Consider context when determining time advancement
            context_data = await self.get_comprehensive_context(ctx)
            npc_count = len(context_data.get("introduced_npcs", []))
            
            # Create prompt for the time manager
            prompt = f"""
            Activity type: {activity_type}
            Current time: Year {year}, Month {month}, Day {day}, {time_of_day}
            Confirm advance: {confirm_advance}
            
            Current context:
            - {npc_count} NPCs present
            - Current location: {context_data.get("current_location", "Unknown")}
            
            Determine if this activity should advance time, and by how much.
            If confirm_advance is True, actually perform the time advancement.
            """
            
            # Run the time manager
            result = await Runner.run(
                self.time_manager,
                prompt,
                context=ctx.context
            )
            
            time_result = result.final_output
            
            # If time advanced and confirmed, update the database
            if time_result.time_advanced and confirm_advance:
                # NEW: Use LoreSystem instead of direct database writes
                from lore.lore_system import LoreSystem
                lore_system = await LoreSystem.get_instance(user_id, conversation_id)
                
                new_time = time_result.new_time
                new_year = new_time.get("year", year)
                new_month = new_time.get("month", month)
                new_day = new_time.get("day", day)
                new_time_of_day = new_time.get("time_of_day", time_of_day)
                
                # Update CurrentRoleplay with new time using LoreSystem
                time_updates = {
                    "CurrentYear": str(new_year),
                    "CurrentMonth": str(new_month),
                    "CurrentDay": str(new_day),
                    "TimeOfDay": new_time_of_day
                }
                
                for key, value in time_updates.items():
                    # Use LoreSystem to update each time component
                    await lore_system.propose_and_enact_change(
                        ctx=ctx,
                        entity_type="CurrentRoleplay",
                        entity_identifier={"user_id": user_id, "conversation_id": conversation_id, "key": key},
                        updates={"value": value},
                        reason=f"Time advancement: {activity_type} activity advanced time from {time_of_day} to {new_time_of_day}"
                    )
                
                # Create a memory about time advancement
                await self.create_memory_for_nyx(
                    ctx,
                    f"Time advanced from {time_of_day} on Day {day} to {new_time_of_day} on Day {new_day}",
                    "observation",
                    5
                )
                    
                    # Invalidate context cache since time changed
                    self.context_cache = {}
                    self.last_retrieval_time = 0
                    context_cache.invalidate("time_info")
                    
                    # If time advanced to a new day's morning, run maintenance
                    if new_time_of_day == "Morning" and new_day > day:
                        await nightly_maintenance(user_id, conversation_id)
                        logging.info("[next_storybeat] Ran nightly maintenance for day rollover.")
                        
                        # Create a memory about nightly maintenance
                        await self.create_memory_for_nyx(
                            ctx,
                            f"Night passed and a new day began. Nightly maintenance completed.",
                            "observation",
                            6
                        )
            
            return time_result
        finally:
            # Stop timer
            if timer_id and self.performance_monitor:
                elapsed = self.performance_monitor.stop_timer(timer_id)
                logger.info(f"Time advancement processing took {elapsed:.3f}s")
    
    @with_governance_permission(AgentType.STORY_DIRECTOR, "check_for_addiction_status")
    async def check_for_addiction_status(self, ctx):
        """
        Check for the player's addiction status.
        
        Returns:
            Dictionary with addiction information
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        player_name = ctx.context.get("player_name", "Chase")
        
        try:
            from logic.addiction_system import get_addiction_status
            status = await get_addiction_status(user_id, conversation_id, player_name)
            return status
        except Exception as e:
            logging.error(f"Error checking addiction status: {e}")
            return {"has_addictions": False}
    
    @with_governance_permission(AgentType.STORY_DIRECTOR, "check_relationship_crossroads")
    async def check_relationship_crossroads(self, ctx):
        """
        Check for relationship crossroads events.
        
        Returns:
            List of crossroads events
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        # Try to get from context system first
        try:
            comprehensive_context = await self.get_comprehensive_context(ctx)
            
            # Check if context already has crossroads info
            if "relationship_crossroads" in comprehensive_context:
                return comprehensive_context["relationship_crossroads"]
        except Exception as e:
            logger.warning(f"Error getting crossroads from context: {e}")
        
        # Fallback to direct database query using the new async context manager
        async with get_db_connection_context() as conn:
            rows = await conn.fetch("""
                SELECT link_id, entity1_type, entity1_id, entity2_type, entity2_id,
                       dynamics
                FROM SocialLinks
                WHERE user_id=$1 AND conversation_id=$2
                ORDER BY link_id
            """, user_id, conversation_id)
            
            crossroads = []
            for row in rows:
                dynamics = row["dynamics"]
                if dynamics:
                    try:
                        dyn_dict = json.loads(dynamics) if isinstance(dynamics, str) else dynamics
                        if dyn_dict.get("crossroads"):
                            crossroads.append({
                                "link_id": row["link_id"],
                                "entity1_type": row["entity1_type"],
                                "entity1_id": row["entity1_id"],
                                "entity2_type": row["entity2_type"],
                                "entity2_id": row["entity2_id"],
                                "crossroads": dyn_dict["crossroads"]
                            })
                    except (json.JSONDecodeError, TypeError):
                        pass
            
            return crossroads
    
    @with_governance(
        agent_type=AgentType.STORY_DIRECTOR,
        action_type="generate_narrative_response",
        action_description="Generated narrative response to player input"
    )
    async def generate_narrative_response(self, ctx, user_input, aggregator_data, npc_responses=None, time_result=None):
        """
        Generate a narrative response with governance oversight.
        
        Args:
            user_input: Player's input text
            aggregator_data: Aggregated game context
            npc_responses: List of NPC responses
            time_result: Time advancement results
            
        Returns:
            Narrative response
        """
        # Start timer for performance tracking
        timer_id = None
        if self.performance_monitor:
            timer_id = self.performance_monitor.start_timer("generate_narrative_response")
        
        try:
            # If there's an override action from a directive, apply it
            if hasattr(self, "current_overrides"):
                for directive_id, override in self.current_overrides.items():
                    if "narrative_response" in override:
                        # Check if the override applies to this input
                        if override.get("for_input") == user_input or not override.get("for_input"):
                            logging.info(f"Applying narrative response override from directive {directive_id}")
                            response_data = override["narrative_response"]
                            
                            # Convert to NarrativeResponse if it's a dict
                            if isinstance(response_data, dict):
                                try:
                                    return NarrativeResponse(**response_data)
                                except:
                                    # If conversion fails, continue with normal processing
                                    pass
            
            # Format NPC responses for context
            npc_response_text = ""
            if npc_responses:
                for resp in npc_responses:
                    npc_response_text += f"{resp.npc_name}: {resp.response}\n"
            
            # Format time result for context
            time_text = ""
            if time_result:
                time_text = f"Time {'advanced' if time_result.time_advanced else 'would advance'} to {time_result.current_time}"
                if time_result.new_time:
                    time_text += f" New time: Year {time_result.new_time.get('year')}, Month {time_result.new_time.get('month')}, Day {time_result.new_time.get('day')}, {time_result.new_time.get('time_of_day')}"
            
            # Check for addiction status
            addiction_status = await self.check_for_addiction_status(ctx)
            addiction_text = ""
            if addiction_status and addiction_status.get("has_addictions"):
                addiction_text = "Player has the following addictions:\n"
                for addiction, details in addiction_status.get("addictions", {}).items():
                    addiction_text += f"- {addiction}: Level {details.get('level', 0)} ({details.get('label', 'Unknown')})\n"
            
            # Check for relationship crossroads
            crossroads = await self.check_relationship_crossroads(ctx)
            crossroads_text = ""
            if crossroads:
                crossroads_text = "Current relationship crossroads:\n"
                for cr in crossroads:
                    crossroads_text += f"- Link {cr['link_id']}: {cr['crossroads'].get('name', 'Unnamed crossroads')}\n"
            
            # Get narratively relevant content using vector search
            relevant_content = await self.get_narratively_relevant_content(
                ctx,
                user_input,
                aggregator_data.get("narrative_stage", {}).get("name")
            )
            
            relevant_content_text = ""
            if relevant_content:
                relevant_content_text = "Narratively relevant content:\n"
                for item in relevant_content[:3]:  # Limit to top 3
                    entity_type = item.get("metadata", {}).get("entity_type", "unknown")
                    content = item.get("metadata", {}).get("content", "")
                    if content:
                        relevant_content_text += f"- {entity_type.capitalize()}: {content[:100]}...\n"
            
            # Get summarized memories relevant to the current input
            memory_context = await self.get_summarized_memories(
                ctx,
                query_text=user_input,
                max_tokens=1000
            )
            
            memory_text = ""
            if memory_context and "memories" in memory_context:
                memory_text = "Relevant memories:\n"
                for mem in memory_context.get("memories", [])[:3]:  # Limit to top 3
                    if isinstance(mem, dict) and "content" in mem:
                        memory_text += f"- {mem['content'][:100]}...\n"
            
            # Get token budget for narrative
            config = get_config()
            narrative_budget = config.get_token_budget("narrative") or 2000
            
            # Store current input for continuity
            self.previous_inputs.append(user_input)
            if len(self.previous_inputs) > 3:  # Keep last 3 inputs
                self.previous_inputs.pop(0)
            
            recent_inputs = "\n".join([f"- {input}" for input in self.previous_inputs[-3:]])
            
            # Build aggregator text
            context_summary = aggregator_data.get("aggregator_text", build_aggregator_text(aggregator_data))
            
            # Create prompt for the narrator
            prompt = f"""
            Player input: "{user_input}"
            
            Recent inputs:
            {recent_inputs}
            
            Game context:
            {context_summary}
            
            NPC Responses:
            {npc_response_text}
            
            Time:
            {time_text}
            
            {addiction_text}
            
            {crossroads_text}
            
            {memory_text}
            
            {relevant_content_text}
            
            Generate an immersive, atmospheric narrative response that:
            1. Acknowledges and responds to the player's input
            2. Incorporates relevant NPC responses
            3. Reflects any time advancement
            4. Maintains the subtle femdom tone and atmosphere
            5. Suggests whether an image should be generated for this scene
            
            Your response should blend velvet darkness and subtle dominance—intimate yet commanding.
            
            Please keep your response within approximately {narrative_budget} tokens.
            """
            
            # Run the narrator
            result = await Runner.run(
                self.narrator,
                prompt,
                context=ctx.context
            )
            
            narrative_response = result.final_output
            
            # Create a memory for this narrative response
            await self.create_memory_for_nyx(
                ctx,
                f"Generated narrative response to player input: '{user_input[:50]}...'",
                "observation",
                5
            )
            
            # Add to narrative manager for progressive summarization
            if self.narrative_manager:
                await self.narrative_manager.add_interaction(
                    content=f"Player: {user_input}\n\nNarrator: {narrative_response.message[:200]}...",
                    npc_name=None,
                    location=aggregator_data.get("current_location"),
                    importance=0.7,  # Medium-high importance
                    tags=["player_interaction", "narrative_response"]
                )
            
            return narrative_response
        finally:
            # Stop timer
            if timer_id and self.performance_monitor:
                elapsed = self.performance_monitor.stop_timer(timer_id)
                logger.info(f"Narrative response generation took {elapsed:.3f}s")
    
    @with_governance(
        agent_type=AgentType.STORY_DIRECTOR,
        action_type="apply_universal_updates",
        action_description="Applied universal updates based on narrative"
    )
    async def apply_universal_updates(self, ctx, narrative_response, additional_data=None):
        """
        Apply universal updates with governance oversight.
        
        Args:
            narrative_response: Narrative response object
            additional_data: Additional data for updates
            
        Returns:
            Update results
        """
        # Start timer for performance tracking
        timer_id = None
        if self.performance_monitor:
            timer_id = self.performance_monitor.start_timer("apply_universal_updates")
        
        try:
            user_id = ctx.context["user_id"]
            conversation_id = ctx.context["conversation_id"]
            
            # Create prompt for the universal updater
            comprehensive_context = await self.get_comprehensive_context(ctx)
            
            # Get relevant memories to use for update context
            memory_context = await self.get_summarized_memories(
                ctx,
                query_text=narrative_response.message,
                max_tokens=800
            )
            
            memory_text = ""
            if memory_context and "memories" in memory_context:
                memory_text = "Relevant memories:\n"
                for mem in memory_context.get("memories", [])[:3]:  # Limit to top 3
                    if isinstance(mem, dict) and "content" in mem:
                        memory_text += f"- {mem['content'][:100]}...\n"
            
            prompt = f"""
            Based on the following narrative, generate appropriate updates to the game state.
            
            Narrative:
            {narrative_response.message}
            
            {memory_text}
            
            Generate updates for:
            - Player stats
            - NPC updates
            - Time advancement
            - Location changes
            - Inventory updates
            - Relationship changes
            - And any other relevant game state changes
            
            Consider:
            - Changes in tone or power dynamics
            - Introduction of new characters or locations
            - Changes in character dynamics or relationships
            - Subtle shifts in control or influence
            """
            
            # Run the universal updater
            result = await Runner.run(
                self.universal_updater,
                prompt,
                context=ctx.context
            )
            
            update_data = result.final_output
            
            # NEW: Use LoreSystem for updates instead of direct database operations
            from lore.lore_system import LoreSystem
            lore_system = await LoreSystem.get_instance(user_id, conversation_id)
            
            # Process each type of update through LoreSystem
            update_results = {}
            
            # Character stat updates
            if update_data.character_stat_updates:
                for stat_name, new_value in update_data.character_stat_updates.items():
                    result = await lore_system.propose_and_enact_change(
                        ctx=ctx,
                        entity_type="PlayerStats",
                        entity_identifier={"user_id": user_id, "conversation_id": conversation_id, "player_name": "Chase"},
                        updates={stat_name: new_value},
                        reason=f"Narrative development: {narrative_response.message[:100]}..."
                    )
                    update_results[f"stat_{stat_name}"] = result
                
                
                # Create a memory for these updates
                update_summary = []
                successful_updates = [k for k, v in update_results.items() if v.get("status") == "committed"]
                if successful_updates:
                    update_summary.append(f"Applied {len(successful_updates)} updates successfully")
                
                summary_text = ", ".join(update_summary)
                await self.create_memory_for_nyx(
                    ctx,
                    f"Applied universal updates: {summary_text}",
                    "observation",
                    4
                )
                
                # Invalidate context cache to reflect updates
                self.context_cache = {}
                self.last_retrieval_time = 0
                context_cache.invalidate()
                
                # Reset context version to force full refresh next time
                self.last_context_version = None
                
                return update_result
        finally:
            # Stop timer
            if timer_id and self.performance_monitor:
                elapsed = self.performance_monitor.stop_timer(timer_id)
                logger.info(f"Universal updates took {elapsed:.3f}s")
    
    @with_governance(
        agent_type=AgentType.STORY_DIRECTOR,
        action_type="generate_image",
        action_description="Generated image based on narrative response"
    )
    async def generate_image_if_needed(self, ctx, narrative_response):
        """
        Generate an image for the scene if needed with governance oversight.
        
        Args:
            narrative_response: Narrative response object
            
        Returns:
            Image generation results
        """
        # Start timer for performance tracking
        timer_id = None
        if self.performance_monitor:
            timer_id = self.performance_monitor.start_timer("generate_image")
        
        try:
            # If image generation not requested, don't proceed
            if not narrative_response.generate_image:
                return {"generated": False}
            
            user_id = ctx.context["user_id"]
            conversation_id = ctx.context["conversation_id"]
            
            # Create image generation data
            scene_data = {
                "narrative": narrative_response.message,
                "image_generation": {
                    "generate": True,
                    "priority": "medium",
                    "focus": "balanced",
                    "framing": "medium_shot",
                    "reason": "Narrative moment"
                }
            }
            
            try:
                # Generate image
                image_result = await generate_roleplay_image_from_gpt(
                    scene_data,
                    user_id,
                    conversation_id
                )
                
                # Process image result
                if image_result and "image_urls" in image_result and image_result["image_urls"]:
                    image_url = image_result["image_urls"][0]
                    
                    # Create a memory for the image generation
                    await self.create_memory_for_nyx(
                        ctx,
                        f"Generated image for scene based on narrative",
                        "observation",
                        4
                    )
                    
                    result = {
                        "generated": True,
                        "image_url": image_url,
                        "prompt_used": image_result.get("prompt_used", "")
                    }
                else:
                    result = {"generated": False, "error": "No image generated"}
                
                return result
            except Exception as e:
                logging.error(f"Error generating image: {e}")
                return {"generated": False, "error": str(e)}
        finally:
            # Stop timer
            if timer_id and self.performance_monitor:
                elapsed = self.performance_monitor.stop_timer(timer_id)
                logger.info(f"Image generation took {elapsed:.3f}s")
    
    @with_governance_permission(AgentType.STORY_DIRECTOR, "store_message")
    async def store_message(self, ctx, sender, content):
        """
        Store a message in the database.
        
        Args:
            sender: Message sender
            content: Message content
            
        Returns:
            Status of the operation
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        # NEW: Use canon module for creating messages
        from lore.core import canon
        
        async with get_db_connection_context() as conn:
            # Create a canon function for messages if it doesn't exist
            # For now, we'll create the message directly through canon
            message_data = {
                "conversation_id": conversation_id,
                "sender": sender,
                "content": content,
                "created_at": datetime.now(timezone.utc)
            }
            
            # If there's no specific canon function for messages, create one or use generic
            await conn.execute("""
                INSERT INTO messages (conversation_id, sender, content)
                VALUES($1, $2, $3)
            """, conversation_id, sender, content)
            
            # Log this as a canonical event
            await canon.log_canonical_event(
                ctx, conn,
                f"Message stored from {sender}: {content[:50]}...",
                tags=["message", "conversation"],
                significance=3
            )
        
        return {"status": "stored"}
    
    @with_governance(
        agent_type=AgentType.STORY_DIRECTOR,
        action_type="process_story_beat",
        action_description="Processed complete story beat for player input"
    )
    async def process_story_beat(self, user_id, conversation_id, user_input, data=None):
        """
        Process a complete story beat with governance oversight.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            user_input: Player's input text
            data: Additional data
            
        Returns:
            Complete response with narrative and updates
        """
        tracker = PerformanceTracker("story_beat")
        tracker.start_phase("initialization")
        
        # Set up context
        context = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "user_input": user_input,
            "data": data or {}
        }
        
        tracker.end_phase()
        
        try:
            # 0. Get the governance system first
            tracker.start_phase("get_governance")
            governance = await get_central_governance(user_id, conversation_id)
            tracker.end_phase()
            
            # 1. Register with governance if needed
            tracker.start_phase("register_with_governance")
            if "storyteller" not in governance.registered_agents:
                await governance.register_agent(
                    agent_type="storyteller",
                    agent_instance=self,
                    agent_id=f"storyteller_{conversation_id}"
                )
            tracker.end_phase()
            
            # 2. Initialize directive handler if needed
            tracker.start_phase("initialize_directive_handler")
            if not self.directive_handler:
                await self.initialize_directive_handler(user_id, conversation_id)
            tracker.end_phase()
            
            # 3. Process any pending directives
            tracker.start_phase("process_directives")
            ctx = type('obj', (object,), {'context': {'user_id': user_id, 'conversation_id': conversation_id, 'user_input': user_input}})
            await self.process_governance_directive(ctx)
            tracker.end_phase()
            
            # 4. Store user message
            tracker.start_phase("store_user_message")
            await self.store_message(ctx, "user", user_input)
            tracker.end_phase()
            
            # 5. Get comprehensive context
            tracker.start_phase("get_context")
            comprehensive_context = await self.get_comprehensive_context(ctx, user_input)
            tracker.end_phase()

            # 5.5 Check for active conflicts and include in context
            tracker.start_phase("check_conflicts")
            conflict_system = await ConflictSystemIntegration.get_instance(user_id, conversation_id)
            active_conflicts = await conflict_system.get_conflicts_with_context(user_input)
            
            # Add conflict summaries to comprehensive context
            if active_conflicts:
                comprehensive_context["active_conflicts"] = []
                for conflict in active_conflicts[:3]:  # Limit to top 3 relevant conflicts
                    summary = await get_conflict_summary(user_id, conversation_id, conflict['conflict_id'])
                    comprehensive_context["active_conflicts"].append(summary)
            tracker.end_phase()
            
            # 6. Process NPC responses
            tracker.start_phase("npc_responses")
            current_location = comprehensive_context.get("current_location")
            npc_responses = await self.process_npc_responses(ctx, user_input, "conversation", current_location)
            tracker.end_phase()

            # 6.5 Process conflict stakeholder reactions
            tracker.start_phase("stakeholder_reactions")
            stakeholder_reactions = []
            if active_conflicts:
                for conflict in active_conflicts:
                    # Check if any NPCs in the scene are stakeholders
                    conflict_stakeholders = {s['npc_id']: s for s in conflict.get('stakeholders', [])}
                    scene_npcs = {npc['npc_id'] for npc in comprehensive_context.get('introduced_npcs', [])}
                    
                    # Find intersection
                    present_stakeholders = scene_npcs & set(conflict_stakeholders.keys())
                    
                    if present_stakeholders:
                        # These stakeholders might react to the player's input
                        for npc_id in present_stakeholders:
                            stakeholder = conflict_stakeholders[npc_id]
                            # Add stakeholder context to NPC responses
                            stakeholder_reactions.append({
                                'npc_id': npc_id,
                                'conflict_role': stakeholder.get('faction_position', 'Independent'),
                                'motivation': stakeholder.get('public_motivation', ''),
                                'conflict_name': conflict['conflict_name']
                            })
            
            # Merge stakeholder reactions into NPC responses
            if stakeholder_reactions:
                for i, npc_response in enumerate(npc_responses):
                    for reaction in stakeholder_reactions:
                        if npc_response.npc_id == reaction['npc_id']:
                            # Enhance the NPC response with conflict context
                            npc_responses[i].response += f"\n[{reaction['conflict_role']} in {reaction['conflict_name']}]"
            tracker.end_phase()
            
            # 7. Process time advancement
            tracker.start_phase("time_advancement")
            confirm_advance = data.get("confirm_time_advance", False) if data else False
            time_result = await self.process_time_advancement(ctx, "conversation", confirm_advance)
            tracker.end_phase()
            
            # 8. Generate narrative response
            tracker.start_phase("narrative_response")
            narrative_response = await self.generate_narrative_response(
                ctx,
                user_input,
                comprehensive_context,
                npc_responses,
                time_result
            )
            tracker.end_phase()

            # 8.5 Process conflict evolution based on player action
            tracker.start_phase("conflict_evolution")
            if active_conflicts:
                # Notify conflict system of player action
                await on_player_major_action(
                    user_id,
                    conversation_id,
                    "player_input",  # action type
                    {
                        "description": user_input,
                        "involved_npcs": [resp.npc_id for resp in npc_responses],
                        "location": current_location
                    }
                )
            tracker.end_phase()

            # 9. Apply emergent addiction analysis
            tracker.start_phase("emergent_addiction")
            try:
                # Prepare narrative/events for analysis (choose what to pass: user_input, context, narrative)
                emergent_events_text = f"{user_input}\n\n{narrative_response.message}"
        
                emergent_result = await analyze_and_apply_emergent_addictions(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    player_name=player_name if 'player_name' in locals() else "Chase",
                    recent_narrative=emergent_events_text,
                    npcs=comprehensive_context.get("introduced_npcs", [])
                )
        
                # Optionally, add to the response dict (this also allows you to debug or display new addictions in client)
                if emergent_result.get("applied_suggestions"):
                    response["addiction_emergence"] = emergent_result["applied_suggestions"]
                if emergent_result.get("update_results"):
                    response["addiction_emergence_results"] = emergent_result["update_results"]
                if emergent_result.get("player_addiction_status"):
                    response["addiction_status"] = emergent_result["player_addiction_status"]
            except Exception as e:
                logger.error(f"Emergent addiction error: {e}")
            tracker.end_phase()            
            
            # 10. Store assistant message
            tracker.start_phase("store_assistant_message")
            await self.store_message(ctx, "Nyx", narrative_response.message)
            tracker.end_phase()
            
            # 11. Apply universal updates
            tracker.start_phase("universal_updates")
            update_result = await self.apply_universal_updates(ctx, narrative_response)
            tracker.end_phase()
            
            # 12. Generate image if needed
            tracker.start_phase("image_generation")
            image_result = await self.generate_image_if_needed(ctx, narrative_response)
            tracker.end_phase()
            
            # 13. Report action summary to governance
            tracker.start_phase("report_to_governance")
            await self.report_action_to_governance(
                ctx,
                action_type="complete_story_beat",
                action_description=f"Processed complete story beat for input: {user_input[:50]}...",
                result={
                    "npc_responses": len(npc_responses),
                    "time_advanced": time_result.time_advanced if hasattr(time_result, "time_advanced") else False,
                    "image_generated": image_result.get("generated", False) if image_result else False,
                    "narrative_length": len(narrative_response.message),
                    "updates_applied": bool(update_result.get("success", False)) if update_result else False
                }
            )
            tracker.end_phase()
            
            # 14. Create a memory about this story beat
            tracker.start_phase("create_memory")
            await self.create_memory_for_nyx(
                ctx,
                f"Completed story beat processing for input: {user_input[:50]}...",
                "observation",
                5
            )
            tracker.end_phase()
            
            # 15. Build and return the final response
            tracker.start_phase("build_response")
            
            # Get performance metrics
            performance_metrics = self.performance_monitor.get_metrics()
            
            response = {
                "message": narrative_response.message,
                "tension_level": narrative_response.tension_level,
                "time_result": time_result.dict() if hasattr(time_result, "dict") else time_result,
                "confirm_needed": time_result.would_advance and not confirm_advance if hasattr(time_result, "would_advance") else False,
                "npc_responses": [resp.dict() for resp in npc_responses],
                "performance_metrics": {**tracker.get_metrics(), **performance_metrics},
                "governance_approved": True,
                "context_source": "integrated_system",
                "context_version": self.last_context_version
            }
            
            if hasattr(narrative_response, "environment_update") and narrative_response.environment_update:
                response["environment_update"] = narrative_response.environment_update
            
            if image_result and image_result.get("generated"):
                response["image"] = {
                    "image_url": image_result["image_url"],
                    "prompt_used": image_result.get("prompt_used", "")
                }
                
            tracker.end_phase()
            
            return response
            
        except Exception as e:
            if tracker.current_phase:
                tracker.end_phase()
            
            logging.exception("[process_story_beat] Error")
            
            # Create memory for error
            try:
                ctx = type('obj', (object,), {'context': {'user_id': user_id, 'conversation_id': conversation_id}})
                await self.create_memory_for_nyx(
                    ctx,
                    f"Error processing story beat: {str(e)}",
                    "error",
                    8
                )
            except:
                pass
            
            return {
                "error": str(e),
                "performance": tracker.get_metrics()
            }

# ----- Registration with Nyx Governance -----

async def register_with_governance(user_id: int, conversation_id: int) -> None:
    """
    Register the Storyteller Agent with the Nyx governance system.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
    """
    try:
        # Get the governance system
        governance = await get_central_governance(user_id, conversation_id)
        
        # Create the agent
        storyteller = get_storyteller()
        
        # Register with governance
        await governance.register_agent(
            agent_type="storyteller",
            agent_instance=storyteller,
            agent_id=f"storyteller_{conversation_id}"
        )
        
        # Initialize directive handler
        await storyteller.initialize_directive_handler(user_id, conversation_id)
        
        # Issue directive to process user messages
        await governance.issue_directive(
            agent_type="storyteller",
            agent_id=f"storyteller_{conversation_id}", 
            directive_type=DirectiveType.ACTION,
            directive_data={
                "instruction": "Process player messages and maintain narrative coherence",
                "scope": "storytelling"
            },
            priority=DirectivePriority.HIGH,
            duration_minutes=24*60  # 24 hours
        )
        
        logging.info(f"Storyteller registered with Nyx governance system for user {user_id}, conversation {conversation_id}")
    except Exception as e:
        logging.error(f"Error registering Storyteller with governance: {e}")

# ----- Helper Functions -----

# Singleton instance for the storyteller
_storyteller_instance = None

def get_storyteller():
    """
    Get (or create) the Nyx-integrated storyteller agent.
    
    Returns:
        The storyteller agent
    """
    global _storyteller_instance
    
    if _storyteller_instance is None:
        _storyteller_instance = StorytellerAgent()
    
    return _storyteller_instance

# For backward compatibility
def get_governed_storyteller():
    """
    Get (or create) the storyteller agent.
    For backward compatibility.
    
    Returns:
        The storyteller agent (same as get_storyteller)
    """
    return get_storyteller()
