# nyx/nyx_governance.py

"""
Unified governance system for Nyx to control all agents (NPCs and beyond).

This file provides a comprehensive governance system that handles:
1. Central authority over all agents (NPCs, story, specialized).
2. Permission checking for all actions (NPC or other).
3. Directive management for all agent types.
4. Action reporting, monitoring, and override capabilities.
5. NPC-specific and general-agent logic in one place.
6. Enhanced decision making with memory integration.
7. User preference adaptation.
8. Cross-agent coordination for narrative coherence.
9. Performance monitoring and feedback loops.
10. Temporal consistency enforcement.
"""

import logging
import json
import asyncio
import inspect
import importlib
import pkgutil
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set

import asyncpg

# The agent trace utility
from agents import trace

# Database connection helper
from db.connection import get_db_connection

# Memory system references
from memory.wrapper import MemorySystem
from nyx.nyx_memory_system import NyxMemorySystem

# Integration with LLM services for reasoning
from nyx.llm_integration import generate_text_completion, generate_reflection

# Caching utilities
from utils.caching import CACHE_TTL, NPC_DIRECTIVE_CACHE, AGENT_DIRECTIVE_CACHE

logger = logging.getLogger(__name__)


class DirectiveType:
    """Constants for directive types."""
    ACTION = "action"
    MOVEMENT = "movement"
    DIALOGUE = "dialogue"
    RELATIONSHIP = "relationship"
    EMOTION = "emotion"
    PROHIBITION = "prohibition"
    SCENE = "scene"
    OVERRIDE = "override"


class DirectivePriority:
    """Constants for directive priorities."""
    LOW = 1
    MEDIUM = 5
    HIGH = 8
    CRITICAL = 10


class AgentType:
    """Constants for agent types."""
    NPC = "npc"
    STORY_DIRECTOR = "story_director"
    CONFLICT_ANALYST = "conflict_analyst"
    NARRATIVE_CRAFTER = "narrative_crafter"
    RESOURCE_OPTIMIZER = "resource_optimizer"
    RELATIONSHIP_MANAGER = "relationship_manager"
    ACTIVITY_ANALYZER = "activity_analyzer"
    SCENE_MANAGER = "scene_manager"
    UNIVERSAL_UPDATER = "universal_updater"
    MEMORY_MANAGER = "memory_manager"


class NyxUnifiedGovernor:
    """
    Unified governance system for Nyx to control all agents (NPC and non-NPC).

    This class provides:
      1. Central authority over all agents (NPCs, StoryDirector, etc.).
      2. Permission checking for all agent actions.
      3. Directive management across all systems.
      4. Action reporting and monitoring.
      5. Global override capabilities.
      6. Inter-agent communication management.
      7. NPC-specific intervention logic merged with general agent logic.
      8. Database table setup for directives and action tracking.
      9. Access to memory system for logging, reflection, and context.
      10. Enhanced conflict resolution between competing directives.
      11. Feedback mechanisms for agent improvement.
      12. Dynamic agent discovery and registration.
      13. Centralized game state management across all agents.
      14. Temporal consistency enforcement for narrative coherence.
      15. Decision explanation systems for debugging and analysis.
      16. Memory-based decision enhancement.
      17. Multi-agent coordinated planning.
      18. Agent performance metrics and evaluation.
      19. User preference integration for personalized experiences.
    """

    def __init__(self, user_id: int, conversation_id: int):
        """
        Initialize the unified governance system.

        Args:
            user_id: The user/player ID
            conversation_id: The current conversation/scene ID
        """
        self.user_id = user_id
        self.conversation_id = conversation_id

        # Memory system will be lazily loaded
        self.memory_system: Optional[NyxMemorySystem] = None

        # Directive caches (one for NPC, one for non-NPC)
        self.npc_directive_cache = {}
        self.agent_directive_cache = {}

        # Track action history
        self.action_history = {}

        # Lock for concurrency
        self.lock = asyncio.Lock()

        # Track agent instances for direct communication
        self.registered_agents = {}

        # Sub-managers placeholders
        self._story_manager = None
        self._resource_manager = None
        self._relationship_manager = None
        
        # Initialize the centralized game state
        self.game_state = None

    # ---------------------------------------------------------------------
    # INITIALIZATION AND SETUP
    # ---------------------------------------------------------------------
    async def initialize(self):
        """Initialize all systems and prepare the governor for operation."""
        # Setup database tables
        await self.setup_database_tables()
        
        # Initialize game state
        await self.initialize_game_state()
        
        # Discover and register available agents
        await self.discover_and_register_agents()
        
        logger.info(f"NyxUnifiedGovernor initialized for user {self.user_id}, conversation {self.conversation_id}")
        
    async def setup_database_tables(self):
        """Set up necessary database tables for the governance system."""
        async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
            async with pool.acquire() as conn:
                # For NPC directives
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS NyxNPCDirectives (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER NOT NULL,
                        conversation_id INTEGER NOT NULL,
                        npc_id INTEGER NOT NULL,
                        directive JSONB NOT NULL,
                        expires_at TIMESTAMP WITH TIME ZONE,
                        priority INTEGER DEFAULT 5,
                        scene_id VARCHAR(50),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

                        CONSTRAINT npc_directives_user_conversation_fk
                            FOREIGN KEY (user_id, conversation_id)
                            REFERENCES conversations(user_id, id)
                            ON DELETE CASCADE
                    )
                """)
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_npc_directives_npc
                    ON NyxNPCDirectives(user_id, conversation_id, npc_id)
                """)
                
                # For non-NPC directives
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS NyxAgentDirectives (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER NOT NULL,
                        conversation_id INTEGER NOT NULL,
                        agent_type VARCHAR(50) NOT NULL,
                        agent_id VARCHAR(50) NOT NULL,
                        directive JSONB NOT NULL,
                        priority INTEGER DEFAULT 5,
                        expires_at TIMESTAMP WITH TIME ZONE,
                        scene_id VARCHAR(50),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

                        CONSTRAINT agent_directives_user_conversation_fk
                            FOREIGN KEY (user_id, conversation_id)
                            REFERENCES conversations(user_id, id)
                            ON DELETE CASCADE
                    )
                """)
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_agent_directives_agent
                    ON NyxAgentDirectives(user_id, conversation_id, agent_type, agent_id)
                """)

                # For action tracking (covers both NPC and non-NPC)
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS NyxActionTracking (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER NOT NULL,
                        conversation_id INTEGER NOT NULL,
                        agent_type VARCHAR(50),
                        agent_id VARCHAR(50),
                        npc_id INTEGER,
                        action_type VARCHAR(50),
                        action_data JSONB,
                        result_data JSONB,
                        status VARCHAR(20),
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

                        CONSTRAINT action_tracking_user_conversation_fk
                            FOREIGN KEY (user_id, conversation_id)
                            REFERENCES conversations(user_id, id)
                            ON DELETE CASCADE
                    )
                """)
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_action_tracking_agent
                    ON NyxActionTracking(user_id, conversation_id, agent_type, agent_id)
                """)
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_action_tracking_npc
                    ON NyxActionTracking(user_id, conversation_id, npc_id)
                """)
                
                # For tracking responses to directives
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS NyxDirectiveResponses (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER NOT NULL,
                        conversation_id INTEGER NOT NULL,
                        directive_id INTEGER NOT NULL,
                        response_data JSONB NOT NULL,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

                        CONSTRAINT directive_responses_user_conversation_fk
                            FOREIGN KEY (user_id, conversation_id)
                            REFERENCES conversations(user_id, id)
                            ON DELETE CASCADE
                    )
                """)
                
                # For decision logging and explanation
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS NyxDecisionLog (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER NOT NULL,
                        conversation_id INTEGER NOT NULL,
                        decision_type VARCHAR(50) NOT NULL,
                        original_data JSONB NOT NULL,
                        result_data JSONB NOT NULL,
                        reasoning TEXT,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

                        CONSTRAINT decision_log_user_conversation_fk
                            FOREIGN KEY (user_id, conversation_id)
                            REFERENCES conversations(user_id, id)
                            ON DELETE CASCADE
                    )
                """)
                
                # For agent feedback
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS NyxAgentFeedback (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER NOT NULL,
                        conversation_id INTEGER NOT NULL,
                        agent_type VARCHAR(50) NOT NULL,
                        agent_id VARCHAR(50) NOT NULL,
                        feedback_data JSONB NOT NULL,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

                        CONSTRAINT agent_feedback_user_conversation_fk
                            FOREIGN KEY (user_id, conversation_id)
                            REFERENCES conversations(user_id, id)
                            ON DELETE CASCADE
                    )
                """)
                
                # For game state tracking
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS NyxGameState (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER NOT NULL,
                        conversation_id INTEGER NOT NULL,
                        state JSONB NOT NULL,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

                        CONSTRAINT game_state_user_conversation_fk
                            FOREIGN KEY (user_id, conversation_id)
                            REFERENCES conversations(user_id, id)
                            ON DELETE CASCADE
                    )
                """)
                
                logger.info("Created NyxUnifiedGovernor database tables")

    # ---------------------------------------------------------------------
    # MEMORY SYSTEM
    # ---------------------------------------------------------------------
    async def get_memory_system(self) -> NyxMemorySystem:
        """
        Lazy-load the memory system.
        """
        if self.memory_system is None:
            self.memory_system = NyxMemorySystem(self.user_id, self.conversation_id)
        return self.memory_system

    # ---------------------------------------------------------------------
    # AGENT REGISTRATION AND DISCOVERY
    # ---------------------------------------------------------------------
    async def register_agent(self, agent_type: str, agent_instance: Any) -> None:
        """
        Register an agent with the governance system.

        Args:
            agent_type: The type of agent (use AgentType constants)
            agent_instance: The agent instance
        """
        self.registered_agents[agent_type] = agent_instance
        logger.info(f"Registered agent of type {agent_type}")

    async def discover_and_register_agents(self) -> Dict[str, Any]:
        """
        Scan for available agents and register them automatically with governance.
        
        Returns:
            Dictionary with registration results
        """
        # Define paths to scan for agent modules
        agent_module_paths = [
            "logic.npc_agents",
            "logic.story_agents",
            "logic.conflict_system",
            "logic.resource_management",
            "memory",
            "nyx"
        ]
        
        registration_results = {}
        
        # Load and inspect modules
        for module_path in agent_module_paths:
            try:
                registration_results[module_path] = await self._scan_module_for_agents(module_path)
            except ImportError:
                logger.warning(f"Could not import module path: {module_path}")
                registration_results[module_path] = {"error": "Module not found", "registered": 0}
        
        # Log registration activity
        await self._log_agent_registration_activity(registration_results)
        
        # Calculate summary
        total_registered = sum(
            result.get("registered", 0) 
            for result in registration_results.values()
        )
        
        logger.info(f"Dynamic agent discovery completed. {total_registered} agents registered.")
        
        return {
            "summary": {
                "total_registered": total_registered,
                "timestamp": datetime.now().isoformat()
            },
            "module_results": registration_results
        }

    async def _scan_module_for_agents(self, module_path: str) -> Dict[str, Any]:
        """
        Scan a module for agent classes and register them.
        
        Args:
            module_path: Dot-notation path to the module
            
        Returns:
            Dictionary with scan results
        """
        result = {
            "module": module_path,
            "found": 0,
            "registered": 0,
            "agents": []
        }
        
        try:
            # Import the module
            module = importlib.import_module(module_path)
            
            # Get all submodules
            if hasattr(module, "__path__"):
                submodule_info = pkgutil.iter_modules(module.__path__, module_path + ".")
                submodules = [importlib.import_module(name) for finder, name, ispkg in submodule_info]
            else:
                submodules = [module]
            
            # Scan each submodule for agent classes
            for submodule in submodules:
                for name, obj in inspect.getmembers(submodule):
                    # Check if it's a class and might be an agent
                    if inspect.isclass(obj) and (
                        "Agent" in name or 
                        hasattr(obj, "process_directive") or 
                        hasattr(obj, "handle_action")
                    ):
                        result["found"] += 1
                        
                        # Determine agent type
                        agent_type = self._determine_agent_type(obj, name)
                        
                        if agent_type:
                            # Create instance and register
                            try:
                                if agent_type == AgentType.NPC:
                                    # Special handling for NPCs
                                    continue  # NPCs are registered differently
                                else:
                                    # For other agent types
                                    agent_instance = obj(self.user_id, self.conversation_id)
                                    await self.register_agent(agent_type, agent_instance)
                                    
                                    result["registered"] += 1
                                    result["agents"].append({
                                        "name": name,
                                        "type": agent_type,
                                        "module": submodule.__name__
                                    })
                            except Exception as e:
                                logger.error(f"Error instantiating agent {name}: {e}")
        
            return result
        except Exception as e:
            logger.error(f"Error scanning module {module_path}: {e}")
            return {"module": module_path, "error": str(e), "registered": 0}

    def _determine_agent_type(self, agent_class, class_name: str) -> Optional[str]:
        """Determine the agent type from a class."""
        # Check for explicit type attribute
        if hasattr(agent_class, "AGENT_TYPE"):
            return getattr(agent_class, "AGENT_TYPE")
        
        # Check name-based heuristics
        if "NPC" in class_name or "Character" in class_name:
            return AgentType.NPC
        elif "Story" in class_name or "Narrative" in class_name or "Plot" in class_name:
            return AgentType.STORY_DIRECTOR
        elif "Conflict" in class_name:
            return AgentType.CONFLICT_ANALYST
        elif "Resource" in class_name:
            return AgentType.RESOURCE_OPTIMIZER
        elif "Relation" in class_name:
            return AgentType.RELATIONSHIP_MANAGER
        elif "Activity" in class_name or "Action" in class_name:
            return AgentType.ACTIVITY_ANALYZER
        elif "Scene" in class_name:
            return AgentType.SCENE_MANAGER
        elif "Memory" in class_name:
            return AgentType.MEMORY_MANAGER
        
        # Check method-based heuristics
        methods = [method_name for method_name, _ in inspect.getmembers(agent_class, predicate=inspect.isfunction)]
        
        if "process_story" in methods or "generate_narrative" in methods:
            return AgentType.STORY_DIRECTOR
        elif "analyze_conflict" in methods:
            return AgentType.CONFLICT_ANALYST
        elif "manage_resources" in methods:
            return AgentType.RESOURCE_OPTIMIZER
        elif "process_memory" in methods:
            return AgentType.MEMORY_MANAGER
        
        # No clear type detected
        return None

    async def _log_agent_registration_activity(self, registration_results: Dict[str, Any]) -> None:
        """Log agent registration activity."""
        memory_system = await self.get_memory_system()
        
        total_registered = sum(
            result.get("registered", 0) 
            for result in registration_results.values()
        )
        
        await memory_system.add_memory(
            memory_text=f"Performed dynamic agent discovery and registered {total_registered} agents",
            memory_type="system",
            memory_scope="game",
            significance=4,
            tags=["agent_registration", "system_activity"],
            metadata={
                "registration_results": registration_results,
                "timestamp": datetime.now().isoformat()
            }
        )

    # ---------------------------------------------------------------------
    # PERMISSION CHECK
    # ---------------------------------------------------------------------
    async def check_action_permission(
        self,
        agent_type: str,
        agent_id: Union[int, str],
        action_type: str,
        action_details: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Check if an agent (NPC or otherwise) is allowed to perform an action.

        - If it's an NPC (AgentType.NPC), apply NPC logic.
        - Otherwise, apply the logic for other agent types.

        Returns:
            Dict with fields like:
              - approved (bool)
              - directive_applied (bool)
              - override_action (optional dict)
              - reasoning (str)
              - tracking_id (int)
        """
        # Enhance action details with memories
        enhanced_details = await self.enhance_decision_with_memories(
            "permission_check", 
            {
                "agent_type": agent_type,
                "agent_id": agent_id,
                "action_type": action_type,
                **action_details
            }
        )
        
        # Apply user preferences
        enhanced_details = await self.apply_user_preferences(enhanced_details)
        
        # Check temporal consistency
        consistency_check = await self.ensure_temporal_consistency(enhanced_details, agent_type, agent_id)
        if not consistency_check["is_consistent"]:
            return {
                "approved": False,
                "directive_applied": False,
                "reasoning": f"Action fails temporal consistency check: {', '.join(consistency_check['time_issues'] + consistency_check['location_issues'] + consistency_check['causal_issues'])}",
                "suggestions": consistency_check["suggestions"],
                "tracking_id": -1
            }

        with trace(workflow_name=f"Agent {agent_type} Permission Check"):
            if agent_type == AgentType.NPC:
                # NPC logic
                directives = await self.get_npc_directives(int(agent_id))
                
                # Resolve any conflicts in directives
                if len(directives) > 1:
                    directives = await self.resolve_directive_conflicts(directives)

                response = {
                    "approved": True,
                    "directive_applied": False,
                    "override_action": None,
                    "reasoning": "No applicable directives found",
                    "tracking_id": await self._track_action_request_npc(
                        npc_id=int(agent_id),
                        action_type=action_type,
                        action_details=enhanced_details,
                        context=context
                    )
                }

                # Check for prohibitions first
                for directive in directives:
                    if directive.get("type") == DirectiveType.PROHIBITION:
                        prohibited_actions = directive.get("prohibited_actions", [])
                        if action_type in prohibited_actions or "*" in prohibited_actions:
                            response.update({
                                "approved": False,
                                "directive_applied": True,
                                "directive_id": directive.get("id"),
                                "reasoning": directive.get(
                                    "reason",
                                    "Action prohibited by Nyx directive"
                                )
                            })
                            # If there's an alternative action
                            if "alternative_action" in directive:
                                response["override_action"] = directive["alternative_action"]
                            return response

                # Check for override directives
                for directive in directives:
                    if directive.get("type") == DirectiveType.OVERRIDE:
                        applies_to = directive.get("applies_to", [])
                        if action_type in applies_to or "*" in applies_to:
                            response.update({
                                "approved": False,
                                "directive_applied": True,
                                "directive_id": directive.get("id"),
                                "override_action": directive.get("override_action"),
                                "reasoning": directive.get("reason", "Action overridden by Nyx directive")
                            })
                            return response

                # Check for action-specific directives
                for directive in directives:
                    if directive.get("type") == action_type:
                        response.update({
                            "approved": True,
                            "directive_applied": True,
                            "directive_id": directive.get("id"),
                            "action_modifications": directive.get("modifications", {}),
                            "reasoning": directive.get("reason", "Action modified by Nyx directive")
                        })
                        return response

                # No specific directives
                return response

            else:
                # Non-NPC logic
                # Get agent directives
                directives = await self.get_agent_directives(agent_type, agent_id)
                
                # Resolve any conflicts in directives
                if len(directives) > 1:
                    directives = await self.resolve_directive_conflicts(directives)

                response = {
                    "approved": True,
                    "directive_applied": False,
                    "override_action": None,
                    "reasoning": f"No applicable directives found for {agent_type}",
                    "tracking_id": await self._track_action_request_general(
                        agent_type=agent_type,
                        agent_id=agent_id,
                        action_type=action_type,
                        action_details=enhanced_details,
                        context=context
                    )
                }

                # Check for prohibitions first
                for directive in directives:
                    if directive.get("type") == DirectiveType.PROHIBITION:
                        prohibited_actions = directive.get("prohibited_actions", [])
                        if action_type in prohibited_actions or "*" in prohibited_actions:
                            response.update({
                                "approved": False,
                                "directive_applied": True,
                                "directive_id": directive.get("id"),
                                "reasoning": directive.get(
                                    "reason",
                                    f"Action prohibited by Nyx directive for {agent_type}"
                                )
                            })
                            # Check alternative
                            if "alternative_action" in directive:
                                response["override_action"] = directive["alternative_action"]
                            return response

                # Check for override directives
                for directive in directives:
                    if directive.get("type") == DirectiveType.OVERRIDE:
                        applies_to = directive.get("applies_to", [])
                        if action_type in applies_to or "*" in applies_to:
                            response.update({
                                "approved": False,
                                "directive_applied": True,
                                "directive_id": directive.get("id"),
                                "override_action": directive.get("override_action"),
                                "reasoning": directive.get(
                                    "reason",
                                    f"Action overridden by Nyx directive for {agent_type}"
                                )
                            })
                            return response

                # Check for action-specific directives
                for directive in directives:
                    if directive.get("type") == action_type:
                        response.update({
                            "approved": True,
                            "directive_applied": True,
                            "directive_id": directive.get("id"),
                            "action_modifications": directive.get("modifications", {}),
                            "reasoning": directive.get(
                                "reason",
                                f"Action modified by Nyx directive for {agent_type}"
                            )
                        })
                        return response

                # No specific directives
                return response

    # ---------------------------------------------------------------------
    # DIRECTIVE CONFLICT RESOLUTION
    # ---------------------------------------------------------------------
    async def resolve_directive_conflicts(self, directives: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Resolve conflicts between competing directives using narrative consistency, user preferences, 
        and gameplay goals.
        
        Args:
            directives: List of potentially conflicting directives
            
        Returns:
            List of resolved directives with conflicts addressed
        """
        if not directives or len(directives) <= 1:
            return directives
        
        # Sort by priority first
        sorted_directives = sorted(
            directives, 
            key=lambda d: d.get("priority", DirectivePriority.MEDIUM), 
            reverse=True
        )
        
        # Group directives by type for conflict detection
        directives_by_type = {}
        for directive in sorted_directives:
            directive_type = directive.get("type", "unknown")
            if directive_type not in directives_by_type:
                directives_by_type[directive_type] = []
            directives_by_type[directive_type].append(directive)
        
        # Check for and resolve conflicts within each type
        resolved_directives = []
        for directive_type, type_directives in directives_by_type.items():
            if len(type_directives) == 1:
                resolved_directives.append(type_directives[0])
                continue
            
            # Handle conflicts by directive type
            if directive_type == DirectiveType.ACTION:
                resolved = await self._resolve_action_conflicts(type_directives)
                resolved_directives.extend(resolved)
            elif directive_type == DirectiveType.MOVEMENT:
                resolved = await self._resolve_movement_conflicts(type_directives)
                resolved_directives.extend(resolved)
            elif directive_type == DirectiveType.DIALOGUE:
                resolved = await self._resolve_dialogue_conflicts(type_directives)
                resolved_directives.extend(resolved)
            elif directive_type == DirectiveType.PROHIBITION:
                resolved = await self._merge_prohibition_directives(type_directives)
                resolved_directives.extend(resolved)
            else:
                # Default: take the highest priority directive
                resolved_directives.append(type_directives[0])
        
        # Check cross-type conflicts (e.g., ACTION vs PROHIBITION)
        final_directives = await self._resolve_cross_type_conflicts(resolved_directives)
        
        # Add to decision log for transparency
        decision_id = await self._log_conflict_resolution(directives, final_directives)
        
        return final_directives

    async def _resolve_action_conflicts(self, action_directives: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Resolve conflicts between action directives."""
        # Get the narrative context
        narrative_data = await self._get_current_narrative_context()
        
        # If there's an active arc requiring specific actions, prioritize those
        if narrative_data.get("active_arcs"):
            for arc in narrative_data["active_arcs"]:
                for directive in action_directives:
                    if arc.get("name") in directive.get("source", ""):
                        # This directive comes from the active narrative arc
                        return [directive]
        
        # Otherwise check for agent synergy
        compatible_directives = []
        for i, directive in enumerate(action_directives):
            compatible = True
            for other in action_directives:
                if directive != other and not self._are_actions_compatible(directive, other):
                    compatible = False
                    break
            
            if compatible:
                compatible_directives.append(directive)
        
        if compatible_directives:
            return compatible_directives
        
        # If no compatible actions, take the highest priority one
        return [action_directives[0]]

    async def _resolve_movement_conflicts(self, movement_directives: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Resolve conflicts between movement directives."""
        # Can't move to multiple locations - take highest priority
        return [movement_directives[0]]

    async def _resolve_dialogue_conflicts(self, dialogue_directives: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Resolve conflicts between dialogue directives."""
        # Try to merge dialogue directives if possible
        if len(dialogue_directives) <= 2:
            try:
                # Create a merged directive that captures the intent of both
                merged = dialogue_directives[0].copy()
                
                # Extract dialogue content
                primary_content = dialogue_directives[0].get("content", "")
                secondary_content = dialogue_directives[1].get("content", "")
                
                # Use memory system to generate a merged dialogue
                merged_content = await self._generate_merged_dialogue(primary_content, secondary_content)
                
                merged["content"] = merged_content
                merged["merged_from"] = [d.get("id") for d in dialogue_directives]
                
                return [merged]
            except Exception as e:
                logger.error(f"Error merging dialogue directives: {e}")
        
        # Otherwise, take the highest priority one
        return [dialogue_directives[0]]

    async def _generate_merged_dialogue(self, primary_content: str, secondary_content: str) -> str:
        """Generate merged dialogue content from multiple sources."""
        prompt = f"""
        Merge these two dialogue directives into one cohesive response:
        
        Primary dialogue: "{primary_content}"
        
        Secondary dialogue: "{secondary_content}"
        
        Create a single dialogue that preserves the main intents and tone of both.
        """
        
        try:
            merged = await generate_text_completion(
                system_prompt="You are merging dialogue directives to create cohesive agent responses.",
                user_prompt=prompt,
                temperature=0.4,
                max_tokens=200
            )
            return merged
        except Exception as e:
            logger.error(f"Error generating merged dialogue: {e}")
            # Fallback: concatenate with transition
            return f"{primary_content} Additionally, {secondary_content}"

    async def _merge_prohibition_directives(self, prohibition_directives: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge multiple prohibition directives into a comprehensive one."""
        if len(prohibition_directives) == 1:
            return prohibition_directives
        
        # Create a merged prohibition directive
        merged = prohibition_directives[0].copy()
        prohibited_actions = set(merged.get("prohibited_actions", []))
        
        # Collect all prohibited actions
        for directive in prohibition_directives[1:]:
            actions = directive.get("prohibited_actions", [])
            prohibited_actions.update(actions)
        
        merged["prohibited_actions"] = list(prohibited_actions)
        merged["merged_from"] = [d.get("id") for d in prohibition_directives]
        
        return [merged]

    async def _resolve_cross_type_conflicts(self, directives: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Resolve conflicts between directives of different types."""
        # Find prohibition directives
        prohibition_directives = [d for d in directives if d.get("type") == DirectiveType.PROHIBITION]
        
        if not prohibition_directives:
            return directives
        
        # Check each directive against prohibitions
        final_directives = prohibition_directives.copy()
        
        for directive in directives:
            if directive.get("type") == DirectiveType.PROHIBITION:
                continue
            
            # Check if this directive violates any prohibition
            violates_prohibition = False
            for prohibition in prohibition_directives:
                prohibited_actions = prohibition.get("prohibited_actions", [])
                if directive.get("type") in prohibited_actions or "*" in prohibited_actions:
                    violates_prohibition = True
                    break
            
            if not violates_prohibition:
                final_directives.append(directive)
        
        return final_directives

    def _are_actions_compatible(self, action1: Dict[str, Any], action2: Dict[str, Any]) -> bool:
        """Check if two action directives are compatible."""
        # Extract action details
        action1_target = action1.get("target", "")
        action2_target = action2.get("target", "")
        action1_description = action1.get("description", "").lower()
        action2_description = action2.get("description", "").lower()
        
        # If actions target different entities, they're compatible
        if action1_target != action2_target:
            return True
        
        # Check for opposing actions
        opposing_pairs = [
            (["attack", "harm", "hurt"], ["protect", "defend", "save"]),
            (["leave", "exit", "depart"], ["stay", "remain", "wait"]),
            (["hide", "conceal"], ["reveal", "show"]),
            (["open"], ["close", "shut"]),
            (["increase", "raise"], ["decrease", "lower"]),
        ]
        
        for words1, words2 in opposing_pairs:
            if any(word in action1_description for word in words1) and any(word in action2_description for word in words2):
                return False
            if any(word in action2_description for word in words1) and any(word in action1_description for word in words2):
                return False
        
        return True

    async def _log_conflict_resolution(self, original_directives: List[Dict[str, Any]], resolved_directives: List[Dict[str, Any]]) -> int:
        """Log the directive conflict resolution process."""
        try:
            # Create a entry for the decision log
            async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                async with pool.acquire() as conn:
                    row = await conn.fetchrow("""
                        INSERT INTO NyxDecisionLog (
                            user_id, conversation_id, decision_type, 
                            original_data, result_data, timestamp
                        )
                        VALUES ($1, $2, $3, $4, $5, NOW())
                        RETURNING id
                    """,
                    self.user_id,
                    self.conversation_id,
                    "directive_conflict_resolution",
                    json.dumps(original_directives),
                    json.dumps(resolved_directives)
                    )
                    
                    return row["id"]
        except Exception as e:
            logger.error(f"Error logging conflict resolution: {e}")
            return -1

    # ---------------------------------------------------------------------
    # GET DIRECTIVES
    # ---------------------------------------------------------------------
    async def get_npc_directives(self, npc_id: int) -> List[Dict[str, Any]]:
        """
        Get all active directives for an NPC.
        """
        # Check cache first
        cache_key = f"directives:{self.user_id}:{self.conversation_id}:{npc_id}"
        cached = NPC_DIRECTIVE_CACHE.get(cache_key)
        if cached:
            return cached

        try:
            async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                async with pool.acquire() as conn:
                    rows = await conn.fetch("""
                        SELECT id, directive, priority, expires_at
                        FROM NyxNPCDirectives
                        WHERE user_id = $1
                          AND conversation_id = $2
                          AND npc_id = $3
                          AND expires_at > NOW()
                        ORDER BY priority DESC
                    """, self.user_id, self.conversation_id, npc_id)

                    directives = []
                    for row in rows:
                        directive_data = json.loads(row["directive"])
                        directive_data["id"] = row["id"]
                        directive_data["priority"] = row["priority"]
                        directive_data["expires_at"] = row["expires_at"].isoformat()
                        directives.append(directive_data)

                    # Cache the result
                    NPC_DIRECTIVE_CACHE.set(cache_key, directives, CACHE_TTL["directives"])
                    return directives

        except Exception as e:
            logger.error(f"Error fetching directives for NPC {npc_id}: {e}")
            return []

    async def get_agent_directives(
        self,
        agent_type: str,
        agent_id: Union[int, str]
    ) -> List[Dict[str, Any]]:
        """
        Get all active directives for a non-NPC agent.
        """
        # Check cache
        cache_key = f"directives:{self.user_id}:{self.conversation_id}:{agent_type}:{agent_id}"
        cached = AGENT_DIRECTIVE_CACHE.get(cache_key)
        if cached:
            return cached

        try:
            async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                async with pool.acquire() as conn:
                    rows = await conn.fetch("""
                        SELECT id, directive, priority, expires_at
                        FROM NyxAgentDirectives
                        WHERE user_id = $1
                          AND conversation_id = $2
                          AND agent_type = $3
                          AND agent_id = $4
                          AND expires_at > NOW()
                        ORDER BY priority DESC
                    """, self.user_id, self.conversation_id, agent_type, str(agent_id))

                    directives = []
                    for row in rows:
                        directive_data = json.loads(row["directive"])
                        directive_data["id"] = row["id"]
                        directive_data["priority"] = row["priority"]
                        directive_data["expires_at"] = row["expires_at"].isoformat()
                        directives.append(directive_data)

                    AGENT_DIRECTIVE_CACHE.set(cache_key, directives, CACHE_TTL["directives"])
                    return directives

        except Exception as e:
            logger.error(f"Error fetching directives for agent {agent_type}/{agent_id}: {e}")
            return []

    # ---------------------------------------------------------------------
    # ISSUE DIRECTIVE
    # ---------------------------------------------------------------------
    async def issue_directive(
        self,
        agent_type: str,
        agent_id: Union[int, str],
        directive_type: str,
        directive_data: Dict[str, Any],
        priority: int = DirectivePriority.MEDIUM,
        duration_minutes: int = 30,
        scene_id: str = None
    ) -> int:
        """
        Issue a new directive to any agent (NPC or non-NPC).
        """
        # Enhance directive with memory context
        enhanced_directive = await self.enhance_decision_with_memories(
            "directive_issuance",
            {
                "agent_type": agent_type,
                "agent_id": agent_id,
                "directive_type": directive_type,
                "directive_data": directive_data
            }
        )
        
        # Extract directive_data if it was wrapped
        if "directive_data" in enhanced_directive:
            directive_data = enhanced_directive["directive_data"]
            
        if agent_type == AgentType.NPC:
            # NPC path
            try:
                npc_id = int(agent_id)
                directive = {
                    "type": directive_type,
                    "timestamp": datetime.now().isoformat(),
                    **directive_data
                }
                async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                    async with pool.acquire() as conn:
                        row = await conn.fetchrow("""
                            INSERT INTO NyxNPCDirectives (
                                user_id, conversation_id, npc_id, directive,
                                expires_at, priority, scene_id
                            )
                            VALUES ($1, $2, $3, $4, NOW() + $5::INTERVAL, $6, $7)
                            RETURNING id
                        """,
                        self.user_id,
                        self.conversation_id,
                        npc_id,
                        json.dumps(directive),
                        f"{duration_minutes} minutes",
                        priority,
                        scene_id
                        )

                        directive_id = row["id"]
                        # Invalidate cache
                        cache_key = f"directives:{self.user_id}:{self.conversation_id}:{npc_id}"
                        NPC_DIRECTIVE_CACHE.delete(cache_key)

                        # Log directive
                        await self._log_directive_npc(npc_id, directive_id, directive_type, directive_data)
                        return directive_id

            except Exception as e:
                logger.error(f"Error issuing directive to NPC {agent_id}: {e}")
                return -1

        else:
            # Non-NPC path
            try:
                directive = {
                    "type": directive_type,
                    "timestamp": datetime.now().isoformat(),
                    **directive_data
                }
                async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                    async with pool.acquire() as conn:
                        row = await conn.fetchrow("""
                            INSERT INTO NyxAgentDirectives (
                                user_id, conversation_id, agent_type, agent_id, directive,
                                expires_at, priority, scene_id
                            )
                            VALUES ($1, $2, $3, $4, $5, NOW() + $6::INTERVAL, $7, $8)
                            RETURNING id
                        """,
                        self.user_id,
                        self.conversation_id,
                        agent_type,
                        str(agent_id),
                        json.dumps(directive),
                        f"{duration_minutes} minutes",
                        priority,
                        scene_id
                        )
                        directive_id = row["id"]

                        # Invalidate cache
                        cache_key = f"directives:{self.user_id}:{self.conversation_id}:{agent_type}:{agent_id}"
                        AGENT_DIRECTIVE_CACHE.delete(cache_key)

                        # Log directive
                        await self._log_directive_general(
                            agent_type, agent_id, directive_id, directive_type, directive_data
                        )
                        return directive_id

            except Exception as e:
                logger.error(f"Error issuing directive to agent {agent_type}/{agent_id}: {e}")
                return -1

    # ---------------------------------------------------------------------
    # REVOKE DIRECTIVE
    # ---------------------------------------------------------------------
    async def revoke_directive(self, directive_id: int, agent_type: str = None) -> bool:
        """
        Revoke a directive immediately.

        Args:
            directive_id: ID of the directive
            agent_type: Optional agent type to speed up lookups
        """
        if agent_type == AgentType.NPC:
            # NPC path
            return await self._revoke_directive_npc(directive_id)

        # If not sure or it's non-NPC, handle universal approach
        try:
            async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                async with pool.acquire() as conn:
                    # 1) If agent_type is None or unknown, we attempt to see if it's an NPC directive
                    if agent_type is None:
                        # Try NPC first
                        row_npc = await conn.fetchrow("""
                            SELECT npc_id FROM NyxNPCDirectives
                            WHERE id = $1 AND user_id = $2 AND conversation_id = $3
                        """, directive_id, self.user_id, self.conversation_id)

                        if row_npc:
                            # It's an NPC directive
                            npc_id = row_npc["npc_id"]
                            return await self._revoke_directive_npc(directive_id)

                        # Otherwise check agent directives
                        row_agent = await conn.fetchrow("""
                            SELECT agent_type, agent_id FROM NyxAgentDirectives
                            WHERE id = $1 AND user_id = $2 AND conversation_id = $3
                        """, directive_id, self.user_id, self.conversation_id)
                        if not row_agent:
                            return False

                        found_agent_type = row_agent["agent_type"]
                        found_agent_id = row_agent["agent_id"]

                        # Revoke in agent directives
                        result = await conn.execute("""
                            UPDATE NyxAgentDirectives
                            SET expires_at = NOW()
                            WHERE id = $1
                        """, directive_id)
                        # Invalidate cache
                        cache_key = f"directives:{self.user_id}:{self.conversation_id}:{found_agent_type}:{found_agent_id}"
                        AGENT_DIRECTIVE_CACHE.delete(cache_key)
                        return True

                    else:
                        # We have an agent_type that is presumably non-NPC
                        row_agent = await conn.fetchrow("""
                            SELECT agent_type, agent_id FROM NyxAgentDirectives
                            WHERE id = $1 AND user_id = $2 AND conversation_id = $3
                        """, directive_id, self.user_id, self.conversation_id)

                        if not row_agent:
                            return False

                        found_agent_type = row_agent["agent_type"]
                        found_agent_id = row_agent["agent_id"]

                        # Expire it
                        await conn.execute("""
                            UPDATE NyxAgentDirectives
                            SET expires_at = NOW()
                            WHERE id = $1
                        """, directive_id)

                        # Invalidate cache
                        cache_key = f"directives:{self.user_id}:{self.conversation_id}:{found_agent_type}:{found_agent_id}"
                        AGENT_DIRECTIVE_CACHE.delete(cache_key)
                        return True

        except Exception as e:
            logger.error(f"Error revoking directive {directive_id}: {e}")
            return False

    async def _revoke_directive_npc(self, directive_id: int) -> bool:
        """
        Internal helper to revoke an NPC directive.
        """
        try:
            async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                async with pool.acquire() as conn:
                    row = await conn.fetchrow("""
                        SELECT npc_id FROM NyxNPCDirectives
                        WHERE id = $1 AND user_id = $2 AND conversation_id = $3
                    """, directive_id, self.user_id, self.conversation_id)
                    if not row:
                        return False
                    npc_id = row["npc_id"]

                    # Expire
                    await conn.execute("""
                        UPDATE NyxNPCDirectives
                        SET expires_at = NOW()
                        WHERE id = $1
                    """, directive_id)

                    # Invalidate cache
                    cache_key = f"directives:{self.user_id}:{self.conversation_id}:{npc_id}"
                    NPC_DIRECTIVE_CACHE.delete(cache_key)
                    return True
        except Exception as e:
            logger.error(f"Error revoking directive {directive_id}: {e}")
            return False

    # ---------------------------------------------------------------------
    # PROCESS ACTION REPORT
    # ---------------------------------------------------------------------
    async def process_agent_action_report(
        self,
        agent_type: str,
        agent_id: Union[int, str],
        action: Dict[str, Any],
        result: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process an action report from any agent (NPC or otherwise)
        and determine if intervention is needed.
        """
        # First generate feedback
        feedback = await self.generate_agent_feedback(agent_type, agent_id, action, result)

        async with self.lock:
            if agent_type == AgentType.NPC:
                # NPC path
                tracking_id = await self._track_completed_action_npc(int(agent_id), action, result, context)
                memory_system = await self.get_memory_system()

                npc_name = await self._get_npc_name(int(agent_id))
                await memory_system.add_memory(
                    memory_text=f"{npc_name} performed: {action.get('description', 'unknown action')}",
                    memory_type="observation",
                    memory_scope="game",
                    significance=min(abs(result.get("emotional_impact", 0)) + 4, 10),
                    tags=["npc_action", f"npc_{agent_id}"],
                    metadata={
                        "npc_id": agent_id,
                        "action": action,
                        "result": result,
                        "tracking_id": tracking_id,
                        "feedback": feedback
                    }
                )

                # Check if intervention needed - use memory-enhanced check
                enhanced_action = await self.enhance_decision_with_memories(
                    "intervention_check",
                    {
                        "agent_type": agent_type,
                        "agent_id": agent_id,
                        "action": action,
                        "result": result
                    }
                )
                
                intervention_needed = await self._check_if_intervention_needed_npc(
                    int(agent_id), 
                    enhanced_action.get("action", action), 
                    enhanced_action.get("result", result)
                )
                
                # Also update game state with action result
                agent_key = f"npcs.{agent_id}"
                action_record = {
                    "last_action": {
                        "type": action.get("type", "unknown"),
                        "description": action.get("description", "unknown"),
                        "result": result.get("success", False),
                        "timestamp": datetime.now().isoformat()
                    }
                }
                await self.update_game_state(agent_key, action_record, agent_type, agent_id)
                
                if intervention_needed:
                    # Issue override directive
                    directive_id = await self.issue_directive(
                        agent_type=AgentType.NPC,
                        agent_id=agent_id,
                        directive_type=DirectiveType.OVERRIDE,
                        directive_data={
                            "reason": intervention_needed.get("reason"),
                            "override_action": intervention_needed.get("override_action"),
                            "applies_to": ["*"],
                            "source": "action_evaluation"
                        },
                        priority=DirectivePriority.HIGH,
                        duration_minutes=10
                    )
                    return {
                        "intervention": True,
                        "directive_id": directive_id,
                        "reason": intervention_needed.get("reason"),
                        "tracking_id": tracking_id
                    }

                return {
                    "intervention": False,
                    "tracking_id": tracking_id,
                    "feedback": feedback
                }

            else:
                # Non-NPC path
                tracking_id = await self._track_completed_action_general(agent_type, agent_id, action, result, context)
                memory_system = await self.get_memory_system()
                agent_identifier = await self._get_agent_identifier(agent_type, agent_id)

                # Add memory
                await memory_system.add_memory(
                    memory_text=f"{agent_identifier} performed: {action.get('description', 'unknown action')}",
                    memory_type="observation",
                    memory_scope="game",
                    significance=6,
                    tags=["agent_action", f"{agent_type}_{agent_id}"],
                    metadata={
                        "agent_type": agent_type,
                        "agent_id": agent_id,
                        "action": action,
                        "result": result,
                        "tracking_id": tracking_id,
                        "feedback": feedback
                    }
                )

                # Enhance with memory context
                enhanced_action = await self.enhance_decision_with_memories(
                    "intervention_check",
                    {
                        "agent_type": agent_type,
                        "agent_id": agent_id,
                        "action": action,
                        "result": result
                    }
                )
                
                # Decide if we intervene
                intervention_needed = await self._check_if_intervention_needed_general(
                    agent_type, 
                    agent_id, 
                    enhanced_action.get("action", action), 
                    enhanced_action.get("result", result)
                )
                
                # Update game state with action result
                agent_key = f"agents.{agent_type}.{agent_id}"
                action_record = {
                    "last_action": {
                        "type": action.get("type", "unknown"),
                        "description": action.get("description", "unknown"),
                        "result": result.get("success", False),
                        "timestamp": datetime.now().isoformat()
                    }
                }
                await self.update_game_state(agent_key, action_record, agent_type, agent_id)
                
                if intervention_needed:
                    directive_id = await self.issue_directive(
                        agent_type=agent_type,
                        agent_id=agent_id,
                        directive_type=DirectiveType.OVERRIDE,
                        directive_data={
                            "reason": intervention_needed.get("reason"),
                            "override_action": intervention_needed.get("override_action"),
                            "applies_to": ["*"],
                            "source": "action_evaluation"
                        },
                        priority=DirectivePriority.HIGH,
                        duration_minutes=10
                    )
                    return {
                        "intervention": True,
                        "directive_id": directive_id,
                        "reason": intervention_needed.get("reason"),
                        "tracking_id": tracking_id
                    }

                return {
                    "intervention": False,
                    "tracking_id": tracking_id,
                    "feedback": feedback
                }

    # ---------------------------------------------------------------------
    # NPC-SPECIFIC INTERVENTION LOGIC
    # ---------------------------------------------------------------------
    async def _check_if_intervention_needed_npc(
        self,
        npc_id: int,
        action: Dict[str, Any],
        result: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Determine if Nyx should intervene based on NPC action and its results.
        """
        emotional_impact = result.get("emotional_impact", 0)
        action_type = action.get("type", "unknown")
        target = action.get("target", "unknown")
        description = action.get("description", "")

        # Narrative context check
        narrative_context = await self._get_current_narrative_context()
        current_arcs = narrative_context.get("active_arcs", [])

        # 1. Contradiction with narrative arcs
        for arc in current_arcs:
            for npc_role in arc.get("npc_roles", []):
                if npc_role.get("npc_id") == npc_id:
                    required_relationship = npc_role.get("relationship")
                    required_action = npc_role.get("required_action")
                    # Relationship contradiction
                    if required_relationship == "friendly" and action_type in ["mock", "attack", "threaten"]:
                        return {
                            "reason": "Action contradicts NPC's friendly role in current narrative arc",
                            "override_action": {
                                "type": "talk",
                                "description": "speak in a more friendly manner",
                                "target": target
                            }
                        }
                    # Required action contradiction
                    if required_action == "help" and action_type == "leave":
                        return {
                            "reason": "NPC is required to help in current narrative arc",
                            "override_action": {
                                "type": "assist",
                                "description": "offer assistance instead of leaving",
                                "target": target
                            }
                        }

        # 2. Excessive emotional impact
        if abs(emotional_impact) > 7:
            return {
                "reason": "Action has excessive emotional impact",
                "override_action": {
                    "type": action_type,
                    "description": description.replace("forcefully", "moderately").replace("aggressively", "assertively"),
                    "target": target
                }
            }

        # 3. Coordinated group actions
        if target in ["group", "player"] and action_type in ["attack", "threaten", "dominate"]:
            # Check recent similar actions
            recent_actions = await self._get_recent_similar_actions_npc(action_type, target)
            if len(recent_actions) >= 2:
                return {
                    "reason": "Too many NPCs performing similar actions against the same target",
                    "override_action": {
                        "type": "observe",
                        "description": "wait and observe instead of joining the others",
                        "target": "environment"
                    }
                }

        # No intervention needed
        return None

    async def _get_recent_similar_actions_npc(self, action_type: str, target: str) -> List[Dict[str, Any]]:
        """
        Get recent similar actions by other NPCs.
        """
        try:
            async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                async with pool.acquire() as conn:
                    rows = await conn.fetch("""
                        SELECT action_data
                        FROM NyxActionTracking
                        WHERE user_id = $1
                          AND conversation_id = $2
                          AND action_data->>'type' = $3
                          AND action_data->>'target' = $4
                          AND timestamp > NOW() - INTERVAL '5 minutes'
                        ORDER BY timestamp DESC
                    """, self.user_id, self.conversation_id, action_type, target)
                    actions = [json.loads(row["action_data"]) for row in rows]
                    return actions
        except Exception as e:
            logger.error(f"Error getting recent similar actions: {e}")
            return []

    async def _check_memory_manager_intervention(self, agent_id, action, result):
        """
        Check if intervention is needed for memory manager actions.
        
        Args:
            agent_id: Agent ID
            action: Action performed
            result: Result of the action
        
        Returns:
            Intervention data or None
        """
        # Check for specific scenarios requiring intervention
        
        # 1. If creating potentially harmful memories
        if action.get("operation") == "remember" and "error" in result:
            return {
                "reason": f"Memory creation error: {result.get('error')}",
                "override_action": {
                    "type": "recover",
                    "description": "Recover from memory creation error",
                    "parameters": {"retry": True}
                }
            }
        
        # 2. If retrieving excessive memories
        if action.get("operation") == "recall" and len(result.get("memories", [])) > 20:
            return {
                "reason": "Excessive memory retrieval may impact performance",
                "override_action": {
                    "type": "optimize",
                    "description": "Limit memory retrieval to improve performance",
                    "parameters": {"max_memories": 20}
                }
            }
        
        # 3. If creating beliefs with extreme confidence
        if action.get("operation") == "create_belief" and action.get("confidence", 0) > 0.95:
            return {
                "reason": "Creating beliefs with excessive confidence",
                "override_action": {
                    "type": "moderate",
                    "description": "Reduce belief confidence to more reasonable level",
                    "parameters": {"max_confidence": 0.9}
                }
            }
        
        # No intervention needed
        return None

    async def _check_narrative_crafter_intervention(self, agent_id, action, result):
        """Check if intervention is needed for narrative crafter actions."""
        action_type = action.get("type", "unknown")
        
        # Check for lore inconsistencies
        if action_type == "generate_lore" and result.get("inconsistencies", []):
            return {
                "reason": "Inconsistencies detected in generated lore",
                "override_action": {
                    "type": "fix_inconsistencies",
                    "description": "Resolve detected lore inconsistencies",
                    "parameters": {"inconsistencies": result.get("inconsistencies", [])}
                }
            }
        
        # Check for unreasonable lore complexity
        if action_type == "generate_lore" and result.get("complexity_score", 0) > 8:
            return {
                "reason": "Generated lore is too complex for current context",
                "override_action": {
                    "type": "simplify_lore",
                    "description": "Simplify lore to more manageable complexity",
                    "parameters": {"max_complexity": 7}
                }
            }
        
        # Prevent overwhelming NPC knowledge
        if action_type == "integrate_lore_with_npcs" and result.get("average_knowledge_per_npc", 0) > 12:
            return {
                "reason": "Too much knowledge being assigned to NPCs",
                "override_action": {
                    "type": "limit_npc_knowledge",
                    "description": "Limit the amount of lore knowledge given to NPCs",
                    "parameters": {"max_knowledge_per_npc": 10}
                }
            }
        
        return None

    # ---------------------------------------------------------------------
    # GENERAL-AGENT INTERVENTION LOGIC
    # ---------------------------------------------------------------------
    async def _check_if_intervention_needed_general(
        self,
        agent_type: str,
        agent_id: Union[int, str],
        action: Dict[str, Any],
        result: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Determine if Nyx should intervene for non-NPC agents
        (story_director, conflict_analyst, etc.).
        """
        # Check memory analysis from enhancement
        if "memory_analysis" in action and "memory_supported_intervention" in action:
            if action["memory_supported_intervention"]:
                return {
                    "reason": f"Memory analysis suggests intervention: {action['memory_analysis']}",
                    "override_action": {
                        "type": "adjust",
                        "description": "Adjust action based on historical context",
                        "parameters": {"context_adjustment": True}
                    }
                }

        # For specialized agent types, check the relevant sub-method:
        intervention_rules = {
            AgentType.STORY_DIRECTOR: self._check_story_director_intervention,
            AgentType.CONFLICT_ANALYST: self._check_conflict_analyst_intervention,
            AgentType.NARRATIVE_CRAFTER: self._check_narrative_crafter_intervention,
            AgentType.RESOURCE_OPTIMIZER: self._check_resource_optimizer_intervention,
            AgentType.RELATIONSHIP_MANAGER: self._check_relationship_manager_intervention,
            AgentType.ACTIVITY_ANALYZER: self._check_activity_analyzer_intervention,
            AgentType.SCENE_MANAGER: self._check_scene_manager_intervention,
            AgentType.UNIVERSAL_UPDATER: self._check_universal_updater_intervention,
            AgentType.MEMORY_MANAGER: self._check_memory_manager_intervention
        }

        if agent_type in intervention_rules:
            return await intervention_rules[agent_type](agent_id, action, result)

        # Otherwise, generic logic
        return await self._check_generic_agent_intervention(agent_type, agent_id, action, result)

    # Sub-check methods:
    async def _check_story_director_intervention(self, agent_id, action, result):
        action_type = action.get("type", "unknown")
        # Prevent excessively rapid narrative progression
        if action_type == "advance_narrative" and result.get("progression_rate", 0) > 0.3:
            return {
                "reason": "Narrative is advancing too quickly",
                "override_action": {
                    "type": "slow_progression",
                    "description": "Reduce the rate of narrative advancement",
                    "parameters": {"max_progression_rate": 0.2}
                }
            }
        # Prevent too many conflicts
        if action_type == "generate_conflict" and result.get("active_conflicts", 0) > 3:
            return {
                "reason": "Too many active conflicts would overwhelm the player",
                "override_action": {
                    "type": "delay_conflict",
                    "description": "Delay conflict generation until existing conflicts are resolved",
                    "parameters": {"max_active_conflicts": 3}
                }
            }
        return None

    async def _check_conflict_analyst_intervention(self, agent_id, action, result):
        action_type = action.get("type", "unknown")
        
        # Check for balance issues in conflicts
        if action_type == "create_conflict" and result.get("difficulty_rating", 0) > 8:
            return {
                "reason": "Created conflict is too difficult for current player level",
                "override_action": {
                    "type": "adjust_difficulty",
                    "description": "Reduce conflict difficulty",
                    "parameters": {"max_difficulty": 7}
                }
            }
            
        # Prevent excessive conflict density
        if action_type == "add_conflict" and result.get("conflict_density", 0) > 0.7:
            return {
                "reason": "Scene has too many active conflicts simultaneously",
                "override_action": {
                    "type": "stage_conflicts",
                    "description": "Stage conflicts to trigger sequentially rather than simultaneously",
                    "parameters": {"max_simultaneous": 2}
                }
            }
            
        # Ensure conflict matches narrative tone
        if action_type in ["create_conflict", "modify_conflict"] and result.get("tone_mismatch", False):
            return {
                "reason": "Conflict tone doesn't match current narrative",
                "override_action": {
                    "type": "align_tone",
                    "description": "Adjust conflict to match narrative tone",
                    "parameters": {"preserve_core_challenge": True}
                }
            }
            
        return None

    async def _check_resource_optimizer_intervention(self, agent_id, action, result):
        # Prevent drastic resource changes
        if action.get("type") == "adjust_resources" and abs(result.get("money_change", 0)) > 500:
            return {
                "reason": "Resource adjustment is too extreme",
                "override_action": {
                    "type": "moderate_resource_change",
                    "description": "Apply a more moderate resource adjustment",
                    "parameters": {"max_change": 500}
                }
            }
            
        # Prevent resource inbalance
        if action.get("type") == "distribute_rewards" and result.get("reward_imbalance", 0) > 0.6:
            return {
                "reason": "Reward distribution is too imbalanced",
                "override_action": {
                    "type": "balance_rewards",
                    "description": "Distribute rewards more evenly",
                    "parameters": {"max_imbalance": 0.4}
                }
            }
            
        # Ensure economic stability
        if action.get("type") in ["set_prices", "adjust_economy"] and result.get("inflation_rate", 0) > 0.15:
            return {
                "reason": "Economic change would cause excessive inflation",
                "override_action": {
                    "type": "stabilize_economy",
                    "description": "Implement price controls to prevent inflation",
                    "parameters": {"target_inflation_rate": 0.05}
                }
            }
            
        return None

    async def _check_relationship_manager_intervention(self, agent_id, action, result):
        action_type = action.get("type", "unknown")
        
        # Prevent extreme relationship changes
        if action_type == "modify_relationship" and abs(result.get("change_magnitude", 0)) > 30:
            return {
                "reason": "Relationship change is too extreme for a single interaction",
                "override_action": {
                    "type": "gradual_change",
                    "description": "Implement relationship change gradually over multiple interactions",
                    "parameters": {"max_per_interaction": 15}
                }
            }
            
        # Ensure relationship changes align with character personalities
        if action_type == "create_relationship" and result.get("personality_mismatch", False):
            return {
                "reason": "Relationship doesn't align with established character personalities",
                "override_action": {
                    "type": "personality_aligned_relationship",
                    "description": "Adjust relationship to better match character personalities",
                    "parameters": {"preserve_core_dynamic": True}
                }
            }
            
        # Prevent contradicting established relationships
        if action_type in ["modify_relationship", "create_interaction"] and result.get("contradicts_history", False):
            return {
                "reason": "Action contradicts established relationship history",
                "override_action": {
                    "type": "consistent_interaction",
                    "description": "Modify interaction to be consistent with relationship history",
                    "parameters": {"reference_past_interactions": True}
                }
            }
            
        return None

    async def _check_activity_analyzer_intervention(self, agent_id, action, result):
        action_type = action.get("type", "unknown")
        
        # Prevent incorrect analysis
        if action_type == "analyze_pattern" and result.get("confidence", 1.0) < 0.6:
            return {
                "reason": "Pattern analysis has low confidence",
                "override_action": {
                    "type": "collect_more_data",
                    "description": "Gather more data before making conclusions",
                    "parameters": {"min_confidence": 0.7}
                }
            }
            
        # Ensure recommendations match player preferences
        if action_type == "recommend_activity" and result.get("preference_mismatch", 0) > 0.5:
            return {
                "reason": "Recommended activity doesn't match player preferences",
                "override_action": {
                    "type": "preference_aligned_recommendation",
                    "description": "Generate recommendation better aligned with player preferences",
                    "parameters": {"preference_alignment": True}
                }
            }
            
        return None

    async def _check_scene_manager_intervention(self, agent_id, action, result):
        action_type = action.get("type", "unknown")
        
        # Prevent scene pacing issues
        if action_type == "transition_scene" and result.get("pacing_too_fast", False):
            return {
                "reason": "Scene transition is happening too quickly",
                "override_action": {
                    "type": "add_transition_element",
                    "description": "Add transitional elements to smooth the scene change",
                    "parameters": {"gradual_transition": True}
                }
            }
            
        # Prevent inconsistent environments
        if action_type == "modify_environment" and result.get("inconsistency_detected", False):
            return {
                "reason": "Environmental change creates inconsistency with established setting",
                "override_action": {
                    "type": "consistent_environment_change",
                    "description": "Modify environment while maintaining continuity",
                    "parameters": {"respect_established_elements": True}
                }
            }
            
        # Ensure appropriate NPC density 
        if action_type == "populate_scene" and result.get("npc_density", 0) > 0.8:
            return {
                "reason": "Scene has too many NPCs for effective interaction",
                "override_action": {
                    "type": "optimize_npc_count",
                    "description": "Reduce number of active NPCs in scene",
                    "parameters": {"target_density": 0.6}
                }
            }
            
        return None

    async def _check_universal_updater_intervention(self, agent_id, action, result):
        action_type = action.get("type", "unknown")
        
        # Prevent large-scale changes without proper setup
        if action_type == "update_world_state" and not result.get("properly_foreshadowed", True):
            return {
                "reason": "World state change lacks proper narrative foreshadowing",
                "override_action": {
                    "type": "stage_world_change",
                    "description": "Create preliminary hints and events before major change",
                    "parameters": {"foreshadowing_required": True}
                }
            }
            
        # Ensure time progression is appropriate
        if action_type == "advance_time" and result.get("skips_important_events", False):
            return {
                "reason": "Time advancement would skip important narrative events",
                "override_action": {
                    "type": "incremental_time_advance",
                    "description": "Advance time in smaller increments to address important events",
                    "parameters": {"handle_skipped_events": True}
                }
            }
            
        return None

    async def _check_generic_agent_intervention(self, agent_type, agent_id, action, result):
        """
        Generic intervention logic for any agent type.
        """
        # Check for critical errors
        if result.get("error") and result.get("critical", False):
            return {
                "reason": f"Critical error in {agent_type} action: {result.get('error')}",
                "override_action": {
                    "type": "recover",
                    "description": "Attempt recovery from error state",
                    "parameters": {"reset_state": True}
                }
            }

        # Check for severe consequences
        if result.get("severity", 0) > 8:
            return {
                "reason": "Action has potentially severe consequences",
                "override_action": {
                    "type": "moderate",
                    "description": "Reduce severity of action outcome",
                    "parameters": {"max_severity": 7}
                }
            }
        return None

    # ---------------------------------------------------------------------
    # TRACKING ACTION REQUESTS/COMPLETION
    # ---------------------------------------------------------------------
    async def _track_action_request_npc(
        self,
        npc_id: int,
        action_type: str,
        action_details: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> int:
        """
        Track an action request for an NPC.
        """
        try:
            async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                async with pool.acquire() as conn:
                    row = await conn.fetchrow("""
                        INSERT INTO NyxActionTracking (
                            user_id, conversation_id, npc_id,
                            action_type, action_data, status, timestamp
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, NOW())
                        RETURNING id
                    """,
                    self.user_id,
                    self.conversation_id,
                    npc_id,
                    action_type,
                    json.dumps(action_details),
                    "requested"
                    )
                    return row["id"]
        except Exception as e:
            logger.error(f"Error tracking action request: {e}")
            return -1

    async def _track_completed_action_npc(
        self,
        npc_id: int,
        action: Dict[str, Any],
        result: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> int:
        """
        Track a completed NPC action.
        """
        try:
            async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                async with pool.acquire() as conn:
                    row = await conn.fetchrow("""
                        INSERT INTO NyxActionTracking (
                            user_id, conversation_id, npc_id,
                            action_type, action_data, result_data, status, timestamp
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                        RETURNING id
                    """,
                    self.user_id,
                    self.conversation_id,
                    npc_id,
                    action.get("type", "unknown"),
                    json.dumps(action),
                    json.dumps(result),
                    "completed"
                    )
                    return row["id"]
        except Exception as e:
            logger.error(f"Error tracking completed action: {e}")
            return -1

    async def _track_action_request_general(
        self,
        agent_type: str,
        agent_id: Union[int, str],
        action_type: str,
        action_details: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> int:
        """
        Track an action request for non-NPC agents.
        """
        try:
            async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                async with pool.acquire() as conn:
                    row = await conn.fetchrow("""
                        INSERT INTO NyxActionTracking (
                            user_id, conversation_id, agent_type, agent_id,
                            action_type, action_data, status, timestamp
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                        RETURNING id
                    """,
                    self.user_id,
                    self.conversation_id,
                    agent_type,
                    str(agent_id),
                    action_type,
                    json.dumps(action_details),
                    "requested"
                    )
                    return row["id"]
        except Exception as e:
            logger.error(f"Error tracking action request: {e}")
            return -1

    async def _track_completed_action_general(
        self,
        agent_type: str,
        agent_id: Union[int, str],
        action: Dict[str, Any],
        result: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> int:
        """
        Track a completed action for non-NPC agents.
        """
        try:
            async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                async with pool.acquire() as conn:
                    row = await conn.fetchrow("""
                        INSERT INTO NyxActionTracking (
                            user_id, conversation_id, agent_type, agent_id,
                            action_type, action_data, result_data, status, timestamp
                        )
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
                        RETURNING id
                    """,
                    self.user_id,
                    self.conversation_id,
                    agent_type,
                    str(agent_id),
                    action.get("type", "unknown"),
                    json.dumps(action),
                    json.dumps(result),
                    "completed"
                    )
                    return row["id"]
        except Exception as e:
            logger.error(f"Error tracking completed action: {e}")
            return -1

    # ---------------------------------------------------------------------
    # LOGGING DIRECTIVES
    # ---------------------------------------------------------------------
    async def _log_directive_npc(
        self,
        npc_id: int,
        directive_id: int,
        directive_type: str,
        directive_data: Dict[str, Any]
    ):
        """
        Log an NPC directive issuance.
        """
        memory_system = await self.get_memory_system()
        npc_name = await self._get_npc_name(npc_id)
        await memory_system.add_memory(
            memory_text=f"I issued a {directive_type} directive to {npc_name}",
            memory_type="observation",
            memory_scope="game",
            significance=6,
            tags=["directive", directive_type, f"npc_{npc_id}"],
            metadata={
                "directive_id": directive_id,
                "directive_type": directive_type,
                "directive_data": directive_data,
                "npc_id": npc_id
            }
        )

    async def _log_directive_general(
        self,
        agent_type: str,
        agent_id: Union[int, str],
        directive_id: int,
        directive_type: str,
        directive_data: Dict[str, Any]
    ):
        """
        Log a directive issuance to a non-NPC agent.
        """
        memory_system = await self.get_memory_system()
        agent_identifier = await self._get_agent_identifier(agent_type, agent_id)
        await memory_system.add_memory(
            memory_text=f"I issued a {directive_type} directive to {agent_identifier}",
            memory_type="observation",
            memory_scope="game",
            significance=6,
            tags=["directive", directive_type, f"{agent_type}_{agent_id}"],
            metadata={
                "directive_id": directive_id,
                "directive_type": directive_type,
                "directive_data": directive_data,
                "agent_type": agent_type,
                "agent_id": agent_id
            }
        )

    # ---------------------------------------------------------------------
    # GETTING NPC NAME & NARRATIVE CONTEXT
    # ---------------------------------------------------------------------
    async def _get_npc_name(self, npc_id: int) -> str:
        """
        Get the name of an NPC.
        """
        try:
            async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                async with pool.acquire() as conn:
                    row = await conn.fetchrow("""
                        SELECT npc_name FROM NPCStats
                        WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
                    """, npc_id, self.user_id, self.conversation_id)

                    if row:
                        return row["npc_name"]
                    return f"NPC {npc_id}"
        except Exception as e:
            logger.error(f"Error getting NPC name: {e}")
            return f"NPC {npc_id}"

    async def _get_current_narrative_context(self) -> Dict[str, Any]:
        """
        Get the current narrative context from the database.
        """
        try:
            async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                async with pool.acquire() as conn:
                    row = await conn.fetchrow("""
                        SELECT value FROM CurrentRoleplay
                        WHERE user_id = $1 AND conversation_id = $2 AND key = 'NyxNarrativeArcs'
                    """, self.user_id, self.conversation_id)

                    if row and row["value"]:
                        return json.loads(row["value"])
            return {}
        except Exception as e:
            logger.error(f"Error getting narrative context: {e}")
            return {}

    # ---------------------------------------------------------------------
    # GETTING "AGENT IDENTIFIER"
    # ---------------------------------------------------------------------
    async def _get_agent_identifier(self, agent_type: str, agent_id: Union[int, str]) -> str:
        """
        Get a readable identifier for any agent.

        If NPC, we use `_get_npc_name()`.
        If not NPC, format agent_type + agent_id.
        """
        if agent_type == AgentType.NPC:
            return await self._get_npc_name(int(agent_id))
        # else
        agent_type_formatted = agent_type.replace("_", " ").title()
        return f"{agent_type_formatted} Agent {agent_id}"

    # ---------------------------------------------------------------------
    # COORDINATE AGENTS
    # ---------------------------------------------------------------------
    async def coordinate_agents(
        self,
        action_type: str,
        primary_agent_type: str,
        action_details: Dict[str, Any],
        supporting_agents: List[str] = None
    ) -> Dict[str, Any]:
        """
        Coordinate multiple agents for a complex action.
        """
        supporting_agents = supporting_agents or []

        # First, check permission with primary agent
        primary_permission = await self.check_action_permission(
            primary_agent_type, "primary", action_type, action_details
        )
        if not primary_permission["approved"]:
            return {
                "success": False,
                "reason": primary_permission["reasoning"],
                "action_type": action_type
            }

        # Track the coordinated action
        tracking_id = await self._track_action_request_general(
            agent_type=primary_agent_type,
            agent_id="coordinated",
            action_type=action_type,
            action_details={**action_details, "supporting_agents": supporting_agents}
        )
        result = {
            "success": True,
            "action_type": action_type,
            "primary_agent": primary_agent_type,
            "supporting_agents": supporting_agents,
            "tracking_id": tracking_id,
            "agent_results": {}
        }

        # Execute the primary agent's action
        try:
            if primary_agent_type in self.registered_agents:
                primary_agent = self.registered_agents[primary_agent_type]
                primary_result = await self._execute_agent_action(
                    primary_agent, action_type, action_details
                )
            else:
                # Simulate
                primary_result = {
                    "status": "simulated",
                    "action_type": action_type,
                    "message": f"Simulated {primary_agent_type} response"
                }

            result["agent_results"][primary_agent_type] = primary_result

        except Exception as e:
            logger.error(f"Error executing primary agent action: {e}")
            result["agent_results"][primary_agent_type] = {
                "error": str(e),
                "action_type": action_type
            }
            result["success"] = False

        # Execute supporting agents
        for ag_type in supporting_agents:
            try:
                support_permission = await self.check_action_permission(
                    ag_type, "support", action_type, action_details
                )
                if not support_permission["approved"]:
                    result["agent_results"][ag_type] = {
                        "skipped": True,
                        "reason": support_permission["reasoning"]
                    }
                    continue

                if ag_type in self.registered_agents:
                    agent_inst = self.registered_agents[ag_type]
                    agent_result = await self._execute_agent_action(
                        agent_inst, action_type, action_details
                    )
                else:
                    agent_result = {
                        "status": "simulated",
                        "action_type": action_type,
                        "message": f"Simulated {ag_type} supporting response"
                    }
                result["agent_results"][ag_type] = agent_result

            except Exception as e:
                logger.error(f"Error executing supporting agent action: {e}")
                result["agent_results"][ag_type] = {
                    "error": str(e),
                    "action_type": action_type
                }

        # Track completed
        await self._track_completed_action_general(
            agent_type=primary_agent_type,
            agent_id="coordinated",
            action={**action_details, "supporting_agents": supporting_agents},
            result=result
        )
        return result

    async def _execute_agent_action(self, agent, action_type, action_details):
        """
        Execute an action on an agent instance.
        """
        # Check for specific execution method
        if hasattr(agent, "execute_action"):
            return await agent.execute_action(action_type, action_details)
            
        # Check for action-specific methods
        method_name = f"handle_{action_type}"
        if hasattr(agent, method_name):
            method = getattr(agent, method_name)
            return await method(action_details)
            
        # Check for process_directive method
        if hasattr(agent, "process_directive"):
            return await agent.process_directive({
                "type": action_type,
                **action_details
            })
            
        # Default implementation
        return {
            "status": "executed",
            "action_type": action_type,
            "details": action_details
        }

    # ---------------------------------------------------------------------
    # BROADCAST TO ALL AGENTS
    # ---------------------------------------------------------------------
    async def broadcast_to_all_agents(self, message_type: str, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Broadcast a message to all registered agents.
        """
        results = {
            "message_type": message_type,
            "broadcast_time": datetime.now().isoformat(),
            "recipients": len(self.registered_agents),
            "responses": {}
        }

        # Log in memory
        memory_system = await self.get_memory_system()
        await memory_system.add_memory(
            memory_text=f"Broadcast message of type '{message_type}' to all agents",
            memory_type="system",
            memory_scope="game",
            significance=4,
            tags=["broadcast", message_type],
            metadata={"message_data": message_data}
        )

        # Broadcast
        for ag_type, agent_inst in self.registered_agents.items():
            try:
                # Check if agent can process message
                if hasattr(agent_inst, "can_process_message") and not agent_inst.can_process_message(message_type):
                    results["responses"][ag_type] = {
                        "skipped": True,
                        "reason": "Agent does not process this message type"
                    }
                    continue

                # Check permission
                permission = await self.check_action_permission(
                    ag_type, "system", "receive_broadcast",
                    {"message_type": message_type}
                )
                if not permission["approved"]:
                    results["responses"][ag_type] = {
                        "skipped": True,
                        "reason": permission["reasoning"]
                    }
                    continue

                # Send
                if hasattr(agent_inst, "process_broadcast"):
                    response = await agent_inst.process_broadcast(message_type, message_data)
                    results["responses"][ag_type] = response
                else:
                    results["responses"][ag_type] = {
                        "status": "no_handler", "message_type": message_type
                    }

            except Exception as e:
                logger.error(f"Error broadcasting to {ag_type}: {e}")
                results["responses"][ag_type] = {"error": str(e), "message_type": message_type}

        return results
        
    # ---------------------------------------------------------------------
    # GET NARRATIVE STATUS
    # ---------------------------------------------------------------------
    async def get_narrative_status(self) -> Dict[str, Any]:
        """
        Get the current status of the narrative as a whole.
        """
        memory_system = await self.get_memory_system()
        recent_memories = await memory_system.get_recent_memories(limit=5)

        # Attempt to get current narrative stage
        try:
            from logic.narrative_progression import get_current_narrative_stage
            narrative_stage = await get_current_narrative_stage(self.user_id, self.conversation_id)
        except Exception as e:
            logger.error(f"Error retrieving narrative stage: {e}")
            narrative_stage = None

        # Active conflicts
        active_conflicts = []
        try:
            from logic.conflict_system.conflict_manager import ConflictManager
            conflict_manager = ConflictManager(self.user_id, self.conversation_id)
            active_conflicts = await conflict_manager.get_active_conflicts()
        except Exception as e:
            logger.error(f"Error getting active conflicts: {e}")

        # Key NPCs
        key_npcs = []
        try:
            from story_agent.tools import get_key_npcs
            class ContextMock:
                def __init__(self, user_id, conversation_id):
                    self.context = {"user_id": user_id, "conversation_id": conversation_id}
            ctx_mock = ContextMock(self.user_id, self.conversation_id)
            key_npcs = await get_key_npcs(ctx_mock, limit=5)
        except Exception as e:
            logger.error(f"Error getting key NPCs: {e}")

        # Resource status
        resources = {}
        try:
            from logic.resource_management import ResourceManager
            resource_manager = ResourceManager(self.user_id, self.conversation_id)
            resources = await resource_manager.get_resources()
            vitals = await resource_manager.get_vitals()
            resources.update(vitals)
        except Exception as e:
            logger.error(f"Error getting resources: {e}")

        # Build the status
        directive_count = 0
        # We'll reuse the NPC directive call; 'all' isn't real, but we mimic the old call:
        try:
            directives_all = await self.get_npc_directives(npc_id="all")  # not truly supported, but for example
            directive_count = len(directives_all)
        except:
            pass

        return {
            "narrative_stage": {
                "name": narrative_stage.name if narrative_stage else "Unknown",
                "description": narrative_stage.description if narrative_stage else ""
            },
            "recent_memories": recent_memories,
            "active_conflicts": active_conflicts,
            "key_npcs": key_npcs,
            "resources": resources,
            "directive_count": directive_count,
            "timestamp": datetime.now().isoformat()
        }

    # ---------------------------------------------------------------------
    # GET/CREATE AGENT MEMORY
    # ---------------------------------------------------------------------
    async def get_agent_memory(self, agent_type: str, agent_id: Union[int, str], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get memories related to a specific agent.
        """
        memory_system = await self.get_memory_system()
        memories = await memory_system.retrieve_memories(
            query="",
            memory_types=["observation", "reflection", "abstraction"],
            scopes=["game"],
            limit=limit,
            context={"tags": [f"{agent_type}_{agent_id}"]}
        )
        return memories

    async def create_agent_memory(
        self,
        agent_type: str,
        agent_id: Union[int, str],
        memory_text: str,
        significance: int = 5,
        tags: List[str] = None
    ) -> int:
        """
        Create a memory about an agent.
        """
        memory_system = await self.get_memory_system()
        tags = tags or []
        agent_tag = f"{agent_type}_{agent_id}"
        if agent_tag not in tags:
            tags.append(agent_tag)

        memory_id = await memory_system.add_memory(
            memory_text=memory_text,
            memory_type="observation",
            memory_scope="game",
            significance=significance,
            tags=tags,
            metadata={
                "agent_type": agent_type,
                "agent_id": agent_id,
                "created_at": datetime.now().isoformat()
            }
        )
        return memory_id

    async def generate_agent_reflection(
        self,
        agent_type: str,
        agent_id: Union[int, str],
        topic: str = None
    ) -> Dict[str, Any]:
        """
        Generate a reflection about an agent's actions and behavior.
        """
        memory_system = await self.get_memory_system()
        agent_tag = f"{agent_type}_{agent_id}"
        memories = await memory_system.retrieve_memories(
            query="",
            memory_types=["observation"],
            scopes=["game"],
            limit=20,
            context={"tags": [agent_tag]}
        )
        if not memories:
            return {
                "reflection": f"I don't have sufficient memories about {agent_type} {agent_id} to form a reflection.",
                "confidence": 0.1,
                "agent_type": agent_type,
                "agent_id": agent_id
            }

        memory_texts = [m["memory_text"] for m in memories]
        
        reflection_text = await generate_reflection(
            memory_texts=memory_texts,
            topic=topic or f"Behavior and actions of {agent_type} {agent_id}",
            context={"agent_type": agent_type, "agent_id": agent_id}
        )
        reflection_memory_id = await memory_system.add_memory(
            memory_text=reflection_text,
            memory_type="reflection",
            memory_scope="game",
            significance=6,
            tags=[agent_tag, "reflection"],
            metadata={
                "agent_type": agent_type,
                "agent_id": agent_id,
                "topic": topic,
                "created_at": datetime.now().isoformat()
            }
        )
        return {
            "reflection": reflection_text,
            "confidence": 0.7,
            "agent_type": agent_type,
            "agent_id": agent_id,
            "memory_id": reflection_memory_id
        }
        
    # ---------------------------------------------------------------------
    # AGENT FEEDBACK MECHANISM
    # ---------------------------------------------------------------------
    async def generate_agent_feedback(
        self,
        agent_type: str,
        agent_id: Union[int, str],
        action: Dict[str, Any],
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate structured feedback for an agent based on its action and results.
        
        Args:
            agent_type: Type of agent (use AgentType constants)
            agent_id: ID of agent instance
            action: Information about the action performed
            result: Result of the action
            
        Returns:
            Feedback data
        """
        # Get agent's recent action history
        recent_actions = await self._get_agent_recent_actions(agent_type, agent_id)
        
        # Get agent's performance metrics
        performance = await self.get_agent_performance_metrics(agent_type, agent_id)
        
        # Identify areas for improvement
        improvement_areas = []
        improvement_suggestions = {}
        
        # Check for patterns in rejected actions
        rejected_actions = [a for a in recent_actions if a.get("result_data", {}).get("approved", True) == False]
        if len(rejected_actions) > 2:
            # Pattern of rejected actions - suggest improvements
            rejection_reasons = [a.get("result_data", {}).get("reasoning", "") for a in rejected_actions]
            common_reason = self._find_common_substring(rejection_reasons)
            if common_reason:
                improvement_areas.append("permission_approval")
                improvement_suggestions["permission_approval"] = f"Consider addressing this common rejection reason: {common_reason}"
        
        # Check for action consistency
        action_types = [a.get("action_type") for a in recent_actions]
        if len(set(action_types)) == 1 and len(action_types) > 3:
            # Agent is only using one type of action repeatedly
            improvement_areas.append("action_variety")
            improvement_suggestions["action_variety"] = "Consider using a wider variety of actions for more dynamic behavior"
        
        # Check for narrative contribution
        if agent_type == AgentType.NPC and "dialogue" in action.get("type", ""):
            narrative_contribution = self._evaluate_narrative_contribution(action, result)
            if narrative_contribution < 0.5:
                improvement_areas.append("narrative_contribution")
                improvement_suggestions["narrative_contribution"] = "Consider making dialogue more relevant to current narrative arcs"
        
        # Check for user preference alignment
        if "user_reaction" in result:
            user_reaction = result.get("user_reaction", 0)
            if user_reaction < 0:
                improvement_areas.append("user_alignment")
                improvement_suggestions["user_alignment"] = "Consider better aligning actions with user preferences"
        
        # Compile the feedback
        feedback = {
            "action_id": result.get("tracking_id"),
            "timestamp": datetime.now().isoformat(),
            "performance_rating": self._calculate_performance_rating(action, result, performance),
            "improvement_areas": improvement_areas,
            "improvement_suggestions": improvement_suggestions,
            "positive_aspects": self._identify_positive_aspects(action, result),
            "context_considerations": self._identify_context_considerations(agent_type)
        }
        
        # Log the feedback
        await self._log_agent_feedback(agent_type, agent_id, feedback)
        
        return feedback

    async def _get_agent_recent_actions(self, agent_type: str, agent_id: Union[int, str]) -> List[Dict[str, Any]]:
        """Get recent actions for an agent."""
        try:
            async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                async with pool.acquire() as conn:
                    if agent_type == AgentType.NPC:
                        rows = await conn.fetch("""
                            SELECT action_type, action_data, result_data, status, timestamp
                            FROM NyxActionTracking
                            WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
                            ORDER BY timestamp DESC
                            LIMIT 10
                        """, self.user_id, self.conversation_id, int(agent_id))
                    else:
                        rows = await conn.fetch("""
                            SELECT action_type, action_data, result_data, status, timestamp
                            FROM NyxActionTracking
                            WHERE user_id = $1 AND conversation_id = $2 
                            AND agent_type = $3 AND agent_id = $4
                            ORDER BY timestamp DESC
                            LIMIT 10
                        """, self.user_id, self.conversation_id, agent_type, str(agent_id))
                    
                    return [
                        {
                            "action_type": row["action_type"],
                            "action_data": json.loads(row["action_data"]) if row["action_data"] else {},
                            "result_data": json.loads(row["result_data"]) if row["result_data"] else {},
                            "status": row["status"],
                            "timestamp": row["timestamp"]
                        }
                        for row in rows
                    ]
        except Exception as e:
            logger.error(f"Error getting agent recent actions: {e}")
            return []

    def _find_common_substring(self, strings: List[str], min_length: int = 10) -> str:
        """Find common substring in a list of strings."""
        if not strings:
            return ""
        
        # Use the shortest string as reference
        reference = min(strings, key=len)
        
        # Try decreasing substring lengths
        for length in range(len(reference), min_length - 1, -1):
            for i in range(len(reference) - length + 1):
                substring = reference[i:i+length]
                if all(substring in s for s in strings):
                    return substring
        
        return ""

    def _evaluate_narrative_contribution(self, action: Dict[str, Any], result: Dict[str, Any]) -> float:
        """Evaluate how much an action contributes to the narrative."""
        # Simple scoring based on keywords
        score = 0.5  # Default middle score
        
        narrative_keywords = ["arc", "story", "plot", "development", "character", "progression"]
        description = action.get("description", "").lower()
        
        # Add score for narrative-relevant content
        for keyword in narrative_keywords:
            if keyword in description:
                score += 0.1
        
        # Cap score
        return min(1.0, score)

    def _calculate_performance_rating(self, action: Dict[str, Any], result: Dict[str, Any], performance: Dict[str, Any]) -> float:
        """Calculate a performance rating for an action."""
        base_score = 0.7  # Start with a decent score
        
        # Adjust based on action complexity
        complexity = len(action.get("description", "")) / 100  # Proxy for complexity
        base_score += min(0.1, complexity * 0.05)  # Reward complexity up to a point
        
        # Adjust based on success and impact
        if "success" in result and result["success"]:
            base_score += 0.1
        if "impact" in result and result["impact"] > 0:
            base_score += min(0.1, result["impact"] * 0.02)
        
        # Adjust based on historical performance
        avg_rating = performance.get("average_rating", 0.7)
        base_score = (base_score * 0.8) + (avg_rating * 0.2)  # Blend with history
        
        # Ensure score is within bounds
        return max(0.1, min(1.0, base_score))

    def _identify_positive_aspects(self, action: Dict[str, Any], result: Dict[str, Any]) -> List[str]:
        """Identify positive aspects of an agent's action."""
        positive_aspects = []
        
        # Check for narrative relevance
        if "arc" in action.get("description", "").lower() or "story" in action.get("description", "").lower():
            positive_aspects.append("narrative_relevance")
        
        # Check for creative approach
        if "creative" in result or "novel" in result:
            positive_aspects.append("creative_approach")
        
        # Check for user engagement
        if "user_engagement" in result and result["user_engagement"] > 0.6:
            positive_aspects.append("high_user_engagement")
        
        # Check for character consistency
        if "character_consistency" in result and result["character_consistency"] > 0.7:
            positive_aspects.append("strong_character_consistency")
        
        return positive_aspects

    def _identify_context_considerations(self, agent_type: str) -> List[str]:
        """Identify context considerations for an agent type."""
        if agent_type == AgentType.NPC:
            return ["character_motivation", "relationship_dynamics", "scene_context"]
        elif agent_type == AgentType.STORY_DIRECTOR:
            return ["narrative_pacing", "player_engagement", "theme_consistency"]
        elif agent_type == AgentType.CONFLICT_ANALYST:
            return ["tension_balance", "player_challenge", "narrative_impact"]
        else:
            return ["game_context", "player_preferences", "narrative_integration"]

    async def _log_agent_feedback(self, agent_type: str, agent_id: Union[int, str], feedback: Dict[str, Any]) -> None:
        """Log feedback given to an agent."""
        try:
            async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                async with pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO NyxAgentFeedback (
                            user_id, conversation_id, 
                            agent_type, agent_id, 
                            feedback_data, timestamp
                        )
                        VALUES ($1, $2, $3, $4, $5, NOW())
                    """,
                    self.user_id,
                    self.conversation_id,
                    agent_type,
                    str(agent_id),
                    json.dumps(feedback)
                    )
        except Exception as e:
            logger.error(f"Error logging agent feedback: {e}")
    
    # ---------------------------------------------------------------------
    # CENTRALIZED STATE MANAGEMENT
    # ---------------------------------------------------------------------
    class SharedGameState:
        """
        Centralized state object that all agents can access through governance.
        
        This provides a consistent view of the game world to all agents and tracks
        changes over time for monitoring and temporal consistency enforcement.
        """
        
        def __init__(self, user_id: int, conversation_id: int):
            self.user_id = user_id
            self.conversation_id = conversation_id
            self.state = {}
            self.history = []  # History of state changes
            self.last_updated = datetime.now()
            self.lock = asyncio.Lock()
            self.change_listeners = {}
            self.last_save_time = datetime.now()
            self.save_interval = 60  # seconds
        
        async def initialize(self):
            """Load initial state from the database."""
            try:
                async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                    async with pool.acquire() as conn:
                        row = await conn.fetchrow("""
                            SELECT state FROM NyxGameState
                            WHERE user_id = $1 AND conversation_id = $2
                            ORDER BY updated_at DESC
                            LIMIT 1
                        """, self.user_id, self.conversation_id)
                        
                        if row and row["state"]:
                            self.state = json.loads(row["state"])
                            logger.info(f"Loaded initial game state for user {self.user_id}, conversation {self.conversation_id}")
                        else:
                            # Initialize with defaults
                            self._initialize_default_state()
            except Exception as e:
                logger.error(f"Error loading initial game state: {e}")
                # Initialize with defaults
                self._initialize_default_state()
        
        def _initialize_default_state(self):
            """Initialize with default state values."""
            self.state = {
                "environment": {
                    "current_location": "unknown",
                    "time_of_day": "day",
                    "weather": "clear"
                },
                "narrative": {
                    "current_act": "introduction",
                    "tension_level": 1,
                    "active_arcs": []
                },
                "player": {
                    "stats": {},
                    "inventory": [],
                    "relationships": {}
                },
                "npcs": {},
                "world": {
                    "locations": {},
                    "factions": {},
                    "global_variables": {}
                }
            }
        
        async def get_state(self, path: str = None) -> Any:
            """
            Get the current state or a specific path within it.
            
            Args:
                path: Optional dot-notation path to a specific state value
                
            Returns:
                The requested state value or the entire state
            """
            async with self.lock:
                if not path:
                    return self.state.copy()
                
                # Navigate the path
                parts = path.split(".")
                current = self.state
                for part in parts:
                    if isinstance(current, dict) and part in current:
                        current = current[part]
                    else:
                        return None  # Path doesn't exist
                
                # Return a copy to prevent direct modification
                if isinstance(current, dict):
                    return current.copy()
                elif isinstance(current, list):
                    return current.copy()
                else:
                    return current
        
        async def update_state(self, path: str, value: Any, agent_info: Dict[str, Any] = None) -> Dict[str, Any]:
            """
            Update a specific path in the state.
            
            Args:
                path: Dot-notation path to update
                value: New value to set
                agent_info: Optional information about the agent making the update
                
            Returns:
                Dictionary with update results
            """
            async with self.lock:
                old_value = None
                # Navigate to parent object
                parts = path.split(".")
                parent_path = ".".join(parts[:-1])
                key = parts[-1]
                
                parent = self.state
                if parent_path:
                    for part in parent_path.split("."):
                        if part not in parent:
                            parent[part] = {}
                        parent = parent[part]
                
                # Save old value for history
                if key in parent:
                    old_value = parent[key]
                
                # Update value
                parent[key] = value
                
                # Record change in history
                change_entry = {
                    "path": path,
                    "old_value": old_value,
                    "new_value": value,
                    "timestamp": datetime.now().isoformat(),
                    "agent_info": agent_info
                }
                
                self.history.append(change_entry)
                
                # Keep history at a reasonable size
                if len(self.history) > 100:
                    self.history = self.history[-100:]
                
                self.last_updated = datetime.now()
                
                # Maybe save to database
                if (self.last_updated - self.last_save_time).total_seconds() > self.save_interval:
                    await self.save_state()
                
                # Notify listeners
                await self._notify_change_listeners(path, old_value, value)
                
                return {
                    "success": True,
                    "path": path,
                    "old_value": old_value,
                    "new_value": value,
                    "timestamp": change_entry["timestamp"]
                }
        
        async def batch_update(self, updates: List[Dict[str, Any]], agent_info: Dict[str, Any] = None) -> Dict[str, Any]:
            """
            Perform multiple updates as an atomic operation.
            
            Args:
                updates: List of {path, value} dictionaries
                agent_info: Optional information about the agent making the updates
                
            Returns:
                Dictionary with batch update results
            """
            results = []
            async with self.lock:
                for update in updates:
                    path = update.get("path")
                    value = update.get("value")
                    
                    if not path:
                        continue
                    
                    # Individual update (reuse logic but within the lock)
                    result = await self._update_without_lock(path, value, agent_info)
                    results.append(result)
                
                # Save to database
                await self.save_state()
                
                return {
                    "success": True,
                    "update_count": len(results),
                    "results": results
                }
        
        async def _update_without_lock(self, path: str, value: Any, agent_info: Dict[str, Any] = None) -> Dict[str, Any]:
            """Update without acquiring the lock (for internal use)."""
            old_value = None
            # Navigate to parent object
            parts = path.split(".")
            parent_path = ".".join(parts[:-1])
            key = parts[-1]
            
            parent = self.state
            if parent_path:
                for part in parent_path.split("."):
                    if part not in parent:
                        parent[part] = {}
                    parent = parent[part]
            
            # Save old value for history
            if key in parent:
                old_value = parent[key]
            
            # Update value
            parent[key] = value
            
            # Record change in history
            change_entry = {
                "path": path,
                "old_value": old_value,
                "new_value": value,
                "timestamp": datetime.now().isoformat(),
                "agent_info": agent_info
            }
            
            self.history.append(change_entry)
            
            return {
                "success": True,
                "path": path,
                "old_value": old_value,
                "new_value": value
            }
        
        async def register_change_listener(self, path_prefix: str, callback: Callable) -> str:
            """
            Register a callback for state changes.
            
            Args:
                path_prefix: Path prefix to listen for changes
                callback: Async function to call when changes occur
                
            Returns:
                Listener ID
            """
            listener_id = f"listener_{len(self.change_listeners) + 1}_{int(datetime.now().timestamp())}"
            self.change_listeners[listener_id] = {"path_prefix": path_prefix, "callback": callback}
            return listener_id
        
        async def unregister_change_listener(self, listener_id: str) -> bool:
            """Unregister a change listener."""
            if listener_id in self.change_listeners:
                del self.change_listeners[listener_id]
                return True
            return False
        
        async def _notify_change_listeners(self, path: str, old_value: Any, new_value: Any) -> None:
            """Notify listeners of a state change."""
            for listener_id, listener in self.change_listeners.items():
                path_prefix = listener["path_prefix"]
                callback = listener["callback"]
                
                if path.startswith(path_prefix):
                    try:
                        await callback(path, old_value, new_value)
                    except Exception as e:
                        logger.error(f"Error in state change listener {listener_id}: {e}")
        
        async def save_state(self) -> bool:
            """Save the current state to the database."""
            try:
                async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                    async with pool.acquire() as conn:
                        await conn.execute("""
                            INSERT INTO NyxGameState (
                                user_id, conversation_id, state, updated_at
                            )
                            VALUES ($1, $2, $3, NOW())
                        """, 
                        self.user_id, 
                        self.conversation_id, 
                        json.dumps(self.state)
                        )
                
                self.last_save_time = datetime.now()
                return True
            except Exception as e:
                logger.error(f"Error saving game state: {e}")
                return False
        
        async def get_history(self, path_prefix: str = None, limit: int = 10) -> List[Dict[str, Any]]:
            """
            Get state change history.
            
            Args:
                path_prefix: Optional path prefix to filter history
                limit: Maximum number of history entries to return
                
            Returns:
                List of state change entries
            """
            async with self.lock:
                if not path_prefix:
                    return self.history[-limit:]
                
                # Filter by path prefix
                filtered_history = [
                    entry for entry in self.history
                    if entry["path"].startswith(path_prefix)
                ]
                
                return filtered_history[-limit:]

    async def initialize_game_state(self):
        """Initialize the shared game state."""
        self.game_state = self.SharedGameState(self.user_id, self.conversation_id)
        await self.game_state.initialize()
        logger.info(f"Initialized game state for user {self.user_id}, conversation {self.conversation_id}")

    async def get_game_state(self, path: str = None) -> Any:
        """
        Get the current game state or a specific value.
        
        Args:
            path: Optional dot-notation path to a specific state value
            
        Returns:
            The requested state value or the entire state
        """
        if not hasattr(self, "game_state"):
            await self.initialize_game_state()
        
        return await self.game_state.get_state(path)

    async def update_game_state(
        self,
        path: str,
        value: Any,
        agent_type: str = None,
        agent_id: Union[int, str] = None
    ) -> Dict[str, Any]:
        """
        Update a specific path in the game state.
        
        Args:
            path: Dot-notation path to update
            value: New value to set
            agent_type: Optional type of agent making the update
            agent_id: Optional ID of agent making the update
            
        Returns:
            Dictionary with update results
        """
        if not hasattr(self, "game_state"):
            await self.initialize_game_state()
        
        agent_info = None
        if agent_type:
            agent_info = {
                "agent_type": agent_type,
                "agent_id": agent_id
            }
        
        result = await self.game_state.update_state(path, value, agent_info)
        
        # Add entry to memory if it's a significant change
        if result.get("success") and (not result.get("old_value") or result.get("old_value") != value):
            memory_system = await self.get_memory_system()
            
            # Make a readable description of the change
            if agent_type:
                if agent_type == AgentType.NPC:
                    agent_name = await self._get_npc_name(int(agent_id))
                    change_description = f"{agent_name} changed game state: {path} is now {str(value)[:50]}"
                else:
                    change_description = f"{agent_type} agent changed game state: {path} is now {str(value)[:50]}"
            else:
                change_description = f"Game state updated: {path} is now {str(value)[:50]}"
            
            await memory_system.add_memory(
                memory_text=change_description,
                memory_type="system",
                memory_scope="game",
                significance=3,
                tags=["game_state", "state_change"],
                metadata={
                    "path": path,
                    "old_value": result.get("old_value"),
                    "new_value": value,
                    "agent_type": agent_type,
                    "agent_id": agent_id,
                    "timestamp": result.get("timestamp")
                }
            )
        
        return result
        
    # ---------------------------------------------------------------------
    # TEMPORAL CONSISTENCY ENFORCEMENT
    # ---------------------------------------------------------------------
    async def ensure_temporal_consistency(
        self,
        proposed_action: Dict[str, Any],
        agent_type: str,
        agent_id: Union[int, str]
    ) -> Dict[str, Any]:
        """
        Check if an action is consistent with the current narrative timeline.
        
        Args:
            proposed_action: The action to check
            agent_type: Type of agent proposing the action
            agent_id: ID of agent proposing the action
            
        Returns:
            Consistency check results
        """
        # Get current game state for time-related information
        current_time = await self.get_game_state("environment.time_of_day")
        current_date = await self.get_game_state("environment.current_date")
        current_location = await self.get_game_state("environment.current_location")
        
        # Get agent's recent actions
        recent_actions = await self._get_agent_recent_actions(agent_type, agent_id)
        
        # Check for time-based inconsistencies
        time_issues = []
        
        # Check if action references things that haven't happened yet
        if "future_reference" in json.dumps(proposed_action).lower():
            time_issues.append("References future events")
        
        # Check if action assumes a different time of day
        action_json = json.dumps(proposed_action).lower()
        if current_time == "day" and ("night" in action_json or "evening" in action_json or "dark" in action_json):
            time_issues.append(f"Action assumes night/evening but current time is {current_time}")
        elif current_time == "night" and ("day" in action_json or "morning" in action_json or "noon" in action_json):
            time_issues.append(f"Action assumes day/morning but current time is {current_time}")
        
        # Check for location-based inconsistencies
        location_issues = []
        
        # For NPCs, check if they can be at this location
        if agent_type == AgentType.NPC:
            npc_location = await self._get_npc_location(int(agent_id))
            
            if npc_location != current_location:
                # NPC is not in the current location
                location_issues.append(f"NPC is in {npc_location} but action is for {current_location}")
                
                # Check if the action mentions teleporting or impossible movement
                if not any(word in action_json for word in ["walk", "move", "travel", "go to", "arrive"]):
                    location_issues.append("Action doesn't account for movement between locations")
        
        # Check for causal inconsistencies
        causal_issues = []
        
        # Check if action depends on something that hasn't happened
        dependencies = proposed_action.get("dependencies", [])
        for dependency in dependencies:
            dependency_state = await self.get_game_state(dependency.get("path"))
            if dependency_state != dependency.get("expected_value"):
                causal_issues.append(f"Depends on {dependency.get('path')} = {dependency.get('expected_value')} but it's {dependency_state}")
        
        # Compile all issues
        all_issues = time_issues + location_issues + causal_issues
        
        # Overall consistency assessment
        is_consistent = len(all_issues) == 0
        
        # If inconsistent, generate suggestions for correction
        suggestions = []
        if not is_consistent:
            for issue in all_issues:
                if "NPC is in" in issue:
                    suggestions.append(f"Add movement action before this action")
                elif "assumes night" in issue:
                    suggestions.append(f"Adjust action to match current time ({current_time})")
                elif "Depends on" in issue:
                    suggestions.append(f"Add prerequisite action to establish dependency")
        
        return {
            "is_consistent": is_consistent,
            "time_issues": time_issues,
            "location_issues": location_issues,
            "causal_issues": causal_issues,
            "suggestions": suggestions
        }

    async def _get_npc_location(self, npc_id: int) -> str:
        """Get the current location of an NPC."""
        try:
            async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                async with pool.acquire() as conn:
                    row = await conn.fetchrow("""
                        SELECT current_location FROM NPCStats
                        WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
                    """, self.user_id, self.conversation_id, npc_id)
                    
                    if row and row["current_location"]:
                        return row["current_location"]
                    return "unknown"
        except Exception as e:
            logger.error(f"Error getting NPC location: {e}")
            return "unknown"

    async def advance_time(self, time_increment: str = "1 hour") -> Dict[str, Any]:
        """
        Advance the game time.
        
        Args:
            time_increment: Amount of time to advance (e.g., "1 hour", "30 minutes")
            
        Returns:
            Time advancement results
        """
        # Get current time information
        current_time = await self.get_game_state("environment.time_of_day") or "day"
        current_hour = await self.get_game_state("environment.hour") or 12
        
        # Parse increment
        increment_value = int(time_increment.split()[0])
        increment_unit = time_increment.split()[1].lower()
        
        # Convert to hours
        hours_to_advance = 0
        if increment_unit in ["hour", "hours"]:
            hours_to_advance = increment_value
        elif increment_unit in ["minute", "minutes"]:
            hours_to_advance = increment_value / 60
        elif increment_unit in ["day", "days"]:
            hours_to_advance = increment_value * 24
        
        # Calculate new hour
        new_hour = (current_hour + hours_to_advance) % 24
        
        # Determine new time of day
        new_time_of_day = current_time
        if 6 <= new_hour < 18:
            new_time_of_day = "day"
        else:
            new_time_of_day = "night"
        
        # Update game state
        await self.update_game_state("environment.hour", new_hour)
        await self.update_game_state("environment.time_of_day", new_time_of_day)
        
        # Inform agents of time change
        update_directives = []
        for agent_type, agent_instance in self.registered_agents.items():
            directive_id = await self.issue_directive(
                agent_type=agent_type,
                agent_id="default",
                directive_type=DirectiveType.ACTION,
                directive_data={
                    "type": "time_update",
                    "new_time_of_day": new_time_of_day,
                    "new_hour": new_hour,
                    "advanced_by": hours_to_advance
                },
                priority=DirectivePriority.MEDIUM,
                duration_minutes=10
            )
            update_directives.append(directive_id)
        
        # Add to memory
        memory_system = await self.get_memory_system()
        await memory_system.add_memory(
            memory_text=f"Time advanced by {time_increment}. It is now {new_time_of_day}, hour {int(new_hour)}.",
            memory_type="system",
            memory_scope="game",
            significance=4,
            tags=["time_advancement", "environment_change"]
        )
        
        return {
            "new_time_of_day": new_time_of_day,
            "new_hour": new_hour,
            "advanced_by": hours_to_advance,
            "update_directives_issued": len(update_directives)
        }
    
    # ---------------------------------------------------------------------
    # DECISION EXPLANATION SYSTEM
    # ---------------------------------------------------------------------
    async def explain_governance_decision(self, decision_id: int) -> Dict[str, Any]:
        """
        Provide detailed explanation for a governance decision.
        
        Args:
            decision_id: ID of the decision to explain
            
        Returns:
            Decision explanation
        """
        try:
            async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                async with pool.acquire() as conn:
                    row = await conn.fetchrow("""
                        SELECT * FROM NyxDecisionLog
                        WHERE id = $1 AND user_id = $2 AND conversation_id = $3
                    """, decision_id, self.user_id, self.conversation_id)
                    
                    if not row:
                        return {"error": f"Decision with ID {decision_id} not found"}
                    
                    decision_type = row["decision_type"]
                    original_data = json.loads(row["original_data"])
                    result_data = json.loads(row["result_data"])
                    timestamp = row["timestamp"]
                    
                    # Get related memories
                    memory_system = await self.get_memory_system()
                    relevant_memories = await memory_system.retrieve_memories(
                        query=f"decision {decision_id}",
                        memory_types=["system"],
                        scopes=["game"],
                        limit=3
                    )
                    
                    # Get narrative context at decision time
                    narrative_context = await self._get_narrative_context_at_time(timestamp)
                    
                    # Format explanation
                    explanation_components = []
                    
                    if decision_type == "permission_check":
                        agent_type = original_data.get("agent_type", "unknown")
                        agent_id = original_data.get("agent_id", "unknown")
                        action_type = original_data.get("action_type", "unknown")
                        
                        if agent_type == AgentType.NPC:
                            agent_name = await self._get_npc_name(int(agent_id))
                            explanation_components.append(f"NPC {agent_name} requested permission for {action_type} action")
                        else:
                            explanation_components.append(f"{agent_type} agent requested permission for {action_type} action")
                        
                        if result_data.get("approved"):
                            explanation_components.append("The action was approved")
                            if result_data.get("directive_applied"):
                                explanation_components.append(f"A directive was applied: {result_data.get('reasoning')}")
                        else:
                            explanation_components.append(f"The action was denied because: {result_data.get('reasoning')}")
                            if result_data.get("override_action"):
                                explanation_components.append("An alternative action was suggested")
                        
                    elif decision_type == "directive_conflict_resolution":
                        explanation_components.append(f"Resolved conflicts between {len(original_data)} directives")
                        explanation_components.append(f"Resulted in {len(result_data)} final directives")
                        
                        # Identify what changed
                        original_types = [d.get("type") for d in original_data]
                        result_types = [d.get("type") for d in result_data]
                        
                        if set(original_types) != set(result_types):
                            explanation_components.append(f"Changed directive types from {original_types} to {result_types}")
                        
                        for directive in result_data:
                            if "merged_from" in directive:
                                explanation_components.append(f"Merged {len(directive['merged_from'])} directives of type {directive.get('type')}")
                    
                    elif decision_type == "action_reporting":
                        agent_type = original_data.get("agent_type", "unknown")
                        agent_id = original_data.get("agent_id", "unknown")
                        action_type = original_data.get("action", {}).get("type", "unknown")
                        
                        if agent_type == AgentType.NPC:
                            agent_name = await self._get_npc_name(int(agent_id))
                            explanation_components.append(f"NPC {agent_name} reported {action_type} action")
                        else:
                            explanation_components.append(f"{agent_type} agent reported {action_type} action")
                        
                        if result_data.get("intervention"):
                            explanation_components.append(f"Governance intervened because: {result_data.get('reason')}")
                            explanation_components.append(f"Issued directive ID {result_data.get('directive_id')}")
                    
                    # Create contextual explanation based on narrative
                    if narrative_context:
                        active_arcs = narrative_context.get("active_arcs", [])
                        if active_arcs:
                            arc_names = [arc.get("name", "Unnamed Arc") for arc in active_arcs]
                            explanation_components.append(f"Active narrative arcs at decision time: {', '.join(arc_names)}")
                            
                            # See if decision aligned with active arcs
                            for arc in active_arcs:
                                arc_keywords = arc.get("keywords", [])
                                decision_text = json.dumps(original_data) + json.dumps(result_data)
                                
                                matching_keywords = [kw for kw in arc_keywords if kw.lower() in decision_text.lower()]
                                if matching_keywords:
                                    explanation_components.append(f"Decision aligned with arc '{arc.get('name')}' through keywords: {', '.join(matching_keywords)}")
                    
                    # Format final explanation
                    formatted_explanation = "\n".join([f"- {component}" for component in explanation_components])
                    
                    # Get current game state variables that influenced this decision
                    state_variables = await self._identify_influential_state_variables(decision_type, original_data, result_data)
                    
                    return {
                        "decision_id": decision_id,
                        "timestamp": timestamp.isoformat(),
                        "decision_type": decision_type,
                        "original_request": original_data,
                        "decision_result": result_data,
                        "explanation": formatted_explanation,
                        "related_memories": [m.get("memory_text") for m in relevant_memories],
                        "narrative_context": narrative_context,
                        "influential_state_variables": state_variables
                    }
                    
        except Exception as e:
            logger.error(f"Error explaining governance decision: {e}")
            return {"error": f"Error explaining decision: {str(e)}"}

    async def _get_narrative_context_at_time(self, timestamp) -> Dict[str, Any]:
        """Get narrative context at a specific time."""
        try:
            # Try to get the closest snapshot before the given timestamp
            async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                async with pool.acquire() as conn:
                    row = await conn.fetchrow("""
                        SELECT value FROM CurrentRoleplay
                        WHERE user_id = $1 AND conversation_id = $2 
                        AND key = 'NyxNarrativeArcs'
                        AND updated_at <= $3
                        ORDER BY updated_at DESC
                        LIMIT 1
                    """, self.user_id, self.conversation_id, timestamp)
                    
                    if row and row["value"]:
                        return json.loads(row["value"])
                    
                    # If no snapshot found before timestamp, get current context
                    return await self._get_current_narrative_context()
        except Exception as e:
            logger.error(f"Error getting narrative context at time: {e}")
            return {}

    async def _identify_influential_state_variables(
        self,
        decision_type: str,
        original_data: Dict[str, Any],
        result_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Identify game state variables that influenced a decision."""
        influential_vars = {}
        
        if not hasattr(self, "game_state"):
            return influential_vars
        
        # Different variables are relevant for different decision types
        if decision_type == "permission_check":
            # For permission checks, location and time often matter
            influential_vars["current_location"] = await self.get_game_state("environment.current_location")
            influential_vars["time_of_day"] = await self.get_game_state("environment.time_of_day")
            
            # For NPCs, relationships often matter
            if original_data.get("agent_type") == AgentType.NPC:
                npc_id = original_data.get("agent_id")
                influential_vars["relationship_to_player"] = await self.get_game_state(f"npcs.{npc_id}.relationship_to_player")
        
        elif decision_type == "directive_conflict_resolution":
            # For directive conflicts, narrative state matters
            influential_vars["tension_level"] = await self.get_game_state("narrative.tension_level")
            influential_vars["current_act"] = await self.get_game_state("narrative.current_act")
        
        elif decision_type == "action_reporting":
            # For action reporting, environment factors often matter
            influential_vars["environment_danger"] = await self.get_game_state("environment.danger_level")
            influential_vars["recent_player_actions"] = await self.get_game_state("player.recent_actions")
        
        return influential_vars
        
    # ---------------------------------------------------------------------
    # MEMORY-BASED DECISION MAKING
    # ---------------------------------------------------------------------
    async def enhance_decision_with_memories(
        self,
        decision_type: str,
        action_details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Use relevant memories to enhance governance decisions.
        
        Args:
            decision_type: Type of decision to enhance
            action_details: Original action details
            
        Returns:
            Enhanced action details
        """
        # Don't modify the original
        enhanced_details = action_details.copy()
        
        # Get memory system
        memory_system = await self.get_memory_system()
        
        # Create a search query based on action details
        search_terms = []
        
        # Add agent information if available
        if "agent_type" in enhanced_details:
            search_terms.append(enhanced_details["agent_type"])
        if "agent_id" in enhanced_details and enhanced_details["agent_type"] == AgentType.NPC:
            npc_name = await self._get_npc_name(int(enhanced_details["agent_id"]))
            search_terms.append(npc_name)
        
        # Add action type
        if "action_type" in enhanced_details:
            search_terms.append(enhanced_details["action_type"])
        elif "type" in enhanced_details:
            search_terms.append(enhanced_details["type"])
        
        # Add important details
        if "description" in enhanced_details:
            # Extract key nouns and verbs
            description = enhanced_details["description"]
            words = description.split()
            
            # Simple extraction of key words (could be more sophisticated)
            key_words = [w for w in words if len(w) > 4 and w.lower() not in ["about", "there", "these", "those", "their", "would", "could"]]
            search_terms.extend(key_words[:3])  # Add up to 3 key words
        
        # Create query
        query = " ".join(search_terms)
        
        # Retrieve relevant memories
        memories = await memory_system.retrieve_memories(
            query=query,
            memory_types=["observation", "reflection", "abstraction"],
            scopes=["game"],
            limit=5,
            min_significance=4
        )
        
        if not memories:
            # No relevant memories found
            return enhanced_details
        
        # Extract insights from memories
        memory_texts = [memory["memory_text"] for memory in memories]
        
        # Generate a merged insight
        insight = await self._generate_memory_insight(memory_texts, decision_type)
        
        # Enhance decision based on decision type
        if decision_type == "permission_check":
            # Add memory context
            if "context" not in enhanced_details:
                enhanced_details["context"] = {}
            enhanced_details["context"]["memory_insight"] = insight
            enhanced_details["context"]["relevant_memory_ids"] = [memory["id"] for memory in memories]
            
        elif decision_type == "directive_issuance":
            # Enhance directive data
            if "directive_data" not in enhanced_details:
                enhanced_details["directive_data"] = {}
            
            # Add memory-based enhancements
            enhanced_details["directive_data"]["memory_context"] = insight
            enhanced_details["directive_data"]["memory_relevance"] = "high" if len(memories) >= 3 else "medium"
            
        elif decision_type == "intervention_check":
            # For checking if intervention needed
            enhanced_details["memory_analysis"] = insight
            enhanced_details["memory_supported_intervention"] = "negative" in insight.lower() or "danger" in insight.lower()
        
        # Track the memories used
        enhanced_details["enhanced_with_memories"] = [memory["id"] for memory in memories]
        
        return enhanced_details

    async def _generate_memory_insight(self, memory_texts: List[str], decision_type: str) -> str:
        """Generate an insight from memories for a specific decision type."""
        
        # Customize prompt based on decision type
        if decision_type == "permission_check":
            prompt = f"""
            Based on these relevant memories:
            
            {memory_texts}
            
            Generate a concise insight about whether the requested action should be permitted.
            Consider patterns, previous permissions, and potential consequences.
            """
        elif decision_type == "directive_issuance":
            prompt = f"""
            Based on these relevant memories:
            
            {memory_texts}
            
            Generate a concise insight about how to enhance a directive being issued.
            Consider specific instructions, constraints, or objectives that would make sense.
            """
        elif decision_type == "intervention_check":
            prompt = f"""
            Based on these relevant memories:
            
            {memory_texts}
            
            Generate a concise insight about whether Nyx should intervene in the current action.
            Consider past interventions, consequences, and narrative impacts.
            """
        else:
            prompt = f"""
            Based on these relevant memories:
            
            {memory_texts}
            
            Generate a concise insight relevant to the current decision.
            Highlight patterns or important considerations.
            """
        
        try:
            insight = await generate_text_completion(
                system_prompt="You are generating memory-based insights for governance decisions.",
                user_prompt=prompt,
                temperature=0.4,
                max_tokens=150
            )
            return insight
        except Exception as e:
            logger.error(f"Error generating memory insight: {e}")
            # Fallback: simple concatenation with prefix
            return f"Memory insight: Past similar situations suggest caution. {memory_texts[0]}"
            
    # ---------------------------------------------------------------------
    # MULTI-AGENT PLANNING
    # ---------------------------------------------------------------------
    async def coordinate_narrative_plan(
        self,
        goal: str,
        involved_agents: List[str],
        timeframe: str
    ) -> Dict[str, Any]:
        """
        Create a coordinated plan across multiple agents toward a narrative goal.
        
        Args:
            goal: The narrative goal to achieve
            involved_agents: List of agent types to involve in planning
            timeframe: Timeframe for the plan (e.g., "short", "medium", "long")
            
        Returns:
            Coordinated plan
        """
        # Translate timeframe to actual time periods
        duration_minutes = {
            "immediate": 5,
            "short": 30,
            "medium": 60,
            "long": 120
        }.get(timeframe.lower(), 60)
        
        # Get necessary context for planning
        narrative_context = await self._get_current_narrative_context()
        current_location = await self.get_game_state("environment.current_location")
        available_npcs = await self._get_npcs_at_location(current_location)
        
        # Create initial plan
        plan = {
            "goal": goal,
            "timeframe": timeframe,
            "involved_agents": involved_agents,
            "created_at": datetime.now().isoformat(),
            "steps": [],
            "contingencies": [],
            "success_criteria": [],
            "coordination_id": f"plan_{int(datetime.now().timestamp())}"
        }
        
        # Track participating agents and their responses
        agent_responses = {}
        
        # Request plan contributions from each agent
        for agent_type in involved_agents:
            if agent_type in self.registered_agents:
                # Agent is available through direct registration
                agent = self.registered_agents[agent_type]
                
                try:
                    agent_contribution = await self._request_agent_plan_contribution(
                        agent, agent_type, goal, timeframe, narrative_context
                    )
                    agent_responses[agent_type] = agent_contribution
                except Exception as e:
                    logger.error(f"Error getting plan contribution from {agent_type}: {e}")
                    agent_responses[agent_type] = {"error": str(e)}
            
            elif agent_type == AgentType.NPC:
                # For NPCs, get contributions from available ones
                npc_responses = {}
                for npc in available_npcs:
                    try:
                        npc_contribution = await self._request_npc_plan_contribution(
                            npc["npc_id"], goal, timeframe, narrative_context
                        )
                        npc_responses[npc["npc_name"]] = npc_contribution
                    except Exception as e:
                        logger.error(f"Error getting plan contribution from NPC {npc['npc_name']}: {e}")
                
                agent_responses[agent_type] = npc_responses
        
        # Integrate contributions into a coherent plan
        integrated_plan = await self._integrate_plan_contributions(plan, agent_responses)
        
        # Create directives for each involved agent
        directive_ids = await self._create_plan_directives(integrated_plan)
        
        # Store plan in game state
        plan_id = f"plan_{int(datetime.now().timestamp())}"
        await self.update_game_state(f"narrative.active_plans.{plan_id}", integrated_plan)
        
        # Add to memory
        memory_system = await self.get_memory_system()
        await memory_system.add_memory(
            memory_text=f"Created narrative plan: {goal} involving {', '.join(involved_agents)}",
            memory_type="system",
            memory_scope="game",
            significance=6,
            tags=["narrative_plan", "coordination"],
            metadata={
                "plan": integrated_plan,
                "agent_responses": agent_responses,
                "directive_ids": directive_ids
            }
        )
        
        return {
            "plan_id": plan_id,
            "plan": integrated_plan,
            "directives_created": len(directive_ids),
            "directive_ids": directive_ids
        }

    async def _get_npcs_at_location(self, location: str) -> List[Dict[str, Any]]:
        """Get NPCs currently at a specific location."""
        try:
            async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                async with pool.acquire() as conn:
                    rows = await conn.fetch("""
                        SELECT npc_id, npc_name, archetype
                        FROM NPCStats
                        WHERE user_id = $1 AND conversation_id = $2 
                        AND current_location = $3
                        AND is_active = TRUE
                    """, self.user_id, self.conversation_id, location)
                    
                    return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting NPCs at location: {e}")
            return []

    async def _request_agent_plan_contribution(
        self,
        agent,
        agent_type: str,
        goal: str,
        timeframe: str,
        narrative_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Request a plan contribution from a specific agent."""
        # Check if agent has a plan_contribution method
        if hasattr(agent, "plan_contribution"):
            return await agent.plan_contribution(goal, timeframe, narrative_context)
        
        # For agents without specific planning methods, use a standardized approach
        # Create a directive to get plan contribution
        directive_id = await self.issue_directive(
            agent_type=agent_type,
            agent_id="default",
            directive_type=DirectiveType.ACTION,
            directive_data={
                "type": "plan_contribution",
                "goal": goal,
                "timeframe": timeframe,
                "narrative_context": narrative_context,
                "deadline": (datetime.now() + timedelta(minutes=5)).isoformat()
            },
            priority=DirectivePriority.HIGH,
            duration_minutes=10
        )
        
        # Wait a moment for the agent to process the directive
        await asyncio.sleep(2)
        
        # Check if agent has responded to directive
        response = await self._check_directive_response(directive_id)
        
        if response and "plan_contribution" in response:
            return response["plan_contribution"]
        
        # Fallback - generate a generic contribution based on agent type
        return await self._generate_generic_contribution(agent_type, goal, timeframe)

    async def _request_npc_plan_contribution(
        self,
        npc_id: int,
        goal: str,
        timeframe: str,
        narrative_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Request a plan contribution from an NPC."""
        # Get NPC information
        npc_name = await self._get_npc_name(npc_id)
        
        # Issue directive to the NPC
        directive_id = await self.issue_directive(
            agent_type=AgentType.NPC,
            agent_id=npc_id,
            directive_type=DirectiveType.ACTION,
            directive_data={
                "type": "plan_contribution",
                "goal": goal,
                "timeframe": timeframe,
                "narrative_context": narrative_context,
                "deadline": (datetime.now() + timedelta(minutes=5)).isoformat()
            },
            priority=DirectivePriority.HIGH,
            duration_minutes=10
        )
        
        # In a real system, you'd have a way to check the NPC's response to the directive
        # For this implementation, we'll generate a plausible contribution
        
        # Get NPC archetype to inform contribution
        async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
            async with pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT archetype FROM NPCStats
                    WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
                """, npc_id, self.user_id, self.conversation_id)
                
                archetype = row["archetype"] if row else "unknown"
        
        # Generate contribution based on archetype
        prompt = f"""
        Generate a plan contribution for NPC {npc_name} (archetype: {archetype}) for this narrative goal:
        
        Goal: {goal}
        Timeframe: {timeframe}
        
        Generate:
        1. 1-2 specific actions this NPC can take toward the goal
        2. How these actions fit with the NPC's character
        3. Any conditions or requirements for the NPC's participation
        
        Format as JSON with keys: actions, rationale, conditions
        """
        
        try:
            response_text = await generate_text_completion(
                system_prompt="You are generating narrative plan contributions for NPCs.",
                user_prompt=prompt,
                temperature=0.7,
                max_tokens=300
            )
            
            # Try to parse JSON response
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                # Extract information through simple parsing
                sections = response_text.split("\n\n")
                contribution = {
                    "actions": [],
                    "rationale": "",
                    "conditions": []
                }
                
                for section in sections:
                    if "actions:" in section.lower():
                        contribution["actions"] = [a.strip() for a in section.split("\n")[1:] if a.strip()]
                    elif "rationale:" in section.lower():
                        contribution["rationale"] = section.split(":", 1)[1].strip()
                    elif "conditions:" in section.lower():
                        contribution["conditions"] = [c.strip() for c in section.split("\n")[1:] if c.strip()]
                
                return contribution
                
        except Exception as e:
            logger.error(f"Error generating NPC plan contribution: {e}")
            # Provide a minimal fallback
            return {
                "actions": [f"{npc_name} will support the goal in a way that fits their character"],
                "rationale": f"As a {archetype}, {npc_name} can contribute to this goal",
                "conditions": ["Must align with NPC's established motivations"]
            }

    async def _check_directive_response(self, directive_id: int) -> Optional[Dict[str, Any]]:
        """Check if an agent has responded to a directive."""
        try:
            async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                async with pool.acquire() as conn:
                    row = await conn.fetchrow("""
                        SELECT response_data FROM NyxDirectiveResponses
                        WHERE directive_id = $1
                    """, directive_id)
                    
                    if row and row["response_data"]:
                        return json.loads(row["response_data"])
                    return None
        except Exception as e:
            logger.error(f"Error checking directive response: {e}")
            return None

    async def _generate_generic_contribution(
        self,
        agent_type: str,
        goal: str,
        timeframe: str
    ) -> Dict[str, Any]:
        """Generate a generic plan contribution based on agent type."""
        # Tailor contributions to agent type
        if agent_type == AgentType.STORY_DIRECTOR:
            return {
                "actions": [
                    "Create narrative framework for goal",
                    "Design key story events to advance goal"
                ],
                "rationale": "The Story Director will structure the narrative to naturally lead toward this goal",
                "conditions": [
                    "Must maintain narrative consistency",
                    "Should incorporate player's past choices"
                ]
            }
        elif agent_type == AgentType.CONFLICT_ANALYST:
            return {
                "actions": [
                    "Identify potential obstacles to goal",
                    "Design balanced challenges related to goal"
                ],
                "rationale": "The Conflict Analyst will ensure the goal involves engaging challenges",
                "conditions": [
                    "Challenges must be appropriate to player's skill level",
                    "Conflicts should have multiple resolution paths"
                ]
            }
        else:
            # Generic contribution
            return {
                "actions": [
                    "Support the goal through specialized functions",
                    "Coordinate with other agents as needed"
                ],
                "rationale": f"The {agent_type} agent will contribute through its area of expertise",
                "conditions": [
                    "Actions must remain within agent's functional domain"
                ]
            }

    async def _integrate_plan_contributions(
        self,
        base_plan: Dict[str, Any],
        agent_responses: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Integrate multiple agents' contributions into a coherent plan."""
        # Start with the base plan
        integrated_plan = base_plan.copy()
        
        # Collect actions from all agents
        all_actions = []
        all_conditions = []
        
        for agent_type, response in agent_responses.items():
            if agent_type == AgentType.NPC:
                # Handle NPC responses (which is a dict of NPC names to responses)
                for npc_name, npc_response in response.items():
                    if "actions" in npc_response:
                        for action in npc_response["actions"]:
                            all_actions.append({
                                "action": action,
                                "agent": f"NPC:{npc_name}",
                                "conditions": npc_response.get("conditions", [])
                            })
                        all_conditions.extend(npc_response.get("conditions", []))
            else:
                # Handle other agent responses
                if isinstance(response, dict) and "actions" in response:
                    for action in response["actions"]:
                        all_actions.append({
                            "action": action,
                            "agent": agent_type,
                            "conditions": response.get("conditions", [])
                        })
                    all_conditions.extend(response.get("conditions", []))
        
        # Build a sequence of steps - we need to order them logically
        ordered_steps = self._order_actions_logically(all_actions)
        
        # Deduplicate conditions
        unique_conditions = list(set(all_conditions))
        
        # Add to the plan
        integrated_plan["steps"] = ordered_steps
        integrated_plan["conditions"] = unique_conditions
        
        # Generate success criteria
        integrated_plan["success_criteria"] = [
            f"Goal achieved: {base_plan['goal']}",
            "All involved agents have completed their actions",
            "Player engagement maintained throughout"
        ]
        
        # Add contingencies
        integrated_plan["contingencies"] = [
            {
                "trigger": "Player rejects initial approach",
                "adaptation": "Agents adjust to more indirect methods"
            },
            {
                "trigger": "Required NPC becomes unavailable",
                "adaptation": "Substitute with alternative NPCs or restructure plan"
            },
            {
                "trigger": "Time constraints prevent completion",
                "adaptation": "Extend plan timeframe or simplify remaining steps"
            }
        ]
        
        return integrated_plan

    def _order_actions_logically(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Order actions in a logical sequence for a plan."""
        # This is a simplistic approach - a real implementation would be more sophisticated
        
        # First, identify setup/preparation actions
        setup_actions = []
        main_actions = []
        conclusion_actions = []
        
        setup_keywords = ["prepare", "create", "identify", "plan", "design", "establish"]
        conclusion_keywords = ["conclude", "finalize", "complete", "end", "resolve"]
        
        for action in actions:
            action_text = action["action"].lower()
            
            if any(keyword in action_text for keyword in setup_keywords):
                setup_actions.append(action)
            elif any(keyword in action_text for keyword in conclusion_keywords):
                conclusion_actions.append(action)
            else:
                main_actions.append(action)
        
        # Order each group by any explicit sequence numbers in the text
        for action_group in [setup_actions, main_actions, conclusion_actions]:
            action_group.sort(key=lambda a: self._extract_sequence_hint(a["action"]))
        
        # Combine the groups in order
        return setup_actions + main_actions + conclusion_actions

    def _extract_sequence_hint(self, action_text: str) -> int:
        """Extract any sequence hints from action text."""
        # Look for patterns like "First, ...", "Step 1: ...", etc.
        if "first" in action_text.lower():
            return 1
        elif "second" in action_text.lower():
            return 2
        elif "third" in action_text.lower():
            return 3
        elif "finally" in action_text.lower():
            return 999  # Very high number to put it last
        
        # Look for step numbers
        import re
        step_match = re.search(r'step\s+(\d+)', action_text.lower())
        if step_match:
            return int(step_match.group(1))
        
        # Default - no sequence hint
        return 500  # Middle priority

    async def _create_plan_directives(self, plan: Dict[str, Any]) -> List[int]:
        """Create directives for each agent involved in the plan."""
        directive_ids = []
        
        # Create a directive for each step in the plan
        for i, step in enumerate(plan["steps"]):
            agent_info = step["agent"].split(":")
            
            if len(agent_info) == 2 and agent_info[0] == "NPC":
                # This is an NPC action
                agent_type = AgentType.NPC
                
                # Look up NPC ID from name
                npc_name = agent_info[1]
                npc_id = await self._get_npc_id_from_name(npc_name)
                
                if npc_id:
                    directive_id = await self.issue_directive(
                        agent_type=agent_type,
                        agent_id=npc_id,
                        directive_type=DirectiveType.ACTION,
                        directive_data={
                            "type": "plan_step",
                            "plan_id": plan.get("coordination_id"),
                            "step_number": i + 1,
                            "action": step["action"],
                            "conditions": step.get("conditions", []),
                            "goal": plan["goal"],
                            "timeframe": plan["timeframe"]
                        },
                        priority=DirectivePriority.MEDIUM,
                        duration_minutes=120  # Longer duration for plan completion
                    )
                    directive_ids.append(directive_id)
            else:
                # This is a regular agent action
                agent_type = agent_info[0]
                
                directive_id = await self.issue_directive(
                    agent_type=agent_type,
                    agent_id="default",
                    directive_type=DirectiveType.ACTION,
                    directive_data={
                        "type": "plan_step",
                        "plan_id": plan.get("coordination_id"),
                        "step_number": i + 1,
                        "action": step["action"],
                        "conditions": step.get("conditions", []),
                        "goal": plan["goal"],
                        "timeframe": plan["timeframe"]
                    },
                    priority=DirectivePriority.MEDIUM,
                    duration_minutes=120  # Longer duration for plan completion
                )
                directive_ids.append(directive_id)
        
        return directive_ids

    async def _get_npc_id_from_name(self, npc_name: str) -> Optional[int]:
        """Get NPC ID from name."""
        try:
            async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                async with pool.acquire() as conn:
                    row = await conn.fetchrow("""
                        SELECT npc_id FROM NPCStats
                        WHERE npc_name = $1 AND user_id = $2 AND conversation_id = $3
                    """, npc_name, self.user_id, self.conversation_id)
                    
                    if row:
                        return row["npc_id"]
                    return None
        except Exception as e:
            logger.error(f"Error getting NPC ID from name: {e}")
            return None
            
    # ---------------------------------------------------------------------
    # AGENT PERFORMANCE METRICS
    # ---------------------------------------------------------------------
    async def get_agent_performance_metrics(
        self,
        agent_type: str,
        agent_id: Union[int, str]
    ) -> Dict[str, Any]:
        """
        Get metrics on how well an agent is performing its role.
        
        Args:
            agent_type: Type of agent (use AgentType constants)
            agent_id: ID of agent instance
            
        Returns:
            Dictionary with performance metrics
        """
        # Get agent's recent actions
        recent_actions = await self._get_agent_recent_actions(agent_type, agent_id)
        
        # Get agent's directives
        if agent_type == AgentType.NPC:
            directives = await self.get_npc_directives(int(agent_id))
        else:
            directives = await self.get_agent_directives(agent_type, agent_id)
        
        # Calculate metrics
        metrics = {
            "action_count": len(recent_actions),
            "directive_count": len(directives),
            "success_rate": self._calculate_success_rate(recent_actions),
            "response_time": self._calculate_response_time(recent_actions, directives),
            "narrative_contribution": await self._calculate_narrative_contribution(agent_type, agent_id),
            "consistency_score": await self._calculate_consistency_score(agent_type, agent_id),
            "player_impact_score": await self._calculate_player_impact(agent_type, agent_id),
            "adaptability_score": self._calculate_adaptability(recent_actions, directives),
            "last_updated": datetime.now().isoformat()
        }
        
        # Additional metrics for specific agent types
        if agent_type == AgentType.NPC:
            # Get NPC-specific metrics
            metrics.update(await self._get_npc_specific_metrics(int(agent_id)))
        elif agent_type == AgentType.STORY_DIRECTOR:
            # Get story director metrics
            metrics.update(await self._get_story_director_metrics())
        elif agent_type == AgentType.MEMORY_MANAGER:
            # Get memory manager metrics
            metrics.update(await self._get_memory_manager_metrics())
        
        # Generate performance suggestions
        metrics["improvement_suggestions"] = await self._generate_performance_suggestions(
            agent_type, agent_id, metrics
        )
        
        # Calculate overall score
        scores = [
            metrics["success_rate"] * 0.25,
            metrics["narrative_contribution"] * 0.2,
            metrics["consistency_score"] * 0.2,
            metrics["player_impact_score"] * 0.25,
            metrics["adaptability_score"] * 0.1
        ]
        metrics["overall_score"] = sum(scores)
        
        # Add rating level
        if metrics["overall_score"] >= 0.85:
            metrics["performance_rating"] = "excellent"
        elif metrics["overall_score"] >= 0.7:
            metrics["performance_rating"] = "good"
        elif metrics["overall_score"] >= 0.5:
            metrics["performance_rating"] = "adequate"
        else:
            metrics["performance_rating"] = "needs_improvement"
        
        return metrics

    async def _calculate_narrative_contribution(self, agent_type: str, agent_id: Union[int, str]) -> float:
        """Calculate narrative contribution score."""
        # Get current narrative arc data
        narrative_data = await self._get_current_narrative_context()
        
        if not narrative_data or not narrative_data.get("active_arcs"):
            return 0.5  # Default middle value when no arcs
        
        active_arcs = narrative_data.get("active_arcs", [])
        
        # Count how many active arcs involve this agent
        involvement_count = 0
        for arc in active_arcs:
            if agent_type == AgentType.NPC:
                # Check NPC involvement
                for npc_role in arc.get("npc_roles", []):
                    if npc_role.get("npc_id") == int(agent_id):
                        involvement_count += 1
                        break
            else:
                # Check other agent types
                agent_roles = arc.get("agent_roles", [])
                if any(role.get("agent_type") == agent_type for role in agent_roles):
                    involvement_count += 1
        
        # Calculate score based on involvement
        if not active_arcs:
            return 0.5
        
        # Base score on percentage of arcs the agent is involved in
        base_score = involvement_count / len(active_arcs)
        
        # Adjust for recent contributions through memory
        memory_system = await self.get_memory_system()
        
        # Query for memories about this agent related to narrative
        search_term = f"{agent_type} {'NPC' if agent_type == AgentType.NPC else 'agent'} narrative"
        
        # For NPCs, include name
        if agent_type == AgentType.NPC:
            npc_name = await self._get_npc_name(int(agent_id))
            search_term = f"{npc_name} narrative"
        
        narrative_memories = await memory_system.retrieve_memories(
            query=search_term,
            memory_types=["observation", "reflection"],
            scopes=["game"],
            limit=5
        )
        
        # Boost score based on recent narrative memories
        if narrative_memories:
            recent_contribution_boost = min(0.3, len(narrative_memories) * 0.06)
            base_score += recent_contribution_boost
        
        return min(1.0, base_score)

    async def _calculate_consistency_score(self, agent_type: str, agent_id: Union[int, str]) -> float:
        """Calculate consistency score for an agent."""
        if agent_type == AgentType.NPC:
            # For NPCs, check character consistency
            return await self._calculate_npc_consistency(int(agent_id))
        else:
            # For other agents, check functional consistency
            return await self._calculate_functional_consistency(agent_type, agent_id)

    async def _calculate_npc_consistency(self, npc_id: int) -> float:
        """Calculate character consistency for an NPC."""
        # Get NPC archetype and personality
        async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
            async with pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT personality_traits, archetype, backstory
                    FROM NPCStats
                    WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
                """, npc_id, self.user_id, self.conversation_id)
                
                if not row:
                    return 0.5  # Default middle value
                
                personality = row["personality_traits"]
                archetype = row["archetype"]
                backstory = row["backstory"]
        
        # Get recent dialogues
        recent_dialogues = await self._get_npc_recent_dialogues(npc_id)
        
        if not recent_dialogues:
            return 0.6  # Slightly above middle when no data
        
        # Use LLM to evaluate consistency
        prompt = f"""
        Evaluate the consistency of this NPC's character based on their recent dialogues:
        
        NPC Type: {archetype}
        Personality Traits: {personality}
        Backstory Summary: {backstory[:200]}...
        
        Recent Dialogues:
        {recent_dialogues}
        
        Rate the character consistency from 0.0 to 1.0, where:
        - 1.0 means perfectly consistent with archetype, personality, and backstory
        - 0.5 means somewhat consistent but with noticeable deviations
        - 0.0 means completely inconsistent with established character
        
        Provide your rating as a single number between 0.0 and 1.0.
        """
        
        try:
            response = await generate_text_completion(
                system_prompt="You are evaluating NPC character consistency.",
                user_prompt=prompt,
                temperature=0.3,
                max_tokens=50
            )
            
            # Extract the numerical score
            score_match = re.search(r'(\d+\.\d+|\d+)', response)
            if score_match:
                return float(score_match.group(1))
            return 0.6  # Default if parsing fails
        except Exception as e:
            logger.error(f"Error calculating NPC consistency: {e}")
            return 0.6  # Default value

    async def _get_npc_recent_dialogues(self, npc_id: int) -> str:
        """Get recent dialogues for an NPC."""
        try:
            async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                async with pool.acquire() as conn:
                    rows = await conn.fetch("""
                        SELECT content, timestamp
                        FROM NPCDialogues
                        WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
                        ORDER BY timestamp DESC
                        LIMIT 10
                    """, npc_id, self.user_id, self.conversation_id)
                    
                    if not rows:
                        return "No recent dialogues found."
                    
                    dialogues = []
                    for row in rows:
                        dialogues.append(f"[{row['timestamp']}] {row['content']}")
                    
                    return "\n".join(dialogues)
        except Exception as e:
            logger.error(f"Error getting NPC dialogues: {e}")
            return "Error retrieving dialogues."

    async def _calculate_functional_consistency(self, agent_type: str, agent_id: Union[int, str]) -> float:
        """Calculate functional consistency for a non-NPC agent."""
        # Get recent actions
        actions = await self._get_agent_recent_actions(agent_type, agent_id)
        
        if not actions:
            return 0.7  # Default good score when no data
        
        # Calculate consistency based on action types
        action_types = [action.get("action_type", "unknown") for action in actions]
        
        # If agent is focusing on its primary functions, it's consistent
        expected_action_types = self._get_expected_action_types(agent_type)
        
        # Count how many actions match expected types
        matching_actions = sum(1 for action_type in action_types if action_type in expected_action_types)
        
        if not action_types:
            return 0.7
        
        # Calculate score based on percentage of expected actions
        consistency_score = matching_actions / len(action_types)
        
        # Ensure reasonable bounds
        return max(0.3, min(1.0, consistency_score))

    def _get_expected_action_types(self, agent_type: str) -> List[str]:
        """Get expected action types for an agent type."""
        # Define expected actions by agent type
        expected_actions = {
            AgentType.STORY_DIRECTOR: [
                "advance_narrative", "create_scene", "introduce_character", 
                "develop_arc", "resolve_arc", "plant_story_seed"
            ],
            AgentType.CONFLICT_ANALYST: [
                "analyze_conflict", "create_conflict", "resolve_conflict",
                "balance_challenge", "scale_difficulty"
            ],
            AgentType.NARRATIVE_CRAFTER: [
                "create_lore", "extend_backstory", "develop_setting",
                "craft_dialogue", "generate_descriptions"
            ],
            AgentType.RESOURCE_OPTIMIZER: [
                "allocate_resources", "balance_economy", "adjust_availability",
                "create_reward", "scale_costs"
            ],
            AgentType.RELATIONSHIP_MANAGER: [
                "adjust_relationship", "create_connection", "develop_tension",
                "resolve_conflict", "introduce_dynamic"
            ],
            AgentType.ACTIVITY_ANALYZER: [
                "analyze_pattern", "identify_trend", "recommend_activity",
                "track_engagement", "predict_behavior"
            ],
            AgentType.MEMORY_MANAGER: [
                "create_memory", "recall_memory", "organize_memories",
                "prune_memories", "consolidate_memories"
            ]
        }
        
        return expected_actions.get(agent_type, ["generic_action"])

    async def _calculate_player_impact(self, agent_type: str, agent_id: Union[int, str]) -> float:
        """Calculate player impact score for an agent."""
        # Get memories specifically about player reactions to this agent
        memory_system = await self.get_memory_system()
        
        search_term = f"player reaction {agent_type}"
        if agent_type == AgentType.NPC:
            npc_name = await self._get_npc_name(int(agent_id))
            search_term = f"player reaction {npc_name}"
        
        reaction_memories = await memory_system.retrieve_memories(
            query=search_term,
            memory_types=["observation", "reflection"],
            scopes=["game"],
            limit=10
        )
        
        if not reaction_memories:
            return 0.5  # Default middle value when no data
        
        # Analyze sentiment in the memories
        positive_sentiment = 0
        negative_sentiment = 0
        
        positive_words = ["enjoyed", "liked", "loved", "positive", "good", "great", "interesting", "engaged"]
        negative_words = ["disliked", "hated", "negative", "bad", "boring", "uninterested", "frustrated"]
        
        for memory in reaction_memories:
            text = memory["memory_text"].lower()
            
            for word in positive_words:
                if word in text:
                    positive_sentiment += 1
            
            for word in negative_words:
                if word in text:
                    negative_sentiment += 1
        
        # Calculate impact score
        if positive_sentiment + negative_sentiment == 0:
            return 0.5  # Neutral when no sentiment detected
        
        # Convert to a 0-1 scale with 0.5 as neutral
        impact_score = 0.5 + (positive_sentiment - negative_sentiment) / (positive_sentiment + negative_sentiment) * 0.5
        
        return max(0.0, min(1.0, impact_score))

    def _calculate_adaptability(
        self,
        actions: List[Dict[str, Any]],
        directives: List[Dict[str, Any]]
    ) -> float:
        """Calculate adaptability score based on response to directives."""
        if not actions or not directives:
            return 0.6  # Default slightly above middle when no data
        
        # Count directives with significant changes
        significant_changes = 0
        total_directives = len(directives)
        
        for directive in directives:
            # Check if directive requested a change in behavior
            if "override" in json.dumps(directive).lower() or "change" in json.dumps(directive).lower():
                # Look for actions that implemented the change
                directive_text = json.dumps(directive).lower()
                
                for action in actions:
                    action_text = json.dumps(action).lower()
                    
                    # Simple matching - check if action addresses directive themes
                    if any(word in action_text for word in directive_text.split()[:5]):
                        significant_changes += 1
                        break
        
        if total_directives == 0:
            return 0.6
        
        # Calculate adaptability score based on response to change directives
        adaptability_score = significant_changes / total_directives
        
        # Ensure reasonable bounds
        return max(0.3, min(1.0, adaptability_score))

    async def _get_npc_specific_metrics(self, npc_id: int) -> Dict[str, Any]:
        """Get NPC-specific performance metrics."""
        try:
            async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                async with pool.acquire() as conn:
                    # Get NPC data
                    row = await conn.fetchrow("""
                        SELECT 
                            relationship_to_player, 
                            times_encountered,
                            dialogue_count,
                            last_interaction
                        FROM NPCStats
                        WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
                    """, npc_id, self.user_id, self.conversation_id)
                    
                    if not row:
                        return {}
                    
                    # Get dialogue engagement metrics
                    dialogue_rows = await conn.fetch("""
                        SELECT AVG(word_count) as avg_length
                        FROM NPCDialogues
                        WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
                    """, npc_id, self.user_id, self.conversation_id)
                    
                    dialogue_metrics = {}
                    if dialogue_rows:
                        dialogue_metrics["average_dialogue_length"] = dialogue_rows[0]["avg_length"] or 0
                    
                    return {
                        "relationship_to_player": row["relationship_to_player"],
                        "times_encountered": row["times_encountered"],
                        "dialogue_count": row["dialogue_count"],
                        "days_since_last_interaction": (datetime.now() - row["last_interaction"]).days if row["last_interaction"] else None,
                        **dialogue_metrics
                    }
        except Exception as e:
            logger.error(f"Error getting NPC-specific metrics: {e}")
            return {}

    async def _get_story_director_metrics(self) -> Dict[str, Any]:
        """Get specific metrics for the Story Director."""
        try:
            # Get narrative progression data
            narrative_data = await self._get_current_narrative_context()
            
            active_arcs = narrative_data.get("active_arcs", [])
            completed_arcs = narrative_data.get("completed_arcs", [])
            
            # Calculate average progression of active arcs
            progression_rates = [arc.get("progress", 0) / 100 for arc in active_arcs if "progress" in arc]
            avg_progression = sum(progression_rates) / len(progression_rates) if progression_rates else 0
            
            # Calculate completion rate
            total_arcs = len(active_arcs) + len(completed_arcs)
            completion_rate = len(completed_arcs) / total_arcs if total_arcs > 0 else 0
            
            return {
                "active_arcs_count": len(active_arcs),
                "completed_arcs_count": len(completed_arcs),
                "average_progression_rate": avg_progression,
                "completion_rate": completion_rate
            }
        except Exception as e:
            logger.error(f"Error getting Story Director metrics: {e}")
            return {}

    async def _get_memory_manager_metrics(self) -> Dict[str, Any]:
        """Get specific metrics for the Memory Manager."""
        try:
            # Get memory system stats
            memory_system = await self.get_memory_system()
            
            # Use memory system's methods to get data
            memory_stats = await memory_system.get_stats()
            
            return {
                "total_memories": memory_stats.get("total_count", 0),
                "observations_count": memory_stats.get("observation_count", 0),
                "reflections_count": memory_stats.get("reflection_count", 0),
                "abstractions_count": memory_stats.get("abstraction_count", 0),
                "average_significance": memory_stats.get("average_significance", 0),
                "pruned_count": memory_stats.get("pruned_count", 0),
                "consolidated_count": memory_stats.get("consolidated_count", 0)
            }
        except Exception as e:
            logger.error(f"Error getting Memory Manager metrics: {e}")
            return {}

    async def _generate_performance_suggestions(
        self,
        agent_type: str,
        agent_id: Union[int, str],
        metrics: Dict[str, Any]
    ) -> List[str]:
        """Generate suggestions for improving agent performance."""
        suggestions = []
        
        # Check success rate
        if metrics.get("success_rate", 0) < 0.6:
            suggestions.append("Improve action success rate by simplifying actions or providing better context")
        
        # Check narrative contribution
        if metrics.get("narrative_contribution", 0) < 0.5:
            suggestions.append("Increase involvement in active narrative arcs")
        
        # Check consistency
        if metrics.get("consistency_score", 0) < 0.6:
            if agent_type == AgentType.NPC:
                suggestions.append("Improve character consistency by better following established personality and backstory")
            else:
                suggestions.append("Focus more on core functions appropriate for this agent type")
        
        # Check player impact
        if metrics.get("player_impact_score", 0) < 0.5:
            suggestions.append("Improve engagement with player and generate more memorable interactions")
        
        # Check response time
        if metrics.get("response_time", 0) > 10:  # More than 10 minutes
            suggestions.append("Reduce response time to directives")
        
        # Agent-specific suggestions
        if agent_type == AgentType.NPC:
            # NPC-specific suggestions
            if metrics.get("relationship_to_player", 0) < 0:
                suggestions.append("Consider developing a more positive relationship with the player")
            
            if metrics.get("dialogue_count", 0) < 5:
                suggestions.append("Increase dialogue interactions with player")
            
            if metrics.get("average_dialogue_length", 0) < 10:
                suggestions.append("Provide more substantial dialogue responses")
        
        elif agent_type == AgentType.STORY_DIRECTOR:
            # Story Director suggestions
            if metrics.get("active_arcs_count", 0) < 2:
                suggestions.append("Introduce more simultaneous narrative arcs")
            
            if metrics.get("average_progression_rate", 0) < 0.1:
                suggestions.append("Increase pace of narrative progression")
        
        elif agent_type == AgentType.MEMORY_MANAGER:
            # Memory Manager suggestions
            if metrics.get("reflections_count", 0) < metrics.get("observations_count", 0) / 10:
                suggestions.append("Generate more reflections from observations")
            
            if metrics.get("consolidated_count", 0) < 1:
                suggestions.append("Perform regular memory consolidation")
        
        return suggestions
        
    # ---------------------------------------------------------------------
    # USER PREFERENCE INTEGRATION
    # ---------------------------------------------------------------------
    async def apply_user_preferences(
        self,
        action_details: Dict[str, Any],
        user_model: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Modify action details based on user preferences from model.
        
        Args:
            action_details: Original action details
            user_model: Optional user model data (will be fetched if not provided)
            
        Returns:
            Modified action details
        """
        # Get user model if not provided
        if not user_model:
            user_model = await self._get_user_model()
        
        if not user_model:
            return action_details  # No modifications if model not available
        
        # Create a modified copy
        modified_details = action_details.copy()
        
        # Apply modifications based on action type
        if "narrative" in modified_details.get("type", "").lower() or "scene" in modified_details.get("type", "").lower():
            # Modify narrative content based on preferences
            modified_details = await self._apply_narrative_preferences(modified_details, user_model)
        
        elif "dialogue" in modified_details.get("type", "").lower():
            # Modify dialogue based on preferences
            modified_details = await self._apply_dialogue_preferences(modified_details, user_model)
        
        elif "conflict" in modified_details.get("type", "").lower():
            # Modify conflict content based on preferences
            modified_details = await self._apply_conflict_preferences(modified_details, user_model)
        
        # Apply general modifications for content intensity
        modified_details = await self._apply_intensity_preferences(modified_details, user_model)
        
        # Apply kink-specific preferences if relevant
        modified_details = await self._apply_kink_preferences(modified_details, user_model)
        
        # Add a flag indicating the action was modified for user preferences
        modified_details["modified_for_user_preferences"] = True
        
        return modified_details

    async def _get_user_model(self) -> Dict[str, Any]:
        """Get the user model from the User Model Manager."""
        try:
            from nyx.nyx_model_manager import UserModelManager
            
            user_model_manager = UserModelManager(self.user_id, self.conversation_id)
            return await user_model_manager.get_user_model()
        except Exception as e:
            logger.error(f"Error getting user model: {e}")
            return {}

    async def _apply_narrative_preferences(
        self,
        action_details: Dict[str, Any],
        user_model: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply narrative preferences from user model."""
        modified_details = action_details.copy()
        
        # Get narrative preferences
        narrative_prefs = user_model.get("narrative_preferences", {})
        
        # Adjust pacing based on preferences
        pacing_pref = narrative_prefs.get("pacing", "medium")
        if "pacing" in modified_details:
            if pacing_pref == "fast":
                modified_details["pacing"] = "accelerated"
            elif pacing_pref == "slow":
                modified_details["pacing"] = "deliberate"
        
        # Adjust focus based on preferences
        focus_pref = narrative_prefs.get("focus", "balanced")
        if "focus" in modified_details:
            if focus_pref == "character":
                modified_details["focus"] = "character_development"
            elif focus_pref == "plot":
                modified_details["focus"] = "plot_progression"
        
        # Content tone adjustments
        tone_pref = narrative_prefs.get("tone", "neutral")
        if "tone" in modified_details:
            modified_details["tone"] = tone_pref
        
        return modified_details

    async def _apply_dialogue_preferences(
        self,
        action_details: Dict[str, Any],
        user_model: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply dialogue preferences from user model."""
        modified_details = action_details.copy()
        
        # Get dialogue preferences
        dialogue_prefs = user_model.get("dialogue_preferences", {})
        
        # Adjust dialogue length
        length_pref = dialogue_prefs.get("length", "medium")
        if "content" in modified_details:
            content = modified_details["content"]
            
            if length_pref == "brief" and len(content.split()) > 50:
                # Shorten dialogue
                try:
                    shorter_content = await generate_text_completion(
                        system_prompt="You are shortening dialogue while preserving key information.",
                        user_prompt=f"Shorten this dialogue to be brief while preserving key information:\n\n{content}",
                        temperature=0.3,
                        max_tokens=len(content.split()) // 2
                    )
                    modified_details["content"] = shorter_content
                except Exception as e:
                    logger.error(f"Error shortening dialogue: {e}")
            
            elif length_pref == "detailed" and len(content.split()) < 30:
                # Expand dialogue
                try:
                    expanded_content = await generate_text_completion(
                        system_prompt="You are expanding dialogue with more detail while preserving character voice.",
                        user_prompt=f"Expand this dialogue to be more detailed while preserving the character's voice:\n\n{content}",
                        temperature=0.4,
                        max_tokens=len(content.split()) * 2
                    )
                    modified_details["content"] = expanded_content
                except Exception as e:
                    logger.error(f"Error expanding dialogue: {e}")
        
        # Adjust dialogue style
        style_pref = dialogue_prefs.get("style", "natural")
        if "style" in modified_details:
            modified_details["style"] = style_pref
        
        return modified_details
        
    async def _apply_conflict_preferences(
        self,
        action_details: Dict[str, Any],
        user_model: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply conflict preferences from user model."""
        modified_details = action_details.copy()
        
        # Get conflict preferences
        conflict_prefs = user_model.get("conflict_preferences", {})
        
        # Adjust difficulty
        difficulty_pref = conflict_prefs.get("difficulty", "medium")
        if "difficulty" in modified_details:
            if difficulty_pref == "easy":
                modified_details["difficulty"] = max(1, modified_details["difficulty"] - 2)
            elif difficulty_pref == "hard":
                modified_details["difficulty"] = min(10, modified_details["difficulty"] + 2)
        
        # Adjust conflict type preference
        conflict_type_pref = conflict_prefs.get("preferred_types", [])
        if conflict_type_pref and "conflict_type" in modified_details:
            # If current conflict type isn't preferred, try to shift it
            if modified_details["conflict_type"] not in conflict_type_pref:
                # If feasible, change to a preferred type
                modified_details["conflict_type"] = conflict_type_pref[0]
                
                # Update conflict description if needed
                if "description" in modified_details:
                    try:
                        new_description = await generate_text_completion(
                            system_prompt="You are modifying conflict descriptions to match a preferred conflict type.",
                            user_prompt=f"Modify this conflict description to focus on {conflict_type_pref[0]} conflict type while preserving the core challenge:\n\n{modified_details['description']}",
                            temperature=0.4,
                            max_tokens=len(modified_details['description'].split()) + 20
                        )
                        modified_details["description"] = new_description
                    except Exception as e:
                        logger.error(f"Error modifying conflict description: {e}")
        
        return modified_details
        
    async def _apply_intensity_preferences(
        self,
        action_details: Dict[str, Any],
        user_model: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply content intensity preferences."""
        modified_details = action_details.copy()
        
        # Get intensity preferences
        content_prefs = user_model.get("content_preferences", {})
        intensity_pref = content_prefs.get("intensity", "medium")
        
        # Check if there's intensity field to modify directly
        if "intensity" in modified_details:
            if intensity_pref == "low":
                modified_details["intensity"] = max(1, modified_details["intensity"] - 2)
            elif intensity_pref == "high":
                modified_details["intensity"] = min(10, modified_details["intensity"] + 2)
        
        # Otherwise, if there's description text, modify language intensity
        elif "description" in modified_details:
            description = modified_details["description"]
            if intensity_pref == "low" and any(word in description.lower() for word in ["intense", "extreme", "violent", "graphic"]):
                try:
                    milder_description = await generate_text_completion(
                        system_prompt="You are toning down intense content descriptions.",
                        user_prompt=f"Rewrite this description with lower intensity while preserving the core meaning:\n\n{description}",
                        temperature=0.3,
                        max_tokens=len(description.split()) + 10
                    )
                    modified_details["description"] = milder_description
                except Exception as e:
                    logger.error(f"Error modifying description intensity: {e}")
            
            elif intensity_pref == "high" and all(word not in description.lower() for word in ["intense", "extreme", "violent", "graphic"]):
                try:
                    stronger_description = await generate_text_completion(
                        system_prompt="You are enhancing content descriptions with more intensity.",
                        user_prompt=f"Rewrite this description with higher intensity while preserving the core meaning:\n\n{description}",
                        temperature=0.4,
                        max_tokens=len(description.split()) + 20
                    )
                    modified_details["description"] = stronger_description
                except Exception as e:
                    logger.error(f"Error modifying description intensity: {e}")
        
        return modified_details
        
    async def _apply_kink_preferences(
        self,
        action_details: Dict[str, Any],
        user_model: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply specific kink preferences if relevant to the action."""
        modified_details = action_details.copy()
        
        # Get kink preferences
        kink_prefs = user_model.get("kink_preferences", {})
        
        # If no kink preferences or not a kink-related action, return unchanged
        if not kink_prefs or not self._is_kink_related_action(modified_details):
            return modified_details
        
        # Extract preferences and limits
        preferred_kinks = kink_prefs.get("preferred", [])
        hard_limits = kink_prefs.get("hard_limits", [])
        
        # Check description for hard limits
        if "description" in modified_details:
            description = modified_details["description"]
            
            # Check for hard limits and replace if found
            for limit in hard_limits:
                if limit.lower() in description.lower():
                    try:
                        modified_description = await generate_text_completion(
                            system_prompt="You are removing content that violates user limits while preserving the action.",
                            user_prompt=f"Rewrite this description to remove any '{limit}' content while preserving the core action:\n\n{description}",
                            temperature=0.3,
                            max_tokens=len(description.split()) + 10
                        )
                        modified_details["description"] = modified_description
                    except Exception as e:
                        logger.error(f"Error removing limit from description: {e}")
        
        # Check if we can enhance with preferred kinks
        if "type" in modified_details and modified_details["type"] == "scene_setup":
            # For scene setup, we can incorporate preferred elements
            if preferred_kinks and "elements" in modified_details:
                elements = modified_details["elements"]
                # Add a preferred kink if not already present
                for kink in preferred_kinks:
                    if not any(kink.lower() in elem.lower() for elem in elements):
                        elements.append(f"subtle {kink} dynamic")
                modified_details["elements"] = elements
        
        return modified_details
    
    def _is_kink_related_action(self, action_details: Dict[str, Any]) -> bool:
        """Determine if an action is potentially kink-related."""
        # Check action type
        kink_related_types = ["intimate_scene", "relationship", "scene_setup", "dialogue"]
        if "type" in action_details and action_details["type"] in kink_related_types:
            return True
        
        # Check for kink-related keywords in description
        if "description" in action_details:
            description = action_details["description"].lower()
            kink_keywords = ["dominance", "submission", "intimate", "bdsm", "fetish", "kink", 
                            "collar", "leash", "bondage", "restraint", "punishment", "discipline"]
            if any(keyword in description for keyword in kink_keywords):
                return True
        
        return False
