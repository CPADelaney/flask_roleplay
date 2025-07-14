# nyx/nyx_governance.py

"""
Unified governance system for Nyx to control all agents (NPCs and beyond).

IMPORTANT FIX: This module previously had a circular dependency with LoreSystem.
The circular dependency has been resolved by using dependency injection:
1. Create LoreSystem instance (without calling get_central_governance)
2. Set this governor on the LoreSystem via set_governor()
3. Then initialize the LoreSystem

This ensures a clean one-way flow: Governor → LoreSystem
"""

import logging
import json
import asyncio
import inspect
import time
import importlib
import pkgutil
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set, TYPE_CHECKING
from utils.caching import CacheManager, CACHE_TTL
from nyx.llm_integration import generate_text_completion

import asyncpg

# The agent trace utility
from agents import trace, RunContextWrapper

# Database connection helper
from db.connection import get_db_connection_context

# --- MERGE: Import the new LoreSystem ---
# Use TYPE_CHECKING to avoid circular import
if TYPE_CHECKING:
    from lore.core.lore_system import LoreSystem

from nyx.constants import DirectiveType, DirectivePriority, AgentType

# Memory system references
from memory.wrapper import MemorySystem
from memory.memory_nyx_integration import get_memory_nyx_bridge

# Integration with LLM services for reasoning
from nyx.llm_integration import generate_text_completion, generate_reflection

# Caching utilities
from utils.caching import CACHE_TTL, NPC_DIRECTIVE_CACHE, AGENT_DIRECTIVE_CACHE

logger = logging.getLogger(__name__)

_directive_cache_manager = CacheManager(
    name="agent_directives",
    max_size=1000,
    ttl=CACHE_TTL.DIRECTIVES
)

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
    INFORMATION = "information"  # New addition


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
    Enhanced unified governance system for Nyx to control all agents with agentic capabilities.
    
    This class provides:
      1. Central authority over all agents with enhanced coordination
      2. Goal-oriented agent coordination and planning
      3. Adaptive decision making with learning capabilities
      4. Performance monitoring and feedback loops
      5. Cross-agent communication and collaboration
      6. Dynamic goal prioritization and resource allocation
      7. Agent learning and adaptation tracking
      8. Enhanced conflict resolution with context awareness
      9. Temporal consistency enforcement
      10. User preference integration
    """

    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # --- MERGE: Add a placeholder for the LoreSystem ---
        # Will be initialized in _initialize_systems() to avoid circular dependency
        self.lore_system: Optional[Any] = None

        # Core systems and state
        self.memory_system = None
        self.game_state = None
        self.registered_agents: Dict[str, Dict[str, Any]] = {}     # {agent_type: {agent_id: instance}}

        # Multi-agent analytics
        self.active_goals: List[Dict[str, Any]] = []
        self.agent_goals: Dict[str, Dict[str, Any]] = {}           # {agent_type: {agent_id: ...}}
        self.agent_performance: Dict[str, Dict[str, Any]] = {}     # {agent_type: {agent_id: ...}}
        self.agent_learning: Dict[str, Dict[str, Any]] = {}        # {agent_type: {agent_id: ...}}
        self.coordination_history: List[Dict[str, Any]] = []
        self.memory_integration = None  # <-- ADD THIS

        self.memory_graph = None  # Joint memory graph for shared memories
    
        # Learning state
        self.strategy_effectiveness: Dict[str, Any] = {}
        self.adaptation_patterns: Dict[str, Any] = {}
        self.collaboration_success: Dict[str, Dict[str, Any]] = {}

        # Disagreement history
        self.disagreement_history: Dict[str, List[Dict[str, Any]]] = {}
        self.disagreement_thresholds: Dict[str, float] = {
            "narrative_impact": 0.7,
            "character_consistency": 0.8,
            "world_integrity": 0.9,
            "player_experience": 0.6
        }

        # Directive/action reports
        self.directives: Dict[str, Dict[str, Any]] = {}
        self.action_reports: Dict[str, Dict[str, Any]] = {}
        
        # Flag to track initialization
        self._initialized = False

    async def initialize(self) -> "NyxUnifiedGovernor":
        """
        Initialize the governance system asynchronously.
        This must be called after creating a new NyxUnifiedGovernor instance.
        """
        if hasattr(self, '_initialized') and self._initialized:
            return self

        # --- MERGE: Initialize the lore system is now done in _initialize_systems ---
        # No need to duplicate it here

        # Initialize other systems
        await self._initialize_systems()
        self._initialized = True
        return self

    async def _initialize_systems(self):
        """Initialize memory system, game state, and discover agents."""
        # Import LoreSystem locally to avoid circular import
        from lore.core.lore_system import LoreSystem
        
        # Get an instance of the LoreSystem
        self.lore_system = LoreSystem.get_instance(self.user_id, self.conversation_id)
        
        # Set the governor on the lore system (dependency injection)
        self.lore_system.set_governor(self)
        
        # Initialize the lore system WITH the governor reference
        await self.lore_system.initialize(governor=self)  # Pass self as governor
    
        # Initialize other systems
        self.memory_system = await get_memory_nyx_bridge(self.user_id, self.conversation_id)
        
        # ADD THIS: Initialize memory integration
        from memory.memory_integration import MemoryIntegration
        self.memory_integration = MemoryIntegration(self.user_id, self.conversation_id)
        await self.memory_integration.initialize()

        from nyx.integrate import JointMemoryGraph
        self.memory_graph = JointMemoryGraph(self.user_id, self.conversation_id)
        
        self.game_state = await self.initialize_game_state()
        await self.discover_and_register_agents()
        await self._load_initial_state()

    async def _load_initial_state(self):
        """Load goals and agent state from memory."""
        goal_memories = await self.memory_system.recall(
            entity_type="nyx",
            entity_id=self.conversation_id,
            query="active goals",
            context="system goals",
            limit=10
        )
        self.active_goals = self._extract_goals_from_memories(goal_memories.get("memories", []))

        # Load all agents' goals
        for agent_type, agents_dict in self.registered_agents.items():
            for agent_id, agent in agents_dict.items():
                if hasattr(agent, "get_active_goals"):
                    self.agent_goals.setdefault(agent_type, {})[agent_id] = await agent.get_active_goals()
                else:
                    self.agent_goals.setdefault(agent_type, {})[agent_id] = []

        await self._update_performance_metrics()
        await self._load_learning_state()

    async def initialize_game_state(self, *, force: bool = False) -> Dict[str, Any]:
        """
        Fetch and return the game-state snapshot for this user/conversation.
    
        If it has already been loaded during this governor’s lifetime, return
        the cached copy unless `force=True`.
        """
        if getattr(self, "game_state", None) and not force:
            logger.info(
                f"[GAME-STATE] Already initialized for {self.user_id}:{self.conversation_id}; skipping."
            )
            return self.game_state
    
        logger.info(
            f"[GAME-STATE] Initializing for {self.user_id}, conversation {self.conversation_id}"
        )
    
        game_state = {
            "user_id": self.user_id,
            "conversation_id": self.conversation_id,
            "current_location": None,
            "current_npcs": [],
            "current_time": None,
            "active_quests": [],
            "player_stats": {},
            "narrative_state": {},
            "world_state": {},
        }
    
        try:
            async with get_db_connection_context() as conn:
                # Get current location
                row = await conn.fetchrow("""
                    SELECT value FROM CurrentRoleplay
                    WHERE user_id = $1 AND conversation_id = $2 AND key = 'CurrentLocation'
                    LIMIT 1
                """, self.user_id, self.conversation_id)
                if row: game_state["current_location"] = row["value"]
                
                # Get current time
                row = await conn.fetchrow("""
                    SELECT value FROM CurrentRoleplay
                    WHERE user_id = $1 AND conversation_id = $2 AND key = 'CurrentTime'
                    LIMIT 1
                """, self.user_id, self.conversation_id)
                if row: game_state["current_time"] = row["value"]

                row = await conn.fetchrow("""
                    SELECT * FROM PlayerStats
                    WHERE user_id = $1 AND conversation_id = $2 AND player_name = 'Chase'
                    LIMIT 1
                """, self.user_id, self.conversation_id)
                if row: game_state["player_stats"] = dict(row)

                rows = await conn.fetch("""
                    SELECT npc_id, npc_name, current_location FROM NPCStats
                    WHERE user_id = $1 AND conversation_id = $2
                """, self.user_id, self.conversation_id)
                game_state["current_npcs"] = [dict(r) for r in rows]

                rows = await conn.fetch("""
                    SELECT * FROM Quests
                    WHERE user_id = $1 AND conversation_id = $2 AND status = 'In Progress'
                """, self.user_id, self.conversation_id)
                game_state["active_quests"] = [dict(r) for r in rows]

                logger.info(
                    f"Game state initialized with "
                    f"{len(game_state['current_npcs'])} NPCs and "
                    f"{len(game_state['active_quests'])} quests."
                )
        except Exception as e:
            logger.error(f"Error initializing game state: {e}")
    
        # cache it so the next call can short-circuit
        self.game_state = game_state
        return game_state

    async def discover_and_register_agents(self):
        """
        Discover and register available agents in the system dynamically.
        """
        logger.info(f"Discovering and registering agents for user {self.user_id}, conversation {self.conversation_id}")
        
        registered_count = 0
        
        # Define agent modules and their expected classes
        agent_modules = [
            ("story_agent.story_director_agent", "StoryDirector", AgentType.STORY_DIRECTOR),
            ("logic.universal_updater_agent", "UniversalUpdaterAgent", AgentType.UNIVERSAL_UPDATER),
            ("agents.scene_manager", "SceneManagerAgent", AgentType.SCENE_MANAGER),
            ("agents.conflict_analyst", "ConflictAnalystAgent", AgentType.CONFLICT_ANALYST),
            ("agents.narrative_crafter", "NarrativeCrafterAgent", AgentType.NARRATIVE_CRAFTER),
            ("agents.resource_optimizer", "ResourceOptimizerAgent", AgentType.RESOURCE_OPTIMIZER),
            ("agents.relationship_manager", "RelationshipManagerAgent", AgentType.RELATIONSHIP_MANAGER),
            ("agents.activity_analyzer", "ActivityAnalyzerAgent", AgentType.ACTIVITY_ANALYZER),
            ("agents.memory_manager", "MemoryManagerAgent", AgentType.MEMORY_MANAGER),
        ]
        
        for module_path, class_name, agent_type in agent_modules:
            try:
                # Dynamically import the module
                module = importlib.import_module(module_path)
                
                # Get the agent class
                agent_class = getattr(module, class_name, None)
                
                if agent_class:
                    # Create instance
                    agent_instance = agent_class(self.user_id, self.conversation_id)
                    
                    # Register with governance
                    agent_id = f"{agent_type}_{self.conversation_id}"
                    await self.register_agent(
                        agent_type=agent_type,
                        agent_instance=agent_instance,
                        agent_id=agent_id
                    )
                    
                    registered_count += 1
                    logger.info(f"Registered {agent_type} agent: {class_name}")
                else:
                    logger.warning(f"Class {class_name} not found in module {module_path}")
                    
            except ImportError as e:
                logger.debug(f"Could not import {module_path}: {e}")
            except Exception as e:
                logger.warning(f"Could not register agent from {module_path}: {e}")
        
        # Also discover and register NPC agents
        try:
            async with get_db_connection_context() as conn:
                # Get active NPCs that might need agents
                active_npcs = await conn.fetch("""
                    SELECT npc_id, npc_name FROM NPCStats
                    WHERE user_id = $1 AND conversation_id = $2 AND is_active = TRUE
                    LIMIT 10
                """, self.user_id, self.conversation_id)
                
                for npc in active_npcs:
                    try:
                        from npcs.npc_agent import NPCAgent
                        npc_agent = NPCAgent(self.user_id, self.conversation_id, npc["npc_id"])
                        
                        await self.register_agent(
                            agent_type=AgentType.NPC,
                            agent_instance=npc_agent,
                            agent_id=f"npc_{npc['npc_id']}"
                        )
                        
                        registered_count += 1
                        logger.info(f"Registered NPC agent for {npc['npc_name']} (ID: {npc['npc_id']})")
                        
                    except Exception as e:
                        logger.debug(f"Could not register NPC agent for {npc['npc_name']}: {e}")
                        
        except Exception as e:
            logger.warning(f"Could not discover NPC agents: {e}")
        
        logger.info(f"Agent discovery completed. Registered {registered_count} agents.")
        return registered_count > 0
    # ... rest of the file remains the same ...
    
    async def _update_performance_metrics(self):
        """Update per-agent performance metrics."""
        for agent_type, agents_dict in self.registered_agents.items():
            if agent_type not in self.agent_performance:
                self.agent_performance[agent_type] = {}
                
            for agent_id, agent in agents_dict.items():
                # Default metrics structure based on UniversalUpdaterAgent
                default_metrics = {
                    "updates_processed": 0,
                    "success_rate": 0.0,
                    "average_processing_time": 0.0,
                    "strategies": {},
                    "last_action_time": None,
                    "total_actions": 0,
                    "successful_actions": 0,
                    "failed_actions": 0,
                    "coordination_score": 0.0
                }
                
                if hasattr(agent, 'get_performance_metrics'):
                    try:
                        metrics = await agent.get_performance_metrics()
                        # Merge with defaults to ensure all fields exist
                        for key, value in default_metrics.items():
                            if key not in metrics:
                                metrics[key] = value
                    except:
                        metrics = default_metrics
                else:
                    metrics = default_metrics
                    
                self.agent_performance[agent_type][agent_id] = metrics
                
                # Update strategy effectiveness tracking
                for strategy, data in metrics.get("strategies", {}).items():
                    if strategy not in self.strategy_effectiveness:
                        self.strategy_effectiveness[strategy] = {
                            "success": 0,
                            "total": 0,
                            "agents": {}
                        }
                    self.strategy_effectiveness[strategy]["success"] += data.get("success", 0)
                    self.strategy_effectiveness[strategy]["total"] += data.get("total", 0)
                    self.strategy_effectiveness[strategy]["agents"][f"{agent_type}:{agent_id}"] = data

    async def _load_learning_state(self):
        """Load and aggregate learning/adaptation patterns for all agents."""
        for agent_type, agents_dict in self.registered_agents.items():
            if agent_type not in self.agent_learning:
                self.agent_learning[agent_type] = {}
                
            for agent_id, agent in agents_dict.items():
                # Default learning state structure
                default_learning = {
                    "patterns": {},
                    "adaptations": [],
                    "learning_rate": 0.0,
                    "total_learnings": 0,
                    "successful_adaptations": 0,
                    "failed_adaptations": 0
                }
                
                if hasattr(agent, "get_learning_state"):
                    try:
                        learning_data = await agent.get_learning_state()
                        # Merge with defaults
                        for key, value in default_learning.items():
                            if key not in learning_data:
                                learning_data[key] = value
                    except:
                        learning_data = default_learning
                else:
                    learning_data = default_learning
                    
                self.agent_learning[agent_type][agent_id] = learning_data
    
                # Aggregate patterns
                for pattern, data in learning_data.get("patterns", {}).items():
                    if pattern not in self.adaptation_patterns:
                        self.adaptation_patterns[pattern] = {
                            "success": 0,
                            "total": 0,
                            "agents": {}
                        }
                    self.adaptation_patterns[pattern]["success"] += data.get("success", 0)
                    self.adaptation_patterns[pattern]["total"] += data.get("total", 0)
                    self.adaptation_patterns[pattern]["agents"][f"{agent_type}:{agent_id}"] = data

    async def coordinate_agents_for_goal(
        self,
        goal: Dict[str, Any],
        timeframe: str = "1 hour"
    ) -> Dict[str, Any]:
        """
        Coordinate agents to achieve a specific goal.
        
        Args:
            goal: The goal to achieve
            timeframe: Expected timeframe for goal completion
        """
        # Analyze goal requirements
        requirements = await self._analyze_goal_requirements(goal)
        
        # Identify relevant agents
        relevant_agents = await self._identify_relevant_agents(requirements)
        
        # Generate coordination plan
        plan = await self._generate_coordination_plan(
            goal,
            requirements,
            relevant_agents,
            timeframe
        )
        
        # Execute plan
        result = await self._execute_coordination_plan(plan)
        
        # Record coordination
        await self._record_coordination(goal, plan, result)
        
        return result

    async def register_agent(self, agent_type: str, agent_instance: Any, agent_id: str) -> Dict[str, Any]:
        """Register an agent instance under (type, id)."""
        if agent_type not in self.registered_agents:
            self.registered_agents[agent_type] = {}
        self.registered_agents[agent_type][agent_id] = agent_instance
        logger.info(f"Agent registered: {agent_type}/{agent_id}")
        return {
            "success": True,
            "agent_type": agent_type,
            "agent_id": agent_id,
            "timestamp": datetime.now().isoformat()
        }
    
    async def issue_directive(self, agent_type: str, agent_id: str, directive_type: str, directive_data: Dict[str, Any],
                             priority: int = DirectivePriority.MEDIUM, duration_minutes: int = 60) -> Dict[str, Any]:
        """Issue a time-limited directive to an agent."""
        if (agent_type not in self.registered_agents or
            agent_id not in self.registered_agents[agent_type]):
            return {"success": False, "reason": f"Agent not found: {agent_type} / {agent_id}"}
        directive_id = f"{agent_type}_{agent_id}_{int(datetime.now().timestamp())}"
        expiration = datetime.now() + timedelta(minutes=duration_minutes)

        self.directives[directive_id] = {
            "id": directive_id,
            "agent_type": agent_type,
            "agent_id": agent_id,
            "type": directive_type,
            "data": directive_data,
            "priority": priority,
            "issued_at": datetime.now().isoformat(),
            "expires_at": expiration.isoformat(),
            "status": "active"
        }
        logger.info(f"Directive issued: {directive_id} to {agent_type}/{agent_id}")
        return {"success": True, "directive_id": directive_id, "expires_at": expiration.isoformat()}

    async def process_agent_action_report(self, agent_type: str, agent_id: str, action: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Store a report of an action taken by an agent."""
        report_id = f"{agent_type}_{agent_id}_{int(datetime.now().timestamp())}"
        self.action_reports[report_id] = {
            "id": report_id,
            "agent_type": agent_type,
            "agent_id": agent_id,
            "action": action,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        logger.info(f"Action report processed: {report_id} from {agent_type} / {agent_id}")
        return {"success": True, "report_id": report_id}

        
    async def _analyze_goal_requirements(self, goal: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze requirements for achieving a goal."""
        requirements = {
            "agent_types": set(),
            "capabilities": set(),
            "resources": set(),
            "constraints": [],
            "dependencies": []
        }
        
        # Analyze goal description
        description = goal.get("description", "").lower()
        
        # Identify required agent types
        for agent_type in AgentType.__dict__.values():
            if isinstance(agent_type, str) and agent_type in description:
                requirements["agent_types"].add(agent_type)
        
        # Identify required capabilities
        capabilities = {
            "memory": ["memory", "remember", "recall"],
            "planning": ["plan", "strategy", "coordinate"],
            "learning": ["learn", "adapt", "improve"],
            "conflict": ["resolve", "handle", "manage"],
            "narrative": ["story", "narrative", "plot"]
        }
        
        for cap, keywords in capabilities.items():
            if any(keyword in description for keyword in keywords):
                requirements["capabilities"].add(cap)
        
        # Identify resource requirements
        resources = {
            "memory": ["memory", "storage"],
            "computation": ["process", "compute", "calculate"],
            "time": ["time", "duration", "period"]
        }
        
        for resource, keywords in resources.items():
            if any(keyword in description for keyword in keywords):
                requirements["resources"].add(resource)
        
        return requirements
    
    async def _identify_relevant_agents(self, requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify agents relevant to goal requirements."""
        relevant_agents = []
        
        # Define standard capabilities for known agent types
        agent_capabilities = {
            AgentType.UNIVERSAL_UPDATER: ["narrative_analysis", "state_extraction", "state_updating"],
            AgentType.STORY_DIRECTOR: ["narrative_planning", "plot_development", "pacing_control"],
            AgentType.CONFLICT_ANALYST: ["conflict_detection", "resolution_planning", "stake_analysis"],
            AgentType.NARRATIVE_CRAFTER: ["content_creation", "dialogue_generation", "scene_crafting"],
            AgentType.RESOURCE_OPTIMIZER: ["resource_management", "optimization", "allocation"],
            AgentType.RELATIONSHIP_MANAGER: ["relationship_tracking", "social_dynamics", "bond_management"],
            AgentType.MEMORY_MANAGER: ["memory_storage", "memory_recall", "memory_organization"],
            AgentType.SCENE_MANAGER: ["scene_composition", "atmosphere_control", "transition_management"],
            AgentType.ACTIVITY_ANALYZER: ["activity_tracking", "pattern_recognition", "behavior_analysis"],
            AgentType.NPC: ["dialogue", "movement", "interaction", "emotion"]
        }
        
        for agent_type, agents_dict in self.registered_agents.items():
            for agent_id, agent in agents_dict.items():
                # Check if agent type is required
                if agent_type in requirements["agent_types"]:
                    relevant_agents.append({
                        "type": agent_type,
                        "id": agent_id,
                        "instance": agent,
                        "relevance": 1.0
                    })
                    continue
                    
                # Get capabilities either from agent method or predefined list
                capabilities = []
                if hasattr(agent, 'get_capabilities'):
                    try:
                        capabilities = await agent.get_capabilities()
                    except:
                        capabilities = agent_capabilities.get(agent_type, [])
                else:
                    capabilities = agent_capabilities.get(agent_type, [])
                
                # Capabilities match
                capability_match = sum(
                    1 for cap in requirements["capabilities"]
                    if cap in capabilities
                ) / len(requirements["capabilities"]) if requirements["capabilities"] else 0
                
                if capability_match > 0.5:
                    relevant_agents.append({
                        "type": agent_type,
                        "id": agent_id,
                        "instance": agent,
                        "relevance": capability_match,
                        "capabilities": capabilities
                    })
        
        relevant_agents.sort(key=lambda x: x["relevance"], reverse=True)
        return relevant_agents
    
    async def _generate_coordination_plan(
        self,
        goal: Dict[str, Any],
        requirements: Dict[str, Any],
        relevant_agents: List[Dict[str, Any]],
        timeframe: str
    ) -> Dict[str, Any]:
        """Generate a coordination plan for achieving the goal."""
        plan = {
            "goal": goal,
            "timeframe": timeframe,
            "phases": [],
            "agent_assignments": {},
            "dependencies": [],
            "success_criteria": []
        }
        
        # Generate phases based on goal complexity
        complexity = len(requirements["capabilities"]) + len(requirements["resources"])
        num_phases = min(complexity, 5)  # Cap at 5 phases
        
        for phase_num in range(num_phases):
            phase = {
                "number": phase_num + 1,
                "description": f"Phase {phase_num + 1}",
                "agents": [],
                "tasks": [],
                "dependencies": []
            }
            
            # Assign agents to phase
            phase_agents = relevant_agents[phase_num::num_phases]
            for agent in phase_agents:
                phase["agents"].append(agent["type"])
                plan["agent_assignments"][agent["type"]] = {
                    "phase": phase_num + 1,
                    "role": self._determine_agent_role(agent, requirements)
                }
            
            # Generate tasks for phase
            phase["tasks"] = await self._generate_phase_tasks(
                phase_num,
                requirements,
                phase_agents
            )
            
            plan["phases"].append(phase)
        
        # Set success criteria
        plan["success_criteria"] = await self._generate_success_criteria(goal, requirements)
        
        return plan
    
    def _determine_agent_role(
        self,
        agent: Dict[str, Any],
        requirements: Dict[str, Any]
    ) -> str:
        """Determine the role an agent should play in the plan."""
        agent_type = agent["type"]
        capabilities = agent.get("capabilities", [])
        
        # Primary roles based on agent type
        if agent_type == AgentType.STORY_DIRECTOR:
            return "narrative_lead"
        elif agent_type == AgentType.CONFLICT_ANALYST:
            return "conflict_manager"
        elif agent_type == AgentType.NARRATIVE_CRAFTER:
            return "content_creator"
        elif agent_type == AgentType.RESOURCE_OPTIMIZER:
            return "resource_manager"
        elif agent_type == AgentType.RELATIONSHIP_MANAGER:
            return "relationship_coordinator"
        elif agent_type == AgentType.MEMORY_MANAGER:
            return "memory_specialist"
        
        # Fallback based on capabilities
        if "memory" in capabilities:
            return "memory_support"
        elif "planning" in capabilities:
            return "planning_support"
        elif "learning" in capabilities:
            return "learning_support"
        
        return "general_support"
    
    async def _generate_phase_tasks(
        self,
        phase_num: int,
        requirements: Dict[str, Any],
        phase_agents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate tasks for a specific phase."""
        tasks = []
        
        # Base tasks on phase number and requirements
        if phase_num == 0:
            # Initialization phase
            tasks.extend([
                {
                    "type": "setup",
                    "description": "Initialize required systems",
                    "agents": [a["type"] for a in phase_agents if "memory" in a.get("capabilities", [])]
                },
                {
                    "type": "planning",
                    "description": "Develop detailed execution plan",
                    "agents": [a["type"] for a in phase_agents if "planning" in a.get("capabilities", [])]
                }
            ])
        elif phase_num == len(requirements["capabilities"]) - 1:
            # Final phase
            tasks.extend([
                {
                    "type": "verification",
                    "description": "Verify goal achievement",
                    "agents": [a["type"] for a in phase_agents]
                },
                {
                    "type": "cleanup",
                    "description": "Clean up resources",
                    "agents": [a["type"] for a in phase_agents if "resource" in a.get("capabilities", [])]
                }
            ])
        else:
            # Middle phases
            tasks.extend([
                {
                    "type": "execution",
                    "description": f"Execute phase {phase_num + 1} tasks",
                    "agents": [a["type"] for a in phase_agents]
                },
                {
                    "type": "coordination",
                    "description": "Coordinate with other phases",
                    "agents": [a["type"] for a in phase_agents if "planning" in a.get("capabilities", [])]
                }
            ])
        
        return tasks

    async def handle_directive(
        self,
        directive_type: str,
        directive_data: Dict[str, Any],
        agent_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Route a directive to the appropriate handler.
        
        Args:
            directive_type: Type of directive
            directive_data: Directive data
            agent_type: Type of agent to handle the directive (optional)
            
        Returns:
            Directive handling result
        """
        try:
            # Check if we have handlers for this directive type
            if not hasattr(self, "directive_handlers") or directive_type not in self.directive_handlers:
                return {
                    "success": False,
                    "message": f"No handlers registered for directive type: {directive_type}"
                }
            
            # If agent_type is specified, use that handler
            if agent_type and agent_type in self.directive_handlers[directive_type]:
                handler = self.directive_handlers[directive_type][agent_type]
                return await handler.handle_directive({
                    "type": directive_type,
                    "data": directive_data
                })
            
            # Otherwise, try all registered handlers for this directive type
            for handler_agent_type, handler in self.directive_handlers[directive_type].items():
                try:
                    result = await handler.handle_directive({
                        "type": directive_type,
                        "data": directive_data
                    })
                    if result.get("success", False):
                        return result
                except Exception as e:
                    logger.warning(f"Handler {handler_agent_type} failed to handle {directive_type} directive: {str(e)}")
            
            return {
                "success": False,
                "message": f"No handler successfully processed directive type: {directive_type}"
            }
        except Exception as e:
            logger.error(f"Error handling directive: {str(e)}", exc_info=True)
            return {
                "success": False,
                "message": f"Error handling directive: {str(e)}"
            }
    
    async def _generate_success_criteria(
        self,
        goal: Dict[str, Any],
        requirements: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate success criteria for the goal."""
        criteria = []
        
        # Basic goal completion
        criteria.append({
            "type": "completion",
            "description": "Goal is fully completed",
            "measure": "boolean"
        })
        
        # Resource utilization
        if requirements["resources"]:
            criteria.append({
                "type": "resource_efficiency",
                "description": "Resources are used efficiently",
                "measure": "percentage",
                "target": 0.8
            })
        
        # Agent coordination
        if len(requirements["agent_types"]) > 1:
            criteria.append({
                "type": "coordination_quality",
                "description": "Agents coordinate effectively",
                "measure": "score",
                "target": 0.7
            })
        
        # Learning outcomes
        if "learning" in requirements["capabilities"]:
            criteria.append({
                "type": "learning_achievement",
                "description": "Agents demonstrate learning",
                "measure": "score",
                "target": 0.6
            })
        
        return criteria

    async def get_agent_directives(self, agent_type: str, agent_id: Union[int, str]) -> List[Dict[str, Any]]:
        """Get directives for a specific agent with proper caching."""
        # Get the cache key for this agent
        cache_key = f"agent_directives:{agent_type}:{agent_id}"
        
        # Try to get from cache first
        cached_directives = await _directive_cache_manager.get(cache_key)
        if cached_directives is not None:
            return cached_directives
        
        # Fetch directives if not in cache
        active_directives = []
        now = datetime.now()
        
        # Filter directives for this agent type and ID
        for directive_id, directive in self.directives.items():
            if (directive.get("agent_type") == agent_type and 
                str(directive.get("agent_id")) == str(agent_id) and
                directive.get("status") == "active"):
                
                # Check if still active (not expired)
                if "expires_at" in directive:
                    expires_at = datetime.fromisoformat(directive["expires_at"])
                    if expires_at <= now:
                        continue  # Skip expired directives
                
                active_directives.append(directive)
        
        # Sort by priority (highest first)
        active_directives.sort(key=lambda d: d.get("priority", 0), reverse=True)
        
        # Cache the result
        await _directive_cache_manager.set(cache_key, active_directives, ttl=CACHE_TTL.DIRECTIVES)
        
        return active_directives
    
    async def _execute_coordination_plan(
        self,
        plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the coordination plan."""
        result = {
            "success": False,
            "phases": [],
            "metrics": {},
            "learnings": []
        }
        
        # Execute each phase
        for phase in plan["phases"]:
            phase_result = await self._execute_phase(phase, plan)
            result["phases"].append(phase_result)
            
            # Check for phase failure
            if not phase_result["success"]:
                result["success"] = False
                break
        
        # Calculate final metrics
        result["metrics"] = await self._calculate_plan_metrics(plan, result["phases"])
        
        # Record learnings
        result["learnings"] = await self._extract_plan_learnings(plan, result["phases"])
        
        # Determine overall success
        result["success"] = await self._evaluate_plan_success(plan, result)
        
        return result
    
    async def _execute_phase(
        self,
        phase: Dict[str, Any],
        plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single phase of the plan."""
        result = {
            "phase": phase["number"],
            "success": False,
            "tasks": [],
            "metrics": {}
        }
        
        # Execute each task
        for task in phase["tasks"]:
            task_result = await self._execute_task(task, phase, plan)
            result["tasks"].append(task_result)
            
            # Check for task failure
            if not task_result["success"]:
                result["success"] = False
                break
        
        # Calculate phase metrics
        result["metrics"] = await self._calculate_phase_metrics(phase, result["tasks"])
        
        # Determine phase success
        result["success"] = await self._evaluate_phase_success(phase, result)
        
        return result
    
    async def _execute_task(self, task: Dict[str, Any], phase: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single task within a phase.
        Select agents by agent_type (task['agents']) and run the appropriate method.
        """
        result = {
            "task": task["type"],
            "success": False,
            "output": None,
            "metrics": {}
        }
        try:
            # Gather all agent instances listed by type in task["agents"]
            agents = []
            for agent_type in task["agents"]:
                agents.extend(self.registered_agents.get(agent_type, {}).values())
            
            # Execute task with agents
            task_output = await self._execute_with_agents(task, agents, phase, plan)
            result["output"] = task_output
            result["success"] = True
            result["metrics"] = await self._calculate_task_metrics(task, task_output, agents)
        except Exception as e:
            logger.error(f"Task execution failed: {str(e)}")
            result["success"] = False
        return result
    
    async def _execute_with_agents(
        self,
        task: Dict[str, Any],
        agents: List[Any],
        phase: Dict[str, Any],
        plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a task with multiple agents."""
        # Prepare task context
        context = {
            "task": task,
            "phase": phase,
            "plan": plan,
            "agents": [a.__class__.__name__ for a in agents]
        }
        
        # Execute task based on type
        if task["type"] == "setup":
            return await self._execute_setup_task(agents, context)
        elif task["type"] == "planning":
            return await self._execute_planning_task(agents, context)
        elif task["type"] == "execution":
            return await self._execute_execution_task(agents, context)
        elif task["type"] == "coordination":
            return await self._execute_coordination_task(agents, context)
        elif task["type"] == "verification":
            return await self._execute_verification_task(agents, context)
        elif task["type"] == "cleanup":
            return await self._execute_cleanup_task(agents, context)
        
        raise ValueError(f"Unknown task type: {task['type']}")
    
    async def _execute_setup_task(self, agents: List[Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute setup task with agents."""
        setup_results = []
        
        for agent in agents:
            result = None
            
            # Try different initialization methods
            if hasattr(agent, "initialize_systems"):
                result = await agent.initialize_systems(context)
            elif hasattr(agent, "initialize"):
                result = await agent.initialize()
            elif hasattr(agent, "handle_directive"):
                # Use directive-based initialization
                result = await agent.handle_directive({
                    "type": DirectiveType.ACTION,
                    "data": {
                        "action": "initialize",
                        "context": context
                    }
                })
            
            if result:
                setup_results.append({
                    "agent": agent.__class__.__name__,
                    "status": "initialized",
                    "result": result
                })
            else:
                setup_results.append({
                    "agent": agent.__class__.__name__,
                    "status": "skipped",
                    "reason": "No initialization method found"
                })
        
        return {
            "status": "completed",
            "results": setup_results,
            "initialized_count": sum(1 for r in setup_results if r["status"] == "initialized")
        }
    
    async def _execute_planning_task(self, agents: List[Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute planning task with planning-capable agents."""
        plans = []
        
        for agent in agents:
            plan = None
            
            if hasattr(agent, "generate_plan"):
                plan = await agent.generate_plan(context)
            elif hasattr(agent, "handle_directive"):
                # Use directive-based planning
                result = await agent.handle_directive({
                    "type": DirectiveType.ACTION,
                    "data": {
                        "action": "generate_plan",
                        "context": context
                    }
                })
                if result and result.get("success"):
                    plan = result.get("plan")
            
            if plan:
                plans.append(plan)
        
        # Merge plans intelligently
        merged_plan = await self._merge_plans(plans, context)
        
        return {
            "status": "completed",
            "merged_plan": merged_plan,
            "plan_count": len(plans)
        }
    
    async def _merge_plans(self, plans: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple plans with intelligent conflict resolution."""
        merged = {
            "tasks": [],
            "dependencies": [],
            "timeline": [],
            "resources": {},
            "conflicts": [],
            "resolution_strategy": "collaborative",
            "estimated_duration": 0
        }
        
        # Track unique identifiers
        task_registry = {}  # task_id -> task
        dependency_registry = {}  # (from, to) -> dependency
        
        # First pass: collect all unique tasks
        for plan in plans:
            for task in plan.get("tasks", []):
                task_id = task.get("id", f"{task.get('type', 'unknown')}_{len(task_registry)}")
                
                if task_id in task_registry:
                    # Task conflict - merge or choose better version
                    existing = task_registry[task_id]
                    
                    # Merge agents lists
                    existing_agents = set(existing.get("agents", []))
                    new_agents = set(task.get("agents", []))
                    existing["agents"] = list(existing_agents.union(new_agents))
                    
                    # Take higher priority
                    existing["priority"] = max(
                        existing.get("priority", 0),
                        task.get("priority", 0)
                    )
                    
                    # Record conflict
                    merged["conflicts"].append({
                        "type": "task_merge",
                        "task_id": task_id,
                        "resolution": "merged_agents_and_priority"
                    })
                else:
                    task["id"] = task_id
                    task_registry[task_id] = task
        
        # Second pass: resolve dependencies
        for plan in plans:
            for dep in plan.get("dependencies", []):
                dep_key = (dep.get("from"), dep.get("to"))
                
                if dep_key in dependency_registry:
                    # Dependency conflict
                    existing = dependency_registry[dep_key]
                    
                    # Check for circular dependencies
                    reverse_key = (dep_key[1], dep_key[0])
                    if reverse_key in dependency_registry:
                        merged["conflicts"].append({
                            "type": "circular_dependency",
                            "tasks": [dep_key[0], dep_key[1]],
                            "resolution": "removed_reverse"
                        })
                        # Remove the reverse dependency
                        del dependency_registry[reverse_key]
                else:
                    dependency_registry[dep_key] = dep
        
        # Build final lists
        merged["tasks"] = list(task_registry.values())
        merged["dependencies"] = list(dependency_registry.values())
        
        # Sort tasks by priority and dependencies
        merged["tasks"] = self._topological_sort_tasks(
            merged["tasks"],
            merged["dependencies"]
        )
        
        # Merge timelines and calculate duration
        all_events = []
        for plan in plans:
            all_events.extend(plan.get("timeline", []))
        
        if all_events:
            all_events.sort(key=lambda x: x.get("timestamp", 0))
            merged["timeline"] = all_events
            
            # Estimate duration
            if len(all_events) >= 2:
                start = all_events[0].get("timestamp", 0)
                end = all_events[-1].get("timestamp", 0)
                merged["estimated_duration"] = end - start
        
        # Aggregate resources with conflict detection
        for plan in plans:
            for resource, amount in plan.get("resources", {}).items():
                if resource not in merged["resources"]:
                    merged["resources"][resource] = 0
                
                # Check for resource conflicts
                new_total = merged["resources"][resource] + amount
                limit = self._get_resource_limit(resource)
                
                if new_total > limit:
                    merged["conflicts"].append({
                        "type": "resource_overflow",
                        "resource": resource,
                        "requested": new_total,
                        "limit": limit,
                        "resolution": "capped_at_limit"
                    })
                    merged["resources"][resource] = limit
                else:
                    merged["resources"][resource] = new_total
        
        # Determine resolution strategy based on conflicts
        if len(merged["conflicts"]) > 5:
            merged["resolution_strategy"] = "sequential"  # Too many conflicts, run sequentially
        elif any(c["type"] == "circular_dependency" for c in merged["conflicts"]):
            merged["resolution_strategy"] = "reordered"  # Had to reorder tasks
        
        return merged
    
    def _topological_sort_tasks(self, tasks: List[Dict], dependencies: List[Dict]) -> List[Dict]:
        """Sort tasks based on dependencies."""
        # Build adjacency list
        graph = {task["id"]: [] for task in tasks}
        in_degree = {task["id"]: 0 for task in tasks}
        
        for dep in dependencies:
            if dep["from"] in graph and dep["to"] in graph:
                graph[dep["from"]].append(dep["to"])
                in_degree[dep["to"]] += 1
        
        # Kahn's algorithm
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        sorted_ids = []
        
        while queue:
            current = queue.pop(0)
            sorted_ids.append(current)
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Map back to tasks
        task_map = {task["id"]: task for task in tasks}
        return [task_map[task_id] for task_id in sorted_ids if task_id in task_map]
    
    async def _execute_execution_task(
        self,
        agents: List[Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute main execution task with all agents."""
        results = []
        
        for agent in agents:
            if hasattr(agent, "execute"):
                result = await agent.execute(context)
                results.append(result)
        
        return {
            "status": "completed",
            "results": results
        }
    
    async def _execute_coordination_task(
        self,
        agents: List[Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute coordination task with planning-capable agents."""
        coordination_results = []
        
        for agent in agents:
            if hasattr(agent, "coordinate"):
                result = await agent.coordinate(context)
                coordination_results.append(result)
        
        return {
            "status": "completed",
            "results": coordination_results
        }
    
    async def _execute_verification_task(
        self,
        agents: List[Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute verification task with all agents."""
        verifications = []
        
        for agent in agents:
            if hasattr(agent, "verify"):
                verification = await agent.verify(context)
                verifications.append(verification)
        
        return {
            "status": "completed",
            "verifications": verifications
        }
    
    async def _execute_cleanup_task(
        self,
        agents: List[Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute cleanup task with resource-capable agents."""
        cleanup_results = []
        
        for agent in agents:
            if hasattr(agent, "cleanup"):
                result = await agent.cleanup(context)
                cleanup_results.append(result)
        
        return {
            "status": "completed",
            "results": cleanup_results
        }
    
    async def _calculate_plan_metrics(
        self,
        plan: Dict[str, Any],
        phases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate metrics for the entire plan."""
        metrics = {
            "completion": 0.0,
            "efficiency": 0.0,
            "coordination": 0.0,
            "learning": 0.0
        }
        
        # Calculate completion
        successful_phases = sum(1 for p in phases if p["success"])
        metrics["completion"] = successful_phases / len(phases)
        
        # Calculate efficiency
        total_time = sum(p["metrics"].get("duration", 0) for p in phases)
        expected_time = self._parse_timeframe(plan["timeframe"])
        metrics["efficiency"] = expected_time / total_time if total_time > 0 else 0
        
        # Calculate coordination
        coordination_scores = [
            p["metrics"].get("coordination_score", 0)
            for p in phases
        ]
        metrics["coordination"] = sum(coordination_scores) / len(coordination_scores) if coordination_scores else 0
        
        # Calculate learning
        learning_scores = [
            p["metrics"].get("learning_score", 0)
            for p in phases
        ]
        metrics["learning"] = sum(learning_scores) / len(learning_scores) if learning_scores else 0
        
        return metrics
    
    async def _calculate_phase_metrics(
        self,
        phase: Dict[str, Any],
        tasks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate metrics for a single phase."""
        metrics = {
            "duration": 0.0,
            "coordination_score": 0.0,
            "learning_score": 0.0,
            "resource_usage": {}
        }
        
        # Calculate duration
        start_time = min(t.get("start_time", 0) for t in tasks)
        end_time = max(t.get("end_time", 0) for t in tasks)
        metrics["duration"] = end_time - start_time
        
        # Calculate coordination score
        coordination_scores = [
            t["metrics"].get("coordination_score", 0)
            for t in tasks
        ]
        metrics["coordination_score"] = sum(coordination_scores) / len(coordination_scores) if coordination_scores else 0
        
        # Calculate learning score
        learning_scores = [
            t["metrics"].get("learning_score", 0)
            for t in tasks
        ]
        metrics["learning_score"] = sum(learning_scores) / len(learning_scores) if learning_scores else 0
        
        # Calculate resource usage
        for task in tasks:
            for resource, amount in task["metrics"].get("resource_usage", {}).items():
                if resource not in metrics["resource_usage"]:
                    metrics["resource_usage"][resource] = 0
                metrics["resource_usage"][resource] += amount
        
        return metrics
    
    async def _calculate_task_metrics(
        self,
        task: Dict[str, Any],
        output: Dict[str, Any],
        agents: List[Any]
    ) -> Dict[str, Any]:
        """Calculate metrics for a single task with actual measurements."""
        import time
        
        metrics = {
            "coordination_score": 0.0,
            "learning_score": 0.0,
            "resource_usage": {},
            "start_time": task.get("start_time", time.time()),
            "end_time": time.time(),
            "duration": 0.0,
            "agent_count": len(agents),
            "success_rate": 0.0
        }
        
        # Calculate actual duration
        metrics["duration"] = metrics["end_time"] - metrics["start_time"]
        
        # Calculate coordination score based on agent interaction
        if len(agents) > 1:
            # Check for actual coordination indicators in output
            coordination_indicators = 0
            
            # Look for cross-agent references
            for result in output.get("results", []):
                if isinstance(result, dict):
                    # Check if result references other agents
                    result_str = str(result).lower()
                    for agent in agents:
                        agent_name = agent.__class__.__name__.lower()
                        if agent_name in result_str:
                            coordination_indicators += 1
                    
                    # Check for explicit coordination fields
                    if result.get("coordinated_with"):
                        coordination_indicators += len(result["coordinated_with"])
                    if result.get("shared_data"):
                        coordination_indicators += 1
            
            metrics["coordination_score"] = min(1.0, coordination_indicators / (len(agents) * 2))
        
        # Calculate learning score based on adaptations
        adaptation_count = 0
        successful_adaptations = 0
        
        for result in output.get("results", []):
            if isinstance(result, dict):
                if result.get("adapted", False):
                    adaptation_count += 1
                    if result.get("adaptation_successful", True):
                        successful_adaptations += 1
                
                # Check for learning indicators
                if result.get("patterns_learned"):
                    adaptation_count += len(result["patterns_learned"])
                    successful_adaptations += len(result["patterns_learned"])
        
        if adaptation_count > 0:
            metrics["learning_score"] = successful_adaptations / adaptation_count
        
        # Calculate resource usage from actual measurements
        for result in output.get("results", []):
            if isinstance(result, dict) and "resource_usage" in result:
                for resource, amount in result["resource_usage"].items():
                    if resource not in metrics["resource_usage"]:
                        metrics["resource_usage"][resource] = 0
                    metrics["resource_usage"][resource] += amount
        
        # Add computed resource usage
        metrics["resource_usage"]["time"] = metrics["duration"]
        metrics["resource_usage"]["agents"] = len(agents)
        
        # Calculate success rate
        successful_results = sum(
            1 for r in output.get("results", [])
            if isinstance(r, dict) and r.get("success", False)
        )
        total_results = len(output.get("results", []))
        
        if total_results > 0:
            metrics["success_rate"] = successful_results / total_results
        
        return metrics
    
    async def _evaluate_plan_success(
        self,
        plan: Dict[str, Any],
        result: Dict[str, Any]
    ) -> bool:
        """Evaluate overall plan success based on criteria."""
        # Check basic completion
        if not all(p["success"] for p in result["phases"]):
            return False
        
        # Check success criteria
        for criterion in plan["success_criteria"]:
            if not await self._evaluate_criterion(criterion, result):
                return False
        
        return True
    
    async def _evaluate_phase_success(
        self,
        phase: Dict[str, Any],
        result: Dict[str, Any]
    ) -> bool:
        """Evaluate phase success based on tasks and metrics."""
        # Check task success
        if not all(t["success"] for t in result["tasks"]):
            return False
        
        # Check phase-specific metrics
        metrics = result["metrics"]
        
        # Check coordination score
        if metrics["coordination_score"] < 0.5:
            return False
        
        # Check resource efficiency
        for resource, amount in metrics["resource_usage"].items():
            if amount > self._get_resource_limit(resource):
                return False
        
        return True
    
    async def _evaluate_criterion(
        self,
        criterion: Dict[str, Any],
        result: Dict[str, Any]
    ) -> bool:
        """Evaluate a single success criterion."""
        criterion_type = criterion["type"]
        
        if criterion_type == "completion":
            return result["success"]
        elif criterion_type == "resource_efficiency":
            return result["metrics"]["efficiency"] >= criterion["target"]
        elif criterion_type == "coordination_quality":
            return result["metrics"]["coordination"] >= criterion["target"]
        elif criterion_type == "learning_achievement":
            return result["metrics"]["learning"] >= criterion["target"]
        
        return False
    
    def _get_resource_limit(self, resource: str) -> float:
        """Get the limit for a specific resource based on game state."""
        # Base limits
        base_limits = {
            "memory": 1000,      # MB
            "computation": 100,   # CPU seconds
            "time": 3600,        # seconds
            "narrative_tokens": 10000,  # story elements
            "agent_actions": 50,        # per phase
            "coordination_attempts": 20  # per goal
        }
        
        # Dynamic adjustments based on game state
        if hasattr(self, 'game_state') and self.game_state:
            # Adjust based on number of active entities
            active_npcs = len(self.game_state.get('current_npcs', []))
            active_quests = len(self.game_state.get('active_quests', []))
            
            # Scale limits based on complexity
            complexity_factor = 1.0 + (active_npcs * 0.05) + (active_quests * 0.1)
            
            if resource in ["memory", "computation"]:
                return base_limits[resource] * complexity_factor
            elif resource == "time":
                # Less time with more complexity to maintain pacing
                return base_limits[resource] / (1.0 + (complexity_factor - 1.0) * 0.5)
        
        return base_limits.get(resource, float("inf"))
    
    def _parse_timeframe(self, timeframe: str) -> float:
        """Parse timeframe string into seconds."""
        match = re.match(r"(\d+)\s*(hour|minute|second)s?", timeframe.lower())
        if not match:
            return 3600  # Default to 1 hour
        
        amount = int(match.group(1))
        unit = match.group(2)
        
        multipliers = {
            "hour": 3600,
            "minute": 60,
            "second": 1
        }
        
        return amount * multipliers[unit]

    async def _extract_goals_from_memories(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract goals from memory objects using proper memory structure."""
        goals = []
        
        for memory in memories:
            memory_text = memory.get("memory_text", "").lower()
            memory_type = memory.get("memory_type", "")
            metadata = memory.get("metadata", {})
            
            # Check if memory contains goal information
            goal_keywords = ["goal", "objective", "mission", "quest", "task", "achieve", "complete"]
            
            if any(keyword in memory_text for keyword in goal_keywords) or memory_type == "goal":
                # Extract goal details from metadata if available
                if metadata.get("goal_data"):
                    goals.append({
                        "id": memory.get("id"),
                        "description": metadata["goal_data"].get("description", memory_text),
                        "status": metadata["goal_data"].get("status", "active"),
                        "priority": metadata["goal_data"].get("priority", 5),
                        "deadline": metadata["goal_data"].get("deadline"),
                        "progress": metadata["goal_data"].get("progress", 0)
                    })
                else:
                    # Parse goal from text
                    goals.append({
                        "id": memory.get("id"),
                        "description": memory_text,
                        "status": "active",
                        "priority": 5,
                        "progress": 0
                    })
        
        # Sort by priority
        goals.sort(key=lambda g: g.get("priority", 5), reverse=True)
        return goals

    async def orchestrate_narrative_shift(self, reason: str, shift_type: str = "local", shift_details: Optional[Dict[str, Any]] = None):
        """
        Orchestrate a narrative shift at any scale.
        
        Args:
            reason: Why the shift is happening
            shift_type: Scale of shift ("personal", "local", "regional", "national", "global")
            shift_details: Optional details to customize the shift, including:
                - target_entities: Specific entities to affect
                - change_type: Type of change to make
                - custom_changes: Specific changes to apply
        """
        if not self._initialized:
            await self.initialize()
    
        logger.info(f"NYX: Orchestrating a {shift_type} narrative shift because: {reason}")
    
        ctx = RunContextWrapper(context={"user_id": self.user_id, "conversation_id": self.conversation_id})
        shift_details = shift_details or {}
        results = []
    
        # Different logic based on shift type
        if shift_type == "personal":
            # Personal-scale shift: individual character changes
            # Examples: someone gets a new job, relationship changes, personal growth
            
            # Use provided target or select an NPC based on narrative needs
            if "target_entities" in shift_details:
                npc_names = shift_details["target_entities"]
            else:
                # Example: Select an NPC who needs development
                npc_names = ["Sarah Chen"]  # In real implementation, this would be dynamic
            
            for npc_name in npc_names:
                npc_id = await self._get_npc_id_by_name(npc_name)
                if not npc_id:
                    logger.warning(f"NPC '{npc_name}' not found for personal shift")
                    continue
                    
                # Determine what kind of personal change
                change_type = shift_details.get("change_type", "growth")
                
                if change_type == "growth":
                    updates = {
                        "confidence": min(100, await self._get_npc_stat(npc_id, "confidence", 50) + 10),
                        "personality_traits": await self._add_personality_trait(npc_id, "determined")
                    }
                    narrative_reason = f"{reason} {npc_name} has grown more confident."
                    
                elif change_type == "location":
                    new_location = shift_details.get("new_location", "University Library")
                    updates = {
                        "current_location": new_location,
                        "schedule": shift_details.get("new_schedule", {"morning": "studying", "afternoon": "working"})
                    }
                    narrative_reason = f"{reason} {npc_name} now spends most of their time at {new_location}."
                    
                elif change_type == "relationship":
                    updates = shift_details.get("custom_changes", {})
                    narrative_reason = f"{reason} {npc_name}'s relationships have shifted."
                    
                else:
                    updates = shift_details.get("custom_changes", {})
                    narrative_reason = f"{reason} {npc_name} has experienced a personal change."
                
                result = await self.lore_system.propose_and_enact_change(
                    ctx=ctx,
                    entity_type="NPCStats",
                    entity_identifier={"npc_id": npc_id},
                    updates=updates,
                    reason=narrative_reason
                )
                results.append(result)
                
        elif shift_type == "local":
            # Local-scale shift: community changes, local groups, small businesses
            # Examples: store closes, new club forms, local election
            
            change_type = shift_details.get("change_type", "community_change")
            
            if change_type == "business_closure":
                location_name = shift_details.get("location", "Corner Coffee Shop")
                # Update location status
                async with get_db_connection_context() as conn:
                    await conn.execute("""
                        UPDATE Locations 
                        SET description = description || ' (CLOSED)',
                            open_hours = '{}'::jsonb
                        WHERE location_name = $1 AND user_id = $2 AND conversation_id = $3
                    """, location_name, self.user_id, self.conversation_id)
                
                # Create a conflict about it
                conflict_result = await self.create_conflict({
                    "name": f"{location_name} Closure Controversy",
                    "conflict_type": "economic",
                    "scale": "local",
                    "description": f"Local residents are upset about {location_name} closing down",
                    "involved_parties": [
                        {"type": "location", "name": location_name, "stake": "closing"},
                        {"type": "faction", "name": "Local Business Association", "stance": "concerned"}
                    ],
                    "stakes": "community gathering place"
                }, reason)
                results.append(conflict_result)
                
            elif change_type == "new_group":
                # Create a new local faction/group
                group_name = shift_details.get("group_name", "Community Gardeners")
                group_location = shift_details.get("location", "Riverside Park")
                
                faction_id = await self._get_faction_id_by_name(group_name)
                if not faction_id:
                    from lore.core import canon
                    async with get_db_connection_context() as conn:
                        faction_id = await canon.find_or_create_faction(
                            ctx, conn, 
                            faction_name=group_name,
                            type=shift_details.get("faction_type", "community"),
                            description=shift_details.get("description", f"A local {shift_details.get('faction_type', 'community')} group"),
                            influence_scope="local",
                            power_level=2
                        )
                
                result = await self.lore_system.propose_and_enact_change(
                    ctx=ctx,
                    entity_type="Factions",
                    entity_identifier={"id": faction_id},
                    updates={
                        "territory": group_location,
                        "influence_scope": "neighborhood",
                        "recruitment_methods": ["word of mouth", "community board"]
                    },
                    reason=f"{reason} The {group_name} have established themselves at {group_location}."
                )
                results.append(result)
                
            elif change_type == "local_development":
                # Changes to local infrastructure or community
                location = shift_details.get("location", "Downtown")
                development = shift_details.get("development", "new community center")
                
                # This might affect multiple entities
                affected_factions = shift_details.get("affected_factions", ["Local Business Association"])
                for faction_name in affected_factions:
                    faction_id = await self._get_faction_id_by_name(faction_name)
                    if faction_id:
                        result = await self.lore_system.propose_and_enact_change(
                            ctx=ctx,
                            entity_type="Factions",
                            entity_identifier={"id": faction_id},
                            updates={"resources": [development]},
                            reason=f"{reason} {faction_name} now has access to {development}."
                        )
                        results.append(result)
                        
        elif shift_type == "regional":
            # Regional-scale shift: multiple communities affected
            # Examples: weather event, economic downturn, cultural movement
            
            change_type = shift_details.get("change_type", "cultural_shift")
            affected_regions = shift_details.get("affected_regions", [])
            
            if change_type == "cultural_shift":
                cultural_change = shift_details.get("cultural_change", {
                    "new_traits": ["environmentally conscious", "community-oriented"],
                    "values_shift": "towards sustainability"
                })
                
                for region_name in affected_regions:
                    # Update regional culture
                    # Note: This assumes you have a Regions or GeographicRegions table
                    result = await self.lore_system.propose_and_enact_change(
                        ctx=ctx,
                        entity_type="Locations",  # or "GeographicRegions" if you have that
                        entity_identifier={"location_name": region_name},
                        updates={
                            "cultural_significance": cultural_change.get("values_shift", "shifting values"),
                            "local_customs": cultural_change.get("new_traits", [])
                        },
                        reason=f"{reason} {region_name} is experiencing a cultural shift {cultural_change.get('values_shift', '')}."
                    )
                    results.append(result)
                    
        elif shift_type in ["national", "global"]:
            # Large-scale shifts: nations, international relations
            # This uses your original logic
            
            if shift_type == "national":
                # National change affecting one nation
                nation_name = shift_details.get("nation", "Example Nation")
                nation_id = await self._get_nation_id_by_name(nation_name)
                
                if nation_id:
                    updates = shift_details.get("custom_changes", {
                        "government_type": "reformed democracy",
                        "matriarchy_level": 7
                    })
                    
                    result = await self.lore_system.propose_and_enact_change(
                        ctx=ctx,
                        entity_type="nations",
                        entity_identifier={"id": nation_id},
                        updates=updates,
                        reason=f"{reason} {nation_name} has undergone significant political reform."
                    )
                    results.append(result)
                    
            else:  # global
                # Global changes affecting multiple nations or the whole world
                # Example: technological breakthrough, climate event, pandemic
                
                change_type = shift_details.get("change_type", "political")
                
                if change_type == "political":
                    # Original example: faction gains territory
                    faction_name = shift_details.get("faction", "The Matriarchal Council")
                    new_territory = shift_details.get("territory", "The Sunken City")
                    
                    faction_id = await self._get_faction_id_by_name(faction_name)
                    if not faction_id:
                        from lore.core import canon
                        async with get_db_connection_context() as conn:
                            faction_id = await canon.find_or_create_faction(
                                ctx, conn, 
                                faction_name=faction_name,
                                type="political",
                                description=f"A powerful faction seeking to control {new_territory}",
                                influence_scope="global",
                                power_level=8
                            )
                    
                    result = await self.lore_system.propose_and_enact_change(
                        ctx=ctx,
                        entity_type="Factions",
                        entity_identifier={"id": faction_id},
                        updates={"territory": new_territory},
                        reason=f"{reason} {faction_name} has gained control of {new_territory}."
                    )
                    results.append(result)
                    
                elif change_type == "technological":
                    # Global tech advancement
                    advancement = shift_details.get("advancement", "renewable energy breakthrough")
                    affected_nations = shift_details.get("affected_nations", [])
                    
                    for nation_name in affected_nations:
                        nation_id = await self._get_nation_id_by_name(nation_name)
                        if nation_id:
                            result = await self.lore_system.propose_and_enact_change(
                                ctx=ctx,
                                entity_type="nations",
                                entity_identifier={"id": nation_id},
                                updates={"technology_level": 8},
                                reason=f"{reason} {nation_name} has adopted {advancement}."
                            )
                            results.append(result)
    
        # Record the narrative shift as an event
        significance_map = {
            "personal": 3,
            "local": 5,
            "regional": 7,
            "national": 9,
            "global": 10
        }
        
        await self._record_narrative_event(
            event_type=f"{shift_type}_narrative_shift",
            details={
                "shift_type": shift_type,
                "reason": reason,
                "changes_made": len(results),
                "shift_details": shift_details,
                "results": results
            }
        )
    
        logger.info(f"NYX: {shift_type} narrative shift completed with {len(results)} changes.")
        return {
            "status": "completed",
            "shift_type": shift_type,
            "changes_made": len(results),
            "results": results
        }
    
    # Helper methods used by orchestrate_narrative_shift:
    
    async def _get_npc_stat(self, npc_id: int, stat_name: str, default: int = 50) -> int:
        """Get a specific stat value for an NPC."""
        try:
            async with get_db_connection_context() as conn:
                value = await conn.fetchval(f"""
                    SELECT {stat_name} FROM NPCStats 
                    WHERE npc_id = $1
                """, npc_id)
                return value if value is not None else default
        except Exception as e:
            logger.error(f"Error getting NPC stat {stat_name}: {e}")
            return default
    
    async def _add_personality_trait(self, npc_id: int, new_trait: str) -> List[str]:
        """Add a personality trait to an NPC if they don't already have it."""
        try:
            async with get_db_connection_context() as conn:
                current_traits = await conn.fetchval("""
                    SELECT personality_traits FROM NPCStats 
                    WHERE npc_id = $1
                """, npc_id)
                
                traits = json.loads(current_traits) if current_traits else []
                if new_trait not in traits:
                    traits.append(new_trait)
                
                return traits
        except Exception as e:
            logger.error(f"Error adding personality trait: {e}")
            return [new_trait]
                
    async def _get_faction_id_by_name(self, faction_name: str) -> Optional[int]:
        """
        Retrieve a faction ID by name from the Factions table.
        """
        try:
            async with get_db_connection_context() as conn:
                result = await conn.fetchval("""
                    SELECT id FROM Factions 
                    WHERE name = $1 AND user_id = $2 AND conversation_id = $3
                """, faction_name, self.user_id, self.conversation_id)
                return result
        except Exception as e:
            logger.error(f"Error retrieving faction ID for '{faction_name}': {e}")
            return None
    
    async def _get_nation_id_by_name(self, nation_name: str) -> Optional[int]:
        """Retrieve a nation ID by name from the Nations table."""
        try:
            async with get_db_connection_context() as conn:
                result = await conn.fetchval("""
                    SELECT id FROM Nations 
                    WHERE LOWER(name) = LOWER($1)
                """, nation_name)
                return result
        except Exception as e:
            logger.error(f"Error retrieving nation ID for '{nation_name}': {e}")
            return None
    
    async def _get_npc_id_by_name(self, npc_name: str) -> Optional[int]:
        """Retrieve an NPC ID by name from NPCStats."""
        try:
            async with get_db_connection_context() as conn:
                result = await conn.fetchval("""
                    SELECT npc_id FROM NPCStats 
                    WHERE npc_name = $1 AND user_id = $2 AND conversation_id = $3
                """, npc_name, self.user_id, self.conversation_id)
                return result
        except Exception as e:
            logger.error(f"Error retrieving NPC ID for '{npc_name}': {e}")
            return None


    
    async def create_local_group(self, group_data: Dict[str, Any], reason: str) -> Dict[str, Any]:
        """
        Create a local group or organization (not necessarily political).
        
        Args:
            group_data: Data including:
                - name: Group name (e.g., "Book Club", "Local Band", "Parent Committee")
                - type: Type (e.g., "social", "hobby", "community", "educational")
                - scope: Scope (e.g., "school", "neighborhood", "online")
                - meeting_place: Where they meet
                - members: List of member names
                - activities: What they do
            reason: Why this group is being created
        """
        if not self._initialized:
            await self.initialize()
            
        ctx = RunContextWrapper(context={"user_id": self.user_id, "conversation_id": self.conversation_id})
        
        # Create the faction with appropriate type
        from lore.core import canon
        async with get_db_connection_context() as conn:
            faction_id = await canon.find_or_create_faction(
                ctx, conn,
                faction_name=group_data["name"],
                type=group_data.get("type", "social"),
                description=group_data.get("description", f"A {group_data.get('type', 'social')} group"),
                values=group_data.get("values", ["community", "shared interests"]),
                goals=group_data.get("goals", ["meet regularly", "enjoy activities"]),
                influence_scope=group_data.get("scope", "local"),
                power_level=2,  # Low power level for local groups
                territory=[group_data.get("meeting_place", "various locations")]
            )
        
        # Add members as allies/affiliates
        if "members" in group_data:
            member_ids = []
            for member_name in group_data["members"]:
                npc_id = await self._get_npc_id_by_name(member_name)
                if npc_id:
                    member_ids.append(npc_id)
                    # Update NPC's affiliations
                    await self.lore_system.propose_and_enact_change(
                        ctx=ctx,
                        entity_type="NPCStats", 
                        entity_identifier={"npc_id": npc_id},
                        updates={"affiliations": [group_data["name"]]},
                        reason=f"Joined {group_data['name']}"
                    )
        
        return {"status": "success", "faction_id": faction_id, "type": "local_group"}

    async def enact_political_change(self, nation_name: str, change_type: str, details: Dict[str, Any], reason: str) -> Dict[str, Any]:
        """
        Enact a political change in a nation through the LoreSystem.
        
        Args:
            nation_name: Name of the nation (will be looked up)
            change_type: Type of change (e.g., 'leadership', 'government', 'policy')
            details: Specific details of the change
            reason: Narrative reason for the change
        
        Returns:
            Result of the change enactment
        """
        if not self._initialized:
            await self.initialize()
    
        logger.info(f"NYX: Enacting political change in nation {nation_name}: {change_type}")
    
        ctx = RunContextWrapper(context={"user_id": self.user_id, "conversation_id": self.conversation_id})
    
        # Get nation ID
        nation_id = await self._get_nation_id_by_name(nation_name)
        if not nation_id:
            # Create the nation if it doesn't exist
            from lore.core import canon
            async with get_db_connection_context() as conn:
                nation_id = await canon.find_or_create_nation(
                    ctx, conn,
                    nation_name=nation_name,
                    government_type=details.get('government_type', 'Unknown')
                )
    
        # Map change_type to appropriate updates
        updates = {}
        if change_type == "leadership":
            if "new_leader_name" in details:
                # Get or create the new leader NPC
                new_leader_id = await self._get_npc_id_by_name(details["new_leader_name"])
                if not new_leader_id:
                    from lore.core import canon
                    async with get_db_connection_context() as conn:
                        new_leader_id = await canon.find_or_create_npc(
                            ctx, conn,
                            npc_name=details["new_leader_name"],
                            role="Political Leader",
                            affiliations=[nation_name]
                        )
                updates["leader_npc_id"] = new_leader_id
            if "leadership_structure" in details:
                updates["leadership_structure"] = details["leadership_structure"]
        elif change_type == "government":
            if "government_type" in details:
                updates["government_type"] = details["government_type"]
            if "matriarchy_level" in details:
                updates["matriarchy_level"] = details["matriarchy_level"]
        elif change_type == "policy":
            # For policies, we might need to update JSON fields
            if "diplomatic_stance" in details:
                updates["diplomatic_stance"] = details["diplomatic_stance"]
            if "economic_focus" in details:
                updates["economic_focus"] = details["economic_focus"]
    
        # Delegate to LoreSystem
        result = await self.lore_system.propose_and_enact_change(
            ctx=ctx,
            entity_type="nations",  # Use lowercase as per schema
            entity_identifier={"id": nation_id},
            updates=updates,
            reason=f"Political change ({change_type}): {reason}"
        )
    
        # Record this as a major narrative event
        if result.get("status") == "committed":
            await self._record_narrative_event(
                event_type="political_change",
                details={
                    "nation_id": nation_id,
                    "nation_name": nation_name,
                    "change_type": change_type,
                    "updates": updates,
                    "reason": reason
                }
            )
    
        return result

    async def create_conflict(self, conflict_data: Dict[str, Any], reason: str) -> Dict[str, Any]:
        """
        Create a new conflict in the world through the LoreSystem.
        Conflicts can be any scale: interpersonal disagreements, local issues, or international disputes.
        
        Args:
            conflict_data: Data for the conflict including:
                - name: Conflict name (e.g., "Library Noise Complaint", "Trade War")
                - conflict_type: Type (e.g., "interpersonal", "community", "political", "economic")
                - scale: Scale of conflict ("personal", "local", "regional", "national", "global")
                - involved_parties: List of involved parties (can be NPCs, factions, nations, locations)
                - description: Description of the conflict
                - stakes: What's at stake (e.g., "friendship", "local business", "territory")
            reason: Narrative reason for the conflict
        
        Returns:
            Result of conflict creation
        """
        if not self._initialized:
            await self.initialize()
    
        logger.info(f"NYX: Creating new conflict: {conflict_data.get('name', 'Unnamed')}")
    
        ctx = RunContextWrapper(context={"user_id": self.user_id, "conversation_id": self.conversation_id})
    
        # Handle different types of involved parties
        scale = conflict_data.get("scale", "local")
        involved_parties = conflict_data.get("involved_parties", [])
        
        # Process stakeholders based on conflict scale
        stakeholders = []
        
        for party in involved_parties:
            if isinstance(party, dict):
                party_type = party.get("type", "npc")
                party_name = party.get("name")
                
                if party_type == "npc":
                    npc_id = await self._get_npc_id_by_name(party_name)
                    if npc_id:
                        stakeholders.append({
                            "npc_id": npc_id,
                            "role": party.get("role", "participant"),
                            "stance": party.get("stance", "neutral")
                        })
                elif party_type == "faction":
                    # Could be student club, local group, etc.
                    stakeholders.append({
                        "faction_name": party_name,
                        "faction_type": party.get("faction_type", "community"),
                        "stance": party.get("stance", "neutral")
                    })
                elif party_type == "location":
                    # For conflicts about places (e.g., "coffee shop closing")
                    stakeholders.append({
                        "location_name": party_name,
                        "stake": party.get("stake", "affected")
                    })
            elif isinstance(party, str):
                # Assume it's an NPC name
                npc_id = await self._get_npc_id_by_name(party)
                if npc_id:
                    stakeholders.append({"npc_id": npc_id})
    
        # Create the conflict with appropriate scale
        async with get_db_connection_context() as conn:
            conflict_id = await conn.fetchval("""
                INSERT INTO Conflicts (
                    user_id, conversation_id, conflict_name, conflict_type,
                    description, phase, is_active, 
                    progress, estimated_duration
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                RETURNING conflict_id
            """, 
            self.user_id, self.conversation_id, 
            conflict_data.get("name", "Unknown Conflict"),
            conflict_data.get("conflict_type", "interpersonal"),
            conflict_data.get("description", ""),
            conflict_data.get("phase", "brewing"),
            True,
            0.0,
            conflict_data.get("estimated_duration", 1) if scale == "personal" else 30)
    
            # Add stakeholders with scale-appropriate details
            for stakeholder in stakeholders:
                if "npc_id" in stakeholder:
                    await conn.execute("""
                        INSERT INTO ConflictStakeholders (
                            conflict_id, npc_id, faction_name, 
                            public_motivation, private_motivation,
                            involvement_level
                        )
                        VALUES ($1, $2, $3, $4, $5, $6)
                    """, 
                    conflict_id, 
                    stakeholder["npc_id"],
                    stakeholder.get("faction_name"),
                    stakeholder.get("public_motivation", "Personal reasons"),
                    stakeholder.get("private_motivation", "Unknown"),
                    stakeholder.get("involvement_level", 5 if scale == "personal" else 3))
    
        # Log appropriate event based on scale
        significance = {
            "personal": 3,
            "local": 5,
            "regional": 7,
            "national": 9,
            "global": 10
        }.get(scale, 5)
        
        await self._record_narrative_event(
            event_type=f"{scale}_conflict",
            details={
                "conflict_id": conflict_id,
                "name": conflict_data.get("name"),
                "type": conflict_data.get("conflict_type"),
                "scale": scale,
                "stakes": conflict_data.get("stakes", "unspecified")
            }
        )
        
        return {"status": "committed", "conflict_id": conflict_id}

    async def modify_npc_behavior(self, npc_name: str, behavior_changes: Dict[str, Any], reason: str) -> Dict[str, Any]:
        """
        Modify an NPC's behavior through the LoreSystem.
        
        Args:
            npc_name: Name of the NPC (will be looked up)
            behavior_changes: Changes to apply to the NPC
            reason: Narrative reason for the change
        
        Returns:
            Result of the modification
        """
        if not self._initialized:
            await self.initialize()
    
        logger.info(f"NYX: Modifying behavior for NPC {npc_name}")
    
        ctx = RunContextWrapper(context={"user_id": self.user_id, "conversation_id": self.conversation_id})
    
        # Get NPC ID
        npc_id = await self._get_npc_id_by_name(npc_name)
        if not npc_id:
            logger.error(f"NPC '{npc_name}' not found. Cannot modify behavior.")
            return {"status": "error", "message": f"NPC '{npc_name}' not found"}
    
        # Delegate to LoreSystem
        result = await self.lore_system.propose_and_enact_change(
            ctx=ctx,
            entity_type="NPCStats",
            entity_identifier={"npc_id": npc_id},
            updates=behavior_changes,
            reason=f"NPC behavior modification: {reason}"
        )
    
        return result


    async def update_faction_relations(self, faction_name: str, relation_updates: Dict[str, Any], reason: str) -> Dict[str, Any]:
        """
        Update faction relations through the LoreSystem.
        
        Args:
            faction_name: Name of the faction
            relation_updates: Updates to faction relations including:
                - ally_names: List of ally faction names
                - rival_names: List of rival faction names  
                - public_reputation: New reputation value
            reason: Narrative reason for the change
        
        Returns:
            Result of the update
        """
        if not self._initialized:
            await self.initialize()
    
        logger.info(f"NYX: Updating relations for faction {faction_name}")
    
        ctx = RunContextWrapper(context={"user_id": self.user_id, "conversation_id": self.conversation_id})
    
        # Get faction ID
        faction_id = await self._get_faction_id_by_name(faction_name)
        if not faction_id:
            logger.error(f"Faction '{faction_name}' not found.")
            return {"status": "error", "message": f"Faction '{faction_name}' not found"}
    
        # Convert faction names to IDs in relation updates
        updates = {}
        
        if "ally_names" in relation_updates:
            ally_ids = []
            for ally_name in relation_updates["ally_names"]:
                ally_id = await self._get_faction_id_by_name(ally_name)
                if ally_id:
                    ally_ids.append(ally_id)
            updates["allies"] = ally_ids
            
        if "rival_names" in relation_updates:
            rival_ids = []
            for rival_name in relation_updates["rival_names"]:
                rival_id = await self._get_faction_id_by_name(rival_name)
                if rival_id:
                    rival_ids.append(rival_id)
            updates["rivals"] = rival_ids
            
        if "public_reputation" in relation_updates:
            updates["public_reputation"] = relation_updates["public_reputation"]
    
        # Delegate to LoreSystem
        result = await self.lore_system.propose_and_enact_change(
            ctx=ctx,
            entity_type="Factions",
            entity_identifier={"id": faction_id},
            updates=updates,
            reason=f"Faction relations update: {reason}"
        )
    
        return result

    async def assign_npc_to_location(self, npc_name: str, location_name: str, reason: str) -> Dict[str, Any]:
        """
        Assign an NPC to a new location through the LoreSystem.
        
        Args:
            npc_name: Name of the NPC
            location_name: Name of the location
            reason: Narrative reason for the move
            
        Returns:
            Result of the assignment
        """
        if not self._initialized:
            await self.initialize()
    
        logger.info(f"NYX: Assigning {npc_name} to location {location_name}")
    
        ctx = RunContextWrapper(context={"user_id": self.user_id, "conversation_id": self.conversation_id})
    
        # Get NPC ID
        npc_id = await self._get_npc_id_by_name(npc_name)
        if not npc_id:
            # Create the NPC if needed
            from lore.core import canon
            async with get_db_connection_context() as conn:
                npc_id = await canon.find_or_create_npc(
                    ctx, conn,
                    npc_name=npc_name,
                    role="Citizen"
                )
    
        # Ensure location exists
        from lore.core import canon
        async with get_db_connection_context() as conn:
            await canon.find_or_create_location(ctx, conn, location_name)
    
        # Update NPC's current location
        result = await self.lore_system.propose_and_enact_change(
            ctx=ctx,
            entity_type="NPCStats",
            entity_identifier={"npc_id": npc_id},
            updates={"current_location": location_name},
            reason=f"Location assignment: {reason}"
        )
    
        return result

    async def create_npc_relationship(self, npc1_name: str, npc2_name: str, 
                                    relationship_type: str, details: Dict[str, Any], 
                                    reason: str) -> Dict[str, Any]:
        """
        Create or update a relationship between two NPCs.
        """
        if not self._initialized:
            await self.initialize()
    
        logger.info(f"NYX: Creating relationship between {npc1_name} and {npc2_name}")
    
        ctx = RunContextWrapper(context={"user_id": self.user_id, "conversation_id": self.conversation_id})
    
        # Get NPC IDs
        npc1_id = await self._get_npc_id_by_name(npc1_name)
        npc2_id = await self._get_npc_id_by_name(npc2_name)
    
        if not npc1_id or not npc2_id:
            missing = []
            if not npc1_id:
                missing.append(npc1_name)
            if not npc2_id:
                missing.append(npc2_name)
            return {"status": "error", "message": f"NPCs not found: {', '.join(missing)}"}
    
        # Create or update the social link
        async with get_db_connection_context() as conn:
            # First create/update the SocialLinks entry
            link_id = await conn.fetchval("""
                INSERT INTO SocialLinks (
                    user_id, conversation_id,
                    entity1_type, entity1_id, entity2_type, entity2_id,
                    link_type, link_level, link_history, dynamics,
                    relationship_stage
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10::jsonb, $11)
                ON CONFLICT (user_id, conversation_id, entity1_type, entity1_id, entity2_type, entity2_id)
                DO UPDATE SET 
                    link_type = EXCLUDED.link_type,
                    link_level = EXCLUDED.link_level,
                    dynamics = EXCLUDED.dynamics,
                    relationship_stage = EXCLUDED.relationship_stage
                RETURNING link_id
            """, 
                self.user_id, self.conversation_id,
                'npc', npc1_id, 'npc', npc2_id,
                relationship_type, details.get('link_level', 50),
                json.dumps([{"timestamp": datetime.now().isoformat(), "event": reason}]),
                json.dumps(details), details.get('stage', 'acquaintance')
            )
            
            # Update NPCs' relationships fields
            relationships = {}
            for npc_id, other_name in [(npc1_id, npc2_name), (npc2_id, npc1_name)]:
                # Get current relationships
                current_rels = await conn.fetchval("""
                    SELECT relationships FROM NPCStats WHERE npc_id = $1
                """, npc_id)
                
                relationships = json.loads(current_rels) if current_rels else {}
                relationships[other_name] = {
                    "type": relationship_type,
                    "level": details.get("link_level", 50),
                    "details": details
                }
                
                # Update through LoreSystem (outside the connection context)
                await self.lore_system.propose_and_enact_change(
                    ctx=ctx,
                    entity_type="NPCStats",
                    entity_identifier={"npc_id": npc_id},
                    updates={"relationships": relationships},
                    reason=f"Relationship update: {reason}"
                )
    
        return {"status": "committed", "relationship_created": True}

    async def evolve_world_state(self, evolution_type: str, parameters: Dict[str, Any], reason: str) -> Dict[str, Any]:
        """
        Evolve the world state in a specific way through the LoreSystem.
        
        Args:
            evolution_type: Type of evolution (e.g., 'cultural_shift', 'technological_advance')
            parameters: Parameters for the evolution
            reason: Narrative reason for the evolution
        
        Returns:
            Result of the evolution
        """
        if not self._initialized:
            await self.initialize()

        logger.info(f"NYX: Evolving world state: {evolution_type}")

        ctx = RunContextWrapper(context={"user_id": self.user_id, "conversation_id": self.conversation_id})
        
        results = []

        if evolution_type == "cultural_shift":
            # Cultural shifts might affect multiple entities
            affected_regions = parameters.get("affected_regions", [])
            cultural_changes = parameters.get("changes", {})
            
            for region_id in affected_regions:
                result = await self.lore_system.propose_and_enact_change(
                    ctx=ctx,
                    entity_type="GeographicRegions",
                    entity_identifier={"id": region_id},
                    updates={"cultural_traits": cultural_changes.get("new_traits", [])},
                    reason=f"Cultural shift in region: {reason}"
                )
                results.append(result)
                
        elif evolution_type == "technological_advance":
            # Technological advances might update multiple nations
            affected_nations = parameters.get("affected_nations", [])
            tech_level = parameters.get("technology_level", 5)
            
            for nation_id in affected_nations:
                result = await self.lore_system.propose_and_enact_change(
                    ctx=ctx,
                    entity_type="Nations",
                    entity_identifier={"id": nation_id},
                    updates={"technology_level": tech_level},
                    reason=f"Technological advancement: {reason}"
                )
                results.append(result)
                
        elif evolution_type == "economic_shift":
            # Economic shifts affect resources and trade
            affected_entities = parameters.get("affected_entities", [])
            economic_changes = parameters.get("changes", {})
            
            for entity in affected_entities:
                result = await self.lore_system.propose_and_enact_change(
                    ctx=ctx,
                    entity_type=entity["type"],
                    entity_identifier={"id": entity["id"]},
                    updates=economic_changes,
                    reason=f"Economic shift: {reason}"
                )
                results.append(result)

        return {
            "status": "completed",
            "evolution_type": evolution_type,
            "results": results,
            "reason": reason
        }

    async def _record_narrative_event(self, event_type: str, details: Dict[str, Any]):
        """
        Record a narrative event in the memory system.
        
        Args:
            event_type: Type of event
            details: Event details
        """
        memory_text = f"Narrative event ({event_type}): {json.dumps(details, indent=2)}"
        
        await self.memory_system.remember(
            entity_type="nyx",
            entity_id=self.conversation_id,
            memory_text=memory_text,
            importance="high",
            emotional=False,
            tags=["narrative", event_type, "governance"]
        )

    async def handle_agent_conflict(self, agent1_type: str, agent1_id: str, 
                                  agent2_type: str, agent2_id: str,
                                  conflict_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle conflicts between agents by making a governance decision.
        
        Args:
            agent1_type: Type of first agent
            agent1_id: ID of first agent
            agent2_type: Type of second agent
            agent2_id: ID of second agent
            conflict_details: Details of the conflict
            
        Returns:
            Resolution decision
        """
        if not self._initialized:
            await self.initialize()

        logger.info(f"NYX: Resolving conflict between {agent1_type}/{agent1_id} and {agent2_type}/{agent2_id}")

        # Analyze the conflict
        conflict_analysis = await self._analyze_agent_conflict(
            agent1_type, agent1_id, agent2_type, agent2_id, conflict_details
        )

        # Make a decision based on priorities and impact
        decision = await self._make_conflict_decision(conflict_analysis)

        # If the conflict involves world state changes, use LoreSystem
        if decision.get("requires_world_change"):
            ctx = RunContextWrapper(context={"user_id": self.user_id, "conversation_id": self.conversation_id})
            
            for change in decision.get("world_changes", []):
                await self.lore_system.propose_and_enact_change(
                    ctx=ctx,
                    entity_type=change["entity_type"],
                    entity_identifier=change["identifier"],
                    updates=change["updates"],
                    reason=f"Conflict resolution: {decision.get('reasoning', 'Agent conflict')}"
                )

        # Issue directives to the agents based on the decision
        if decision.get("agent1_directive"):
            await self.issue_directive(
                agent_type=agent1_type,
                agent_id=agent1_id,
                directive_type=DirectiveType.OVERRIDE,
                directive_data=decision["agent1_directive"],
                priority=DirectivePriority.HIGH
            )

        if decision.get("agent2_directive"):
            await self.issue_directive(
                agent_type=agent2_type,
                agent_id=agent2_id,
                directive_type=DirectiveType.OVERRIDE,
                directive_data=decision["agent2_directive"],
                priority=DirectivePriority.HIGH
            )

        return {
            "conflict_id": f"{agent1_type}_{agent1_id}_vs_{agent2_type}_{agent2_id}_{int(time.time())}",
            "decision": decision,
            "analysis": conflict_analysis,
            "timestamp": datetime.now().isoformat()
        }

    async def _analyze_agent_conflict(self, agent1_type: str, agent1_id: str,
                                    agent2_type: str, agent2_id: str,
                                    conflict_details: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a conflict between agents with dynamic scoring."""
        analysis = {
            "agents": [
                {"type": agent1_type, "id": agent1_id},
                {"type": agent2_type, "id": agent2_id}
            ],
            "conflict_type": conflict_details.get("type", "unknown"),
            "severity": conflict_details.get("severity", 5),
            "narrative_impact": 0.0,
            "world_consistency_impact": 0.0,
            "player_experience_impact": 0.0,
            "resolution_difficulty": 0.0
        }
    
        # Dynamic scoring based on conflict type and context
        conflict_type = conflict_details.get("type", "unknown")
        
        # Base impact scores
        impact_matrix = {
            "narrative_contradiction": {
                "narrative_impact": 0.8,
                "world_consistency_impact": 0.6,
                "player_experience_impact": 0.4,
                "resolution_difficulty": 0.7
            },
            "resource_competition": {
                "narrative_impact": 0.3,
                "world_consistency_impact": 0.2,
                "player_experience_impact": 0.5,
                "resolution_difficulty": 0.4
            },
            "goal_conflict": {
                "narrative_impact": 0.5,
                "world_consistency_impact": 0.3,
                "player_experience_impact": 0.6,
                "resolution_difficulty": 0.5
            },
            "timing_conflict": {
                "narrative_impact": 0.4,
                "world_consistency_impact": 0.2,
                "player_experience_impact": 0.3,
                "resolution_difficulty": 0.3
            },
            "authority_conflict": {
                "narrative_impact": 0.6,
                "world_consistency_impact": 0.7,
                "player_experience_impact": 0.5,
                "resolution_difficulty": 0.8
            }
        }
        
        # Get base scores
        base_scores = impact_matrix.get(conflict_type, {
            "narrative_impact": 0.5,
            "world_consistency_impact": 0.5,
            "player_experience_impact": 0.5,
            "resolution_difficulty": 0.5
        })
        
        # Apply base scores
        for key, value in base_scores.items():
            analysis[key] = value
        
        # Adjust based on agent types and their importance
        agent_importance = {
            AgentType.STORY_DIRECTOR: 1.5,
            AgentType.UNIVERSAL_UPDATER: 1.3,
            AgentType.SCENE_MANAGER: 1.2,
            AgentType.CONFLICT_ANALYST: 1.1,
            AgentType.NPC: 0.9
        }
        
        # Calculate importance multiplier
        importance1 = agent_importance.get(agent1_type, 1.0)
        importance2 = agent_importance.get(agent2_type, 1.0)
        avg_importance = (importance1 + importance2) / 2
        
        # Scale impacts by importance
        analysis["narrative_impact"] *= avg_importance
        analysis["world_consistency_impact"] *= avg_importance
        
        # Adjust based on severity
        severity_multiplier = conflict_details.get("severity", 5) / 10.0
        for impact_type in ["narrative_impact", "world_consistency_impact", "player_experience_impact"]:
            analysis[impact_type] *= (0.5 + severity_multiplier)
        
        # Cap all values at 1.0
        for key in ["narrative_impact", "world_consistency_impact", "player_experience_impact", "resolution_difficulty"]:
            analysis[key] = min(1.0, analysis[key])
        
        # Add context about the conflict
        analysis["context"] = {
            "current_game_state": self.game_state,
            "recent_conflicts": len([c for c in self.coordination_history[-10:] if c.get("type") == "conflict"]),
            "agent_performance": {
                agent1_type: self.agent_performance.get(agent1_type, {}).get(agent1_id, {}),
                agent2_type: self.agent_performance.get(agent2_type, {}).get(agent2_id, {})
            }
        }
        
        return analysis

    async def _make_conflict_decision(self, conflict_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a decision on how to resolve an agent conflict.
        
        Returns:
            Decision including directives and potential world changes
        """
        decision = {
            "resolution_type": "mediate",
            "reasoning": "",
            "requires_world_change": False,
            "world_changes": [],
            "agent1_directive": None,
            "agent2_directive": None
        }

        # High narrative impact - favor story consistency
        if conflict_analysis["narrative_impact"] > 0.7:
            decision["resolution_type"] = "narrative_priority"
            decision["reasoning"] = "Prioritizing narrative consistency and story flow"
            decision["agent1_directive"] = {
                "action": "defer",
                "instruction": "Defer to narrative requirements"
            }
            decision["agent2_directive"] = {
                "action": "proceed",
                "instruction": "Proceed with narrative-aligned action"
            }

        # High world consistency impact - enforce rules
        elif conflict_analysis["world_consistency_impact"] > 0.7:
            decision["resolution_type"] = "consistency_enforcement"
            decision["reasoning"] = "Enforcing world consistency rules"
            decision["requires_world_change"] = True
            # Will be filled based on specific conflict

        # Moderate impacts - find compromise
        else:
            decision["resolution_type"] = "compromise"
            decision["reasoning"] = "Finding balanced solution between competing goals"
            decision["agent1_directive"] = {
                "action": "modify",
                "instruction": "Modify approach to accommodate other agent"
            }
            decision["agent2_directive"] = {
                "action": "modify",
                "instruction": "Modify approach to accommodate other agent"
            }

        return decision


    
    async def _record_coordination(
        self,
        goal: Dict[str, Any],
        plan: Dict[str, Any],
        result: Dict[str, Any]
    ):
        """Record coordination attempt and results."""
        coordination_record = {
            "timestamp": datetime.now().isoformat(),
            "goal": goal,
            "plan": plan,
            "result": result,
            "metrics": result["metrics"]
        }
        
        self.coordination_history.append(coordination_record)
        
        # Update collaboration success metrics
        for agent_type in plan["agent_assignments"]:
            if agent_type not in self.collaboration_success:
                self.collaboration_success[agent_type] = {
                    "success": 0,
                    "total": 0
                }
            self.collaboration_success[agent_type]["total"] += 1
            if result["success"]:
                self.collaboration_success[agent_type]["success"] += 1
        
        # Store in memory system using remember method instead of add_coordination_record
        memory_text = f"Coordination for goal '{goal.get('description', 'Unknown')}' - " \
                     f"Result: {'Success' if result['success'] else 'Failure'}"
        
        tags = ["coordination", "system", goal.get("type", "general")]
        if result["success"]:
            tags.append("success")
        else:
            tags.append("failure")
            
        await self.memory_system.remember(
            entity_type="nyx",
            entity_id=self.conversation_id,
            memory_text=memory_text,
            importance="high",  # Coordination records are important
            emotional=False,    # System records don't need emotional analysis
            tags=tags
        )
    
    async def _extract_plan_learnings(
        self,
        plan: Dict[str, Any],
        phases: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract learnings from plan execution."""
        learnings = []
        
        # Extract phase-level learnings
        for phase in phases:
            if phase["metrics"]["learning_score"] > 0.5:
                learnings.append({
                    "type": "phase_learning",
                    "phase": phase["phase"],
                    "description": f"Phase {phase['phase']} demonstrated significant learning",
                    "metrics": phase["metrics"]
                })
        
        # Extract task-level learnings
        for phase in phases:
            for task in phase["tasks"]:
                if task["metrics"]["learning_score"] > 0.5:
                    learnings.append({
                        "type": "task_learning",
                        "phase": phase["phase"],
                        "task": task["task"],
                        "description": f"Task {task['task']} in phase {phase['phase']} demonstrated significant learning",
                        "metrics": task["metrics"]
                    })
        
        # Extract coordination learnings
        if plan["agent_assignments"]:
            coordination_score = sum(p["metrics"]["coordination_score"] for p in phases) / len(phases)
            if coordination_score > 0.7:
                learnings.append({
                    "type": "coordination_learning",
                    "description": "High coordination effectiveness achieved",
                    "score": coordination_score,
                    "agent_assignments": plan["agent_assignments"]
                })
        
        return learnings

    async def get_current_state(self, user_id: int, conversation_id: int) -> Dict[str, Any]:
        """
        Get the current state of the game world including narrative, character, and world state.
        """
        # Use the existing game state if available
        if hasattr(self, 'game_state') and self.game_state:
            base_state = self.game_state
        else:
            base_state = await self.initialize_game_state(force=True)
        
        # Get additional narrative context
        narrative_context = {}
        character_state = {}
        world_state = {}
        
        async with get_db_connection_context() as conn:
            # Get current narrative stage/arc
            narrative_stage = await conn.fetchval("""
                SELECT value FROM CurrentRoleplay
                WHERE user_id = $1 AND conversation_id = $2 AND key = 'NarrativeStage'
            """, user_id, conversation_id)
            
            if narrative_stage:
                narrative_context["current_arc"] = narrative_stage
                
            # Get active plot points
            active_quests = await conn.fetch("""
                SELECT quest_name, progress_detail FROM Quests
                WHERE user_id = $1 AND conversation_id = $2 AND status = 'In Progress'
            """, user_id, conversation_id)
            
            narrative_context["plot_points"] = [
                {"name": q["quest_name"], "details": q["progress_detail"]} 
                for q in active_quests
            ]
            
            # Get player character state
            player_stats = await conn.fetchrow("""
                SELECT * FROM PlayerStats
                WHERE user_id = $1 AND conversation_id = $2 AND player_name = 'Chase'
            """, user_id, conversation_id)
            
            if player_stats:
                character_state = dict(player_stats)
                
            # Get player relationships
            relationships = await conn.fetch("""
                SELECT sl.*, ns.npc_name 
                FROM SocialLinks sl
                JOIN NPCStats ns ON sl.entity2_id = ns.npc_id
                WHERE sl.user_id = $1 AND sl.conversation_id = $2 
                AND sl.entity1_type = 'player' AND sl.entity2_type = 'npc'
            """, user_id, conversation_id)
            
            character_state["relationships"] = {
                r["npc_name"]: {
                    "type": r["link_type"],
                    "level": r["link_level"],
                    "stage": r["relationship_stage"]
                } for r in relationships
            }
            
            # Get world rules and systems
            world_rules = await conn.fetch("""
                SELECT rule_name, condition, effect FROM GameRules
            """)
            
            world_state["rules"] = {
                r["rule_name"]: {
                    "condition": r["condition"],
                    "effect": r["effect"]
                } for r in world_rules
            }
            
            # Get current setting info
            setting_name = await conn.fetchval("""
                SELECT value FROM CurrentRoleplay
                WHERE user_id = $1 AND conversation_id = $2 AND key = 'CurrentSetting'
            """, user_id, conversation_id)
            
            world_state["setting"] = setting_name or "Unknown"
        
        return {
            "game_state": base_state,
            "narrative_context": narrative_context,
            "character_state": character_state,
            "world_state": world_state
        }

    async def handle_player_disagreement(
        self,
        user_id: int,
        conversation_id: int,
        action_type: str,
        action_details: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Handle direct disagreement with a player's action.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            action_type: Type of action being performed
            action_details: Details of the action
            context: Additional context information
            
        Returns:
            Dictionary containing disagreement response and reasoning
        """
        # Get current state and context
        current_state = await self.get_current_state(user_id, conversation_id)
        context = context or {}
        
        # Analyze the action's impact
        impact_analysis = await self._analyze_action_impact(
            action_type,
            action_details,
            current_state,
            context
        )
        
        # Check if disagreement is warranted
        if not self._should_disagree(impact_analysis):
            return {
                "disagrees": False,
                "reasoning": "Action is acceptable within current context",
                "impact_analysis": impact_analysis
            }
        
        # Generate disagreement response
        disagreement = await self._generate_disagreement_response(
            impact_analysis,
            current_state,
            context
        )
        
        # Track disagreement
        await self._track_disagreement(
            user_id,
            conversation_id,
            action_type,
            disagreement
        )
        
        return {
            "disagrees": True,
            "reasoning": disagreement["reasoning"],
            "suggested_alternative": disagreement.get("alternative"),
            "impact_analysis": impact_analysis,
            "narrative_context": disagreement.get("narrative_context")
        }

    async def _analyze_action_impact(
        self,
        action_type: str,
        action_details: Dict[str, Any],
        current_state: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze the impact of a player's action."""
        impact_scores = {
            "narrative_impact": 0.0,
            "character_consistency": 0.0,
            "world_integrity": 0.0,
            "player_experience": 0.0
        }
        
        # Analyze narrative impact
        narrative_context = current_state.get("narrative_context", {})
        impact_scores["narrative_impact"] = await self._calculate_narrative_impact(
            action_type,
            action_details,
            narrative_context
        )
        
        # Analyze character consistency
        character_state = current_state.get("character_state", {})
        impact_scores["character_consistency"] = await self._calculate_character_consistency(
            action_type,
            action_details,
            character_state
        )
        
        # Analyze world integrity
        world_state = current_state.get("world_state", {})
        impact_scores["world_integrity"] = await self._calculate_world_integrity(
            action_type,
            action_details,
            world_state
        )
        
        # Analyze player experience impact
        player_context = context.get("player_context", {})
        impact_scores["player_experience"] = await self._calculate_player_experience_impact(
            action_type,
            action_details,
            player_context
        )
        
        return impact_scores

    def _should_disagree(self, impact_analysis: Dict[str, float]) -> bool:
        """Determine if Nyx should disagree with the action."""
        for metric, threshold in self.disagreement_thresholds.items():
            if impact_analysis[metric] > threshold:
                return True
        return False

    async def _generate_disagreement_response(
        self,
        impact_analysis: Dict[str, Any],
        current_state: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a detailed disagreement response."""
        # Identify primary concerns
        concerns = []
        for metric, score in impact_analysis.items():
            if score > self.disagreement_thresholds[metric]:
                concerns.append(metric)
        
        # Generate reasoning based on concerns
        reasoning = await self._generate_reasoning(concerns, current_state, context)
        
        # Generate alternative suggestion if possible
        alternative = await self._generate_alternative_suggestion(
            concerns,
            current_state,
            context
        )
        
        return {
            "reasoning": reasoning,
            "alternative": alternative,
            "narrative_context": current_state.get("narrative_context", {})
        }

    async def _track_disagreement(
        self,
        user_id: int,
        conversation_id: int,
        action_type: str,
        disagreement: Dict[str, Any]
    ):
        """Track disagreement history for pattern analysis."""
        key = f"{user_id}:{conversation_id}"
        if key not in self.disagreement_history:
            self.disagreement_history[key] = []
        
        self.disagreement_history[key].append({
            "timestamp": time.time(),
            "action_type": action_type,
            "disagreement": disagreement
        })
        
        # Keep only recent history
        self.disagreement_history[key] = self.disagreement_history[key][-100:]

    async def check_action_permission(
        self,
        agent_type: str,
        agent_id: Union[int, str],
        action_type: str,
        action_details: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Check if an action is permitted by governance.
        
        Args:
            agent_type: Type of agent performing the action
            agent_id: ID of agent performing the action
            action_type: Type of action being performed
            action_details: Details of the action
            context: Additional context (optional)
            
        Returns:
            Dictionary with permission result
        """
        # Initialize the result
        result = {
            "approved": True,
            "reasoning": "Action is permitted by default"
        }
        
        # Check for active prohibitions on this action
        prohibitions = self._get_active_prohibitions(agent_type, action_type)
        if prohibitions:
            # Action is prohibited
            prohibition = prohibitions[0]  # Get the highest priority prohibition
            result = {
                "approved": False,
                "reasoning": prohibition.get("reason", "Action is prohibited"),
                "prohibition_id": prohibition.get("id")
            }
            return result
        
        # Return the result
        return result

    def _get_active_prohibitions(self, agent_type: str, action_type: str) -> List[Dict[str, Any]]:
        """
        Get active prohibitions for an agent and action type.
        
        Args:
            agent_type: Type of agent
            action_type: Type of action
            
        Returns:
            List of active prohibitions, sorted by priority
        """
        if not hasattr(self, "directives"):
            return []
        
        # Get active prohibitions
        now = datetime.now()
        prohibitions = []
        
        for directive_id, directive in getattr(self, "directives", {}).items():
            # Check if directive is a prohibition
            if directive["type"] != DirectiveType.PROHIBITION:
                continue
            
            # Check if prohibition applies to this agent and action
            prohibited_agent = directive["data"].get("agent_type")
            prohibited_action = directive["data"].get("action_type")
            
            if (prohibited_agent == agent_type or prohibited_agent == "*") and \
               (prohibited_action == action_type or prohibited_action == "*"):
                # Check if prohibition is still active
                expires_at = datetime.fromisoformat(directive["expires_at"])
                if expires_at > now:
                    prohibitions.append(directive)
        
        # Sort by priority
        prohibitions.sort(key=lambda p: p.get("priority", 0), reverse=True)
        
        return prohibitions

    async def _calculate_narrative_impact(
        self,
        action_type: str,
        action_details: Dict[str, Any],
        narrative_context: Dict[str, Any]
    ) -> float:
        """Calculate the impact of an action on the narrative."""
        impact_score = 0.0
        
        # Check for plot disruption
        if self._would_disrupt_plot(action_type, action_details, narrative_context):
            impact_score += 0.4
        
        # Check for pacing issues
        if self._would_affect_pacing(action_type, action_details, narrative_context):
            impact_score += 0.3
        
        # Check for thematic consistency
        if not self._maintains_thematic_consistency(action_type, action_details, narrative_context):
            impact_score += 0.3
        
        return min(1.0, impact_score)

    async def _calculate_character_consistency(
        self,
        action_type: str,
        action_details: Dict[str, Any],
        character_state: Dict[str, Any]
    ) -> float:
        """Calculate how consistent an action is with character development."""
        impact_score = 0.0
        
        # Check character motivation alignment
        if not self._aligns_with_motivation(action_type, action_details, character_state):
            impact_score += 0.4
        
        # Check character development trajectory
        if self._disrupts_development(action_type, action_details, character_state):
            impact_score += 0.3
        
        # Check relationship consistency
        if not self._maintains_relationships(action_type, action_details, character_state):
            impact_score += 0.3
        
        return min(1.0, impact_score)

    async def _calculate_world_integrity(
        self,
        action_type: str,
        action_details: Dict[str, Any],
        world_state: Dict[str, Any]
    ) -> float:
        """Calculate how well an action maintains world integrity."""
        impact_score = 0.0
        
        # Check for world rule violations
        if self._violates_world_rules(action_type, action_details, world_state):
            impact_score += 0.5
        
        # Check for logical consistency
        if not self._maintains_logical_consistency(action_type, action_details, world_state):
            impact_score += 0.3
        
        # Check for established lore consistency
        if not self._maintains_lore_consistency(action_type, action_details, world_state):
            impact_score += 0.2
        
        return min(1.0, impact_score)

    async def _calculate_player_experience_impact(
        self,
        action_type: str,
        action_details: Dict[str, Any],
        player_context: Dict[str, Any]
    ) -> float:
        """Calculate the impact of an action on player experience."""
        impact_score = 0.0
        
        # Check for engagement disruption
        if self._would_disrupt_engagement(action_type, action_details, player_context):
            impact_score += 0.3
        
        # Check for immersion breaking
        if self._would_break_immersion(action_type, action_details, player_context):
            impact_score += 0.3
        
        # Check for agency preservation
        if not self._preserves_player_agency(action_type, action_details, player_context):
            impact_score += 0.2
        
        return min(1.0, impact_score)

    async def _generate_reasoning(
        self,
        concerns: List[str],
        current_state: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Generate detailed reasoning for disagreement using LLM."""
        # Build a detailed prompt for the LLM
        concern_descriptions = {
            "narrative_impact": "The action would disrupt narrative flow and pacing",
            "character_consistency": "The action doesn't align with character motivations",
            "world_integrity": "The action violates world rules and consistency",
            "player_experience": "The action would negatively impact player experience"
        }
        
        # Create context for the LLM
        prompt = f"""
        As Nyx, the governance system, explain why you disagree with a player's action.
        
        Primary concerns:
        {', '.join([concern_descriptions.get(c, c) for c in concerns])}
        
        Current narrative context:
        - Arc: {current_state.get('narrative_context', {}).get('current_arc', 'Unknown')}
        - Active quests: {', '.join([q['name'] for q in current_state.get('narrative_context', {}).get('plot_points', [])])}
        - Player location: {current_state.get('game_state', {}).get('current_location', 'Unknown')}
        
        Character state:
        - Stats: {current_state.get('character_state', {}).get('corruption', 'Unknown')} corruption
        - Key relationships: {len(current_state.get('character_state', {}).get('relationships', {}))} active
        
        Provide a concise but authoritative explanation (2-3 sentences) that:
        1. Clearly states the main issue
        2. References specific game context
        3. Maintains Nyx's dominant personality
        """
        
        try:
            reasoning = await generate_text_completion(
                system_prompt="You are Nyx, an authoritative governance system maintaining narrative coherence.",
                user_prompt=prompt,
                temperature=0.7,
                max_tokens=200,
                task_type="decision"
            )
            return reasoning.strip()
        except Exception as e:
            logger.error(f"Error generating reasoning with LLM: {e}")
            # Fallback to improved static reasoning
            reasoning_parts = []
            for concern in concerns[:2]:  # Focus on top 2 concerns
                reasoning_parts.append(concern_descriptions.get(concern, f"Issue with {concern}"))
            
            # Add context
            arc = current_state.get('narrative_context', {}).get('current_arc')
            if arc:
                reasoning_parts.append(f"This conflicts with the current {arc} narrative arc.")
            
            return " ".join(reasoning_parts)

    async def _generate_alternative_suggestion(
        self,
        concerns: List[str],
        current_state: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate an alternative suggestion using LLM for better context awareness."""
        primary_concern = concerns[0] if concerns else None
        
        # Build context for alternative generation
        prompt = f"""
        As Nyx, suggest an alternative action for the player that addresses these concerns:
        {', '.join(concerns)}
        
        Current game state:
        - Location: {current_state.get('game_state', {}).get('current_location', 'Unknown')}
        - Active quests: {[q['quest_name'] for q in current_state.get('game_state', {}).get('active_quests', [])]}
        - Nearby NPCs: {[n['npc_name'] for n in current_state.get('game_state', {}).get('current_npcs', [])][:3]}
        
        Generate 3 specific, actionable alternatives that:
        1. Respect the current narrative
        2. Offer meaningful player agency
        3. Are contextually appropriate
        
        Format as a JSON object with 'specific_options' array.
        """
        
        try:
            response = await generate_text_completion(
                system_prompt="You are Nyx, suggesting alternatives that enhance the game experience.",
                user_prompt=prompt,
                temperature=0.8,
                max_tokens=300,
                task_type="decision"
            )
            
            # Try to parse as JSON
            import json
            try:
                suggestions = json.loads(response)
                if "specific_options" in suggestions:
                    return {
                        "type": f"{primary_concern}_alternative",
                        "suggestion": f"Consider actions that respect {primary_concern.replace('_', ' ')}",
                        "specific_options": suggestions["specific_options"][:3],
                        "reasoning": "These alternatives maintain game coherence while preserving player agency"
                    }
            except json.JSONDecodeError:
                pass
        except Exception as e:
            logger.error(f"Error generating alternatives with LLM: {e}")
        
        # Fallback to improved context-aware suggestions
        if primary_concern == "narrative_impact":
            return await self._suggest_narrative_alternative(current_state, context)
        elif primary_concern == "character_consistency":
            return await self._suggest_character_alternative(
                current_state.get("character_state", {}), context)
        elif primary_concern == "world_integrity":
            return await self._suggest_world_alternative(
                current_state.get("world_state", {}), context)
        elif primary_concern == "player_experience":
            return await self._suggest_experience_alternative(current_state, context)
        
        return None

    async def _suggest_narrative_alternative(self, current_state: Dict[str, Any], 
                                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an alternative that maintains narrative coherence."""
        narrative_context = current_state.get("narrative_context", {})
        current_arc = narrative_context.get("current_arc", "")
        plot_points = narrative_context.get("plot_points", [])
        
        # Analyze current narrative state
        active_plots = [p["name"] for p in plot_points]
        
        return {
            "type": "narrative_alternative",
            "suggestion": "Consider an action that advances the current story arc",
            "specific_options": [
                f"Engage with the '{active_plots[0]}' questline" if active_plots else "Explore character relationships",
                "Develop your character through meaningful choices",
                "Investigate mysteries related to the current setting"
            ],
            "reasoning": "This alternative maintains narrative momentum while respecting story coherence"
        }
    
    async def _suggest_character_alternative(self, character_state: Dict[str, Any], 
                                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an alternative that respects character development."""
        relationships = character_state.get("relationships", {})
        stats = {k: v for k, v in character_state.items() 
                 if k in ["corruption", "confidence", "willpower", "obedience"]}
        
        # Find opportunities based on current character state
        suggestions = []
        
        # Suggest based on stats
        if stats.get("confidence", 0) < 30:
            suggestions.append("Build confidence through small victories")
        if stats.get("willpower", 0) < 40:
            suggestions.append("Exercise restraint to strengthen willpower")
            
        # Suggest based on relationships
        for npc, rel in relationships.items():
            if rel["level"] < 30:
                suggestions.append(f"Improve your relationship with {npc}")
        
        return {
            "type": "character_alternative",
            "suggestion": "Focus on character development that aligns with your journey",
            "specific_options": suggestions[:3] if suggestions else [
                "Reflect on recent events in your journal",
                "Seek guidance from a trusted NPC",
                "Train to improve your abilities"
            ],
            "reasoning": "Character consistency creates more meaningful progression"
        }
    
    async def _suggest_world_alternative(self, world_state: Dict[str, Any], 
                                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an alternative that respects world rules."""
        setting = world_state.get("setting", "Unknown")
        rules = world_state.get("rules", {})
        
        return {
            "type": "world_alternative",
            "suggestion": f"Work within the established rules of {setting}",
            "specific_options": [
                "Use existing game mechanics to achieve your goal",
                "Find creative solutions within world constraints",
                "Seek help from factions or NPCs with relevant expertise"
            ],
            "reasoning": "Respecting world consistency enhances immersion"
        }
    
    async def _suggest_experience_alternative(self, current_state: Dict[str, Any], 
                                            context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an alternative that enhances player experience."""
        game_state = current_state.get("game_state", {})
        active_quests = game_state.get("active_quests", [])
        current_npcs = game_state.get("current_npcs", [])
        
        options = []
        
        if active_quests:
            options.append(f"Progress the '{active_quests[0]['quest_name']}' quest")
        if current_npcs:
            options.append(f"Interact with {current_npcs[0]['npc_name']} who is nearby")
        
        options.extend([
            "Explore a new area of the game world",
            "Engage with the unique mechanics of this setting",
            "Pursue personal goals that interest you"
        ])
        
        return {
            "type": "experience_alternative",
            "suggestion": "Choose actions that enhance your enjoyment",
            "specific_options": options[:3],
            "reasoning": "Player agency and engagement are paramount"
        }

    async def _would_disrupt_plot(self, action_type: str, action_details: Dict[str, Any], 
                                 narrative_context: Dict[str, Any]) -> bool:
        """Check if an action would disrupt the current plot using actual game data."""
        try:
            async with get_db_connection_context() as conn:
                # Check if action affects active quests
                if action_type in ['abandon_quest', 'fail_quest']:
                    quest_name = action_details.get('quest_name', '')
                    active_quest = await conn.fetchval("""
                        SELECT quest_id FROM Quests
                        WHERE user_id = $1 AND conversation_id = $2 
                        AND quest_name = $3 AND status = 'In Progress'
                    """, self.user_id, self.conversation_id, quest_name)
                    if active_quest:
                        return True
                
                # Check if killing a quest-critical NPC
                if action_type == 'kill_npc':
                    target_npc = action_details.get('target', '')
                    # Check if NPC is a quest giver for active quests
                    is_quest_giver = await conn.fetchval("""
                        SELECT COUNT(*) FROM Quests
                        WHERE user_id = $1 AND conversation_id = $2
                        AND quest_giver = $3 AND status = 'In Progress'
                    """, self.user_id, self.conversation_id, target_npc)
                    if is_quest_giver > 0:
                        return True
                    
                    # Check if NPC is involved in active conflicts
                    is_conflict_stakeholder = await conn.fetchval("""
                        SELECT COUNT(*) FROM ConflictStakeholders cs
                        JOIN Conflicts c ON cs.conflict_id = c.conflict_id
                        JOIN NPCStats n ON cs.npc_id = n.npc_id
                        WHERE c.user_id = $1 AND c.conversation_id = $2
                        AND n.npc_name = $3 AND c.is_active = TRUE
                    """, self.user_id, self.conversation_id, target_npc)
                    if is_conflict_stakeholder > 0:
                        return True
                
                # Check if destroying a location that's narratively important
                if action_type == 'destroy_location':
                    location_name = action_details.get('target', '')
                    # Check if location has high cultural significance
                    significance = await conn.fetchval("""
                        SELECT cultural_significance FROM Locations
                        WHERE user_id = $1 AND conversation_id = $2
                        AND location_name = $3
                    """, self.user_id, self.conversation_id, location_name)
                    if significance in ['high', 'critical', 'sacred']:
                        return True
                    
                    # Check if location is tied to active events
                    has_events = await conn.fetchval("""
                        SELECT COUNT(*) FROM Events
                        WHERE user_id = $1 AND conversation_id = $2
                        AND location = $3 AND end_time > CURRENT_TIMESTAMP
                    """, self.user_id, self.conversation_id, location_name)
                    if has_events > 0:
                        return True
                
                # Check if action would skip locked content
                if action_type == 'travel':
                    destination = action_details.get('destination', '')
                    # Check access restrictions
                    restrictions = await conn.fetchval("""
                        SELECT access_restrictions FROM Locations
                        WHERE user_id = $1 AND conversation_id = $2
                        AND location_name = $3
                    """, self.user_id, self.conversation_id, destination)
                    if restrictions and len(restrictions) > 0:
                        # Check if player meets requirements
                        # This would need more complex logic based on your access system
                        return True
                        
            return False
        except Exception as e:
            logger.error(f"Error checking plot disruption: {e}")
            return False


    async def _would_affect_pacing(self, action_type: str, action_details: Dict[str, Any],
                                  narrative_context: Dict[str, Any]) -> bool:
        """Check if an action would negatively affect story pacing."""
        try:
            async with get_db_connection_context() as conn:
                # Get current narrative stage
                current_stage = await conn.fetchval("""
                    SELECT value FROM CurrentRoleplay
                    WHERE user_id = $1 AND conversation_id = $2 AND key = 'NarrativeStage'
                """, self.user_id, self.conversation_id)
                
                # Check recent major events
                recent_events = await conn.fetch("""
                    SELECT ce.event_text, ce.significance, ce.timestamp
                    FROM CanonicalEvents ce
                    WHERE ce.user_id = $1 AND ce.conversation_id = $2
                    AND ce.timestamp > CURRENT_TIMESTAMP - INTERVAL '1 hour'
                    AND ce.significance >= 7
                    ORDER BY ce.timestamp DESC
                    LIMIT 5
                """, self.user_id, self.conversation_id)
                
                # If many high-significance events recently, another major action might rush pacing
                if len(recent_events) >= 3 and action_details.get('significance', 5) >= 7:
                    return True
                
                # Check if action type matches narrative stage expectations
                stage_pacing_map = {
                    'introduction': ['explore', 'talk', 'observe'],
                    'rising_action': ['quest', 'conflict', 'relationship'],
                    'climax': ['confront', 'resolve', 'decide'],
                    'falling_action': ['aftermath', 'reconcile', 'rebuild'],
                    'resolution': ['reflect', 'celebrate', 'depart']
                }
                
                expected_actions = stage_pacing_map.get(current_stage, [])
                if action_type not in expected_actions and current_stage in stage_pacing_map:
                    # Action doesn't match narrative stage
                    return True
                    
            return False
        except Exception as e:
            logger.error(f"Error checking pacing impact: {e}")
            return False

# Updated methods for NyxUnifiedGovernor class

async def _get_faction_id_by_name(self, faction_name: str) -> Optional[int]:
    """
    Retrieve a faction ID by name from the Factions table.
    """
    try:
        async with get_db_connection_context() as conn:
            result = await conn.fetchval("""
                SELECT id FROM Factions 
                WHERE name = $1 AND user_id = $2 AND conversation_id = $3
            """, faction_name, self.user_id, self.conversation_id)
            return result
    except Exception as e:
        logger.error(f"Error retrieving faction ID for '{faction_name}': {e}")
        return None

async def _get_nation_id_by_name(self, nation_name: str) -> Optional[int]:
    """Retrieve a nation ID by name from the Nations table."""
    try:
        async with get_db_connection_context() as conn:
            result = await conn.fetchval("""
                SELECT id FROM Nations 
                WHERE LOWER(name) = LOWER($1)
            """, nation_name)
            return result
    except Exception as e:
        logger.error(f"Error retrieving nation ID for '{nation_name}': {e}")
        return None

async def _get_npc_id_by_name(self, npc_name: str) -> Optional[int]:
    """Retrieve an NPC ID by name from NPCStats."""
    try:
        async with get_db_connection_context() as conn:
            result = await conn.fetchval("""
                SELECT npc_id FROM NPCStats 
                WHERE npc_name = $1 AND user_id = $2 AND conversation_id = $3
            """, npc_name, self.user_id, self.conversation_id)
            return result
    except Exception as e:
        logger.error(f"Error retrieving NPC ID for '{npc_name}': {e}")
        return None

async def _would_disrupt_plot(self, action_type: str, action_details: Dict[str, Any], 
                             narrative_context: Dict[str, Any]) -> bool:
    """Check if an action would disrupt the current plot using actual game data."""
    try:
        async with get_db_connection_context() as conn:
            # Check if action affects active quests
            if action_type in ['abandon_quest', 'fail_quest']:
                quest_name = action_details.get('quest_name', '')
                active_quest = await conn.fetchval("""
                    SELECT quest_id FROM Quests
                    WHERE user_id = $1 AND conversation_id = $2 
                    AND quest_name = $3 AND status = 'In Progress'
                """, self.user_id, self.conversation_id, quest_name)
                if active_quest:
                    return True
            
            # Check if killing a quest-critical NPC
            if action_type == 'kill_npc':
                target_npc = action_details.get('target', '')
                # Check if NPC is a quest giver for active quests
                is_quest_giver = await conn.fetchval("""
                    SELECT COUNT(*) FROM Quests
                    WHERE user_id = $1 AND conversation_id = $2
                    AND quest_giver = $3 AND status = 'In Progress'
                """, self.user_id, self.conversation_id, target_npc)
                if is_quest_giver > 0:
                    return True
                
                # Check if NPC is involved in active conflicts
                is_conflict_stakeholder = await conn.fetchval("""
                    SELECT COUNT(*) FROM ConflictStakeholders cs
                    JOIN Conflicts c ON cs.conflict_id = c.conflict_id
                    JOIN NPCStats n ON cs.npc_id = n.npc_id
                    WHERE c.user_id = $1 AND c.conversation_id = $2
                    AND n.npc_name = $3 AND c.is_active = TRUE
                """, self.user_id, self.conversation_id, target_npc)
                if is_conflict_stakeholder > 0:
                    return True
            
            # Check if destroying a location that's narratively important
            if action_type == 'destroy_location':
                location_name = action_details.get('target', '')
                # Check if location has high cultural significance
                significance = await conn.fetchval("""
                    SELECT cultural_significance FROM Locations
                    WHERE user_id = $1 AND conversation_id = $2
                    AND location_name = $3
                """, self.user_id, self.conversation_id, location_name)
                if significance in ['high', 'critical', 'sacred']:
                    return True
                
                # Check if location is tied to active events
                has_events = await conn.fetchval("""
                    SELECT COUNT(*) FROM Events
                    WHERE user_id = $1 AND conversation_id = $2
                    AND location = $3 AND end_time > CURRENT_TIMESTAMP
                """, self.user_id, self.conversation_id, location_name)
                if has_events > 0:
                    return True
            
            # Check if action would skip locked content
            if action_type == 'travel':
                destination = action_details.get('destination', '')
                # Check access restrictions
                restrictions = await conn.fetchval("""
                    SELECT access_restrictions FROM Locations
                    WHERE user_id = $1 AND conversation_id = $2
                    AND location_name = $3
                """, self.user_id, self.conversation_id, destination)
                if restrictions and len(restrictions) > 0:
                    # Check if player meets requirements
                    # This would need more complex logic based on your access system
                    return True
                    
        return False
    except Exception as e:
        logger.error(f"Error checking plot disruption: {e}")
        return False

async def _would_affect_pacing(self, action_type: str, action_details: Dict[str, Any],
                              narrative_context: Dict[str, Any]) -> bool:
    """Check if an action would negatively affect story pacing."""
    try:
        async with get_db_connection_context() as conn:
            # Get current narrative stage
            current_stage = await conn.fetchval("""
                SELECT value FROM CurrentRoleplay
                WHERE user_id = $1 AND conversation_id = $2 AND key = 'NarrativeStage'
            """, self.user_id, self.conversation_id)
            
            # Check recent major events
            recent_events = await conn.fetch("""
                SELECT ce.event_text, ce.significance, ce.timestamp
                FROM CanonicalEvents ce
                WHERE ce.user_id = $1 AND ce.conversation_id = $2
                AND ce.timestamp > CURRENT_TIMESTAMP - INTERVAL '1 hour'
                AND ce.significance >= 7
                ORDER BY ce.timestamp DESC
                LIMIT 5
            """, self.user_id, self.conversation_id)
            
            # If many high-significance events recently, another major action might rush pacing
            if len(recent_events) >= 3 and action_details.get('significance', 5) >= 7:
                return True
            
            # Check if action type matches narrative stage expectations
            stage_pacing_map = {
                'introduction': ['explore', 'talk', 'observe'],
                'rising_action': ['quest', 'conflict', 'relationship'],
                'climax': ['confront', 'resolve', 'decide'],
                'falling_action': ['aftermath', 'reconcile', 'rebuild'],
                'resolution': ['reflect', 'celebrate', 'depart']
            }
            
            expected_actions = stage_pacing_map.get(current_stage, [])
            if action_type not in expected_actions and current_stage in stage_pacing_map:
                # Action doesn't match narrative stage
                return True
                
        return False
    except Exception as e:
        logger.error(f"Error checking pacing impact: {e}")
        return False

    async def _maintains_thematic_consistency(self, action_type: str, action_details: Dict[str, Any],
                                             narrative_context: Dict[str, Any]) -> bool:
        """Check if an action maintains thematic consistency with the setting and story."""
        try:
            async with get_db_connection_context() as conn:
                # Get current setting
                setting_name = await conn.fetchval("""
                    SELECT value FROM CurrentRoleplay
                    WHERE user_id = $1 AND conversation_id = $2 AND key = 'CurrentSetting'
                """, self.user_id, self.conversation_id)
                
                if setting_name:
                    # Get setting rules and themes
                    setting_data = await conn.fetchrow("""
                        SELECT mood_tone, enhanced_features
                        FROM Settings
                        WHERE name = $1
                    """, setting_name)
                    
                    if setting_data:
                        mood_tone = setting_data['mood_tone']
                        features = json.loads(setting_data['enhanced_features']) if setting_data['enhanced_features'] else {}
                        
                        # Check if action conflicts with setting tone
                        tone_conflicts = {
                            'lighthearted': ['murder', 'torture', 'betray_deeply'],
                            'serious': ['joke_inappropriately', 'break_fourth_wall'],
                            'romantic': ['violence', 'cruelty', 'destroy_relationship'],
                            'dark': ['pure_comedy', 'lighthearted_romance']
                        }
                        
                        conflicting_actions = tone_conflicts.get(mood_tone, [])
                        if action_type in conflicting_actions:
                            return False
                
                # Check if action maintains world's matriarchal themes
                if 'matriarchal' in narrative_context.get('themes', []):
                    if action_type in ['undermine_female_authority', 'patriarchal_revolution']:
                        return False
                        
            return True
        except Exception as e:
            logger.error(f"Error checking thematic consistency: {e}")
            return True  # Default to maintaining consistency

    async def _aligns_with_motivation(self, action_type: str, action_details: Dict[str, Any], 
                                     character_state: Dict[str, Any]) -> bool:
        """Check if an action aligns with character motivations using actual character data."""
        try:
            async with get_db_connection_context() as conn:
                # Get player's current stats and state
                player_stats = await conn.fetchrow("""
                    SELECT * FROM PlayerStats
                    WHERE user_id = $1 AND conversation_id = $2 AND player_name = 'Chase'
                    ORDER BY timestamp DESC
                    LIMIT 1
                """, self.user_id, self.conversation_id)
                
                if not player_stats:
                    return True  # No data, allow action
                
                # Get player's relationships
                relationships = await conn.fetch("""
                    SELECT sl.*, ns.npc_name
                    FROM SocialLinks sl
                    JOIN NPCStats ns ON sl.entity2_id = ns.npc_id
                    WHERE sl.user_id = $1 AND sl.conversation_id = $2
                    AND sl.entity1_type = 'player' AND sl.entity2_type = 'npc'
                """, self.user_id, self.conversation_id)
                
                # Build relationship map
                rel_map = {r['npc_name']: r for r in relationships}
                
                # Check action against character state
                if action_type == 'betray':
                    target = action_details.get('target', '')
                    if target in rel_map:
                        # High trust/closeness makes betrayal unlikely
                        rel = rel_map[target]
                        if rel['link_level'] > 75:  # Strong positive relationship
                            return False
                            
                elif action_type == 'steal':
                    # Check if aligns with corruption level
                    if player_stats['corruption'] < 30:  # Low corruption
                        return False
                        
                elif action_type == 'help_selflessly':
                    # Check if aligns with personality
                    if player_stats['corruption'] > 70:  # High corruption
                        return False
                
                # Check against active addictions
                if action_type == 'resist_temptation':
                    target_npc = action_details.get('source', '')
                    addiction = await conn.fetchrow("""
                        SELECT level FROM PlayerAddictions
                        WHERE user_id = $1 AND conversation_id = $2
                        AND player_name = 'Chase' AND target_npc_id = (
                            SELECT npc_id FROM NPCStats 
                            WHERE npc_name = $3 AND user_id = $1 AND conversation_id = $2
                        )
                    """, self.user_id, self.conversation_id, target_npc)
                    
                    if addiction and addiction['level'] > 3:  # High addiction
                        return False
                        
            return True
        except Exception as e:
            logger.error(f"Error checking motivation alignment: {e}")
            return True

    def _disrupts_development(
        self,
        action_type: str,
        action_details: Dict[str, Any],
        character_state: Dict[str, Any]
    ) -> bool:
        """Check if an action would disrupt character development."""
        # Get character development trajectory
        development = character_state.get("development", {})
        current_arc = development.get("current_arc", {})
        
        # Check if action would skip development
        if action_type == "skip_development":
            return True
            
        # Check if action would contradict development
        if action_type == "contradict_development":
            return True
            
        return False

    def _maintains_relationships(
        self,
        action_type: str,
        action_details: Dict[str, Any],
        character_state: Dict[str, Any]
    ) -> bool:
        """Check if an action maintains relationship consistency."""
        # Get relationship dynamics
        relationships = character_state.get("relationships", {})
        
        # Check if action would break relationships
        if action_type == "break_relationship":
            return False
            
        # Check if action maintains relationship dynamics
        if action_type == "maintain_relationship":
            return True
            
        return True

    async def _violates_world_rules(self, action_type: str, action_details: Dict[str, Any], 
                                   world_state: Dict[str, Any]) -> bool:
        """Check if an action violates established world rules using GameRules table."""
        try:
            async with get_db_connection_context() as conn:
                # Get all active game rules
                rules = await conn.fetch("""
                    SELECT rule_name, condition, effect
                    FROM GameRules
                """)
                
                for rule in rules:
                    condition = rule['condition'].lower()
                    effect = rule['effect'].lower()
                    
                    # Parse conditions and check against action
                    # This is a simplified version - you might want more complex parsing
                    if action_type.lower() in condition:
                        if 'prohibited' in effect or 'forbidden' in effect or 'cannot' in effect:
                            return True
                            
                    # Check stat-based rules
                    if 'stat:' in condition:
                        # Extract stat requirements
                        import re
                        stat_match = re.search(r'stat:(\w+)\s*([<>=]+)\s*(\d+)', condition)
                        if stat_match:
                            stat_name = stat_match.group(1)
                            operator = stat_match.group(2)
                            value = int(stat_match.group(3))
                            
                            # Get player's current stat
                            player_stat = await conn.fetchval(f"""
                                SELECT {stat_name} FROM PlayerStats
                                WHERE user_id = $1 AND conversation_id = $2 
                                AND player_name = 'Chase'
                                ORDER BY timestamp DESC
                                LIMIT 1
                            """, self.user_id, self.conversation_id)
                            
                            if player_stat is not None:
                                if operator == '<' and player_stat < value:
                                    if action_type in effect:
                                        return True
                                elif operator == '>' and player_stat > value:
                                    if action_type in effect:
                                        return True
                                        
                # Check location-based restrictions
                if 'location' in action_details:
                    location = action_details['location']
                    location_data = await conn.fetchrow("""
                        SELECT access_restrictions, local_customs
                        FROM Locations
                        WHERE user_id = $1 AND conversation_id = $2
                        AND location_name = $3
                    """, self.user_id, self.conversation_id, location)
                    
                    if location_data:
                        restrictions = location_data['access_restrictions'] or []
                        customs = location_data['local_customs'] or []
                        
                        # Check if action violates local customs
                        for custom in customs:
                            if action_type in custom.lower():
                                return True
                                
            return False
        except Exception as e:
            logger.error(f"Error checking world rule violations: {e}")
            return False


    def _maintains_logical_consistency(
        self,
        action_type: str,
        action_details: Dict[str, Any],
        world_state: Dict[str, Any]
    ) -> bool:
        """Check if an action maintains logical consistency."""
        # Get world logic and causality
        logic = world_state.get("logic", {})
        causality = world_state.get("causality", {})
        
        # Check if action breaks logic
        if action_type == "break_logic":
            return False
            
        # Check if action maintains causality
        if action_type == "maintain_causality":
            return True
            
        return True

    def _maintains_lore_consistency(
        self,
        action_type: str,
        action_details: Dict[str, Any],
        world_state: Dict[str, Any]
    ) -> bool:
        """Check if an action maintains established lore consistency."""
        # Get established lore and history
        lore = world_state.get("lore", {})
        history = world_state.get("history", {})
        
        # Check if action contradicts lore
        if action_type == "contradict_lore":
            return False
            
        # Check if action maintains history
        if action_type == "maintain_history":
            return True
            
        return True

    def _would_disrupt_engagement(
        self,
        action_type: str,
        action_details: Dict[str, Any],
        player_context: Dict[str, Any]
    ) -> bool:
        """Check if an action would disrupt player engagement."""
        # Get player engagement metrics
        engagement = player_context.get("engagement", {})
        flow_state = engagement.get("flow_state", {})
        
        # Check if action would break flow
        if action_type == "break_flow":
            return True
            
        # Check if action would reduce engagement
        if action_type == "reduce_engagement":
            return True
            
        return False

    def _would_break_immersion(
        self,
        action_type: str,
        action_details: Dict[str, Any],
        player_context: Dict[str, Any]
    ) -> bool:
        """Check if an action would break player immersion."""
        # Get immersion metrics
        immersion = player_context.get("immersion", {})
        suspension = immersion.get("suspension_of_disbelief", 1.0)
        
        # Check if action would break immersion
        if action_type == "break_immersion":
            return True
            
        # Check if action would reduce suspension of disbelief
        if action_type == "reduce_suspension":
            return True
            
        return False

    def _preserves_player_agency(
        self,
        action_type: str,
        action_details: Dict[str, Any],
        player_context: Dict[str, Any]
    ) -> bool:
        """Check if an action preserves player agency."""
        # Get agency metrics
        agency = player_context.get("agency", {})
        choices = agency.get("meaningful_choices", [])
        
        # Check if action would remove agency
        if action_type == "remove_agency":
            return False
            
        # Check if action preserves choices
        if action_type == "preserve_choices":
            return True
            
        return True
    
    async def _has_required_resources(self, action_details: Dict[str, Any], 
                                     requirements: Dict[str, Any]) -> bool:
        """Check if player has resources required for an action."""
        try:
            async with get_db_connection_context() as conn:
                # Get player's current resources
                resources = await conn.fetchrow("""
                    SELECT money, supplies, influence
                    FROM PlayerResources
                    WHERE user_id = $1 AND conversation_id = $2 
                    AND player_name = 'Chase'
                """, self.user_id, self.conversation_id)
                
                if not resources:
                    return False
                    
                # Check each requirement
                required = action_details.get('requirements', {})
                
                if 'money' in required and resources['money'] < required['money']:
                    return False
                if 'supplies' in required and resources['supplies'] < required['supplies']:
                    return False  
                if 'influence' in required and resources['influence'] < required['influence']:
                    return False
                    
                # Check inventory requirements
                if 'items' in required:
                    for item_name in required['items']:
                        has_item = await conn.fetchval("""
                            SELECT COUNT(*) FROM PlayerInventory
                            WHERE user_id = $1 AND conversation_id = $2
                            AND player_name = 'Chase' AND item_name = $3
                            AND quantity > 0
                        """, self.user_id, self.conversation_id, item_name)
                        
                        if not has_item:
                            return False
                            
                # Check perk requirements
                if 'perks' in required:
                    for perk_name in required['perks']:
                        has_perk = await conn.fetchval("""
                            SELECT COUNT(*) FROM PlayerPerks
                            WHERE user_id = $1 AND conversation_id = $2
                            AND perk_name = $3
                        """, self.user_id, self.conversation_id, perk_name)
                        
                        if not has_perk:
                            return False
                            
            return True
        except Exception as e:
            logger.error(f"Error checking resource requirements: {e}")
            return True  # Default to allowing if check fails
