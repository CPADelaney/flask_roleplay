# nyx/nyx_governance.py

"""
Unified governance system for Nyx to control all agents (NPCs and beyond).
...
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

import asyncpg

# The agent trace utility
from agents import trace, RunContextWrapper

# Database connection helper
from db.connection import get_db_connection_context

# --- MERGE: Import the new LoreSystem ---
# Use TYPE_CHECKING to avoid circular import
if TYPE_CHECKING:
    from lore.lore_system import LoreSystem

from nyx.constants import DirectiveType, DirectivePriority, AgentType

# Memory system references
from memory.wrapper import MemorySystem
from memory.memory_nyx_integration import get_memory_nyx_bridge

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
        # Use Optional[Any] instead of Optional[LoreSystem] to avoid needing the import
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

        # --- MERGE: Get an instance of the LoreSystem during initialization ---
        # Nyx needs its primary tool to execute its will.
        self.lore_system = await LoreSystem.get_instance(self.user_id, self.conversation_id)

        # Initialize other systems
        await self._initialize_systems()
        self._initialized = True
        return self

    async def _initialize_systems(self):
        """Initialize memory system, game state, and discover agents."""
        # --- MERGE: Import LoreSystem locally to avoid circular import ---
        from lore.lore_system import LoreSystem
        
        # --- MERGE: Get an instance of the LoreSystem during initialization ---
        # Nyx needs its primary tool to execute its will.
        self.lore_system = LoreSystem.get_instance(self.user_id, self.conversation_id)
        
        # Initialize the lore system
        await self.lore_system.initialize()
    
        # Initialize other systems
        self.memory_system = await get_memory_nyx_bridge(self.user_id, self.conversation_id)
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

    async def initialize_game_state(self) -> Dict[str, Any]:
        """Fetch and return the game state for current user/conversation."""
        logger.info(f"Initializing game state for user {self.user_id}, conversation {self.conversation_id}")

        game_state = dict(
            user_id=self.user_id,
            conversation_id=self.conversation_id,
            current_location=None,
            current_npcs=[],
            current_time=None,
            active_quests=[],
            player_stats={},
            narrative_state={},
            world_state={},
        )
        
        try:
            # Fetch current roleplay state from database
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

            logger.info(f"Game state initialized with {len(game_state['current_npcs'])} NPCs and {len(game_state['active_quests'])} quests.")
            return game_state
        except Exception as e:
            logger.error(f"Error initializing game state: {e}")
            return game_state

    async def discover_and_register_agents(self):
        """
        Discover and register available agents in the system.
        """
        logger.info(f"Discovering and registering agents for user {self.user_id}, conversation {self.conversation_id}")
        
        # Example of registering some default agents
        try:
            # Register story director if available
            try:
                from story_agent.story_director_agent import StoryDirector
                story_director = StoryDirector(self.user_id, self.conversation_id)
                await self.register_agent(
                    agent_type=AgentType.STORY_DIRECTOR,
                    agent_instance=story_director,
                    agent_id="story_director"
                )
                logger.info("Registered story director agent")
            except (ImportError, Exception) as e:
                logger.warning(f"Could not register story director: {e}")
            
            # Register universal updater if available
            try:
                from logic.universal_updater_agent import UniversalUpdaterAgent
                universal_updater = UniversalUpdaterAgent(self.user_id, self.conversation_id)
                await self.register_agent(
                    agent_type=AgentType.UNIVERSAL_UPDATER,
                    agent_instance=universal_updater,
                    agent_id="universal_updater"
                )
                logger.info("Registered universal updater agent")
            except (ImportError, Exception) as e:
                logger.warning(f"Could not register universal updater: {e}")
            
            return True
        except Exception as e:
            logger.error(f"Error during agent discovery and registration: {e}")
            return False    
    
    async def _update_performance_metrics(self):
        """Update per-agent performance metrics."""
        for agent_type, agents_dict in self.registered_agents.items():
            if agent_type not in self.agent_performance:
                self.agent_performance[agent_type] = {}
            for agent_id, agent in agents_dict.items():
                if hasattr(agent, 'get_performance_metrics'):
                    metrics = await agent.get_performance_metrics()
                    self.agent_performance[agent_type][agent_id] = metrics
                    for strategy, data in metrics.get("strategies", {}).items():
                        if strategy not in self.strategy_effectiveness:
                            self.strategy_effectiveness[strategy] = {
                                "success": 0,
                                "total": 0,
                                "agents": {}
                            }
                        self.strategy_effectiveness[strategy]["success"] += data["success"]
                        self.strategy_effectiveness[strategy]["total"] += data["total"]
                        self.strategy_effectiveness[strategy]["agents"][f"{agent_type}:{agent_id}"] = data

    async def _load_learning_state(self):
        """Load and aggregate learning/adaptation patterns for all agents."""
        for agent_type, agents_dict in self.registered_agents.items():
            if agent_type not in self.agent_learning:
                self.agent_learning[agent_type] = {}
            for agent_id, agent in agents_dict.items():
                if hasattr(agent, "get_learning_state"):
                    learning_data = await agent.get_learning_state()
                    self.agent_learning[agent_type][agent_id] = learning_data

                    for pattern, data in learning_data.get("patterns", {}).items():
                        if pattern not in self.adaptation_patterns:
                            self.adaptation_patterns[pattern] = {
                                "success": 0,
                                "total": 0,
                                "agents": {}
                            }
                        self.adaptation_patterns[pattern]["success"] += data["success"]
                        self.adaptation_patterns[pattern]["total"] += data["total"]
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
                # Capabilities match
                capabilities = []
                if hasattr(agent, 'get_capabilities'):
                    capabilities = await agent.get_capabilities()
                capability_match = sum(
                    1 for cap in requirements["capabilities"]
                    if cap in capabilities
                ) / len(requirements["capabilities"]) if requirements["capabilities"] else 0
                if capability_match > 0.5:
                    relevant_agents.append({
                        "type": agent_type,
                        "id": agent_id,
                        "instance": agent,
                        "relevance": capability_match
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
        """
        Get directives for a specific agent.
        
        Args:
            agent_type: Type of agent
            agent_id: ID of agent
            
        Returns:
            List of active directives for the agent
        """
        # Get the cache key for this agent
        cache_key = f"agent_directives:{agent_type}:{agent_id}"
        
        # Define the function to fetch directives if not in cache
        async def fetch_agent_directives():
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
            return active_directives
    
        # Try to get from cache first, fall back to fetching
        if AGENT_DIRECTIVE_CACHE is not None:
            try:
                # Don't use keyword arguments with dict.get
                ttl_value = getattr(CACHE_TTL, 'DIRECTIVES', 300)
                
                # Use a cache method that supports TTL properly
                if hasattr(AGENT_DIRECTIVE_CACHE, 'get_with_ttl'):
                    directives = await AGENT_DIRECTIVE_CACHE.get_with_ttl(
                        cache_key, 
                        fetch_agent_directives,
                        ttl_value
                    )
                # Or implement a fallback approach
                elif cache_key in AGENT_DIRECTIVE_CACHE:
                    directives = AGENT_DIRECTIVE_CACHE[cache_key]
                else:
                    directives = await fetch_agent_directives()
                    AGENT_DIRECTIVE_CACHE[cache_key] = directives
                    # Set expiration separately if needed
                
                return directives
            except Exception as e:
                logger.error(f"Error fetching agent directives from cache: {e}")
        
        # Direct fetch if cache fails or isn't available
        return await fetch_agent_directives()
    
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
    
    async def _execute_setup_task(
        self,
        agents: List[Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute setup task with memory-capable agents."""
        setup_results = []
        
        for agent in agents:
            if hasattr(agent, "initialize_systems"):
                result = await agent.initialize_systems(context)
                setup_results.append(result)
        
        return {
            "status": "completed",
            "results": setup_results
        }
    
    async def _execute_planning_task(
        self,
        agents: List[Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute planning task with planning-capable agents."""
        plans = []
        
        for agent in agents:
            if hasattr(agent, "generate_plan"):
                plan = await agent.generate_plan(context)
                plans.append(plan)
        
        # Merge plans
        merged_plan = await self._merge_plans(plans)
        
        return {
            "status": "completed",
            "merged_plan": merged_plan
        }
    
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
    
    async def _merge_plans(self, plans: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple plans into a single coherent plan."""
        merged = {
            "tasks": [],
            "dependencies": [],
            "timeline": [],
            "resources": {}
        }
        
        # Merge tasks
        for plan in plans:
            merged["tasks"].extend(plan.get("tasks", []))
            merged["dependencies"].extend(plan.get("dependencies", []))
            merged["timeline"].extend(plan.get("timeline", []))
            
            # Merge resources
            for resource, amount in plan.get("resources", {}).items():
                if resource not in merged["resources"]:
                    merged["resources"][resource] = 0
                merged["resources"][resource] += amount
        
        # Deduplicate and sort
        merged["tasks"] = list({t["id"]: t for t in merged["tasks"]}.values())
        merged["dependencies"] = list({d["id"]: d for d in merged["dependencies"]}.values())
        merged["timeline"].sort(key=lambda x: x["timestamp"])
        
        return merged
    
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
        """Calculate metrics for a single task."""
        metrics = {
            "coordination_score": 0.0,
            "learning_score": 0.0,
            "resource_usage": {},
            "start_time": time.time(),
            "end_time": time.time()
        }
        
        # Calculate coordination score based on agent interaction
        if len(agents) > 1:
            interaction_count = sum(
                1 for result in output.get("results", [])
                if result.get("interaction_count", 0) > 0
            )
            metrics["coordination_score"] = interaction_count / len(agents)
        
        # Calculate learning score based on adaptations
        adaptation_count = sum(
            1 for result in output.get("results", [])
            if result.get("adapted", False)
        )
        metrics["learning_score"] = adaptation_count / len(agents)
        
        # Calculate resource usage
        for result in output.get("results", []):
            for resource, amount in result.get("resource_usage", {}).items():
                if resource not in metrics["resource_usage"]:
                    metrics["resource_usage"][resource] = 0
                metrics["resource_usage"][resource] += amount
        
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
        """Get the limit for a specific resource."""
        limits = {
            "memory": 1000,  # MB
            "computation": 100,  # CPU seconds
            "time": 3600  # seconds
        }
        return limits.get(resource, float("inf"))
    
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


    def _extract_goals_from_memories(self, memories):
        """Extract goals from memory objects"""
        goals = []
        for memory in memories:
            # Look for goal info in memory text or metadata
            if "goal" in memory.get("text", "").lower():
                # Parse goal information from memory
                # This is a placeholder implementation - enhance based on your memory format
                goals.append({
                    "id": memory.get("id"),
                    "description": memory.get("text"),
                    "status": "active"
                })
            
            # Check metadata for goal information
            metadata = memory.get("metadata", {})
            if metadata.get("type") == "goal":
                goals.append({
                    "id": memory.get("id"),
                    "description": memory.get("text"),
                    "status": metadata.get("status", "active")
                })
                
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
        Retrieve a faction ID by name.
        Note: Your schema doesn't have a dedicated Factions table, 
        so this assumes you'll add one or use an existing pattern.
        """
        try:
            async with get_db_connection_context() as conn:
                # If you have a Factions table (not in current schema)
                result = await conn.fetchval("""
                    SELECT id FROM Factions 
                    WHERE name = $1 AND user_id = $2 AND conversation_id = $3
                """, faction_name, self.user_id, self.conversation_id)
                return result
        except Exception as e:
            logger.error(f"Error retrieving faction ID for '{faction_name}': {e}")
            return None
    
    async def _get_nation_id_by_name(self, nation_name: str) -> Optional[int]:
        """Retrieve a nation ID by name."""
        try:
            async with get_db_connection_context() as conn:
                result = await conn.fetchval("""
                    SELECT id FROM nations 
                    WHERE LOWER(name) = LOWER($1)
                """, nation_name)
                return result
        except Exception as e:
            logger.error(f"Error retrieving nation ID for '{nation_name}': {e}")
            return None
    
    async def _get_npc_id_by_name(self, npc_name: str) -> Optional[int]:
        """Retrieve an NPC ID by name."""
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
        
        Args:
            npc1_name: Name of first NPC
            npc2_name: Name of second NPC
            relationship_type: Type of relationship
            details: Relationship details
            reason: Narrative reason
            
        Returns:
            Result of the operation
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
            await conn.execute("""...""", ...)
            
            # Update NPC relationships field
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
        
        # Now update through LoreSystem (outside the connection context)
        for npc_id, other_name in [(npc1_id, npc2_name), (npc2_id, npc1_name)]:
            # Update through LoreSystem
            await self.lore_system.propose_and_enact_change(
                ctx=ctx,
                entity_type="NPCStats",
                entity_identifier={"npc_id": npc_id},
                updates={"relationships": json.dumps(relationships)},
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
        """
        Analyze a conflict between agents.
        
        Returns:
            Analysis results including impact assessment
        """
        analysis = {
            "agents": [
                {"type": agent1_type, "id": agent1_id},
                {"type": agent2_type, "id": agent2_id}
            ],
            "conflict_type": conflict_details.get("type", "unknown"),
            "severity": conflict_details.get("severity", 5),
            "narrative_impact": 0.0,
            "world_consistency_impact": 0.0,
            "player_experience_impact": 0.0
        }

        # Calculate impacts based on conflict type and agent types
        if conflict_details.get("type") == "narrative_contradiction":
            analysis["narrative_impact"] = 0.8
            analysis["world_consistency_impact"] = 0.6
        elif conflict_details.get("type") == "resource_competition":
            analysis["narrative_impact"] = 0.3
            analysis["world_consistency_impact"] = 0.2
        elif conflict_details.get("type") == "goal_conflict":
            analysis["narrative_impact"] = 0.5
            analysis["player_experience_impact"] = 0.4

        # Adjust based on agent types
        if AgentType.STORY_DIRECTOR in [agent1_type, agent2_type]:
            analysis["narrative_impact"] *= 1.5
        if AgentType.UNIVERSAL_UPDATER in [agent1_type, agent2_type]:
            analysis["world_consistency_impact"] *= 1.5

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
        """Generate detailed reasoning for disagreement."""
        reasoning_parts = []
        
        for concern in concerns:
            if concern == "narrative_impact":
                reasoning_parts.append(
                    "This action would significantly disrupt the current narrative flow and pacing."
                )
            elif concern == "character_consistency":
                reasoning_parts.append(
                    "This action doesn't align with the established character motivations and development."
                )
            elif concern == "world_integrity":
                reasoning_parts.append(
                    "This action would violate established world rules and logical consistency."
                )
            elif concern == "player_experience":
                reasoning_parts.append(
                    "This action would negatively impact the overall player experience and immersion."
                )
        
        # Add narrative context if available
        narrative_context = current_state.get("narrative_context", {})
        if narrative_context:
            reasoning_parts.append(
                f"Current narrative context: {narrative_context.get('description', '')}"
            )
        
        return " ".join(reasoning_parts)

    async def _generate_alternative_suggestion(
        self,
        concerns: List[str],
        current_state: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate an alternative suggestion that addresses concerns."""
        # Get current narrative elements
        narrative_context = current_state.get("narrative_context", {})
        character_state = current_state.get("character_state", {})
        world_state = current_state.get("world_state", {})
        
        # Generate alternative based on primary concern
        primary_concern = concerns[0] if concerns else None
        
        if primary_concern == "narrative_impact":
            return await self._suggest_narrative_alternative(
                current_state,
                context
            )
        elif primary_concern == "character_consistency":
            return await self._suggest_character_alternative(
                character_state,
                context
            )
        elif primary_concern == "world_integrity":
            return await self._suggest_world_alternative(
                world_state,
                context
            )
        elif primary_concern == "player_experience":
            return await self._suggest_experience_alternative(
                current_state,
                context
            )
        
        return None

    def _would_disrupt_plot(
        self,
        action_type: str,
        action_details: Dict[str, Any],
        narrative_context: Dict[str, Any]
    ) -> bool:
        """Check if an action would disrupt the current plot."""
        # Get current plot points and story arcs
        current_arc = narrative_context.get("current_arc", {})
        plot_points = narrative_context.get("plot_points", [])
        
        # Check if action would skip or invalidate plot points
        if action_type == "skip_plot_point":
            return True
            
        # Check if action would contradict established plot elements
        if action_type == "contradict_plot":
            return True
            
        # Check if action would break story arc progression
        if action_type == "break_arc_progression":
            return True
            
        return False

    def _would_affect_pacing(
        self,
        action_type: str,
        action_details: Dict[str, Any],
        narrative_context: Dict[str, Any]
    ) -> bool:
        """Check if an action would negatively affect story pacing."""
        # Get current pacing information
        current_pacing = narrative_context.get("pacing", {})
        tension_level = current_pacing.get("tension_level", 0.5)
        
        # Check if action would break tension
        if action_type == "break_tension":
            return True
            
        # Check if action would rush or slow pacing
        if action_type == "affect_pacing":
            return True
            
        return False

    def _maintains_thematic_consistency(
        self,
        action_type: str,
        action_details: Dict[str, Any],
        narrative_context: Dict[str, Any]
    ) -> bool:
        """Check if an action maintains thematic consistency."""
        # Get current themes and motifs
        themes = narrative_context.get("themes", [])
        motifs = narrative_context.get("motifs", [])
        
        # Check if action aligns with themes
        if action_type == "contradict_theme":
            return False
            
        # Check if action maintains motif consistency
        if action_type == "break_motif":
            return False
            
        return True

    def _aligns_with_motivation(
        self,
        action_type: str,
        action_details: Dict[str, Any],
        character_state: Dict[str, Any]
    ) -> bool:
        """Check if an action aligns with character motivations."""
        # Get character motivations and goals
        motivations = character_state.get("motivations", [])
        goals = character_state.get("goals", [])
        
        # Check if action contradicts motivations
        if action_type == "contradict_motivation":
            return False
            
        # Check if action aligns with goals
        if action_type == "align_with_goal":
            return True
            
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

    def _violates_world_rules(
        self,
        action_type: str,
        action_details: Dict[str, Any],
        world_state: Dict[str, Any]
    ) -> bool:
        """Check if an action violates established world rules."""
        # Get world rules and systems
        rules = world_state.get("rules", {})
        systems = world_state.get("systems", {})
        
        # Check if action violates rules
        if action_type == "violate_rule":
            return True
            
        # Check if action breaks systems
        if action_type == "break_system":
            return True
            
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
    
