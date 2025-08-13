# nyx/governance/agents.py
"""
Agent coordination and management.
"""
import logging
import importlib
import pkgutil
import inspect
import time
from typing import Dict, Any, Optional, List, Union, Set, Callable
from datetime import datetime, timedelta
from .constants import DirectiveType, DirectivePriority, AgentType
from utils.caching import CACHE_TTL, NPC_DIRECTIVE_CACHE, AGENT_DIRECTIVE_CACHE
from utils.cache_manager import CacheManager
from db.connection import get_db_connection_context

import asyncio, importlib, inspect
from contextlib import suppress

logger = logging.getLogger(__name__)


_directive_cache_manager = CacheManager(
    name="agent_directives",
    max_size=1000,
    ttl=CACHE_TTL.DIRECTIVES
)

ASYNC_IMPORT_TIMEOUT = 4.0      # sec per module
AGENT_INIT_TIMEOUT   = 6.0      # sec per agent.__init__
MAX_TOTAL_DISCOVERY  = 20.0     # fail-safe ceiling for the whole routine

class AgentGovernanceMixin:
    """Handles agent coordination and management functions."""
    def __init__(self):
        self._discovery_completed = False  # Add flag
    
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

    def is_agent_registered(self, agent_id: str, agent_type: Optional[str] = None) -> bool:
        """
        Check if an agent is already registered.
        
        Args:
            agent_id: ID of the agent to check
            agent_type: Optional agent type to narrow the search
            
        Returns:
            True if the agent is registered, False otherwise
        """
        if agent_type:
            # Check specific agent type
            return (agent_type in self.registered_agents and 
                    agent_id in self.registered_agents.get(agent_type, {}))
        else:
            # Check all agent types
            for agents_dict in self.registered_agents.values():
                if agent_id in agents_dict:
                    return True
            return False
    
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
    
    @staticmethod  # Add static method decorator
    async def _safe_import(module_path: str):
        """
        Import a module in a worker-thread so that any slow I/O
        (network, disk, heavy static initialisation) can't stall the loop.
        """
        return await asyncio.to_thread(importlib.import_module, module_path)
    
    
    async def discover_and_register_agents(self) -> None:
        """
        Discover and register available agents with the governance system.
        This method automatically finds and registers agents from various sources.
        """
        if self._discovery_completed:
            logger.info("[discover] already completed, skipping")
            return
        
        logger.info(f"[discover] starting for user={self.user_id}, conv={self.conversation_id}")
        
        registrations = []
        
        # 1. Core story agents
        story_agent_types = [
            AgentType.WORLD_DIRECTOR,
            AgentType.UNIVERSAL_UPDATER,
            AgentType.CONFLICT_ANALYST,
            AgentType.SCENE_MANAGER,
            AgentType.NARRATIVE_CRAFTER,
        ]
        
        for agent_type in story_agent_types:
            # Special case: World Director
            if agent_type == AgentType.WORLD_DIRECTOR:
                try:
                    from story_agent.world_director_agent import CompleteWorldDirector
                    world_director = CompleteWorldDirector(self.user_id, self.conversation_id)
                    agent_id = f"world_director_{self.conversation_id}"
                    registrations.append((agent_type, agent_id))
                    logger.info(f"[discover] queued {agent_type.value} → {agent_id}")
                except ImportError as e:
                    logger.error(f"[discover] Failed to import world director: {e}")
                continue
            
            # Special case: Universal Updater
            if agent_type == AgentType.UNIVERSAL_UPDATER:
                try:
                    from logic.universal_updater_agent import UniversalUpdaterAgent
                    universal_updater = UniversalUpdaterAgent(self.user_id, self.conversation_id)
                    agent_id = f"universal_updater_{self.conversation_id}"
                    registrations.append((agent_type, agent_id))
                    logger.info(f"[discover] queued {agent_type.value} → {agent_id}")
                except ImportError as e:
                    logger.error(f"[discover] Failed to import universal updater: {e}")
                continue
            
            # Special case: Conflict system agents
            if agent_type == AgentType.CONFLICT_ANALYST:
                try:
                    # Import conflict system integration
                    from logic.conflict_system.conflict_integration import (
                        ConflictSystemIntegration,
                        register_enhanced_integration
                    )
                    
                    # Create and register conflict system
                    logger.info(f"[discover] initializing conflict system integration")
                    
                    # Use the static registration method
                    result = await register_enhanced_integration(self.user_id, self.conversation_id)
                    
                    if result.get("success"):
                        conflict_system = result.get("integration")
                        if conflict_system:
                            agent_id = f"conflict_system_{self.conversation_id}"
                            registrations.append((agent_type, agent_id))
                            logger.info(f"[discover] queued conflict_system → {agent_id}")
                    else:
                        logger.error(f"[discover] Failed to register conflict system: {result.get('message', 'Unknown error')}")
                        
                except ImportError as e:
                    logger.error(f"[discover] Failed to import conflict system: {e}")
                except Exception as e:
                    logger.error(f"[discover] Failed to initialize conflict system: {e}")
                continue
        
        # 2. Specialized open-world agents (from story_agent.specialized_agents)
        try:
            from story_agent.specialized_agents import (
                initialize_specialized_agents,
                OpenWorldAgentType
            )
            
            specialized = initialize_specialized_agents()
            for key, agent_instance in specialized.items():
                # Map to governance agent types if available
                agent_id = f"{key}_{self.conversation_id}"
                # Note: These are slice-of-life agents, not conflict agents
                # They handle daily life, relationships, etc.
        except ImportError as e:
            logger.error(f"[discover] Failed to load specialized agents: {e}")
        except Exception as e:
            logger.error(f"[discover] Failed to load specialized agents: {e}")
        
        # 3. SDK-based agents
        sdk_agents = [
            ("memory_manager", AgentType.MEMORY_MANAGER),
            ("scene_manager", AgentType.SCENE_MANAGER),
        ]
        
        for agent_name, agent_type in sdk_agents:
            agent_id = f"{agent_name}_{self.conversation_id}"
            registrations.append((agent_type, agent_id))
            logger.info(f"[discover] queued SDK agent {agent_name} → {agent_id}")
        
        # 4. Discover NPC agents
        try:
            async with get_db_connection_context() as conn:
                # Get active NPCs
                npcs = await conn.fetch("""
                    SELECT npc_id, npc_name FROM NPCStats
                    WHERE user_id = $1 AND conversation_id = $2
                      AND introduced = true
                    LIMIT 10
                """, self.user_id, self.conversation_id)
                
                for npc in npcs:
                    agent_id = f"npc_{npc['npc_id']}"
                    registrations.append((AgentType.NPC, agent_id))
                    logger.info(f"[discover] queued NPC {npc['npc_name']} → {agent_id}")
                    
        except Exception as e:
            logger.error(f"[discover] Failed to discover NPC agents: {e}")
        
        # 5. Relationship manager
        try:
            from logic.relationship_integration import RelationshipIntegration
            rel_manager = RelationshipIntegration(self.user_id, self.conversation_id)
            agent_id = f"relationship_manager_{self.conversation_id}"
            registrations.append((AgentType.RELATIONSHIP_MANAGER, agent_id))
            logger.info(f"[discover] queued relationship_manager → {agent_id}")
        except ImportError as e:
            logger.error(f"[discover] Failed to import relationship manager: {e}")
        
        # 6. Resource optimizer (if available)
        try:
            from logic.resource_optimizer import ResourceOptimizer
            optimizer = ResourceOptimizer(self.user_id, self.conversation_id)
            agent_id = f"resource_optimizer_{self.conversation_id}"
            registrations.append((AgentType.RESOURCE_OPTIMIZER, agent_id))
            logger.info(f"[discover] queued resource_optimizer → {agent_id}")
        except ImportError:
            # Resource optimizer is optional
            pass
        
        # 7. Register all discovered agents
        for agent_type, agent_id in registrations:
            await self.register_agent(
                agent_type=agent_type,
                agent_id=agent_id,
                agent_instance=None  # Will be created lazily
            )
        
        self._discovery_completed = True
        logger.info(f"[discover] completed – {len(registrations)} agents queued")
    
    
    async def _discover_npc_agents(governor):
        """background helper – NPC lookup may hit the DB"""
        try:
            async with get_db_connection_context() as conn:
                rows = await conn.fetch("""
                    SELECT npc_id, npc_name
                    FROM   NPCStats
                    WHERE  user_id=$1 AND conversation_id=$2 AND is_active
                    LIMIT  10
                """, governor.user_id, governor.conversation_id)
    
            for row in rows:
                from npcs.npc_agent import NPCAgent
                npc_agent = NPCAgent(governor.user_id,
                                     governor.conversation_id,
                                     row["npc_id"])
                await governor.register_agent(
                    agent_type=AgentType.NPC,
                    agent_id=f"npc_{row['npc_id']}",
                    agent_instance=npc_agent
                )
                logger.info(f"[discover] NPC agent queued for {row['npc_name']}")
        except Exception as e:
            logger.warning(f"[discover] NPC discovery failed: {e!r}")

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
            # World director now manages overarching world simulation
            AgentType.WORLD_DIRECTOR: ["narrative_planning", "plot_development", "pacing_control"],
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
        if agent_type == AgentType.WORLD_DIRECTOR:
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
        initialized_systems = set()
        
        for agent in agents:
            result = None
            agent_name = agent.__class__.__name__
            
            try:
                # Check if agent has already been initialized in this context
                if hasattr(agent, '_initialized') and agent._initialized:
                    setup_results.append({
                        "agent": agent_name,
                        "status": "already_initialized",
                        "result": {"success": True}
                    })
                    continue
                
                # Try different initialization methods based on agent type
                if hasattr(agent, "initialize_systems"):
                    result = await agent.initialize_systems(context)
                elif hasattr(agent, "initialize"):
                    result = await agent.initialize()
                elif hasattr(agent, "setup"):
                    result = await agent.setup(context)
                elif hasattr(agent, "handle_directive"):
                    # Use directive-based initialization
                    result = await agent.handle_directive({
                        "type": DirectiveType.ACTION,
                        "data": {
                            "action": "initialize",
                            "context": context,
                            "systems": list(initialized_systems)
                        }
                    })
                
                if result:
                    setup_results.append({
                        "agent": agent_name,
                        "status": "initialized",
                        "result": result
                    })
                    
                    # Track initialized systems
                    if isinstance(result, dict) and "initialized_systems" in result:
                        initialized_systems.update(result["initialized_systems"])
                        
                    # Mark agent as initialized
                    if hasattr(agent, '_initialized'):
                        agent._initialized = True
                else:
                    setup_results.append({
                        "agent": agent_name,
                        "status": "no_initialization_needed",
                        "reason": "Agent requires no setup"
                    })
                    
            except Exception as e:
                logger.error(f"Error initializing {agent_name}: {e}")
                setup_results.append({
                    "agent": agent_name,
                    "status": "error",
                    "error": str(e)
                })
        
        return {
            "status": "completed",
            "results": setup_results,
            "initialized_count": sum(1 for r in setup_results if r["status"] == "initialized"),
            "initialized_systems": list(initialized_systems)
        }
    
    async def _execute_planning_task(self, agents: List[Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute planning task with planning-capable agents."""
        plans = []
        planning_metadata = {
            "constraints": [],
            "opportunities": [],
            "risks": []
        }
        
        for agent in agents:
            agent_name = agent.__class__.__name__
            plan = None
            
            try:
                # Check agent capabilities
                capabilities = self.agent_capabilities.get(
                    agent.__class__.__name__, 
                    getattr(agent, 'capabilities', [])
                )
                
                if "planning" in capabilities or "strategy_development" in capabilities:
                    if hasattr(agent, "generate_plan"):
                        plan = await agent.generate_plan(context)
                    elif hasattr(agent, "create_strategy"):
                        plan = await agent.create_strategy(context)
                    elif hasattr(agent, "plan_execution"):
                        plan = await agent.plan_execution(context)
                    elif hasattr(agent, "handle_directive"):
                        # Use directive-based planning
                        result = await agent.handle_directive({
                            "type": DirectiveType.ACTION,
                            "data": {
                                "action": "generate_plan",
                                "context": context,
                                "existing_plans": plans
                            }
                        })
                        if result and result.get("success"):
                            plan = result.get("plan")
                    
                    if plan:
                        plans.append({
                            "agent": agent_name,
                            "plan": plan,
                            "priority": getattr(agent, 'planning_priority', 5)
                        })
                        
                        # Extract metadata
                        if isinstance(plan, dict):
                            planning_metadata["constraints"].extend(
                                plan.get("constraints", [])
                            )
                            planning_metadata["opportunities"].extend(
                                plan.get("opportunities", [])
                            )
                            planning_metadata["risks"].extend(
                                plan.get("risks", [])
                            )
                            
            except Exception as e:
                logger.error(f"Error getting plan from {agent_name}: {e}")
        
        # Merge plans intelligently
        merged_plan = await self._merge_plans(plans, context)
        merged_plan["metadata"] = planning_metadata
        
        return {
            "status": "completed",
            "merged_plan": merged_plan,
            "plan_count": len(plans),
            "contributing_agents": [p["agent"] for p in plans]
        }

    async def _merge_plans(self, plans: List[Dict[str, Any]], context: Dict[str, Any] = None) -> Dict[str, Any]:
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
        # Base limits that can be configured
        base_limits = {
            "memory": 1000,      # MB
            "computation": 100,   # CPU seconds
            "time": 3600,        # seconds
            "narrative_tokens": 10000,  # story elements
            "agent_actions": 50,        # per phase
            "coordination_attempts": 20,  # per goal
            "influence": 100,     # influence points
            "money": 10000,      # currency
            "supplies": 1000     # generic supplies
        }
        
        # Get actual limits from database if available
        try:
            # This would need async, but for now we'll use cached values
            if hasattr(self, '_resource_limits_cache'):
                base_limits.update(self._resource_limits_cache)
        except:
            pass
        
        # Dynamic adjustments based on game state
        if hasattr(self, 'game_state') and self.game_state:
            # Adjust based on number of active entities
            active_npcs = len(self.game_state.get('current_npcs', []))
            active_quests = len(self.game_state.get('active_quests', []))
            active_conflicts = len(getattr(self, 'active_conflicts', []))
            
            # Calculate complexity factor
            complexity_factor = 1.0
            complexity_factor += (active_npcs * 0.05)  # 5% per NPC
            complexity_factor += (active_quests * 0.1)  # 10% per quest
            complexity_factor += (active_conflicts * 0.15)  # 15% per conflict
            
            # Apply different scaling based on resource type
            if resource in ["memory", "computation"]:
                # Scale up for processing resources
                return base_limits.get(resource, 100) * complexity_factor
            elif resource == "time":
                # Less time with more complexity to maintain pacing
                return base_limits.get(resource, 3600) / (1.0 + (complexity_factor - 1.0) * 0.5)
            elif resource in ["agent_actions", "coordination_attempts"]:
                # More actions allowed with complexity
                return base_limits.get(resource, 50) * (1.0 + (complexity_factor - 1.0) * 0.3)
            elif resource == "narrative_tokens":
                # Narrative tokens scale with story complexity
                narrative_stage = self.game_state.get('narrative_state', {}).get('stage', 'beginning')
                stage_multipliers = {
                    'beginning': 0.8,
                    'rising_action': 1.0,
                    'climax': 1.5,
                    'falling_action': 1.2,
                    'resolution': 0.9
                }
                stage_mult = stage_multipliers.get(narrative_stage, 1.0)
                return base_limits.get(resource, 10000) * complexity_factor * stage_mult
        
        return base_limits.get(resource, float("inf"))
    
    async def _load_resource_limits(self):
        """Load resource limits from configuration or database."""
        try:
            async with get_db_connection_context() as conn:
                # Load system configuration
                config_rows = await conn.fetch("""
                    SELECT config_key, config_value
                    FROM SystemConfig
                    WHERE category = 'resource_limits'
                """)
                
                self._resource_limits_cache = {}
                for row in config_rows:
                    try:
                        self._resource_limits_cache[row['config_key']] = float(row['config_value'])
                    except:
                        pass
                        
        except Exception as e:
            logger.warning(f"Could not load resource limits: {e}")
            self._resource_limits_cache = {}

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
