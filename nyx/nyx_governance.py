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
from db.connection import get_db_connection_context

# Memory system references
from memory.wrapper import MemorySystem
from memory.memory_nyx_integration.py import MemoryNyxBridge

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
        """Initialize the enhanced unified governance system."""
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Core systems
        self.memory_system = None
        self.game_state = None
        self.registered_agents = {}
        
        # Agentic state
        self.active_goals = []
        self.agent_goals = {}
        self.agent_performance = {}
        self.agent_learning = {}
        self.coordination_history = []
        
        # Learning state
        self.strategy_effectiveness = {}
        self.adaptation_patterns = {}
        self.collaboration_success = {}
        
        # Disagreement state
        self.disagreement_history = {}
        self.disagreement_thresholds = {
            "narrative_impact": 0.7,
            "character_consistency": 0.8,
            "world_integrity": 0.9,
            "player_experience": 0.6
        }
        
        # Initialize systems
        self._initialize_systems()
    
    async def _initialize_systems(self):
        """Initialize core systems and load initial state."""
        # Initialize memory system
        self.memory_system = await NyxMemorySystem.get_instance(
            self.user_id,
            self.conversation_id
        )
        
        # Initialize game state
        self.game_state = await self.initialize_game_state()
        
        # Discover and register agents
        await self.discover_and_register_agents()
        
        # Load initial state
        await self._load_initial_state()
    
    async def _load_initial_state(self):
        """Load initial state and goals."""
        # Load active goals
        self.active_goals = await self.memory_system.get_active_goals()
        
        # Load agent-specific goals
        for agent_type, agent in self.registered_agents.items():
            self.agent_goals[agent_type] = await agent.get_active_goals()
        
        # Load performance metrics
        await self._update_performance_metrics()
        
        # Load learning state
        await self._load_learning_state()
    
    async def _update_performance_metrics(self):
        """Update performance tracking for all agents."""
        for agent_type, agent in self.registered_agents.items():
            metrics = await agent.get_performance_metrics()
            self.agent_performance[agent_type] = metrics
            
            # Update strategy effectiveness
            for strategy, data in metrics.get("strategies", {}).items():
                if strategy not in self.strategy_effectiveness:
                    self.strategy_effectiveness[strategy] = {
                        "success": 0,
                        "total": 0,
                        "agents": {}
                    }
                self.strategy_effectiveness[strategy]["success"] += data["success"]
                self.strategy_effectiveness[strategy]["total"] += data["total"]
                self.strategy_effectiveness[strategy]["agents"][agent_type] = data
    
    async def _load_learning_state(self):
        """Load and analyze learning state across agents."""
        for agent_type, agent in self.registered_agents.items():
            learning_data = await agent.get_learning_state()
            self.agent_learning[agent_type] = learning_data
            
            # Update adaptation patterns
            for pattern, data in learning_data.get("patterns", {}).items():
                if pattern not in self.adaptation_patterns:
                    self.adaptation_patterns[pattern] = {
                        "success": 0,
                        "total": 0,
                        "agents": {}
                    }
                self.adaptation_patterns[pattern]["success"] += data["success"]
                self.adaptation_patterns[pattern]["total"] += data["total"]
                self.adaptation_patterns[pattern]["agents"][agent_type] = data
    
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
    
    async def _identify_relevant_agents(
        self,
        requirements: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify agents relevant to goal requirements."""
        relevant_agents = []
        
        for agent_type, agent in self.registered_agents.items():
            # Check if agent type is required
            if agent_type in requirements["agent_types"]:
                relevant_agents.append({
                    "type": agent_type,
                    "instance": agent,
                    "relevance": 1.0
                })
                continue
            
            # Check capabilities
            capabilities = await agent.get_capabilities()
            capability_match = sum(
                1 for cap in requirements["capabilities"]
                if cap in capabilities
            ) / len(requirements["capabilities"]) if requirements["capabilities"] else 0
            
            if capability_match > 0.5:
                relevant_agents.append({
                    "type": agent_type,
                    "instance": agent,
                    "relevance": capability_match
                })
        
        # Sort by relevance
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
    
    async def _execute_task(
        self,
        task: Dict[str, Any],
        phase: Dict[str, Any],
        plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single task within a phase."""
        result = {
            "task": task["type"],
            "success": False,
            "output": None,
            "metrics": {}
        }
        
        try:
            # Get assigned agents
            agents = [
                self.registered_agents[agent_type]
                for agent_type in task["agents"]
            ]
            
            # Execute task with agents
            task_output = await self._execute_with_agents(
                task,
                agents,
                phase,
                plan
            )
            
            result["output"] = task_output
            result["success"] = True
            
            # Calculate task metrics
            result["metrics"] = await self._calculate_task_metrics(
                task,
                task_output,
                agents
            )
            
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
        
        # Store in memory system
        await self.memory_system.add_coordination_record(coordination_record)
    
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
        Enhanced permission checking with disagreement handling.
        """
        # First check basic permissions
        basic_permission = await super().check_action_permission(
            agent_type,
            agent_id,
            action_type,
            action_details,
            context
        )
        
        if not basic_permission["approved"]:
            return basic_permission
        
        # If this is a player action, check for disagreements
        if agent_type == "player":
            disagreement = await self.handle_player_disagreement(
                action_details.get("user_id"),
                action_details.get("conversation_id"),
                action_type,
                action_details,
                context
            )
            
            if disagreement["disagrees"]:
                return {
                    "approved": False,
                    "reasoning": disagreement["reasoning"],
                    "suggested_alternative": disagreement.get("suggested_alternative"),
                    "impact_analysis": disagreement["impact_analysis"],
                    "narrative_context": disagreement.get("narrative_context")
                }
        
        return basic_permission

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
