# mcp_orchestrator.py

import asyncio
import datetime
import json
import logging
import os
import re
import socket
import threading
import time
import uuid
import importlib
import inspect
import signal
import weakref
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Set, Tuple, Callable

import aiohttp
import asyncio_extras
from pydantic import BaseModel, Field, root_validator

logger = logging.getLogger(__name__)

class TaskPurpose(str, Enum):
    """Enumeration of possible task purposes"""
    ANALYZE = "analyze"
    WRITE = "write"
    DEPLOY = "deploy"
    SEARCH = "search"
    COMMUNICATE = "communicate"
    DATABASE = "database"
    FILE_MANIPULATION = "file_manipulation"
    CODE = "code"
    VISUALIZATION = "visualization"
    OTHER = "other"

class TaskRequirement(BaseModel):
    """Model representing a task's requirements"""
    purpose: TaskPurpose = Field(..., description="The primary purpose of the task")
    tools_required: List[str] = Field(default_factory=list, description="MCP tools required for this task")
    complexity: int = Field(default=1, description="Task complexity from 1-5")
    priority: int = Field(default=2, description="Task priority from 1-5")
    estimated_time: int = Field(default=60, description="Estimated time in seconds")
    
    @root_validator
    def check_constraints(cls, values):
        """Validate constraints on fields"""
        if values.get("complexity") < 1 or values.get("complexity") > 5:
            values["complexity"] = max(1, min(values.get("complexity", 1), 5))
        
        if values.get("priority") < 1 or values.get("priority") > 5:
            values["priority"] = max(1, min(values.get("priority", 2), 5))
            
        if values.get("estimated_time") < 0:
            values["estimated_time"] = 60
            
        return values

class MCPTool(BaseModel):
    """Model representing an MCP tool/server"""
    name: str = Field(..., description="Name of the MCP tool")
    endpoint: str = Field(..., description="Endpoint URL for the MCP tool")
    description: str = Field(..., description="Description of what the tool does")
    capabilities: List[str] = Field(default_factory=list, description="List of capabilities this tool provides")
    version: str = Field(default="1.0.0", description="Tool version")
    requires_auth: bool = Field(default=False, description="Whether the tool requires authentication")
    auth_type: Optional[str] = Field(default=None, description="Type of authentication required")
    auth_credentials: Optional[Dict[str, Any]] = Field(default=None, description="Authentication credentials")
    cached_status: Optional[Dict[str, Any]] = Field(default=None, description="Cached status information")
    health_check_path: Optional[str] = Field(default="/health", description="Path for health checks")
    last_check_time: Optional[str] = Field(default=None, description="Last health check time")
    status: str = Field(default="unknown", description="Current status of the tool")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with secrets masked"""
        result = self.dict(exclude={'auth_credentials'})
        if self.auth_credentials:
            result['auth_credentials'] = {k: '***' for k in self.auth_credentials.keys()}
        return result
    
    def as_agent_tool(self) -> Dict[str, Any]:
        """Format as a tool configuration for an agent"""
        return {
            "name": self.name,
            "description": self.description,
            "endpoint": self.endpoint
        }

class Agent(BaseModel):
    """Model representing an agent with capabilities"""
    name: str = Field(..., description="Name of the agent")
    description: str = Field(..., description="Description of the agent's purpose")
    capabilities: List[str] = Field(default_factory=list, description="List of agent's capabilities")
    specializations: List[TaskPurpose] = Field(default_factory=list, description="Task purposes this agent specializes in")
    assigned_tools: List[str] = Field(default_factory=list, description="MCP tools assigned to this agent")
    success_rate: float = Field(default=0.0, description="Success rate for tasks (0.0-1.0)")
    total_tasks: int = Field(default=0, description="Total number of tasks handled")
    avg_response_time: float = Field(default=0.0, description="Average response time in seconds")
    max_token_limit: Optional[int] = Field(default=None, description="Maximum token limit for this agent")
    timeout: int = Field(default=60, description="Timeout in seconds")
    system_prompt: Optional[str] = Field(default=None, description="System prompt for the agent")
    temperature: float = Field(default=0.7, description="Temperature setting (0.0-1.0)")
    last_active: Optional[str] = Field(default=None, description="Last time this agent was active")
    access_level: int = Field(default=1, description="Access level (1-5)")
    is_active: bool = Field(default=True, description="Whether this agent is active")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = self.dict()
        result["specializations"] = [s.value if isinstance(s, TaskPurpose) else s for s in self.specializations]
        return result
    
    def record_task_result(self, success: bool, response_time: float):
        """Update agent metrics based on task result"""
        self.total_tasks += 1
        
        # Update success rate using weighted average
        if self.total_tasks == 1:
            self.success_rate = 1.0 if success else 0.0
        else:
            success_value = 1.0 if success else 0.0
            self.success_rate = ((self.success_rate * (self.total_tasks - 1)) + success_value) / self.total_tasks
        
        # Update average response time
        if self.total_tasks == 1:
            self.avg_response_time = response_time
        else:
            self.avg_response_time = ((self.avg_response_time * (self.total_tasks - 1)) + response_time) / self.total_tasks
        
        # Update last active timestamp
        self.last_active = datetime.datetime.now().isoformat()

class TaskResult(BaseModel):
    """Model representing the result of a task execution"""
    task_id: str = Field(..., description="Unique ID for the task")
    agent_name: str = Field(..., description="Name of the agent that handled the task")
    success: bool = Field(..., description="Whether the task was successful")
    start_time: str = Field(..., description="Start time of the task")
    end_time: str = Field(..., description="End time of the task")
    duration: float = Field(..., description="Duration in seconds")
    tools_used: List[str] = Field(default_factory=list, description="MCP tools used for this task")
    error: Optional[str] = Field(default=None, description="Error if any")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return self.dict()

class TaskClassifier:
    """Classifies tasks based on natural language input"""
    
    def __init__(self):
        """Initialize the task classifier"""
        # Regex patterns for tool detection
        self.tool_patterns = {
            "filesystem": r"file|directory|folder|path|storage|document",
            "browser": r"browser|web|website|url|chrome|firefox|safari|internet",
            "database": r"database|sql|query|mongodb|postgresql|mysql|sqlite|db",
            "api": r"api|endpoint|request|http|json|rest",
            "terminal": r"terminal|shell|bash|command line|cli|cmd",
            "slack": r"slack|message|channel|chat",
            "email": r"email|gmail|outlook|smtp|mail",
            "google_sheets": r"google sheets|spreadsheet|excel|csv",
            "calendar": r"calendar|schedule|appointment|meeting|event",
            "code": r"code|program|script|function|class|method|repository|github",
            "kubernetes": r"kubernetes|k8s|pod|container|deployment|cluster",
            "aws": r"aws|amazon web services|s3|ec2|lambda|cloudformation",
            "gcp": r"gcp|google cloud|gke|bigquery",
            "azure": r"azure|microsoft cloud",
            "docker": r"docker|container|image",
            "visualization": r"visualization|chart|graph|plot|dashboard"
        }
        
        # Regex patterns for purpose detection
        self.purpose_patterns = {
            TaskPurpose.ANALYZE: r"analyze|assess|evaluate|examine|study|review|understand|investigate",
            TaskPurpose.WRITE: r"write|create|compose|draft|author|generate|document",
            TaskPurpose.DEPLOY: r"deploy|publish|release|launch|ship|distribute|roll out",
            TaskPurpose.SEARCH: r"search|find|look up|locate|discover|seek|query",
            TaskPurpose.COMMUNICATE: r"communicate|message|email|slack|notify|inform|share|contact",
            TaskPurpose.DATABASE: r"database|query|sql|table|record|schema|store|fetch",
            TaskPurpose.FILE_MANIPULATION: r"file|save|read|write|upload|download|copy|move|delete",
            TaskPurpose.CODE: r"code|program|script|develop|implement|function|class|method",
            TaskPurpose.VISUALIZATION: r"visualize|chart|graph|plot|diagram|dashboard"
        }
    
    async def classify(self, input_text: str) -> TaskRequirement:
        """
        Classify the task based on natural language input
        
        Args:
            input_text: The natural language input to classify
            
        Returns:
            TaskRequirement object with classification results
        """
        input_text = input_text.lower()
        
        # Detect purpose
        purpose = TaskPurpose.OTHER
        max_matches = 0
        
        for p, pattern in self.purpose_patterns.items():
            matches = len(re.findall(pattern, input_text))
            if matches > max_matches:
                max_matches = matches
                purpose = p
        
        # Detect tools required
        tools_required = []
        for tool, pattern in self.tool_patterns.items():
            if re.search(pattern, input_text):
                tools_required.append(f"{tool}_mcp")
        
        # Estimate complexity (1-5)
        complexity = 1
        complexity_signals = {
            1: ["simple", "quick", "basic", "easy", "straightforward"],
            2: ["moderate", "normal", "standard"],
            3: ["complex", "detailed", "thorough", "comprehensive"],
            4: ["intricate", "advanced", "sophisticated"],
            5: ["extremely complex", "highly sophisticated", "expert level"]
        }
        
        for level, signals in complexity_signals.items():
            if any(signal in input_text for signal in signals):
                complexity = level
                break
        
        # Estimate time based on complexity
        time_estimates = {
            1: 30,    # 30 seconds
            2: 60,    # 1 minute
            3: 300,   # 5 minutes
            4: 600,   # 10 minutes
            5: 1800   # 30 minutes
        }
        
        # Detect priority (1-5)
        priority = 2  # Default is medium priority
        priority_signals = {
            5: ["urgent", "immediate", "critical", "emergency", "asap", "right now"],
            4: ["high priority", "important", "vital", "essential"],
            3: ["medium priority", "moderate priority"],
            2: ["low priority", "when you can", "no rush", "take your time"],
            1: ["lowest priority", "whenever", "background task"]
        }
        
        for level, signals in priority_signals.items():
            if any(signal in input_text for signal in signals):
                priority = level
                break
        
        return TaskRequirement(
            purpose=purpose,
            tools_required=tools_required,
            complexity=complexity,
            priority=priority,
            estimated_time=time_estimates.get(complexity, 60)
        )

class AgentRegistry:
    """Registry for managing agents and their capabilities"""
    
    def __init__(self):
        """Initialize the agent registry"""
        self.agents: Dict[str, Agent] = {}
        self.disabled_agents: Dict[str, Agent] = {}
        self.known_capabilities: Set[str] = set()
        self.registry_file = "agent_registry.json"
        self.load_from_file()
    
    def load_from_file(self) -> bool:
        """Load the registry from a file"""
        try:
            if os.path.exists(self.registry_file):
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                
                # Load agents
                for agent_data in data.get("agents", []):
                    agent = Agent(**agent_data)
                    self.agents[agent.name] = agent
                
                # Load disabled agents
                for agent_data in data.get("disabled_agents", []):
                    agent = Agent(**agent_data)
                    self.disabled_agents[agent.name] = agent
                
                # Load known capabilities
                self.known_capabilities.update(data.get("known_capabilities", []))
                
                logger.info(f"Loaded {len(self.agents)} agents from registry file")
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading agent registry: {str(e)}")
            return False
    
    def save_to_file(self) -> bool:
        """Save the registry to a file"""
        try:
            data = {
                "agents": [agent.to_dict() for agent in self.agents.values()],
                "disabled_agents": [agent.to_dict() for agent in self.disabled_agents.values()],
                "known_capabilities": list(self.known_capabilities),
                "last_updated": datetime.datetime.now().isoformat()
            }
            
            with open(self.registry_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(self.agents)} agents to registry file")
            return True
        except Exception as e:
            logger.error(f"Error saving agent registry: {str(e)}")
            return False
    
    def register_agent(self, agent: Agent) -> bool:
        """Register a new agent"""
        if agent.name in self.agents:
            logger.warning(f"Agent {agent.name} already exists, updating")
        
        self.agents[agent.name] = agent
        
        # Update known capabilities
        self.known_capabilities.update(agent.capabilities)
        
        # Save to file
        self.save_to_file()
        
        return True
    
    def unregister_agent(self, agent_name: str) -> bool:
        """Unregister an agent"""
        if agent_name in self.agents:
            # Move to disabled agents
            self.disabled_agents[agent_name] = self.agents[agent_name]
            del self.agents[agent_name]
            
            # Save to file
            self.save_to_file()
            
            return True
        
        return False
    
    def get_agent(self, agent_name: str) -> Optional[Agent]:
        """Get an agent by name"""
        return self.agents.get(agent_name)
    
    def find_agents_by_capability(self, capability: str) -> List[Agent]:
        """Find agents with a specific capability"""
        return [agent for agent in self.agents.values() if capability in agent.capabilities]
    
    def find_agents_by_purpose(self, purpose: TaskPurpose) -> List[Agent]:
        """Find agents specialized for a specific purpose"""
        return [agent for agent in self.agents.values() 
                if purpose in agent.specializations and agent.is_active]
    
    def find_best_agent_for_task(self, task: TaskRequirement) -> Optional[Agent]:
        """Find the best agent for a specific task"""
        # First, filter agents by required purpose
        candidates = self.find_agents_by_purpose(task.purpose)
        
        if not candidates:
            # Fall back to any active agent if no specialized agents are found
            candidates = [agent for agent in self.agents.values() if agent.is_active]
        
        if not candidates:
            return None
        
        # Score candidates based on multiple factors
        scored_candidates = []
        for agent in candidates:
            # Base score starts with success rate (0-1)
            score = agent.success_rate
            
            # Boost score for every matching capability or assigned tool
            for tool in task.tools_required:
                if tool in agent.assigned_tools:
                    score += 0.2
            
            # Penalize agents with very high response times for high-priority tasks
            if task.priority >= 4 and agent.avg_response_time > 5.0:
                score -= 0.2
            
            # Penalize agents that are likely to timeout based on estimated time
            if agent.timeout < task.estimated_time:
                score -= 0.5
            
            scored_candidates.append((agent, score))
        
        # Sort by score descending
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Return highest-scored agent
        if scored_candidates:
            return scored_candidates[0][0]
        
        return None
    
    def update_agent_metrics(self, agent_name: str, success: bool, response_time: float) -> bool:
        """Update agent metrics based on task result"""
        agent = self.get_agent(agent_name)
        if agent:
            agent.record_task_result(success, response_time)
            self.save_to_file()
            return True
        
        return False
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get statistics about all agents"""
        stats = {
            "total_agents": len(self.agents),
            "active_agents": sum(1 for a in self.agents.values() if a.is_active),
            "disabled_agents": len(self.disabled_agents),
            "total_tasks_processed": sum(a.total_tasks for a in self.agents.values()),
            "average_success_rate": 0.0,
            "average_response_time": 0.0,
            "capabilities_distribution": {}
        }
        
        # Calculate average success rate and response time
        active_agents = [a for a in self.agents.values() if a.is_active and a.total_tasks > 0]
        if active_agents:
            stats["average_success_rate"] = sum(a.success_rate for a in active_agents) / len(active_agents)
            stats["average_response_time"] = sum(a.avg_response_time for a in active_agents) / len(active_agents)
        
        # Calculate capabilities distribution
        for capability in self.known_capabilities:
            stats["capabilities_distribution"][capability] = len(self.find_agents_by_capability(capability))
        
        # Get top agents by success rate
        top_agents = sorted(
            [a for a in self.agents.values() if a.total_tasks > 5],
            key=lambda x: x.success_rate,
            reverse=True
        )[:5]
        
        stats["top_agents"] = [
            {"name": a.name, "success_rate": a.success_rate, "tasks": a.total_tasks}
            for a in top_agents
        ]
        
        return stats

class MCPToolRegistry:
    """Registry for managing MCP tools"""
    
    def __init__(self):
        """Initialize the MCP tool registry"""
        self.tools: Dict[str, MCPTool] = {}
        self.registry_file = "mcp_tools_registry.json"
        self.last_health_check = None
        self.load_from_file()
    
    def load_from_file(self) -> bool:
        """Load the registry from a file"""
        try:
            if os.path.exists(self.registry_file):
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                
                # Load tools
                for tool_data in data.get("tools", []):
                    tool = MCPTool(**tool_data)
                    self.tools[tool.name] = tool
                
                logger.info(f"Loaded {len(self.tools)} MCP tools from registry file")
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading MCP tool registry: {str(e)}")
            return False
    
    def save_to_file(self) -> bool:
        """Save the registry to a file"""
        try:
            # Convert to dict but mask auth credentials
            tool_dicts = [tool.to_dict() for tool in self.tools.values()]
            
            data = {
                "tools": tool_dicts,
                "last_updated": datetime.datetime.now().isoformat()
            }
            
            with open(self.registry_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(self.tools)} MCP tools to registry file")
            return True
        except Exception as e:
            logger.error(f"Error saving MCP tool registry: {str(e)}")
            return False
    
    def register_tool(self, tool: MCPTool) -> bool:
        """Register a new MCP tool"""
        if tool.name in self.tools:
            logger.warning(f"MCP tool {tool.name} already exists, updating")
        
        self.tools[tool.name] = tool
        
        # Save to file
        self.save_to_file()
        
        return True
    
    def unregister_tool(self, tool_name: str) -> bool:
        """Unregister an MCP tool"""
        if tool_name in self.tools:
            del self.tools[tool_name]
            
            # Save to file
            self.save_to_file()
            
            return True
        
        return False
    
    def get_tool(self, tool_name: str) -> Optional[MCPTool]:
        """Get a tool by name"""
        return self.tools.get(tool_name)
    
    def find_tools_by_capability(self, capability: str) -> List[MCPTool]:
        """Find tools with a specific capability"""
        return [tool for tool in self.tools.values() if capability in tool.capabilities]
    
    async def check_tool_health(self, tool_name: str) -> Dict[str, Any]:
        """Check the health of a specific tool"""
        tool = self.get_tool(tool_name)
        if not tool:
            return {"error": f"Tool {tool_name} not found"}
        
        try:
            # Create health check URL
            health_url = tool.endpoint
            if tool.health_check_path:
                if not health_url.endswith('/') and not tool.health_check_path.startswith('/'):
                    health_url += '/'
                health_url += tool.health_check_path
            
            # Make request
            async with aiohttp.ClientSession() as session:
                async with session.get(health_url, timeout=5) as response:
                    status_code = response.status
                    try:
                        response_json = await response.json()
                    except:
                        response_json = {"text": await response.text()}
            
            # Update tool status
            if 200 <= status_code < 300:
                tool.status = "healthy"
            else:
                tool.status = "unhealthy"
            
            # Update cached status
            tool.cached_status = response_json
            tool.last_check_time = datetime.datetime.now().isoformat()
            
            # Save to file
            self.save_to_file()
            
            return {
                "status": tool.status,
                "status_code": status_code,
                "response": response_json,
                "check_time": tool.last_check_time
            }
        except Exception as e:
            # Update tool status
            tool.status = "error"
            tool.cached_status = {"error": str(e)}
            tool.last_check_time = datetime.datetime.now().isoformat()
            
            # Save to file
            self.save_to_file()
            
            return {
                "status": "error",
                "error": str(e),
                "check_time": tool.last_check_time
            }
    
    async def check_all_tools_health(self) -> Dict[str, Any]:
        """Check the health of all tools"""
        results = {}
        
        for tool_name in self.tools:
            results[tool_name] = await self.check_tool_health(tool_name)
        
        self.last_health_check = datetime.datetime.now().isoformat()
        
        return {
            "results": results,
            "healthy_count": sum(1 for r in results.values() if r.get("status") == "healthy"),
            "unhealthy_count": sum(1 for r in results.values() if r.get("status") == "unhealthy"),
            "error_count": sum(1 for r in results.values() if r.get("status") == "error"),
            "timestamp": self.last_health_check
        }
    
    def get_tool_statistics(self) -> Dict[str, Any]:
        """Get statistics about all tools"""
        stats = {
            "total_tools": len(self.tools),
            "healthy_tools": sum(1 for t in self.tools.values() if t.status == "healthy"),
            "unhealthy_tools": sum(1 for t in self.tools.values() if t.status == "unhealthy"),
            "unknown_status_tools": sum(1 for t in self.tools.values() if t.status == "unknown"),
            "last_health_check": self.last_health_check,
            "capabilities_count": {}
        }
        
        # Count tools by capability
        all_capabilities = set()
        for tool in self.tools.values():
            all_capabilities.update(tool.capabilities)
        
        for capability in all_capabilities:
            stats["capabilities_count"][capability] = len(self.find_tools_by_capability(capability))
        
        return stats

class ToolInjector:
    """Injects appropriate MCP tools into agent configurations"""
    
    def __init__(self, tool_registry: MCPToolRegistry):
        """
        Initialize the tool injector
        
        Args:
            tool_registry: Registry of available MCP tools
        """
        self.tool_registry = tool_registry
    
    def assemble_agent_toolset(self, 
                             agent: Agent, 
                             task_requirement: TaskRequirement) -> Dict[str, Any]:
        """
        Assemble a runtime toolset for an agent based on task requirements
        
        Args:
            agent: The agent to assemble toolset for
            task_requirement: Requirements for the task
            
        Returns:
            Agent configuration with appropriate tools
        """
        # Start with base configuration
        agent_config = {
            "name": agent.name,
            "system_prompt": agent.system_prompt or f"You are {agent.name}, an expert assistant.",
            "tools": [],
            "temperature": agent.temperature,
            "timeout": agent.timeout,
            "max_tokens": agent.max_token_limit
        }
        
        # Add assigned tools that are healthy
        for tool_name in agent.assigned_tools:
            tool = self.tool_registry.get_tool(tool_name)
            if tool and tool.status == "healthy":
                agent_config["tools"].append(tool.as_agent_tool())
        
        # Add required tools for this task that aren't already assigned
        for tool_name in task_requirement.tools_required:
            if tool_name not in agent.assigned_tools:
                tool = self.tool_registry.get_tool(tool_name)
                if tool and tool.status == "healthy":
                    agent_config["tools"].append(tool.as_agent_tool())
        
        # Adjust timeout based on task estimated time
        agent_config["timeout"] = max(agent.timeout, task_requirement.estimated_time * 1.5)
        
        return agent_config
    
    def merge_agent_tool_permissions(self, 
                                  agent: Agent, 
                                  tools: List[str], 
                                  temporary: bool = True) -> Dict[str, Any]:
        """
        Merge new tools into an agent's permissions
        
        Args:
            agent: The agent to update
            tools: List of tool names to add
            temporary: Whether this is a temporary assignment
            
        Returns:
            Updated agent configuration
        """
        # Start with current assigned tools
        current_tools = set(agent.assigned_tools)
        
        # Add new tools
        for tool_name in tools:
            if tool_name not in current_tools and self.tool_registry.get_tool(tool_name):
                current_tools.add(tool_name)
        
        if not temporary:
            # Update agent permanently
            agent.assigned_tools = list(current_tools)
        
        # Return updated configuration
        return {
            "name": agent.name,
            "assigned_tools": list(current_tools)
        }

class FeedbackLoop:
    """Monitors agent performance and handles escalation"""
    
    def __init__(self, agent_registry: AgentRegistry):
        """
        Initialize the feedback loop
        
        Args:
            agent_registry: Registry of available agents
        """
        self.agent_registry = agent_registry
        self.task_history: List[TaskResult] = []
        self.max_history_size = 1000
        self.failure_thresholds = {
            "high_priority_failure": 2,  # Consecutive failures on high priority tasks
            "general_failure": 5,        # Consecutive failures overall
            "success_rate_minimum": 0.5  # Minimum acceptable success rate
        }
        self.escalation_rules = {
            "max_escalation_attempts": 3,
            "escalation_timeout": 300  # 5 minutes
        }
        self.current_escalations: Dict[str, Dict[str, Any]] = {}  # task_id -> escalation info
    
    def record_task_result(self, result: TaskResult) -> None:
        """
        Record the result of a task
        
        Args:
            result: The task result to record
        """
        # Add to history
        self.task_history.append(result)
        
        # Trim history if needed
        if len(self.task_history) > self.max_history_size:
            self.task_history = self.task_history[-self.max_history_size:]
        
        # Update agent metrics
        self.agent_registry.update_agent_metrics(
            agent_name=result.agent_name,
            success=result.success,
            response_time=result.duration
        )
        
        # Check for failure patterns
        self._check_failure_patterns(result)
    
    def _check_failure_patterns(self, result: TaskResult) -> None:
        """
        Check for failure patterns that may require intervention
        
        Args:
            result: The task result to check
        """
        if not result.success:
            # Check agent's recent performance
            agent = self.agent_registry.get_agent(result.agent_name)
            if not agent:
                return
            
            # Check for consecutive failures
            recent_tasks = [t for t in self.task_history if t.agent_name == result.agent_name][-5:]
            consecutive_failures = 0
            
            for task in reversed(recent_tasks):
                if not task.success:
                    consecutive_failures += 1
                else:
                    break
            
            # Check escalation conditions
            should_escalate = False
            
            # High priority failure
            if result.metadata.get("priority", 0) >= 4 and consecutive_failures >= self.failure_thresholds["high_priority_failure"]:
                should_escalate = True
            
            # General failure threshold
            if consecutive_failures >= self.failure_thresholds["general_failure"]:
                should_escalate = True
            
            # Success rate too low
            if agent.total_tasks >= 10 and agent.success_rate < self.failure_thresholds["success_rate_minimum"]:
                should_escalate = True
            
            if should_escalate:
                self._initiate_escalation(result)
    
    def _initiate_escalation(self, result: TaskResult) -> None:
        """
        Initiate the escalation process for a failed task
        
        Args:
            result: The failed task result
        """
        # Check if already escalated
        if result.task_id in self.current_escalations:
            # Update existing escalation
            escalation = self.current_escalations[result.task_id]
            escalation["attempts"] += 1
            
            # Check if maximum attempts reached
            if escalation["attempts"] >= self.escalation_rules["max_escalation_attempts"]:
                # Remove from active escalations
                del self.current_escalations[result.task_id]
                
                # Consider disabling the agent if performance is very poor
                agent = self.agent_registry.get_agent(result.agent_name)
                if agent and agent.total_tasks >= 20 and agent.success_rate < 0.3:
                    logger.warning(f"Considering disabling agent {agent.name} due to poor performance")
                    # This could trigger an admin notification or automatic agent disabling
            
            return
        
        # Create new escalation
        self.current_escalations[result.task_id] = {
            "task_id": result.task_id,
            "original_agent": result.agent_name,
            "start_time": datetime.datetime.now().isoformat(),
            "attempts": 1,
            "status": "active"
        }
        
        logger.info(f"Escalation initiated for task {result.task_id} from agent {result.agent_name}")
    
    def get_escalation_candidates(self, failed_task: TaskResult) -> List[Agent]:
        """
        Get candidate agents for escalation
        
        Args:
            failed_task: The failed task
            
        Returns:
            List of potential agents for escalation
        """
        # Determine task requirements from metadata
        purpose = TaskPurpose(failed_task.metadata.get("purpose", "other"))
        
        # Get all active agents except the one that failed
        candidates = [
            agent for agent in self.agent_registry.agents.values()
            if agent.is_active and agent.name != failed_task.agent_name
        ]
        
        # Prioritize based on purpose and success rate
        scored_candidates = []
        for agent in candidates:
            score = agent.success_rate
            
            # Boost score for matching purpose
            if purpose in agent.specializations:
                score += 0.3
            
            scored_candidates.append((agent, score))
        
        # Sort by score descending
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Return candidate agents
        return [agent for agent, _ in scored_candidates]
    
    def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get statistics about task performance and escalations"""
        stats = {
            "total_tasks": len(self.task_history),
            "successful_tasks": sum(1 for t in self.task_history if t.success),
            "failed_tasks": sum(1 for t in self.task_history if not t.success),
            "current_escalations": len(self.current_escalations),
            "average_duration": 0.0,
            "agent_performance": {}
        }
        
        # Calculate average duration
        if stats["total_tasks"] > 0:
            stats["average_duration"] = sum(t.duration for t in self.task_history) / stats["total_tasks"]
        
        # Calculate per-agent statistics
        for agent_name in self.agent_registry.agents:
            agent_tasks = [t for t in self.task_history if t.agent_name == agent_name]
            
            if agent_tasks:
                stats["agent_performance"][agent_name] = {
                    "tasks": len(agent_tasks),
                    "success_rate": sum(1 for t in agent_tasks if t.success) / len(agent_tasks),
                    "average_duration": sum(t.duration for t in agent_tasks) / len(agent_tasks),
                    "escalations": sum(1 for e in self.current_escalations.values() if e["original_agent"] == agent_name)
                }
        
        # Get recent failures
        stats["recent_failures"] = [
            {
                "task_id": t.task_id,
                "agent": t.agent_name,
                "time": t.end_time,
                "error": t.error
            }
            for t in sorted(
                [t for t in self.task_history if not t.success],
                key=lambda x: x.end_time,
                reverse=True
            )[:5]
        ]
        
        return stats

class MCPOrchestrator:
    """Main orchestrator that coordinates MCP server delegation"""
    
    def __init__(self):
        """Initialize the MCP orchestrator"""
        # Initialize components
        self.task_classifier = TaskClassifier()
        self.agent_registry = AgentRegistry()
        self.tool_registry = MCPToolRegistry()
        self.tool_injector = ToolInjector(self.tool_registry)
        self.feedback_loop = FeedbackLoop(self.agent_registry)
        
        # Task tracking
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.task_counter = 0
        
        # Auto-discovery
        self.discovery_service = AgentDiscoveryService(self)
        self.agent_watcher = None
        
        logger.info("MCP Orchestrator initialized")
    
    async def initialize(self) -> bool:
        """Initialize the orchestrator and check components"""
        # Run health checks on MCP tools
        await self.tool_registry.check_all_tools_health()
        
        # Start agent discovery service
        await self.discovery_service.start()
        
        logger.info("MCP Orchestrator initialization complete")
        return True
    
    async def process_input(self, input_text: str, user_id: str = "default") -> Dict[str, Any]:
        """
        Process natural language input and delegate to appropriate agent
        
        Args:
            input_text: Natural language input to process
            user_id: ID of the user making the request
            
        Returns:
            Processing results including assigned agent and tools
        """
        # Generate task ID
        self.task_counter += 1
        task_id = f"task_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{self.task_counter}"
        
        start_time = datetime.datetime.now()
        
        # 1. Classify the task
        task_requirements = await self.task_classifier.classify(input_text)
        
        # 2. Find the best agent for this task
        agent = self.agent_registry.find_best_agent_for_task(task_requirements)
        
        if not agent:
            return {
                "success": False,
                "error": "No suitable agent found for this task",
                "task_id": task_id,
                "requirements": task_requirements.dict()
            }
        
        # 3. Assemble the toolset for this agent
        agent_config = self.tool_injector.assemble_agent_toolset(agent, task_requirements)
        
        # 4. Track the active task
        self.active_tasks[task_id] = {
            "task_id": task_id,
            "start_time": start_time.isoformat(),
            "agent_name": agent.name,
            "input_text": input_text,
            "user_id": user_id,
            "requirements": task_requirements.dict(),
            "status": "assigned"
        }
        
        return {
            "success": True,
            "task_id": task_id,
            "agent_name": agent.name,
            "agent_config": agent_config,
            "requirements": task_requirements.dict(),
            "tools_assigned": [tool["name"] for tool in agent_config["tools"]]
        }
    
    async def complete_task(self, 
                         task_id: str, 
                         success: bool, 
                         result: Dict[str, Any],
                         error: Optional[str] = None) -> Dict[str, Any]:
        """
        Mark a task as complete and update metrics
        
        Args:
            task_id: ID of the task to complete
            success: Whether the task was successful
            result: Task result data
            error: Error message if any
            
        Returns:
            Updated task information
        """
        if task_id not in self.active_tasks:
            return {
                "success": False,
                "error": f"Task {task_id} not found"
            }
        
        # Get task information
        task_info = self.active_tasks[task_id]
        
        # Calculate duration
        start_time = datetime.datetime.fromisoformat(task_info["start_time"])
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Create task result
        task_result = TaskResult(
            task_id=task_id,
            agent_name=task_info["agent_name"],
            success=success,
            start_time=task_info["start_time"],
            end_time=end_time.isoformat(),
            duration=duration,
            tools_used=result.get("tools_used", []),
            error=error,
            metadata={
                "user_id": task_info["user_id"],
                "purpose": task_info["requirements"]["purpose"],
                "priority": task_info["requirements"]["priority"],
                "complexity": task_info["requirements"]["complexity"]
            }
        )
        
        # Record the result
        self.feedback_loop.record_task_result(task_result)
        
        # Update task status
        task_info["status"] = "completed" if success else "failed"
        task_info["end_time"] = end_time.isoformat()
        task_info["duration"] = duration
        task_info["result"] = result
        
        if error:
            task_info["error"] = error
        
        # Remove from active tasks
        del self.active_tasks[task_id]
        
        return {
            "success": True,
            "task_id": task_id,
            "status": task_info["status"],
            "duration": duration,
            "metrics_updated": True
        }
    
    async def escalate_task(self, task_id: str, reason: str) -> Dict[str, Any]:
        """
        Escalate a task to a different agent
        
        Args:
            task_id: ID of the task to escalate
            reason: Reason for escalation
            
        Returns:
            Escalation results including new agent
        """
        if task_id not in self.active_tasks:
            return {
                "success": False,
                "error": f"Task {task_id} not found"
            }
        
        # Get task information
        task_info = self.active_tasks[task_id]
        
        # Fail the original task
        await self.complete_task(
            task_id=task_id,
            success=False,
            result={"escalated": True},
            error=f"Escalated: {reason}"
        )
        
        # Create an escalation task
        escalation_task_id = f"{task_id}_escalated"
        
        # Find an alternative agent
        # We need to reconstruct the task result for the feedback loop
        original_agent = task_info["agent_name"]
        start_time = datetime.datetime.fromisoformat(task_info["start_time"])
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        temp_task_result = TaskResult(
            task_id=task_id,
            agent_name=original_agent,
            success=False,
            start_time=task_info["start_time"],
            end_time=end_time.isoformat(),
            duration=duration,
            tools_used=[],
            error=reason,
            metadata=task_info["requirements"]
        )
        
        # Get escalation candidates
        candidates = self.feedback_loop.get_escalation_candidates(temp_task_result)
        
        if not candidates:
            return {
                "success": False,
                "error": "No suitable agents found for escalation"
            }
        
        # Select the top candidate
        new_agent = candidates[0]
        
        # Process the input again with the new agent
        task_requirements = TaskRequirement(**task_info["requirements"])
        
        # Assemble a new toolset with potentially more tools
        agent_config = self.tool_injector.assemble_agent_toolset(new_agent, task_requirements)
        
        # Add additional tools for the escalation
        for tool_name, tool in self.tool_registry.tools.items():
            if tool.status == "healthy" and any(cap in task_requirements.tools_required for cap in tool.capabilities):
                if not any(t["name"] == tool_name for t in agent_config["tools"]):
                    agent_config["tools"].append(tool.as_agent_tool())
        
        # Track the escalated task
        self.active_tasks[escalation_task_id] = {
            "task_id": escalation_task_id,
            "start_time": datetime.datetime.now().isoformat(),
            "agent_name": new_agent.name,
            "input_text": task_info["input_text"],
            "user_id": task_info["user_id"],
            "requirements": task_requirements.dict(),
            "status": "escalated",
            "original_task_id": task_id,
            "original_agent": original_agent,
            "escalation_reason": reason
        }
        
        return {
            "success": True,
            "task_id": escalation_task_id,
            "agent_name": new_agent.name,
            "agent_config": agent_config,
            "original_task_id": task_id,
            "requirements": task_requirements.dict(),
            "tools_assigned": [tool["name"] for tool in agent_config["tools"]]
        }
    
    async def register_new_agent(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register a new agent
        
        Args:
            agent_data: Agent configuration data
            
        Returns:
            Registration results
        """
        try:
            # Create agent object
            agent = Agent(**agent_data)
            
            # Register the agent
            success = self.agent_registry.register_agent(agent)
            
            return {
                "success": success,
                "agent_name": agent.name,
                "message": "Agent registered successfully" if success else "Failed to register agent"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def register_new_tool(self, tool_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register a new MCP tool
        
        Args:
            tool_data: Tool configuration data
            
        Returns:
            Registration results
        """
        try:
            # Create tool object
            tool = MCPTool(**tool_data)
            
            # Register the tool
            success = self.tool_registry.register_tool(tool)
            
            # Check the tool's health
            health_result = await self.tool_registry.check_tool_health(tool.name)
            
            return {
                "success": success,
                "tool_name": tool.name,
                "health_status": health_result,
                "message": "Tool registered successfully" if success else "Failed to register tool"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get the overall system status"""
        # Get agent statistics
        agent_stats = self.agent_registry.get_agent_statistics()
        
        # Get tool statistics
        tool_stats = self.tool_registry.get_tool_statistics()
        
        # Get feedback statistics
        feedback_stats = self.feedback_loop.get_feedback_statistics()
        
        # Combine into system status
        status = {
            "agents": agent_stats,
            "tools": tool_stats,
            "tasks": feedback_stats,
            "active_tasks": len(self.active_tasks),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return status
    
    def get_active_task_status(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get status of active tasks
        
        Args:
            task_id: Specific task ID to check, or None for all tasks
            
        Returns:
            Task status information
        """
        if task_id:
            if task_id in self.active_tasks:
                return {
                    "success": True,
                    "task": self.active_tasks[task_id]
                }
            else:
                return {
                    "success": False,
                    "error": f"Task {task_id} not found or no longer active"
                }
        else:
            return {
                "success": True,
                "active_tasks": len(self.active_tasks),
                "tasks": self.active_tasks
            }

class AuditUI:
    """Provides audit and control UI functionality"""
    
    def __init__(self, orchestrator: MCPOrchestrator):
        """
        Initialize the audit UI
        
        Args:
            orchestrator: Reference to the MCP orchestrator
        """
        self.orchestrator = orchestrator
        
        # UI state
        self.web_ui_enabled = False
        self.ui_port = 8080
        self.auth_enabled = True
        self.audit_log = []
        self.max_audit_log_size = 10000
    
    def log_action(self, action_type: str, details: Dict[str, Any]) -> None:
        """Log an action to the audit log"""
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "action_type": action_type,
            "details": details
        }
        
        self.audit_log.append(log_entry)
        
        # Trim log if needed
        if len(self.audit_log) > self.max_audit_log_size:
            self.audit_log = self.audit_log[-self.max_audit_log_size:]
    
    def get_audit_log(self, 
                    action_type: Optional[str] = None, 
                    limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get entries from the audit log
        
        Args:
            action_type: Filter by action type
            limit: Maximum number of entries to return
            
        Returns:
            Filtered audit log entries
        """
        filtered_log = self.audit_log
        
        if action_type:
            filtered_log = [entry for entry in filtered_log if entry["action_type"] == action_type]
        
        # Sort by timestamp descending (newest first)
        sorted_log = sorted(filtered_log, key=lambda x: x["timestamp"], reverse=True)
        
        # Return limited results
        return sorted_log[:limit]
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for the dashboard"""
        system_status = asyncio.run(self.orchestrator.get_system_status())
        
        # Enhance with audit data
        system_status["recent_actions"] = self.get_audit_log(limit=10)
        
        return system_status
    
    async def enable_web_ui(self, port: int = 8080) -> Dict[str, Any]:
        """
        Enable the web UI
        
        Args:
            port: Port to run the web UI on
            
        Returns:
            Status of the web UI
        """
        self.web_ui_enabled = True
        self.ui_port = port
        
        # This would start a web server in a real implementation
        # For now, we'll just return status
        return {
            "web_ui_enabled": self.web_ui_enabled,
            "port": self.ui_port,
            "message": f"Web UI enabled on port {port}"
        }
    
    def disable_web_ui(self) -> Dict[str, Any]:
        """
        Disable the web UI
        
        Returns:
            Status of the web UI
        """
        self.web_ui_enabled = False
        
        return {
            "web_ui_enabled": self.web_ui_enabled,
            "message": "Web UI disabled"
        }
    
    def get_web_ui_status(self) -> Dict[str, Any]:
        """
        Get status of the web UI
        
        Returns:
            Web UI status
        """
        return {
            "web_ui_enabled": self.web_ui_enabled,
            "port": self.ui_port,
            "auth_enabled": self.auth_enabled
        }

# Example usage
async def main():
    """Example usage of the MCP orchestrator"""
    # Create the orchestrator
    orchestrator = MCPOrchestrator()
    
    # Initialize - this will start the agent discovery service
    await orchestrator.initialize()
    
    # Register an example MCP tool
    tool_data = {
        "name": "filesystem_mcp",
        "endpoint": "http://localhost:8000/mcp/filesystem",
        "description": "MCP tool for filesystem operations",
        "capabilities": ["file_read", "file_write", "directory_listing"],
        "version": "1.0.0"
    }
    
    await orchestrator.register_new_tool(tool_data)
    
    # No need to manually register agents! They'll be discovered automatically
    print("Waiting for agent discovery...")
    await asyncio.sleep(5)  # Wait for some agents to be discovered
    
    # Process an example input
    result = await orchestrator.process_input("Can you analyze this document and summarize its contents?")
    
    print("Processing result:", result)
    
    # Complete the task
    complete_result = await orchestrator.complete_task(
        task_id=result["task_id"],
        success=True,
        result={"summary": "This is a sample document summary."}
    )
    
    print("Task completion result:", complete_result)
    
    # Get system status
    status = await orchestrator.get_system_status()
    
    print("System status:", status)

# Agent Discovery Service Classes
class AgentDiscoveryService:
    """Service for automatically discovering and registering agents"""
    
    def __init__(self, orchestrator: 'MCPOrchestrator'):
        """Initialize the agent discovery service"""
        self.orchestrator = orchestrator
        self.running = False
        self.discovery_methods = []
        self.registered_agents = set()
        self.scan_interval = 30  # seconds
        self._discovery_task = None
        
        # Initialize discovery methods
        self._init_discovery_methods()
    
    def _init_discovery_methods(self):
        """Initialize all agent discovery methods"""
        # Module discovery (finds agents in Python modules)
        self.discovery_methods.append(ModuleAgentDiscovery(self))
        
        # Process discovery (finds agents running as separate processes)
        self.discovery_methods.append(ProcessAgentDiscovery(self))
        
        # Network discovery (finds agents running on the network)
        self.discovery_methods.append(NetworkAgentDiscovery(self))
        
        # Registration server (agents can register themselves)
        self.discovery_methods.append(AgentRegistrationServer(self))
        
        # Environment variable discovery
        self.discovery_methods.append(EnvVarAgentDiscovery(self))
        
        # Container discovery (Docker, Kubernetes)
        self.discovery_methods.append(ContainerAgentDiscovery(self))
    
    async def start(self):
        """Start the discovery service"""
        if self.running:
            return
        
        self.running = True
        
        # Start all discovery methods
        for method in self.discovery_methods:
            await method.start()
        
        # Start periodic scanning
        self._discovery_task = asyncio.create_task(self._periodic_scan())
        
        logger.info("Agent discovery service started")
    
    async def stop(self):
        """Stop the discovery service"""
        if not self.running:
            return
        
        self.running = False
        
        # Stop discovery task
        if self._discovery_task:
            self._discovery_task.cancel()
            try:
                await self._discovery_task
            except asyncio.CancelledError:
                pass
        
        # Stop all discovery methods
        for method in self.discovery_methods:
            await method.stop()
        
        logger.info("Agent discovery service stopped")
    
    async def _periodic_scan(self):
        """Periodically scan for new agents"""
        while self.running:
            try:
                await self.scan_for_agents()
                await asyncio.sleep(self.scan_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic agent scan: {str(e)}")
                await asyncio.sleep(10)  # Shorter interval on error
    
    async def scan_for_agents(self):
        """Scan for new agents using all discovery methods"""
        for method in self.discovery_methods:
            try:
                await method.discover_agents()
            except Exception as e:
                logger.error(f"Error in discovery method {method.__class__.__name__}: {str(e)}")
    
    async def register_discovered_agent(self, agent_info: Dict[str, Any]) -> bool:
        """Register a discovered agent"""
        # Generate a unique ID for tracking
        agent_id = agent_info.get("id") or f"{agent_info.get('name', 'agent')}_{uuid.uuid4().hex[:8]}"
        
        # Skip if already registered
        if agent_id in self.registered_agents:
            return False
        
        try:
            # Create agent data for registration
            agent_data = {
                "name": agent_info.get("name", agent_id),
                "description": agent_info.get("description", "Automatically discovered agent"),
                "capabilities": agent_info.get("capabilities", []),
                "specializations": agent_info.get("specializations", []),
                "assigned_tools": agent_info.get("assigned_tools", []),
                "system_prompt": agent_info.get("system_prompt"),
                "is_active": True
            }
            
            # Register with orchestrator
            result = await self.orchestrator.register_new_agent(agent_data)
            
            if result.get("success", False):
                self.registered_agents.add(agent_id)
                logger.info(f"Successfully registered discovered agent: {agent_data['name']}")
                return True
            else:
                logger.warning(f"Failed to register discovered agent: {result.get('error', 'Unknown error')}")
                return False
        
        except Exception as e:
            logger.error(f"Error registering discovered agent: {str(e)}")
            return False

class DiscoveryMethod:
    """Base class for agent discovery methods"""
    
    def __init__(self, discovery_service: AgentDiscoveryService):
        """Initialize the discovery method"""
        self.discovery_service = discovery_service
        self.running = False
    
    async def start(self):
        """Start the discovery method"""
        self.running = True
    
    async def stop(self):
        """Stop the discovery method"""
        self.running = False
    
    async def discover_agents(self) -> List[Dict[str, Any]]:
        """Discover agents - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement discover_agents")

class ModuleAgentDiscovery(DiscoveryMethod):
    """Discovers agents in loaded Python modules"""
    
    async def discover_agents(self) -> List[Dict[str, Any]]:
        """Discover agents in loaded modules"""
        discovered_agents = []
        
        # Get all loaded modules
        for module_name, module in list(sys.modules.items()):
            if not module or not module_name:
                continue
            
            # Skip standard library and third-party modules
            if (module_name.startswith('_') or 
                module_name.startswith('importlib') or
                module_name.startswith('pkg_resources') or
                module_name in ('builtins', 'sys', 'os')):
                continue
            
            try:
                # Look for Agent classes or agent instances
                for name, obj in inspect.getmembers(module):
                    # Check if it's an agent class
                    if (inspect.isclass(obj) and 
                        (name.endswith('Agent') or hasattr(obj, 'is_agent')) and
                        not name.startswith('_')):
                        
                        agent_info = self._extract_agent_info(obj, name, is_class=True)
                        if agent_info:
                            discovered_agents.append(agent_info)
                            await self.discovery_service.register_discovered_agent(agent_info)
                    
                    # Check if it's an agent instance
                    elif (not inspect.isclass(obj) and
                          not inspect.isfunction(obj) and
                          not inspect.ismodule(obj) and
                          (name.endswith('Agent') or 
                           (hasattr(obj, 'is_agent') and obj.is_agent) or
                           (hasattr(obj, 'name') and hasattr(obj, 'description')))):
                        
                        agent_info = self._extract_agent_info(obj, name)
                        if agent_info:
                            discovered_agents.append(agent_info)
                            await self.discovery_service.register_discovered_agent(agent_info)
            
            except Exception as e:
                logger.debug(f"Error inspecting module {module_name}: {str(e)}")
        
        return discovered_agents
    
    def _extract_agent_info(self, obj: Any, name: str, is_class: bool = False) -> Optional[Dict[str, Any]]:
        """Extract agent information from an object"""
        try:
            # For classes, we need to check class attributes
            if is_class:
                # Skip abstract classes or base classes
                if hasattr(obj, '__abstractmethods__') and obj.__abstractmethods__:
                    return None
                
                # Basic agent information
                agent_info = {
                    "name": getattr(obj, 'agent_name', name),
                    "description": getattr(obj, 'description', f"Discovered {name} class"),
                    "id": f"module_{name}_{id(obj)}",
                    "capabilities": getattr(obj, 'capabilities', []),
                    "specializations": getattr(obj, 'specializations', [])
                }
                
                return agent_info
            
            # For instances, we can check instance attributes
            else:
                # Basic agent information
                agent_info = {
                    "name": getattr(obj, 'name', name),
                    "description": getattr(obj, 'description', f"Discovered {name} instance"),
                    "id": f"module_{name}_{id(obj)}",
                    "capabilities": getattr(obj, 'capabilities', []),
                    "specializations": getattr(obj, 'specializations', [])
                }
                
                # Check for additional attributes
                if hasattr(obj, 'system_prompt'):
                    agent_info["system_prompt"] = obj.system_prompt
                
                if hasattr(obj, 'assigned_tools'):
                    agent_info["assigned_tools"] = obj.assigned_tools
                
                return agent_info
        
        except Exception as e:
            logger.debug(f"Error extracting agent info from {name}: {str(e)}")
            return None

class ProcessAgentDiscovery(DiscoveryMethod):
    """Discovers agents running as separate processes"""
    
    def __init__(self, discovery_service: AgentDiscoveryService):
        """Initialize the process discovery method"""
        super().__init__(discovery_service)
        self.agent_process_patterns = [
            r'agent[_-].*\.py',
            r'.*[_-]agent.*\.py',
            r'llm[_-]agent.*',
            r'ai[_-]assistant.*',
            r'.*mcp[_-]server.*'
        ]
    
    async def discover_agents(self) -> List[Dict[str, Any]]:
        """Discover agents running as separate processes"""
        discovered_agents = []
        
        try:
            # This would use psutil or similar to get process information
            # As a placeholder, we'll just generate some sample data
            
            # In a real implementation, this would scan process names, command lines, etc.
            sample_processes = [
                {"pid": 1000, "name": "agent_web_browser.py", "cmdline": ["python", "agent_web_browser.py", "--port=8000"]},
                {"pid": 1001, "name": "llm_agent_server", "cmdline": ["llm_agent_server", "--model=gpt4", "--addr=localhost:8001"]},
                {"pid": 1002, "name": "code_assistant_agent.py", "cmdline": ["python", "code_assistant_agent.py"]}
            ]
            
            for proc in sample_processes:
                if any(re.match(pattern, proc["name"]) for pattern in self.agent_process_patterns):
                    # Extract agent info from process
                    agent_info = self._extract_agent_info_from_process(proc)
                    if agent_info:
                        discovered_agents.append(agent_info)
                        await self.discovery_service.register_discovered_agent(agent_info)
        
        except Exception as e:
            logger.error(f"Error discovering agents in processes: {str(e)}")
        
        return discovered_agents
    
    def _extract_agent_info_from_process(self, proc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract agent information from a process"""
        try:
            name = proc["name"]
            cmdline = proc["cmdline"]
            
            # Extract capabilities from command line or name
            capabilities = []
            if "web" in name or "browser" in name:
                capabilities.append("web_browsing")
            if "code" in name:
                capabilities.append("code_generation")
            if "llm" in name or "gpt" in name:
                capabilities.append("text_generation")
            
            # Extract specializations from name
            specializations = []
            if "code" in name:
                specializations.append(TaskPurpose.CODE)
            if "web" in name or "browser" in name:
                specializations.append(TaskPurpose.SEARCH)
            if "write" in name:
                specializations.append(TaskPurpose.WRITE)
            
            # Basic agent information
            agent_info = {
                "name": name.replace(".py", "").replace("_", " ").title(),
                "description": f"Agent discovered in process {proc['pid']}",
                "id": f"process_{proc['pid']}",
                "capabilities": capabilities,
                "specializations": specializations
            }
            
            return agent_info
        
        except Exception as e:
            logger.debug(f"Error extracting agent info from process: {str(e)}")
            return None

class NetworkAgentDiscovery(DiscoveryMethod):
    """Discovers agents running on the network"""
    
    def __init__(self, discovery_service: AgentDiscoveryService):
        """Initialize the network discovery method"""
        super().__init__(discovery_service)
        self.discovery_ports = [8000, 8001, 8080, 5000, 3000]  # Common API ports
        self.discovery_endpoints = [
            "/agent-info",
            "/info",
            "/api/agent",
            "/api/info",
            "/mcp/info"
        ]
        self._scan_task = None
    
    async def start(self):
        """Start the network discovery method"""
        await super().start()
        self._scan_task = asyncio.create_task(self._network_scan_loop())
    
    async def stop(self):
        """Stop the network discovery method"""
        if self._scan_task:
            self._scan_task.cancel()
            try:
                await self._scan_task
            except asyncio.CancelledError:
                pass
        await super().stop()
    
    async def _network_scan_loop(self):
        """Periodically scan the network for agents"""
        while self.running:
            try:
                await self.discover_agents()
                await asyncio.sleep(60)  # Scan every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in network scan loop: {str(e)}")
                await asyncio.sleep(30)
    
    async def discover_agents(self) -> List[Dict[str, Any]]:
        """Discover agents on the network"""
        discovered_agents = []
        
        # Get local hostname
        local_host = socket.gethostname()
        
        # Scan localhost and local hostname
        hosts = ["localhost", "127.0.0.1", local_host]
        
        for host in hosts:
            for port in self.discovery_ports:
                for endpoint in self.discovery_endpoints:
                    try:
                        url = f"http://{host}:{port}{endpoint}"
                        async with aiohttp.ClientSession() as session:
                            async with session.get(url, timeout=2) as response:
                                if response.status == 200:
                                    try:
                                        data = await response.json()
                                        if isinstance(data, dict) and ("agent" in data or "name" in data):
                                            agent_info = self._extract_agent_info_from_response(data, host, port)
                                            if agent_info:
                                                discovered_agents.append(agent_info)
                                                await self.discovery_service.register_discovered_agent(agent_info)
                                    except Exception:
                                        # Not JSON or not the right format
                                        pass
                    except (aiohttp.ClientError, asyncio.TimeoutError):
                        # Connection failed, continue to next
                        pass
                    except Exception as e:
                        logger.debug(f"Error connecting to {host}:{port}{endpoint}: {str(e)}")
        
        return discovered_agents
    
    def _extract_agent_info_from_response(self, data: Dict[str, Any], host: str, port: int) -> Optional[Dict[str, Any]]:
        """Extract agent information from a network response"""
        try:
            # Check if the response contains agent information
            if "agent" in data:
                agent_data = data["agent"]
            else:
                agent_data = data
            
            # Basic agent information
            agent_info = {
                "name": agent_data.get("name", f"Network Agent {host}:{port}"),
                "description": agent_data.get("description", f"Agent discovered at {host}:{port}"),
                "id": f"network_{host}_{port}",
                "capabilities": agent_data.get("capabilities", []),
                "specializations": agent_data.get("specializations", []),
                "system_prompt": agent_data.get("system_prompt")
            }
            
            # If the endpoint is an MCP server, mark it accordingly
            if host != "localhost" and host != "127.0.0.1":
                agent_info["mcp_endpoint"] = f"http://{host}:{port}"
            
            return agent_info
        
        except Exception as e:
            logger.debug(f"Error extracting agent info from response: {str(e)}")
            return None

class AgentRegistrationServer(DiscoveryMethod):
    """Server that allows agents to register themselves"""
    
    def __init__(self, discovery_service: AgentDiscoveryService):
        """Initialize the registration server"""
        super().__init__(discovery_service)
        self.port = 5678  # Port for registration server
        self.server = None
        self.registered_endpoints = {}  # endpoint: agent_data
    
    async def start(self):
        """Start the registration server"""
        await super().start()
        
        # In a real implementation, this would start an HTTP server
        # For now, we'll use a mock implementation
        logger.info(f"Agent registration server started on port {self.port}")
    
    async def stop(self):
        """Stop the registration server"""
        # In a real implementation, this would stop the HTTP server
        await super().stop()
    
    async def discover_agents(self) -> List[Dict[str, Any]]:
        """Return list of registered agents"""
        return list(self.registered_endpoints.values())
    
    async def register_agent(self, agent_data: Dict[str, Any], endpoint: str) -> Dict[str, Any]:
        """Register an agent via the API"""
        try:
            # Store agent data
            agent_id = f"api_{endpoint}_{uuid.uuid4().hex[:8]}"
            agent_data["id"] = agent_id
            
            # Add to registered endpoints
            self.registered_endpoints[endpoint] = agent_data
            
            # Register with discovery service
            await self.discovery_service.register_discovered_agent(agent_data)
            
            return {"success": True, "message": "Agent registered successfully", "id": agent_id}
        except Exception as e:
            logger.error(f"Error registering agent via API: {str(e)}")
            return {"success": False, "error": str(e)}

class EnvVarAgentDiscovery(DiscoveryMethod):
    """Discovers agents defined in environment variables"""
    
    async def discover_agents(self) -> List[Dict[str, Any]]:
        """Discover agents defined in environment variables"""
        discovered_agents = []
        
        try:
            # Check environment variables for agent definitions
            for name, value in os.environ.items():
                if name.startswith("MCP_AGENT_") or name.startswith("LLM_AGENT_"):
                    try:
                        # Parse JSON definition
                        agent_data = json.loads(value)
                        agent_data["id"] = f"env_{name}"
                        
                        discovered_agents.append(agent_data)
                        await self.discovery_service.register_discovered_agent(agent_data)
                    except json.JSONDecodeError:
                        # If it's not JSON, treat it as a simple name-endpoint pair
                        parts = value.split(':')
                        if len(parts) == 2:
                            agent_name, endpoint = parts
                            agent_data = {
                                "name": agent_name,
                                "description": f"Agent defined in environment variable {name}",
                                "id": f"env_{name}",
                                "capabilities": [],
                                "specializations": [],
                                "mcp_endpoint": endpoint
                            }
                            
                            discovered_agents.append(agent_data)
                            await self.discovery_service.register_discovered_agent(agent_data)
        
        except Exception as e:
            logger.error(f"Error discovering agents in environment variables: {str(e)}")
        
        return discovered_agents

class ContainerAgentDiscovery(DiscoveryMethod):
    """Discovers agents running in containers (Docker, Kubernetes)"""
    
    def __init__(self, discovery_service: AgentDiscoveryService):
        """Initialize the container discovery method"""
        super().__init__(discovery_service)
        self.container_labels = ["ai.agent", "ai.assistant", "mcp.server"]
    
    async def discover_agents(self) -> List[Dict[str, Any]]:
        """Discover agents running in containers"""
        discovered_agents = []
        
        try:
            # In a real implementation, this would use Docker/K8s APIs
            # For now, we'll use a mock implementation
            
            # Example container data (would come from Docker/K8s APIs)
            containers = [
                {
                    "id": "container1",
                    "name": "web_browser_agent",
                    "labels": {"ai.agent": "true", "agent.capabilities": "web_browsing,screenshot,navigation"},
                    "ports": {"8000/tcp": [{"HostIp": "0.0.0.0", "HostPort": "8000"}]}
                },
                {
                    "id": "container2",
                    "name": "filesystem_mcp_server",
                    "labels": {"mcp.server": "true", "mcp.capabilities": "file_read,file_write,directory_list"},
                    "ports": {"5000/tcp": [{"HostIp": "0.0.0.0", "HostPort": "5001"}]}
                },
            ]
            
            for container in containers:
                if any(label in container["labels"] for label in self.container_labels):
                    # Extract agent info from container
                    agent_info = self._extract_agent_info_from_container(container)
                    if agent_info:
                        discovered_agents.append(agent_info)
                        await self.discovery_service.register_discovered_agent(agent_info)
        
        except Exception as e:
            logger.error(f"Error discovering agents in containers: {str(e)}")
        
        return discovered_agents
    
    def _extract_agent_info_from_container(self, container: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract agent information from a container"""
        try:
            container_id = container["id"]
            name = container["name"]
            labels = container["labels"]
            
            # Extract capabilities from labels
            capabilities = []
            if "agent.capabilities" in labels:
                capabilities = labels["agent.capabilities"].split(",")
            elif "mcp.capabilities" in labels:
                capabilities = labels["mcp.capabilities"].split(",")
            
            # Extract specializations from name and capabilities
            specializations = []
            if "web" in name or "browser" in capabilities:
                specializations.append(TaskPurpose.SEARCH)
            if "file" in name or any("file" in cap for cap in capabilities):
                specializations.append(TaskPurpose.FILE_MANIPULATION)
            
            # Basic agent information
            agent_info = {
                "name": name.replace("_", " ").title(),
                "description": f"Agent discovered in container {container_id}",
                "id": f"container_{container_id}",
                "capabilities": capabilities,
                "specializations": specializations
            }
            
            # Extract MCP endpoint if it's an MCP server
            if "mcp.server" in labels:
                # Find the exposed port
                for port_key, port_mappings in container["ports"].items():
                    if port_mappings:
                        host_port = port_mappings[0]["HostPort"]
                        agent_info["mcp_endpoint"] = f"http://localhost:{host_port}"
                        break
            
            return agent_info
        
        except Exception as e:
            logger.debug(f"Error extracting agent info from container: {str(e)}")
            return None

class AgentWatcher:
    """Watches for agent creation in the current process"""
    
    def __init__(self, orchestrator: 'MCPOrchestrator'):
        """Initialize the agent watcher"""
        self.orchestrator = orchestrator
        self.watched_types = []
        self.agent_refs = {}  # weakrefs to detected agents
    
    def register_agent_type(self, agent_type: type):
        """Register a type to watch for"""
        self.watched_types.append(agent_type)
        
        # Patch the __init__ method to detect new instances
        original_init = agent_type.__init__
        
        def patched_init(self_agent, *args, **kwargs):
            # Call original init
            original_init(self_agent, *args, **kwargs)
            
            # Register the new agent
            self.on_agent_created(self_agent)
        
        agent_type.__init__ = patched_init
    
    def on_agent_created(self, agent):
        """Called when a new agent is created"""
        # Store weak reference to avoid memory leaks
        agent_id = id(agent)
        if agent_id not in self.agent_refs:
            self.agent_refs[agent_id] = weakref.ref(agent, lambda ref: self.on_agent_destroyed(agent_id))
            
            # Extract agent info
            agent_info = self._extract_agent_info(agent)
            
            # Register with orchestrator
            if agent_info:
                asyncio.create_task(
                    self.orchestrator.discovery_service.register_discovered_agent(agent_info)
                )
    
    def on_agent_destroyed(self, agent_id):
        """Called when an agent is destroyed"""
        if agent_id in self.agent_refs:
            del self.agent_refs[agent_id]
    
    def _extract_agent_info(self, agent) -> Optional[Dict[str, Any]]:
        """Extract agent information from an agent object"""
        try:
            agent_type = type(agent).__name__
            
            # Basic agent information
            agent_info = {
                "name": getattr(agent, 'name', agent_type),
                "description": getattr(agent, 'description', f"{agent_type} agent"),
                "id": f"watcher_{id(agent)}",
                "capabilities": getattr(agent, 'capabilities', []),
                "specializations": getattr(agent, 'specializations', [])
            }
            
            # Additional attributes
            if hasattr(agent, 'system_prompt'):
                agent_info["system_prompt"] = agent.system_prompt
            
            return agent_info
        
        except Exception as e:
            logger.debug(f"Error extracting agent info: {str(e)}")
            return None

if __name__ == "__main__":
    asyncio.run(main())
