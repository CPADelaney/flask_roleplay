# nyx/core/brain/system_health_checker.py

import logging
import importlib
import inspect
import asyncio
import sys
import traceback
import time
from typing import Dict, List, Any, Optional, Callable, Union
from pathlib import Path

logger = logging.getLogger(__name__)

class SystemHealthChecker:
    """
    System for checking the health and operational status of Nyx components,
    functions, and agents to ensure everything is properly running and accessible.
    """
    
    def __init__(self, brain):
        """
        Initialize the system health checker
        
        Args:
            brain: Reference to the NyxBrain instance
        """
        self.brain = brain
        self.health_check_history = []
        self.component_registry = {}
        self.last_check_results = {}
        
        # Initialize standard component list if brain is available
        if brain:
            self._initialize_component_registry()
        
        logger.info("System health checker initialized")
    
    def _initialize_component_registry(self):
        """Initialize the component registry with standard Nyx components"""
        # Core components that should be available
        core_components = [
            "emotional_core",
            "memory_core",
            "reflection_engine",
            "experience_interface",
            "internal_feedback",
            "dynamic_adaptation",
            "meta_core",
            "knowledge_core",
            "memory_orchestrator",
            "reasoning_core",
            "identity_evolution",
            "experience_consolidation",
            "cross_user_manager",
            "reflexive_system",
            "hormone_system",
            "attentional_controller",
            "multimodal_integrator",
            "reward_system",
            "temporal_perception",
            "procedural_memory",
            "agent_enhanced_memory",
            "processing_manager",
            "self_config_manager"
        ]
        
        # Register core components
        for component_name in core_components:
            self.register_component(
                component_name,
                lambda brain=self.brain, name=component_name: getattr(brain, name, None),
                f"Core brain component: {component_name}"
            )
        
        # Check for module optimizer
        self.register_component(
            "module_optimizer",
            lambda brain=self.brain: getattr(brain, "module_optimizer", None),
            "Module optimizer for code improvements"
        )
        
        # Check for function tools
        if hasattr(self.brain, "brain_agent") and hasattr(self.brain.brain_agent, "tools"):
            for tool in self.brain.brain_agent.tools:
                if hasattr(tool, "name"):
                    self.register_component(
                        f"function_tool.{tool.name}",
                        lambda brain=self.brain, t=tool: t,
                        f"Function tool: {tool.name}"
                    )
    
    def register_component(self, 
                         name: str, 
                         accessor: Callable, 
                         description: str = None,
                         required: bool = False):
        """
        Register a component to be checked
        
        Args:
            name: Component name
            accessor: Function that returns the component or None if not available
            description: Description of the component
            required: Whether this component is required for system operation
        """
        self.component_registry[name] = {
            "accessor": accessor,
            "description": description or f"Component: {name}",
            "required": required
        }
    
    async def check_system_health(self, detailed: bool = True) -> Dict[str, Any]:
        """
        Run a comprehensive health check on all registered components
        
        Args:
            detailed: Whether to include detailed information in the results
            
        Returns:
            Health check results
        """
        start_time = time.time()
        
        # Initialize results
        results = {
            "timestamp": time.time(),
            "overall_status": "healthy",
            "components": {},
            "agents": {},
            "function_tools": {},
            "issues": [],
            "stats": {
                "total_components": 0,
                "healthy_components": 0,
                "unhealthy_components": 0,
                "missing_components": 0,
                "total_agents": 0,
                "healthy_agents": 0,
                "unhealthy_agents": 0, 
                "total_function_tools": 0,
                "accessible_function_tools": 0,
                "inaccessible_function_tools": 0
            }
        }
        
        # Check registered components
        for name, config in self.component_registry.items():
            component_result = await self._check_component(name, config, detailed)
            
            # Categorize by component type
            if name.startswith("function_tool."):
                results["function_tools"][name] = component_result
                results["stats"]["total_function_tools"] += 1
                if component_result["status"] == "healthy":
                    results["stats"]["accessible_function_tools"] += 1
                else:
                    results["stats"]["inaccessible_function_tools"] += 1
                    if component_result.get("error"):
                        results["issues"].append({
                            "component": name,
                            "type": "function_tool",
                            "error": component_result["error"],
                            "severity": "high" if config["required"] else "medium"
                        })
            elif "agent" in name.lower():
                results["agents"][name] = component_result
                results["stats"]["total_agents"] += 1
                if component_result["status"] == "healthy":
                    results["stats"]["healthy_agents"] += 1
                else:
                    results["stats"]["unhealthy_agents"] += 1
                    if component_result.get("error"):
                        results["issues"].append({
                            "component": name,
                            "type": "agent",
                            "error": component_result["error"],
                            "severity": "high" if config["required"] else "medium"
                        })
            else:
                results["components"][name] = component_result
                results["stats"]["total_components"] += 1
                if component_result["status"] == "healthy":
                    results["stats"]["healthy_components"] += 1
                elif component_result["status"] == "missing":
                    results["stats"]["missing_components"] += 1
                    if config["required"]:
                        results["issues"].append({
                            "component": name,
                            "type": "component",
                            "error": "Required component is missing",
                            "severity": "critical"
                        })
                else:
                    results["stats"]["unhealthy_components"] += 1
                    if component_result.get("error"):
                        results["issues"].append({
                            "component": name,
                            "type": "component",
                            "error": component_result["error"],
                            "severity": "high" if config["required"] else "medium"
                        })
        
        # Check for critical issues
        if any(issue["severity"] == "critical" for issue in results["issues"]):
            results["overall_status"] = "critical"
        elif len(results["issues"]) > 0:
            results["overall_status"] = "issues"
        
        # Add performance data
        results["execution_time"] = time.time() - start_time
        
        # Store results in history
        self.health_check_history.append({
            "timestamp": results["timestamp"],
            "overall_status": results["overall_status"],
            "stats": results["stats"],
            "issues_count": len(results["issues"])
        })
        
        # Store as last check results
        self.last_check_results = results
        
        return results
    
    async def _check_component(self, 
                           name: str, 
                           config: Dict[str, Any],
                           detailed: bool) -> Dict[str, Any]:
        """
        Check the health of a specific component
        
        Args:
            name: Component name
            config: Component configuration
            detailed: Whether to include detailed information
            
        Returns:
            Component health check results
        """
        result = {
            "name": name,
            "description": config["description"],
            "required": config["required"]
        }
        
        try:
            # Get the component using the accessor
            component = config["accessor"]()
            
            if component is None:
                result["status"] = "missing"
                return result
            
            # Basic checks for all components
            result["status"] = "healthy"
            result["type"] = type(component).__name__
            
            # For function tools, check if they can be called
            if name.startswith("function_tool."):
                if hasattr(component, "function") and callable(component.function):
                    result["callable"] = True
                else:
                    result["callable"] = False
                    result["status"] = "unhealthy"
                    result["error"] = "Function tool is not callable"
            
            # For agents, check if they have required attributes
            elif "agent" in name.lower():
                if hasattr(component, "name") and hasattr(component, "instructions"):
                    result["agent_name"] = component.name
                    result["has_tools"] = hasattr(component, "tools") and component.tools is not None
                    if detailed:
                        result["tool_count"] = len(component.tools) if hasattr(component, "tools") and component.tools else 0
                else:
                    result["status"] = "unhealthy"
                    result["error"] = "Agent missing required attributes"
            
            # For other components, check for basic functionality
            else:
                # Try to detect if the component is properly initialized
                if hasattr(component, "initialized"):
                    result["initialized"] = component.initialized
                    if not component.initialized and config["required"]:
                        result["status"] = "unhealthy"
                        result["error"] = "Required component is not initialized"
                
                # Check if component has async methods (indicating it's properly set up)
                has_async_methods = False
                for attr_name, attr_value in inspect.getmembers(component):
                    if inspect.iscoroutinefunction(attr_value) and not attr_name.startswith("_"):
                        has_async_methods = True
                        break
                
                result["has_async_methods"] = has_async_methods
                
                # For detailed checks, list available methods
                if detailed:
                    methods = []
                    for attr_name, attr_value in inspect.getmembers(component):
                        if callable(attr_value) and not attr_name.startswith("_"):
                            methods.append({
                                "name": attr_name, 
                                "async": inspect.iscoroutinefunction(attr_value)
                            })
                    
                    result["methods"] = methods
            
            return result
        except Exception as e:
            # Handle exceptions during check
            result["status"] = "unhealthy"
            result["error"] = str(e)
            if detailed:
                result["traceback"] = traceback.format_exc()
            
            return result
    
    async def test_function_tool(self, tool_name: str) -> Dict[str, Any]:
        """
        Test a specific function tool
        
        Args:
            tool_name: Name of the function tool to test
            
        Returns:
            Test results
        """
        # Only attempt to test tools that are registered and available
        full_name = f"function_tool.{tool_name}"
        if full_name not in self.component_registry:
            return {
                "success": False,
                "error": f"Function tool '{tool_name}' not registered"
            }
        
        try:
            # Get the tool
            config = self.component_registry[full_name]
            tool = config["accessor"]()
            
            if not tool:
                return {
                    "success": False, 
                    "error": f"Function tool '{tool_name}' not available"
                }
            
            # Inspect the tool to get its parameters
            if not hasattr(tool, "function") or not callable(tool.function):
                return {
                    "success": False,
                    "error": f"Function tool '{tool_name}' is not callable"
                }
            
            # Get function signature
            sig = inspect.signature(tool.function)
            params = list(sig.parameters.keys())
            
            # Check if it's a ctx-style tool
            is_ctx_tool = len(params) > 0 and params[0] == "ctx"
            
            # We can't safely call the tool without knowing the parameters,
            # but we can verify its structure and accessibility
            return {
                "success": True,
                "tool_name": tool_name,
                "is_ctx_tool": is_ctx_tool,
                "parameters": params,
                "docstring": inspect.getdoc(tool.function),
                "accessible": True
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    async def check_agent_capabilities(self, agent_name: str) -> Dict[str, Any]:
        """
        Check the capabilities of a specific agent
        
        Args:
            agent_name: Name of the agent to check
            
        Returns:
            Agent capabilities assessment
        """
        # Find the agent
        agent = None
        
        # First check if it's a registered component
        if agent_name in self.component_registry:
            agent = self.component_registry[agent_name]["accessor"]()
        else:
            # Try to find the agent as an attribute of the brain
            if hasattr(self.brain, agent_name):
                agent = getattr(self.brain, agent_name)
            
            # Check in any agent registries
            elif hasattr(self.brain, "agents") and agent_name in self.brain.agents:
                agent = self.brain.agents[agent_name]
            elif hasattr(self.brain, "agent_integration") and hasattr(self.brain.agent_integration, "agents"):
                if agent_name in self.brain.agent_integration.agents:
                    agent = self.brain.agent_integration.agents[agent_name]
        
        if not agent:
            return {
                "success": False,
                "error": f"Agent '{agent_name}' not found"
            }
        
        try:
            # Basic agent information
            result = {
                "success": True,
                "agent_name": agent.name if hasattr(agent, "name") else agent_name,
                "has_instructions": hasattr(agent, "instructions") and agent.instructions is not None,
                "tools": []
            }
            
            # Check tools
            if hasattr(agent, "tools") and agent.tools:
                for tool in agent.tools:
                    tool_info = {
                        "name": tool.name if hasattr(tool, "name") else "unknown",
                        "type": type(tool).__name__
                    }
                    
                    if hasattr(tool, "function") and tool.function:
                        tool_info["function_name"] = tool.function.__name__
                        tool_info["docstring"] = inspect.getdoc(tool.function)
                    
                    result["tools"].append(tool_info)
            
            result["tool_count"] = len(result["tools"])
            
            # Check if agent can be run
            result["can_run"] = hasattr(self.brain, "Runner") and hasattr(self.brain.Runner, "run")
            
            return result
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    async def get_system_overview(self) -> Dict[str, Any]:
        """
        Get a high-level overview of the system's operational status
        
        Returns:
            System overview data
        """
        # First perform a health check if none exists
        if not self.last_check_results:
            await self.check_system_health(detailed=False)
        
        results = self.last_check_results
        
        # Get system information
        overview = {
            "status": results["overall_status"],
            "components": {
                "total": results["stats"]["total_components"],
                "healthy": results["stats"]["healthy_components"],
                "unhealthy": results["stats"]["unhealthy_components"],
                "missing": results["stats"]["missing_components"]
            },
            "agents": {
                "total": results["stats"]["total_agents"],
                "healthy": results["stats"]["healthy_agents"],
                "unhealthy": results["stats"]["unhealthy_agents"]
            },
            "function_tools": {
                "total": results["stats"]["total_function_tools"],
                "accessible": results["stats"]["accessible_function_tools"],
                "inaccessible": results["stats"]["inaccessible_function_tools"]
            },
            "issues": {
                "count": len(results["issues"]),
                "critical": sum(1 for issue in results["issues"] if issue["severity"] == "critical"),
                "high": sum(1 for issue in results["issues"] if issue["severity"] == "high"),
                "medium": sum(1 for issue in results["issues"] if issue["severity"] == "medium")
            },
            "health_check_history": len(self.health_check_history),
            "last_check_time": time.ctime(results["timestamp"]) if "timestamp" in results else "Unknown"
        }
        
        # Add brain information if available
        if hasattr(self.brain, "user_id"):
            overview["brain_info"] = {
                "user_id": self.brain.user_id,
                "conversation_id": getattr(self.brain, "conversation_id", None),
                "interaction_count": getattr(self.brain, "interaction_count", 0),
                "initialized": getattr(self.brain, "initialized", False)
            }
        
        # Add critical issues
        if overview["issues"]["critical"] > 0:
            overview["critical_issues"] = [
                {"component": issue["component"], "error": issue["error"]}
                for issue in results["issues"] 
                if issue["severity"] == "critical"
            ]
        
        return overview
    
    async def verify_module_imports(self, module_names: List[str] = None) -> Dict[str, Any]:
        """
        Verify that specified modules can be imported
        
        Args:
            module_names: List of module names to check, or None for default core modules
            
        Returns:
            Import verification results
        """
        if not module_names:
            # Default core modules to check
            module_names = [
                "nyx.core.brain.base",
                "nyx.core.brain.models",
                "nyx.core.brain.function_tools",
                "nyx.core.brain.processing.manager",
                "nyx.core.brain.adaptation.self_config",
                "nyx.core.brain.adaptation.context_detection",
                "nyx.core.brain.adaptation.strategy",
                "nyx.core.brain.utils.task_manager"
            ]
        
        results = {
            "success": True,
            "modules": {},
            "import_count": 0,
            "failed_count": 0
        }
        
        for module_name in module_names:
            try:
                # Try to import the module
                module = importlib.import_module(module_name)
                
                # Get some basic info about the module
                module_info = {
                    "imported": True,
                    "path": getattr(module, "__file__", "Unknown"),
                    "has_init": hasattr(module, "__init__"),
                    "attributes": dir(module)[:10]  # First 10 attributes for brevity
                }
                
                results["modules"][module_name] = module_info
                results["import_count"] += 1
            except (ImportError, ModuleNotFoundError) as e:
                results["modules"][module_name] = {
                    "imported": False,
                    "error": str(e)
                }
                results["failed_count"] += 1
                results["success"] = False
        
        return results
    
    def get_component_docs(self, component_name: str) -> Dict[str, Any]:
        """
        Get documentation for a component
        
        Args:
            component_name: Name of the component
            
        Returns:
            Component documentation
        """
        # Find the component
        component = None
        
        # First check if it's a registered component
        if component_name in self.component_registry:
            component = self.component_registry[component_name]["accessor"]()
        else:
            # Try to find the component as an attribute of the brain
            if hasattr(self.brain, component_name):
                component = getattr(self.brain, component_name)
        
        if not component:
            return {
                "success": False,
                "error": f"Component '{component_name}' not found"
            }
        
        try:
            # Get documentation
            result = {
                "success": True,
                "component_name": component_name,
                "type": type(component).__name__,
                "docstring": inspect.getdoc(component),
                "methods": []
            }
            
            # Get method documentation
            for name, method in inspect.getmembers(component, predicate=inspect.ismethod):
                if not name.startswith("_"):  # Skip private methods
                    method_doc = {
                        "name": name,
                        "docstring": inspect.getdoc(method),
                        "signature": str(inspect.signature(method)),
                        "is_async": inspect.iscoroutinefunction(method)
                    }
                    result["methods"].append(method_doc)
            
            return result
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
