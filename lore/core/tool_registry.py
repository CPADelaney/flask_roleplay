# lore/core/tool_registry.py

import inspect
import logging
from typing import Dict, Any, List, Callable, Optional, Type, Union
import functools

from agents import function_tool, Agent

class FunctionToolRegistry:
    """
    Central registry for function tools across the lore system.
    Enables cross-module discovery, invocation, and composition.
    """
    
    _instance = None  # Singleton instance
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FunctionToolRegistry, cls).__new__(cls)
            cls._instance._tools = {}
            cls._instance._categories = {}
            cls._instance._dependencies = {}
        return cls._instance
    
    def register_tool(self, function: Callable, category: str = "general", dependencies: List[str] = None):
        """
        Register a function tool in the registry.
        """
        # Skip if not a function tool
        if not hasattr(function, "_is_function_tool") or not function._is_function_tool:
            logging.warning(f"Attempted to register non-function tool: {getattr(function, '__name__', str(function))}")
            return
        
        # Create a unique ID for the tool - get module name and function name safely
        module_name = getattr(function, "__module__", "unknown_module")
        function_name = getattr(function, "__name__", f"unnamed_tool_{id(function)}")
        tool_id = f"{module_name}.{function_name}"
        
        # Register the tool
        self._tools[tool_id] = function
        
        # Add to category
        if category not in self._categories:
            self._categories[category] = []
        self._categories[category].append(tool_id)
        
        # Register dependencies
        if dependencies:
            self._dependencies[tool_id] = dependencies
        
        logging.info(f"Registered function tool: {tool_id} in category: {category}")
    
    def get_tool(self, tool_id: str) -> Optional[Callable]:
        """Get a registered function tool by its ID."""
        return self._tools.get(tool_id)
    
    def get_tools_by_category(self, category: str) -> List[Callable]:
        """Get all function tools in a category."""
        tool_ids = self._categories.get(category, [])
        return [self._tools[tool_id] for tool_id in tool_ids if tool_id in self._tools]
    
    def get_all_tools(self) -> Dict[str, Callable]:
        """Get all registered function tools."""
        return self._tools.copy()
    
    def get_tool_info(self, tool_id: str) -> Dict[str, Any]:
        """Get detailed information about a registered tool."""
        tool = self.get_tool(tool_id)
        if not tool:
            return {}
        
        # Extract information from the function
        signature = inspect.signature(tool)
        
        # Build parameter info
        params = []
        for name, param in signature.parameters.items():
            # Skip self and ctx parameters
            if name in ["self", "cls", "ctx"]:
                continue
                
            param_info = {
                "name": name,
                "required": param.default == inspect.Parameter.empty,
            }
            
            # Add type hint if available
            if param.annotation != inspect.Parameter.empty:
                param_info["type"] = str(param.annotation)
            
            # Add default value if available
            if param.default != inspect.Parameter.empty:
                param_info["default"] = str(param.default)
                
            params.append(param_info)
        
        return {
            "id": tool_id,
            "name": tool.__name__,
            "description": tool.__doc__ or "No description available",
            "parameters": params,
            "return_type": str(signature.return_annotation) if signature.return_annotation != inspect.Parameter.empty else "Any",
            "category": next((cat for cat, ids in self._categories.items() if tool_id in ids), "unknown"),
            "dependencies": self._dependencies.get(tool_id, [])
        }
    
    def create_agent_with_tools(
        self, 
        agent: Agent, 
        categories: List[str] = None, 
        tool_ids: List[str] = None
    ) -> Agent:
        """
        Create a clone of an agent with selected function tools attached.
        
        Args:
            agent: The agent to clone and add tools to
            categories: Categories of tools to include
            tool_ids: Specific tool IDs to include
            
        Returns:
            A new agent with the requested tools
        """
        # Collect tools to add
        tools = []
        
        # Add tools by category
        if categories:
            for category in categories:
                tools.extend(self.get_tools_by_category(category))
        
        # Add tools by ID
        if tool_ids:
            for tool_id in tool_ids:
                tool = self.get_tool(tool_id)
                if tool and tool not in tools:
                    tools.append(tool)
        
        # Clone agent with tools
        return agent.clone(tools=tools)
    
    def make_registered_tool(self, category: str = "general", dependencies: List[str] = None):
        """
        Decorator to register a function as a tool and ensure it's added to the registry.
        
        Args:
            category: Category for the tool
            dependencies: List of tool IDs this tool depends on
            
        Returns:
            Decorated function that is both a function_tool and registered
        """
        def decorator(func):
            # Make it a function tool if it's not already
            if not hasattr(func, "_is_function_tool") or not func._is_function_tool:
                func = function_tool(func)
            
            # Register the tool
            self.register_tool(func, category, dependencies)
            
            return func
        return decorator

# Create singleton instance
tool_registry = FunctionToolRegistry()

# Helper decorator for easy registration
def registered_tool(category: str = "general", dependencies: List[str] = None):
    """
    Decorator to register a function as a tool in the registry.
    
    Args:
        category: Category for grouping related tools
        dependencies: List of tool IDs that this tool depends on
        
    Returns:
        Decorated function that is both a function_tool and registered
    """
    return tool_registry.make_registered_tool(category, dependencies)
