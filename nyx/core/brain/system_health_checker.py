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

    # Add these methods to the SystemHealthChecker class
    
    async def discover_all_modules(self, base_dir: str = None) -> Dict[str, Any]:
        """
        Scan all Python modules in the Nyx codebase and analyze their structure
        
        Args:
            base_dir: Base directory to scan (defaults to detecting the nyx/core directory)
            
        Returns:
            Complete module map with dependencies and theoretical capabilities
        """
        # Determine base directory if not provided
        if not base_dir:
            if hasattr(self.brain, "__file__"):
                # Get the directory of the brain module
                brain_file = getattr(self.brain, "__file__")
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(brain_file)))
            else:
                # Try to find the nyx directory in sys.path
                for path in sys.path:
                    potential_path = os.path.join(path, 'nyx', 'core')
                    if os.path.isdir(potential_path):
                        base_dir = potential_path
                        break
                
                if not base_dir:
                    return {
                        "success": False,
                        "error": "Could not determine base directory for nyx/core"
                    }
        
        # Verify the directory exists
        if not os.path.isdir(base_dir):
            return {
                "success": False,
                "error": f"Directory not found: {base_dir}"
            }
        
        logger.info(f"Scanning modules in: {base_dir}")
        
        # Results container
        results = {
            "modules": {},
            "dependencies": {},
            "classes": {},
            "functions": {},
            "stats": {
                "total_modules": 0,
                "total_classes": 0,
                "total_functions": 0,
                "total_dependencies": 0
            }
        }
        
        # Scan all Python files recursively
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.py'):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, base_dir)
                    
                    # Convert to module path format
                    module_path = os.path.splitext(rel_path)[0].replace(os.path.sep, '.')
                    
                    # Get full module path including nyx.core prefix if needed
                    if not module_path.startswith('nyx.core'):
                        # Detect the right prefix by checking parent directories
                        parts = []
                        current = os.path.dirname(base_dir)
                        while os.path.basename(current) != 'nyx' and current != os.path.dirname(current):
                            parts.insert(0, os.path.basename(current))
                            current = os.path.dirname(current)
                        
                        if os.path.basename(current) == 'nyx':
                            prefix = 'nyx.' + '.'.join(parts)
                            module_path = f"{prefix}.{module_path}"
                    
                    # Analyze this module
                    module_info = await self._analyze_module_code(full_path, module_path)
                    
                    if module_info:
                        results["modules"][module_path] = module_info
                        results["stats"]["total_modules"] += 1
                        results["stats"]["total_classes"] += len(module_info.get("classes", []))
                        results["stats"]["total_functions"] += len(module_info.get("functions", []))
                        results["stats"]["total_dependencies"] += len(module_info.get("imports", []))
                        
                        # Add to dependencies map
                        results["dependencies"][module_path] = module_info.get("imports", [])
                        
                        # Track classes and functions for later analysis
                        for class_info in module_info.get("classes", []):
                            class_name = class_info["name"]
                            full_name = f"{module_path}.{class_name}"
                            results["classes"][full_name] = {
                                "module": module_path,
                                "info": class_info
                            }
                        
                        for func_info in module_info.get("functions", []):
                            func_name = func_info["name"]
                            full_name = f"{module_path}.{func_name}"
                            results["functions"][full_name] = {
                                "module": module_path,
                                "info": func_info
                            }
        
        # Analyze dependencies to determine theoretical accessibility
        results["theoretical_accessibility"] = await self._analyze_theoretical_accessibility(results)
        
        # Add missing components - compare with what we can actually access
        results["missing_components"] = await self._find_missing_components(results)
        
        return {
            "success": True,
            "results": results
        }

    async def quick_health_check(self) -> Dict[str, Any]:
        """
        Perform a quick health check focusing on critical components only
        
        Returns:
            Quick health check results
        """
        try:
            # Quick check of critical components only
            critical_components = [
                "emotional_core",
                "memory_core", 
                "reasoning_core",
                "global_workspace"
            ]
            
            issues = []
            for component_name in critical_components:
                if component_name in self.component_registry:
                    config = self.component_registry[component_name]
                    component = config["accessor"]()
                    
                    if component is None and config.get("required", False):
                        issues.append({
                            "component": component_name,
                            "error": "Required component is missing"
                        })
            
            # Return simple status
            return {
                "status": "healthy" if not issues else "unhealthy",
                "issues": issues,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Quick health check failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def _analyze_module_code(self, file_path: str, module_path: str) -> Dict[str, Any]:
        """
        Analyze a Python module file to extract its structure
        
        Args:
            file_path: Path to the Python file
            module_path: Dot-notation module path
            
        Returns:
            Module information including imports, classes, functions
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            # Parse the code
            tree = ast.parse(code)
            
            # Extract imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        imports.append(name.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        for name in node.names:
                            imports.append(f"{node.module}.{name.name}")
            
            # Extract classes
            classes = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = {
                        "name": node.name,
                        "methods": [],
                        "attributes": [],
                        "bases": [base.id if hasattr(base, 'id') else "complex_base" for base in node.bases if hasattr(base, 'id')]
                    }
                    
                    # Get docstring
                    class_info["docstring"] = ast.get_docstring(node)
                    
                    # Get methods
                    for method in [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
                        method_info = {
                            "name": method.name,
                            "is_async": isinstance(method, ast.AsyncFunctionDef),
                            "args": [arg.arg for arg in method.args.args],
                            "has_docstring": ast.get_docstring(method) is not None
                        }
                        class_info["methods"].append(method_info)
                    
                    # Get attributes (approximation - this is not perfect)
                    for attr in [n for n in node.body if isinstance(n, ast.Assign)]:
                        for target in attr.targets:
                            if isinstance(target, ast.Name):
                                class_info["attributes"].append(target.id)
                    
                    classes.append(class_info)
            
            # Extract top-level functions
            functions = []
            for node in [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
                func_info = {
                    "name": node.name,
                    "is_async": isinstance(node, ast.AsyncFunctionDef),
                    "args": [arg.arg for arg in node.args.args],
                    "has_docstring": ast.get_docstring(node) is not None,
                    "is_decorated": bool(node.decorator_list)
                }
                
                # Check if it might be a function tool
                if func_info["is_decorated"]:
                    for decorator in node.decorator_list:
                        if hasattr(decorator, 'id') and decorator.id == 'function_tool':
                            func_info["is_function_tool"] = True
                        elif isinstance(decorator, ast.Call) and hasattr(decorator.func, 'id') and decorator.func.id == 'function_tool':
                            func_info["is_function_tool"] = True
                
                functions.append(func_info)
            
            # Get module docstring
            module_docstring = ast.get_docstring(tree)
            
            return {
                "path": file_path,
                "module": module_path,
                "imports": imports,
                "classes": classes,
                "functions": functions,
                "docstring": module_docstring,
                "loc": len(code.splitlines())
            }
        except Exception as e:
            logger.error(f"Error analyzing module {module_path}: {str(e)}")
            return None
    
    async def _analyze_theoretical_accessibility(self, scan_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the dependency graph to determine what should theoretically be accessible
        
        Args:
            scan_results: Results from module scanning
            
        Returns:
            Theoretical accessibility analysis
        """
        # Build dependency graph
        import networkx as nx
        G = nx.DiGraph()
        
        # Add all modules as nodes
        for module in scan_results["modules"]:
            G.add_node(module)
        
        # Add dependencies as edges
        for module, deps in scan_results["dependencies"].items():
            for dep in deps:
                # Only add edges for dependencies we know about
                if dep in scan_results["modules"]:
                    G.add_edge(module, dep)
        
        # Find entry points (likely used by brain.py)
        entry_points = []
        for module in scan_results["modules"]:
            if "brain.base" in module or module.endswith("__init__"):
                entry_points.append(module)
        
        # Determine what should be accessible from the entry points
        accessible = set()
        for entry in entry_points:
            # Add the entry point
            accessible.add(entry)
            
            # Add everything reachable from this entry
            if entry in G:
                for dep in nx.descendants(G, entry):
                    accessible.add(dep)
        
        # Map of what each module should have access to
        accessibility_map = {}
        for module in scan_results["modules"]:
            reachable = set()
            if module in G:
                for dep in nx.descendants(G, module):
                    reachable.add(dep)
            
            accessibility_map[module] = {
                "can_access": list(reachable),
                "accessible_from_entry": module in accessible
            }
        
        # Analyze what components should be accessible based on import patterns
        imported_components = {}
        for module, module_info in scan_results["modules"].items():
            for imp in module_info.get("imports", []):
                # Check if this import corresponds to a known module
                for known_module in scan_results["modules"]:
                    if imp == known_module or imp.startswith(known_module + "."):
                        # This module is importing something from a known module
                        if known_module not in imported_components:
                            imported_components[known_module] = set()
                        
                        # If it's importing a specific component from the module
                        if imp != known_module:
                            component = imp[len(known_module)+1:]
                            imported_components[known_module].add(component)
        
        return {
            "accessibility_map": accessibility_map,
            "entry_points": entry_points,
            "imported_components": {k: list(v) for k, v in imported_components.items()}
        }
    
    async def _find_missing_components(self, scan_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find components that should be accessible but aren't
        
        Args:
            scan_results: Results from module scanning
            
        Returns:
            List of missing components
        """
        missing = []
        
        # Check each class that should be accessible
        for full_class_name, class_info in scan_results["classes"].items():
            module_path = class_info["module"]
            
            # If this class is in a module accessible from entry points
            if scan_results["theoretical_accessibility"]["accessibility_map"].get(
                module_path, {}).get("accessible_from_entry", False):
                
                # Try to import and access it
                try:
                    module_name, class_name = full_class_name.rsplit(".", 1)
                    module = importlib.import_module(module_name)
                    
                    if not hasattr(module, class_name):
                        missing.append({
                            "type": "class",
                            "name": full_class_name,
                            "module": module_path,
                            "reason": "Class defined but not accessible via import"
                        })
                except (ImportError, ModuleNotFoundError) as e:
                    missing.append({
                        "type": "class",
                        "name": full_class_name,
                        "module": module_path,
                        "reason": f"Module import failed: {str(e)}"
                    })
        
        # Check function tools
        for full_func_name, func_info in scan_results["functions"].items():
            if func_info["info"].get("is_function_tool", False):
                # This should be a function tool accessible to the brain
                function_name = func_info["info"]["name"]
                
                # Check if it's registered with the brain
                tool_found = False
                if hasattr(self.brain, "brain_agent") and hasattr(self.brain.brain_agent, "tools"):
                    for tool in self.brain.brain_agent.tools:
                        if (hasattr(tool, "function") and hasattr(tool.function, "__name__") 
                                and tool.function.__name__ == function_name):
                            tool_found = True
                            break
                
                if not tool_found:
                    missing.append({
                        "type": "function_tool",
                        "name": function_name,
                        "full_name": full_func_name,
                        "module": func_info["module"],
                        "reason": "Function tool defined but not registered with brain agent"
                    })
        
        return missing
    
    async def compare_theoretical_vs_actual(self) -> Dict[str, Any]:
        """
        Compare what should theoretically be accessible vs what is actually accessible
        
        Returns:
            Comparison results
        """
        # Run the module discovery
        discovery_results = await self.discover_all_modules()
        if not discovery_results.get("success", False):
            return discovery_results
        
        # Run a health check to get actual component status
        health_results = await self.check_system_health(detailed=True)
        
        # Compare the results
        comparison = {
            "mismatches": [],
            "missing_components": [],
            "unexpected_components": [],
            "health_status": health_results["overall_status"]
        }
        
        # Check for missing components from discovery
        for missing in discovery_results["results"]["missing_components"]:
            comparison["missing_components"].append({
                "name": missing["name"],
                "type": missing["type"],
                "reason": missing["reason"]
            })
        
        # Check for unexpected components not found in code analysis
        for component_name in health_results["components"]:
            found = False
            # Check against classes from discovery
            for full_class_name in discovery_results["results"]["classes"]:
                if full_class_name.endswith("." + component_name):
                    found = True
                    break
            
            if not found:
                comparison["unexpected_components"].append({
                    "name": component_name,
                    "type": "component",
                    "status": health_results["components"][component_name]["status"]
                })
        
        # Check for function tool mismatches
        theoretical_function_tools = []
        for func_name, func_info in discovery_results["results"]["functions"].items():
            if func_info["info"].get("is_function_tool", False):
                theoretical_function_tools.append(func_name.split(".")[-1])
        
        actual_function_tools = []
        for tool_name in health_results["function_tools"]:
            if tool_name.startswith("function_tool."):
                actual_function_tools.append(tool_name.replace("function_tool.", ""))
        
        # Find tools that should exist but don't
        for tool in theoretical_function_tools:
            if tool not in actual_function_tools:
                comparison["mismatches"].append({
                    "name": tool,
                    "type": "function_tool",
                    "issue": "Tool defined in code but not available at runtime"
                })
        
        # Find tools that exist but weren't found in code analysis
        for tool in actual_function_tools:
            if tool not in theoretical_function_tools:
                comparison["mismatches"].append({
                    "name": tool,
                    "type": "function_tool",
                    "issue": "Tool available at runtime but not found in code analysis"
                })
        
        # Summary statistics
        comparison["stats"] = {
            "theoretical_components": len(discovery_results["results"]["classes"]),
            "actual_components": len(health_results["components"]),
            "theoretical_function_tools": len(theoretical_function_tools),
            "actual_function_tools": len(actual_function_tools),
            "missing_count": len(comparison["missing_components"]),
            "unexpected_count": len(comparison["unexpected_components"]),
            "mismatch_count": len(comparison["mismatches"])
        }
        
        return {
            "success": True,
            "comparison": comparison,
            "discovery": discovery_results["results"],
            "health_check": health_results
        }
    
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
