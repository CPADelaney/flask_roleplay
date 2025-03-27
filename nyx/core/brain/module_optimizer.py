# nyx/core/brain/module_optimizer.py

import logging
import os
import inspect
import importlib
import datetime
import ast
import asyncio
import re
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

from agents import Agent, Runner, trace

logger = logging.getLogger(__name__)

class ModuleOptimizer:
    """
    System for analyzing and suggesting improvements to Nyx modules
    or creating entirely new modules with enhanced functionality.
    """
    
    def __init__(self, brain):
        """
        Initialize the module optimizer
        
        Args:
            brain: Reference to the NyxBrain instance
        """
        self.brain = brain
        self.optimization_history = []
        self.optimized_modules = {}  # module_name -> {version -> content}
        
        # Default optimization directory - create if it doesn't exist
        self.optimization_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "optimized_modules"
        )
        os.makedirs(self.optimization_dir, exist_ok=True)
        
        # Load any existing optimized modules
        self._load_optimized_modules()
        
        # Create optimization agent
        self.optimization_agent = self._create_optimization_agent()
        
        logger.info("Module optimizer initialized")
    
    def _load_optimized_modules(self):
        """Load existing optimized modules from disk"""
        try:
            if not os.path.exists(self.optimization_dir):
                return
                
            # Pattern to match optimized module files: modulename_v123.py
            pattern = re.compile(r'(.+)_v(\d+)\.py$')
            
            for filename in os.listdir(self.optimization_dir):
                match = pattern.match(filename)
                if match:
                    module_name = match.group(1)
                    version = int(match.group(2))
                    
                    file_path = os.path.join(self.optimization_dir, filename)
                    with open(file_path, 'r') as f:
                        content = f.read()
                        
                    if module_name not in self.optimized_modules:
                        self.optimized_modules[module_name] = {}
                    
                    self.optimized_modules[module_name][version] = {
                        'content': content,
                        'path': file_path,
                        'timestamp': datetime.datetime.fromtimestamp(
                            os.path.getmtime(file_path)
                        ).isoformat()
                    }
            
            logger.info(f"Loaded {len(self.optimized_modules)} optimized modules")
        except Exception as e:
            logger.error(f"Error loading optimized modules: {str(e)}")
    
    def _create_optimization_agent(self) -> Agent:
        """Create the agent that will optimize modules"""
        return Agent(
            name="Module Optimization Agent",
            instructions="""
            You are a specialized code optimization agent for the Nyx system.
            Your task is to analyze existing Python modules and suggest improvements
            or create entirely new modules based on requirements.
            
            When optimizing a module, consider:
            1. Code organization and readability
            2. Performance improvements
            3. Error handling and robustness
            4. Integration with other Nyx components
            5. Documentation completeness
            
            When creating a new module, ensure:
            1. Consistency with Nyx coding style and patterns
            2. Proper error handling
            3. Complete documentation
            4. Appropriate integration points with existing components
            5. Comprehensive typing
            
            Your output should be a complete Python module with all necessary imports,
            classes, functions, and documentation.
            """
        )
    
    async def analyze_module(self, 
                          module_path: str,
                          detailed: bool = False) -> Dict[str, Any]:
        """
        Analyze a module for potential improvements
        
        Args:
            module_path: Path to the module (e.g., 'nyx.core.brain.base')
            detailed: Whether to provide detailed analysis
            
        Returns:
            Analysis results
        """
        try:
            # Convert path to filename
            if '.' in module_path:
                module_name = module_path.split('.')[-1]
                full_path = module_path.replace('.', '/') + '.py'
            else:
                module_name = os.path.basename(module_path)
                if module_name.endswith('.py'):
                    module_name = module_name[:-3]
                full_path = module_path
            
            # Load the module content
            module_content = self._load_module_content(full_path)
            if not module_content:
                return {
                    "success": False,
                    "error": f"Could not load module: {module_path}"
                }
            
            # Parse the module
            try:
                module_ast = ast.parse(module_content)
            except SyntaxError as e:
                return {
                    "success": False,
                    "error": f"Syntax error in module: {str(e)}"
                }
            
            # Perform basic analysis
            analysis = self._analyze_module_structure(module_ast, module_content)
            
            # Add module metadata
            analysis["module_name"] = module_name
            analysis["module_path"] = module_path
            analysis["module_size"] = len(module_content)
            analysis["line_count"] = module_content.count('\n') + 1
            
            # For detailed analysis, use the optimization agent
            if detailed:
                prompt = f"""
                Analyze the following Python module and identify potential improvements:
                
                Module name: {module_name}
                
                ```python
                {module_content}
                ```
                
                Provide the following analysis:
                1. Code quality assessment
                2. Performance concerns
                3. Error handling review
                4. Integration points with other modules
                5. Documentation completeness
                6. Specific recommendations for improvement
                
                Format your analysis as structured data that could be parsed as JSON.
                """
                
                try:
                    agent_result = await Runner.run(self.optimization_agent, prompt)
                    analysis["detailed_analysis"] = agent_result.final_output
                except Exception as e:
                    logger.error(f"Error in detailed analysis: {str(e)}")
                    analysis["detailed_analysis"] = f"Error in analysis: {str(e)}"
            
            return {
                "success": True,
                "analysis": analysis
            }
        except Exception as e:
            logger.error(f"Error analyzing module {module_path}: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _analyze_module_structure(self, module_ast: ast.Module, content: str) -> Dict[str, Any]:
        """Analyze the structure of a module"""
        # Count different types of nodes
        class_count = sum(1 for node in ast.walk(module_ast) if isinstance(node, ast.ClassDef))
        function_count = sum(1 for node in ast.walk(module_ast) if isinstance(node, ast.FunctionDef))
        async_function_count = sum(1 for node in ast.walk(module_ast) if isinstance(node, ast.AsyncFunctionDef))
        import_count = sum(1 for node in ast.walk(module_ast) if isinstance(node, (ast.Import, ast.ImportFrom)))
        
        # Check for docstrings
        has_module_docstring = ast.get_docstring(module_ast) is not None
        
        class_docstrings = 0
        method_docstrings = 0
        
        for node in ast.walk(module_ast):
            if isinstance(node, ast.ClassDef) and ast.get_docstring(node) is not None:
                class_docstrings += 1
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and ast.get_docstring(node) is not None:
                method_docstrings += 1
        
        # Check for type hints
        typed_functions = 0
        for node in ast.walk(module_ast):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.returns or any(arg.annotation for arg in node.args.args):
                    typed_functions += 1
        
        # Check for try/except blocks (error handling)
        error_handling_blocks = sum(1 for node in ast.walk(module_ast) if isinstance(node, ast.Try))
        
        # Get all class and function names
        classes = [node.name for node in ast.walk(module_ast) if isinstance(node, ast.ClassDef)]
        functions = [node.name for node in ast.walk(module_ast) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        
        # Calculate documentation percentage
        doc_percentage = 0
        if class_count + function_count + async_function_count > 0:
            doc_percentage = ((class_docstrings + method_docstrings) / 
                             (class_count + function_count + async_function_count)) * 100
        
        # Calculate typing percentage
        typing_percentage = 0
        if function_count + async_function_count > 0:
            typing_percentage = (typed_functions / (function_count + async_function_count)) * 100
        
        return {
            "class_count": class_count,
            "function_count": function_count,
            "async_function_count": async_function_count,
            "import_count": import_count,
            "error_handling_blocks": error_handling_blocks,
            "has_module_docstring": has_module_docstring,
            "class_docstrings": class_docstrings,
            "method_docstrings": method_docstrings,
            "typed_functions": typed_functions,
            "doc_percentage": doc_percentage,
            "typing_percentage": typing_percentage,
            "classes": classes,
            "functions": functions
        }
    
    def _load_module_content(self, module_path: str) -> Optional[str]:
        """Load a module's content"""
        try:
            # First try as a file path
            if os.path.exists(module_path):
                with open(module_path, 'r') as f:
                    return f.read()
            
            # Then try as a module name
            if module_path.endswith('.py'):
                module_path = module_path[:-3]
            
            module_path = module_path.replace('/', '.')
            
            # Try to import the module
            try:
                module = importlib.import_module(module_path)
                if hasattr(module, '__file__'):
                    with open(module.__file__, 'r') as f:
                        return f.read()
            except (ImportError, ModuleNotFoundError):
                pass
            
            # Try to find the module in the Nyx directory
            nyx_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            potential_path = os.path.join(nyx_dir, module_path.replace('.', '/') + '.py')
            
            if os.path.exists(potential_path):
                with open(potential_path, 'r') as f:
                    return f.read()
            
            return None
        except Exception as e:
            logger.error(f"Error loading module content: {str(e)}")
            return None
    
    async def optimize_module(self, 
                           module_path: str, 
                           optimization_goal: str = "general",
                           ensure_backwards_compatible: bool = True) -> Dict[str, Any]:
        """
        Create an optimized version of an existing module
        
        Args:
            module_path: Path to the module to optimize
            optimization_goal: Specific goal for optimization (e.g., "performance", "error_handling")
            ensure_backwards_compatible: Whether the optimized module should be backwards compatible
            
        Returns:
            Optimization results
        """
        try:
            # Get module name
            if '.' in module_path:
                module_name = module_path.split('.')[-1]
            else:
                module_name = os.path.basename(module_path)
                if module_name.endswith('.py'):
                    module_name = module_name[:-3]
            
            # Load the module content
            module_content = self._load_module_content(module_path)
            if not module_content:
                return {
                    "success": False,
                    "error": f"Could not load module: {module_path}"
                }
            
            # Create prompt for the optimization agent
            prompt = f"""
            Optimize the following Python module:
            
            Module name: {module_name}
            Optimization goal: {optimization_goal}
            Must be backwards compatible: {ensure_backwards_compatible}
            
            ```python
            {module_content}
            ```
            
            Create an improved version of this module that:
            1. Addresses the optimization goal: {optimization_goal}
            2. Maintains or improves code quality
            3. Enhances error handling and robustness
            4. Improves performance where possible
            5. Maintains or improves integration with other Nyx components
            
            Your response should be ONLY the complete, optimized Python code for the module,
            with no additional explanations or markdown. The optimized code should be a drop-in
            replacement for the original module.
            
            Include a module docstring at the top that explains the optimizations made.
            """
            
            # Run the optimization agent
            agent_result = await Runner.run(self.optimization_agent, prompt)
            optimized_code = agent_result.final_output
            
            # Clean up the code if needed (remove markdown code blocks if present)
            optimized_code = self._clean_code_output(optimized_code)
            
            # Generate new version number
            version = 1
            if module_name in self.optimized_modules:
                version = max(self.optimized_modules[module_name].keys()) + 1
            
            # Save the optimized module
            file_name = f"{module_name}_v{version}.py"
            file_path = os.path.join(self.optimization_dir, file_name)
            
            with open(file_path, 'w') as f:
                f.write(optimized_code)
            
            # Update the optimization history
            timestamp = datetime.datetime.now().isoformat()
            
            if module_name not in self.optimized_modules:
                self.optimized_modules[module_name] = {}
            
            self.optimized_modules[module_name][version] = {
                'content': optimized_code,
                'path': file_path,
                'timestamp': timestamp,
                'optimization_goal': optimization_goal
            }
            
            self.optimization_history.append({
                'module_name': module_name,
                'version': version,
                'timestamp': timestamp,
                'optimization_goal': optimization_goal,
                'backward_compatible': ensure_backwards_compatible
            })
            
            return {
                "success": True,
                "module_name": module_name,
                "version": version,
                "path": file_path,
                "optimized_code": optimized_code
            }
        except Exception as e:
            logger.error(f"Error optimizing module {module_path}: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _clean_code_output(self, code: str) -> str:
        """Clean up code output from the agent"""
        # Remove markdown code blocks if present
        if code.startswith('```python'):
            code = code.replace('```python', '', 1)
            if code.endswith('```'):
                code = code[:-3]
        elif code.startswith('```'):
            code = code.replace('```', '', 1)
            if code.endswith('```'):
                code = code[:-3]
        
        return code.strip()
    
    async def create_new_module(self, 
                             module_name: str, 
                             description: str,
                             requirements: List[str] = None,
                             integration_points: List[str] = None) -> Dict[str, Any]:
        """
        Create a new module from scratch
        
        Args:
            module_name: Name for the new module
            description: Description of the module's purpose
            requirements: List of specific requirements for the module
            integration_points: List of modules this should integrate with
            
        Returns:
            Creation results
        """
        try:
            # Clean up module name
            if module_name.endswith('.py'):
                module_name = module_name[:-3]
            
            # Create prompt for the optimization agent
            prompt = f"""
            Create a new Python module for Nyx:
            
            Module name: {module_name}
            Description: {description}
            
            Requirements:
            {requirements or []}
            
            Should integrate with these modules:
            {integration_points or []}
            
            Create a complete Python module that:
            1. Fulfills the description and requirements
            2. Follows Nyx coding style and patterns
            3. Has comprehensive error handling
            4. Is well-documented with docstrings
            5. Uses proper typing throughout
            6. Properly integrates with the specified modules
            
            Your response should be ONLY the complete Python code for the module,
            with no additional explanations or markdown.
            
            Include a comprehensive module docstring at the top.
            """
            
            # Run the optimization agent
            agent_result = await Runner.run(self.optimization_agent, prompt)
            new_code = agent_result.final_output
            
            # Clean up the code if needed
            new_code = self._clean_code_output(new_code)
            
            # Generate version number
            version = 1
            
            # Save the new module
            file_name = f"{module_name}_v{version}.py"
            file_path = os.path.join(self.optimization_dir, file_name)
            
            with open(file_path, 'w') as f:
                f.write(new_code)
            
            # Update the optimization history
            timestamp = datetime.datetime.now().isoformat()
            
            if module_name not in self.optimized_modules:
                self.optimized_modules[module_name] = {}
            
            self.optimized_modules[module_name][version] = {
                'content': new_code,
                'path': file_path,
                'timestamp': timestamp,
                'new_module': True
            }
            
            self.optimization_history.append({
                'module_name': module_name,
                'version': version,
                'timestamp': timestamp,
                'new_module': True,
                'description': description
            })
            
            return {
                "success": True,
                "module_name": module_name,
                "version": version,
                "path": file_path,
                "code": new_code
            }
        except Exception as e:
            logger.error(f"Error creating new module {module_name}: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def edit_optimized_module(self, 
                                 module_name: str, 
                                 version: int,
                                 edits_description: str) -> Dict[str, Any]:
        """
        Edit an optimized module
        
        Args:
            module_name: Name of the module to edit
            version: Version of the optimized module
            edits_description: Description of the edits to make
            
        Returns:
            Editing results
        """
        try:
            # Check if module exists
            if module_name not in self.optimized_modules:
                return {
                    "success": False,
                    "error": f"No optimized versions of module {module_name} found"
                }
            
            if version not in self.optimized_modules[module_name]:
                return {
                    "success": False,
                    "error": f"Version {version} of module {module_name} not found"
                }
            
            # Get the current content
            current_content = self.optimized_modules[module_name][version]['content']
            
            # Create prompt for the optimization agent
            prompt = f"""
            Edit the following Python module according to these instructions:
            
            Edits to make: {edits_description}
            
            Current module content:
            ```python
            {current_content}
            ```
            
            Make the requested edits while maintaining:
            1. Code quality and readability
            2. Error handling
            3. Documentation
            4. Type hints
            5. Integration with other components
            
            Your response should be ONLY the complete, edited Python code for the module,
            with no additional explanations or markdown.
            
            Include a comment in the module docstring indicating what edits were made.
            """
            
            # Run the optimization agent
            agent_result = await Runner.run(self.optimization_agent, prompt)
            edited_code = agent_result.final_output
            
            # Clean up the code if needed
            edited_code = self._clean_code_output(edited_code)
            
            # Generate new version number
            new_version = max(self.optimized_modules[module_name].keys()) + 1
            
            # Save the edited module
            file_name = f"{module_name}_v{new_version}.py"
            file_path = os.path.join(self.optimization_dir, file_name)
            
            with open(file_path, 'w') as f:
                f.write(edited_code)
            
            # Update the optimization history
            timestamp = datetime.datetime.now().isoformat()
            
            self.optimized_modules[module_name][new_version] = {
                'content': edited_code,
                'path': file_path,
                'timestamp': timestamp,
                'edited_from_version': version,
                'edits_description': edits_description
            }
            
            self.optimization_history.append({
                'module_name': module_name,
                'version': new_version,
                'previous_version': version,
                'timestamp': timestamp,
                'edits_description': edits_description
            })
            
            return {
                "success": True,
                "module_name": module_name,
                "version": new_version,
                "previous_version": version,
                "path": file_path,
                "edited_code": edited_code
            }
        except Exception as e:
            logger.error(f"Error editing module {module_name} v{version}: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_optimized_module(self, 
                           module_name: str, 
                           version: Optional[int] = None) -> Dict[str, Any]:
        """
        Get an optimized module
        
        Args:
            module_name: Name of the module
            version: Specific version to get, or latest if None
            
        Returns:
            Module data
        """
        try:
            # Check if module exists
            if module_name not in self.optimized_modules:
                return {
                    "success": False,
                    "error": f"No optimized versions of module {module_name} found"
                }
            
            versions = self.optimized_modules[module_name]
            
            if not version:
                version = max(versions.keys())
            
            if version not in versions:
                return {
                    "success": False,
                    "error": f"Version {version} of module {module_name} not found"
                }
            
            module_data = versions[version]
            
            return {
                "success": True,
                "module_name": module_name,
                "version": version,
                "content": module_data['content'],
                "path": module_data['path'],
                "timestamp": module_data['timestamp']
            }
        except Exception as e:
            logger.error(f"Error getting optimized module {module_name}: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def list_optimized_modules(self) -> Dict[str, Any]:
        """
        List all optimized modules
        
        Returns:
            List of optimized modules
        """
        try:
            result = {}
            
            for module_name, versions in self.optimized_modules.items():
                result[module_name] = []
                
                for version, data in sorted(versions.items()):
                    result[module_name].append({
                        "version": version,
                        "timestamp": data['timestamp'],
                        "path": data['path'],
                        "new_module": data.get('new_module', False),
                        "optimization_goal": data.get('optimization_goal', None),
                        "edited_from_version": data.get('edited_from_version', None)
                    })
            
            return {
                "success": True,
                "modules": result,
                "optimization_history": self.optimization_history
            }
        except Exception as e:
            logger.error(f"Error listing optimized modules: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def import_and_test_module(self, 
                                  module_name: str, 
                                  version: int) -> Dict[str, Any]:
        """
        Import and run basic tests on an optimized module
        
        Args:
            module_name: Name of the module
            version: Version to test
            
        Returns:
            Test results
        """
        try:
            # Get the module
            module_info = self.get_optimized_module(module_name, version)
            if not module_info.get("success", False):
                return module_info
            
            # Write to a temporary location with proper Python package structure
            import tempfile
            test_dir = tempfile.mkdtemp()
            
            # Create necessary directories and __init__.py files
            os.makedirs(os.path.join(test_dir, 'nyx', 'core', 'brain'), exist_ok=True)
            
            for init_path in ['nyx/__init__.py', 'nyx/core/__init__.py', 'nyx/core/brain/__init__.py']:
                with open(os.path.join(test_dir, init_path), 'w') as f:
                    f.write('# Generated by module_optimizer')
            
            # Write the module
            with open(os.path.join(test_dir, f'nyx/core/brain/{module_name}.py'), 'w') as f:
                f.write(module_info["content"])
            
            # Add to path
            import sys
            sys.path.insert(0, test_dir)
            
            # Try to import the module
            try:
                importlib.invalidate_caches()
                imported_module = importlib.import_module(f'nyx.core.brain.{module_name}')
                
                # Get module attributes
                classes = [name for name, obj in inspect.getmembers(imported_module, inspect.isclass)]
                functions = [name for name, obj in inspect.getmembers(imported_module, inspect.isfunction)]
                
                # Check for any failures during initialization
                failures = []
                
                # Try to instantiate classes with simple initialization
                instance_tests = []
                for class_name in classes:
                    try:
                        class_obj = getattr(imported_module, class_name)
                        # Check if class takes no arguments or only self
                        try:
                            instance = class_obj()
                            instance_tests.append({
                                "class": class_name,
                                "success": True
                            })
                        except TypeError:
                            # Try with brain parameter if available
                            try:
                                if self.brain:
                                    instance = class_obj(self.brain)
                                    instance_tests.append({
                                        "class": class_name,
                                        "success": True
                                    })
                            except:
                                instance_tests.append({
                                    "class": class_name,
                                    "success": False,
                                    "reason": "Could not instantiate with default or brain parameter"
                                })
                    except Exception as e:
                        failures.append({
                            "item": class_name,
                            "error": str(e)
                        })
                
                result = {
                    "success": True,
                    "module_name": module_name,
                    "version": version,
                    "imported": True,
                    "classes": classes,
                    "functions": functions,
                    "instance_tests": instance_tests,
                    "failures": failures
                }
            except Exception as e:
                result = {
                    "success": False,
                    "module_name": module_name,
                    "version": version,
                    "error": f"Import error: {str(e)}"
                }
            
            # Clean up
            sys.path.remove(test_dir)
            import shutil
            shutil.rmtree(test_dir)
            
            return result
        except Exception as e:
            logger.error(f"Error testing module {module_name} v{version}: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
