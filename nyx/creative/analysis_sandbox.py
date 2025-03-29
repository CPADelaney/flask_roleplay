# nyx/creative/analysis_sandbox.py

import os
import sys
import ast
import importlib
import inspect
import json
import subprocess
import tempfile
import threading
import time
import logging
from typing import Dict, List, Any, Optional, Union, Set
import asyncio
import io
import contextlib
import traceback

logger = logging.getLogger(__name__)

class CodeAnalyzer:
    """
    System for analyzing, reviewing, and understanding code.
    """
    
    def __init__(self, creative_content_system=None):
        """
        Initialize the code analyzer.
        
        Args:
            creative_content_system: Optional system for storing analysis results
        """
        self.creative_content_system = creative_content_system
    
    async def analyze_module(self, module_path: str) -> Dict[str, Any]:
        """
        Analyze a Python module to understand its structure.
        
        Args:
            module_path: Path to the module file
            
        Returns:
            Analysis results
        """
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            # Parse the AST
            tree = ast.parse(code)
            
            # Extract module information
            classes = []
            functions = []
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = []
                    class_attrs = []
                    
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            methods.append({
                                "name": item.name,
                                "args": [arg.arg for arg in item.args.args],
                                "doc": ast.get_docstring(item),
                                "line_number": item.lineno
                            })
                        elif isinstance(item, ast.Assign):
                            for target in item.targets:
                                if isinstance(target, ast.Name):
                                    class_attrs.append({
                                        "name": target.id,
                                        "line_number": item.lineno
                                    })
                    
                    classes.append({
                        "name": node.name,
                        "methods": methods,
                        "attributes": class_attrs,
                        "doc": ast.get_docstring(node),
                        "line_number": node.lineno
                    })
                
                elif isinstance(node, ast.FunctionDef) and node.parent_field != 'body':
                    functions.append({
                        "name": node.name,
                        "args": [arg.arg for arg in node.args.args],
                        "doc": ast.get_docstring(node),
                        "line_number": node.lineno
                    })
                
                elif isinstance(node, ast.Import):
                    for name in node.names:
                        imports.append({
                            "name": name.name,
                            "alias": name.asname,
                            "line_number": node.lineno
                        })
                
                elif isinstance(node, ast.ImportFrom):
                    for name in node.names:
                        imports.append({
                            "name": f"{node.module}.{name.name}",
                            "alias": name.asname,
                            "line_number": node.lineno
                        })
            
            # Basic code metrics
            lines_of_code = len(code.splitlines())
            comment_lines = sum(1 for line in code.splitlines() if line.strip().startswith("#"))
            
            analysis_result = {
                "module_path": module_path,
                "module_name": os.path.basename(module_path),
                "classes": classes,
                "functions": functions,
                "imports": imports,
                "metrics": {
                    "lines_of_code": lines_of_code,
                    "comment_lines": comment_lines,
                    "comment_ratio": comment_lines / lines_of_code if lines_of_code > 0 else 0,
                    "class_count": len(classes),
                    "function_count": len(functions),
                    "import_count": len(imports)
                }
            }
            
            # Store analysis if creative content system is available
            if self.creative_content_system:
                await self.creative_content_system.store_content(
                    content_type="analysis",
                    title=f"Code Analysis: {os.path.basename(module_path)}",
                    content=self._format_analysis_to_markdown(analysis_result),
                    metadata={
                        "module_path": module_path,
                        "analysis_type": "module_structure"
                    }
                )
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing module {module_path}: {e}")
            return {"error": str(e)}
    
    def _format_analysis_to_markdown(self, analysis_result: Dict[str, Any]) -> str:
        """Format analysis results as markdown for better readability."""
        md = f"# Code Analysis: {analysis_result['module_name']}\n\n"
        md += f"**Module Path:** {analysis_result['module_path']}\n\n"
        
        # Add metrics section
        md += "## Metrics\n\n"
        metrics = analysis_result["metrics"]
        md += f"- **Lines of Code:** {metrics['lines_of_code']}\n"
        md += f"- **Comment Lines:** {metrics['comment_lines']} ({metrics['comment_ratio']:.2%})\n"
        md += f"- **Classes:** {metrics['class_count']}\n"
        md += f"- **Functions:** {metrics['function_count']}\n"
        md += f"- **Imports:** {metrics['import_count']}\n\n"
        
        # Add classes section
        if analysis_result["classes"]:
            md += "## Classes\n\n"
            for cls in analysis_result["classes"]:
                md += f"### {cls['name']}\n\n"
                
                if cls["doc"]:
                    md += f"{cls['doc']}\n\n"
                
                if cls["methods"]:
                    md += "**Methods:**\n\n"
                    for method in cls["methods"]:
                        args_str = ", ".join(method["args"])
                        md += f"- `{method['name']}({args_str})` (line {method['line_number']})\n"
                    md += "\n"
                
                if cls["attributes"]:
                    md += "**Attributes:**\n\n"
                    for attr in cls["attributes"]:
                        md += f"- `{attr['name']}` (line {attr['line_number']})\n"
                    md += "\n"
        
        # Add functions section
        if analysis_result["functions"]:
            md += "## Functions\n\n"
            for func in analysis_result["functions"]:
                args_str = ", ".join(func["args"])
                md += f"### `{func['name']}({args_str})`\n\n"
                if func["doc"]:
                    md += f"{func['doc']}\n\n"
                md += f"Defined at line {func['line_number']}\n\n"
        
        # Add imports section
        if analysis_result["imports"]:
            md += "## Imports\n\n"
            for imp in analysis_result["imports"]:
                if imp["alias"]:
                    md += f"- `{imp['name']}` as `{imp['alias']}` (line {imp['line_number']})\n"
                else:
                    md += f"- `{imp['name']}` (line {imp['line_number']})\n"
        
        return md
    
    async def analyze_codebase(self, base_dir: str, extensions: List[str] = None) -> Dict[str, Any]:
        """
        Analyze an entire codebase directory.
        
        Args:
            base_dir: Base directory of the codebase
            extensions: File extensions to include (default: ['.py'])
            
        Returns:
            Analysis results
        """
        if extensions is None:
            extensions = ['.py']
        
        # Find all relevant files
        files = []
        for root, dirs, filenames in os.walk(base_dir):
            for filename in filenames:
                if any(filename.endswith(ext) for ext in extensions):
                    files.append(os.path.join(root, filename))
        
        # Analyze each file
        results = {}
        for file_path in files:
            relative_path = os.path.relpath(file_path, base_dir)
            results[relative_path] = await self.analyze_module(file_path)
        
        # Aggregate metrics
        total_loc = sum(file_result["metrics"]["lines_of_code"] for file_result in results.values())
        total_classes = sum(file_result["metrics"]["class_count"] for file_result in results.values())
        total_functions = sum(file_result["metrics"]["function_count"] for file_result in results.values())
        
        # Create module dependency graph
        dependency_graph = self._create_dependency_graph(results)
        
        # Store aggregated analysis if creative content system is available
        if self.creative_content_system:
            summary_md = f"# Codebase Analysis: {os.path.basename(base_dir)}\n\n"
            summary_md += f"**Base Directory:** {base_dir}\n"
            summary_md += f"**Files Analyzed:** {len(files)}\n"
            summary_md += f"**Total Lines of Code:** {total_loc}\n"
            summary_md += f"**Total Classes:** {total_classes}\n"
            summary_md += f"**Total Functions:** {total_functions}\n\n"
            
            summary_md += "## Module Dependency Graph\n\n"
            for module, deps in dependency_graph.items():
                if deps:
                    summary_md += f"- **{module}** depends on: {', '.join(deps)}\n"
            
            await self.creative_content_system.store_content(
                content_type="analysis",
                title=f"Codebase Analysis: {os.path.basename(base_dir)}",
                content=summary_md,
                metadata={
                    "base_dir": base_dir,
                    "analysis_type": "codebase_structure",
                    "files_analyzed": len(files)
                }
            )
        
        return {
            "base_dir": base_dir,
            "files_analyzed": len(files),
            "file_results": results,
            "aggregated_metrics": {
                "total_lines_of_code": total_loc,
                "total_classes": total_classes,
                "total_functions": total_functions,
                "files_count": len(files)
            },
            "dependency_graph": dependency_graph
        }
    
    def _create_dependency_graph(self, file_results: Dict[str, Any]) -> Dict[str, List[str]]:
        """Create a dependency graph from file analysis results."""
        graph = {}
        
        # Map filenames to module names
        filename_to_module = {}
        for file_path, result in file_results.items():
            module_name = os.path.splitext(os.path.basename(file_path))[0]
            filename_to_module[file_path] = module_name
            graph[module_name] = []
        
        # Populate dependencies
        for file_path, result in file_results.items():
            module_name = filename_to_module[file_path]
            
            for imp in result.get("imports", []):
                import_name = imp["name"].split(".")[0]
                
                # Find corresponding module
                for other_module in graph.keys():
                    if import_name == other_module:
                        graph[module_name].append(other_module)
                        break
        
        return graph
    
    async def review_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """
        Review code for improvements, bugs, and style issues.
        
        Args:
            code: The code to review
            language: The programming language
            
        Returns:
            Review results
        """
        if language.lower() != "python":
            return {"error": f"Language {language} not currently supported for review"}
        
        review_results = {
            "issues": [],
            "suggestions": [],
            "strengths": []
        }
        
        try:
            # Parse the AST
            tree = ast.parse(code)
            
            # Look for common issues
            for node in ast.walk(tree):
                # Check for except without specific exceptions
                if isinstance(node, ast.ExceptHandler) and node.type is None:
                    review_results["issues"].append({
                        "type": "bare_except",
                        "message": "Using bare 'except:' without specifying exceptions",
                        "line": node.lineno
                    })
                
                # Check for mutable default arguments
                if isinstance(node, ast.FunctionDef):
                    for arg_idx, default in enumerate(node.args.defaults):
                        if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                            arg_name = node.args.args[len(node.args.args) - len(node.args.defaults) + arg_idx].arg
                            review_results["issues"].append({
                                "type": "mutable_default",
                                "message": f"Using mutable default argument for '{arg_name}'",
                                "line": node.lineno
                            })
            
            # Simple code style checks
            lines = code.splitlines()
            for i, line in enumerate(lines):
                # Check line length
                if len(line) > 100:
                    review_results["issues"].append({
                        "type": "line_length",
                        "message": f"Line exceeds 100 characters ({len(line)})",
                        "line": i + 1
                    })
                
                # Check for TODO comments
                if "TODO" in line or "FIXME" in line:
                    review_results["issues"].append({
                        "type": "todo",
                        "message": f"TODO/FIXME comment: '{line.strip()}'",
                        "line": i + 1
                    })
            
            # Add strengths based on the analysis
            docstring_count = sum(1 for node in ast.walk(tree) 
                               if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)) 
                               and ast.get_docstring(node))
            
            function_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
            
            if docstring_count / max(1, function_count) > 0.8:
                review_results["strengths"].append({
                    "type": "documentation",
                    "message": "Good docstring coverage"
                })
            
            if len(review_results["issues"]) == 0:
                review_results["strengths"].append({
                    "type": "clean_code",
                    "message": "No common issues detected"
                })
            
            # Add suggestions for improvements
            for issue in review_results["issues"]:
                if issue["type"] == "bare_except":
                    review_results["suggestions"].append({
                        "type": "specify_exceptions",
                        "message": "Specify exceptions to catch (e.g., 'except ValueError:' instead of 'except:')",
                        "related_issue": issue
                    })
                elif issue["type"] == "mutable_default":
                    review_results["suggestions"].append({
                        "type": "use_none_default",
                        "message": "Use None as default and initialize the mutable value in the function body",
                        "related_issue": issue
                    })
            
            # Store review if creative content system is available
            if self.creative_content_system:
                review_md = self._format_review_to_markdown(review_results, code)
                
                await self.creative_content_system.store_content(
                    content_type="analysis",
                    title=f"Code Review: {self._get_code_title(code)}",
                    content=review_md,
                    metadata={
                        "language": language,
                        "analysis_type": "code_review",
                        "issues_count": len(review_results["issues"])
                    }
                )
            
            return review_results
        
        except Exception as e:
            logger.error(f"Error reviewing code: {e}")
            return {"error": str(e)}
    
    def _get_code_title(self, code: str) -> str:
        """Extract a title from code."""
        lines = code.splitlines()
        for line in lines:
            if line.startswith("# "):
                return line[2:].strip()
            elif line.startswith("class "):
                return f"Class {line[6:].split('(')[0].split(':')[0].strip()}"
            elif line.startswith("def "):
                return f"Function {line[4:].split('(')[0].strip()}"
        return "Untitled Code"
    
    def _format_review_to_markdown(self, review_results: Dict[str, Any], code: str) -> str:
        """Format code review results as markdown."""
        md = f"# Code Review: {self._get_code_title(code)}\n\n"
        
        # Add summary section
        issue_count = len(review_results["issues"])
        suggestion_count = len(review_results["suggestions"])
        strength_count = len(review_results["strengths"])
        
        md += "## Summary\n\n"
        md += f"- **Issues Found:** {issue_count}\n"
        md += f"- **Suggestions:** {suggestion_count}\n"
        md += f"- **Strengths:** {strength_count}\n\n"
        
        # Add issues section
        if issue_count > 0:
            md += "## Issues\n\n"
            for issue in review_results["issues"]:
                md += f"- **Line {issue['line']}:** {issue['message']}\n"
            md += "\n"
        
        # Add suggestions section
        if suggestion_count > 0:
            md += "## Suggestions\n\n"
            for suggestion in review_results["suggestions"]:
                if "related_issue" in suggestion:
                    related = suggestion["related_issue"]
                    md += f"- **For issue on line {related['line']}:** {suggestion['message']}\n"
                else:
                    md += f"- {suggestion['message']}\n"
            md += "\n"
        
        # Add strengths section
        if strength_count > 0:
            md += "## Strengths\n\n"
            for strength in review_results["strengths"]:
                md += f"- {strength['message']}\n"
            md += "\n"
        
        # Add code section
        md += "## Code Reviewed\n\n"
        md += "```python\n"
        md += code
        md += "\n```\n"
        
        return md


class SandboxExecutor:
    """
    Secure sandbox for executing AI-generated code in a controlled environment.
    """
    
    def __init__(self, 
                creative_content_system=None,
                max_execution_time=10,  # 10 seconds max
                max_memory=50 * 1024 * 1024):  # 50 MB max
        """
        Initialize the sandbox executor.
        
        Args:
            creative_content_system: Optional system for storing execution results
            max_execution_time: Maximum execution time in seconds
            max_memory: Maximum memory usage in bytes
        """
        self.creative_content_system = creative_content_system
        self.max_execution_time = max_execution_time
        self.max_memory = max_memory
    
    async def execute_code(self, 
                       code: str, 
                       language: str = "python",
                       save_output: bool = True) -> Dict[str, Any]:
        """
        Execute code in a sandbox environment.
        
        Args:
            code: Code to execute
            language: Programming language
            save_output: Whether to save output to the content system
            
        Returns:
            Execution results
        """
        if language.lower() != "python":
            return {"error": f"Language {language} not currently supported for sandbox execution"}
        
        # Create a temporary file for the code
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as tmp:
            tmp_name = tmp.name
            tmp.write(code.encode('utf-8'))
        
        try:
            # Capture output
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            
            # Set up execution environment
            env = os.environ.copy()
            
            # Add resource limits
            timeout_happened = False
            start_time = time.time()
            
            def timeout_handler():
                nonlocal timeout_happened
                timeout_happened = True
            
            # Execute the code in a separate process
            process = subprocess.Popen(
                [sys.executable, tmp_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True
            )
            
            # Set up timer for timeout
            timer = threading.Timer(self.max_execution_time, timeout_handler)
            timer.start()
            
            try:
                stdout, stderr = process.communicate()
            finally:
                timer.cancel()
            
            execution_time = time.time() - start_time
            exit_code = process.returncode
            
            # Prepare result
            result = {
                "success": exit_code == 0 and not timeout_happened,
                "execution_time": execution_time,
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": exit_code
            }
            
            if timeout_happened:
                result["error"] = f"Execution timed out after {self.max_execution_time} seconds"
            
            # Store execution results if requested
            if save_output and self.creative_content_system:
                content_title = f"Code Execution: {self._get_code_title(code)}"
                
                # Format as markdown
                md_content = self._format_execution_to_markdown(code, result)
                
                await self.creative_content_system.store_content(
                    content_type="code",
                    title=content_title,
                    content=md_content,
                    metadata={
                        "language": language,
                        "execution_time": execution_time,
                        "success": result["success"]
                    }
                )
            
            return result
        
        except Exception as e:
            logger.error(f"Error executing code: {e}")
            return {"error": str(e), "success": False}
        
        finally:
            # Clean up the temporary file
            try:
                os.unlink(tmp_name)
            except:
                pass
    
    def _get_code_title(self, code: str) -> str:
        """Extract a title from code."""
        lines = code.splitlines()
        for line in lines:
            if line.startswith("# "):
                return line[2:].strip()
            elif line.startswith("class "):
                return f"Class {line[6:].split('(')[0].split(':')[0].strip()}"
            elif line.startswith("def "):
                return f"Function {line[4:].split('(')[0].strip()}"
        return "Untitled Code"
    
    def _format_execution_to_markdown(self, code: str, result: Dict[str, Any]) -> str:
        """Format execution results as markdown."""
        md = f"# Code Execution: {self._get_code_title(code)}\n\n"
        
        # Add execution summary
        md += "## Execution Summary\n\n"
        md += f"- **Success:** {result['success']}\n"
        md += f"- **Execution Time:** {result['execution_time']:.3f} seconds\n"
        md += f"- **Exit Code:** {result['exit_code']}\n\n"
        
        # Add stdout section
        md += "## Standard Output\n\n"
        md += "```\n"
        md += result['stdout'] if result['stdout'] else "(No output)"
        md += "\n```\n\n"
        
        # Add stderr section if there's any error
        if result['stderr']:
            md += "## Standard Error\n\n"
            md += "```\n"
            md += result['stderr']
            md += "\n```\n\n"
        
        # Add error section if there's a specific error message
        if "error" in result:
            md += "## Error\n\n"
            md += f"{result['error']}\n\n"
        
        # Add code section
        md += "## Code Executed\n\n"
        md += "```python\n"
        md += code
        md += "\n```\n"
        
        return md
