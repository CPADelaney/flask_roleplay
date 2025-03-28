import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Callable, TypeVar, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

T = TypeVar('T')
ToolFunction = Callable[..., Any]

@dataclass
class ToolExecutionResult:
    """Result of a tool execution."""
    tool_name: str
    success: bool
    result: Any
    execution_time: float
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ParallelToolExecutor:
    """Executes compatible tools in parallel for improved performance."""
    
    def __init__(self, max_concurrent=5):
        """
        Initialize the parallel tool executor.
        
        Args:
            max_concurrent: Maximum number of tools to execute concurrently
        """
        self.max_concurrent = max_concurrent
        self.execution_history = []
        
    async def execute_tools(self, tools_info: List[Dict[str, Any]], context: Any = None) -> List[ToolExecutionResult]:
        """
        Execute multiple tools in parallel.
        
        Args:
            tools_info: List of tool information dictionaries, each containing 'tool' and 'args'
            context: Optional context to pass to the tools
            
        Returns:
            List of tool execution results
        """
        start_time = time.time()
        logger.info(f"Starting parallel execution of {len(tools_info)} tools")
        
        # Execute tools in parallel with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def _bounded_execute(tool_info):
            async with semaphore:
                return await self._execute_single_tool(tool_info, context)
        
        # Create tasks for all tools
        tasks = [_bounded_execute(tool_info) for tool_info in tools_info]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        
        # Record execution metadata
        execution_time = time.time() - start_time
        execution_record = {
            "timestamp": time.time(),
            "tool_count": len(tools_info),
            "total_execution_time": execution_time,
            "successful_tools": sum(1 for r in results if r.success),
            "failed_tools": sum(1 for r in results if not r.success)
        }
        self.execution_history.append(execution_record)
        
        logger.info(f"Completed parallel execution in {execution_time:.3f}s. Success: {execution_record['successful_tools']}, Failed: {execution_record['failed_tools']}")
        return results
    
    async def _execute_single_tool(self, tool_info: Dict[str, Any], context: Any) -> ToolExecutionResult:
        """
        Execute a single tool and handle any exceptions.
        
        Args:
            tool_info: Dictionary containing the tool and arguments
            context: Optional context to pass to the tool
            
        Returns:
            ToolExecutionResult containing success status and result or error
        """
        tool = tool_info["tool"]
        args = tool_info["args"]
        tool_name = getattr(tool, "name", str(tool))
        metadata = tool_info.get("metadata", {})
        
        start_time = time.time()
        logger.debug(f"Executing tool: {tool_name}")
        
        try:
            # Handle different types of tools
            if hasattr(tool, "execute"):
                # Tool with an execute method
                if context is not None:
                    result = await tool.execute(args, context)
                else:
                    result = await tool.execute(args)
            elif hasattr(tool, "on_invoke_tool"):
                # Function tool from Agent SDK
                result = await tool.on_invoke_tool(context, args)
            elif callable(tool):
                # Direct callable function
                if context is not None:
                    result = await tool(context, **args) if asyncio.iscoroutinefunction(tool) else tool(context, **args)
                else:
                    result = await tool(**args) if asyncio.iscoroutinefunction(tool) else tool(**args)
            else:
                raise ValueError(f"Unsupported tool type: {type(tool)}")
            
            execution_time = time.time() - start_time
            return ToolExecutionResult(
                tool_name=tool_name,
                success=True,
                result=result,
                execution_time=execution_time,
                metadata=metadata
            )
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error executing tool {tool_name}: {str(e)}")
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=str(e),
                execution_time=execution_time,
                metadata=metadata
            )
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get statistics about tool executions."""
        if not self.execution_history:
            return {"executions": 0}
        
        total_executions = len(self.execution_history)
        total_tools = sum(exec["tool_count"] for exec in self.execution_history)
        total_successful = sum(exec["successful_tools"] for exec in self.execution_history)
        total_failed = sum(exec["failed_tools"] for exec in self.execution_history)
        avg_execution_time = sum(exec["total_execution_time"] for exec in self.execution_history) / total_executions
        
        return {
            "executions": total_executions,
            "total_tools": total_tools,
            "successful_tools": total_successful,
            "failed_tools": total_failed,
            "success_rate": total_successful / total_tools if total_tools > 0 else 0,
            "average_execution_time": avg_execution_time
        }

# Usage example with Agent SDK
async def tool_execution_example():
    # Create a parallel tool executor
    executor = ParallelToolExecutor(max_concurrent=3)
    
    # Define some example tools
    async def get_user_data(user_id: str) -> Dict[str, Any]:
        await asyncio.sleep(1)  # Simulate API call
        return {"id": user_id, "name": "Test User", "age": 30}
    
    async def get_weather(location: str) -> Dict[str, Any]:
        await asyncio.sleep(0.5)  # Simulate API call
        return {"location": location, "temperature": 72, "conditions": "sunny"}
    
    async def get_recommendations(user_id: str, category: str) -> List[str]:
        await asyncio.sleep(1.5)  # Simulate API call
        return ["Item 1", "Item 2", "Item 3"]
    
    # Set up tools info
    tools_info = [
        {"tool": get_user_data, "args": {"user_id": "user123"}, "metadata": {"purpose": "profile"}},
        {"tool": get_weather, "args": {"location": "San Francisco"}, "metadata": {"purpose": "forecast"}},
        {"tool": get_recommendations, "args": {"user_id": "user123", "category": "books"}, "metadata": {"purpose": "recommendations"}}
    ]
    
    # Execute tools in parallel
    results = await executor.execute_tools(tools_info)
    
    # Print results
    for result in results:
        print(f"Tool: {result.tool_name}")
        print(f"Success: {result.success}")
        print(f"Execution time: {result.execution_time:.3f}s")
        print(f"Result: {result.result}")
        print()
    
    # Print stats
    print("Execution stats:", executor.get_execution_stats())
