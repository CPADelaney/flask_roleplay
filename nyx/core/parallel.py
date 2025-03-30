# nyx/core/parallel.py

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from agents import RunContextWrapper

logger = logging.getLogger(__name__)

@dataclass
class ToolExecutionResult:
    """Result of a tool execution."""
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None

class ParallelToolExecutor:
    """Executes tools in parallel."""
    
    def __init__(self, max_concurrent: int = 5):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def execute_tool(self, tool_info: Dict[str, Any]) -> ToolExecutionResult:
        """Execute a single tool with error handling."""
        start_time = time.time()
        tool = tool_info["tool"]
        args = tool_info.get("args", {})
        metadata = tool_info.get("metadata", {})
        tool_name = tool.__name__ if hasattr(tool, "__name__") else str(tool)
        
        try:
            async with self.semaphore:
                # Create dummy context if needed
                if "ctx" not in args:
                    args["ctx"] = RunContextWrapper(context=None)
                
                # Execute tool
                result = await tool(**args)
                
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
            logger.error(f"Error executing tool {tool_name}: {e}")
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=str(e),
                execution_time=execution_time,
                metadata=metadata
            )
    
    async def execute_tools(self, tools_info: List[Dict[str, Any]]) -> List[ToolExecutionResult]:
        """Execute multiple tools in parallel."""
        tasks = [self.execute_tool(tool_info) for tool_info in tools_info]
        results = await asyncio.gather(*tasks)
        return results
