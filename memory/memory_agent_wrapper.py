# memory/memory_agent_wrapper.py

from __future__ import annotations
"""
Refactored MemoryAgentWrapper (v2.0)
------------------------------------
* Removes unsupported `metadata` field from the message payload that is
  sent to the OpenAI Responses API.
* Encodes all operation‑specific arguments into the `content` field as
  JSON so the underlying agent can parse structured inputs.
* Optionally forwards the same dict to `trace_metadata` so it still
  appears in OpenAI Agents trace UIs.
* Centralises message construction in `_build_input()` to avoid
  repetition across the various wrapper methods.
* Retains the public API (`recall`, `remember`, `create_belief`, etc.)
  so callers do not need to change.
"""

from typing import Dict, List, Any, Optional
import json
import logging

from agents import Agent, Runner, RunConfig, RunContextWrapper  # type: ignore

logger = logging.getLogger(__name__)


class MemoryAgentWrapper:
    """Compatibility layer between *your* memory system and the OpenAI
    Agents SDK.

    The wrapper simply serialises a structured payload describing the
    requested operation into the *content* of a single chat message.
    That message is passed to :pyfunc:`agents.Runner.run`.  The wrapped
    *agent* is responsible for parsing ``json.loads(message.content)``
    and producing an appropriate response.
    """

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def __init__(self, agent: Agent, context: Any | None = None):
        self.agent = agent
        self.context = context
    
        # Expose a subset of the agent's attributes
        self.handoffs = getattr(agent, "handoffs", [])
        self.output_type = getattr(agent, "output_type", None)
        self.name = getattr(agent, "name", "MemoryAgent")
        self.instructions = getattr(agent, "instructions", "")
        self.tools = getattr(agent, "tools", [])
        self.input_guardrails = getattr(agent, "input_guardrails", [])
        self.output_guardrails = getattr(agent, "output_guardrails", [])  # Add this line
        self._hooks = None
        self.model_settings = getattr(agent, "model_settings", None)

    # Public proxies ----------------------------------------------------

    @property
    def hooks(self):
        return getattr(self.agent, "hooks", None)

    @property
    def model(self):
        return getattr(self.agent, "model", None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_input(self, role: str, operation: str, **fields: Any) -> Dict[str, Any]:
        """Serialise ``operation`` + ``fields`` into a chat‑message."""
        content = json.dumps({"operation": operation, **fields}, ensure_ascii=False)
        return {"role": role, "content": content}

    async def _run(self, input_msg: Dict[str, Any], *, trace_meta: dict[str, Any] | None = None):
        """Thin wrapper around :pyfunc:`Runner.run` that sets trace data."""
        # Convert all numeric values to strings in trace_meta
        if trace_meta:
            for key, value in list(trace_meta.items()):
                if isinstance(value, (int, float)):
                    trace_meta[key] = str(value)
        
        return await Runner.run(
            self.agent, [input_msg], context=self.context,
            run_config=RunConfig(trace_metadata=trace_meta or {})
        )
        
        return await Runner.run(
            self.agent, [input_msg], context=self.context,
            run_config=RunConfig(trace_metadata=trace_meta or {})
        )

    # ------------------------------------------------------------------
    # Public API – Memory operations
    # ------------------------------------------------------------------
    
    async def recall(
        self,
        context: Any,
        entity_type: str,
        entity_id: int,
        query: str | None = None,
        context_text: str | None = None,
        limit: int = 5,
    ) -> Dict[str, Any]:
        try:
            meta = {
                "entity_type": entity_type,
                "entity_id": entity_id,
                "query": query,
                "context": context_text,
                "limit": limit,
            }
            result = await self._run(self._build_input("user", "recall", **meta), trace_meta=meta)
            return _coerce_to_dict(result.final_output)
        except Exception as e:
            logger.error("Error in recall: %s", e)
            return {"error": str(e), "memories": []}
          
    async def remember(
        self,
        context: Any,
        entity_type: str,
        entity_id: int,
        memory_text: str,
        importance: str = "medium",
        emotional: bool = True,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        try:
            meta = {
                "entity_type": entity_type,
                "entity_id": entity_id,
                "memory_text": memory_text,
                "importance": importance,  # These will now be required by the function
                "emotional": emotional,    # Make sure to pass them all
                "tags": tags or [],
            }
            result = await self._run(self._build_input("user", "remember", **meta), trace_meta=meta)
            return _coerce_to_dict(result.final_output)
        except Exception as e:
            logger.error("Error in remember: %s", e)
            return {"error": str(e), "memory_id": None}

    async def create_belief(
        self,
        context: Any,
        entity_type: str,
        entity_id: int,
        belief_text: str,
        confidence: float = 0.7,
    ) -> Dict[str, Any]:
        try:
            meta = {
                "entity_type": entity_type,
                "entity_id": entity_id,
                "belief_text": belief_text,
                "confidence": confidence,
            }
            result = await self._run(self._build_input("user", "create_belief", **meta), trace_meta=meta)
            return _coerce_to_dict(result.final_output)
        except Exception as e:
            logger.error("Error in create_belief: %s", e)
            return {"error": str(e), "belief_id": None}

    async def get_beliefs(
        self,
        context: Any,
        entity_type: str,
        entity_id: int,
        topic: str | None = None,
    ) -> List[Dict[str, Any]]:
        try:
            meta = {
                "entity_type": entity_type,
                "entity_id": entity_id,
                "topic": topic,
            }
            result = await self._run(self._build_input("user", "get_beliefs", **meta), trace_meta=meta)
            parsed = _coerce_to_dict(result.final_output)
            if isinstance(parsed, list):
                return parsed
            return parsed.get("beliefs", []) if isinstance(parsed, dict) else []
        except Exception as e:
            logger.error("Error in get_beliefs: %s", e)
            return []

    async def run_maintenance(self, context: Any, entity_type: str, entity_id: int) -> Dict[str, Any]:
        try:
            meta = {"entity_type": entity_type, "entity_id": entity_id}
            result = await self._run(self._build_input("user", "run_maintenance", **meta), trace_meta=meta)
            return _coerce_to_dict(result.final_output)
        except Exception as e:
            logger.error("Error in run_maintenance: %s", e)
            return {"error": str(e), "success": False}

    async def analyze_memories(self, context: Any, entity_type: str, entity_id: int) -> Dict[str, Any]:
        try:
            meta = {"entity_type": entity_type, "entity_id": entity_id}
            result = await self._run(self._build_input("user", "analyze_memories", **meta), trace_meta=meta)
            return _coerce_to_dict(result.final_output)
        except Exception as e:
            logger.error("Error in analyze_memories: %s", e)
            return {"error": str(e), "analysis": None}

    async def generate_schemas(self, context: Any, entity_type: str, entity_id: int) -> Dict[str, Any]:
        try:
            meta = {"entity_type": entity_type, "entity_id": entity_id}
            result = await self._run(self._build_input("user", "generate_schemas", **meta), trace_meta=meta)
            return _coerce_to_dict(result.final_output)
        except Exception as e:
            logger.error("Error in generate_schemas: %s", e)
            return {"error": str(e), "schemas": []}

    # ------------------------------------------------------------------
    # Utility / passthrough helpers
    # ------------------------------------------------------------------

    async def get_system_prompt(self, run_context: RunContextWrapper):  # noqa: D401
        if hasattr(self.agent, "get_system_prompt"):
            return await self.agent.get_system_prompt(run_context)  # type: ignore[arg-type]
        if callable(self.instructions):
            return await self.instructions(run_context, self.agent)  # type: ignore[misc]
        return "You are a memory management assistant that helps manage and retrieve memories."

    def get_tools(self):
        """Get the tools and ensure all ctx parameters have proper type annotations."""
        try:
            tools = self.agent.tools  # Get tools directly from the agent
            for t in tools:
                if "parameters" in t and "properties" in t["parameters"] and "ctx" in t["parameters"]["properties"]:
                    # Ensure the ctx parameter has a type
                    if "type" not in t["parameters"]["properties"]["ctx"]:
                        t["parameters"]["properties"]["ctx"] = {"type": "object"}
            return tools
        except Exception as e:
            logger.error(f"Error in get_tools: {e}")
            return self.agent.tools if hasattr(self.agent, "tools") else []

    def run(self, *args, **kwargs):
        if hasattr(self.agent, "run"):
            return self.agent.run(*args, **kwargs)
        raise NotImplementedError("Run method not implemented by underlying agent")

    def get_name(self):
        return self.agent.get_name() if hasattr(self.agent, "get_name") else self.name


# ----------------------------------------------------------------------
# Helper functions (module‑level)
# ----------------------------------------------------------------------

def _coerce_to_dict(obj: Any) -> Dict[str, Any]:
    """Return *obj* as a dict; if it's a JSON string, parse it."""
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, str):
        try:
            parsed = json.loads(obj)
            return parsed if isinstance(parsed, dict) else {"data": parsed}
        except json.JSONDecodeError:
            return {"message": obj}
    # Fallback – wrap any other type so callers can see *something*
    return {"data": obj}
