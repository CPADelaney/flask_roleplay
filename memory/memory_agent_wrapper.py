# memory/memory_agent_wrapper.py

from __future__ import annotations

"""
Refactored MemoryAgentWrapper (v2.1)
------------------------------------
- Fast path: calls MemorySystem directly; falls back to LLM/agent path.
- Backward-compatible arg names (supports both `context` and `context_text`).
- Centralized message construction and trace metadata sanitization.
- Safer passthroughs to underlying Agent; no extra metadata field leaks.
"""

from typing import Dict, List, Any, Optional
import json
import logging

from agents import Agent, Runner, RunConfig, RunContextWrapper  # type: ignore

logger = logging.getLogger(__name__)


class MemoryAgentWrapper:
    """
    Compatibility layer between your memory system and the OpenAI Agents SDK.

    Public API:
      - recall(...)
      - remember(...)
      - create_belief(...)
      - get_beliefs(...)
      - run_maintenance(...)
      - analyze_memories(...)
      - generate_schemas(...)

    Behavior:
      1) Tries to execute against MemorySystem directly (fast path).
      2) If unavailable or failing, falls back to LLM/agent via Runner.run.
    """

    def __init__(self, agent: Agent, context: Any | None = None):
        self.agent = agent
        self.context = context

        # Expose a subset of the agent's attributes for transparency
        self.handoffs = getattr(agent, "handoffs", [])
        self.output_type = getattr(agent, "output_type", None)
        self.name = getattr(agent, "name", "MemoryAgent")
        self.instructions = getattr(agent, "instructions", "")
        self.tools = getattr(agent, "tools", [])
        self.input_guardrails = getattr(agent, "input_guardrails", [])
        self.output_guardrails = getattr(agent, "output_guardrails", [])
        self.model_settings = getattr(agent, "model_settings", None)

    # ----------------------------- Properties -----------------------------

    @property
    def hooks(self):
        return getattr(self.agent, "hooks", None)

    @property
    def model(self):
        return getattr(self.agent, "model", None)

    # ---------------------------- Internal I/O ----------------------------

    def _build_input(self, role: str, operation: str, **fields: Any) -> Dict[str, Any]:
        """Serialize operation + fields into a single chat message."""
        content = json.dumps({"operation": operation, **fields}, ensure_ascii=False)
        return {"role": role, "content": content}

    def _sanitize_trace_meta(self, meta: Dict[str, Any] | None) -> Dict[str, str]:
        if not meta:
            return {}
        out: Dict[str, str] = {}
        for k, v in meta.items():
            if isinstance(v, (int, float, bool)):
                out[k] = str(v)
            elif v is None:
                out[k] = ""
            else:
                out[k] = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        return out

    async def _run(self, input_msg: Dict[str, Any], *, trace_meta: dict[str, Any] | None = None):
        """Thin wrapper around Runner.run that sets trace metadata."""
        return await Runner.run(
            self.agent,
            [input_msg],
            context=self.context,
            run_config=RunConfig(trace_metadata=self._sanitize_trace_meta(trace_meta)),
        )

    async def _ensure_memory_system(self):
        """
        Ensure self.context.memory_system exists; attempt to lazy-init if possible.
        Returns the memory system or None.
        """
        ms = getattr(self.context, "memory_system", None)
        if ms is not None:
            return ms
        try:
            # Lazy import to avoid circulars
            from memory.wrapper import MemorySystem  # type: ignore
            user_id = getattr(self.context, "user_id", None)
            conversation_id = getattr(self.context, "conversation_id", None)
            if user_id is None or conversation_id is None:
                return None
            ms = await MemorySystem.get_instance(user_id, conversation_id)
            self.context.memory_system = ms
            return ms
        except Exception as e:
            logger.debug("Could not initialize MemorySystem in wrapper: %s", e)
            return None

    # --------------------------- Public Operations ------------------------

    async def recall(
        self,
        run_context: Any,
        *,
        entity_type: str,
        entity_id: int,
        query: str | None = None,
        context: str | None = None,          # preferred name
        context_text: str | None = None,     # backward-compat
        limit: int = 5,
    ) -> Dict[str, Any]:
        ctx_text = context if context is not None else context_text
        # --- Fast path: direct call to memory system ---
        try:
            ms = await self._ensure_memory_system()
            if ms is not None:
                raw = await ms.recall(
                    entity_type=entity_type,
                    entity_id=entity_id,
                    query=query,
                    context=ctx_text,
                    limit=limit,
                )
                return {
                    "memories": _format_memories(raw.get("memories", [])),
                    "count": len(raw.get("memories", [])),
                    **({"flashback": raw.get("flashback")} if "flashback" in raw else {}),
                    **({"mood_influenced": True, "current_emotion": raw.get("current_emotion", {}).get("primary", "neutral")}
                       if raw.get("mood_congruent_recall") else {}),
                }
        except Exception as e:
            logger.warning("Direct recall failed, falling back to agent: %s", e)

        # --- Fallback: route via agent/LLM ---
        try:
            meta = {
                "entity_type": entity_type,
                "entity_id": entity_id,
                "query": query,
                "context": ctx_text,
                "limit": limit,
            }
            result = await self._run(self._build_input("user", "recall", **meta), trace_meta=meta)
            parsed = _coerce_to_dict(result.final_output)
            if "memories" in parsed:
                # normalize shape
                parsed["memories"] = _format_memories(parsed.get("memories", []))
                parsed.setdefault("count", len(parsed["memories"]))
            return parsed
        except Exception as e:
            logger.error("Error in recall (agent path): %s", e)
            return {"error": str(e), "memories": [], "count": 0}

    async def remember(
        self,
        run_context: Any,
        *,
        entity_type: str,
        entity_id: int,
        memory_text: str,
        importance: str = "medium",
        emotional: bool = True,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        # --- Fast path ---
        try:
            ms = await self._ensure_memory_system()
            if ms is not None:
                raw = await ms.remember(
                    entity_type=entity_type,
                    entity_id=entity_id,
                    memory_text=memory_text,
                    importance=importance,
                    emotional=emotional,
                    tags=tags,
                )
                out = {
                    "memory_id": raw.get("memory_id"),
                    "memory_text": memory_text,
                    "importance": importance,
                }
                emo = raw.get("emotion_analysis")
                if isinstance(emo, dict):
                    out["emotional_analysis"] = {
                        "primary_emotion": emo.get("primary_emotion"),
                        "intensity": emo.get("intensity"),
                        "valence": emo.get("valence"),
                    }
                return out
        except Exception as e:
            logger.warning("Direct remember failed, falling back to agent: %s", e)

        # --- Fallback: agent/LLM ---
        try:
            meta = {
                "entity_type": entity_type,
                "entity_id": entity_id,
                "memory_text": memory_text,
                "importance": importance,
                "emotional": emotional,
                "tags": tags or [],
            }
            result = await self._run(self._build_input("user", "remember", **meta), trace_meta=meta)
            return _coerce_to_dict(result.final_output)
        except Exception as e:
            logger.error("Error in remember (agent path): %s", e)
            return {"error": str(e), "memory_id": None}

    async def create_belief(
        self,
        run_context: Any,
        *,
        entity_type: str,
        entity_id: int,
        belief_text: str,
        confidence: float = 0.7,
    ) -> Dict[str, Any]:
        # --- Fast path ---
        try:
            ms = await self._ensure_memory_system()
            if ms is not None:
                raw = await ms.create_belief(
                    entity_type=entity_type,
                    entity_id=entity_id,
                    belief_text=belief_text,
                    confidence=confidence,
                )
                return {
                    "belief_id": raw.get("belief_id"),
                    "belief_text": raw.get("belief_text", belief_text),
                    "confidence": raw.get("confidence", confidence),
                }
        except Exception as e:
            logger.warning("Direct create_belief failed, falling back to agent: %s", e)

        # --- Fallback ---
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
            logger.error("Error in create_belief (agent path): %s", e)
            return {"error": str(e), "belief_id": None}

    async def get_beliefs(
        self,
        run_context: Any,
        *,
        entity_type: str,
        entity_id: int,
        topic: str | None = None,
    ) -> List[Dict[str, Any]]:
        # --- Fast path ---
        try:
            ms = await self._ensure_memory_system()
            if ms is not None:
                raw = await ms.get_beliefs(entity_type=entity_type, entity_id=entity_id, topic=topic)
                return raw if isinstance(raw, list) else []
        except Exception as e:
            logger.warning("Direct get_beliefs failed, falling back to agent: %s", e)

        # --- Fallback ---
        try:
            meta = {"entity_type": entity_type, "entity_id": entity_id, "topic": topic}
            result = await self._run(self._build_input("user", "get_beliefs", **meta), trace_meta=meta)
            parsed = _coerce_to_dict(result.final_output)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return parsed.get("beliefs", [])
            return []
        except Exception as e:
            logger.error("Error in get_beliefs (agent path): %s", e)
            return []

    async def run_maintenance(self, run_context: Any, *, entity_type: str, entity_id: int) -> Dict[str, Any]:
        # --- Fast path ---
        try:
            ms = await self._ensure_memory_system()
            if ms is not None:
                return await ms.maintain(entity_type=entity_type, entity_id=entity_id)
        except Exception as e:
            logger.warning("Direct run_maintenance failed, falling back to agent: %s", e)

        # --- Fallback ---
        try:
            meta = {"entity_type": entity_type, "entity_id": entity_id}
            result = await self._run(self._build_input("user", "run_maintenance", **meta), trace_meta=meta)
            return _coerce_to_dict(result.final_output)
        except Exception as e:
            logger.error("Error in run_maintenance (agent path): %s", e)
            return {"error": str(e), "success": False}

    async def analyze_memories(self, run_context: Any, *, entity_type: str, entity_id: int) -> Dict[str, Any]:
        # --- Fast path ---
        try:
            ms = await self._ensure_memory_system()
            if ms is not None:
                return await ms.analyze_entity_memory(entity_type=entity_type, entity_id=entity_id)
        except Exception as e:
            logger.warning("Direct analyze_memories failed, falling back to agent: %s", e)

        # --- Fallback ---
        try:
            meta = {"entity_type": entity_type, "entity_id": entity_id}
            result = await self._run(self._build_input("user", "analyze_memories", **meta), trace_meta=meta)
            return _coerce_to_dict(result.final_output)
        except Exception as e:
            logger.error("Error in analyze_memories (agent path): %s", e)
            return {"error": str(e), "analysis": None}

    async def generate_schemas(self, run_context: Any, *, entity_type: str, entity_id: int) -> Dict[str, Any]:
        # --- Fast path ---
        try:
            ms = await self._ensure_memory_system()
            if ms is not None:
                return await ms.generate_schemas(entity_type=entity_type, entity_id=entity_id)
        except Exception as e:
            logger.warning("Direct generate_schemas failed, falling back to agent: %s", e)

        # --- Fallback ---
        try:
            meta = {"entity_type": entity_type, "entity_id": entity_id}
            result = await self._run(self._build_input("user", "generate_schemas", **meta), trace_meta=meta)
            return _coerce_to_dict(result.final_output)
        except Exception as e:
            logger.error("Error in generate_schemas (agent path): %s", e)
            return {"error": str(e), "schemas": []}

    # ----------------------- Utility / Passthroughs -----------------------

    async def get_system_prompt(self, run_context: RunContextWrapper):  # noqa: D401
        if hasattr(self.agent, "get_system_prompt"):
            return await self.agent.get_system_prompt(run_context)  # type: ignore[arg-type]
        if callable(self.instructions):
            return await self.instructions(run_context, self.agent)  # type: ignore[misc]
        return "You are a memory management assistant that helps manage and retrieve memories."

    def get_prompt(self, *args, **kwargs):
        """Get the prompt from the underlying agent."""
        if hasattr(self.agent, "get_prompt") and callable(self.agent.get_prompt):
            return self.agent.get_prompt(*args, **kwargs)
        if hasattr(self, "get_system_prompt"):
            # Synchronous fallback string (async prompt not available here)
            return "You are a memory management assistant that helps manage and retrieve memories."
        if hasattr(self.agent, "instructions"):
            return self.agent.instructions
        return "You are a memory management assistant that helps manage and retrieve memories."

    def get_tools(self, *args, **kwargs):
        """Return tools from the underlying agent (no mutation)."""
        try:
            if hasattr(self.agent, "get_tools") and callable(self.agent.get_tools):
                return self.agent.get_tools(*args, **kwargs)
            return getattr(self.agent, "tools", [])
        except Exception as e:
            logger.error("Error in get_tools: %s", e)
            return getattr(self.agent, "tools", [])

    def get_all_tools(self, *args, **kwargs):
        if hasattr(self.agent, "get_all_tools"):
            return self.agent.get_all_tools(*args, **kwargs)
        if hasattr(self.agent, "get_tools"):
            return self.agent.get_tools(*args, **kwargs)
        return getattr(self.agent, "tools", [])

    def run(self, *args, **kwargs):
        if hasattr(self.agent, "run"):
            return self.agent.run(*args, **kwargs)
        raise NotImplementedError("Run method not implemented by underlying agent")

    def get_name(self, *args, **kwargs):
        if hasattr(self.agent, "get_name") and callable(self.agent.get_name):
            return self.agent.get_name(*args, **kwargs)
        return self.name

    def reset_tool_choice(self, *args, **kwargs):
        """Delegate to agent if implemented; otherwise no-op."""
        if hasattr(self.agent, "reset_tool_choice"):
            return self.agent.reset_tool_choice(*args, **kwargs)

    def __getattr__(self, name):
        """Transparent passthrough for unknown attributes."""
        return getattr(self.agent, name)


# ------------------------------ Helpers ---------------------------------


def _format_memories(memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in memories or []:
        if not isinstance(m, dict):
            continue
        out.append(
            {
                "id": m.get("id"),
                "text": m.get("text"),
                "type": m.get("type"),
                "significance": m.get("significance"),
                "emotional_intensity": m.get("emotional_intensity"),
                "timestamp": m.get("timestamp"),
            }
        )
    return out


def _coerce_to_dict(obj: Any) -> Dict[str, Any]:
    """Return obj as a dict; if it's JSON string, parse it; else wrap."""
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, str):
        try:
            parsed = json.loads(obj)
            return parsed if isinstance(parsed, dict) else {"data": parsed}
        except json.JSONDecodeError:
            return {"message": obj}
    return {"data": obj}
