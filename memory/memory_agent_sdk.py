# memory/memory_agent_sdk.py

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from agents import (
    Agent,
    Runner,
    ModelSettings,
    function_tool,
    InputGuardrail,
    GuardrailFunctionOutput,
    RunContextWrapper,
)

from memory.memory_agent_wrapper import MemoryAgentWrapper
# Remove this import: from memory.wrapper import MemorySystem
from memory.core import MemorySignificance  # kept in case callers rely on it


# ----------------------------- models (kept for compatibility) -----------------------------

class MemoryInput(BaseModel):
    entity_type: str = Field(description="Type of entity ('player', 'nyx', etc.)")
    entity_id: int = Field(description="ID of the entity")
    memory_text: str = Field(description="The memory text to record")
    importance: str = Field(
        default="medium",
        description="Importance level: 'trivial', 'low', 'medium', 'high', 'critical'",
    )
    emotional: bool = Field(default=True, description="Whether to analyze emotional content")
    tags: Optional[List[str]] = Field(default=None, description="Optional tags for the memory")


class MemoryQueryInput(BaseModel):
    entity_type: str = Field(description="Type of entity ('player', 'nyx', etc.)")
    entity_id: int = Field(description="ID of the entity")
    query: Optional[str] = Field(default=None, description="Optional search query")
    context: Optional[str] = Field(default=None, description="Current context that might influence recall")
    limit: int = Field(default=5, description="Maximum number of memories to return")


class BeliefInput(BaseModel):
    entity_type: str = Field(description="Type of entity ('player', 'nyx', etc.)")
    entity_id: int = Field(description="ID of the entity")
    belief_text: str = Field(description="The belief statement")
    confidence: float = Field(default=0.7, description="Confidence in this belief (0.0-1.0)")


class BeliefsQueryInput(BaseModel):
    entity_type: str = Field(description="Type of entity ('player', 'nyx', etc.)")
    entity_id: int = Field(description="ID of the entity")
    topic: Optional[str] = Field(default=None, description="Optional topic filter")


class MaintenanceInput(BaseModel):
    entity_type: str = Field(description="Type of entity ('player', 'nyx', etc.)")
    entity_id: int = Field(description="ID of the entity")


class AnalysisInput(BaseModel):
    entity_type: str = Field(description="Type of entity ('player', 'nyx', etc.)")
    entity_id: int = Field(description="ID of the entity")


# ----------------------------- context -----------------------------

class MemorySystemContext:
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.memory_system: Optional[MemorySystem] = None


# ----------------------------- helpers -----------------------------

def _ts(v: Any) -> Any:
    if isinstance(v, datetime):
        return v.isoformat()
    return v


def _json_safe(obj: Any) -> Any:
    """Recursively convert datetimes and unknown objects into JSON-safe values."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, list):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    # Last resort: string repr
    return str(obj)


async def _ensure_memory_system(ctx: RunContextWrapper[MemorySystemContext]) -> 'MemorySystem':
    if ctx.context.memory_system is None:
        # Lazy import to avoid circular dependency
        from memory.wrapper import MemorySystem
        ctx.context.memory_system = await MemorySystem.get_instance(
            ctx.context.user_id, ctx.context.conversation_id
        )
    return ctx.context.memory_system


def _fmt_memory_item(memory: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a single memory record into a simple, JSON-safe dict."""
    return {
        "id": memory.get("id"),
        "text": memory.get("text"),
        "type": memory.get("type") or memory.get("memory_type"),
        "significance": memory.get("significance"),
        "emotional_intensity": memory.get("emotional_intensity"),
        "timestamp": _ts(memory.get("timestamp")),
    }


# ----------------------------- function tools -----------------------------

@function_tool
async def remember(
    ctx: RunContextWrapper[MemorySystemContext],
    entity_type: str,
    entity_id: int,
    memory_text: str,
    importance: str,
    emotional: bool,
    tags: Optional[List[str]],
) -> Dict[str, Any]:
    """
    Record a new memory for an entity.
    """
    try:
        # defaults (SDK-safe)
        importance = importance or "medium"
        emotional = True if emotional is None else emotional
        tags = tags or []

        ms = await _ensure_memory_system(ctx)
        result = await ms.remember(
            entity_type=entity_type,
            entity_id=entity_id,
            memory_text=memory_text,
            importance=importance,
            emotional=emotional,
            tags=tags,
        )

        out = {
            "memory_id": result.get("memory_id"),
            "memory_text": memory_text,
            "importance": importance,
        }

        if "emotion_analysis" in result:
            ea = result["emotion_analysis"]
            out["emotional_analysis"] = {
                "primary_emotion": ea.get("primary_emotion"),
                "intensity": ea.get("intensity"),
                "valence": ea.get("valence"),
            }

        return _json_safe(out)
    except Exception as e:
        return {"error": str(e)}


@function_tool
async def recall(
    ctx: RunContextWrapper[MemorySystemContext],
    entity_type: str,
    entity_id: int,
    query: Optional[str] = None,
    context: Optional[str] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Recall memories for an entity, optionally filtered by a query.
    """
    try:
        limit = int(limit or 5)

        ms = await _ensure_memory_system(ctx)
        result = await ms.recall(
            entity_type=entity_type,
            entity_id=entity_id,
            query=query,
            context=context,
            limit=limit,
        )

        memories = [_fmt_memory_item(m) for m in result.get("memories", [])]

        out: Dict[str, Any] = {"memories": memories, "count": len(memories)}

        if "flashback" in result and result["flashback"]:
            fb = result["flashback"]
            out["flashback"] = {
                "text": fb.get("text"),
                "source_memory_id": fb.get("source_memory_id"),
            }

        if result.get("mood_congruent_recall"):
            out["mood_influenced"] = True
            out["current_emotion"] = (
                result.get("current_emotion", {}).get("primary", "neutral")
            )

        return _json_safe(out)
    except Exception as e:
        return {"error": str(e), "memories": [], "count": 0}


@function_tool
async def create_belief(
    ctx: RunContextWrapper[MemorySystemContext],
    entity_type: str,
    entity_id: int,
    belief_text: str,
    confidence: float,
) -> Dict[str, Any]:
    """
    Create a belief for an entity based on their experiences.
    """
    try:
        confidence = float(confidence or 0.7)
        ms = await _ensure_memory_system(ctx)
        result = await ms.create_belief(
            entity_type=entity_type,
            entity_id=entity_id,
            belief_text=belief_text,
            confidence=confidence,
        )
        out = {
            "belief_id": result.get("belief_id"),
            "belief_text": result.get("belief_text") or belief_text,
            "confidence": result.get("confidence", confidence),
        }
        return _json_safe(out)
    except Exception as e:
        return {"error": str(e)}


@function_tool
async def get_beliefs(
    ctx: RunContextWrapper[MemorySystemContext],
    entity_type: str,
    entity_id: int,
    topic: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Get beliefs held by an entity.
    """
    try:
        ms = await _ensure_memory_system(ctx)
        beliefs = await ms.get_beliefs(
            entity_type=entity_type, entity_id=entity_id, topic=topic
        )
        # ensure JSON-safe list of dicts
        return _json_safe(beliefs)
    except Exception as e:
        return [{"error": str(e)}]


@function_tool
async def run_maintenance(
    ctx: RunContextWrapper[MemorySystemContext], entity_type: str, entity_id: int
) -> Dict[str, Any]:
    """
    Run maintenance tasks on an entity's memories (consolidation, decay, etc.).
    """
    try:
        ms = await _ensure_memory_system(ctx)
        result = await ms.maintain(entity_type=entity_type, entity_id=entity_id)
        return _json_safe(result)
    except Exception as e:
        return {"error": str(e)}


@function_tool
async def analyze_memories(
    ctx: RunContextWrapper[MemorySystemContext], entity_type: str, entity_id: int
) -> Dict[str, Any]:
    """
    Perform a comprehensive analysis of an entity's memories.
    """
    try:
        ms = await _ensure_memory_system(ctx)
        result = await ms.analyze_entity_memory(
            entity_type=entity_type, entity_id=entity_id
        )
        return _json_safe(result)
    except Exception as e:
        return {"error": str(e)}


@function_tool
async def generate_schemas(
    ctx: RunContextWrapper[MemorySystemContext], entity_type: str, entity_id: int
) -> Dict[str, Any]:
    """
    Generate schemas by analyzing memory patterns.
    """
    try:
        ms = await _ensure_memory_system(ctx)
        result = await ms.generate_schemas(entity_type=entity_type, entity_id=entity_id)
        return _json_safe(result)
    except Exception as e:
        return {"error": str(e)}


@function_tool
async def add_journal_entry(
    ctx: RunContextWrapper[MemorySystemContext],
    player_name: str,
    entry_text: str,
    entry_type: str,
    fantasy_flag: bool,
    intensity_level: int,
) -> Dict[str, Any]:
    """
    Add a journal entry to a player's memory.
    """
    try:
        entry_type = entry_type or "observation"
        fantasy_flag = bool(fantasy_flag) if fantasy_flag is not None else False
        intensity_level = int(intensity_level or 0)

        ms = await _ensure_memory_system(ctx)
        journal_id = await ms.add_journal_entry(
            player_name=player_name,
            entry_text=entry_text,
            entry_type=entry_type,
            fantasy_flag=fantasy_flag,
            intensity_level=intensity_level,
        )

        return _json_safe(
            {
                "journal_entry_id": journal_id,
                "player_name": player_name,
                "entry_text": entry_text,
                "entry_type": entry_type,
            }
        )
    except Exception as e:
        return {"error": str(e)}


@function_tool
async def get_journal_history(
    ctx: RunContextWrapper[MemorySystemContext],
    player_name: str,
    entry_type: Optional[str],
    limit: int,
) -> List[Dict[str, Any]]:
    """
    Get a player's journal entries.
    """
    try:
        limit = int(limit or 10)
        ms = await _ensure_memory_system(ctx)
        history = await ms.get_journal_history(
            player_name=player_name, entry_type=entry_type, limit=limit
        )
        return _json_safe(history)
    except Exception as e:
        return [{"error": str(e)}]


# ----------------------------- input guardrail -----------------------------

async def validate_entity_input(
    ctx: RunContextWrapper[MemorySystemContext], agent: Agent[MemorySystemContext], input_data: Any
) -> GuardrailFunctionOutput:
    """
    Validate entity information in input. If the user message mentions
    'entity' or 'memory' but not a clear 'entity_id', trip the guardrail.
    """
    # Normalize input to text
    if isinstance(input_data, list):
        text = []
        for item in input_data:
            if isinstance(item, dict) and "content" in item:
                text.append(str(item["content"]))
            elif isinstance(item, str):
                text.append(item)
        input_text = " ".join(text)
    else:
        input_text = str(input_data)

    if ("entity" in input_text.lower() or "memory" in input_text.lower()) and ("entity_id" not in input_text):
        return GuardrailFunctionOutput(
            output_info={"valid": False, "reason": "Missing entity identification"},
            tripwire_triggered=True,
        )

    return GuardrailFunctionOutput(output_info={"valid": True}, tripwire_triggered=False)


# ----------------------------- agent factory -----------------------------

from textwrap import dedent

FULL_PROMPT = dedent("""
You are a memory management assistant that helps manage, retrieve, and analyze memories.
You have access to a sophisticated memory system that stores and organizes memories for
different entities. Each entity has a type (such as "player" or "nyx") and an entity_id.

You can:
1. Record new memories with varying levels of importance
2. Retrieve memories based on queries or context
3. Create and manage beliefs derived from memories
4. Run maintenance on memories (consolidation, decay, etc.)
5. Analyze memory patterns and generate schemas
6. Manage journal entries for players

Always ask for the entity_type and entity_id when performing memory operations.
When describing memories, focus on their content, emotional aspects, and significance.

For memory importance levels, use:
- "trivial": Everyday minor details
- "low": Minor but somewhat memorable events
- "medium": Standard memories of moderate importance
- "high": Important memories that stand out
- "critical": Extremely important, life-changing memories
""").strip()

COMPACT_PROMPT = (
    "You manage, retrieve, and analyze memories. Always require entity_type and entity_id. "
    "Importance: trivial/low/medium/high/critical."
)

def create_memory_agent(
    user_id: int,
    conversation_id: int,
    *,
    model: str = "gpt-5-nano",
    use_full_prompt: bool = True,
) -> Agent[MemorySystemContext]:
    instructions = FULL_PROMPT if use_full_prompt else COMPACT_PROMPT

    base_agent = Agent[MemorySystemContext](
        name="Memory Manager",
        instructions=instructions,
        tools=[
            remember,
            recall,
            create_belief,
            get_beliefs,
            run_maintenance,
            analyze_memories,
            generate_schemas,
            add_journal_entry,
            get_journal_history,
        ],
        input_guardrails=[InputGuardrail(guardrail_function=validate_entity_input)],
        model_settings=ModelSettings(),
        model=model,
    )
    return base_agent

