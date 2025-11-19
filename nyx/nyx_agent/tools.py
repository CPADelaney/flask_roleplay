# nyx/nyx_agent/tools.py
"""Refactored tools module for Nyx Agent SDK with ContextBundle architecture.

Strict JSON Schema–safe:
- All tool *inputs* use TypedDicts (no Pydantic), avoiding `additionalProperties`.
- Output models can remain Pydantic; they are not part of the input schema.

Notes:
- Keep @function_tool strict_json_schema=True (default).
- If you add new input types, prefer TypedDict (or simple scalars) here.
"""

from __future__ import annotations

import asyncio
import time
import logging
import uuid
from typing import Dict, List, Any, Optional, Union

from typing_extensions import TypedDict
from typing import Mapping, NotRequired

# JSON-friendly value helpers for tool payloads (kept non-recursive so TypedDicts
# remain fully defined when Pydantic inspects them for tool schemas).
# NOTE: Avoid Dict[...] here—Pydantic would emit additionalProperties which the
# agents SDK rejects when building strict JSON schemas for tool inputs.
JSONPrimitive = Union[str, int, float, bool, None]
JSONSequence = List[JSONPrimitive]
JSONValue = Union[JSONPrimitive, JSONSequence]

from agents import function_tool, RunContextWrapper

from nyx.governance import AgentType
from nyx.governance.ids import format_agent_id
from nyx.governance_helpers import check_permission, report_action
from nyx.gateway.lore_tool import handle_lore_operation as gateway_handle_lore_operation

from .context import (
    NyxContext,
    ContextBroker,
    SceneScope,
    ContextBundle,
    PackedContext,
    WORLD_SIMULATION_AVAILABLE,
)
from .models import (
    # OUTPUT models only (safe to keep as they don't affect tool param schema)
    MemoryItem,
    MemorySearchResult,
    MemoryStorageResult,
    UserGuidanceResult,
    ImageGenerationDecision,
    EmotionalStateResult,
    RelationshipUpdate,
    PerformanceMetrics,
    ActivityRecommendationsResult,
    DecisionScoringResult,
    ConflictDetectionResult,
)

logger = logging.getLogger(__name__)


LORE_TIMEOUT_ERROR_MESSAGE = "Lore request timed out (Codex Task 7). Please try again."
LORE_GENERIC_ERROR_MESSAGE = "Lore request failed unexpectedly (Codex Task 7). Please try again later."
WORLD_SYSTEMS_DISABLED_ERROR_MESSAGE = "World systems are not available in this environment."
WORLD_SYSTEMS_UNAVAILABLE_ERROR_MESSAGE = "World systems are still initializing. Please try again shortly."
WORLD_OPERATION_GENERIC_ERROR_MESSAGE = "World operation failed unexpectedly. Please try again later."
WORLD_GOVERNANCE_DENIED_TEMPLATE = "Governance denied this world action: {reason}"


async def lore_handle_operation_tool_wrapper(
    *,
    user_id: int,
    conversation_id: int,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Invoke the lore gateway with a strict timeout and rich logging."""

    extra = {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "aspects": payload.get("aspects"),
    }

    start = time.monotonic()
    outcome: Dict[str, Any]

    try:
        outcome = await asyncio.wait_for(
            gateway_handle_lore_operation(
                user_id=user_id,
                conversation_id=conversation_id,
                payload=payload,
            ),
            timeout=3.0,
        )
    except asyncio.TimeoutError:
        duration_ms = int((time.monotonic() - start) * 1000)
        logger.warning(
            "Lore operation timed out",
            extra={**extra, "duration_ms": duration_ms, "success": False, "error_kind": "timeout"},
        )
        return {"ok": False, "error": LORE_TIMEOUT_ERROR_MESSAGE}
    except Exception:
        duration_ms = int((time.monotonic() - start) * 1000)
        logger.exception(
            "Lore operation failed",
            extra={**extra, "duration_ms": duration_ms, "success": False, "error_kind": "internal"},
        )
        return {"ok": False, "error": LORE_GENERIC_ERROR_MESSAGE}

    duration_ms = int((time.monotonic() - start) * 1000)
    # Successful call: log once with basic metrics
    logger.info(
        "Lore operation completed",
        extra={**extra, "duration_ms": duration_ms, "success": True, "error_kind": None},
    )
    return outcome


def _normalize_extra_items(items: Any) -> Dict[str, Any]:
    """Normalize optional key/value pairs coming from tool payloads."""

    if isinstance(items, Mapping):
        return {str(key): value for key, value in items.items()}

    normalized: Dict[str, Any] = {}
    if isinstance(items, list):
        for entry in items:
            if not isinstance(entry, Mapping):
                continue
            key = entry.get("key")
            if key is None:
                continue
            normalized[str(key)] = entry.get("value")
    return normalized


async def _resolve_world_orchestrator(nyx_ctx: NyxContext) -> Optional[Any]:
    """Ensure the world orchestrator is initialized and ready."""

    orchestrator = getattr(nyx_ctx, "world_orchestrator", None)
    if orchestrator is not None:
        return orchestrator

    awaiter = getattr(nyx_ctx, "await_orchestrator", None)
    if callable(awaiter):
        try:
            ready = await awaiter("world")
        except Exception:  # pragma: no cover - defensive guard
            logger.debug("World orchestrator await failed", exc_info=True)
            return None
        if not ready:
            return None
        return getattr(nyx_ctx, "world_orchestrator", None)

    return None


async def world_handle_operation_tool_wrapper(
    *,
    ctx: RunContextWrapper,
    action_type: str,
    payload: Optional[WorldOperationPayload],
) -> Dict[str, Any]:
    """Governance-aware bridge for imperative world operations."""

    if not WORLD_SIMULATION_AVAILABLE:
        return {"ok": False, "error": WORLD_SYSTEMS_DISABLED_ERROR_MESSAGE}

    if not isinstance(action_type, str) or not action_type.strip():
        raise ValueError("world_handle_operation requires a non-empty action_type string")

    nyx_ctx: Optional[NyxContext]
    if hasattr(ctx, "context") and isinstance(ctx.context, NyxContext):
        nyx_ctx = ctx.context
    elif isinstance(ctx, NyxContext):
        nyx_ctx = ctx
    else:
        nyx_ctx = None

    if nyx_ctx is None:
        raise ValueError("Nyx context missing for world operation")

    user_id = getattr(nyx_ctx, "user_id", None)
    conversation_id = getattr(nyx_ctx, "conversation_id", None)
    if user_id is None or conversation_id is None:
        raise ValueError("Nyx context missing identifiers for world operation")

    orchestrator = await _resolve_world_orchestrator(nyx_ctx)
    extra = {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "action_type": action_type.strip().lower(),
    }
    if orchestrator is None:
        logger.info(
            "World operation requested before orchestrator ready",
            extra={**extra, "duration_ms": 0, "success": False, "error_kind": "unavailable"},
        )
        return {"ok": False, "error": WORLD_SYSTEMS_UNAVAILABLE_ERROR_MESSAGE}

    forward_payload: Dict[str, Any] = dict(payload or {})
    scope_payload = forward_payload.get("scope")
    if isinstance(scope_payload, Mapping):
        forward_payload["scope"] = dict(scope_payload)

    for field in ("parameters", "extra"):
        normalized_items = _normalize_extra_items(forward_payload.get(field))
        if normalized_items:
            forward_payload[field] = normalized_items
        else:
            forward_payload.pop(field, None)

    normalized_action = action_type.strip().lower()
    forward_payload.setdefault("operation", normalized_action)

    action_details: Dict[str, Any] = {
        "operation": forward_payload.get("operation"),
        "handler": forward_payload.get("handler"),
    }
    for detail_key in ("scope", "aspects", "entities", "depth"):
        if forward_payload.get(detail_key) is not None:
            action_details[detail_key] = forward_payload[detail_key]

    permission = await check_permission(
        user_id=user_id,
        conversation_id=conversation_id,
        agent_type=AgentType.WORLD_ORCHESTRATOR,
        agent_id=format_agent_id(AgentType.WORLD_ORCHESTRATOR, conversation_id),
        action_type=normalized_action,
        action_details=action_details,
    )

    if not permission.get("approved", True):
        reason = permission.get("reasoning") or "Not approved"
        logger.info(
            "World operation denied by governance",
            extra={**extra, "duration_ms": 0, "success": False, "error_kind": "governance"},
        )
        return {
            "ok": False,
            "error": WORLD_GOVERNANCE_DENIED_TEMPLATE.format(reason=reason),
            "governance_blocked": True,
        }

    override = permission.get("override_action")
    if isinstance(override, Mapping):
        override_payload = override.get("payload")
        if isinstance(override_payload, Mapping):
            forward_payload.update(dict(override_payload))
        for key in ("operation", "handler"):
            if key in override:
                forward_payload[key] = override[key]

    start = time.monotonic()
    try:
        outcome = await orchestrator.handle_world_operation(forward_payload)
    except Exception:
        duration_ms = int((time.monotonic() - start) * 1000)
        logger.exception(
            "World operation failed",
            extra={**extra, "duration_ms": duration_ms, "success": False, "error_kind": "exception"},
        )
        return {"ok": False, "error": WORLD_OPERATION_GENERIC_ERROR_MESSAGE}

    duration_ms = int((time.monotonic() - start) * 1000)
    logger.info(
        "World operation completed",
        extra={**extra, "duration_ms": duration_ms, "success": True, "error_kind": None},
    )

    normalized_result: Dict[str, Any]
    if isinstance(outcome, Mapping):
        normalized_result = dict(outcome)
    else:
        normalized_result = {"result": outcome}

    normalized_result.setdefault("success", normalized_result.get("ok", True))
    normalized_result.setdefault("ok", bool(normalized_result.get("success", True)))

    tracking_id = permission.get("tracking_id")
    if tracking_id not in (None, ""):
        normalized_result["governance_tracking_id"] = tracking_id

    action_record = {
        "type": normalized_action,
        "operation": forward_payload.get("operation"),
        "handler": forward_payload.get("handler"),
        "scope": forward_payload.get("scope"),
    }
    try:
        await report_action(
            user_id=user_id,
            conversation_id=conversation_id,
            agent_type=AgentType.WORLD_ORCHESTRATOR,
            agent_id=format_agent_id(AgentType.WORLD_ORCHESTRATOR, conversation_id),
            action=action_record,
            result=normalized_result,
        )
    except Exception:
        logger.debug("World action report failed", exc_info=True)

    return normalized_result

# ╭──────────────────────────────────────────────────────────────────────────────╮
# │ TypedDict INPUT MODELS (strict JSON schema–friendly)                         │
# ╰──────────────────────────────────────────────────────────────────────────────╯

class EmptyInput(TypedDict):
    pass


class RetrieveMemoriesInput(TypedDict):
    query: NotRequired[str]
    limit: int


class AddMemoryInput(TypedDict):
    memory_text: str
    memory_type: str
    significance: Union[int, float]
    entities: NotRequired[List[Union[int, str]]]


class DecideImageGenerationInput(TypedDict):
    scene_text: str


class UpdateEmotionalStateInput(TypedDict):
    triggering_event: str
    valence_change: Union[int, float]
    arousal_change: Union[int, float]
    dominance_change: Union[int, float]


class UpdateRelationshipStateInput(TypedDict):
    entity_id: Union[int, str]
    trust_change: Union[int, float]
    attraction_change: Union[int, float]
    respect_change: Union[int, float]
    triggering_event: NotRequired[str]


class DecisionOptionInput(TypedDict, total=False):
    # Optional keys; we’ll normalize in code.
    id: str
    description: str


class ScoreDecisionOptionsInput(TypedDict):
    options: List[Union[str, DecisionOptionInput]]


class LoreOperationExtraItem(TypedDict):
    """Key/value entries passed to the lore orchestrator."""

    key: str
    value: JSONValue


class LoreOperationInput(TypedDict, total=False):
    aspects: List[str]
    location_id: Union[int, str]
    location_name: str
    npc_ids: List[Union[int, str]]
    detail_level: str
    extra: List[LoreOperationExtraItem]


class WorldOperationScope(TypedDict, total=False):
    location_id: Union[int, str]
    location_name: str
    npc_ids: List[Union[int, str]]
    topics: List[str]
    tags: List[str]


class WorldOperationExtraItem(TypedDict):
    key: str
    value: JSONValue


class WorldOperationPayload(TypedDict, total=False):
    operation: str
    handler: str
    scope: WorldOperationScope
    entities: List[Union[int, str]]
    aspects: List[str]
    depth: str
    parameters: List[WorldOperationExtraItem]
    extra: List[WorldOperationExtraItem]


# ╭──────────────────────────────────────────────────────────────────────────────╮
# │ Bundle expansion utilities                                                   │
# ╰──────────────────────────────────────────────────────────────────────────────╯

def _log_task_exc(task: asyncio.Task) -> None:
    """Log exceptions from background tasks."""
    try:
        task.result()
    except Exception:
        logger.exception("Background task failed")


async def _get_context_broker(ctx: RunContextWrapper) -> ContextBroker:
    """Get or create the ContextBroker from wrapped context."""
    nyx_ctx = ctx.context if isinstance(ctx.context, NyxContext) else ctx
    if not hasattr(nyx_ctx, 'broker'):
        nyx_ctx.broker = ContextBroker(nyx_ctx)
    return nyx_ctx.broker


async def _get_bundle(ctx: RunContextWrapper, expand_sections: Optional[List[str]] = None) -> ContextBundle:
    """Get the current context bundle, optionally expanding specific sections."""
    broker = await _get_context_broker(ctx)
    nyx_ctx = ctx.context if isinstance(ctx.context, NyxContext) else ctx

    # Cache on the actual NyxContext, not the wrapper
    if not hasattr(nyx_ctx, '_current_bundle'):
        scene_scope = await broker.compute_scene_scope(
            getattr(nyx_ctx, 'last_user_input', ''),
            getattr(nyx_ctx, 'current_state', {})
        )
        nyx_ctx._current_bundle = await broker.load_or_fetch_bundle(scene_scope)

    bundle = nyx_ctx._current_bundle

    # Expand missing sections in parallel
    if expand_sections:
        missing = [s for s in expand_sections if s not in bundle.expanded_sections]
        if missing:
            await asyncio.gather(*(broker.expand_bundle_section(bundle, s) for s in missing))

    # Mirror onto wrapper for compatibility
    setattr(ctx, '_current_bundle', bundle)
    return bundle


async def _get_packed_context(
    ctx: RunContextWrapper,
    token_budget: int = 8000,
    must_include: Optional[List[str]] = None
) -> PackedContext:
    """Get packed context optimized for token budget."""
    bundle = await _get_bundle(ctx)
    return bundle.pack(token_budget, must_include or ['canon'])


# ╭──────────────────────────────────────────────────────────────────────────────╮
# │ Memory tools                                                                 │
# ╰──────────────────────────────────────────────────────────────────────────────╯

@function_tool
async def retrieve_memories(
    ctx: RunContextWrapper,
    payload: RetrieveMemoriesInput
) -> Dict[str, Any]:
    """Enhanced memory retrieval using ContextBundle's memory section."""
    bundle = await _get_bundle(ctx, expand_sections=['memory'])

    # Extract memory data from bundle (avoid mutation)
    memory_data = bundle.sections.get('memory', {})
    memories = list(memory_data.get('memories', []))  # Copy to avoid mutation
    graph_links = memory_data.get('graph_links', {})

    # Define graph_keys and scene entities BEFORE the loop, normalize to strings
    graph_keys = {str(k) for k in graph_links.keys()}
    scene_entity_ids = {str(e) for e in getattr(bundle.scene_scope, 'entity_ids', [])}

    # If we need more memories, fetch via broker
    if payload['limit'] > len(memories):
        broker = await _get_context_broker(ctx)
        additional = await broker.fetch_memories(
            query=payload.get('query'),
            limit=payload['limit'] - len(memories),
            exclude_ids=[str(m.get('id')) for m in memories]
        )
        memories.extend(additional)

    # Apply relevance filtering with graph boost
    scored_memories: List[Dict[str, Any]] = []
    for mem in memories:
        base_score = mem.get('relevance', 0.0)
        link_boost = 0.0
        mem_id = str(mem.get('id', ''))

        # Boost if memory is graph-linked to current scene entities
        if mem_id in graph_keys:
            linked_entities = {str(e) for e in graph_links[mem_id]}
            if linked_entities & scene_entity_ids:  # Set intersection
                link_boost = 0.2  # Emergent connection bonus

        scored_memories.append({**mem, 'relevance': min(1.0, base_score + link_boost)})

    # Sort by relevance and limit
    scored_memories.sort(key=lambda m: m['relevance'], reverse=True)
    final_memories = scored_memories[: payload['limit']]

    # Format with confidence markers
    formatted = []
    for mem in final_memories:
        relevance = mem['relevance']
        confidence = (
            "vividly recall" if relevance > 0.8 else
            "remember" if relevance > 0.6 else
            "think I recall" if relevance > 0.4 else
            "vaguely remember"
        )
        formatted.append(f"I {confidence}: {mem['text']}")

    # Count graph connections (using pre-defined graph_keys)
    graph_connections = sum(1 for m in final_memories if str(m.get('id')) in graph_keys)

    return MemorySearchResult(
        memories=[MemoryItem(**m) for m in final_memories],
        formatted_text="\n".join(formatted) if formatted else "No relevant memories found.",
        graph_connections=graph_connections
    ).model_dump()


@function_tool
async def add_memory(
    ctx: RunContextWrapper,
    payload: AddMemoryInput
) -> Dict[str, Any]:
    """Store memory and update context bundle."""
    broker = await _get_context_broker(ctx)

    # Store via broker (handles canon logging)
    result = await broker.store_memory(
        memory_text=payload['memory_text'],
        memory_type=payload['memory_type'],
        significance=payload['significance'],
        entities=payload.get('entities', []),  # Link to scene entities
    )

    # Invalidate memory section of bundle to force refresh
    nyx_ctx = ctx.context if isinstance(ctx.context, NyxContext) else ctx
    if hasattr(nyx_ctx, '_current_bundle'):
        nyx_ctx._current_bundle.invalidate_section('memory')

    return MemoryStorageResult(
        memory_id=result['memory_id'],
        success=True,
        linked_entities=result.get('linked_entities', [])
    ).model_dump()


# ╭──────────────────────────────────────────────────────────────────────────────╮
# │ Narrative tools                                                              │
# ╰──────────────────────────────────────────────────────────────────────────────╯

@function_tool
async def tool_narrate_slice_of_life_scene(
    ctx: RunContextWrapper,
    scene_type: str = "routine",
    player_action: Optional[str] = None
) -> Dict[str, Any]:
    """Generate narration using packed context optimized for scene."""
    # Get scene-optimized bundle
    bundle = await _get_bundle(ctx, expand_sections=['npcs', 'location', 'conflict'])
    packed = await _get_packed_context(ctx, token_budget=6000, must_include=['npcs', 'canon'])

    # Extract relevant data from bundle
    npcs = bundle.sections.get('npcs', {}).get('active', [])
    location = bundle.sections.get('location', {})
    conflicts = bundle.sections.get('conflict', {}).get('active', [])

    # Build scene context from bundle (safer NPC ID handling)
    participants: List[str] = []
    for npc in npcs[:3]:
        if isinstance(npc, dict):
            npc_id = npc.get('id')
            participants.append(str(npc_id) if npc_id is not None else 'unknown')
        else:
            participants.append(str(npc))

    scene_context = {
        'participants': participants,
        'location': location.get('name', 'current_location'),
        'atmosphere': location.get('atmosphere', ''),
        'tensions': [c['name'] for c in conflicts if c.get('intensity', 0) > 0.5],
    }

    # Check for emergent narrative opportunities
    broker = await _get_context_broker(ctx)
    emergent = await broker.detect_emergent_patterns(bundle) or {}

    # Generate narration with packed context
    narration = await broker.generate_narration(
        scene_type=scene_type,
        scene_context=scene_context,
        packed_context=packed,
        player_action=player_action,
        emergent_themes=emergent.get('themes', [])
    )

    return {
        'narrative': narration['text'],
        'atmosphere': narration.get('atmosphere', ''),
        'nyx_commentary': narration.get('commentary'),
        'emergent_connections': emergent.get('connections', []),
        'context_aware': True,
        'governance_approved': True
    }


@function_tool
async def orchestrate_slice_scene(
    ctx: RunContextWrapper,
    scene_type: str = "routine"
) -> Dict[str, Any]:
    """Orchestrate complete scene with all subsystems via ContextBroker."""
    broker = await _get_context_broker(ctx)

    # Read from underlying NyxContext, not wrapper
    nyx_ctx = ctx.context if hasattr(ctx, 'context') else ctx

    # Compute scene scope from current state
    scene_scope = await broker.compute_scene_scope(
        getattr(nyx_ctx, 'last_user_input', ''),
        {'scene_type': scene_type}
    )

    # Parallel fetch all scene components
    bundle = await broker.fetch_bundle(scene_scope)

    # Generate scene with emergent connections
    scene = await broker.orchestrate_scene(
        bundle=bundle,
        scene_type=scene_type,
        enable_emergence=True
    )

    return {
        'scene': scene['narrative'],
        'world_state': bundle.sections.get('world', {}),
        'active_npcs': bundle.sections.get('npcs', {}).get('active', []),
        'conflicts': bundle.sections.get('conflict', {}).get('active', []),
        'emergent_events': scene.get('emergent_events', []),
        'choices': scene.get('choices', [])
    }


@function_tool
async def generate_npc_dialogue(
    ctx: RunContextWrapper,
    npc_id: int,
    context_hint: Optional[str] = None
) -> Dict[str, Any]:
    """Generate NPC dialogue with relationship-aware context."""
    # Expand NPC section for specific character
    bundle = await _get_bundle(ctx, expand_sections=['npcs', 'memory'])
    broker = await _get_context_broker(ctx)

    # Get NPC data from bundle
    npc_data = await broker.get_npc_details(bundle, npc_id)

    # Find memories involving this NPC (normalize IDs to strings)
    memories = bundle.sections.get('memory', {}).get('memories', [])
    npc_id_str = str(npc_id)
    npc_memories = [m for m in memories if npc_id_str in [str(e) for e in m.get('entities', [])]]

    # Generate dialogue with context
    dialogue = await broker.generate_npc_dialogue(
        npc_data=npc_data,
        recent_memories=npc_memories[-3:],
        context_hint=context_hint,
        relationship_state=npc_data.get('relationship', {})
    )

    # Check if dialogue references a memory
    ref = str(dialogue.get('references_memory', ''))
    reveals = any(str(m.get('id')) == ref for m in npc_memories)

    return {
        'npc_name': npc_data['name'],
        'dialogue': dialogue['text'],
        'subtext': dialogue.get('subtext'),
        'relationship_change': dialogue.get('relationship_delta', 0),
        'reveals_memory': reveals
    }


# ╭──────────────────────────────────────────────────────────────────────────────╮
# │ World state tools                                                            │
# ╰──────────────────────────────────────────────────────────────────────────────╯

@function_tool
async def check_world_state(
    ctx: RunContextWrapper,
    payload: EmptyInput
) -> Dict[str, Any]:
    """Get world state from context bundle."""
    bundle = await _get_bundle(ctx, expand_sections=['world'])
    world_data = bundle.sections.get('world', {})

    # Ensure we have all canonical world facts
    if not world_data.get('canon_verified'):
        broker = await _get_context_broker(ctx)
        await broker.verify_canon(bundle, 'world')

    return {
        'time_of_day': world_data.get('time_of_day', 'unknown'),
        'weather': world_data.get('weather', 'clear'),
        'location': world_data.get('current_location', 'unknown'),
        'world_mood': world_data.get('mood', 'neutral'),
        'tension_level': world_data.get('tension', 0.5),
        'active_events': world_data.get('events', []),
        'canon_facts': world_data.get('canon', {})  # Use {} for consistency
    }


@function_tool
async def generate_emergent_event(
    ctx: RunContextWrapper,
    trigger: Optional[str] = None
) -> Dict[str, Any]:
    """Generate emergent events based on bundle's graph connections."""
    broker = await _get_context_broker(ctx)
    bundle = await _get_bundle(ctx)

    # Detect emergent patterns from graph (guard against None)
    patterns = await broker.detect_emergent_patterns(bundle) or {}
    connections = patterns.get('connections', [])

    # Generate event based on patterns
    if connections:
        event = await broker.generate_emergent_event(
            patterns=patterns,
            trigger=trigger,
            bundle=bundle
        )

        # Update bundle with new event
        bundle.add_event(event)

        return {
            'event': event['description'],
            'type': event['type'],
            'impacts': event.get('impacts', []),
            'emerges_from': connections[:3],
            'canon_consistent': True
        }

    return {
        'event': None,
        'reason': 'No emergent patterns detected'
    }


@function_tool
async def simulate_npc_autonomy(
    ctx: RunContextWrapper,
    npc_id: Optional[int] = None,
    time_skip: Optional[int] = 0
) -> Dict[str, Any]:
    """Simulate NPC actions using bundle's cached NPC state."""
    bundle = await _get_bundle(ctx, expand_sections=['npcs'])
    broker = await _get_context_broker(ctx)

    # Get NPCs from bundle
    npcs = bundle.sections.get('npcs', {}).get('active', [])
    if npc_id is not None:
        npc_id_str = str(npc_id)
        npcs = [n for n in npcs if str(n.get('id', '')) == npc_id_str]

    # Simulate actions in parallel
    actions = await broker.simulate_npc_actions(
        npcs=npcs,
        time_skip=time_skip,
        context_bundle=bundle
    )

    # Update bundle with new NPC states
    if actions:
        bundle.update_npc_states(actions)

    return {
        'npc_actions': actions,
        'world_changes': bundle.get_pending_changes(),
        'maintains_canon': True
    }


# ╭──────────────────────────────────────────────────────────────────────────────╮
# │ Emotional & relationship tools                                               │
# ╰──────────────────────────────────────────────────────────────────────────────╯

@function_tool
async def calculate_and_update_emotional_state(
    ctx: RunContextWrapper,
    payload: UpdateEmotionalStateInput
) -> Dict[str, Any]:
    """Update emotional state with bundle-aware context."""
    bundle = await _get_bundle(ctx, expand_sections=['emotional'])
    broker = await _get_context_broker(ctx)

    # Get current emotional state from bundle
    current_state = bundle.sections.get('emotional', {})

    # Calculate new state with graph-aware influences
    patterns = await broker.detect_emotional_patterns(bundle)

    new_state = await broker.update_emotional_state(
        current=current_state,
        event=payload['triggering_event'],
        valence_delta=payload['valence_change'],
        arousal_delta=payload['arousal_change'],
        dominance_delta=payload['dominance_change'],
        patterns=patterns
    )

    # Update bundle
    bundle.sections['emotional'] = new_state

    # Return consistent modelled output
    return EmotionalStateResult(
        valence=new_state["valence"],
        arousal=new_state["arousal"],
        dominance=new_state["dominance"],
        emotional_label=new_state.get("label", "neutral"),
        manifestation=new_state.get("manifestation", "")
    ).model_dump()


@function_tool
async def update_relationship_state(
    ctx: RunContextWrapper,
    payload: UpdateRelationshipStateInput
) -> Dict[str, Any]:
    """Update relationship using bundle's relationship graph."""
    bundle = await _get_bundle(ctx, expand_sections=['relationships'])
    broker = await _get_context_broker(ctx)

    # Normalize entity ID to string for bundle storage
    entity_id = payload['entity_id']
    entity_id_str = str(entity_id)

    # Get relationship from bundle
    relationships = bundle.sections.get('relationships', {})
    current = relationships.get(entity_id_str, {})

    # Update with graph-aware context
    updated = await broker.update_relationship(
        entity_id=entity_id,  # original type for broker
        current_state=current,
        trust_delta=payload['trust_change'],
        attraction_delta=payload['attraction_change'],
        respect_delta=payload['respect_change'],
        event=payload.get('triggering_event'),
        graph_context=bundle.get_graph_context(entity_id)
    )

    # Store in bundle with normalized ID
    relationships[entity_id_str] = updated

    return RelationshipUpdate(
        entity_id=entity_id,
        new_trust=updated['trust'],
        new_attraction=updated['attraction'],
        new_respect=updated['respect'],
        relationship_level=updated.get('level', 'neutral'),
        change_description=updated.get('change_description', '')
    ).model_dump()


# ╭──────────────────────────────────────────────────────────────────────────────╮
# │ Decision & conflict tools                                                    │
# ╰──────────────────────────────────────────────────────────────────────────────╯

@function_tool
async def detect_conflicts_and_instability(
    ctx: RunContextWrapper,
    payload: EmptyInput
) -> Dict[str, Any]:
    """Detect conflicts using bundle's graph analysis."""
    bundle = await _get_bundle(ctx, expand_sections=['conflict', 'npcs'])
    broker = await _get_context_broker(ctx)

    # Analyze graph for tension points
    tensions = await broker.analyze_tensions(bundle)

    # Detect active and potential conflicts
    conflicts = bundle.sections.get('conflict', {}).get('active', [])
    potential = await broker.detect_potential_conflicts(bundle, tensions)

    return ConflictDetectionResult(
        active_conflicts=conflicts,
        potential_conflicts=potential,
        tension_level=tensions.get('overall', 0.5),
        hot_spots=tensions.get('hot_spots', []),
        recommendations=tensions.get('recommendations', [])
    ).model_dump()


@function_tool
async def score_decision_options(
    ctx: RunContextWrapper,
    payload: ScoreDecisionOptionsInput
) -> Dict[str, Any]:
    """Score decisions with full bundle context."""
    bundle = await _get_bundle(ctx)
    broker = await _get_context_broker(ctx)

    # Normalize options: support List[str] or List[DecisionOptionInput]-like dicts
    normalized: List[str] = []
    for o in payload['options']:
        if isinstance(o, str):
            normalized.append(o)
        elif isinstance(o, dict):
            normalized.append(o.get('id') or o.get('description') or str(o))
        else:
            normalized.append(str(o))

    # Score each option with multi-factor analysis
    scores: Dict[str, Dict[str, Any]] = {}
    for option in normalized:
        score = await broker.score_decision(
            option=option,
            bundle=bundle,
            criteria={
                'canon_consistency': 1.0,      # Top priority
                'character_alignment': 0.8,
                'narrative_impact': 0.7,
                'emergent_potential': 0.6,
                'player_satisfaction': 0.5
            }
        )
        scores[option] = score

    # Sort by score
    ranked = sorted(scores.items(), key=lambda x: x[1]['total'], reverse=True)

    return DecisionScoringResult(
        options={k: v['total'] for k, v in scores.items()},
        recommended=ranked[0][0] if ranked else None,
        reasoning={k: v['reasoning'] for k, v in scores.items()}
    ).model_dump()


# ╭──────────────────────────────────────────────────────────────────────────────╮
# │ Image generation tools                                                       │
# ╰──────────────────────────────────────────────────────────────────────────────╯

@function_tool
async def decide_image_generation(
    ctx: RunContextWrapper,
    payload: DecideImageGenerationInput
) -> Dict[str, Any]:
    """Decide on image generation using bundle's scene analysis."""
    bundle = await _get_bundle(ctx, expand_sections=['visual'])
    broker = await _get_context_broker(ctx)

    # Analyze scene for visual potential
    visual_analysis = await broker.analyze_visual_potential(
        scene_text=payload['scene_text'],
        bundle=bundle
    )

    # Check pacing constraints
    recent_images = bundle.sections.get('visual', {}).get('recent', [])
    time_since_last = broker.get_time_since_last_image(recent_images)

    should_generate = (
        visual_analysis['score'] > 0.7 and
        time_since_last > 300 and  # 5 minutes
        not bundle.is_dialogue_heavy()
    )

    result = ImageGenerationDecision(
        should_generate=should_generate,
        scene_score=visual_analysis['score'],
        prompt=visual_analysis.get('prompt', '') if should_generate else None,
        reason=visual_analysis.get('reasoning', ''),
        style_hints=visual_analysis.get('style_hints', [])
    )

    # Update bundle if generating
    if should_generate:
        bundle.mark_image_generated()

    return result.model_dump()


# ╭──────────────────────────────────────────────────────────────────────────────╮
# │ System tools                                                                 │
# ╰──────────────────────────────────────────────────────────────────────────────╯

@function_tool
async def generate_universal_updates(
    ctx: RunContextWrapper,
    payload: EmptyInput
) -> Dict[str, Any]:
    """Generate updates across all systems using bundle."""
    broker = await _get_context_broker(ctx)
    bundle = await _get_bundle(ctx)

    # Collect all pending changes from bundle
    updates = await broker.collect_universal_updates(bundle)

    # Apply updates in background with error handling
    task = asyncio.create_task(broker.apply_updates_async(updates))
    task.add_done_callback(_log_task_exc)

    return {
        'world_updates': updates.get('world', {}),
        'npc_updates': updates.get('npcs', {}),
        'memory_consolidation': updates.get('memory', {}),
        'conflict_progression': updates.get('conflicts', {}),
        'lore_evolution': updates.get('lore', {}),
        'applied_async': True
    }


@function_tool
async def check_performance_metrics(
    ctx: RunContextWrapper,
    payload: EmptyInput
) -> Dict[str, Any]:
    """Get performance metrics from bundle's telemetry."""
    bundle = await _get_bundle(ctx)
    metrics = bundle.get_performance_metrics()

    return PerformanceMetrics(
        response_time_ms=metrics.get('response_time', 0),
        tokens_used=metrics.get('tokens', 0),
        cache_hits=metrics.get('cache_hits', 0),
        cache_misses=metrics.get('cache_misses', 0),
        parallel_fetches=metrics.get('parallel_fetches', 0),
        bundle_size_kb=metrics.get('bundle_size', 0),
        sections_loaded=list(bundle.sections.keys())
    ).model_dump()


# ╭──────────────────────────────────────────────────────────────────────────────╮
# │ Advanced narrative tools                                                     │
# ╰──────────────────────────────────────────────────────────────────────────────╯

@function_tool
async def generate_ambient_narration(
    ctx: RunContextWrapper,
    focus: str = "atmosphere",
    intensity: float = 0.5
) -> Dict[str, Any]:
    """Generate ambient narration from bundle's atmosphere data."""
    bundle = await _get_bundle(ctx, expand_sections=['location', 'world'])
    broker = await _get_context_broker(ctx)

    atmosphere_data = {
        'location': bundle.sections.get('location', {}),
        'world': bundle.sections.get('world', {}),
        'active_themes': bundle.get_active_themes(),
        'tension': bundle.get_tension_level()
    }

    narration = await broker.generate_ambient(
        focus=focus,
        intensity=intensity,
        atmosphere_data=atmosphere_data
    )

    return {
        'description': narration['text'],
        'affects_mood': narration.get('mood_impact', False),
        'reflects_systems': narration.get('systems_reflected', [])
    }


@function_tool
async def get_activity_recommendations(
    ctx: RunContextWrapper,
    scenario_type: str = "general",
    npc_ids: Optional[List[int]] = None
) -> Dict[str, Any]:
    """Get activity recommendations using bundle context."""
    bundle = await _get_bundle(ctx, expand_sections=['activities', 'npcs', 'relationships'])
    broker = await _get_context_broker(ctx)

    # Analyze context for appropriate activities
    activity_analysis = await broker.analyze_activity_opportunities(
        bundle=bundle,
        scenario_type=scenario_type,
        npc_ids=npc_ids or []
    )

    # Generate recommendations based on:
    # - Current relationships
    # - Time of day
    # - Recent activities
    # - NPC availability
    recommendations = await broker.generate_activity_recommendations(
        analysis=activity_analysis,
        bundle=bundle,
        limit=5
    )

    return ActivityRecommendationsResult(
        recommendations=recommendations,
        total_available=len(activity_analysis.get('available', [])),
        scenario_context=activity_analysis.get('context', {})
    ).model_dump()


@function_tool
async def detect_narrative_patterns(
    ctx: RunContextWrapper,
    lookback_turns: int = 10
) -> Dict[str, Any]:
    """Detect narrative patterns using bundle's graph analysis."""
    broker = await _get_context_broker(ctx)
    bundle = await _get_bundle(ctx)

    # Analyze patterns across bundle's graph
    patterns = await broker.analyze_narrative_patterns(
        bundle=bundle,
        lookback_turns=lookback_turns
    )

    return {
        'recurring_themes': patterns.get('themes', []),
        'character_arcs': patterns.get('arcs', {}),
        'foreshadowing': patterns.get('foreshadowing', []),
        'emergent_threads': patterns.get('emergent', []),
        'recommendation': patterns.get('next_beat', '')
    }


@function_tool
async def narrate_power_exchange(
    ctx: RunContextWrapper,
    exchange_type: str = "subtle"
) -> Dict[str, Any]:
    """Narrate power dynamics using bundle's relationship data."""
    bundle = await _get_bundle(ctx, expand_sections=['relationships', 'npcs'])
    broker = await _get_context_broker(ctx)

    # Analyze power dynamics from relationships
    dynamics = await broker.analyze_power_dynamics(bundle)

    # Generate power exchange narration
    narration = await broker.narrate_power_exchange(
        exchange_type=exchange_type,
        dynamics=dynamics,
        bundle=bundle
    )

    return {
        'narrative': narration['text'],
        'power_shift': narration.get('shift', 0),
        'participants': narration.get('participants', []),
        'subtext': narration.get('subtext', '')
    }


@function_tool
async def narrate_daily_routine(
    ctx: RunContextWrapper,
    time_period: str = "morning"
) -> Dict[str, Any]:
    """Narrate routine with emergent details from bundle."""
    bundle = await _get_bundle(ctx, expand_sections=['routine', 'npcs'])
    broker = await _get_context_broker(ctx)

    # Get routine data with emergent variations
    routine = await broker.get_routine_with_emergence(
        time_period=time_period,
        bundle=bundle
    )

    return {
        'description': routine['text'],
        'routine_with_dynamics': routine.get('with_dynamics', ''),
        'emergent_details': routine.get('emergent', []),
        'mood_influence': routine.get('mood_factor', 0)
    }


# ╭──────────────────────────────────────────────────────────────────────────────╮
# │ User model tools                                                             │
# ╰──────────────────────────────────────────────────────────────────────────────╯

@function_tool
async def get_user_model_guidance(
    ctx: RunContextWrapper,
    payload: EmptyInput
) -> Dict[str, Any]:
    """Get user model guidance from bundle's analysis."""
    bundle = await _get_bundle(ctx, expand_sections=['user_model'])
    broker = await _get_context_broker(ctx)

    # Analyze user patterns from bundle
    user_data = bundle.sections.get('user_model', {})
    patterns = await broker.analyze_user_patterns(bundle)

    guidance = await broker.generate_user_guidance(
        user_data=user_data,
        patterns=patterns,
        recent_interactions=bundle.get_recent_interactions()
    )

    return UserGuidanceResult(
        preferred_approach=guidance.get('approach', 'standard'),
        tone_suggestions=guidance.get('tone', []),
        topics_to_explore=guidance.get('topics', []),
        boundaries_detected=guidance.get('boundaries', []),
        engagement_level=guidance.get('engagement', 0.5)
    ).model_dump()


@function_tool
async def detect_user_revelations(
    ctx: RunContextWrapper,
    user_input: str
) -> Dict[str, Any]:
    """Detect revelations with bundle context."""
    bundle = await _get_bundle(ctx, expand_sections=['user_model', 'memory'])
    broker = await _get_context_broker(ctx)

    # Analyze input for revelations
    revelations = await broker.detect_revelations(
        input_text=user_input,
        bundle=bundle
    )

    # Store significant revelations
    for rev in revelations:
        if rev.get('significance', 0) > 0.7:
            await broker.store_revelation(rev, bundle)

    return {
        'revelations': revelations,
        'updated_model': bundle.sections.get('user_model', {})
    }


# ╭──────────────────────────────────────────────────────────────────────────────╮
# │ Helper tools                                                                 │
# ╰──────────────────────────────────────────────────────────────────────────────╯

@function_tool
async def expand_context_section(
    ctx: RunContextWrapper,
    section: str,
    depth: str = "standard"
) -> Dict[str, Any]:
    """On-demand expansion of specific context sections."""
    bundle = await _get_bundle(ctx, expand_sections=[section])
    broker = await _get_context_broker(ctx)

    # Expand with requested depth
    expanded = await broker.expand_section_with_depth(
        bundle=bundle,
        section=section,
        depth=depth
    )

    # Get graph links with normalized IDs (guard against None)
    section_links = bundle.get_section_links(section) or {}
    normalized_links = {str(k): [str(v) for v in vals] for k, vals in section_links.items()}

    return {
        'section': section,
        'data': expanded,
        'canon_included': True,
        'graph_links': normalized_links
    }


@function_tool
async def prefetch_next_context(
    ctx: RunContextWrapper,
    predicted_action: str
) -> Dict[str, Any]:
    """Prefetch context for predicted next action."""
    broker = await _get_context_broker(ctx)

    # Predict next scene scope
    predicted_scope = broker.predict_scene_scope(predicted_action)

    # Generate stable task ID
    task_id = str(uuid.uuid4())

    # Prefetch in background with error handling
    task = asyncio.create_task(broker.prefetch_bundle(predicted_scope))
    task.add_done_callback(_log_task_exc)

    # Register task for later status queries
    if hasattr(broker, 'register_task'):
        broker.register_task(task_id, task)

    return {
        'prefetch_started': True,
        'predicted_scope': predicted_scope.to_dict(),
        'task_id': task_id
    }


@function_tool(name_override="lore_handle_operation")
async def lore_handle_operation(
    ctx: RunContextWrapper,
    payload: LoreOperationInput,
) -> Dict[str, Any]:
    """Timeout-protected bridge for lore orchestrator operations."""

    nyx_ctx = ctx.context if isinstance(ctx.context, NyxContext) else ctx
    user_id = getattr(nyx_ctx, "user_id", None)
    conversation_id = getattr(nyx_ctx, "conversation_id", None)

    if user_id is None or conversation_id is None:
        raise ValueError("Nyx context missing identifiers for lore operation")

    forward_payload: Dict[str, Any] = dict(payload) if payload is not None else {}

    # Normalize optional "extra" entries (list of key/value pairs → mapping)
    raw_extra = forward_payload.pop("extra", None)
    normalized_extra: Dict[str, Any] = {}

    if isinstance(raw_extra, Mapping):
        normalized_extra = {str(key): value for key, value in raw_extra.items()}
    elif isinstance(raw_extra, list):
        for item in raw_extra:
            if not isinstance(item, Mapping):
                continue
            key = item.get("key")
            if key is None:
                continue
            normalized_extra[str(key)] = item.get("value")

    if normalized_extra:
        forward_payload["extra"] = normalized_extra

    return await lore_handle_operation_tool_wrapper(
        user_id=int(user_id),
        conversation_id=int(conversation_id),
        payload=forward_payload,
    )


@function_tool(name_override="world_handle_operation")
async def world_handle_operation(
    ctx: RunContextWrapper,
    action_type: str,
    payload: Optional[WorldOperationPayload] = None,
) -> Dict[str, Any]:
    """Governed access point for world orchestrator operations."""

    return await world_handle_operation_tool_wrapper(
        ctx=ctx,
        action_type=action_type,
        payload=payload,
    )


# ╭──────────────────────────────────────────────────────────────────────────────╮
# │ Implementation wrappers & responses                                          │
# ╰──────────────────────────────────────────────────────────────────────────────╯

async def generate_universal_updates_impl(
    ctx: Union[RunContextWrapper, Any],
    narrative: str
) -> Dict[str, Any]:
    """Implementation wrapper for universal updates - handles both old and new context."""
    # Handle both RunContextWrapper and direct NyxContext
    if hasattr(ctx, 'context'):
        app_ctx = ctx.context
    else:
        app_ctx = ctx

    # Create wrapper if needed
    if not isinstance(ctx, RunContextWrapper):
        ctx = RunContextWrapper(context=app_ctx)

    # Direct implementation instead of calling the tool
    broker = await _get_context_broker(ctx)
    bundle = await _get_bundle(ctx)

    # Collect all pending changes from bundle
    updates = await broker.collect_universal_updates(bundle)

    # Apply updates in background with error handling
    task = asyncio.create_task(broker.apply_updates_async(updates))
    task.add_done_callback(_log_task_exc)

    result = {
        'world_updates': updates.get('world', {}),
        'npc_updates': updates.get('npcs', {}),
        'memory_consolidation': updates.get('memory', {}),
        'conflict_progression': updates.get('conflicts', {}),
        'lore_evolution': updates.get('lore', {}),
        'applied_async': True
    }

    # Return a compatible object for the orchestrator
    from .models import UniversalUpdateResult
    return UniversalUpdateResult(
        success=True,
        updates_generated=bool(updates),
        error=None
    )

def ok_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap successful response in standard envelope."""
    return {"ok": True, "data": data}


def error_response(error: str) -> Dict[str, Any]:
    """Wrap error response in standard envelope."""
    return {"ok": False, "error": error}


# ╭──────────────────────────────────────────────────────────────────────────────╮
# │ Compatibility exports                                                        │
# ╰──────────────────────────────────────────────────────────────────────────────╯

# Keep these for backward compatibility
generate_image_from_scene = decide_image_generation
calculate_emotional_impact = calculate_and_update_emotional_state
manage_beliefs = score_decision_options

# Additional compatibility mappings used by other modules
detect_user_revelations_impl = detect_user_revelations
get_user_model_guidance_impl = get_user_model_guidance
