# nyx/nyx_agent/tools.py
"""Function tools for Nyx Agent SDK - Core Tools"""

import json
import time
import logging
from typing import Dict, List, Any, Optional, Union
from contextlib import suppress

from agents import function_tool, RunContextWrapper
from db.connection import get_db_connection_context
from lore.core.canon import (
    find_or_create_npc,
    find_or_create_location, 
    find_or_create_event,
    log_canonical_event,
    create_message,
    update_entity_with_governance,
    ensure_canonical_context
)

from .config import Config
from .models import *
from .utils import (
    _get_app_ctx, _unwrap_tool_ctx, _ensure_world_state, _ensure_world_state_from_ctx,
    _json_safe, _resolve_app_ctx, _ensure_context_map, get_canonical_context,
    _score_scene_text, _build_image_prompt, _calculate_context_relevance,
    _calculate_emotional_alignment, _calculate_pattern_score, _calculate_relationship_impact,
    _get_fallback_decision, _get_memory_emotional_impact, get_context_text_lower,
    _prune_list, _calculate_avg_response_time, _calculate_variance,
    safe_process_metric, get_process_info, bytes_to_mb, extract_runner_response,
    _default_json_encoder
)

logger = logging.getLogger(__name__)

# Import the real narrate_slice_of_life_scene tool
try:
    from story_agent.slice_of_life_narrator import (
        SliceOfLifeNarrator,
        NarrateSliceOfLifeInput,
        SliceOfLifeEvent as SafeSliceOfLifeEvent,
        narrate_slice_of_life_scene as tool_narrate_slice_of_life_scene,
    )
except ImportError:
    logger.warning("Slice-of-life narrator not available")
    tool_narrate_slice_of_life_scene = None

# ===== Power Exchange and Routine Tools =====

@function_tool
async def narrate_power_exchange(
    ctx: RunContextWrapper,
    npc_id: int,
    exchange_type: str = "subtle_control",
    intensity: float = 0.5
) -> str:
    app_ctx = _get_app_ctx(ctx)
    if not app_ctx.slice_of_life_narrator:
        from story_agent.slice_of_life_narrator import SliceOfLifeNarrator
        app_ctx.slice_of_life_narrator = SliceOfLifeNarrator(app_ctx.user_id, app_ctx.conversation_id)
        await app_ctx.slice_of_life_narrator.initialize()

    world_state = await _ensure_world_state_from_ctx(app_ctx)

    from story_agent.world_simulation_models import PowerExchange, PowerDynamicType
    type_map = {
        "subtle_control": PowerDynamicType.SUBTLE_CONTROL,
        "gentle_guidance": PowerDynamicType.GENTLE_GUIDANCE,
        "firm_direction": PowerDynamicType.FIRM_DIRECTION,
        "casual_dominance": PowerDynamicType.CASUAL_DOMINANCE,
        "protective_control": PowerDynamicType.PROTECTIVE_CONTROL,
    }

    exchange = PowerExchange(
        initiator_npc_id=npc_id,
        initiator_id=npc_id,
        initiator_type="npc",
        recipient_type="player",
        recipient_id=1,
        exchange_type=type_map.get(exchange_type, PowerDynamicType.SUBTLE_CONTROL),
        intensity=float(intensity)
    )

    payload = {
        "scene_type": "social",
        "scene": {
            "event_type": "social",
            "title": "power exchange",
            "description": "A brief exchange of subtle control.",
            "location": "current_location",
            "participants": [npc_id],
            "power_exchange": _json_safe(exchange),
        },
        "world_state": _json_safe(world_state),
        "player_action": None,
    }

    return json.dumps({
        "narrator_request": True,
        "payload": payload,
        "hint": "power_exchange"
    }, ensure_ascii=False)

@function_tool
async def narrate_daily_routine(
    ctx: RunContextWrapper,
    activity: str | None = None,
    involved_npcs: list[int] | None = None
) -> dict:
    app_ctx = _unwrap_tool_ctx(ctx)
    if not getattr(app_ctx, "slice_of_life_narrator", None):
        from story_agent.slice_of_life_narrator import SliceOfLifeNarrator
        app_ctx.slice_of_life_narrator = SliceOfLifeNarrator(app_ctx.user_id, app_ctx.conversation_id)
        await app_ctx.slice_of_life_narrator.initialize()

    world_state = await _ensure_world_state(app_ctx)
    participants = (involved_npcs or [])[:3]

    scene_payload = {
        "event_type": "routine",
        "title": "daily routine",
        "description": activity or "A quiet, habitual moment.",
        "location": "current_location",
        "participants": participants,
    }

    payload = {
        "scene_type": "routine",
        "scene": _json_safe(scene_payload),
        "world_state": _json_safe(world_state),
        "player_action": None,
    }

    return {
        "narrator_request": True,
        "payload": payload
    }

# ===== Ambient and Pattern Detection Tools =====

@function_tool
async def generate_ambient_narration(
    ctx: RunContextWrapper,
    focus: str = "atmosphere",
    intensity: float = 0.5
) -> str:
    intensity = max(0.0, min(1.0, float(intensity)))
    app_ctx = _get_app_ctx(ctx)

    if not app_ctx.slice_of_life_narrator:
        from story_agent.slice_of_life_narrator import SliceOfLifeNarrator
        app_ctx.slice_of_life_narrator = SliceOfLifeNarrator(app_ctx.user_id, app_ctx.conversation_id)
        await app_ctx.slice_of_life_narrator.initialize()

    narrator = app_ctx.slice_of_life_narrator
    world_state = await _ensure_world_state_from_ctx(app_ctx)
    ws = _json_safe(world_state)

    prompt = (
        "Write one or two concise sentences of ambient narration reflecting the current world mood.\n"
        "Return STRICT JSON only with keys: {\"ambient\":\"...\"}\n\n"
        f"Focus: {focus}\n"
        f"Intensity: {intensity}\n"
        f"World State: {json.dumps(ws, ensure_ascii=False)}\n"
    )

    from agents import Runner
    resp = await Runner.run(narrator.ambient_narrator, prompt, context=narrator.context)
    raw = extract_runner_response(resp)

    try:
        data = json.loads(raw)
        if not isinstance(data, dict) or "ambient" not in data:
            raise ValueError("bad shape")
    except Exception:
        data = {"ambient": raw.strip() or "The air hums with a quiet, watchful tension."}

    return json.dumps(data, ensure_ascii=False)

@function_tool  
async def detect_narrative_patterns(ctx: RunContextWrapper) -> str:
    """Detect emergent narrative patterns using the narrator."""
    app_ctx = _get_app_ctx(ctx)
    
    if not app_ctx.slice_of_life_narrator:
        from story_agent.slice_of_life_narrator import SliceOfLifeNarrator
        app_ctx.slice_of_life_narrator = SliceOfLifeNarrator(
            app_ctx.user_id,
            app_ctx.conversation_id
        )
        await app_ctx.slice_of_life_narrator.initialize()
    
    narrator = app_ctx.slice_of_life_narrator
    patterns = await narrator.generate_emergent_narrative()
    
    return json.dumps(patterns)

# ===== NPC Dialogue Tool =====

@function_tool
async def generate_npc_dialogue(
    ctx: RunContextWrapper,
    npc_id: int,
    situation: str
) -> str:
    app_ctx = _get_app_ctx(ctx)

    canonical_ctx = ensure_canonical_context({
        'user_id': app_ctx.user_id,
        'conversation_id': app_ctx.conversation_id
    })

    world_state = await _ensure_world_state_from_ctx(app_ctx)

    payload = {
        "scene_type": "social",
        "scene": {
            "event_type": "social",
            "title": "npc dialogue",
            "description": f"NPC {npc_id} responds: {situation}",
            "location": "current_location",
            "participants": [npc_id],
            "dialogue_request": {
                "npc_id": npc_id,
                "situation": situation
            }
        },
        "world_state": _json_safe(world_state),
        "player_action": None,
    }

    return json.dumps({
        "narrator_request": True,
        "payload": payload,
        "hint": "dialogue"
    }, ensure_ascii=False)

# ===== Memory Tools =====

@function_tool
async def retrieve_memories(ctx: RunContextWrapper, payload: RetrieveMemoriesInput) -> str:
    """Retrieve relevant memories for Nyx."""
    data = RetrieveMemoriesInput.model_validate(payload or {})
    query = data.query
    limit = data.limit
    memory_system = ctx.context.memory_system
    
    result = await memory_system.recall(
        entity_type=Config.ENTITY_TYPE_INTEGRATED,
        entity_id=0,
        query=query,
        limit=limit
    )
    
    memories_raw = result.get("memories", [])
    memories = [
        MemoryItem(
            id=str(m.get("id") or m.get("memory_id") or ""),
            text=m["text"],
            relevance=float(m.get("relevance", 0.0)),
            tags=m.get("tags", [])
        )
        for m in memories_raw
    ]
    
    formatted_memories = []
    for memory in memories:
        relevance = memory.relevance
        confidence_marker = "vividly recall" if relevance > Config.VIVID_RECALL_THRESHOLD else \
                          "remember" if relevance > Config.REMEMBER_THRESHOLD else \
                          "think I recall" if relevance > Config.THINK_RECALL_THRESHOLD else \
                          "vaguely remember"
        
        formatted_memories.append(f"I {confidence_marker}: {memory.text}")
    
    formatted_text = "\n".join(formatted_memories) if formatted_memories else "No relevant memories found."

    return MemorySearchResult(
        memories=memories,
        formatted_text=formatted_text
    ).model_dump_json()

@function_tool
async def add_memory(ctx: RunContextWrapper, payload: AddMemoryInput) -> str:
    """Add a new memory for Nyx using canonical system."""
    data = AddMemoryInput.model_validate(payload or {})
    memory_text = data.memory_text
    memory_type = data.memory_type
    significance = data.significance
    
    canonical_ctx = ensure_canonical_context(ctx.context)
    
    async with get_db_connection_context() as conn:
        await log_canonical_event(
            canonical_ctx, conn,
            f"Memory stored: {memory_text[:100]}...",
            tags=["memory", memory_type, "nyx_generated"],
            significance=significance
        )
    
    memory_system = ctx.context.memory_system
    result = await memory_system.remember(
        entity_type="integrated",
        entity_id=0,
        memory_text=memory_text,
        importance="high" if significance >= 8 else "medium",
        emotional=True,
        tags=["agent_generated", memory_type]
    )
    
    return MemoryStorageResult(
        memory_id=str(result.get("memory_id", "unknown")),
        success=True
    ).model_dump_json()

# ===== User Model Tools =====

@function_tool
async def get_user_model_guidance(ctx: RunContextWrapper, payload: EmptyInput) -> str:
    """Get guidance for how Nyx should respond based on the user model."""
    _ = payload  # unused
    user_model_manager = ctx.context.user_model
    guidance = await user_model_manager.get_response_guidance()
    
    top_kinks = guidance.get("top_kinks", [])
    behavior_patterns = guidance.get("behavior_patterns", {})
    suggested_intensity = guidance.get("suggested_intensity", 0.5)
    reflections = guidance.get("reflections", [])
    
    return UserGuidanceResult(
        top_kinks=top_kinks,
        behavior_patterns=dict_to_kvlist(behavior_patterns),
        suggested_intensity=suggested_intensity,
        reflections=reflections
    ).model_dump_json()

@function_tool
async def detect_user_revelations(ctx: RunContextWrapper, payload: DetectUserRevelationsInput) -> str:
    """Detect if user is revealing new preferences or patterns."""
    data = DetectUserRevelationsInput.model_validate(payload or {})
    user_message = data.user_message
    lower_message = user_message.lower()
    revelations = []
    
    kink_keywords = {
        "ass": ["ass", "booty", "behind", "rear"],
        "feet": ["feet", "foot", "toes"],
        "goth": ["goth", "gothic", "dark", "black clothes"],
        "tattoos": ["tattoo", "ink", "inked"],
        "piercings": ["piercing", "pierced", "stud", "ring"],
        "latex": ["latex", "rubber", "shiny"],
        "leather": ["leather", "leathery"],
        "humiliation": ["humiliate", "embarrassed", "ashamed", "pathetic"],
        "submission": ["submit", "obey", "serve", "kneel"]
    }
    
    for kink, keywords in kink_keywords.items():
        if any(keyword in lower_message for keyword in keywords):
            sentiment = "neutral"
            pos_words = ["like", "love", "enjoy", "good", "great", "nice", "yes", "please"]
            neg_words = ["don't", "hate", "dislike", "bad", "worse", "no", "never"]
            
            pos_count = sum(1 for word in pos_words if word in lower_message)
            neg_count = sum(1 for word in neg_words if word in lower_message)
            
            if pos_count > neg_count:
                sentiment = "positive"
                intensity = 0.7
            elif neg_count > pos_count:
                sentiment = "negative" 
                intensity = 0.0
            else:
                intensity = 0.4
                
            revelation_data = {
                "type": "kink_preference",
                "kink": kink,
                "intensity": intensity,
                "source": "explicit_negative_mention" if sentiment == "negative" else "explicit_mention",
                "sentiment": sentiment
            }
            
            revelations.append(dict_to_kvlist(revelation_data))
    
    if "don't tell me what to do" in lower_message or "i won't" in lower_message:
        revelation_data = {
            "type": "behavior_pattern",
            "pattern": "resistance",
            "intensity": 0.6,
            "source": "explicit_statement"
        }
        revelations.append(dict_to_kvlist(revelation_data))
    
    if "yes mistress" in lower_message or "i'll obey" in lower_message:
        revelation_data = {
            "type": "behavior_pattern",
            "pattern": "submission",
            "intensity": 0.8,
            "source": "explicit_statement"
        }
        revelations.append(dict_to_kvlist(revelation_data))
    
    # Save revelations to database if found
    if revelations and ctx.context.user_model:
        for revelation_kv in revelations:
            revelation = kvlist_to_dict(revelation_kv)
            if revelation["type"] == "kink_preference":
                await ctx.context.user_model.update_kink_preference(
                    revelation["kink"],
                    revelation["intensity"],
                    revelation["source"]
                )
    
    return RevelationDetectionResult(
        revelations=revelations,
        has_revelations=len(revelations) > 0
    ).model_dump_json()

# ===== Emotional Tools =====

@function_tool
async def calculate_and_update_emotional_state(ctx: RunContextWrapper, payload: CalculateEmotionalStateInput) -> str:
    """Calculate emotional impact and immediately update the emotional state."""
    data = CalculateEmotionalStateInput.model_validate(payload or {})
    context_dict = kvlist_to_dict(data.context)

    # First calculate the new state
    result = await calculate_emotional_impact(ctx, data)
    emotional_data = json.loads(result)
    
    # Immediately update the context with the new state
    ctx.context.emotional_state.update({
        "valence": emotional_data["valence"],
        "arousal": emotional_data["arousal"],
        "dominance": emotional_data["dominance"]
    })
    
    # Save to database with its own connection
    async with get_db_connection_context() as conn:
        await conn.execute("""
            INSERT INTO NyxAgentState (user_id, conversation_id, emotional_state, updated_at)
            VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
            ON CONFLICT (user_id, conversation_id) 
            DO UPDATE SET emotional_state = $3, updated_at = CURRENT_TIMESTAMP
        """, ctx.context.user_id, ctx.context.conversation_id, json.dumps(ctx.context.emotional_state, ensure_ascii=False))
    
    # Return the result with confirmation of update
    emotional_data["state_updated"] = True
    
    return EmotionalCalculationResult(
        valence=emotional_data["valence"],
        arousal=emotional_data["arousal"],
        dominance=emotional_data["dominance"],
        primary_emotion=emotional_data["primary_emotion"],
        changes=EmotionalChanges(
            valence_change=emotional_data["changes"]["valence_change"],
            arousal_change=emotional_data["changes"]["arousal_change"],
            dominance_change=emotional_data["changes"]["dominance_change"]
        ),
        state_updated=True
    ).model_dump_json()

@function_tool
async def calculate_emotional_impact(ctx: RunContextWrapper, payload: CalculateEmotionalStateInput) -> str:
    """Calculate emotional impact of current context using the emotional core system."""
    data = CalculateEmotionalStateInput.model_validate(payload or {})
    context_dict = kvlist_to_dict(data.context)
    current_state = ctx.context.emotional_state.copy()
    
    # Calculate emotional changes based on context
    valence_change = 0.0
    arousal_change = 0.0
    dominance_change = 0.0
    
    # Analyze context for emotional triggers
    context_text_lower = get_context_text_lower(context_dict)
    
    if "conflict" in context_text_lower:
        arousal_change += 0.2
        valence_change -= 0.1
    if "submission" in context_text_lower:
        dominance_change += 0.1
        arousal_change += 0.1
    if "praise" in context_text_lower or "good" in context_text_lower:
        valence_change += 0.2
    if "resistance" in context_text_lower:
        arousal_change += 0.15
        dominance_change -= 0.05
    
    # Get memory emotional impact
    memory_impact = await _get_memory_emotional_impact(ctx, context_dict)
    valence_change += memory_impact["valence"] * 0.3
    arousal_change += memory_impact["arousal"] * 0.3
    dominance_change += memory_impact["dominance"] * 0.3
    
    # Use EmotionalCore if available for more nuanced analysis
    if ctx.context.emotional_core:
        try:
            core_analysis = ctx.context.emotional_core.analyze(str(context_dict))
            valence_change += core_analysis.get("valence_delta", 0) * 0.5
            arousal_change += core_analysis.get("arousal_delta", 0) * 0.5
        except Exception as e:
            logger.debug(f"EmotionalCore analysis failed: {e}", exc_info=True)
    
    # Apply changes with bounds
    new_valence = max(-1, min(1, current_state["valence"] + valence_change))
    new_arousal = max(0, min(1, current_state["arousal"] + arousal_change))
    new_dominance = max(0, min(1, current_state["dominance"] + dominance_change))
    
    # Determine primary emotion based on VAD model
    primary_emotion = "neutral"
    if new_valence > Config.POSITIVE_VALENCE_THRESHOLD and new_arousal > Config.POSITIVE_VALENCE_THRESHOLD:
        primary_emotion = "excited"
    elif new_valence > Config.POSITIVE_VALENCE_THRESHOLD and new_arousal < Config.POSITIVE_VALENCE_THRESHOLD:
        primary_emotion = "content"
    elif new_valence < Config.NEGATIVE_VALENCE_THRESHOLD and new_arousal > Config.POSITIVE_VALENCE_THRESHOLD:
        primary_emotion = "frustrated"
    elif new_valence < Config.NEGATIVE_VALENCE_THRESHOLD and new_arousal < Config.POSITIVE_VALENCE_THRESHOLD:
        primary_emotion = "disappointed"
    elif new_dominance > Config.HIGH_DOMINANCE_THRESHOLD:
        primary_emotion = "commanding"
    
    return EmotionalCalculationResult(
        valence=new_valence,
        arousal=new_arousal,
        dominance=new_dominance,
        primary_emotion=primary_emotion,
        changes=EmotionalChanges(
            valence_change=valence_change,
            arousal_change=arousal_change,
            dominance_change=dominance_change
        ),
        state_updated=None
    ).model_dump_json()

# ===== Relationship Tools =====

@function_tool
async def update_relationship_state(
    ctx: RunContextWrapper,
    payload: UpdateRelationshipStateInput
) -> str:
    """Update relationship state canonically with full governance integration."""
    data = UpdateRelationshipStateInput.model_validate(payload or {})
    entity_id = data.entity_id
    trust_change = data.trust_change
    power_change = data.power_change
    bond_change = data.bond_change
    
    relationships = ctx.context.relationship_states
    
    # Create canonical context
    canonical_ctx = ensure_canonical_context({
        'user_id': ctx.context.user_id,
        'conversation_id': ctx.context.conversation_id
    })
    
    # Initialize relationship if it doesn't exist
    if entity_id not in relationships:
        relationships[entity_id] = {
            "trust": 0.5,
            "power_dynamic": 0.5,
            "emotional_bond": 0.3,
            "interaction_count": 0,
            "last_interaction": time.time(),
            "type": "neutral"
        }
    
    rel = relationships[entity_id]
    old_state = {
        "trust": rel["trust"],
        "power_dynamic": rel["power_dynamic"],
        "emotional_bond": rel["emotional_bond"]
    }
    
    # Apply changes with bounds checking
    rel["trust"] = max(0, min(1, rel["trust"] + trust_change))
    rel["power_dynamic"] = max(0, min(1, rel["power_dynamic"] + power_change))
    rel["emotional_bond"] = max(0, min(1, rel["emotional_bond"] + bond_change))
    rel["interaction_count"] += 1
    rel["last_interaction"] = time.time()
    
    # Determine relationship type based on new values
    if rel["trust"] > Config.INTIMATE_TRUST_THRESHOLD and rel["emotional_bond"] > Config.INTIMATE_BOND_THRESHOLD:
        rel["type"] = "intimate"
    elif rel["trust"] > Config.FRIENDLY_TRUST_THRESHOLD:
        rel["type"] = "friendly"
    elif rel["trust"] < Config.HOSTILE_TRUST_THRESHOLD:
        rel["type"] = "hostile"
    elif rel["power_dynamic"] > Config.DOMINANT_POWER_THRESHOLD:
        rel["type"] = "dominant"
    elif rel["power_dynamic"] < Config.SUBMISSIVE_POWER_THRESHOLD:
        rel["type"] = "submissive"
    else:
        rel["type"] = "neutral"
    
    # Calculate significance of changes for canonical logging
    total_change = abs(trust_change) + abs(power_change) + abs(bond_change)
    significance = 5  # base
    if total_change > 0.3:
        significance = 7  # major change
    if total_change > 0.5:
        significance = 8  # dramatic change
    if rel["type"] != "neutral":
        significance += 1  # relationship has clear type
    
    # Save to database with canonical integration
    async with get_db_connection_context() as conn:
        try:
            # First, ensure the entity exists canonically (if it's an NPC)
            if entity_id.isdigit():  # Assume numeric entity_id means NPC
                npc_id = int(entity_id)
                
                # Get NPC details for canonical creation
                npc_data = await conn.fetchrow("""
                    SELECT npc_name, role, affiliations 
                    FROM NPCStats 
                    WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
                """, npc_id, ctx.context.user_id, ctx.context.conversation_id)
                
                if npc_data:
                    # Ensure NPC exists in canonical system
                    canonical_npc_id = await find_or_create_npc(
                        canonical_ctx, conn,
                        npc_name=npc_data['npc_name'],
                        role=npc_data.get('role', 'individual'),
                        affiliations=npc_data.get('affiliations', [])
                    )
            
            # Get existing evolution history
            existing = await conn.fetchrow("""
                SELECT evolution_history 
                FROM RelationshipEvolution 
                WHERE user_id = $1 AND conversation_id = $2 
                    AND npc1_id = $3 AND entity2_type = $4 AND entity2_id = $5
            """, ctx.context.user_id, ctx.context.conversation_id, 0, "entity", entity_id)
            
            # Build evolution history
            evolution_history = []
            if existing and existing["evolution_history"]:
                try:
                    evolution_history = json.loads(existing["evolution_history"])
                except (json.JSONDecodeError, TypeError):
                    evolution_history = []
            
            # Add new entry (keep last 50 entries)
            evolution_entry = {
                "timestamp": time.time(),
                "trust": rel["trust"],
                "power": rel["power_dynamic"],
                "bond": rel["emotional_bond"],
                "changes": {
                    "trust": trust_change,
                    "power": power_change,
                    "bond": bond_change
                },
                "interaction_count": rel["interaction_count"],
                "old_type": old_state.get("type", "neutral"),
                "new_type": rel["type"]
            }
            evolution_history.append(evolution_entry)
            evolution_history = evolution_history[-50:]  # Keep last 50
            
            # Use canonical update system
            update_result = await update_entity_with_governance(
                canonical_ctx, conn,
                entity_type="RelationshipEvolution",
                entity_id=0,  # Will be handled by upsert logic
                updates={
                    "relationship_type": rel["type"],
                    "current_stage": rel["type"], 
                    "progress_to_next": min(1.0, total_change * 2),  # Scale change to progress
                    "evolution_history": json.dumps(evolution_history),
                    "trust_level": rel["trust"],
                    "power_dynamic": rel["power_dynamic"],
                    "emotional_bond": rel["emotional_bond"],
                    "last_interaction": rel["last_interaction"]
                },
                reason=f"Relationship evolution: trust{trust_change:+.3f}, power{power_change:+.3f}, bond{bond_change:+.3f}",
                significance=significance
            )
            
            # If canonical update failed, fall back to direct update
            if not update_result.get("success"):
                await conn.execute("""
                    INSERT INTO RelationshipEvolution 
                    (user_id, conversation_id, npc1_id, entity2_type, entity2_id, 
                     relationship_type, current_stage, progress_to_next, evolution_history)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (user_id, conversation_id, npc1_id, entity2_type, entity2_id)
                    DO UPDATE SET 
                        relationship_type = $6,
                        current_stage = $7,
                        progress_to_next = $8,
                        evolution_history = $9
                """, ctx.context.user_id, ctx.context.conversation_id, 0, "entity", entity_id,
                     rel["type"], rel["type"], min(1.0, total_change * 2), json.dumps(evolution_history))
            
            # Log the relationship change canonically
            change_description = []
            if abs(trust_change) > 0.01:
                change_description.append(f"trust {trust_change:+.3f}")
            if abs(power_change) > 0.01:
                change_description.append(f"power {power_change:+.3f}")
            if abs(bond_change) > 0.01:
                change_description.append(f"bond {bond_change:+.3f}")
            
            if change_description:
                await log_canonical_event(
                    canonical_ctx, conn,
                    f"Relationship with {entity_id} evolved: {', '.join(change_description)} → {rel['type']} relationship",
                    tags=["relationship", "evolution", rel["type"], "nyx_update"],
                    significance=significance
                )
            
            # Log relationship type changes
            if old_state.get("type", "neutral") != rel["type"]:
                await log_canonical_event(
                    canonical_ctx, conn,
                    f"Relationship type shifted: {old_state.get('type', 'neutral')} → {rel['type']} with {entity_id}",
                    tags=["relationship", "type_change", rel["type"], "milestone"],
                    significance=significance + 1
                )
            
            # Check for significant thresholds crossed
            threshold_events = []
            
            # Trust thresholds
            if old_state["trust"] < Config.INTIMATE_TRUST_THRESHOLD <= rel["trust"]:
                threshold_events.append("intimate_trust_achieved")
            elif old_state["trust"] >= Config.HOSTILE_TRUST_THRESHOLD > rel["trust"]:
                threshold_events.append("trust_broken")
            
            # Power thresholds  
            if old_state["power_dynamic"] < Config.DOMINANT_POWER_THRESHOLD <= rel["power_dynamic"]:
                threshold_events.append("dominance_established")
            elif old_state["power_dynamic"] >= Config.SUBMISSIVE_POWER_THRESHOLD > rel["power_dynamic"]:
                threshold_events.append("submission_deepened")
            
            # Bond thresholds
            if old_state["emotional_bond"] < Config.INTIMATE_BOND_THRESHOLD <= rel["emotional_bond"]:
                threshold_events.append("deep_bond_formed")
            
            # Log threshold crossings
            for event in threshold_events:
                await log_canonical_event(
                    canonical_ctx, conn,
                    f"Relationship milestone: {event.replace('_', ' ')} with {entity_id}",
                    tags=["relationship", "milestone", event, entity_id],
                    significance=9  # Milestones are highly significant
                )
            
        except Exception as e:
            logger.error(f"Error in canonical relationship update: {e}", exc_info=True)
            # Continue with function to avoid breaking the flow
    
    # Create relationship changes for response
    changes = RelationshipChanges(
        trust=trust_change,
        power=power_change,
        bond=bond_change
    )
    
    # Create relationship state output
    relationship_state_out = RelationshipStateOut(
        trust=rel["trust"],
        power_dynamic=rel["power_dynamic"],
        emotional_bond=rel["emotional_bond"],
        interaction_count=rel["interaction_count"],
        last_interaction=rel["last_interaction"],
        type=rel["type"]
    )
    
    # Build result
    result = RelationshipUpdateResult(
        entity_id=entity_id,
        relationship=relationship_state_out,
        changes=changes
    )
    
    # Store evolution pattern for learning
    pattern_data = {
        "entity_id": entity_id,
        "change_magnitude": total_change,
        "relationship_type": rel["type"],
        "direction": "positive" if total_change > 0 else "negative" if total_change < 0 else "neutral",
        "trust_direction": "up" if trust_change > 0 else "down" if trust_change < 0 else "stable",
        "power_direction": "up" if power_change > 0 else "down" if power_change < 0 else "stable",
        "bond_direction": "up" if bond_change > 0 else "down" if bond_change < 0 else "stable"
    }
    
    # Learn from this relationship interaction
    await ctx.context.learn_from_interaction(
        action=f"relationship_update_{rel['type']}",
        outcome=f"total_change_{total_change:.2f}",
        success=total_change > 0.1  # Consider it successful if meaningful change occurred
    )
    
    return result.model_dump_json()

# ===== Emotional Tools =====

@function_tool
async def calculate_and_update_emotional_state(ctx: RunContextWrapper, payload: CalculateEmotionalStateInput) -> str:
    """Calculate emotional impact and immediately update the emotional state."""
    data = CalculateEmotionalStateInput.model_validate(payload or {})
    context_dict = kvlist_to_dict(data.context)

    # First calculate the new state
    result = await calculate_emotional_impact(ctx, data)
    emotional_data = json.loads(result)
    
    # Immediately update the context with the new state
    ctx.context.emotional_state.update({
        "valence": emotional_data["valence"],
        "arousal": emotional_data["arousal"],
        "dominance": emotional_data["dominance"]
    })
    
    # Save to database with its own connection
    async with get_db_connection_context() as conn:
        await conn.execute("""
            INSERT INTO NyxAgentState (user_id, conversation_id, emotional_state, updated_at)
            VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
            ON CONFLICT (user_id, conversation_id) 
            DO UPDATE SET emotional_state = $3, updated_at = CURRENT_TIMESTAMP
        """, ctx.context.user_id, ctx.context.conversation_id, json.dumps(ctx.context.emotional_state, ensure_ascii=False))
    
    # Return the result with confirmation of update
    emotional_data["state_updated"] = True
    
    return EmotionalCalculationResult(
        valence=emotional_data["valence"],
        arousal=emotional_data["arousal"],
        dominance=emotional_data["dominance"],
        primary_emotion=emotional_data["primary_emotion"],
        changes=EmotionalChanges(
            valence_change=emotional_data["changes"]["valence_change"],
            arousal_change=emotional_data["changes"]["arousal_change"],
            dominance_change=emotional_data["changes"]["dominance_change"]
        ),
        state_updated=True
    ).model_dump_json()

@function_tool
async def calculate_emotional_impact(ctx: RunContextWrapper, payload: CalculateEmotionalStateInput) -> str:
    """Calculate emotional impact of current context using the emotional core system."""
    data = CalculateEmotionalStateInput.model_validate(payload or {})
    context_dict = kvlist_to_dict(data.context)
    current_state = ctx.context.emotional_state.copy()
    
    # Calculate emotional changes based on context
    valence_change = 0.0
    arousal_change = 0.0
    dominance_change = 0.0
    
    # Analyze context for emotional triggers
    context_text_lower = get_context_text_lower(context_dict)
    
    if "conflict" in context_text_lower:
        arousal_change += 0.2
        valence_change -= 0.1
    if "submission" in context_text_lower:
        dominance_change += 0.1
        arousal_change += 0.1
    if "praise" in context_text_lower or "good" in context_text_lower:
        valence_change += 0.2
    if "resistance" in context_text_lower:
        arousal_change += 0.15
        dominance_change -= 0.05
    
    # Get memory emotional impact
    memory_impact = await _get_memory_emotional_impact(ctx, context_dict)
    valence_change += memory_impact["valence"] * 0.3
    arousal_change += memory_impact["arousal"] * 0.3
    dominance_change += memory_impact["dominance"] * 0.3
    
    # Use EmotionalCore if available for more nuanced analysis
    if ctx.context.emotional_core:
        try:
            core_analysis = ctx.context.emotional_core.analyze(str(context_dict))
            valence_change += core_analysis.get("valence_delta", 0) * 0.5
            arousal_change += core_analysis.get("arousal_delta", 0) * 0.5
        except Exception as e:
            logger.debug(f"EmotionalCore analysis failed: {e}", exc_info=True)
    
    # Apply changes with bounds
    new_valence = max(-1, min(1, current_state["valence"] + valence_change))
    new_arousal = max(0, min(1, current_state["arousal"] + arousal_change))
    new_dominance = max(0, min(1, current_state["dominance"] + dominance_change))
    
    # Determine primary emotion based on VAD model
    primary_emotion = "neutral"
    if new_valence > Config.POSITIVE_VALENCE_THRESHOLD and new_arousal > Config.POSITIVE_VALENCE_THRESHOLD:
        primary_emotion = "excited"
    elif new_valence > Config.POSITIVE_VALENCE_THRESHOLD and new_arousal < Config.POSITIVE_VALENCE_THRESHOLD:
        primary_emotion = "content"
    elif new_valence < Config.NEGATIVE_VALENCE_THRESHOLD and new_arousal > Config.POSITIVE_VALENCE_THRESHOLD:
        primary_emotion = "frustrated"
    elif new_valence < Config.NEGATIVE_VALENCE_THRESHOLD and new_arousal < Config.POSITIVE_VALENCE_THRESHOLD:
        primary_emotion = "disappointed"
    elif new_dominance > Config.HIGH_DOMINANCE_THRESHOLD:
        primary_emotion = "commanding"
    
    return EmotionalCalculationResult(
        valence=new_valence,
        arousal=new_arousal,
        dominance=new_dominance,
        primary_emotion=primary_emotion,
        changes=EmotionalChanges(
            valence_change=valence_change,
            arousal_change=arousal_change,
            dominance_change=dominance_change
        ),
        state_updated=None
    ).model_dump_json()

# ===== Relationship Tools =====

@function_tool
async def update_relationship_state(
    ctx: RunContextWrapper,
    payload: UpdateRelationshipStateInput
) -> str:
    """Update relationship state canonically with full governance integration."""
    data = UpdateRelationshipStateInput.model_validate(payload or {})
    entity_id = data.entity_id
    trust_change = data.trust_change
    power_change = data.power_change
    bond_change = data.bond_change
    
    relationships = ctx.context.relationship_states
    
    # Create canonical context
    canonical_ctx = ensure_canonical_context({
        'user_id': ctx.context.user_id,
        'conversation_id': ctx.context.conversation_id
    })
    
    # Initialize relationship if it doesn't exist
    if entity_id not in relationships:
        relationships[entity_id] = {
            "trust": 0.5,
            "power_dynamic": 0.5,
            "emotional_bond": 0.3,
            "interaction_count": 0,
            "last_interaction": time.time(),
            "type": "neutral"
        }
    
    rel = relationships[entity_id]
    old_state = {
        "trust": rel["trust"],
        "power_dynamic": rel["power_dynamic"],
        "emotional_bond": rel["emotional_bond"]
    }
    
    # Apply changes with bounds checking
    rel["trust"] = max(0, min(1, rel["trust"] + trust_change))
    rel["power_dynamic"] = max(0, min(1, rel["power_dynamic"] + power_change))
    rel["emotional_bond"] = max(0, min(1, rel["emotional_bond"] + bond_change))
    rel["interaction_count"] += 1
    rel["last_interaction"] = time.time()
    
    # Determine relationship type based on new values
    if rel["trust"] > Config.INTIMATE_TRUST_THRESHOLD and rel["emotional_bond"] > Config.INTIMATE_BOND_THRESHOLD:
        rel["type"] = "intimate"
    elif rel["trust"] > Config.FRIENDLY_TRUST_THRESHOLD:
        rel["type"] = "friendly"
    elif rel["trust"] < Config.HOSTILE_TRUST_THRESHOLD:
        rel["type"] = "hostile"
    elif rel["power_dynamic"] > Config.DOMINANT_POWER_THRESHOLD:
        rel["type"] = "dominant"
    elif rel["power_dynamic"] < Config.SUBMISSIVE_POWER_THRESHOLD:
        rel["type"] = "submissive"
    else:
        rel["type"] = "neutral"
    
    # Calculate significance of changes for canonical logging
    total_change = abs(trust_change) + abs(power_change) + abs(bond_change)
    significance = 5  # base
    if total_change > 0.3:
        significance = 7  # major change
    if total_change > 0.5:
        significance = 8  # dramatic change
    if rel["type"] != "neutral":
        significance += 1  # relationship has clear type
    
    # Save to database with canonical integration
    async with get_db_connection_context() as conn:
        try:
            # First, ensure the entity exists canonically (if it's an NPC)
            if entity_id.isdigit():  # Assume numeric entity_id means NPC
                npc_id = int(entity_id)
                
                # Get NPC details for canonical creation
                npc_data = await conn.fetchrow("""
                    SELECT npc_name, role, affiliations 
                    FROM NPCStats 
                    WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
                """, npc_id, ctx.context.user_id, ctx.context.conversation_id)
                
                if npc_data:
                    # Ensure NPC exists in canonical system
                    canonical_npc_id = await find_or_create_npc(
                        canonical_ctx, conn,
                        npc_name=npc_data['npc_name'],
                        role=npc_data.get('role', 'individual'),
                        affiliations=npc_data.get('affiliations', [])
                    )
            
            # Get existing evolution history
            existing = await conn.fetchrow("""
                SELECT evolution_history 
                FROM RelationshipEvolution 
                WHERE user_id = $1 AND conversation_id = $2 
                    AND npc1_id = $3 AND entity2_type = $4 AND entity2_id = $5
            """, ctx.context.user_id, ctx.context.conversation_id, 0, "entity", entity_id)
            
            # Build evolution history
            evolution_history = []
            if existing and existing["evolution_history"]:
                try:
                    evolution_history = json.loads(existing["evolution_history"])
                except (json.JSONDecodeError, TypeError):
                    evolution_history = []
            
            # Add new entry (keep last 50 entries)
            evolution_entry = {
                "timestamp": time.time(),
                "trust": rel["trust"],
                "power": rel["power_dynamic"],
                "bond": rel["emotional_bond"],
                "changes": {
                    "trust": trust_change,
                    "power": power_change,
                    "bond": bond_change
                },
                "interaction_count": rel["interaction_count"],
                "old_type": old_state.get("type", "neutral"),
                "new_type": rel["type"]
            }
            evolution_history.append(evolution_entry)
            evolution_history = evolution_history[-50:]  # Keep last 50
            
            # Use canonical update system
            update_result = await update_entity_with_governance(
                canonical_ctx, conn,
                entity_type="RelationshipEvolution",
                entity_id=0,  # Will be handled by upsert logic
                updates={
                    "relationship_type": rel["type"],
                    "current_stage": rel["type"], 
                    "progress_to_next": min(1.0, total_change * 2),  # Scale change to progress
                    "evolution_history": json.dumps(evolution_history),
                    "trust_level": rel["trust"],
                    "power_dynamic": rel["power_dynamic"],
                    "emotional_bond": rel["emotional_bond"],
                    "last_interaction": rel["last_interaction"]
                },
                reason=f"Relationship evolution: trust{trust_change:+.3f}, power{power_change:+.3f}, bond{bond_change:+.3f}",
                significance=significance
            )
            
            # If canonical update failed, fall back to direct update
            if not update_result.get("success"):
                await conn.execute("""
                    INSERT INTO RelationshipEvolution 
                    (user_id, conversation_id, npc1_id, entity2_type, entity2_id, 
                     relationship_type, current_stage, progress_to_next, evolution_history)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (user_id, conversation_id, npc1_id, entity2_type, entity2_id)
                    DO UPDATE SET 
                        relationship_type = $6,
                        current_stage = $7,
                        progress_to_next = $8,
                        evolution_history = $9
                """, ctx.context.user_id, ctx.context.conversation_id, 0, "entity", entity_id,
                     rel["type"], rel["type"], min(1.0, total_change * 2), json.dumps(evolution_history))
            
            # Log the relationship change canonically
            change_description = []
            if abs(trust_change) > 0.01:
                change_description.append(f"trust {trust_change:+.3f}")
            if abs(power_change) > 0.01:
                change_description.append(f"power {power_change:+.3f}")
            if abs(bond_change) > 0.01:
                change_description.append(f"bond {bond_change:+.3f}")
            
            if change_description:
                await log_canonical_event(
                    canonical_ctx, conn,
                    f"Relationship with {entity_id} evolved: {', '.join(change_description)} → {rel['type']} relationship",
                    tags=["relationship", "evolution", rel["type"], "nyx_update"],
                    significance=significance
                )
            
            # Log relationship type changes
            if old_state.get("type", "neutral") != rel["type"]:
                await log_canonical_event(
                    canonical_ctx, conn,
                    f"Relationship type shifted: {old_state.get('type', 'neutral')} → {rel['type']} with {entity_id}",
                    tags=["relationship", "type_change", rel["type"], "milestone"],
                    significance=significance + 1
                )
            
            # Check for significant thresholds crossed
            threshold_events = []
            
            # Trust thresholds
            if old_state["trust"] < Config.INTIMATE_TRUST_THRESHOLD <= rel["trust"]:
                threshold_events.append("intimate_trust_achieved")
            elif old_state["trust"] >= Config.HOSTILE_TRUST_THRESHOLD > rel["trust"]:
                threshold_events.append("trust_broken")
            
            # Power thresholds  
            if old_state["power_dynamic"] < Config.DOMINANT_POWER_THRESHOLD <= rel["power_dynamic"]:
                threshold_events.append("dominance_established")
            elif old_state["power_dynamic"] >= Config.SUBMISSIVE_POWER_THRESHOLD > rel["power_dynamic"]:
                threshold_events.append("submission_deepened")
            
            # Bond thresholds
            if old_state["emotional_bond"] < Config.INTIMATE_BOND_THRESHOLD <= rel["emotional_bond"]:
                threshold_events.append("deep_bond_formed")
            
            # Log threshold crossings
            for event in threshold_events:
                await log_canonical_event(
                    canonical_ctx, conn,
                    f"Relationship milestone: {event.replace('_', ' ')} with {entity_id}",
                    tags=["relationship", "milestone", event, entity_id],
                    significance=9  # Milestones are highly significant
                )
            
        except Exception as e:
            logger.error(f"Error in canonical relationship update: {e}", exc_info=True)
            # Continue with function to avoid breaking the flow
    
    # Create relationship changes for response
    changes = RelationshipChanges(
        trust=trust_change,
        power=power_change,
        bond=bond_change
    )
    
    # Create relationship state output
    relationship_state_out = RelationshipStateOut(
        trust=rel["trust"],
        power_dynamic=rel["power_dynamic"],
        emotional_bond=rel["emotional_bond"],
        interaction_count=rel["interaction_count"],
        last_interaction=rel["last_interaction"],
        type=rel["type"]
    )
    
    # Build result
    result = RelationshipUpdateResult(
        entity_id=entity_id,
        relationship=relationship_state_out,
        changes=changes
    )
    
    # Store evolution pattern for learning
    pattern_data = {
        "entity_id": entity_id,
        "change_magnitude": total_change,
        "relationship_type": rel["type"],
        "direction": "positive" if total_change > 0 else "negative" if total_change < 0 else "neutral",
        "trust_direction": "up" if trust_change > 0 else "down" if trust_change < 0 else "stable",
        "power_direction": "up" if power_change > 0 else "down" if power_change < 0 else "stable",
        "bond_direction": "up" if bond_change > 0 else "down" if bond_change < 0 else "stable"
    }
    
    # Learn from this relationship interaction
    await ctx.context.learn_from_interaction(
        action=f"relationship_update_{rel['type']}",
        outcome=f"total_change_{total_change:.2f}",
        success=total_change > 0.1  # Consider it successful if meaningful change occurred
    )
    
    return result.model_dump_json()
