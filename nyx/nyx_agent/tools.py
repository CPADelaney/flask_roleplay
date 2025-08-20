# nyx/nyx_agent/tools.py
"""Function tools for Nyx Agent SDK - Complete"""

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
    """Narrate a power exchange moment with an NPC."""
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
    """Narrate a daily routine activity."""
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
    """Generate ambient narration for the current scene."""
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
    """Generate contextual dialogue for an NPC."""
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

# ===== Visual/Image Tools =====

@function_tool
async def generate_image_from_scene(
    ctx: RunContextWrapper,
    payload: GenerateImageFromSceneInput
) -> str:
    """Generate an image for the current scene."""
    from routes.ai_image_generator import generate_roleplay_image_from_gpt

    data = GenerateImageFromSceneInput.model_validate(payload or {})
    image_data = data.model_dump()
    
    result = await generate_roleplay_image_from_gpt(
        image_data,
        ctx.context.user_id,
        ctx.context.conversation_id
    )
    
    if result and "image_urls" in result and result["image_urls"]:
        return ImageGenerationResult(
            success=True,
            image_url=result["image_urls"][0],
            error=None
        ).model_dump_json()
    else:
        return ImageGenerationResult(
            success=False,
            image_url=None,
            error="Failed to generate image"
        ).model_dump_json()

@function_tool
async def decide_image_generation(ctx: RunContextWrapper, payload: DecideImageInput) -> str:
    """Decide whether to generate an image for the current scene."""
    try:
        data = DecideImageInput.model_validate(payload or {})
        scene_text = data.scene_text or ""
    except Exception:
        scene_text = (getattr(payload, "scene_text", None)
                      or (isinstance(payload, dict) and payload.get("scene_text"))
                      or "")

    app_ctx = _resolve_app_ctx(ctx)
    context_map, writable = _ensure_context_map(app_ctx)

    score = _score_scene_text(scene_text)
    recent_images = int(context_map.get("recent_image_count", 0)) if isinstance(context_map, dict) else 0
    threshold = 0.7 if recent_images > 3 else 0.6 if recent_images > 1 else 0.5

    should_generate = score > threshold
    image_prompt = _build_image_prompt(scene_text) if should_generate else None

    if should_generate and writable:
        context_map["recent_image_count"] = recent_images + 1

    return ImageGenerationDecision(
        should_generate=should_generate,
        score=score,
        image_prompt=image_prompt,
        reasoning=f"Scene has visual impact score of {score:.2f} (threshold: {threshold:.2f})",
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

async def _get_memory_emotional_impact(ctx: RunContextWrapper, context: Dict[str, Any]) -> Dict[str, float]:
    """Get emotional impact from related memories"""
    return {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}

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
        "emotional_bond": rel["emotional_bond"],
        "type": rel.get("type", "neutral")
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
    
    # Learn from this relationship interaction
    await ctx.context.learn_from_interaction(
        action=f"relationship_update_{rel['type']}",
        outcome=f"total_change_{total_change:.2f}",
        success=total_change > 0.1  # Consider it successful if meaningful change occurred
    )
    
    return result.model_dump_json()

# ===== Performance Tools =====

@function_tool
async def check_performance_metrics(ctx: RunContextWrapper, payload: EmptyInput) -> str:
    """Check current performance metrics and apply remediation if needed."""
    _ = payload  # unused
    metrics = ctx.context.performance_metrics

    # Refresh CPU & RAM values using centralized helper
    process = get_process_info()
    if process:
        memory_info = safe_process_metric(process, 'memory_info')
        metrics["memory_usage"] = bytes_to_mb(memory_info)
    else:
        metrics["memory_usage"] = 0

    metrics["cpu_usage"] = ctx.context.get_cpu_usage()

    suggestions, actions_taken = [], []

    # Health checks
    avg_rt = _calculate_avg_response_time(metrics["response_times"])
    if avg_rt > Config.HIGH_RESPONSE_TIME_THRESHOLD:
        suggestions.append("Response times are high – consider caching frequent queries")

    if metrics["memory_usage"] > Config.HIGH_MEMORY_THRESHOLD_MB:
        suggestions.append("High memory usage detected – triggering cleanup")
        memory_before = metrics["memory_usage"]
        await ctx.context.handle_high_memory_usage()
        # Re-check memory after cleanup
        if process:
            memory_after = bytes_to_mb(safe_process_metric(process, 'memory_info'))
            logger.info(f"Memory cleanup: {memory_before:.2f}MB -> {memory_after:.2f}MB")
        actions_taken.append("memory_cleanup")

    if metrics["total_actions"]:
        success_rate = metrics["successful_actions"] / metrics["total_actions"]
        if success_rate < Config.MIN_SUCCESS_RATE:
            suggestions.append("Success rate below 80% – review error patterns")

    if metrics["error_rates"]["total"] > Config.HIGH_ERROR_COUNT:
        suggestions.append("High error count – clearing old errors")
        ctx.context.error_log = ctx.context.error_log[-Config.MAX_ERROR_LOG_ENTRIES:]
        actions_taken.append("error_log_cleanup")

    return PerformanceMetricsResult(
        metrics=PerformanceNumbers(
            memory_mb=metrics["memory_usage"],
            cpu_percent=metrics["cpu_usage"],
            avg_response_time=(
                sum(metrics["response_times"]) / len(metrics["response_times"])
                if metrics["response_times"] else 0
            ),
            success_rate=(
                metrics["successful_actions"] / metrics["total_actions"]
                if metrics["total_actions"] else 1.0
            )
        ),
        suggestions=suggestions,
        actions_taken=actions_taken,
        health="good" if not suggestions else "needs_attention",
    ).model_dump_json()

# ===== Activity Tools =====

@function_tool
async def get_activity_recommendations(
    ctx: RunContextWrapper,
    payload: GetActivityRecommendationsInput
) -> str:
    """Get activity recommendations based on current context."""
    data = GetActivityRecommendationsInput.model_validate(payload or {})
    scenario_type = data.scenario_type
    npc_ids = data.npc_ids
    
    activities = []
    
    # Copy relationship states to avoid mutation during iteration
    relationship_states_copy = dict(ctx.context.relationship_states)
    
    # Training activities
    if "training" in scenario_type.lower() or any(rel.get("type") == "submissive" 
        for rel in relationship_states_copy.values()):
        activities.extend([
            ActivityRec(
                name="Obedience Training",
                description="Test and improve submission through structured exercises",
                requirements=["trust > 0.4", "submission tendency"],
                duration="15-30 minutes",
                intensity="medium"
            ),
            ActivityRec(
                name="Position Practice",
                description="Learn and perfect submissive positions",
                requirements=["trust > 0.5"],
                duration="10-20 minutes",
                intensity="low-medium"
            )
        ])
    
    # Social activities
    if npc_ids and len(npc_ids) > 0:
        activities.append(ActivityRec(
            name="Group Dynamics Exercise",
            description="Explore power dynamics with multiple participants",
            requirements=["multiple NPCs present"],
            duration="20-40 minutes",
            intensity="variable"
        ))
    
    # Intimate activities
    for entity_id, rel in relationship_states_copy.items():
        if rel.get("type") == "intimate" and rel.get("trust", 0) > 0.7:
            activities.append(ActivityRec(
                name="Intimate Scene",
                description=f"Deepen connection with trusted partner",
                requirements=["high trust", "intimate relationship"],
                duration="30-60 minutes",
                intensity="high",
                partner_id=entity_id
            ))
            break
    
    # Default activities
    activities.extend([
        ActivityRec(
            name="Exploration",
            description="Discover new areas or items",
            requirements=[],
            duration="10-30 minutes",
            intensity="low"
        ),
        ActivityRec(
            name="Conversation",
            description="Engage in meaningful dialogue",
            requirements=[],
            duration="5-15 minutes",
            intensity="low"
        )
    ])
    
    return ActivityRecommendationsResult(
        recommendations=activities[:5],  # Top 5 activities
        total_available=len(activities)
    ).model_dump_json()

# ===== Belief Tools =====

@function_tool
async def manage_beliefs(ctx: RunContextWrapper, payload: ManageBeliefsInput) -> str:
    """Manage belief system operations."""
    data = ManageBeliefsInput.model_validate(payload or {})
    action = data.action
    belief_data = data.belief_data
    
    if not ctx.context.belief_system:
        return BeliefManagementResult(
            result="",
            error="Belief system not available"
        ).model_dump_json()
    
    try:
        if action == "get":
            entity_id = belief_data.entity_id
            beliefs = await ctx.context.belief_system.get_beliefs(entity_id)
            return BeliefManagementResult(
                result=dict_to_kvlist(beliefs),
                error=None
            ).model_dump_json()
        
        elif action == "update":
            entity_id = belief_data.entity_id
            belief_type = belief_data.type
            content = kvlist_to_dict(belief_data.content)
            await ctx.context.belief_system.update_belief(entity_id, belief_type, content)
            return BeliefManagementResult(
                result="Belief updated successfully",
                error=None
            ).model_dump_json()
        
        elif action == "query":
            query = belief_data.query or ""
            results = await ctx.context.belief_system.query_beliefs(query)
            return BeliefManagementResult(
                result=dict_to_kvlist(results),
                error=None
            ).model_dump_json()
        
        else:
            return BeliefManagementResult(
                result="",
                error=f"Unknown action: {action}"
            ).model_dump_json()
            
    except Exception as e:
        logger.error(f"Error managing beliefs: {e}", exc_info=True)
        return BeliefManagementResult(
            result="",
            error=str(e)
        ).model_dump_json()

# ===== Decision Tools =====

@function_tool
async def score_decision_options(
    ctx: RunContextWrapper,
    payload: ScoreDecisionOptionsInput
) -> str:
    """Score decision options using advanced decision engine logic."""
    data = ScoreDecisionOptionsInput.model_validate(payload or {})
    options = data.options
    decision_context = kvlist_to_dict(data.decision_context)
    
    scored_options = []
    
    for option in options:
        # Base score from context relevance
        context_score = _calculate_context_relevance(option.model_dump(), decision_context)
        
        # Emotional alignment score
        emotional_score = _calculate_emotional_alignment(option.model_dump(), ctx.context.emotional_state)
        
        # Pattern-based score
        pattern_score = _calculate_pattern_score(option.model_dump(), ctx.context.learned_patterns)
        
        # Relationship impact score
        relationship_score = _calculate_relationship_impact(option.model_dump(), ctx.context.relationship_states)
        
        # Calculate weighted final score
        weights = {
            "context": 0.3,
            "emotional": 0.25,
            "pattern": 0.25,
            "relationship": 0.2
        }
        
        final_score = (
            context_score * weights["context"] +
            emotional_score * weights["emotional"] +
            pattern_score * weights["pattern"] +
            relationship_score * weights["relationship"]
        )
        
        scored_options.append(ScoredOption(
            option=option,
            score=final_score,
            components=ScoreComponents(
                context=context_score,
                emotional=emotional_score,
                pattern=pattern_score,
                relationship=relationship_score
            )
        ))
    
    # Sort by score
    scored_options.sort(key=lambda x: x.score, reverse=True)
    
    # If all scores are too low, include a fallback
    if all(opt.score < Config.MIN_DECISION_SCORE for opt in scored_options):
        fallback = _get_fallback_decision(options)
        fallback_scored = ScoredOption(
            option=fallback,
            score=Config.FALLBACK_DECISION_SCORE,
            components=ScoreComponents(
                context=Config.FALLBACK_DECISION_SCORE,
                emotional=Config.FALLBACK_DECISION_SCORE,
                pattern=Config.FALLBACK_DECISION_SCORE,
                relationship=Config.FALLBACK_DECISION_SCORE
            ),
            is_fallback=True
        )
        scored_options.insert(0, fallback_scored)
    
    return DecisionScoringResult(
        scored_options=scored_options,
        best_option=scored_options[0].option,
        confidence=scored_options[0].score
    ).model_dump_json()

# ===== Conflict Detection Tools =====

@function_tool
async def detect_conflicts_and_instability(
    ctx: RunContextWrapper,
    payload: DetectConflictsAndInstabilityInput
) -> str:
    """Detect conflicts and emotional instability in current scenario."""
    data = DetectConflictsAndInstabilityInput.model_validate(payload or {})
    scenario_state = kvlist_to_dict(data.scenario_state)
    
    conflicts = []
    instabilities = []
    
    # Check for relationship conflicts
    relationship_items = list(ctx.context.relationship_states.items())
    for i, (entity1_id, rel1) in enumerate(relationship_items):
        for entity2_id, rel2 in relationship_items[i+1:]:
            # Conflicting power dynamics
            if abs(rel1.get("power_dynamic", 0.5) - rel2.get("power_dynamic", 0.5)) > Config.POWER_CONFLICT_THRESHOLD:
                conflicts.append(ConflictItem(
                    type="power_conflict",
                    entities=[entity1_id, entity2_id],
                    severity=abs(rel1["power_dynamic"] - rel2["power_dynamic"]),
                    description="Conflicting power dynamics between entities"
                ))
            
            # Low mutual trust
            if rel1.get("trust", 0.5) < Config.HOSTILE_TRUST_THRESHOLD and rel2.get("trust", 0.5) < Config.HOSTILE_TRUST_THRESHOLD:
                conflicts.append(ConflictItem(
                    type="trust_conflict",
                    entities=[entity1_id, entity2_id],
                    severity=0.7,
                    description="Mutual distrust between entities"
                ))
    
    # Check for emotional instability
    emotional_state = ctx.context.emotional_state
    
    # High arousal with negative valence
    if emotional_state["arousal"] > Config.HIGH_AROUSAL_THRESHOLD and emotional_state["valence"] < Config.NEGATIVE_VALENCE_THRESHOLD:
        instabilities.append(InstabilityItem(
            type="emotional_volatility",
            severity=emotional_state["arousal"],
            description="High arousal with negative emotions",
            recommendation="De-escalation needed"
        ))
    
    # Rapid emotional changes
    if ctx.context.adaptation_history:
        recent_emotions = [h.get("emotional_state", {}) for h in ctx.context.adaptation_history[-5:]]
        if recent_emotions and any(recent_emotions):
            valence_values = [e.get("valence", 0) for e in recent_emotions if e]
            if valence_values:
                valence_variance = _calculate_variance(valence_values)
                if valence_variance > Config.EMOTIONAL_VARIANCE_THRESHOLD:
                    instabilities.append(InstabilityItem(
                        type="emotional_instability",
                        severity=min(1.0, valence_variance),
                        description="Rapid emotional swings detected",
                        recommendation="Stabilization recommended"
                    ))
    
    # Scenario-specific conflicts
    if scenario_state.get("objectives"):
        blocked_objectives = [obj for obj in scenario_state["objectives"] 
                             if obj.get("status") == "blocked"]
        if blocked_objectives:
            conflicts.append(ConflictItem(
                type="objective_conflict",
                severity=len(blocked_objectives) / len(scenario_state["objectives"]),
                description=f"{len(blocked_objectives)} objectives are blocked",
                blocked_objectives=[str(obj) for obj in blocked_objectives]
            ))
    
    # Calculate overall stability
    total_issues = len(conflicts) + len(instabilities)
    overall_stability = max(0.0, 1.0 - (total_issues / Config.MAX_STABILITY_ISSUES))
    
    # Only save scenario state if it's a significant change
    if ctx.context.scenario_state and ctx.context.scenario_state.get("active"):
        should_save = False
        
        # Check if this is a significant change
        if conflicts and any(c.severity > 0.7 for c in conflicts):
            should_save = True
        if instabilities and any(i.severity > 0.7 for i in instabilities):
            should_save = True
        if overall_stability < 0.3:
            should_save = True
            
        if should_save:
            async with get_db_connection_context() as conn:
                # Use UPSERT pattern to maintain one current state
                await conn.execute("""
                    INSERT INTO scenario_states (user_id, conversation_id, state_data, created_at)
                    VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
                    ON CONFLICT (user_id, conversation_id) 
                    DO UPDATE SET state_data = $3, created_at = CURRENT_TIMESTAMP
                """, ctx.context.user_id, ctx.context.conversation_id, 
                json.dumps(ctx.context.scenario_state, ensure_ascii=False))
    
    return ConflictDetectionResult(
        conflicts=conflicts,
        instabilities=instabilities,
        overall_stability=overall_stability,
        stability_note=f"{total_issues} issues detected (0 issues = 1.0 stability, 10+ issues = 0.0 stability)",
        requires_intervention=any(c.severity > 0.8 for c in conflicts + instabilities)
    ).model_dump_json()

# ===== Universal Updates Tool =====

async def generate_universal_updates_impl(
    ctx: Union[RunContextWrapper, Any],
    narrative: str
) -> UniversalUpdateResult:
    """Implementation of generate universal updates from the narrative using the Universal Updater."""
    from logic.universal_updater_agent import process_universal_update

    app_ctx = _unwrap_tool_ctx(ctx)

    user_id = getattr(app_ctx, "user_id", None)
    convo_id = getattr(app_ctx, "conversation_id", None)
    if user_id is None or convo_id is None:
        return UniversalUpdateResult(
            success=False,
            updates_generated=False,
            error="Invalid context: missing user_id/conversation_id"
        )

    try:
        update_result = await process_universal_update(
            user_id=user_id,
            conversation_id=convo_id,
            narrative=narrative,
            context={"source": "nyx_agent"}
        )

        # Normalize updater result to a dict
        if hasattr(update_result, "model_dump"):
            update_result = update_result.model_dump()
        elif not isinstance(update_result, dict):
            update_result = {"success": bool(update_result), "details": None}

        ctx_map, writable = _ensure_context_map(app_ctx)
        if writable and "universal_updates" not in ctx_map:
            ctx_map["universal_updates"] = {}

        # Merge details if present (handle list/dict/pydantic)
        details = update_result.get("details")
        if details:
            if hasattr(details, "model_dump"):
                details = details.model_dump()

            if isinstance(details, list):
                # try helper, otherwise best-effort kv conversion
                try:
                    from logic.universal_updater_agent import array_to_dict
                    details_dict = array_to_dict(details)
                except Exception:
                    details_dict = {}
                    for d in details:
                        if isinstance(d, dict) and "key" in d:
                            details_dict[d["key"]] = d.get("value")
            elif isinstance(details, dict):
                details_dict = details
            else:
                details_dict = {}

            if writable and isinstance(ctx_map.get("universal_updates"), dict):
                for k, v in details_dict.items():
                    ctx_map["universal_updates"][k] = v

        return UniversalUpdateResult(
            success=bool(update_result.get("success", False)),
            updates_generated=bool(update_result.get("details")),
            error=None
        )
    except Exception as e:
        logger.error(f"Error generating universal updates: {e}")
        return UniversalUpdateResult(
            success=False,
            updates_generated=False,
            error=str(e)
        )

@function_tool
async def generate_universal_updates(
    ctx: RunContextWrapper,
    payload: GenerateUniversalUpdatesInput
) -> str:
    """Extract universal state updates from narrative text."""
    # Defensive payload handling
    try:
        data = GenerateUniversalUpdatesInput.model_validate(payload or {})
        narrative = data.narrative
    except Exception:
        narrative = getattr(payload, "narrative", None) or (payload.get("narrative") if isinstance(payload, dict) else "")

    result = await generate_universal_updates_impl(ctx, narrative or "")
    return result.model_dump_json()

# ===== Open World / Slice-of-life Tools =====

@function_tool
async def orchestrate_slice_scene(
    ctx: RunContextWrapper,
    scene_type: str = "routine",
) -> dict:
    """Prepare narrator payload for a slice-of-life scene. No nested tool calls."""
    app_ctx = _unwrap_tool_ctx(ctx)

    # Ensure narrator exists (we won't call it here, but we may need IDs/context)
    if not getattr(app_ctx, "slice_of_life_narrator", None):
        from story_agent.slice_of_life_narrator import SliceOfLifeNarrator
        app_ctx.slice_of_life_narrator = SliceOfLifeNarrator(app_ctx.user_id, app_ctx.conversation_id)
        await app_ctx.slice_of_life_narrator.initialize()

    world_state_obj = await _ensure_world_state(app_ctx)
    world_state_payload = _json_safe(world_state_obj)

    # Participants -> ints
    participants: list[int] = []
    if world_state_obj and getattr(world_state_obj, "active_npcs", None):
        for npc in (world_state_obj.active_npcs or [])[:3]:
            if isinstance(npc, dict) and "npc_id" in npc:
                participants.append(int(npc["npc_id"]))
            elif isinstance(npc, int):
                participants.append(npc)
            elif hasattr(npc, "npc_id"):
                with suppress(Exception):
                    participants.append(int(getattr(npc, "npc_id")))

    activity_value_map = {
        "routine": "routine", "social": "social", "work": "work",
        "intimate": "intimate", "leisure": "leisure", "special": "special",
        "errands": "routine", "chores": "routine",
    }
    event_type_value = activity_value_map.get(scene_type, "routine")

    # Try typed, fall back to dict
    try:
        from story_agent.slice_of_life_narrator import SliceOfLifeEvent as TypedEvent
        from story_agent.world_director_agent import ActivityType as TypedActivity
        event_type = getattr(TypedActivity, event_type_value.upper(), TypedActivity.ROUTINE)
        scene_obj = TypedEvent(
            event_type=event_type,
            title=f"{scene_type} scene",
            description=f"A {scene_type} moment in daily life",
            location="current_location",
            participants=participants,
        )
        scene_payload = _json_safe(scene_obj)
    except Exception:
        scene_payload = {
            "event_type": event_type_value,
            "title": f"{scene_type} scene",
            "description": f"A {scene_type} moment in daily life",
            "location": "current_location",
            "participants": participants,
        }

    narrator_input_payload = {
        "scene_type": scene_type,
        "scene": scene_payload,
        "world_state": world_state_payload,
        "player_action": None,
    }

    # Return a *request* for the host to fulfill
    return {
        "narrator_request": {
            "tool": "tool_narrate_slice_of_life_scene",
            "payload": _json_safe(narrator_input_payload),
        }
    }

@function_tool
async def check_world_state(
    ctx: RunContextWrapper,
    payload: EmptyInput
) -> str:
    """Return a compact, JSON-safe snapshot of the current world state."""
    app_ctx = _get_app_ctx(ctx)
    ws = await _ensure_world_state_from_ctx(app_ctx)

    if ws is None:
        return json.dumps({"error": "world_state_unavailable"}, ensure_ascii=False)

    def _jsafe(x):
        if x is None:
            return None
        if hasattr(x, "model_dump"):
            return x.model_dump()
        if isinstance(x, list):
            return [_jsafe(i) for i in x]
        if isinstance(x, dict):
            return {k: _jsafe(v) for k, v in x.items()}
        return getattr(x, "value", x)

    tod = getattr(getattr(ws, "current_time", None), "time_of_day", None)
    out = {
        "time_of_day": getattr(tod, "value", tod),
        "world_mood": getattr(getattr(ws, "world_mood", None), "value", None),
        "active_npcs": [
            (npc.get("npc_name") or npc.get("name") or npc.get("title"))
            if isinstance(npc, dict) else str(npc)
            for npc in (getattr(ws, "active_npcs", []) or [])
        ],
        "ongoing_events": _jsafe(getattr(ws, "ongoing_events", [])),
        "tensions": _jsafe(getattr(ws, "tension_factors", {})),
        "player_state": {
            "vitals": _jsafe(getattr(ws, "player_vitals", None)),
            "addictions": _jsafe(getattr(ws, "addiction_status", {})),
            "stats": _jsafe(getattr(ws, "hidden_stats", {})),
        },
    }
    return json.dumps(out, ensure_ascii=False)
  
@function_tool
async def generate_emergent_event(
    ctx: RunContextWrapper,
    payload: EmergentEventInput
) -> str:
    """Generate an emergent slice-of-life event."""
    app_ctx = _get_app_ctx(ctx)
    wd = getattr(app_ctx, "world_director", None)
    if not wd:
        return json.dumps({"error": "world_director not available"}, ensure_ascii=False)

    try:
        event = await wd.generate_next_moment()
    except Exception as e:
        return json.dumps({"error": f"director_failed: {e}"}, ensure_ascii=False)

    def _jsafe(x):
        if x is None:
            return None
        if hasattr(x, "model_dump"):
            return x.model_dump()
        if isinstance(x, list):
            return [_jsafe(i) for i in x]
        if isinstance(x, dict):
            return {k: _jsafe(v) for k, v in x.items()}
        return getattr(x, "value", x)

    safe_event = _jsafe(event)
    # Quick human summary
    title = None; etype = None; participants = []; location = None; timestamp = None
    if isinstance(safe_event, dict):
        title = safe_event.get("title") or safe_event.get("moment", {}).get("title")
        etype = safe_event.get("type") or safe_event.get("moment", {}).get("type")
        location = safe_event.get("location") or safe_event.get("moment", {}).get("location")
        timestamp = safe_event.get("time") or safe_event.get("moment", {}).get("time")
        raw_parts = (safe_event.get("participants")
                     or safe_event.get("moment", {}).get("participants") or [])
        if isinstance(raw_parts, list):
            for p in raw_parts:
                if isinstance(p, dict):
                    participants.append(p.get("npc_name") or p.get("name") or p.get("title") or str(p))
                else:
                    participants.append(str(p))

    out = {
        "event": safe_event,
        "event_summary": {
            "title": title,
            "type": etype,
            "location": location,
            "time": timestamp,
            "participants": participants,
        },
        "nyx_commentary": "*smirks* Let's see what that ripple does to your day…",
    }
    return json.dumps(out, ensure_ascii=False)
    
@function_tool
async def simulate_npc_autonomy(
    ctx: RunContextWrapper,
    payload: SimulateAutonomyInput
) -> str:
    """Simulate autonomous NPC actions with context-aware activity processing."""
    app_ctx = _get_app_ctx(ctx)
    wd = getattr(app_ctx, "world_director", None)
    if not wd:
        return json.dumps({"error": "world_director not available"}, ensure_ascii=False)
    
    from logic.stats_logic import process_world_activity
    
    try:
        result = await wd.advance_time(payload.hours)
    except Exception as e:
        return json.dumps({"error": f"advance_time_failed: {e}"}, ensure_ascii=False)
    
    def _jsafe(x):
        if x is None:
            return None
        if hasattr(x, "model_dump"):
            return x.model_dump()
        if isinstance(x, list):
            return [_jsafe(i) for i in x]
        if isinstance(x, dict):
            return {k: _jsafe(v) for k, v in x.items()}
        return getattr(x, "value", x)
    
    safe_result = _jsafe(result)
    
    # Build world context from available data
    world_context = {}
    
    # Extract world state if available
    if hasattr(app_ctx, 'current_world_state') and app_ctx.current_world_state:
        ws = app_ctx.current_world_state
        try:
            # Get world mood
            if hasattr(ws, 'world_mood'):
                mood = ws.world_mood
                world_context['world_mood'] = mood.value if hasattr(mood, 'value') else str(mood)
            
            # Get time of day
            if hasattr(ws, 'current_time'):
                time_data = ws.current_time
                if hasattr(time_data, 'time_of_day'):
                    world_context['time_of_day'] = str(time_data.time_of_day)
            
            # Get stats
            if hasattr(ws, 'hidden_stats'):
                world_context['hidden_stats'] = ws.hidden_stats
            if hasattr(ws, 'visible_stats'):
                world_context['visible_stats'] = ws.visible_stats
            
            # Get NPCs
            if hasattr(ws, 'active_npcs'):
                world_context['active_npcs'] = ws.active_npcs[:3]  # Limit to 3 for context
            
            # Get addictions
            if hasattr(ws, 'addiction_status'):
                world_context['addiction_status'] = ws.addiction_status
            
            # Get relationships
            if hasattr(ws, 'relationship_dynamics'):
                world_context['relationship_dynamics'] = ws.relationship_dynamics
            
            # Get location
            if hasattr(ws, 'location_data'):
                world_context['location'] = ws.location_data
                
        except Exception as e:
            logger.debug(f"Could not extract full world context: {e}")
    
    # Check if vitals update failed
    if isinstance(safe_result, dict):
        vitals_result = safe_result.get('vitals_updated', {})
        
        if not vitals_result.get('success') and 'Unknown activity' in str(vitals_result.get('error', '')):
            # Extract the activity
            activity_name = 'unknown'
            
            # Find activity in results
            if 'time' in safe_result:
                time_data = safe_result['time']
                if isinstance(time_data, dict):
                    activity_name = time_data.get('activity_mood') or time_data.get('activity') or 'unknown'
            
            # Process with full world context
            try:
                activity_result = await process_world_activity(
                    user_id=app_ctx.user_id,
                    conversation_id=app_ctx.conversation_id,
                    activity_name=activity_name,
                    player_name="Chase",
                    world_context=world_context,  # Pass the context!
                    hours=payload.hours
                )
                
                # Replace failed vitals_updated
                safe_result['vitals_updated'] = {
                    'success': True,
                    'activity': activity_name,
                    'effects': activity_result.get('effects', {}),
                    'method': 'contextual_generation',
                    'context_used': bool(world_context)
                }
                
            except Exception as e:
                logger.warning(f"Contextual activity processing failed: {e}")
    
    # Build action log and process NPC actions with context
    action_log = []
    candidate = []
    if isinstance(safe_result, list):
        candidate = safe_result
    elif isinstance(safe_result, dict):
        for key in ("actions", "npc_actions", "events", "log"):
            if isinstance(safe_result.get(key), list):
                candidate = safe_result[key]
                break
    
    for entry in candidate or []:
        if isinstance(entry, dict):
            action_entry = {
                "npc": entry.get("npc") or entry.get("npc_name") or entry.get("name"),
                "action": entry.get("action") or entry.get("current_activity") or entry.get("activity"),
                "time": entry.get("time") or entry.get("timestamp"),
            }
            action_log.append(action_entry)
        else:
            action_log.append({"entry": str(entry)})
    
    out = {
        "advanced_time_hours": payload.hours,
        "npc_actions": safe_result,
        "npc_action_log": action_log,
        "nyx_observation": "While you were away, the others kept moving… and watching.",
        "context_aware": bool(world_context)  # Flag showing we used context
    }
    
    return json.dumps(out, ensure_ascii=False)
