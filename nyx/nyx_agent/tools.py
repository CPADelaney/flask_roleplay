# nyx/nyx_agent/tools.py
"""Refactored tools module for Nyx Agent SDK with ContextBundle architecture"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Dict, List, Any, Optional, Union

from agents import function_tool, RunContextWrapper

from .context import NyxContext, ContextBroker, SceneScope, ContextBundle, PackedContext
from .models import (
    # Input models
    EmptyInput,
    RetrieveMemoriesInput,
    AddMemoryInput,
    DecideImageGenerationInput,
    UpdateEmotionalStateInput,
    UpdateRelationshipStateInput,
    ScoreDecisionOptionsInput,
    
    # Output models  
    MemoryItem,
    MemorySearchResult,
    MemoryStorageResult,
    UserGuidanceResult,
    ImageGenerationDecision,
    EmotionalState,
    RelationshipUpdate,
    PerformanceMetrics,
    ActivityRecommendationsResult,
    DecisionScores,
    ConflictDetection,
)

logger = logging.getLogger(__name__)

# ===== Bundle Expansion Utilities =====

def _log_task_exc(task: asyncio.Task) -> None:
    """Log exceptions from background tasks"""
    try:
        task.result()
    except Exception:
        logger.exception("Background task failed")

async def _get_context_broker(ctx: RunContextWrapper) -> ContextBroker:
    """Get or create the ContextBroker from wrapped context"""
    nyx_ctx = ctx.context if isinstance(ctx.context, NyxContext) else ctx
    if not hasattr(nyx_ctx, 'broker'):
        nyx_ctx.broker = ContextBroker(nyx_ctx)
    return nyx_ctx.broker

async def _get_bundle(ctx: RunContextWrapper, expand_sections: Optional[List[str]] = None) -> ContextBundle:
    """Get the current context bundle, optionally expanding specific sections"""
    broker = await _get_context_broker(ctx)
    nyx_ctx = ctx.context if isinstance(ctx.context, NyxContext) else ctx
    
    # Cache on the actual NyxContext, not the wrapper
    if not hasattr(nyx_ctx, '_current_bundle'):
        scene_scope = broker.compute_scene_scope(
            getattr(nyx_ctx, 'last_user_input', ''),
            getattr(nyx_ctx, 'current_state', {})
        )
        nyx_ctx._current_bundle = await broker.load_or_fetch_bundle(scene_scope)
    
    bundle = nyx_ctx._current_bundle
    
    # Expand missing sections in parallel
    if expand_sections:
        missing = [s for s in expand_sections if s not in bundle.expanded_sections]
        if missing:
            await asyncio.gather(
                *(broker.expand_bundle_section(bundle, s) for s in missing)
            )
    
    # Mirror onto wrapper for compatibility
    setattr(ctx, '_current_bundle', bundle)
    return bundle

async def _get_packed_context(
    ctx: RunContextWrapper, 
    token_budget: int = 8000,
    must_include: Optional[List[str]] = None
) -> PackedContext:
    """Get packed context optimized for token budget"""
    bundle = await _get_bundle(ctx)
    return bundle.pack(token_budget, must_include or ['canon'])

# ===== Memory Tools (Refactored) =====

@function_tool
async def retrieve_memories(
    ctx: RunContextWrapper,
    payload: RetrieveMemoriesInput
) -> Dict[str, Any]:
    """Enhanced memory retrieval using ContextBundle's memory section"""
    bundle = await _get_bundle(ctx, expand_sections=['memory'])
    
    # Extract memory data from bundle (avoid mutation)
    memory_data = bundle.sections.get('memory', {})
    memories = list(memory_data.get('memories', []))  # Copy to avoid mutation
    graph_links = memory_data.get('graph_links', {})
    
    # Define graph_keys and scene entities BEFORE the loop, normalize to strings
    graph_keys = {str(k) for k in graph_links.keys()}
    scene_entity_ids = {str(e) for e in getattr(bundle.scene_scope, 'entity_ids', [])}
    
    # If we need more memories, fetch via broker
    if payload.limit > len(memories):
        broker = await _get_context_broker(ctx)
        additional = await broker.fetch_memories(
            query=payload.query,
            limit=payload.limit - len(memories),
            exclude_ids=[str(m.get('id')) for m in memories]
        )
        memories.extend(additional)
    
    # Apply relevance filtering with graph boost
    scored_memories = []
    for mem in memories:
        base_score = mem.get('relevance', 0.0)
        link_boost = 0.0
        mem_id = str(mem.get('id', ''))
        
        # Boost if memory is graph-linked to current scene entities
        if mem_id in graph_keys:
            linked_entities = {str(e) for e in graph_links[mem_id]}
            if linked_entities & scene_entity_ids:  # Set intersection
                link_boost = 0.2  # Emergent connection bonus
        
        scored_memories.append({
            **mem,
            'relevance': min(1.0, base_score + link_boost)
        })
    
    # Sort by relevance and limit
    scored_memories.sort(key=lambda m: m['relevance'], reverse=True)
    final_memories = scored_memories[:payload.limit]
    
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
    """Store memory and update context bundle"""
    broker = await _get_context_broker(ctx)
    
    # Store via broker (handles canon logging)
    result = await broker.store_memory(
        memory_text=payload.memory_text,
        memory_type=payload.memory_type,
        significance=payload.significance,
        entities=payload.entities  # Link to scene entities
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

# ===== Narrative Tools (Refactored) =====

@function_tool
async def tool_narrate_slice_of_life_scene(
    ctx: RunContextWrapper,
    scene_type: str = "routine",
    player_action: Optional[str] = None
) -> Dict[str, Any]:
    """Generate narration using packed context optimized for scene"""
    # Get scene-optimized bundle
    bundle = await _get_bundle(ctx, expand_sections=['npcs', 'location', 'conflict'])
    packed = await _get_packed_context(ctx, token_budget=6000, must_include=['npcs', 'canon'])
    
    # Extract relevant data from bundle
    npcs = bundle.sections.get('npcs', {}).get('active', [])
    location = bundle.sections.get('location', {})
    conflicts = bundle.sections.get('conflict', {}).get('active', [])
    
    # Build scene context from bundle (safer NPC ID handling)
    participants = []
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
    """Orchestrate complete scene with all subsystems via ContextBroker"""
    broker = await _get_context_broker(ctx)
    
    # Read from underlying NyxContext, not wrapper
    nyx_ctx = ctx.context if hasattr(ctx, 'context') else ctx
    
    # Compute scene scope from current state
    scene_scope = broker.compute_scene_scope(
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
    """Generate NPC dialogue with relationship-aware context"""
    # Expand NPC section for specific character
    bundle = await _get_bundle(ctx, expand_sections=['npcs', 'memory'])
    broker = await _get_context_broker(ctx)
    
    # Get NPC data from bundle
    npc_data = await broker.get_npc_details(bundle, npc_id)
    
    # Find memories involving this NPC (normalize IDs to strings)
    memories = bundle.sections.get('memory', {}).get('memories', [])
    npc_id_str = str(npc_id)
    npc_memories = [
        m for m in memories 
        if npc_id_str in [str(e) for e in m.get('entities', [])]
    ]
    
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

# ===== World State Tools (Refactored) =====

@function_tool
async def check_world_state(
    ctx: RunContextWrapper,
    payload: EmptyInput = None
) -> Dict[str, Any]:
    """Get world state from context bundle"""
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
    """Generate emergent events based on bundle's graph connections"""
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
    """Simulate NPC actions using bundle's cached NPC state"""
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

# ===== Emotional & Relationship Tools (Refactored) =====

@function_tool
async def calculate_and_update_emotional_state(
    ctx: RunContextWrapper,
    payload: UpdateEmotionalStateInput
) -> Dict[str, Any]:
    """Update emotional state with bundle-aware context"""
    bundle = await _get_bundle(ctx, expand_sections=['emotional'])
    broker = await _get_context_broker(ctx)
    
    # Get current emotional state from bundle
    current_state = bundle.sections.get('emotional', {})
    
    # Calculate new state with graph-aware influences
    patterns = await broker.detect_emotional_patterns(bundle)
    
    new_state = await broker.update_emotional_state(
        current=current_state,
        event=payload.triggering_event,
        valence_delta=payload.valence_change,
        arousal_delta=payload.arousal_change, 
        dominance_delta=payload.dominance_change,
        patterns=patterns
    )
    
    # Update bundle
    bundle.sections['emotional'] = new_state
    
    return EmotionalStateResult(
        valence=new_state['valence'],
        arousal=new_state['arousal'],
        dominance=new_state['dominance'],
        emotional_label=new_state.get('label', 'neutral'),
        manifestation=new_state.get('manifestation', '')
    ).model_dump()

@function_tool
async def update_relationship_state(
    ctx: RunContextWrapper,
    payload: UpdateRelationshipStateInput  # Changed
) -> Dict[str, Any]:
    """Update relationship using bundle's relationship graph"""
    bundle = await _get_bundle(ctx, expand_sections=['relationships'])
    broker = await _get_context_broker(ctx)
    
    # Normalize entity ID to string
    entity_id_str = str(payload.entity_id)
    
    # Get relationship from bundle
    relationships = bundle.sections.get('relationships', {})
    current = relationships.get(entity_id_str, {})
    
    # Update with graph-aware context
    updated = await broker.update_relationship(
        entity_id=payload.entity_id,  # Keep original type for broker
        current_state=current,
        trust_delta=payload.trust_change,
        attraction_delta=payload.attraction_change,
        respect_delta=payload.respect_change,
        event=payload.triggering_event,
        graph_context=bundle.get_graph_context(payload.entity_id)
    )
    
    # Store in bundle with normalized ID
    relationships[entity_id_str] = updated
    
    return RelationshipUpdate(
        entity_id=payload.entity_id,
        new_trust=updated['trust'],
        new_attraction=updated['attraction'],
        new_respect=updated['respect'],
        relationship_level=updated.get('level', 'neutral'),
        change_description=updated.get('change_description', '')
    ).model_dump()

# ===== Decision & Conflict Tools (Refactored) =====

@function_tool
async def detect_conflicts_and_instability(
    ctx: RunContextWrapper,
    payload: EmptyInput = None
) -> Dict[str, Any]:
    """Detect conflicts using bundle's graph analysis"""
    bundle = await _get_bundle(ctx, expand_sections=['conflict', 'npcs'])
    broker = await _get_context_broker(ctx)
    
    # Analyze graph for tension points
    tensions = await broker.analyze_tensions(bundle)
    
    # Detect active and potential conflicts
    conflicts = bundle.sections.get('conflict', {}).get('active', [])
    potential = await broker.detect_potential_conflicts(bundle, tensions)
    
    return ConflictDetection(
        active_conflicts=conflicts,
        potential_conflicts=potential,
        tension_level=tensions.get('overall', 0.5),
        hot_spots=tensions.get('hot_spots', []),
        recommendations=tensions.get('recommendations', [])
    ).model_dump()

@function_tool
async def score_decision_options(
    ctx: RunContextWrapper,
    payload: ScoreDecisionOptionsInput  # Changed
) -> Dict[str, Any]:
    """Score decisions with full bundle context"""
    bundle = await _get_bundle(ctx)
    broker = await _get_context_broker(ctx)
    
    # Score each option with multi-factor analysis
    scores = {}
    for option in payload.options:
        score = await broker.score_decision(
            option=option,
            bundle=bundle,
            criteria={
                'canon_consistency': 1.0,  # Top priority
                'character_alignment': 0.8,
                'narrative_impact': 0.7,
                'emergent_potential': 0.6,
                'player_satisfaction': 0.5
            }
        )
        scores[option] = score
    
    # Sort by score
    ranked = sorted(scores.items(), key=lambda x: x[1]['total'], reverse=True)
    
    return DecisionScores(
        options={k: v['total'] for k, v in scores.items()},
        recommended=ranked[0][0] if ranked else None,
        reasoning={k: v['reasoning'] for k, v in scores.items()}
    ).model_dump()

# ===== Image Generation Tools (Refactored) =====

@function_tool
async def decide_image_generation(
    ctx: RunContextWrapper,
    payload: DecideImageGenerationInput
) -> Dict[str, Any]:
    """Decide on image generation using bundle's scene analysis"""
    bundle = await _get_bundle(ctx, expand_sections=['visual'])
    broker = await _get_context_broker(ctx)
    
    # Analyze scene for visual potential
    visual_analysis = await broker.analyze_visual_potential(
        scene_text=payload.scene_text,
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

# ===== System Tools (Refactored) =====

@function_tool
async def generate_universal_updates(
    ctx: RunContextWrapper,
    payload: EmptyInput = None
) -> Dict[str, Any]:
    """Generate updates across all systems using bundle"""
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
    payload: EmptyInput = None
) -> Dict[str, Any]:
    """Get performance metrics from bundle's telemetry"""
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

# ===== Advanced Narrative Tools =====

@function_tool
async def generate_ambient_narration(
    ctx: RunContextWrapper,
    focus: str = "atmosphere",
    intensity: float = 0.5
) -> Dict[str, Any]:
    """Generate ambient narration from bundle's atmosphere data"""
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
) -> str:
    """Get activity recommendations using bundle context"""
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
    
    return ActivityRecommendations(
        recommendations=recommendations,
        total_available=len(activity_analysis.get('available', [])),
        scenario_context=activity_analysis.get('context', {})
    ).model_dump_json()

@function_tool
async def detect_narrative_patterns(
    ctx: RunContextWrapper,
    lookback_turns: int = 10
) -> Dict[str, Any]:
    """Detect narrative patterns using bundle's graph analysis"""
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
    """Narrate power dynamics using bundle's relationship data"""
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
    """Narrate routine with emergent details from bundle"""
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

# ===== User Model Tools (Refactored) =====

@function_tool
async def get_user_model_guidance(
    ctx: RunContextWrapper,
    payload: EmptyInput = None
) -> str:
    """Get user model guidance from bundle's analysis"""
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
    ).model_dump_json()

@function_tool
async def detect_user_revelations(
    ctx: RunContextWrapper,
    user_input: str
) -> str:
    """Detect revelations with bundle context"""
    bundle = await _get_bundle(ctx, expand_sections=['user_model', 'memory'])
    broker = await _get_context_broker(ctx)
    
    # Analyze input for revelations
    revelations = await broker.detect_revelations(
        input_text=user_input,
        bundle=bundle
    )
    
    # Store significant revelations
    for rev in revelations:
        if rev['significance'] > 0.7:
            await broker.store_revelation(rev, bundle)
    
    return json.dumps({
        'revelations': revelations,
        'updated_model': bundle.sections.get('user_model', {})
    })

# ===== Helper Tools =====

@function_tool
async def expand_context_section(
    ctx: RunContextWrapper,
    section: str,
    depth: str = "standard"
) -> Dict[str, Any]:
    """On-demand expansion of specific context sections"""
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
    """Prefetch context for predicted next action"""
    broker = await _get_context_broker(ctx)
    
    # Predict next scene scope
    predicted_scope = broker.predict_scene_scope(predicted_action)
    
    # Generate stable task ID
    task_id = str(uuid.uuid4())
    
    # Prefetch in background with error handling
    task = asyncio.create_task(
        broker.prefetch_bundle(predicted_scope)
    )
    task.add_done_callback(_log_task_exc)
    
    # Register task for later status queries
    if hasattr(broker, 'register_task'):
        broker.register_task(task_id, task)
    
    return {
        'prefetch_started': True,
        'predicted_scope': predicted_scope.to_dict(),
        'task_id': task_id
    }

# ===== Implementation Wrappers =====

async def generate_universal_updates_impl(
    ctx: Union[RunContextWrapper, Any],
    narrative: str
) -> Dict[str, Any]:
    """Implementation wrapper for universal updates - handles both old and new context"""
    # Handle both RunContextWrapper and direct NyxContext
    if hasattr(ctx, 'context'):
        app_ctx = ctx.context
    else:
        app_ctx = ctx
    
    # Create wrapper if needed
    if not isinstance(ctx, RunContextWrapper):
        ctx = RunContextWrapper(context=app_ctx)
    
    return await generate_universal_updates(ctx, EmptyInput())

# ===== Optional Envelope Helpers =====

def ok_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap successful response in standard envelope"""
    return {"ok": True, "data": data}

def error_response(error: str) -> Dict[str, Any]:
    """Wrap error response in standard envelope"""
    return {"ok": False, "error": error}

# ===== Compatibility Exports =====

# Keep these for backward compatibility
generate_image_from_scene = decide_image_generation
calculate_emotional_impact = calculate_and_update_emotional_state
manage_beliefs = score_decision_options

# Additional compatibility mappings used by other modules
detect_user_revelations_impl = detect_user_revelations
get_user_model_guidance_impl = get_user_model_guidance
