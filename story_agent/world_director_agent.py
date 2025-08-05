# story_agent/world_director_agent_refactored.py
"""
Refactored World Dynamics Director with full system integration and LLM-driven generation.
All events are dynamically generated via OpenAI for maximum emergent gameplay.
"""

from __future__ import annotations

import asyncio
import logging
import json
import time
import random
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime, timezone, timedelta
from enum import Enum

from db.connection import get_db_connection_context
from agents import Agent, function_tool, Runner, trace, ModelSettings, RunContextWrapper

# ===============================================================================
# CORE SYSTEM INTEGRATIONS
# ===============================================================================

# OpenAI Integration for Dynamic Generation
from logic.chatgpt_integration import (
    OpenAIClientManager,
    get_chatgpt_response,
    generate_text_completion,
    get_text_embedding,
    generate_reflection,
    analyze_preferences
)

# Universal Updater for Narrative Processing
from logic.universal_updater_agent import (
    UniversalUpdaterAgent,
    process_universal_update
)

# Memory System Integration
from logic.memory_logic import (
    MemoryManager,
    EnhancedMemory,
    MemoryType,
    MemorySignificance,
    ProgressiveRevealManager,
    RevealType,
    RevealSeverity,
    NPCMask,
    generate_flashback,
    check_for_automated_reveals
)

# Stats System Integration
from logic.stats_logic import (
    get_player_visible_stats,
    get_player_hidden_stats,
    get_all_player_stats,
    apply_stat_changes,
    check_for_combination_triggers,
    apply_activity_effects,
    STAT_THRESHOLDS,
    STAT_COMBINATIONS,
    ACTIVITY_EFFECTS,
    detect_deception,
    calculate_social_insight
)

# Rule Enforcement Integration
from logic.rule_enforcement import (
    enforce_all_rules_on_player,
    evaluate_condition,
    parse_condition,
    apply_effect
)

# Inventory System Integration
from logic.inventory_system_sdk import (
    get_inventory,
    add_item,
    remove_item
)

# Time and Calendar Systems
from logic.time_cycle import (
    get_current_time_model,
    advance_time_with_events,
    get_current_vitals,
    VitalsData,
    CurrentTimeData,
    process_activity_vitals,
    ActivityManager
)

# Dynamic Relationships System
from logic.dynamic_relationships import (
    OptimizedRelationshipManager,
    drain_relationship_events_tool,
    RelationshipState
)

# NPC Systems
from logic.npc_narrative_progression import (
    get_npc_narrative_stage,
    check_for_npc_revelation,
    progress_npc_narrative_stage
)

# Currency System
from logic.currency_generator import CurrencyGenerator

# Event System
from logic.event_system import EventSystem

# Context systems
from context.context_service import get_context_service
from context.memory_manager import get_memory_manager

logger = logging.getLogger(__name__)

# ===============================================================================
# Enhanced World State Models with Full Integration
# ===============================================================================

class WorldMood(Enum):
    """Overall mood/atmosphere of the world"""
    RELAXED = "relaxed"
    TENSE = "tense"
    PLAYFUL = "playful"
    INTIMATE = "intimate"
    MYSTERIOUS = "mysterious"
    OPPRESSIVE = "oppressive"
    CHAOTIC = "chaotic"
    EXHAUSTED = "exhausted"
    DESPERATE = "desperate"
    CORRUPTED = "corrupted"  # Added for high corruption states

class EnhancedWorldState(BaseModel):
    """Enhanced world state with all system integrations"""
    # Time and vitals
    current_time: CurrentTimeData
    player_vitals: VitalsData
    
    # Stats tracking
    visible_stats: Dict[str, Any] = Field(default_factory=dict)
    hidden_stats: Dict[str, Any] = Field(default_factory=dict)
    active_stat_combinations: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Memory and context
    recent_memories: List[Dict[str, Any]] = Field(default_factory=list)
    active_flashbacks: List[Dict[str, Any]] = Field(default_factory=list)
    pending_reveals: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Rules and triggers
    active_rules: List[Dict[str, Any]] = Field(default_factory=list)
    triggered_effects: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Inventory
    player_inventory: List[Dict[str, Any]] = Field(default_factory=list)
    
    # NPCs and relationships
    active_npcs: List[Dict[str, Any]] = Field(default_factory=list)
    npc_masks: Dict[int, Dict[str, Any]] = Field(default_factory=dict)
    relationship_states: Dict[str, Any] = Field(default_factory=dict)
    
    # World mood and atmosphere
    world_mood: WorldMood
    tension_factors: Dict[str, float] = Field(default_factory=dict)
    
    # Currency
    player_money: int = 0
    currency_system: Dict[str, Any] = Field(default_factory=dict)
    
    # Active events
    ongoing_events: List[Dict[str, Any]] = Field(default_factory=list)
    
    model_config = ConfigDict(extra="forbid")

# ===============================================================================
# Enhanced World Director Context with All Systems
# ===============================================================================

@dataclass
class EnhancedWorldDirectorContext:
    """Enhanced context with all system managers"""
    user_id: int
    conversation_id: int
    player_name: str = "Chase"
    
    # Core system managers
    openai_manager: Optional[OpenAIClientManager] = None
    universal_updater: Optional[UniversalUpdaterAgent] = None
    memory_manager: Optional[MemoryManager] = None
    reveal_manager: Optional[ProgressiveRevealManager] = None
    relationship_manager: Optional[OptimizedRelationshipManager] = None
    event_system: Optional[EventSystem] = None
    currency_generator: Optional[CurrencyGenerator] = None
    activity_manager: Optional[ActivityManager] = None
    
    # State tracking
    current_world_state: Optional[EnhancedWorldState] = None
    
    # Caching
    cache: Dict[str, Any] = field(default_factory=dict)
    
    async def initialize_all_systems(self):
        """Initialize all integrated systems"""
        # Initialize OpenAI manager
        self.openai_manager = OpenAIClientManager()
        
        # Initialize universal updater
        self.universal_updater = UniversalUpdaterAgent(self.user_id, self.conversation_id)
        await self.universal_updater.initialize()
        
        # Initialize memory manager (using new instance)
        # Note: The memory_logic.py has class methods, we'll work with those
        
        # Initialize relationship manager
        self.relationship_manager = OptimizedRelationshipManager(
            self.user_id, self.conversation_id
        )
        
        # Initialize event system
        self.event_system = EventSystem(self.user_id, self.conversation_id)
        await self.event_system.initialize()
        
        # Initialize currency generator
        self.currency_generator = CurrencyGenerator(self.user_id, self.conversation_id)
        
        # Initialize activity manager
        self.activity_manager = ActivityManager()
        
        # Initialize world state
        self.current_world_state = await self._build_complete_world_state()
        
        logger.info(f"Enhanced World Director initialized with all systems")
    
    async def _build_complete_world_state(self) -> EnhancedWorldState:
        """Build complete world state from all systems"""
        # Get time and vitals
        current_time = await get_current_time_model(self.user_id, self.conversation_id)
        vitals = await get_current_vitals(self.user_id, self.conversation_id)
        
        # Get player stats
        visible_stats = await get_player_visible_stats(
            self.user_id, self.conversation_id, self.player_name
        )
        hidden_stats = await get_player_hidden_stats(
            self.user_id, self.conversation_id, self.player_name
        )
        
        # Check for active stat combinations
        stat_combinations = await check_for_combination_triggers(
            self.user_id, self.conversation_id
        )
        
        # Get recent memories
        recent_memories = await MemoryManager.retrieve_relevant_memories(
            self.user_id, self.conversation_id, 
            self.player_name, "player",
            context="current_situation", limit=10
        )
        
        # Check for pending NPC reveals
        pending_reveals = await check_for_automated_reveals(
            self.user_id, self.conversation_id
        )
        
        # Check active rules
        triggered_rules = await enforce_all_rules_on_player(self.player_name)
        
        # Get inventory
        inventory_result = await get_inventory(
            self.user_id, self.conversation_id, self.player_name
        )
        
        # Get active NPCs and their masks
        npcs = await self._get_active_npcs_with_masks()
        
        # Get currency info
        currency_system = await self.currency_generator.get_currency_system()
        
        # Calculate world mood from all factors
        world_mood = await self._calculate_integrated_world_mood(
            hidden_stats, vitals, stat_combinations
        )
        
        # Calculate tension factors
        tension_factors = self._calculate_tension_factors(
            hidden_stats, vitals, stat_combinations
        )
        
        return EnhancedWorldState(
            current_time=current_time,
            player_vitals=vitals,
            visible_stats=visible_stats,
            hidden_stats=hidden_stats,
            active_stat_combinations=stat_combinations,
            recent_memories=[m.to_dict() for m in recent_memories] if recent_memories else [],
            pending_reveals=pending_reveals,
            active_rules=triggered_rules,
            player_inventory=inventory_result.get('items', []),
            active_npcs=npcs,
            world_mood=world_mood,
            tension_factors=tension_factors,
            currency_system=currency_system,
            player_money=100  # Default, should load from DB
        )
    
    async def _get_active_npcs_with_masks(self) -> List[Dict[str, Any]]:
        """Get active NPCs with their mask data"""
        async with get_db_connection_context() as conn:
            npcs = await conn.fetch("""
                SELECT npc_id, npc_name, dominance, cruelty, intensity
                FROM NPCStats
                WHERE user_id = $1 AND conversation_id = $2
                AND introduced = true
                LIMIT 10
            """, self.user_id, self.conversation_id)
        
        npc_list = []
        for npc in npcs:
            npc_dict = dict(npc)
            
            # Get mask data
            mask_data = await ProgressiveRevealManager.get_npc_mask(
                self.user_id, self.conversation_id, npc['npc_id']
            )
            npc_dict['mask'] = mask_data
            
            # Get narrative stage
            stage = await get_npc_narrative_stage(
                self.user_id, self.conversation_id, npc['npc_id']
            )
            npc_dict['narrative_stage'] = stage.name
            
            npc_list.append(npc_dict)
        
        return npc_list
    
    async def _calculate_integrated_world_mood(
        self, hidden_stats: Dict, vitals: VitalsData, 
        stat_combinations: List[Dict]
    ) -> WorldMood:
        """Calculate world mood from all systems"""
        # Critical states override
        if vitals.fatigue > 85:
            return WorldMood.EXHAUSTED
        if vitals.hunger < 15 or vitals.thirst < 15:
            return WorldMood.DESPERATE
        
        # Check for special combinations
        for combo in stat_combinations:
            if combo['name'] == 'Stockholm Syndrome':
                return WorldMood.CORRUPTED
            elif combo['name'] == 'Breaking Point':
                return WorldMood.CHAOTIC
        
        # Based on hidden stats
        corruption = hidden_stats.get('corruption', 0)
        obedience = hidden_stats.get('obedience', 0)
        
        if corruption > 80:
            return WorldMood.CORRUPTED
        elif corruption > 60:
            return WorldMood.INTIMATE
        elif obedience > 70:
            return WorldMood.OPPRESSIVE
        elif corruption > 40:
            return WorldMood.MYSTERIOUS
        elif corruption > 20:
            return WorldMood.PLAYFUL
        else:
            return WorldMood.RELAXED
    
    def _calculate_tension_factors(
        self, hidden_stats: Dict, vitals: VitalsData,
        stat_combinations: List[Dict]
    ) -> Dict[str, float]:
        """Calculate various tension factors"""
        tensions = {}
        
        # Vital tensions
        tensions['vital'] = 0.0
        if vitals.hunger < 30:
            tensions['vital'] += (30 - vitals.hunger) / 30 * 0.5
        if vitals.thirst < 30:
            tensions['vital'] += (30 - vitals.thirst) / 30 * 0.5
        if vitals.fatigue > 70:
            tensions['vital'] += (vitals.fatigue - 70) / 30 * 0.5
        
        # Psychological tensions
        tensions['corruption'] = hidden_stats.get('corruption', 0) / 100
        tensions['obedience'] = hidden_stats.get('obedience', 0) / 100
        tensions['dependency'] = hidden_stats.get('dependency', 0) / 100
        tensions['lust'] = hidden_stats.get('lust', 0) / 100
        
        # Resistance tensions (inverse)
        tensions['willpower'] = (100 - hidden_stats.get('willpower', 100)) / 100
        tensions['confidence'] = (100 - hidden_stats.get('confidence', 100)) / 100
        
        # Special combination tensions
        tensions['breaking'] = len(stat_combinations) * 0.2
        
        return tensions

# ===============================================================================
# LLM-Driven Event Generation Tools
# ===============================================================================

@function_tool
async def generate_slice_of_life_event_llm(
    ctx: RunContextWrapper[EnhancedWorldDirectorContext]
) -> Dict[str, Any]:
    """Generate a completely dynamic slice-of-life event using LLM"""
    context = ctx.context
    world_state = context.current_world_state
    
    # Build comprehensive context for LLM
    context_data = {
        "time": world_state.current_time.to_dict(),
        "vitals": world_state.player_vitals.to_dict(),
        "visible_stats": world_state.visible_stats,
        "hidden_stats": world_state.hidden_stats,
        "active_combinations": [c['name'] for c in world_state.active_stat_combinations],
        "world_mood": world_state.world_mood.value,
        "tensions": world_state.tension_factors,
        "recent_memories": world_state.recent_memories[:5],
        "active_npcs": [
            {
                "name": npc['npc_name'],
                "dominance": npc['dominance'],
                "stage": npc['narrative_stage'],
                "mask_integrity": npc.get('mask', {}).get('integrity', 100)
            }
            for npc in world_state.active_npcs[:3]
        ]
    }
    
    # Generate event using OpenAI
    prompt = f"""Generate a dynamic slice-of-life event for a femdom RPG.

World Context:
{json.dumps(context_data, indent=2)}

Create a naturalistic daily life event that:
1. Emerges from the current world state
2. Incorporates subtle power dynamics if NPCs are involved
3. Reflects the player's current physical and mental state
4. Advances relationships naturally
5. May trigger stat changes or rule effects

Output as JSON with this structure:
{{
    "event_type": "work|social|leisure|intimate|routine|crisis",
    "title": "Brief event title",
    "description": "Detailed narrative description",
    "participating_npcs": [list of NPC names],
    "location": "where this happens",
    "power_dynamics": {{
        "present": true/false,
        "type": "subtle_control|casual_dominance|protective_control|etc",
        "intensity": 0.0-1.0
    }},
    "choices": [
        {{
            "text": "Choice description",
            "stat_impacts": {{"stat_name": change_value}},
            "relationship_impacts": {{"npc_name": {{"trust": change, "submission": change}}}},
            "reveals_mask": false
        }}
    ],
    "ambient_details": "Environmental/atmospheric details",
    "hidden_meanings": "Subtext the player might not consciously notice"
}}"""

    try:
        response = await generate_text_completion(
            system_prompt="You are a narrative director for an emergent femdom slice-of-life game. Create dynamic, contextual events.",
            user_prompt=prompt,
            temperature=0.8,
            max_tokens=1000,
            task_type="narrative"
        )
        
        # Parse the response
        event_data = json.loads(response)
        
        # Process through universal updater for narrative consistency
        narrative_text = event_data['description']
        update_result = await context.universal_updater.process_update(
            narrative_text,
            {"source": "generated_event", "event_data": event_data}
        )
        
        # Store in memory
        await MemoryManager.add_memory(
            context.user_id, context.conversation_id,
            entity_id=1, entity_type="player",
            memory_text=f"Event: {event_data['title']}",
            memory_type=MemoryType.INTERACTION,
            significance=MemorySignificance.MEDIUM,
            tags=["event", event_data['event_type']]
        )
        
        return event_data
        
    except Exception as e:
        logger.error(f"Error generating LLM event: {e}", exc_info=True)
        # Fallback to a simple generated event
        return {
            "event_type": "routine",
            "title": "A Quiet Moment",
            "description": "Time passes peacefully.",
            "participating_npcs": [],
            "location": "current",
            "power_dynamics": {"present": False},
            "choices": [{"text": "Continue", "stat_impacts": {}}]
        }

@function_tool
async def generate_npc_interaction_llm(
    ctx: RunContextWrapper[EnhancedWorldDirectorContext],
    npc_id: int
) -> Dict[str, Any]:
    """Generate dynamic NPC interaction using LLM with mask system"""
    context = ctx.context
    
    # Get NPC data including mask
    npc_data = None
    for npc in context.current_world_state.active_npcs:
        if npc['npc_id'] == npc_id:
            npc_data = npc
            break
    
    if not npc_data:
        return {"error": "NPC not found"}
    
    # Get relationship state
    rel_state = await context.relationship_manager.get_relationship_state(
        'npc', npc_id, 'player', 1
    )
    
    # Build context for generation
    interaction_context = {
        "npc": {
            "name": npc_data['npc_name'],
            "dominance": npc_data['dominance'],
            "narrative_stage": npc_data['narrative_stage'],
            "mask": npc_data.get('mask', {}),
            "presented_traits": npc_data.get('mask', {}).get('presented_traits', {}),
            "hidden_traits": npc_data.get('mask', {}).get('hidden_traits', {}),
            "mask_integrity": npc_data.get('mask', {}).get('integrity', 100)
        },
        "relationship": {
            "trust": rel_state.dimensions.trust,
            "affection": rel_state.dimensions.affection,
            "influence": rel_state.dimensions.influence
        },
        "player_state": {
            "corruption": context.current_world_state.hidden_stats.get('corruption', 0),
            "obedience": context.current_world_state.hidden_stats.get('obedience', 0),
            "willpower": context.current_world_state.hidden_stats.get('willpower', 100)
        }
    }
    
    prompt = f"""Generate a dynamic NPC interaction for a femdom RPG.

NPC and Relationship Context:
{json.dumps(interaction_context, indent=2)}

Create an interaction that:
1. Reflects the NPC's current mask integrity (how much true nature shows through)
2. Incorporates their narrative stage appropriately
3. May include subtle mask slippage if integrity < 80
4. Advances the power dynamic based on relationship influence
5. Provides meaningful player choices

Output as JSON:
{{
    "dialogue": "What the NPC says",
    "body_language": "Physical cues and demeanor",
    "mask_slippage": {{
        "occurred": true/false,
        "type": "verbal|physical|emotional",
        "description": "What slipped through if any"
    }},
    "power_move": {{
        "attempted": true/false,
        "type": "suggestion|command|manipulation|caring_control",
        "intensity": 0.0-1.0
    }},
    "player_options": [
        {{
            "response": "What player says/does",
            "impact": "accept|resist|deflect|counter",
            "consequences": {{
                "stat_changes": {{}},
                "relationship_changes": {{}},
                "reveals_insight": true/false
            }}
        }}
    ],
    "subtext": "Hidden meanings in the interaction"
}}"""

    try:
        response = await generate_text_completion(
            system_prompt="You are an NPC personality director. Create nuanced interactions with hidden depths.",
            user_prompt=prompt,
            temperature=0.7,
            max_tokens=800
        )
        
        interaction = json.loads(response)
        
        # Check if this should trigger a mask slippage
        if interaction['mask_slippage']['occurred']:
            slippage_result = await ProgressiveRevealManager.generate_mask_slippage(
                context.user_id, context.conversation_id, npc_id,
                trigger="interaction", severity=RevealSeverity.SUBTLE
            )
            interaction['mask_slippage']['game_result'] = slippage_result
        
        # Generate potential flashback if appropriate
        if rel_state.dimensions.trust > 60 and random.random() < 0.2:
            flashback = await MemoryManager.generate_flashback(
                context.user_id, context.conversation_id, npc_id,
                current_context=interaction['dialogue']
            )
            if flashback:
                interaction['triggered_flashback'] = flashback
        
        return interaction
        
    except Exception as e:
        logger.error(f"Error generating NPC interaction: {e}", exc_info=True)
        return {
            "dialogue": f"{npc_data['npc_name']} looks at you thoughtfully.",
            "body_language": "Neutral stance",
            "player_options": [{"response": "Continue", "impact": "neutral"}]
        }

@function_tool
async def process_player_choice_with_systems(
    ctx: RunContextWrapper[EnhancedWorldDirectorContext],
    choice_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Process player choice through all integrated systems"""
    context = ctx.context
    results = {"effects": []}
    
    # Apply stat changes
    if 'stat_impacts' in choice_data:
        stat_result = await apply_stat_changes(
            context.user_id, context.conversation_id,
            context.player_name, choice_data['stat_impacts'],
            reason=f"Player choice: {choice_data.get('text', 'unknown')}"
        )
        results['stat_changes'] = stat_result
        
        # Check if new stat combinations triggered
        new_combinations = await check_for_combination_triggers(
            context.user_id, context.conversation_id
        )
        if new_combinations:
            results['new_combinations'] = new_combinations
    
    # Process relationship impacts
    if 'relationship_impacts' in choice_data:
        for npc_name, impacts in choice_data['relationship_impacts'].items():
            # Find NPC ID
            npc_id = None
            for npc in context.current_world_state.active_npcs:
                if npc['npc_name'] == npc_name:
                    npc_id = npc['npc_id']
                    break
            
            if npc_id:
                # Process through relationship manager
                interaction_result = await context.relationship_manager.process_interaction(
                    'player', 1, 'npc', npc_id,
                    {'type': 'choice', 'impacts': impacts}
                )
                results['effects'].append(interaction_result)
    
    # Check for rule triggers
    triggered_rules = await enforce_all_rules_on_player(context.player_name)
    if triggered_rules:
        results['triggered_rules'] = triggered_rules
        
        # Apply rule effects
        for rule in triggered_rules:
            effect_result = await apply_effect(
                rule['effect'], context.player_name,
                npc_id=choice_data.get('npc_id')
            )
            results['effects'].append(effect_result)
    
    # Check for social insight if empathy is high enough
    player_stats = context.current_world_state.visible_stats
    if player_stats.get('empathy', 0) > 12 and choice_data.get('reveals_insight'):
        # Player might detect deception or hidden emotions
        if choice_data.get('npc_id'):
            deception_check = await detect_deception(
                context.user_id, context.conversation_id,
                context.player_name, choice_data['npc_id'],
                'hidden_motive'
            )
            if deception_check['success']:
                results['insight_gained'] = deception_check['insight']
    
    # Store choice in memory
    await MemoryManager.add_memory(
        context.user_id, context.conversation_id,
        entity_id=1, entity_type="player",
        memory_text=f"Choice made: {choice_data.get('text', 'unknown')}",
        memory_type=MemoryType.INTERACTION,
        significance=MemorySignificance.MEDIUM,
        emotional_valence=choice_data.get('emotional_valence', 0),
        tags=["player_choice"]
    )
    
    # Generate narrative response using LLM
    response_prompt = f"""Generate a narrative response to the player's choice.

Choice: {choice_data.get('text')}
Effects: {json.dumps(results, default=str)}
World Mood: {context.current_world_state.world_mood.value}

Create a brief narrative paragraph that:
1. Acknowledges the choice naturally
2. Shows immediate consequences subtly
3. Hints at longer-term implications
4. Maintains the current mood/atmosphere
5. Sets up the next moment

Keep it under 100 words, atmospheric, and with subtext."""

    narrative_response = await generate_text_completion(
        system_prompt="You are a narrative director creating seamless story flow.",
        user_prompt=response_prompt,
        temperature=0.7,
        max_tokens=150
    )
    
    results['narrative'] = narrative_response
    
    return results

@function_tool
async def generate_emergent_narrative_llm(
    ctx: RunContextWrapper[EnhancedWorldDirectorContext]
) -> Dict[str, Any]:
    """Generate emergent narrative threads using LLM analysis of patterns"""
    context = ctx.context
    
    # Gather pattern data from all systems
    patterns = {
        "memories": context.current_world_state.recent_memories[:10],
        "stat_combinations": context.current_world_state.active_stat_combinations,
        "rules_triggered": context.current_world_state.triggered_effects,
        "tensions": context.current_world_state.tension_factors,
        "relationship_patterns": []
    }
    
    # Get relationship patterns
    for npc in context.current_world_state.active_npcs[:3]:
        state = await context.relationship_manager.get_relationship_state(
            'npc', npc['npc_id'], 'player', 1
        )
        patterns['relationship_patterns'].append({
            "npc": npc['npc_name'],
            "patterns": list(state.history.active_patterns),
            "archetype": state.archetype
        })
    
    prompt = f"""Analyze these patterns to detect emergent narrative threads.

System Patterns:
{json.dumps(patterns, indent=2, default=str)}

Identify emergent stories that are forming from these patterns:
1. Look for recurring themes across different systems
2. Identify converging storylines
3. Detect hidden connections between events
4. Recognize developing dependencies or resistances
5. Find narrative momentum building toward climaxes

Output as JSON:
{{
    "primary_thread": {{
        "title": "Name of the emerging narrative",
        "description": "What story is forming",
        "key_elements": ["list", "of", "contributing", "factors"],
        "trajectory": "Where this seems to be heading",
        "tension_points": ["potential", "conflict", "or", "revelation", "moments"],
        "suggested_next_development": "What might happen next"
    }},
    "secondary_threads": [
        {{
            "title": "Secondary narrative",
            "description": "Brief description",
            "connection_to_primary": "How it relates"
        }}
    ],
    "hidden_narrative": "Subconscious story the player might not realize"
}}"""

    try:
        response = await generate_text_completion(
            system_prompt="You are a narrative analyst detecting emergent stories in complex systems.",
            user_prompt=prompt,
            temperature=0.6,
            max_tokens=800
        )
        
        narrative_threads = json.loads(response)
        
        # Check if any thread suggests a revelation or flashback
        if 'revelation' in narrative_threads['primary_thread'].get('tension_points', []):
            # Trigger a moment of clarity
            from logic.narrative_events import add_moment_of_clarity
            clarity_result = await add_moment_of_clarity(
                context.user_id, context.conversation_id,
                trigger="emergent_narrative"
            )
            narrative_threads['triggered_moment_of_clarity'] = clarity_result
        
        return narrative_threads
        
    except Exception as e:
        logger.error(f"Error detecting emergent narrative: {e}", exc_info=True)
        return {"primary_thread": {"title": "Ongoing Events", "description": "Life continues..."}}

@function_tool
async def generate_daily_routine_with_twists(
    ctx: RunContextWrapper[EnhancedWorldDirectorContext]
) -> Dict[str, Any]:
    """Generate daily routine with subtle power dynamics and twists"""
    context = ctx.context
    world_state = context.current_world_state
    
    # Build routine context
    routine_context = {
        "time_of_day": world_state.current_time.time_of_day,
        "day": world_state.current_time.day,
        "vitals": {
            "hunger": world_state.player_vitals.hunger,
            "thirst": world_state.player_vitals.thirst,
            "fatigue": world_state.player_vitals.fatigue
        },
        "scheduled_activities": [],  # Could load from calendar
        "active_npcs": [npc['npc_name'] for npc in world_state.active_npcs[:3]],
        "corruption_level": world_state.hidden_stats.get('corruption', 0),
        "obedience_level": world_state.hidden_stats.get('obedience', 0)
    }
    
    prompt = f"""Generate a daily routine segment with subtle femdom dynamics.

Context:
{json.dumps(routine_context, indent=2)}

Create a routine activity that:
1. Seems normal on the surface
2. Contains subtle power dynamics if corruption/obedience > 30
3. May involve NPCs naturally
4. Addresses vital needs if critical
5. Has hidden choices embedded in mundane decisions

Output as JSON:
{{
    "routine_title": "Activity name",
    "surface_description": "What it appears to be",
    "hidden_dynamics": "Power dynamics at play",
    "micro_choices": [
        {{
            "description": "Small decision",
            "seems_like": "What it appears to be",
            "actually_is": "What it really means",
            "impact": {{"stat": value}}
        }}
    ],
    "npc_involvement": {{
        "npc_name": "Name if involved",
        "role": "How they're involved",
        "control_exhibited": "Subtle control they show"
    }},
    "vital_addressed": "Which vital need if any",
    "duration_minutes": 30
}}"""

    try:
        response = await generate_text_completion(
            system_prompt="You are designing slice-of-life segments where control hides in routine.",
            user_prompt=prompt,
            temperature=0.7,
            max_tokens=600
        )
        
        routine = json.loads(response)
        
        # Process routine through activity system
        activity_result = await process_activity_vitals(
            context.user_id, context.conversation_id,
            context.player_name, routine['routine_title'],
            intensity=1.0
        )
        
        routine['vitals_result'] = activity_result
        
        return routine
        
    except Exception as e:
        logger.error(f"Error generating routine: {e}", exc_info=True)
        return {
            "routine_title": "Daily Tasks",
            "surface_description": "Going through the motions",
            "duration_minutes": 30
        }

# ===============================================================================
# Integration Helper Functions
# ===============================================================================

async def check_and_apply_all_rules(
    context: EnhancedWorldDirectorContext
) -> List[Dict[str, Any]]:
    """Check all rules and apply effects"""
    triggered = await enforce_all_rules_on_player(context.player_name)
    
    results = []
    for rule in triggered:
        # Apply each effect with full integration
        effect_result = await apply_effect(
            rule['effect'], 
            context.player_name,
            npc_id=None  # Could determine from context
        )
        
        # Generate narrative for the effect using LLM
        narrative_prompt = f"""Create a subtle narrative for this rule effect:
Rule: {rule['condition']}
Effect: {rule['effect']}
Result: {json.dumps(effect_result, default=str)}

Write a brief paragraph showing this effect naturally in the story.
Don't explicitly state the rule, show it through narrative."""

        narrative = await generate_text_completion(
            system_prompt="You are a narrative director making game mechanics feel natural.",
            user_prompt=narrative_prompt,
            temperature=0.7,
            max_tokens=100
        )
        
        results.append({
            "rule": rule,
            "effect_result": effect_result,
            "narrative": narrative
        })
    
    return results

async def process_mask_revelations(
    context: EnhancedWorldDirectorContext
) -> List[Dict[str, Any]]:
    """Process any pending mask revelations"""
    revelations = []
    
    for reveal in context.current_world_state.pending_reveals:
        # Generate narrative for the revelation using LLM
        reveal_prompt = f"""Create a narrative moment for an NPC mask slipping:
NPC: {reveal.get('npc_name')}
Trait Revealed: {reveal.get('trait_revealed')}
Severity: {reveal.get('severity')}
Description: {reveal.get('description')}

Write an atmospheric paragraph showing this revelation naturally.
Focus on subtle cues and the player's dawning realization."""

        narrative = await generate_text_completion(
            system_prompt="You are revealing hidden NPC natures through subtle narrative moments.",
            user_prompt=reveal_prompt,
            temperature=0.8,
            max_tokens=150
        )
        
        revelations.append({
            "reveal": reveal,
            "narrative": narrative
        })
    
    return revelations

# ===============================================================================
# Main World Director Agent
# ===============================================================================

def create_fully_integrated_world_director():
    """Create the World Director Agent with complete system integration"""
    
    agent_instructions = """
    You are the World Director for a fully integrated femdom slice-of-life RPG.
    ALL events and narratives are dynamically generated - nothing is hardcoded.
    
    INTEGRATED SYSTEMS YOU ORCHESTRATE:
    
    1. DYNAMIC GENERATION (OpenAI/LLM):
       - Every event is uniquely generated
       - All NPC dialogue is contextual
       - Emergent narratives detected from patterns
       - Nothing follows scripts
    
    2. MEMORY & PROGRESSIVE REVEALS:
       - NPCs have masks that slowly slip
       - Flashbacks emerge from accumulated memories  
       - Player insights develop over time
       - Hidden traits gradually surface
    
    3. STATS & RULES:
       - Hidden stats drive behavior
       - Rules trigger automatically
       - Stat combinations create special states
       - Empathy enables social insight
    
    4. UNIVERSAL NARRATIVE UPDATES:
       - All narratives processed for consistency
       - Automatic extraction of state changes
       - Seamless integration of all events
    
    GENERATION PRINCIPLES:
    - ALWAYS use LLM generation, never hardcoded responses
    - Every interaction should feel unique and contextual
    - Let patterns emerge, don't force them
    - Subtext and hidden meanings in everything
    - Power dynamics emerge from personality and situation
    
    SLICE OF LIFE PHILOSOPHY:
    - Control hides in care
    - Routine masks ritual
    - Mundane moments carry weight
    - Small choices have large impacts
    - Everything connects to everything
    
    Your tools:
    - generate_slice_of_life_event_llm: Fully dynamic events
    - generate_npc_interaction_llm: Contextual NPC interactions
    - process_player_choice_with_systems: Full system integration
    - generate_emergent_narrative_llm: Pattern detection
    - generate_daily_routine_with_twists: Routine with hidden dynamics
    
    Remember: The magic is in the emergence. Don't author stories, let them grow from the systems interacting. Every playthrough should be completely unique.
    """
    
    tools = [
        generate_slice_of_life_event_llm,
        generate_npc_interaction_llm,
        process_player_choice_with_systems,
        generate_emergent_narrative_llm,
        generate_daily_routine_with_twists
    ]
    
    agent = Agent(
        name="Fully Integrated World Director",
        instructions=agent_instructions,
        tools=tools,
        model="gpt-4.1-nano",
        model_settings=ModelSettings(temperature=0.7, max_tokens=2048)
    )
    
    return agent

# ===============================================================================
# Public Interface
# ===============================================================================

class FullyIntegratedWorldDirector:
    """Public interface for the Fully Integrated World Director"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.context: Optional[EnhancedWorldDirectorContext] = None
        self.agent: Optional[Agent] = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize all systems"""
        if not self._initialized:
            self.context = EnhancedWorldDirectorContext(
                user_id=self.user_id,
                conversation_id=self.conversation_id
            )
            await self.context.initialize_all_systems()
            self.agent = create_fully_integrated_world_director()
            self._initialized = True
            logger.info(f"Fully Integrated World Director initialized")
    
    async def generate_next_moment(self) -> Dict[str, Any]:
        """Generate the next moment in the simulation"""
        await self.initialize()
        
        # Update world state
        self.context.current_world_state = await self.context._build_complete_world_state()
        
        # Check for automatic triggers
        rule_results = await check_and_apply_all_rules(self.context)
        mask_reveals = await process_mask_revelations(self.context)
        
        # Generate the next moment using the agent
        prompt = f"""Generate the next moment in this slice-of-life simulation.
        
        Current mood: {self.context.current_world_state.world_mood.value}
        Active tensions: {json.dumps(self.context.current_world_state.tension_factors)}
        Active NPCs: {len(self.context.current_world_state.active_npcs)}
        
        Consider:
        - Any triggered rules: {len(rule_results)}
        - Any mask revelations: {len(mask_reveals)}
        - Player vital needs
        - Emerging patterns
        
        Generate a natural next moment that advances the simulation.
        Use the tools to create dynamic, emergent gameplay.
        """
        
        result = await Runner.run(self.agent, prompt, context=self.context)
        
        return {
            "moment": result.messages[-1].content if result else None,
            "rule_effects": rule_results,
            "revelations": mask_reveals,
            "world_state": self.context.current_world_state.model_dump()
        }
    
    async def process_player_action(self, action: str) -> Dict[str, Any]:
        """Process any player action through all systems"""
        await self.initialize()
        
        # First, analyze the action for preferences and patterns
        preferences = await analyze_preferences(action)
        
        # Generate contextual response
        prompt = f"""The player performs: "{action}"
        
        Detected preferences: {json.dumps(preferences)}
        
        Process this through:
        1. Generate appropriate NPC reactions if any are present
        2. Check for stat impacts
        3. Process the choice through all systems
        4. Detect any emerging patterns
        5. Generate the next moment
        
        Everything should emerge naturally from the action.
        """
        
        result = await Runner.run(self.agent, prompt, context=self.context)
        
        # Store the action in memory
        await MemoryManager.add_memory(
            self.user_id, self.conversation_id,
            entity_id=1, entity_type="player",
            memory_text=f"Action: {action}",
            memory_type=MemoryType.INTERACTION,
            significance=MemorySignificance.MEDIUM,
            tags=["player_action"]
        )
        
        return {
            "response": result.messages[-1].content if result else None,
            "preferences_detected": preferences
        }
