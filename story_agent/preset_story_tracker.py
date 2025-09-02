# story_agent/preset_story_tracker.py
"""
Refactored preset story tracker for open-world slice-of-life simulation.
Bridges linear preset stories with emergent gameplay while maintaining backwards compatibility.
"""

import json
import logging
import random
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from db.connection import get_db_connection_context
from story_templates.preset_stories import PresetStory, StoryBeat
from story_agent.world_simulation_models import (
    WorldDirector,
    SliceOfLifeEvent,
    PowerExchange,
    WorldMood,
    ActivityType,
    PowerDynamicType,
    NPCRoutine
)

logger = logging.getLogger(__name__)


class BeatTriggerType(Enum):
    """How preset beats map to open-world events"""
    RELATIONSHIP_THRESHOLD = "relationship_threshold"
    POWER_EXCHANGE = "power_exchange"
    WORLD_STATE = "world_state"
    TIME_BASED = "time_based"
    LOCATION_BASED = "location_based"
    EMERGENT = "emergent"


@dataclass
class BeatMapping:
    """Maps a preset story beat to open-world mechanics"""
    beat_id: str
    trigger_type: BeatTriggerType
    slice_event_type: Optional[ActivityType] = None
    power_dynamic_type: Optional[PowerDynamicType] = None
    required_mood: Optional[WorldMood] = None
    tension_threshold: float = 0.0
    can_emerge_naturally: bool = True
    force_after_hours: Optional[int] = None  # Force trigger after X hours if not naturally occurred


class PresetStoryTracker:
    """
    Tracks preset stories within the open-world simulation.
    Instead of forcing linear progression, it monitors for organic beat opportunities.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.current_story_id: Optional[str] = None
        self.completed_beats: List[str] = []
        self.current_act: int = 1  # Kept for compatibility
        self.story_variables: Dict[str, Any] = {}
        
        # New open-world integration
        self.beat_mappings: Dict[str, BeatMapping] = {}
        self.organic_beat_queue: List[str] = []  # Beats ready to trigger naturally
        self.forced_beat_timers: Dict[str, datetime] = {}  # Track forced triggers
        self.world_director: Optional[WorldDirector] = None
        
        # Dynamic entity mapping
        self.npc_mappings: Dict[str, int] = {}  # Map preset NPC roles to actual NPC IDs
        self.location_mappings: Dict[str, str] = {}  # Map preset locations to actual locations
        self.conflict_mappings: Dict[str, str] = {}  # Map preset conflicts to world tensions
        
        # Emergent tracking
        self.emergent_opportunities: List[Dict[str, Any]] = []
        self.relationship_snapshots: Dict[int, Dict[str, float]] = {}
        self.last_world_state_check: Optional[datetime] = None
    
    async def initialize_preset_story(self, story: PresetStory):
        """
        Initialize tracking for a preset story in the open world.
        Maps story elements to world systems rather than forcing linear progression.
        """
        self.current_story_id = story.id
        
        # Initialize world director connection
        self.world_director = WorldDirector(self.user_id, self.conversation_id)
        await self.world_director.initialize()
        
        # Map story beats to open-world triggers
        await self._create_beat_mappings(story)
        
        # Map entities to the dynamic world
        await self._map_preset_to_dynamic_entities(story)
        
        # Store initial state
        async with get_db_connection_context() as conn:
            await conn.execute("""
                INSERT INTO PresetStoryProgress (
                    user_id, conversation_id, story_id, 
                    current_act, completed_beats, story_variables,
                    beat_mappings, entity_mappings
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (user_id, conversation_id) 
                DO UPDATE SET 
                    story_id = $3,
                    beat_mappings = $7,
                    entity_mappings = $8,
                    updated_at = NOW()
            """, 
            self.user_id, self.conversation_id, story.id, 
            1, json.dumps([]), json.dumps({}),
            json.dumps(self._serialize_beat_mappings()),
            json.dumps({
                'npcs': self.npc_mappings,
                'locations': self.location_mappings,
                'conflicts': self.conflict_mappings
            }))
        
        logger.info(f"Initialized preset story '{story.name}' in open-world mode")
    
    async def check_beat_triggers(self, context: Dict[str, Any]) -> Optional[StoryBeat]:
        """
        Check if any story beats should trigger based on current world state.
        Now checks for organic emergence rather than forced progression.
        """
        story = await self._get_current_story()
        if not story:
            return None
        
        # Get current world state
        world_state = await self.world_director.get_world_state()
        
        # Check each unmapped beat for trigger conditions
        for beat in story.story_beats:
            if beat.id in self.completed_beats:
                continue
            
            mapping = self.beat_mappings.get(beat.id)
            if not mapping:
                continue
            
            # Check if beat can trigger organically
            if await self._check_organic_trigger(beat, mapping, world_state, context):
                logger.info(f"Beat '{beat.name}' triggered organically")
                return beat
            
            # Check if beat should be forced (time limit exceeded)
            if await self._check_forced_trigger(beat, mapping):
                logger.info(f"Beat '{beat.name}' force-triggered after timeout")
                return beat
        
        return None
    
    async def _check_organic_trigger(
        self, 
        beat: StoryBeat, 
        mapping: BeatMapping,
        world_state: Any,
        context: Dict[str, Any]
    ) -> bool:
        """Check if a beat can trigger naturally based on world conditions"""
        
        if mapping.trigger_type == BeatTriggerType.RELATIONSHIP_THRESHOLD:
            # Check relationship levels
            for npc_id in self.npc_mappings.values():
                relationship = await self._get_relationship_state(npc_id)
                for condition, threshold in beat.trigger_conditions.items():
                    if condition in relationship and relationship[condition] >= threshold:
                        return True
        
        elif mapping.trigger_type == BeatTriggerType.POWER_EXCHANGE:
            # Check if recent power exchanges match the beat's theme
            recent_exchanges = world_state.recent_power_exchanges
            for exchange in recent_exchanges:
                if mapping.power_dynamic_type and exchange['type'] == mapping.power_dynamic_type.value:
                    return True
        
        elif mapping.trigger_type == BeatTriggerType.WORLD_STATE:
            # Check world mood and tensions
            if mapping.required_mood and world_state.world_mood == mapping.required_mood:
                dominant_tension, level = world_state.world_tension.get_dominant_tension()
                if level >= mapping.tension_threshold:
                    return True
        
        elif mapping.trigger_type == BeatTriggerType.TIME_BASED:
            # Check if enough time has passed
            time_condition = beat.trigger_conditions.get('hours_elapsed', 0)
            if context.get('hours_played', 0) >= time_condition:
                return True
        
        elif mapping.trigger_type == BeatTriggerType.LOCATION_BASED:
            # Check if player is in the right location
            current_location = context.get('current_location', '')
            for required_loc in beat.required_locations:
                if self.location_mappings.get(required_loc, '') == current_location:
                    return True
        
        elif mapping.trigger_type == BeatTriggerType.EMERGENT:
            # Check if emergent conditions align
            if await self._check_emergent_alignment(beat, world_state):
                return True
        
        return False
    
    async def _check_forced_trigger(self, beat: StoryBeat, mapping: BeatMapping) -> bool:
        """Check if a beat should be forced due to time constraints"""
        if not mapping.force_after_hours:
            return False
        
        if beat.id not in self.forced_beat_timers:
            self.forced_beat_timers[beat.id] = datetime.now()
            return False
        
        time_elapsed = (datetime.now() - self.forced_beat_timers[beat.id]).total_seconds() / 3600
        return time_elapsed >= mapping.force_after_hours
    
    async def _check_emergent_alignment(self, beat: StoryBeat, world_state: Any) -> bool:
        """Check if emergent gameplay has naturally aligned with a beat's themes"""
        
        # Check if current events match beat themes
        for event in world_state.ongoing_events:
            # Check thematic alignment
            if self._themes_align(beat.dialogue_hints, event.description):
                return True
            
            # Check participant alignment
            required_npcs = set(beat.required_npcs)
            event_npcs = set(str(npc_id) for npc_id in event.participants)
            if required_npcs.intersection(event_npcs):
                return True
        
        # Check if relationship dynamics match
        if world_state.relationship_dynamics:
            dynamics = world_state.relationship_dynamics
            
            # Map beat conditions to dynamic values
            condition_map = {
                'submission': dynamics.player_submission_level,
                'control': dynamics.collective_control,
                'visibility': dynamics.power_visibility,
                'resistance': dynamics.resistance_level,
                'acceptance': dynamics.acceptance_level
            }
            
            for condition, required_value in beat.trigger_conditions.items():
                if condition in condition_map:
                    if condition_map[condition] >= required_value:
                        return True
        
        return False
    
    def _themes_align(self, dialogue_hints: List[str], event_description: str) -> bool:
        """Check if themes from a beat align with an event"""
        if not dialogue_hints:
            return False
        
        event_lower = event_description.lower()
        for hint in dialogue_hints:
            key_words = hint.lower().split()
            matching_words = sum(1 for word in key_words if word in event_lower)
            if matching_words >= len(key_words) * 0.4:  # 40% word match
                return True
        
        return False
    
    async def map_preset_to_dynamic_entities(self, story: PresetStory):
        """Map preset story entities to actual game entities in the open world"""
        
        # Map NPCs based on traits and roles
        for preset_npc in story.required_npcs:
            actual_npc = await self._find_or_create_matching_npc(preset_npc)
            if actual_npc:
                self.npc_mappings[preset_npc.get('id', preset_npc.get('role'))] = actual_npc['npc_id']
        
        # Map locations to world locations
        for preset_location in story.required_locations:
            actual_location = await self._find_or_create_matching_location(preset_location)
            if actual_location:
                self.location_mappings[preset_location.get('id', preset_location.get('name'))] = actual_location
        
        # Map conflicts to world tensions
        for preset_conflict in story.required_conflicts:
            tension_type = self._map_conflict_to_tension(preset_conflict)
            self.conflict_mappings[preset_conflict.get('id', preset_conflict.get('type'))] = tension_type
    
    async def _find_or_create_matching_npc(self, preset_npc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find an NPC that matches preset requirements or adapt/create one"""
        
        async with get_db_connection_context() as conn:
            # First, try to find existing NPC with matching traits
            query_conditions = []
            query_params = [self.user_id, self.conversation_id]
            param_count = 2
            
            if 'min_dominance' in preset_npc:
                param_count += 1
                query_conditions.append(f"dominance >= ${param_count}")
                query_params.append(preset_npc['min_dominance'])
            
            if 'gender' in preset_npc:
                param_count += 1
                query_conditions.append(f"gender = ${param_count}")
                query_params.append(preset_npc['gender'])
            
            if 'archetype' in preset_npc:
                param_count += 1
                query_conditions.append(f"archetype = ${param_count}")
                query_params.append(preset_npc['archetype'])
            
            where_clause = " AND ".join(query_conditions) if query_conditions else "TRUE"
            
            npcs = await conn.fetch(f"""
                SELECT npc_id, npc_name, dominance, personality_traits
                FROM NPCStats
                WHERE user_id = $1 AND conversation_id = $2 AND {where_clause}
                ORDER BY dominance DESC
                LIMIT 10
            """, *query_params)
            
            # Score NPCs based on match quality
            best_match = None
            best_score = 0
            
            for npc in npcs:
                score = self._calculate_npc_match_score(npc, preset_npc)
                if score > best_score:
                    best_score = score
                    best_match = npc
            
            if best_match and best_score > 0.6:  # 60% match threshold
                return dict(best_match)
            
            # No good match - create a new NPC that fits
            # This would integrate with your NPC creation system
            return await self._create_npc_for_preset(preset_npc)
    
    def _calculate_npc_match_score(self, npc: Any, preset_npc: Dict[str, Any]) -> float:
        """Calculate how well an NPC matches preset requirements"""
        score = 0.0
        checks = 0
        
        # Check dominance
        if 'min_dominance' in preset_npc:
            checks += 1
            if npc['dominance'] >= preset_npc['min_dominance']:
                score += 1.0
            else:
                # Partial credit for being close
                score += max(0, 1.0 - (preset_npc['min_dominance'] - npc['dominance']) / 50)
        
        # Check personality traits
        if 'required_traits' in preset_npc and npc.get('personality_traits'):
            checks += 1
            npc_traits = json.loads(npc['personality_traits']) if isinstance(npc['personality_traits'], str) else npc['personality_traits']
            required_traits = preset_npc['required_traits']
            matching_traits = sum(1 for trait in required_traits if trait in npc_traits)
            score += matching_traits / len(required_traits) if required_traits else 0
        
        return score / checks if checks > 0 else 0
    
    async def _create_npc_for_preset(self, preset_npc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create an NPC specifically for the preset story"""
        # This would integrate with your NPC creation system
        # For now, return None to indicate no automatic creation
        logger.warning(f"Would create NPC for preset: {preset_npc}")
        return None
    
    async def _find_or_create_matching_location(self, preset_location: Dict[str, Any]) -> Optional[str]:
        """Find a location that matches preset requirements"""
        
        async with get_db_connection_context() as conn:
            # Try to find existing location
            location = await conn.fetchrow("""
                SELECT location_name FROM Locations
                WHERE user_id = $1 AND conversation_id = $2
                AND LOWER(location_name) LIKE LOWER($3)
                LIMIT 1
            """, self.user_id, self.conversation_id, f"%{preset_location.get('name', '')}%")
            
            if location:
                return location['location_name']
            
            # Check for type-based match
            if 'type' in preset_location:
                location = await conn.fetchrow("""
                    SELECT location_name FROM Locations
                    WHERE user_id = $1 AND conversation_id = $2
                    AND location_type = $3
                    LIMIT 1
                """, self.user_id, self.conversation_id, preset_location['type'])
                
                if location:
                    return location['location_name']
        
        return None
    
    def _map_conflict_to_tension(self, preset_conflict: Dict[str, Any]) -> str:
        """Map a preset conflict to world tension types"""
        conflict_type = preset_conflict.get('type', '').lower()
        
        tension_map = {
            'romantic': 'sexual',
            'power': 'power',
            'social': 'social',
            'mystery': 'mystery',
            'danger': 'conflict'
        }
        
        for key, value in tension_map.items():
            if key in conflict_type:
                return value
        
        return 'conflict'  # Default
    
    async def translate_beat_to_context(self, beat: StoryBeat) -> Dict[str, Any]:
        """
        Translate preset beat requirements to current game context.
        Makes beats work with the dynamic world rather than forcing specific setups.
        """
        context = {
            "beat_id": beat.id,
            "beat_name": beat.name,
            "can_trigger": True,
            "missing_requirements": []
        }
        
        # Map required NPCs to actual NPCs
        actual_npcs = []
        for preset_npc_id in beat.required_npcs:
            actual_id = self.npc_mappings.get(preset_npc_id)
            if actual_id:
                actual_npcs.append(actual_id)
            else:
                context["missing_requirements"].append(f"NPC: {preset_npc_id}")
                context["can_trigger"] = False
        context["required_npcs"] = actual_npcs
        
        # Map required locations
        actual_locations = []
        for preset_loc_id in beat.required_locations:
            actual_loc = self.location_mappings.get(preset_loc_id)
            if actual_loc:
                actual_locations.append(actual_loc)
            else:
                # Try to use any similar location
                actual_locations.append("any_suitable_location")
        context["required_locations"] = actual_locations
        
        # Add narrative stage requirements
        context["narrative_stage_requirements"] = beat.narrative_stage
        
        # Add world state preferences for organic triggering
        mapping = self.beat_mappings.get(beat.id)
        if mapping:
            context["preferred_activity_type"] = mapping.slice_event_type
            context["preferred_power_dynamic"] = mapping.power_dynamic_type
            context["preferred_mood"] = mapping.required_mood
            context["tension_threshold"] = mapping.tension_threshold
        
        return context
    
    async def complete_story_beat(self, beat_id: str, outcomes: Dict[str, Any]):
        """
        Mark a story beat as completed and apply outcomes to the open world.
        Outcomes now affect world state rather than linear progression.
        """
        self.completed_beats.append(beat_id)
        
        # Apply outcomes to world state
        if self.world_director:
            # Trigger power exchanges based on outcomes
            if 'power_shift' in outcomes:
                await self._apply_power_shift(outcomes['power_shift'])
            
            # Adjust world mood based on outcomes
            if 'mood_change' in outcomes:
                await self.world_director.adjust_world_mood(
                    outcomes['mood_change']['target_mood'],
                    outcomes['mood_change'].get('intensity', 0.5)
                )
            
            # Create follow-up events
            if 'spawn_events' in outcomes:
                for event_data in outcomes['spawn_events']:
                    await self._spawn_slice_event(event_data)
        
        # Update story variables
        if 'variable_changes' in outcomes:
            self.story_variables.update(outcomes['variable_changes'])
        
        # Check for act progression (kept for compatibility but less rigid)
        await self._check_act_progression()
        
        # Store progress
        await self.save_progress()
        
        logger.info(f"Completed beat '{beat_id}' with world integration")
    
    async def _apply_power_shift(self, power_shift: Dict[str, Any]):
        """Apply power shift outcomes to the world"""
        if not self.world_director:
            return
        
        npc_id = power_shift.get('npc_id')
        if npc_id and npc_id in self.npc_mappings.values():
            exchange_type = power_shift.get('type', 'subtle_control')
            intensity = power_shift.get('intensity', 0.5)
            
            # Trigger a power exchange in the world
            await self.world_director.trigger_power_exchange(
                npc_id=npc_id,
                exchange_type=exchange_type,
                intensity=intensity,
                is_public=power_shift.get('is_public', False)
            )
    
    async def _spawn_slice_event(self, event_data: Dict[str, Any]):
        """Spawn a slice-of-life event as a beat outcome"""
        if not self.world_director:
            return
        
        # Map preset event to slice-of-life event
        event_type = event_data.get('type', 'routine')
        involved_npcs = [
            self.npc_mappings.get(npc_id, npc_id) 
            for npc_id in event_data.get('npcs', [])
        ]
        
        event = await self.world_director.generate_slice_of_life_event(
            event_type=event_type,
            involved_npcs=involved_npcs,
            preferred_mood=event_data.get('mood')
        )
        
        logger.info(f"Spawned slice-of-life event: {event.title}")
    
    async def _check_act_progression(self):
        """
        Check for act progression. In open world, acts are more like chapters
        that mark major relationship/story milestones rather than linear progression.
        """
        story = await self._get_current_story()
        if not story:
            return
        
        # Count completed beats per act
        act_beats = {}
        for beat in story.story_beats:
            act = self._get_beat_act(beat, story)
            if act not in act_beats:
                act_beats[act] = {'total': 0, 'completed': 0}
            act_beats[act]['total'] += 1
            if beat.id in self.completed_beats:
                act_beats[act]['completed'] += 1
        
        # Check if current act is mostly complete (60% threshold for flexibility)
        if self.current_act in act_beats:
            completion_ratio = act_beats[self.current_act]['completed'] / act_beats[self.current_act]['total']
            if completion_ratio >= 0.6:
                # Progress to next act
                self.current_act += 1
                logger.info(f"Progressed to Act {self.current_act}")
                
                # Trigger act transition in world
                if self.world_director:
                    # Acts can influence world state
                    await self._apply_act_transition_effects()
    
    def _get_beat_act(self, beat: StoryBeat, story: PresetStory) -> int:
        """Determine which act a beat belongs to"""
        for i, act in enumerate(story.acts, 1):
            if beat.id in act.get('beats', []):
                return i
        return 1  # Default to act 1
    
    async def _apply_act_transition_effects(self):
        """Apply effects when transitioning between acts"""
        if not self.world_director:
            return
        
        # Each act can shift the world tone
        act_moods = {
            2: WorldMood.TENSE,
            3: WorldMood.INTIMATE,
            4: WorldMood.OPPRESSIVE,
            5: WorldMood.CHAOTIC
        }
        
        if self.current_act in act_moods:
            await self.world_director.adjust_world_mood(
                act_moods[self.current_act].value,
                intensity=0.7
            )
    
    async def save_progress(self):
        """Save current progress to database"""
        async with get_db_connection_context() as conn:
            await conn.execute("""
                UPDATE PresetStoryProgress
                SET current_act = $3,
                    completed_beats = $4,
                    story_variables = $5,
                    beat_mappings = $6,
                    updated_at = NOW()
                WHERE user_id = $1 AND conversation_id = $2
            """, 
            self.user_id, self.conversation_id,
            self.current_act,
            json.dumps(self.completed_beats),
            json.dumps(self.story_variables),
            json.dumps(self._serialize_beat_mappings()))
    
    async def _create_beat_mappings(self, story: PresetStory):
        """Create mappings between story beats and open-world triggers"""
        for beat in story.story_beats:
            # Analyze beat to determine best trigger type
            trigger_type = self._analyze_beat_trigger_type(beat)
            
            # Map to slice-of-life event types
            slice_type = self._map_beat_to_activity_type(beat)
            
            # Map to power dynamics if relevant
            power_type = self._map_beat_to_power_dynamic(beat)
            
            # Determine if beat needs specific mood
            required_mood = self._analyze_required_mood(beat)
            
            # Create mapping
            self.beat_mappings[beat.id] = BeatMapping(
                beat_id=beat.id,
                trigger_type=trigger_type,
                slice_event_type=slice_type,
                power_dynamic_type=power_type,
                required_mood=required_mood,
                tension_threshold=beat.trigger_conditions.get('tension', 0.0),
                can_emerge_naturally=not beat.can_skip,
                force_after_hours=72 if not beat.can_skip else None  # Force critical beats after 3 days
            )
    
    def _analyze_beat_trigger_type(self, beat: StoryBeat) -> BeatTriggerType:
        """Analyze a beat to determine its trigger type"""
        conditions = beat.trigger_conditions
        
        # Check for relationship conditions
        relationship_keys = ['trust', 'dominance', 'submission', 'corruption', 'affection']
        if any(key in conditions for key in relationship_keys):
            return BeatTriggerType.RELATIONSHIP_THRESHOLD
        
        # Check for time conditions
        if 'hours_elapsed' in conditions or 'days_elapsed' in conditions:
            return BeatTriggerType.TIME_BASED
        
        # Check for location requirements
        if beat.required_locations:
            return BeatTriggerType.LOCATION_BASED
        
        # Check for power dynamics
        if any('power' in hint.lower() or 'control' in hint.lower() for hint in beat.dialogue_hints):
            return BeatTriggerType.POWER_EXCHANGE
        
        # Default to emergent
        return BeatTriggerType.EMERGENT
    
    def _map_beat_to_activity_type(self, beat: StoryBeat) -> Optional[ActivityType]:
        """Map a story beat to slice-of-life activity type"""
        # Analyze dialogue hints and description
        text = f"{beat.description} {' '.join(beat.dialogue_hints)}".lower()
        
        if 'work' in text or 'office' in text:
            return ActivityType.WORK
        elif 'social' in text or 'party' in text or 'gathering' in text:
            return ActivityType.SOCIAL
        elif 'intimate' in text or 'private' in text or 'bedroom' in text:
            return ActivityType.INTIMATE
        elif 'routine' in text or 'daily' in text:
            return ActivityType.ROUTINE
        elif 'special' in text or 'unique' in text:
            return ActivityType.SPECIAL
        
        return ActivityType.LEISURE
    
    def _map_beat_to_power_dynamic(self, beat: StoryBeat) -> Optional[PowerDynamicType]:
        """Map a story beat to power dynamic type"""
        text = f"{beat.description} {' '.join(beat.dialogue_hints)}".lower()
        
        if 'subtle' in text or 'gentle' in text:
            return PowerDynamicType.SUBTLE_CONTROL
        elif 'casual' in text or 'confident' in text:
            return PowerDynamicType.CASUAL_DOMINANCE
        elif 'protect' in text or 'care' in text:
            return PowerDynamicType.PROTECTIVE_CONTROL
        elif 'tease' in text or 'playful' in text:
            return PowerDynamicType.PLAYFUL_TEASING
        elif 'ritual' in text or 'pattern' in text:
            return PowerDynamicType.RITUAL_SUBMISSION
        elif 'financial' in text or 'money' in text:
            return PowerDynamicType.FINANCIAL_CONTROL
        elif 'public' in text or 'social' in text:
            return PowerDynamicType.SOCIAL_HIERARCHY
        elif 'command' in text or 'order' in text:
            return PowerDynamicType.INTIMATE_COMMAND
        
        return None
    
    def _analyze_required_mood(self, beat: StoryBeat) -> Optional[WorldMood]:
        """Determine if a beat requires a specific world mood"""
        text = f"{beat.description} {' '.join(beat.dialogue_hints)}".lower()
        
        if 'relaxed' in text or 'calm' in text:
            return WorldMood.RELAXED
        elif 'tense' in text or 'nervous' in text:
            return WorldMood.TENSE
        elif 'playful' in text or 'fun' in text:
            return WorldMood.PLAYFUL
        elif 'intimate' in text or 'close' in text:
            return WorldMood.INTIMATE
        elif 'mysterious' in text or 'unknown' in text:
            return WorldMood.MYSTERIOUS
        elif 'oppressive' in text or 'heavy' in text:
            return WorldMood.OPPRESSIVE
        elif 'chaotic' in text or 'wild' in text:
            return WorldMood.CHAOTIC
        
        return None
    
    def _serialize_beat_mappings(self) -> List[Dict[str, Any]]:
        """Serialize beat mappings for storage"""
        return [
            {
                'beat_id': mapping.beat_id,
                'trigger_type': mapping.trigger_type.value,
                'slice_event_type': mapping.slice_event_type.value if mapping.slice_event_type else None,
                'power_dynamic_type': mapping.power_dynamic_type.value if mapping.power_dynamic_type else None,
                'required_mood': mapping.required_mood.value if mapping.required_mood else None,
                'tension_threshold': mapping.tension_threshold,
                'can_emerge_naturally': mapping.can_emerge_naturally,
                'force_after_hours': mapping.force_after_hours
            }
            for mapping in self.beat_mappings.values()
        ]
    
    async def _get_current_story(self) -> Optional[PresetStory]:
        """Get the current preset story object"""
        if not self.current_story_id:
            return None
        
        # This would load from your story storage
        # For now, return None
        return None
    
    async def _get_relationship_state(self, npc_id: int) -> Dict[str, float]:
        """Get relationship state with an NPC"""
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow("""
                SELECT trust, respect, fear, arousal, resistance,
                       dominance, submission, affection, corruption
                FROM NPCRelationships
                WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
            """, self.user_id, self.conversation_id, npc_id)
            
            if row:
                return dict(row)
            
            return {}
    
    # Backwards compatibility methods
    async def get_active_beat(self) -> Optional[str]:
        """Get currently active beat (backwards compatibility)"""
        # In open world, there's no single active beat
        # Return the next organic beat if any
        if self.organic_beat_queue:
            return self.organic_beat_queue[0]
        return None
    
    async def force_beat_trigger(self, beat_id: str) -> bool:
        """Force a specific beat to trigger (backwards compatibility)"""
        if beat_id not in self.beat_mappings:
            return False
        
        # Add to organic queue to trigger next
        if beat_id not in self.organic_beat_queue:
            self.organic_beat_queue.append(beat_id)
        
        return True
