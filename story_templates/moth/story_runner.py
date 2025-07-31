# story_templates/moth/story_runner.py
"""
Main story runner for Queen of Thorns - Unified Slice-of-Life Sandbox
Features:
- Dynamic cast rotation with any NPC able to take spotlight
- Lilith can fade completely offscreen or return via RetconGate
- Episodes with dormancy and reactivation
- Proximity-based interactions
- Full agent integration with canon governance
"""

import logging
import json
import asyncio
from typing import Dict, Any, Optional, List, Tuple, Set
from datetime import datetime, timedelta
import random
from pydantic import BaseModel, Field, ValidationError
from functools import lru_cache
import os
from uuid import uuid4
from enum import Enum
from dataclasses import dataclass, field

from db.connection import get_db_connection_context
from story_templates.moth.story_initializer import (
    QueenOfThornsStoryInitializer, 
    QueenOfThornsStoryProgression,
    GPTService,
    CanonIntegrationService,
    with_canon_governance,
    propose_canonical_change,
    AgentType
)
from story_templates.moth.poem_enhanced_generation import ThornsEnhancedTextGenerator, integrate_thorns_enhancement
from story_templates.moth.npcs.lilith_mechanics import LilithMechanicsHandler
from story_templates.moth.lore.consistency_guide import QueenOfThornsConsistencyGuide
from memory.wrapper import MemorySystem
from lore.core import canon
from lore.core.context import CanonicalContext
from nyx.governance_helpers import remember_with_governance

logger = logging.getLogger(__name__)

# Environment variables for model selection
_INTENT_MODEL = os.getenv("OPENAI_INTENT_MODEL", "gpt-4.1-nano")
_CONTEXT_MODEL = os.getenv("OPENAI_CONTEXT_MODEL", "gpt-4.1-nano") 
_NARRATIVE_MODEL = os.getenv("OPENAI_NARRATIVE_MODEL", "gpt-4.1-nano")
_THREAT_MODEL = os.getenv("OPENAI_THREAT_MODEL", "gpt-4.1-nano")
_INNER_VOICE_MODEL = os.getenv("OPENAI_INNER_VOICE_MODEL", "gpt-4.1-nano")
_SCENE_MODEL = os.getenv("OPENAI_SCENE_MODEL", "gpt-4.1-nano")
_DIALOGUE_MODEL = os.getenv("OPENAI_DIALOGUE_MODEL", "gpt-4.1-nano")
_CONTINUITY_MODEL = os.getenv("OPENAI_CONTINUITY_MODEL", "gpt-4.1-nano")
_MEMORY_MODEL = os.getenv("OPENAI_MEMORY_MODEL", "gpt-4.1-nano")
_RIPPLE_MODEL = os.getenv("OPENAI_RIPPLE_MODEL", "gpt-4.1-nano")
_SCHEDULER_MODEL = os.getenv("OPENAI_SCHEDULER_MODEL", "gpt-4.1-nano")
_WEAVER_MODEL = os.getenv("OPENAI_WEAVER_MODEL", "gpt-4.1-nano")
_PULSE_MODEL = os.getenv("OPENAI_PULSE_MODEL", "gpt-4.1-nano")
_FOCUS_MODEL = os.getenv("OPENAI_FOCUS_MODEL", "gpt-4.1-nano")

# Enhanced Proximity with OFFSCREEN
class Proximity(str, Enum):
    TOGETHER = "together"
    SAME_VENUE = "same-venue" 
    DIFFERENT_VENUE = "different-venue"
    OFFSCREEN = "offscreen"  # Queen fully exits the story

# NPC tracking for dynamic cast
@dataclass
class NPCTrait:
    npc_id: int
    name: str
    affinity: int = 0          # -100 to +100
    visibility: int = 0        # scene-time counter
    spotlight: float = 0.0     # auto-calculated = aff*0.6 + vis*0.4
    canonical_tags: set = field(default_factory=set)
    last_seen: datetime = field(default_factory=datetime.now)
    
    def calculate_spotlight(self):
        """Update spotlight score based on affinity and visibility"""
        self.spotlight = (self.affinity * 0.6) + (self.visibility * 0.4)
        return self.spotlight

# Enhanced Episode model with cast tracking
class Episode(BaseModel):
    """Lightweight story container with cast and reactivation support"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    premise: str = Field(..., description="What's happening")
    stakes: str = Field(..., description="What matters about this")
    progress: int = Field(default=0, ge=0, le=100, description="0-100% completion")
    open_threads: List[str] = Field(default_factory=list, description="Unresolved elements")
    tags: List[str] = Field(default_factory=list, description="Genre/type tags")
    location_relevant: Optional[str] = Field(None, description="Specific location if any")
    network_related: bool = Field(default=False, description="Is this network business")
    lilith_involvement: str = Field(default="optional", description="How Lilith relates: central/interested/optional/none")
    cast: List[int] = Field(default_factory=list, description="NPC IDs involved")
    created_at: datetime = Field(default_factory=datetime.now)
    last_active: datetime = Field(default_factory=datetime.now)
    # Dormancy fields
    last_state: Optional[str] = Field(None, description="State when went dormant")
    unresolved_stakes: Optional[str] = Field(None, description="What was left hanging")
    reactivation_score: int = Field(default=0, description="How ready to return")
    dormant_since: Optional[datetime] = Field(None, description="When it went dormant")
    parent_ids: List[str] = Field(default_factory=list, description="Episodes this spawned from")
    child_ids: List[str] = Field(default_factory=list, description="Episodes this spawned")

# Focus Allocator output
class FocusAllocation(BaseModel):
    """Output of the Focus Allocator agent"""
    promote: Optional[int] = Field(None, description="NPC ID to promote to spotlight")
    demote: List[int] = Field(default_factory=list, description="NPC IDs to demote")
    new_episode_for: Optional[int] = Field(None, description="Create episode for this NPC")
    reasoning: str = Field(..., description="Why these focus changes")

# Enhanced Memory model
class MemoryShard(BaseModel):
    """Structured memory with actors and tags"""
    text: str
    actors: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    epoch: int = Field(default_factory=lambda: int(datetime.now().timestamp()))
    importance: int = Field(default=50, ge=0, le=100)

# Agent models
class IntentClassification(BaseModel):
    """Agent 1: Intent & Tone Classifier - enhanced for banishment"""
    intent: str = Field(..., description="Primary intent: question, request, action, observation, threat, invite, leave, banish")
    emotion: str = Field(..., description="Emotional tone: neutral, curious, aggressive, submissive, fearful")
    politeness: float = Field(..., ge=0, le=10, description="Politeness level 0-10")
    threat_level: float = Field(..., ge=0, le=10, description="Threat assessment 0-10")
    network_relevance: float = Field(..., ge=0, le=10, description="Relevance to network business")
    key_topics: List[str] = Field(default_factory=list, max_items=5)
    proximity_intent: Optional[str] = Field(None, description="Does player want to stay/leave/invite/banish")

class QueenSchedule(BaseModel):
    """Queen Scheduler output - enhanced for offscreen state"""
    location: str = Field(..., description="Where she is right now")
    activity: str = Field(..., description="What she's doing")
    mood: str = Field(..., description="Current emotional state")
    available_for: List[str] = Field(default_factory=list, description="What interactions work")
    next_transition: str = Field(..., description="When she'll move/change activity")
    join_preference: str = Field(default="neutral", description="How she feels about player joining")
    next_desire: Optional[str] = Field(None, description="What she wants from player")
    offscreen_status: Optional[str] = Field(None, description="If offscreen, what she's doing in her own life")

class SituationWeave(BaseModel):
    """Situation Weaver output - considers spotlight NPC"""
    spotlight_episode: Optional[str] = Field(None, description="Episode ID to highlight")
    new_episode: Optional[Episode] = Field(None, description="New episode to spawn")
    episode_updates: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    reasoning: str = Field(..., description="Why this situation now")
    tension_level: float = Field(default=0.5, ge=0, le=1)
    spotlight_npc_bias: Optional[int] = Field(None, description="Favor episodes with this NPC")

class DialogueScript(BaseModel):
    """Agent 7: Dialogue Playwright - handles any cast"""
    script: List[Dict[str, str]] = Field(..., description="Multi-speaker dialogue")
    subtext: Dict[str, str] = Field(default_factory=dict, description="Hidden meanings")
    power_dynamics: str = Field(..., max_length=200, description="Power flow description")
    priority_speaker: str = Field(..., description="Who gets focus")
    cast_present: List[str] = Field(default_factory=list, description="All speakers in scene")

class ContinuityCheck(BaseModel):
    """Agent 8: Continuity QA - checks offscreen violations"""
    ok: bool = Field(..., description="Passes continuity check")
    issues: List[str] = Field(default_factory=list, description="Any continuity problems")
    canon_violations: List[str] = Field(default_factory=list, description="Canon breaks")
    episode_logic: Dict[str, bool] = Field(default_factory=dict, description="Per-episode consistency")
    queen_offscreen_violation: bool = Field(default=False, description="Queen spoke while offscreen")

class LilithInnerVoice(BaseModel):
    """Agent 5: Lilith Inner-Voice - silent when offscreen"""
    mask_slip: bool = Field(..., description="Does mask slip in this moment")
    mask_integrity: int = Field(..., ge=0, le=100, description="Current mask strength")
    poetry_line: Optional[str] = Field(None, max_length=150, description="Poetic utterance if any")
    trauma_flash: Optional[str] = Field(None, max_length=100, description="Trauma memory if triggered")
    emotional_state: str = Field(..., description="Current emotional state")
    vulnerability_level: int = Field(..., ge=0, le=10)
    separation_feeling: Optional[str] = Field(None, description="How she feels about distance")
    meanwhile_glimpse: Optional[str] = Field(None, description="What she's doing when apart")
    is_offscreen: bool = Field(default=False, description="No output if true")

# Keep other agent models...
class ContextSummary(BaseModel):
    """Agent 2: Context Stitcher"""
    queen_summary: str = Field(..., max_length=200, description="1-2 sentence Queen state summary")
    location_summary: str = Field(..., max_length=200, description="1-2 sentence location summary")
    player_state_summary: str = Field(..., max_length=200, description="1-2 sentence player state")
    spotlight_npc_summary: Optional[str] = Field(None, max_length=200, description="Current spotlight NPC")
    relevant_history: List[str] = Field(default_factory=list, max_items=3, description="Key recent events")

class LooseThreadIndex(BaseModel):
    """Output of the LooseThreadIndexer agent"""
    high_weight_threads: List[Dict[str, Any]] = Field(..., max_items=10)
    reentry_opportunities: Dict[str, str] = Field(default_factory=dict)
    recommended_reactivations: List[str] = Field(default_factory=list)
    queen_return_potential: float = Field(default=0.0, ge=0, le=1, description="Should Queen return?")

class ReflectionLens(BaseModel):
    """Output when viewing scene without primary cast"""
    ambient_details: List[str] = Field(..., max_items=5, description="What's happening around")
    queen_echoes: List[str] = Field(default_factory=list, max_items=3, description="Her influence felt")
    npc_gossip: Optional[str] = Field(None, description="What others say")
    news_item: Optional[str] = Field(None, description="Media mention if relevant")
    spotlight_npc_presence: Optional[str] = Field(None, description="How spotlight NPC affects space")

class QueenOfThornsSliceRunner:
    """
    Unified slice-of-life sandbox runner for Queen of Thorns.
    - Dynamic cast with any NPC able to take spotlight
    - Lilith can fade completely offscreen or return
    - Episodes emerge and resolve organically
    - Full canon governance integration
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.story_id = "queen_of_thorns"
        
        # Component handlers
        self.text_generator = None
        self.lilith_mechanics = None
        self.memory_system = None
        self.gpt_service = GPTService()
        
        # Core state
        self.proximity: Proximity = Proximity.TOGETHER
        self.lilith_npc_id = None
        self.lilith_affinity = 50  # Start neutral
        self.cutoff_threshold = -75  # Below this, Queen goes offscreen
        
        # NPC cast tracking
        self.npc_traits: Dict[int, NPCTrait] = {}
        self.current_spotlight_npc: Optional[int] = None
        
        # Queen scheduling
        self.queen_schedule: Dict[str, str] = {}
        self.queen_current_state: Optional[QueenSchedule] = None
        
        # Episode management
        self.active_episodes: List[Episode] = []
        self.dormant_episodes: List[Episode] = []
        self.spotlight_episode: Optional[Episode] = None
        self.completed_episodes: List[str] = []
        
        # Relationship tracking
        self.relationship_tension: int = 0  # -100 to +100
        self.last_proximity_change: datetime = datetime.now()
        self.queen_presence_log: List[Dict[str, Any]] = []
        
        # Shared goals
        self.queen_goals = {
            "immediate": [],
            "ongoing": [],
            "relationship": []
        }
        
        # Memory system
        self.memory_shards: List[MemoryShard] = []
        
        # Network state
        self.network_awareness = 0
        self.information_layer = "public"
        self.player_rank = "outsider"
        
        # Bay Area state
        self.neighborhood_pulse = None
        self.current_time = datetime.now()
        
        # Dynamic mode control
        self.dynamic_mode = True
        
        # Agent caching
        self._last_intent = None
        self._last_threat_assessment = None
        self._pending_consequences = []
        self._last_thread_index = None
        
        # Tracking
        self._initialized = False
        self._last_simulation_run = None
        self._last_schedule_update = None
        self._last_thread_index_run = None
        self._last_visibility_decay = None
    
    async def initialize(self, dynamic: bool = True) -> Dict[str, Any]:
        """Initialize the unified sandbox experience"""
        self.dynamic_mode = dynamic
        
        try:
            story_exists = await self._check_story_exists()
            
            if not story_exists:
                logger.info(f"Initializing new Queen of Thorns sandbox for user {self.user_id}")
                
                ctx = CanonicalContext(user_id=self.user_id, conversation_id=self.conversation_id)
                
                # Initialize story with preset
                init_result = await QueenOfThornsStoryInitializer.initialize_story(
                    ctx, self.user_id, self.conversation_id, dynamic=dynamic
                )
                
                if init_result['status'] != 'success':
                    return init_result
                
                self.lilith_npc_id = init_result['main_npc_id']
                
                # Initialize NPC traits for all NPCs
                await self._initialize_npc_traits(init_result.get('support_npc_ids', []))
                
                # Initialize starter episodes
                await self._initialize_starter_episodes()
                
                logger.info(f"Sandbox initialized with Lilith ID: {self.lilith_npc_id}")
            else:
                # Load existing state
                await self._load_story_state()
                    
                logger.info(f"Resumed sandbox with {len(self.active_episodes)} active episodes")
            
            # Initialize components
            await self._initialize_components()
            
            # Set initial spotlight (Lilith by default)
            if not self.current_spotlight_npc:
                self.current_spotlight_npc = self.lilith_npc_id
            
            # Update Queen's schedule
            await self._update_queen_schedule()
            
            # Run initial simulations
            if self.dynamic_mode:
                await self._run_neighborhood_pulse()
                await self._run_visibility_decay()
            
            self._initialized = True
            
            spotlight_name = await self._get_npc_name(self.current_spotlight_npc)
            
            return {
                "status": "success",
                "message": "Unified sandbox ready",
                "new_story": not story_exists,
                "active_episodes": len(self.active_episodes),
                "current_spotlight": spotlight_name,
                "queen_status": "offscreen" if self.proximity == Proximity.OFFSCREEN else "available",
                "queen_location": self.queen_current_state.location if self.queen_current_state else "Unknown",
                "cast_size": len(self.npc_traits),
                "setting": "San Francisco Bay Area, 2025",
                "dynamic_mode": self.dynamic_mode
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize sandbox: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "message": "Failed to initialize unified sandbox"
            }

    async def _initialize_starter_episodes(self):
        """Create initial episodes for new game"""
        starter_episodes = [
            Episode(
                premise="The Rose Garden Café is unusually busy with tech workers today",
                stakes="Potential recruitment opportunities, but also risk of exposure",
                open_threads=["Assess the new faces", "Lily Chen seems nervous"],
                tags=["slice-social", "network-adjacent", "low-stakes"],
                location_relevant="Rose Garden Café",
                network_related=True,
                lilith_involvement="interested"
            ),
            Episode(
                premise="A new art installation in SOMA is drawing mysterious crowds at night",
                stakes="Could be supernatural, could be mundane - worth investigating",
                open_threads=["Visit after dark", "Ask locals about it"],
                tags=["slice-mystery", "neighborhood", "optional"],
                location_relevant="SOMA",
                network_related=False,
                lilith_involvement="curious"
            ),
            Episode(
                premise="Lilith mentioned wanting to check on 'an old friend' in the Marina",
                stakes="Personal glimpse into her past, building trust",
                open_threads=["Offer to accompany her", "Learn about this friend"],
                tags=["slice-personal", "relationship", "queen-driven"],
                location_relevant="Marina District",
                network_related=False,
                lilith_involvement="central"
            )
        ]
        
        self.active_episodes = starter_episodes
    
    async def _generate_solo_response(
        self,
        context: Dict[str, Any],
        mechanics_results: Dict[str, Any],
        retry: bool = False
    ) -> Dict[str, Any]:
        """Generate response when player is alone"""
        response = {
            'status': 'success',
            'timestamp': self.current_time.isoformat(),
            'proximity': Proximity.DIFFERENT_VENUE.value
        }
        
        # Get scene description with reflection lens
        if context.get('reflection'):
            scene_parts = [
                f"You're alone in {context.get('current_location')}.",
                random.choice(context['reflection']['ambient_details'])
            ]
            
            if context['reflection'].get('queen_echoes'):
                scene_parts.append(random.choice(context['reflection']['queen_echoes']))
            
            response['scene_description'] = " ".join(scene_parts)
        else:
            response['scene_description'] = f"You're alone in {context.get('current_location')}."
        
        # No dialogue with Lilith, but maybe NPC gossip
        if context.get('reflection', {}).get('npc_gossip'):
            response['overheard'] = context['reflection']['npc_gossip']
        
        # Add meanwhile glimpse if available
        if mechanics_results.get('meanwhile'):
            response['meanwhile_lilith'] = mechanics_results['meanwhile']
        
        # Episode content still relevant
        if self.spotlight_episode:
            response['episode_content'] = {
                'current': self.spotlight_episode.premise,
                'stakes': self.spotlight_episode.stakes,
                'progress': self.spotlight_episode.progress
            }
        
        return response

    def _get_base_schedule(self, hour: int) -> QueenSchedule:
        """Get base schedule for non-dynamic mode"""
        if 6 <= hour < 10:
            return QueenSchedule(
                location="Private Chambers",
                activity="Morning routine",
                mood="contemplative",
                available_for=["conversation", "planning"],
                next_transition="2 hours",
                join_preference="neutral"
            )
        elif 10 <= hour < 14:
            return QueenSchedule(
                location="Rose Garden Café", 
                activity="Observing and recruiting",
                mood="watchful",
                available_for=["public interaction"],
                next_transition="4 hours",
                join_preference="professional"
            )
        elif 14 <= hour < 18:
            return QueenSchedule(
                location="Network Business",
                activity="Operations",
                mood="focused",
                available_for=["accompany"],
                next_transition="4 hours",
                join_preference="neutral"
            )
        elif 18 <= hour < 22:
            return QueenSchedule(
                location="Velvet Sanctum",
                activity="Holding court",
                mood="dominant",
                available_for=["public performance"],
                next_transition="4 hours",
                join_preference="busy"
            )
        else:
            return QueenSchedule(
                location="Private Chambers",
                activity="Unwinding",
                mood="vulnerable",
                available_for=["intimacy"],
                next_transition="until morning",
                join_preference="selective"
            )

    def _check_queen_wants_to_leave(self) -> bool:
        """Check if Queen wants to leave for her own activities"""
        if not self.queen_current_state:
            return False
        
        # She has other priorities
        if self.queen_current_state.join_preference == "wants_to_leave":
            return True
        
        # Been together too long and tension is low
        hours_together = (self.current_time - self.last_proximity_change).seconds / 3600
        if hours_together > 4 and self.relationship_tension < 20:
            return random.random() < 0.3  # 30% chance she needs space
        
        return False
    
    def _can_invite_queen(self) -> bool:
        """Check if player can currently invite Queen"""
        if self.proximity == Proximity.TOGETHER:
            return False  # Already together
        
        if self.relationship_tension < -40:
            return False  # She won't come
        
        if self.queen_current_state and self.queen_current_state.join_preference == "busy":
            return False
        
        return True

    async def _handle_situation_results(self, situation: SituationWeave):
        """Process situation weaver results including dormant reactivation"""
        # Handle spotlight
        if situation.spotlight_episode:
            # Check if it's a dormant episode being reactivated
            dormant = next((ep for ep in self.dormant_episodes if ep.id == situation.spotlight_episode), None)
            if dormant:
                # Reactivate it!
                self.dormant_episodes.remove(dormant)
                dormant.last_active = self.current_time
                dormant.progress = max(dormant.progress, 10)  # Minimum progress on return
                self.active_episodes.append(dormant)
                self.spotlight_episode = dormant
                logger.info(f"Reactivated dormant episode: {dormant.premise}")
            else:
                # Regular active episode
                self.spotlight_episode = next(
                    (ep for ep in self.active_episodes if ep.id == situation.spotlight_episode),
                    None
                )
        
        # Handle new episode
        if situation.new_episode:
            self.active_episodes.append(situation.new_episode)
        
        # Handle updates
        for ep_id, updates in situation.episode_updates.items():
            episode = next((ep for ep in self.active_episodes if ep.id == ep_id), None)
            if episode:
                for key, value in updates.items():
                    if hasattr(episode, key):
                        setattr(episode, key, value)
    
    async def process_player_action_slice(
        self, 
        player_input: str, 
        current_location: str,
        scene_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a player action in the slice-of-life sandbox.
        Lilith's presence depends on current proximity state.
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Update current time
            self.current_time = datetime.now()
            
            # Update Queen's schedule if needed
            await self._update_queen_schedule()
            
            # Agent 1: Intent & Tone Classifier
            intent_result = None
            if self.dynamic_mode:
                intent_result = await self._classify_intent_with_proximity(player_input)
                self._last_intent = intent_result
            
            # Handle proximity changes based on intent
            await self._handle_proximity_with_cutoff(intent_result, None, current_location)
            
            # Build complete context with proximity-aware presence
            context = await self._build_slice_context(
                player_input, current_location, scene_context, intent_result
            )
            
            # Agent: Situation Weaver - what episode is relevant now?
            if self.dynamic_mode:
                situation = await self._weave_situation(context)
                await self._handle_situation_results(situation)
            else:
                # Pick a relevant episode based on location
                self._select_relevant_episode(current_location)
            
            # Check network mechanics if relevant
            network_results = {}
            if context.get('is_network_relevant') or self.spotlight_episode and self.spotlight_episode.network_related:
                network_results = await self._check_network_mechanics(context)
            
            # Check Lilith's mechanics only if she's present
            lilith_mechanics_results = {}
            if self.proximity in [Proximity.TOGETHER, Proximity.SAME_VENUE]:
                lilith_mechanics_results = await self._check_lilith_special_mechanics(context)
            
            # Combine mechanics
            all_mechanics = {**network_results, **lilith_mechanics_results}
            
            # Generate response based on proximity
            if self.proximity == Proximity.TOGETHER:
                response = await self._generate_duet_response(context, all_mechanics)
            elif self.proximity == Proximity.SAME_VENUE:
                response = await self._generate_venue_response(context, all_mechanics)
            else:  # DIFFERENT_VENUE or OFFSCREEN
                response = await self._generate_solo_response(context, all_mechanics)
            
            # Agent 8: Continuity check
            if self.dynamic_mode:
                continuity_ok = await self._check_slice_continuity(response, context)
                if not continuity_ok:
                    logger.warning("Continuity check failed, regenerating response")
                    # Regenerate based on proximity
                    if self.proximity == Proximity.TOGETHER:
                        response = await self._generate_duet_response(context, all_mechanics, retry=True)
                    elif self.proximity == Proximity.SAME_VENUE:
                        response = await self._generate_venue_response(context, all_mechanics, retry=True)
                    else:
                        response = await self._generate_solo_response(context, all_mechanics, retry=True)
            
            # Update story state and episode progress
            await self._update_slice_state(context, response)
            
            # Check for episode resolution
            await self._check_episode_lifecycle()
            
            # Add current slice status
            response['slice_status'] = {
                'proximity': self.proximity.value,
                'queen_with_you': self.proximity == Proximity.TOGETHER,
                'queen_nearby': self.proximity == Proximity.SAME_VENUE,
                'queen_accessible': self.proximity != Proximity.OFFSCREEN,
                'queen_location': self.queen_current_state.location if self.queen_current_state else "Unknown",
                'queen_activity': self.queen_current_state.activity if self.queen_current_state else "Unknown",
                'queen_mood': self.queen_current_state.mood if self.queen_current_state and self.proximity != Proximity.OFFSCREEN else "unknown",
                'active_episodes': len(self.active_episodes),
                'spotlight_episode': self.spotlight_episode.premise if self.spotlight_episode else None,
                'neighborhood_vibe': self.neighborhood_pulse.bay_mood if self.neighborhood_pulse else "normal",
                'time': self.current_time.strftime("%I:%M %p"),
                'can_invite_queen': self._can_invite_queen(),
                'can_retcon_queen': self._can_retcon_queen()
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing action: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "message": "Failed to process action"
            }
    
    # 3. Add new method for SAME_VENUE responses
    async def _generate_venue_response(
        self,
        context: Dict[str, Any],
        mechanics_results: Dict[str, Any],
        retry: bool = False
    ) -> Dict[str, Any]:
        """Generate response when Queen is in same venue but not together"""
        response = {
            'status': 'success',
            'timestamp': self.current_time.isoformat(),
            'proximity': Proximity.SAME_VENUE.value
        }
        
        # Get scene description with Queen visible but not interacting
        scene_parts = [
            f"You're in {context.get('current_location')}.",
            f"Lilith is here too, {self.queen_current_state.activity if self.queen_current_state else 'occupied with her own affairs'}."
        ]
        
        # She might acknowledge player but won't engage in full dialogue
        if context.get('player_intent', {}).get('intent') == 'invite':
            if self._check_queen_will_join(context.get('player_intent')):
                scene_parts.append("She notices your invitation and considers it.")
            else:
                scene_parts.append("She notices but seems too busy to join you right now.")
        
        response['scene_description'] = " ".join(scene_parts)
        
        # Limited interaction - maybe a gesture or brief acknowledgment
        if self.dynamic_mode and random.random() < 0.3:
            response['queen_acknowledgment'] = await self._get_queen_acknowledgment(context)
        
        # Episode content still relevant
        if self.spotlight_episode:
            response['episode_content'] = {
                'current': self.spotlight_episode.premise,
                'stakes': self.spotlight_episode.stakes,
                'progress': self.spotlight_episode.progress
            }
        
        return response

    
    async def _initialize_npc_traits(self, support_npc_ids: List[int]):
        """Initialize traits for all NPCs"""
        # Add Lilith
        if self.lilith_npc_id:
            self.npc_traits[self.lilith_npc_id] = NPCTrait(
                npc_id=self.lilith_npc_id,
                name="Lilith Ravencroft",
                affinity=50,
                visibility=100,  # Starts highly visible
                canonical_tags={"queen", "network", "main"}
            )
            self.npc_traits[self.lilith_npc_id].calculate_spotlight()
        
        # Add support NPCs
        for npc_id in support_npc_ids:
            npc_name = await self._get_npc_name(npc_id)
            self.npc_traits[npc_id] = NPCTrait(
                npc_id=npc_id,
                name=npc_name,
                affinity=0,
                visibility=20,  # Start with some visibility
                canonical_tags={"support", "network"}
            )
            self.npc_traits[npc_id].calculate_spotlight()
    
    async def _update_npc_affinity(self, intent_result: Optional[IntentClassification], player_input: str, location: str):
        """Update affinity for NPCs based on interaction"""
        # Determine which NPCs are present
        present_npcs = []
        
        # Spotlight NPC is always relevant
        if self.current_spotlight_npc:
            present_npcs.append(self.current_spotlight_npc)
        
        # Queen if not offscreen
        if self.lilith_npc_id and self.proximity != Proximity.OFFSCREEN:
            if self.proximity == Proximity.TOGETHER:
                present_npcs.append(self.lilith_npc_id)
        
        # NPCs in current episode
        if self.spotlight_episode:
            present_npcs.extend(self.spotlight_episode.cast)
        
        # Update affinity based on interaction tone
        if intent_result:
            affinity_change = 0
            if intent_result.emotion in ["curious", "friendly"]:
                affinity_change = 3
            elif intent_result.emotion in ["aggressive", "hostile"]:
                affinity_change = -5
            elif intent_result.threat_level > 7:
                affinity_change = -10
            
            # Apply to present NPCs
            for npc_id in set(present_npcs):
                if npc_id in self.npc_traits:
                    self.npc_traits[npc_id].affinity = max(-100, min(100, 
                        self.npc_traits[npc_id].affinity + affinity_change))
                    
                    # Update visibility
                    self.npc_traits[npc_id].visibility += 5
                    self.npc_traits[npc_id].last_seen = self.current_time
                    
                    # Special handling for Lilith
                    if npc_id == self.lilith_npc_id:
                        self.lilith_affinity = self.npc_traits[npc_id].affinity
    
    async def _handle_proximity_with_cutoff(self, intent_result: Optional[IntentClassification], stay_with_queen: Optional[bool], current_location: str):
        """Handle proximity changes including cutoff threshold"""
        # Check for cutoff first
        if self.lilith_affinity < self.cutoff_threshold or (intent_result and intent_result.proximity_intent == "banish"):
            if self.proximity != Proximity.OFFSCREEN:
                self.proximity = Proximity.OFFSCREEN
                self.add_memory_shard(
                    "Lilith formally steps out of the player's story.",
                    actors=["Lilith", "Player"],
                    tags=["relationship_end", "severed"],
                    importance=90
                )
                self.queen_presence_log.append({
                    "timestamp": self.current_time,
                    "event": "queen_offscreen",
                    "reason": "affinity_cutoff" if self.lilith_affinity < self.cutoff_threshold else "banished"
                })
                logger.info(f"Queen went offscreen: affinity={self.lilith_affinity}")
            return
        
        # Normal proximity handling if not offscreen
        if self.proximity != Proximity.OFFSCREEN:
            await self._handle_normal_proximity_change(intent_result, stay_with_queen, current_location)

    def _check_queen_will_join(self, intent_result: IntentClassification) -> bool:
        """Check if Lilith will accept invitation to join"""
        # Low relationship tension = she's less likely to join
        if self.relationship_tension < -40:
            return False
        
        # Check her schedule
        if self.queen_current_state:
            # She'll join if available or if she likes you enough
            if "accompany" in self.queen_current_state.available_for:
                return True
            elif self.relationship_tension > 60 and intent_result.politeness > 7:
                return True  # High relationship + polite request
        
        return False
    
    async def _handle_normal_proximity_change(self, intent_result: Optional[IntentClassification], stay_with_queen: Optional[bool], current_location: str):
        """Handle proximity when Queen is not offscreen"""
        # Similar to previous implementation but skip if offscreen
        if stay_with_queen is not None:
            new_proximity = Proximity.TOGETHER if stay_with_queen else Proximity.DIFFERENT_VENUE
        else:
            if intent_result and intent_result.proximity_intent:
                if intent_result.proximity_intent in ["leave", "go_alone"]:
                    new_proximity = Proximity.DIFFERENT_VENUE
                elif intent_result.proximity_intent in ["invite", "join"]:
                    if self._check_queen_will_join(intent_result):
                        new_proximity = Proximity.TOGETHER
                    else:
                        new_proximity = Proximity.SAME_VENUE
                else:
                    new_proximity = self.proximity
            else:
                new_proximity = self.proximity
        
        if new_proximity != self.proximity:
            self.proximity = new_proximity
            self.last_proximity_change = self.current_time
            
            # Update tension
            if new_proximity == Proximity.DIFFERENT_VENUE and self.proximity == Proximity.TOGETHER:
                if self.relationship_tension > 50:
                    self.relationship_tension -= 5
                elif self.relationship_tension < -20:
                    self.relationship_tension += 5
            elif new_proximity == Proximity.TOGETHER and self.proximity != Proximity.TOGETHER:
                self.relationship_tension += 3
    
    async def _run_focus_allocator(self) -> Optional[FocusAllocation]:
        """Agent: Focus Allocator - determine cast changes"""
        if not self.dynamic_mode:
            return None
        
        # Calculate all spotlight scores
        for npc in self.npc_traits.values():
            npc.calculate_spotlight()
        
        # Get current spotlight score
        current_score = 0
        if self.current_spotlight_npc and self.current_spotlight_npc in self.npc_traits:
            current_score = self.npc_traits[self.current_spotlight_npc].spotlight
        
        system_prompt = """You allocate narrative focus among the cast.
Promote NPCs with high spotlight scores (affinity*0.6 + visibility*0.4).
Demote NPCs with spotlight < 10.
Never demote Queen below SAME-VENUE unless affinity < -50."""

        # Prepare cast data
        cast_data = [
            {
                "id": npc.npc_id,
                "name": npc.name,
                "spotlight": npc.spotlight,
                "affinity": npc.affinity,
                "visibility": npc.visibility,
                "is_queen": npc.npc_id == self.lilith_npc_id,
                "is_current": npc.npc_id == self.current_spotlight_npc
            }
            for npc in sorted(self.npc_traits.values(), key=lambda x: x.spotlight, reverse=True)[:10]
        ]

        user_prompt = f"""Current cast spotlight scores:
{json.dumps(cast_data, indent=2)}

Current spotlight holder: {self.current_spotlight_npc}
Current score: {current_score}

Who should be promoted/demoted?"""

        return await self.gpt_service.call_with_validation(
            model=_FOCUS_MODEL,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=FocusAllocation,
            temperature=0.5
        )
    
    async def _handle_focus_changes(self, focus: Optional[FocusAllocation]):
        """Process focus allocator decisions"""
        if not focus:
            return
        
        # Handle promotion
        if focus.promote and focus.promote != self.current_spotlight_npc:
            # Validate with canon
            ctx = CanonicalContext(user_id=self.user_id, conversation_id=self.conversation_id)
            validation = await canon.validate_npc(ctx, focus.promote, "focus_change")
            
            if validation:
                old_spotlight = self.current_spotlight_npc
                self.current_spotlight_npc = focus.promote
                
                old_name = await self._get_npc_name(old_spotlight) if old_spotlight else "None"
                new_name = await self._get_npc_name(focus.promote)
                
                self.add_memory_shard(
                    f"Narrative focus shifts from {old_name} to {new_name}.",
                    actors=[old_name, new_name],
                    tags=["focus_change", "spotlight"],
                    importance=70
                )
                
                # Special handling if promoting Queen from offscreen
                if focus.promote == self.lilith_npc_id and self.proximity == Proximity.OFFSCREEN:
                    await self.attempt_queen_reentry("focus_allocator")
        
        # Handle new episode creation
        if focus.new_episode_for:
            await self._create_npc_episode(focus.new_episode_for)
    
    async def attempt_queen_reentry(self, catalyst: str) -> bool:
        """RetconGate - attempt to bring Queen back from offscreen"""
        if self.proximity != Proximity.OFFSCREEN:
            return False
        
        # Check gates
        gates = [
            self.lilith_affinity > -20,
            "queen_redeems" in catalyst.lower(),
            self._loose_thread_requires_her(),
            catalyst == "focus_allocator" and self.current_spotlight_npc == self.lilith_npc_id
        ]
        
        if any(gates):
            # Validate with canon
            ctx = CanonicalContext(user_id=self.user_id, conversation_id=self.conversation_id)
            if await canon.validate_npc(ctx, self.lilith_npc_id, "reentry"):
                self.proximity = Proximity.SAME_VENUE
                self.relationship_tension = max(self.relationship_tension, -10)
                
                if self.queen_current_state:
                    self.queen_current_state.join_preference = "cautious"
                
                self.add_memory_shard(
                    "Lilith returns to the story, cautious but present.",
                    actors=["Lilith", "Player"],
                    tags=["reunion", "retcon_gate"],
                    importance=80
                )
                
                logger.info(f"Queen re-entered via RetconGate: {catalyst}")
                return True
        
        return False
    
    def _loose_thread_requires_her(self) -> bool:
        """Check if any dormant threads need the Queen"""
        if self._last_thread_index and self._last_thread_index.queen_return_potential > 0.7:
            return True
        
        # Check dormant episodes
        for episode in self.dormant_episodes:
            if episode.lilith_involvement in ["central", "required"]:
                if episode.reactivation_score > 20:
                    return True
        
        return False
    
    def _can_retcon_queen(self) -> bool:
        """Check if Queen can be brought back"""
        return (
            self.proximity == Proximity.OFFSCREEN and
            self.lilith_affinity > -50 and
            len([e for e in self.active_episodes if e.lilith_involvement == "central"]) < 2
        )
    
    async def _weave_situation_with_spotlight(self, context: Dict[str, Any]) -> SituationWeave:
        """Situation Weaver that considers current spotlight NPC"""
        system_prompt = """You weave situations favoring the current spotlight NPC.
Active episodes with that NPC should be prioritized.
Can create new episodes featuring the spotlight character."""

        # Add spotlight bias
        spotlight_name = await self._get_npc_name(self.current_spotlight_npc)
        
        active_summary = [
            {
                "id": ep.id,
                "premise": ep.premise,
                "progress": ep.progress,
                "features_spotlight": self.current_spotlight_npc in ep.cast
            }
            for ep in self.active_episodes[:10]
        ]

        user_prompt = f"""Current situation:
Player action: "{context.get('player_input')}"
Spotlight NPC: {spotlight_name} (ID: {self.current_spotlight_npc})
Active episodes: {json.dumps(active_summary, default=str)}

Weave a situation that features or relates to {spotlight_name}."""

        result = await self.gpt_service.call_with_validation(
            model=_WEAVER_MODEL,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=SituationWeave,
            temperature=0.7
        )
        
        result.spotlight_npc_bias = self.current_spotlight_npc
        return result
    
    async def _check_unified_mechanics(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check mechanics for all present NPCs"""
        all_mechanics = {}
        
        # Network mechanics if relevant
        if context.get('is_network_relevant'):
            network_results = await self._check_network_mechanics(context)
            all_mechanics.update(network_results)
        
        # Queen mechanics if not offscreen
        if self.proximity not in [Proximity.OFFSCREEN, Proximity.DIFFERENT_VENUE]:
            if self.lilith_mechanics:
                lilith_results = await self._check_lilith_special_mechanics(context)
                all_mechanics.update(lilith_results)
        elif self.proximity == Proximity.DIFFERENT_VENUE and self.dynamic_mode:
            # Get meanwhile glimpse
            meanwhile = await self._get_lilith_meanwhile(context)
            if meanwhile.meanwhile_glimpse and not meanwhile.is_offscreen:
                all_mechanics['meanwhile'] = {
                    'glimpse': meanwhile.meanwhile_glimpse,
                    'feeling': meanwhile.separation_feeling
                }
        
        # Spotlight NPC mechanics (simplified for now)
        if self.current_spotlight_npc != self.lilith_npc_id:
            all_mechanics['spotlight_npc'] = {
                'present': True,
                'id': self.current_spotlight_npc,
                'name': await self._get_npc_name(self.current_spotlight_npc)
            }
        
        return all_mechanics
    
    async def _check_unified_continuity(self, response: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Enhanced continuity check including offscreen violations"""
        response_text = json.dumps(response.get('dialogue', {}))
        
        system_prompt = """You check continuity for a dynamic cast story.
Key rules:
- Queen CANNOT speak if proximity is "offscreen"
- NPCs can only speak if they're in the scene
- Canon must be preserved
- Episodes must maintain internal logic"""

        speakers = []
        if response.get('dialogue', {}).get('script'):
            speakers = [line.get('speaker', '') for line in response['dialogue']['script']]

        user_prompt = f"""Check this response:
Response: {response_text}
Speakers in dialogue: {speakers}
Queen proximity: {context.get('proximity')}
Spotlight NPC: {await self._get_npc_name(self.current_spotlight_npc)}

Flag any violations."""

        check_result = await self.gpt_service.call_with_validation(
            model=_CONTINUITY_MODEL,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=ContinuityCheck,
            temperature=0.2
        )
        
        # Additional check for Queen offscreen violation
        if self.proximity == Proximity.OFFSCREEN and "Lilith" in speakers:
            check_result.queen_offscreen_violation = True
            check_result.ok = False
            check_result.issues.append("Queen cannot speak while offscreen")
        
        return check_result.ok
    
    async def _generate_duet_response(self, context: Dict[str, Any], mechanics_results: Dict[str, Any], retry: bool = False) -> Dict[str, Any]:
        """Generate response with multiple cast members"""
        response = {
            'status': 'success',
            'timestamp': self.current_time.isoformat(),
            'proximity': self.proximity.value
        }
        
        # Determine who's present
        cast_present = []
        if self.proximity == Proximity.TOGETHER:
            cast_present.append("Lilith")
        
        spotlight_name = await self._get_npc_name(self.current_spotlight_npc)
        if spotlight_name and spotlight_name != "Lilith":
            cast_present.append(spotlight_name)
        
        # Generate dialogue with present cast
        if self.dynamic_mode and cast_present:
            dialogue_script = await self._write_cast_dialogue(context, mechanics_results, cast_present)
            response['dialogue'] = {
                'script': dialogue_script.script,
                'subtext': dialogue_script.subtext,
                'power_dynamics': dialogue_script.power_dynamics,
                'cast_speaking': dialogue_script.cast_present
            }
        else:
            # Fallback
            response['dialogue'] = {
                'script': [{"speaker": cast_present[0] if cast_present else "Narrator", "line": "..."}],
                'cast_speaking': cast_present
            }
        
        # Scene description
        response['scene_description'] = await self._generate_cast_scene_description(context, cast_present)
        
        # Episode content
        if self.spotlight_episode:
            response['episode_content'] = {
                'current': self.spotlight_episode.premise,
                'stakes': self.spotlight_episode.stakes,
                'progress': self.spotlight_episode.progress,
                'cast': [await self._get_npc_name(npc_id) for npc_id in self.spotlight_episode.cast]
            }
        
        return response

    async def _generate_cast_scene_description(
        self, context: Dict[str, Any], cast_present: List[str]
    ) -> str:
        """Generate scene description with cast awareness"""
        location = context.get('current_location', 'somewhere')
        base_parts = [f"You're in {location}."]
        
        # Describe who's actually present based on proximity
        if "Lilith" in cast_present and self.proximity == Proximity.TOGETHER:
            base_parts.append(f"Lilith is with you, {self.queen_current_state.activity if self.queen_current_state else 'her attention on you'}.")
        elif self.proximity == Proximity.SAME_VENUE:
            base_parts.append(f"Lilith is here but occupied with {self.queen_current_state.activity if self.queen_current_state else 'her own affairs'}.")
        
        # Add other cast members
        other_cast = [c for c in cast_present if c != "Lilith"]
        if other_cast:
            base_parts.append(f"{', '.join(other_cast)} {'is' if len(other_cast) == 1 else 'are'} here.")
        
        base_desc = " ".join(base_parts)
        
        # Enhance with scene details
        enhanced = await self.text_generator.enhance_scene_description(
            base_desc,
            context.get('scene_type', 'dynamic'),
            {
                'cast_size': len(cast_present),
                'proximity': self.proximity.value,
                'emotional_tone': context.get('lilith_emotion', 'neutral')
            }
        )
        
        return enhanced
    
    async def _write_cast_dialogue(self, context: Dict[str, Any], mechanics: Dict[str, Any], cast_present: List[str]) -> DialogueScript:
        """Write dialogue for present cast members"""
        system_prompt = """You write natural dialogue for multiple characters.
Each character should have distinct voice and agenda.
Priority goes to the designated priority speaker."""

        user_prompt = f"""Write dialogue for this scene:
Player said: "{context.get('player_input')}"
Cast present: {cast_present}
Spotlight character: {await self._get_npc_name(self.current_spotlight_npc)}
Special events: {list(mechanics.keys()) if mechanics else 'None'}

Create 2-4 lines with priority to spotlight character."""

        result = await self.gpt_service.call_with_validation(
            model=_DIALOGUE_MODEL,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=DialogueScript,
            temperature=0.7
        )
        
        result.cast_present = cast_present
        result.priority_speaker = await self._get_npc_name(self.current_spotlight_npc) or cast_present[0]
        
        return result
    
    async def _run_visibility_decay(self):
        """Nightly visibility decay for all NPCs"""
        if not self.dynamic_mode:
            return
        
        # Check if a day has passed
        if self._last_visibility_decay:
            if (self.current_time - self._last_visibility_decay).days < 1:
                return
        
        self._last_visibility_decay = self.current_time
        
        for npc in self.npc_traits.values():
            # Decay visibility
            npc.visibility = int(npc.visibility * 0.75)
            
            # Extra decay if offscreen
            if npc.npc_id == self.lilith_npc_id and self.proximity == Proximity.OFFSCREEN:
                npc.visibility = int(npc.visibility * 0.5)
            
            # Recalculate spotlight
            npc.calculate_spotlight()
        
        logger.info("Visibility decay applied to all NPCs")
    
    async def run_simulation_daemon(self):
        """Enhanced simulation with focus allocation"""
        if not self.dynamic_mode:
            return
        
        if self._last_simulation_run:
            time_since = (datetime.now() - self._last_simulation_run).total_seconds() / 60
            if time_since < 30:
                return
        
        self._last_simulation_run = datetime.now()
        
        # Update Queen's schedule (even if offscreen)
        self.queen_current_state = await self._generate_queen_schedule()
        
        # Update neighborhood
        await self._run_neighborhood_pulse()
        
        # Progress episodes
        for episode in self.active_episodes:
            if random.random() < 0.1:
                episode.progress = min(100, episode.progress + random.randint(5, 15))
        
        # Dormant reactivation
        for episode in self.dormant_episodes:
            episode.reactivation_score += random.randint(1, 5)
            
            if episode.reactivation_score > 15 and random.random() < 0.3:
                episode.last_active = self.current_time
                self.active_episodes.append(episode)
                self.dormant_episodes.remove(episode)
                logger.info(f"Dormant episode reactivated: {episode.premise}")
        
        # Run focus allocator
        focus_changes = await self._run_focus_allocator()
        if focus_changes:
            await self._handle_focus_changes(focus_changes)
        
        # Run thread indexer
        await self._run_loose_thread_indexer()
        
        # Visibility decay
        await self._run_visibility_decay()
        
        # Cleanup
        await self._check_episode_lifecycle()
    
    async def _generate_queen_schedule(self) -> QueenSchedule:
        """Generate Queen's schedule even when offscreen"""
        current_hour = self.current_time.hour
        day_of_week = self.current_time.strftime("%A")
        
        system_prompt = """You schedule Lilith's activities.
If she's offscreen, she continues her life without the player.
Consider her emotional state and what she'd realistically be doing."""

        user_prompt = f"""Generate Lilith's current activity:
Time: {day_of_week}, {current_hour}:00
Proximity: {self.proximity.value}
Affinity with player: {self.lilith_affinity}
Last interaction: {(self.current_time - self.last_proximity_change).days} days ago

What is she doing?"""

        result = await self.gpt_service.call_with_validation(
            model=_SCHEDULER_MODEL,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=QueenSchedule,
            temperature=0.6
        )
        
        # Override availability if offscreen
        if self.proximity == Proximity.OFFSCREEN:
            result.available_for = []
            result.join_preference = "unavailable"
            result.offscreen_status = "Living her own life"
        
        return result

    async def _build_unified_context(
        self,
        player_input: str,
        current_location: str,
        scene_context: Optional[Dict[str, Any]],
        intent_result: Optional[IntentClassification] = None
    ) -> Dict[str, Any]:
        """Build unified context with full proximity awareness"""
        
        # Start with base proximity context
        context = await self._build_base_proximity_context(
            player_input, current_location, scene_context
        )
        
        # Add proximity-specific details
        context['proximity'] = self.proximity.value
        context['queen_present'] = self.proximity == Proximity.TOGETHER
        context['queen_nearby'] = self.proximity == Proximity.SAME_VENUE
        context['queen_accessible'] = self.proximity != Proximity.OFFSCREEN
        context['queen_in_story'] = self.proximity != Proximity.OFFSCREEN
        
        # Add intent if available
        if intent_result:
            context['player_intent'] = intent_result.dict()
            context['is_threatening'] = intent_result.threat_level > 6
            context['is_network_relevant'] = intent_result.network_relevance > 5
            context['proximity_intent'] = intent_result.proximity_intent
        
        # Determine current focus NPC info
        if self.current_spotlight_npc:
            context['spotlight_npc'] = {
                'id': self.current_spotlight_npc,
                'name': await self._get_npc_name(self.current_spotlight_npc),
                'is_queen': self.current_spotlight_npc == self.lilith_npc_id
            }
        
        # Add episode context filtered by proximity appropriateness
        appropriate_episodes = []
        for ep in self.active_episodes:
            if self.proximity == Proximity.OFFSCREEN and ep.lilith_involvement == "central":
                continue  # Skip Lilith-centric episodes when she's gone
            appropriate_episodes.append(ep.premise)
        context['appropriate_episodes'] = appropriate_episodes
        
        # Reflection lens for when apart
        if self.proximity in [Proximity.DIFFERENT_VENUE, Proximity.OFFSCREEN] and self.dynamic_mode:
            context['reflection'] = await self._get_reflection_lens(current_location)
        
        return context
        
    async def _get_lilith_meanwhile(self, context: Dict[str, Any]) -> LilithInnerVoice:
        """Get Lilith's inner voice/meanwhile based on proximity"""
        
        if self.proximity == Proximity.OFFSCREEN:
            # She's completely out of the story
            return LilithInnerVoice(
                mask_slip=False,
                mask_integrity=100,
                emotional_state="absent",
                vulnerability_level=0,
                is_offscreen=True,
                meanwhile_glimpse=None  # No glimpses when severed
            )
        elif self.proximity == Proximity.DIFFERENT_VENUE:
            # Get a glimpse of what she's doing elsewhere
            system_prompt = """You provide brief glimpses of what Lilith is doing elsewhere.
    She has her own life, goals, and activities. Show her as a full person.
    Keep it mysterious but grounded."""
    
            user_prompt = f"""Lilith is at {self.queen_current_state.location if self.queen_current_state else 'her own location'}.
    She's {self.queen_current_state.activity if self.queen_current_state else 'pursuing her own agenda'}.
    Relationship tension: {self.relationship_tension}
    Time since together: {(self.current_time - self.last_proximity_change).seconds / 3600:.1f} hours
    
    Give a brief, evocative glimpse of her activities."""
    
            result = await self.gpt_service.call_with_validation(
                model=_INNER_VOICE_MODEL,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=LilithInnerVoice,
                temperature=0.7
            )
            
            # Override to ensure it's marked as meanwhile
            result.is_offscreen = False
            result.mask_slip = False
            result.vulnerability_level = 0
            return result
        elif self.proximity == Proximity.SAME_VENUE:
            # She's aware of player but focused on her own things
            return LilithInnerVoice(
                mask_slip=False,
                mask_integrity=90,
                emotional_state="aware but occupied",
                vulnerability_level=1,
                separation_feeling="maintaining appropriate distance",
                meanwhile_glimpse=f"She {self.queen_current_state.activity if self.queen_current_state else 'continues her business'}, occasionally aware of your presence"
            )
        else:  # TOGETHER
            # Full inner voice when together
            return await self._get_lilith_inner_voice(context)

    
    def _get_top_cast_scores(self) -> List[Dict[str, Any]]:
        """Get top 5 NPCs by spotlight score"""
        sorted_npcs = sorted(self.npc_traits.values(), key=lambda x: x.spotlight, reverse=True)[:5]
        return [
            {
                "name": npc.name,
                "spotlight": round(npc.spotlight, 1),
                "affinity": npc.affinity,
                "is_current": npc.npc_id == self.current_spotlight_npc
            }
            for npc in sorted_npcs
        ]
    
    async def _get_npc_name(self, npc_id: Optional[int]) -> str:
        """Get NPC name from ID"""
        if not npc_id:
            return "Unknown"
        
        if npc_id in self.npc_traits:
            return self.npc_traits[npc_id].name
        
        # Fallback to database
        async with get_db_connection_context() as conn:
            name = await conn.fetchval(
                "SELECT npc_name FROM NPCStats WHERE npc_id = $1",
                npc_id
            )
            return name or "Unknown"
    
    async def _create_npc_episode(self, npc_id: int):
        """Create a new episode featuring specific NPC"""
        npc_name = await self._get_npc_name(npc_id)
        
        new_episode = Episode(
            premise=f"A situation develops involving {npc_name}",
            stakes=f"Your relationship with {npc_name} could change",
            open_threads=[f"Understand {npc_name}'s needs", "Make a choice"],
            tags=["slice-character", "relationship", f"npc-{npc_id}"],
            cast=[npc_id],
            lilith_involvement="none" if npc_id != self.lilith_npc_id else "central"
        )
        
        self.active_episodes.append(new_episode)
        logger.info(f"Created episode for {npc_name}")
    
    async def _check_story_exists(self) -> bool:
        """Check if story already exists for this user/conversation"""
        async with get_db_connection_context() as conn:
            exists = await conn.fetchval(
                """
                SELECT EXISTS(
                    SELECT 1 FROM story_states
                    WHERE user_id = $1 AND conversation_id = $2 AND story_id = $3
                )
                """,
                self.user_id, self.conversation_id, self.story_id
            )
            return exists
    
    async def _load_story_state(self):
        """Load existing story state from database"""
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow(
                """
                SELECT story_flags, 
                       story_flags->>'lilith_npc_id' as lilith_id,
                       story_flags->>'network_awareness' as awareness,
                       story_flags->>'information_layer' as info_layer,
                       story_flags->>'player_rank' as rank,
                       story_flags->>'active_episodes' as episodes,
                       story_flags->>'queen_goals' as goals,
                       story_flags->>'queen_schedule' as schedule
                FROM story_states
                WHERE user_id = $1 AND conversation_id = $2 AND story_id = $3
                """,
                self.user_id, self.conversation_id, self.story_id
            )
            
            if row:
                self.lilith_npc_id = int(row['lilith_id']) if row['lilith_id'] else None
                self.network_awareness = int(row['awareness'] or 0)
                self.information_layer = row['info_layer'] or 'public'
                self.player_rank = row['rank'] or 'outsider'
                
                # Load episodes
                if row['episodes']:
                    episodes_data = json.loads(row['episodes'])
                    self.active_episodes = [Episode(**ep) for ep in episodes_data]
                
                # Load Queen goals
                if row['goals']:
                    self.queen_goals = json.loads(row['goals'])
                
                # Load Queen schedule
                if row['schedule']:
                    self.queen_schedule = json.loads(row['schedule'])
    
    async def _initialize_components(self):
        """Initialize all component handlers"""
        # Text generator
        self.text_generator = ThornsEnhancedTextGenerator(
            self.user_id, self.conversation_id, self.story_id
        )
        await self.text_generator.initialize()
        
        # Lilith mechanics handler
        if self.lilith_npc_id:
            self.lilith_mechanics = LilithMechanicsHandler(
                self.user_id, self.conversation_id, self.lilith_npc_id
            )
            await self.lilith_mechanics.initialize()
        
        # Memory system
        self.memory_system = await MemorySystem.get_instance(
            self.user_id, self.conversation_id
        )

    async def _weave_situation_with_dormant(self, context: Dict[str, Any]) -> SituationWeave:
        """Enhanced situation weaver that considers dormant episodes"""
        system_prompt = """You weave situations from active AND dormant episodes.
Dormant episodes with high reactivation scores might return with new urgency.
Consider the player's current proximity to Lilith when choosing episodes."""

        active_summary = [
            {"id": ep.id, "premise": ep.premise, "progress": ep.progress}
            for ep in self.active_episodes[:10]
        ]
        
        # Include high-score dormant episodes
        dormant_candidates = [
            {
                "id": ep.id, 
                "premise": ep.premise,
                "dormant_since": str(ep.dormant_since),
                "reactivation_score": ep.reactivation_score,
                "unresolved": ep.unresolved_stakes
            }
            for ep in sorted(self.dormant_episodes, key=lambda e: e.reactivation_score, reverse=True)[:5]
        ]

        user_prompt = f"""Current situation:
Player action: "{context.get('player_input')}"
Proximity to Queen: {context.get('proximity')}
Active episodes: {json.dumps(active_summary, default=str)}
Dormant candidates: {json.dumps(dormant_candidates, default=str)}

Should we spotlight an active episode, resurrect a dormant one, or create new?"""

        return await self.gpt_service.call_with_validation(
            model=_WEAVER_MODEL,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=SituationWeave,
            temperature=0.7
        )

    async def _classify_intent_with_proximity(self, player_input: str) -> IntentClassification:
        """Enhanced intent classification that detects proximity desires"""
        system_prompt = """You classify player intent in a slice-of-life story where they can be with or apart from Lilith.
Detect if they want to: invite her, leave her, join her, or neither.
Key phrases: "come with me", "let's go", "I'll go alone", "see you later", "join me", etc."""

        user_prompt = f"""Classify this input including proximity intent:
"{player_input}"

Current proximity: {self.proximity.value}
Queen's availability: {self.queen_current_state.available_for if self.queen_current_state else 'unknown'}"""

        return await self.gpt_service.call_with_validation(
            model=_INTENT_MODEL,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=IntentClassification,
            temperature=0.3,
            use_cache=True
        )
    
    
    async def _update_queen_schedule(self, force: bool = False):
        """Update Lilith's schedule and current activity"""
        # Only update hourly or when forced
        if not force and self._last_schedule_update:
            if (datetime.now() - self._last_schedule_update).seconds < 3600:
                return
        
        self._last_schedule_update = datetime.now()
        
        if self.dynamic_mode:
            self.queen_current_state = await self._generate_queen_schedule()
        else:
            # Default schedule
            hour = self.current_time.hour
            if 6 <= hour < 10:
                self.queen_current_state = QueenSchedule(
                    location="Private Chambers",
                    activity="Morning routine and correspondence",
                    mood="contemplative",
                    available_for=["intimate conversation", "planning"],
                    next_transition="2 hours"
                )
            elif 10 <= hour < 14:
                self.queen_current_state = QueenSchedule(
                    location="Rose Garden Café",
                    activity="Observing and recruiting",
                    mood="watchful",
                    available_for=["public interaction", "coded conversation"],
                    next_transition="4 hours"
                )
            elif 14 <= hour < 18:
                self.queen_current_state = QueenSchedule(
                    location="Various (Network Business)",
                    activity="Checking on operations",
                    mood="focused",
                    available_for=["network talk", "accompanying"],
                    next_transition="4 hours"
                )
            elif 18 <= hour < 22:
                self.queen_current_state = QueenSchedule(
                    location="Velvet Sanctum",
                    activity="Holding court",
                    mood="dominant",
                    available_for=["public performance", "private sessions"],
                    next_transition="4 hours"
                )
            else:
                self.queen_current_state = QueenSchedule(
                    location="Private Chambers",
                    activity="Unwinding",
                    mood="vulnerable",
                    available_for=["deep conversation", "intimacy"],
                    next_transition="until morning"
                )
    
    async def _generate_queen_schedule(self) -> QueenSchedule:
        """Agent: Queen Scheduler - decide where Lilith is and what she's doing"""
        current_hour = self.current_time.hour
        day_of_week = self.current_time.strftime("%A")
        
        system_prompt = """You schedule Lilith Ravencroft's daily activities.
She balances: running the Velvet Sanctum, network operations, personal time, and relationship with player.
Consider time of day, her various roles, and recent events.
She's a complex woman with many responsibilities but also human needs."""

        user_prompt = f"""Generate Lilith's current activity:
Time: {day_of_week}, {current_hour}:00
Trust with player: {self.queen_goals.get('relationship', ['building trust'])}
Recent network activity: {self._last_threat_assessment.kozlov_activity if self._last_threat_assessment else 'moderate'}
Active episodes: {[ep.premise for ep in self.active_episodes if ep.lilith_involvement != 'none'][:3]}

Where is she and what is she doing?"""

        return await self.gpt_service.call_with_validation(
            model=_SCHEDULER_MODEL,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=QueenSchedule,
            temperature=0.6
        )

    async def handle_retcon_attempt(self, catalyst_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle player attempts to bring Queen back via RetconGate"""
        if self.proximity != Proximity.OFFSCREEN:
            return {
                'success': False,
                'reason': 'Queen is not offscreen'
            }
        
        # Check various catalysts
        catalysts = {
            'sincere_apology': context.get('player_input', '').lower().count('sorry') > 0 and self.lilith_affinity > -40,
            'network_crisis': context.get('is_network_relevant') and any(e.network_related and e.lilith_involvement == "required" for e in self.active_episodes),
            'emotional_plea': any(word in context.get('player_input', '').lower() for word in ['need you', 'come back', 'miss you']),
            'time_passed': (self.current_time - self.last_proximity_change).days > 7,
            'high_thread_score': self._last_thread_index and self._last_thread_index.queen_return_potential > 0.7
        }
        
        if catalysts.get(catalyst_type, False):
            success = await self.attempt_queen_reentry(catalyst_type)
            
            if success:
                return {
                    'success': True,
                    'new_proximity': self.proximity.value,
                    'queen_response': await self._generate_return_response(catalyst_type),
                    'affinity_change': 5
                }
        
        return {
            'success': False,
            'reason': f'Conditions not met for {catalyst_type}',
            'hint': 'Perhaps time, sincerity, or necessity will open the way...'
        }

    async def _generate_return_response(self, catalyst: str) -> str:
        """Generate Lilith's response when returning via RetconGate"""
        responses = {
            'sincere_apology': "Your words... they reached me. I'm not ready to be close again, but I'm here.",
            'network_crisis': "The network needs me. And perhaps... so do you. Let's handle this first.",
            'emotional_plea': "I heard you calling. I'm not sure this is wise, but...",
            'time_passed': "Time has a way of softening edges. Hello again.",
            'high_thread_score': "There are threads between us that refuse to be cut. Very well."
        }
        
        return responses.get(catalyst, "I'm here. Let's see where this goes.")
    
    async def _run_neighborhood_pulse(self):
        """Generate neighborhood events considering Queen's availability"""
        if not self.dynamic_mode:
            return
        
        system_prompt = """You generate daily events in San Francisco Bay Area, 2025.
    Mix of: tech culture, underground scenes, supernatural hints, normal city life.
    Consider whether Lilith is available to participate in events."""
    
        queen_status = "offscreen" if self.proximity == Proximity.OFFSCREEN else "available"
        
        user_prompt = f"""Generate today's neighborhood events:
    Day: {self.current_time.strftime('%A')}
    Season: {self._get_season()}
    Queen status: {queen_status}
    Recent player locations: {self._get_recent_locations()}
    
    Create 2-4 interesting happenings. Mark which ones could involve Lilith."""
    
        self.neighborhood_pulse = await self.gpt_service.call_with_validation(
            model=_PULSE_MODEL,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=NeighborhoodPulse,
            temperature=0.8
        )
        
        # Convert events to episodes with appropriate Lilith involvement
        for event in self.neighborhood_pulse.local_events:
            if random.random() < 0.3:  # 30% chance each event becomes available
                # Determine Lilith's potential involvement based on proximity
                if self.proximity == Proximity.OFFSCREEN:
                    lilith_involvement = "none"
                elif self.proximity == Proximity.DIFFERENT_VENUE:
                    lilith_involvement = "optional" if random.random() < 0.3 else "none"
                else:
                    lilith_involvement = "optional" if random.random() < 0.5 else "interested"
                
                new_episode = Episode(
                    premise=event.hook,
                    stakes="A slice of Bay Area life - could lead somewhere interesting",
                    open_threads=[f"Check out {event.place}", "See what this is about"],
                    tags=["slice-neighborhood", f"type-{event.type}", "optional"],
                    location_relevant=event.place,
                    network_related=False,
                    lilith_involvement=lilith_involvement
                )
                self.active_episodes.append(new_episode)

    


    async def _build_base_proximity_context(
        self,
        player_input: str,
        current_location: str,
        scene_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build base context"""
        # Get Lilith's data even if she's not present
        lilith_data = None
        if self.lilith_npc_id:
            async with get_db_connection_context() as conn:
                lilith_row = await conn.fetchrow(
                    """
                    SELECT npc_name, trust, dominance, current_mask,
                           network_role, network_knowledge, operational_knowledge
                    FROM NPCStats
                    WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
                    """,
                    self.user_id, self.conversation_id, self.lilith_npc_id
                )
                
                if lilith_row:
                    lilith_data = dict(lilith_row)
                    for field in ['network_knowledge', 'operational_knowledge']:
                        if lilith_data.get(field):
                            lilith_data[field] = json.loads(lilith_data[field])
        
        # Get location details
        location_data = await self._get_location_data(current_location)
        
        # Build context
        context = {
            'player_input': player_input,
            'player_action': player_input,
            'current_location': current_location,
            'location_data': location_data,
            'queen_data': lilith_data,
            'lilith_data': lilith_data,
            'queen_schedule': self.queen_current_state.dict() if self.queen_current_state else {},
            'relationship_tension': self.relationship_tension,
            'network_awareness': self.network_awareness,
            'information_layer': self.information_layer,
            'player_rank': self.player_rank,
            'active_episodes': [ep.premise for ep in self.active_episodes],
            'dormant_count': len(self.dormant_episodes),
            'spotlight_episode': self.spotlight_episode.dict() if self.spotlight_episode else None,
            'timestamp': self.current_time,
            'time_of_day': self._get_time_period()
        }
        
        if scene_context:
            context.update(scene_context)
        
        return context
    
    async def _build_slice_context(
        self, 
        player_input: str,
        current_location: str,
        scene_context: Optional[Dict[str, Any]],
        intent_result: Optional[IntentClassification] = None
    ) -> Dict[str, Any]:
        """Build context with proximity-aware Queen presence"""
        
        # Get Lilith's current state
        lilith_data = None
        if self.lilith_npc_id and self.proximity != Proximity.OFFSCREEN:
            async with get_db_connection_context() as conn:
                lilith_row = await conn.fetchrow(
                    """
                    SELECT npc_name, trust, dominance, current_mask,
                           network_role, network_knowledge, operational_knowledge
                    FROM NPCStats
                    WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
                    """,
                    self.user_id, self.conversation_id, self.lilith_npc_id
                )
                
                if lilith_row:
                    lilith_data = dict(lilith_row)
                    # Parse JSON fields
                    for field in ['network_knowledge', 'operational_knowledge']:
                        if lilith_data.get(field):
                            lilith_data[field] = json.loads(lilith_data[field])
        
        # Get location details
        location_data = await self._get_location_data(current_location)
        
        # Determine Queen's presence based on proximity
        queen_present = self.proximity == Proximity.TOGETHER
        queen_in_venue = self.proximity in [Proximity.TOGETHER, Proximity.SAME_VENUE]
        queen_accessible = self.proximity != Proximity.OFFSCREEN
        
        # Build context
        context = {
            'player_input': player_input,
            'player_action': player_input,
            'current_location': current_location,
            'location_data': location_data,
            'queen_data': lilith_data if queen_accessible else None,
            'lilith_data': lilith_data if queen_accessible else None,
            'queen_present': queen_present,  # Only true if TOGETHER
            'queen_in_scene': queen_present,  # Same as above
            'queen_in_venue': queen_in_venue,  # True for TOGETHER or SAME_VENUE
            'queen_accessible': queen_accessible,  # False only for OFFSCREEN
            'proximity': self.proximity.value,
            'queen_schedule': self.queen_current_state.dict() if self.queen_current_state and queen_accessible else {},
            'queen_available_for': self.queen_current_state.available_for if self.queen_current_state and queen_accessible else [],
            'network_awareness': self.network_awareness,
            'information_layer': self.information_layer,
            'player_rank': self.player_rank,
            'active_episodes': [ep.premise for ep in self.active_episodes],
            'spotlight_episode': self.spotlight_episode.dict() if self.spotlight_episode else None,
            'timestamp': self.current_time,
            'time_of_day': self._get_time_period()
        }
        
        # Merge with provided scene context
        if scene_context:
            context.update(scene_context)
        
        # Add intent classification if available
        if intent_result:
            context['player_intent'] = intent_result.dict()
            context['is_threatening'] = intent_result.threat_level > 6
            context['is_network_relevant'] = intent_result.network_relevance > 5
        
        # Determine derived context
        context['is_network_business'] = self._is_network_business(context)
        context['is_private'] = self._is_private_location(current_location)
        
        # Only determine Lilith's emotion if she's accessible
        if queen_accessible:
            context['lilith_emotion'] = self._determine_lilith_emotion(context)
        else:
            context['lilith_emotion'] = 'absent'
        
        # Determine scene type from episodes and mood
        context['scene_type'] = self._determine_slice_scene_type(context)
        
        return context

    async def _weave_situation(self, context: Dict[str, Any]) -> SituationWeave:
        """Situation Weaver agent - considers Lilith's availability"""
        system_prompt = """You're a situation weaver for a dynamic story.
    Consider which episodes make sense given character proximity.
    If Lilith is offscreen, avoid episodes requiring her central involvement.
    If she's in same venue, episodes can reference her but not require direct interaction."""
    
        # Filter episodes by appropriateness given proximity
        active_episodes_summary = []
        for ep in self.active_episodes[:10]:
            ep_data = {
                "id": ep.id,
                "premise": ep.premise,
                "progress": ep.progress,
                "location": ep.location_relevant,
                "lilith_involvement": ep.lilith_involvement
            }
            
            # Flag if episode is appropriate for current proximity
            if self.proximity == Proximity.OFFSCREEN:
                ep_data["appropriate"] = ep.lilith_involvement in ["none", "optional"]
            elif self.proximity == Proximity.DIFFERENT_VENUE:
                ep_data["appropriate"] = ep.lilith_involvement != "central"
            else:
                ep_data["appropriate"] = True
                
            active_episodes_summary.append(ep_data)
    
        user_prompt = f"""Current situation:
    Player action: "{context.get('player_input')}"
    Location: {context.get('current_location')}
    Lilith proximity: {self.proximity.value}
    Queen is: {context.get('queen_schedule', {}).get('activity', 'elsewhere') if self.proximity != Proximity.OFFSCREEN else 'gone from your story'}
    Active episodes: {json.dumps(active_episodes_summary, default=str)}
    
    Choose episodes appropriate for current proximity. Don't force Lilith-centric episodes if she's not available."""
    
        return await self.gpt_service.call_with_validation(
            model=_WEAVER_MODEL,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=SituationWeave,
            temperature=0.7
        )
    
    def _select_relevant_episode(self, location: str):
        """Fallback episode selection without AI"""
        # Find location-relevant episodes
        relevant = [
            ep for ep in self.active_episodes
            if ep.location_relevant and ep.location_relevant.lower() in location.lower()
        ]
        
        if relevant:
            self.spotlight_episode = random.choice(relevant)
        else:
            # Pick any low-progress episode
            low_progress = [ep for ep in self.active_episodes if ep.progress < 50]
            if low_progress:
                self.spotlight_episode = random.choice(low_progress)
            else:
                self.spotlight_episode = None
    
    async def _generate_duet_response(
        self,
        context: Dict[str, Any],
        mechanics_results: Dict[str, Any],
        retry: bool = False
    ) -> Dict[str, Any]:
        """Generate response with Lilith as co-protagonist"""
        
        response = {
            'status': 'success',
            'timestamp': self.current_time.isoformat()
        }
        
        # Update context with mechanics
        context['special_mechanics'] = mechanics_results
        context['emotional_intensity'] = self._calculate_slice_intensity(context, mechanics_results)
        
        # Since Lilith is always present, ALWAYS use dialogue playwright
        if self.dynamic_mode:
            dialogue_script = await self._write_duet_dialogue(context, mechanics_results)
            
            response['dialogue'] = {
                'script': dialogue_script.script,
                'subtext': dialogue_script.subtext,
                'power_dynamics': dialogue_script.power_dynamics,
                'lilith_speaks': True  # She always gets at least one line
            }
        else:
            # Fallback dialogue
            response['dialogue'] = {
                'script': [
                    {"speaker": "Lilith", "line": self._get_lilith_line(context)},
                    {"speaker": "Player", "line": "[Your response]"}
                ],
                'lilith_speaks': True
            }
        
        # Generate scene description considering episode
        scene_desc = await self._generate_slice_description(context)
        response['scene_description'] = scene_desc
        
        # Add episode-specific content if relevant
        if self.spotlight_episode:
            response['episode_content'] = {
                'current': self.spotlight_episode.premise,
                'stakes': self.spotlight_episode.stakes,
                'threads': self.spotlight_episode.open_threads,
                'progress': self.spotlight_episode.progress
            }
        
        # Add mechanics results
        if mechanics_results:
            response['special_events'] = mechanics_results
        
        # Add slice-of-life elements
        response['slice_elements'] = {
            'queen_doing': self.queen_current_state.activity if self.queen_current_state else "accompanying you",
            'background_detail': self._get_background_detail(context),
            'available_topics': self._get_conversation_topics(context)
        }
        
        return response
    
    async def _write_duet_dialogue(
        self, context: Dict[str, Any], mechanics: Dict[str, Any]
    ) -> DialogueScript:
        """Agent 7 adapted: Write dialogue based on who's actually present"""
        
        # Determine who can speak based on proximity
        speakers_available = []
        if self.proximity == Proximity.TOGETHER:
            speakers_available.append("Lilith")
        
        # Add spotlight NPC if different from Lilith and present
        if self.current_spotlight_npc and self.current_spotlight_npc != self.lilith_npc_id:
            spotlight_name = await self._get_npc_name(self.current_spotlight_npc)
            if spotlight_name:
                speakers_available.append(spotlight_name)
        
        if not speakers_available:
            # No one to have dialogue with
            return DialogueScript(
                script=[{"speaker": "Narrator", "line": "You are alone with your thoughts."}],
                subtext={},
                power_dynamics="solitary",
                priority_speaker="Narrator",
                cast_present=[]
            )
        
        system_prompt = f"""You write dialogue for available characters.
    Available speakers: {', '.join(speakers_available)}
    Proximity to Lilith: {self.proximity.value}
    Only characters who are TOGETHER with the player can have full dialogue."""
    
        queen_mood = context.get('lilith_emotion', 'contemplative')
        episode_context = f"Current episode: {self.spotlight_episode.premise}" if self.spotlight_episode else "Daily moments"
    
        user_prompt = f"""Write natural dialogue for this moment:
    Player said: "{context.get('player_input')}"
    Speakers present: {speakers_available}
    {"Lilith's mood: " + queen_mood if "Lilith" in speakers_available else ""}
    {episode_context}
    Special events: {list(mechanics.keys()) if mechanics else 'None'}
    
    Create 2-4 lines of natural conversation."""
    
        result = await self.gpt_service.call_with_validation(
            model=_DIALOGUE_MODEL,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=DialogueScript,
            temperature=0.7
        )
        
        # Ensure only available speakers are used
        result.cast_present = speakers_available
        result.script = [line for line in result.script if line.get('speaker') in speakers_available + ['Player', 'Narrator']]
        
        return result

    async def _get_queen_acknowledgment(self, context: Dict[str, Any]) -> str:
        """Get a brief acknowledgment from Queen when in same venue"""
        acknowledgments = {
            'busy': ["*A brief nod in your direction*", "*Glances up momentarily*"],
            'neutral': ["*Catches your eye and gives a small smile*", "*Raises her glass slightly*"],
            'warm': ["*Warm smile before returning to her conversation*", "*Mouths 'one moment' to you*"],
            'cold': ["*Deliberately doesn't look your way*", "*Turns slightly away*"]
        }
        
        mood = self.queen_current_state.mood if self.queen_current_state else 'neutral'
        if self.relationship_tension < -40:
            mood = 'cold'
        elif self.relationship_tension > 60:
            mood = 'warm'
        
        return random.choice(acknowledgments.get(mood, acknowledgments['neutral']))
    
    async def _check_episode_lifecycle(self):
        """Check episodes for resolution and dormancy"""
        resolved = []
        going_dormant = []
        
        for episode in self.active_episodes:
            # Resolve if complete
            if episode.progress >= 100:
                resolved.append(episode)
            # Go dormant if abandoned
            elif (self.current_time - episode.last_active).total_seconds() > 172800:  # 48 hours
                episode.dormant_since = self.current_time
                episode.last_state = f"Progress was {episode.progress}%"
                episode.unresolved_stakes = episode.stakes
                going_dormant.append(episode)
            # Resolve if all threads closed
            elif not episode.open_threads:
                episode.progress = 100
                resolved.append(episode)
        
        # Process resolved
        for episode in resolved:
            self.active_episodes.remove(episode)
            self.completed_episodes.append(episode.id)
            logger.info(f"Episode resolved: {episode.premise}")
        
        # Process dormant
        for episode in going_dormant:
            self.active_episodes.remove(episode)
            self.dormant_episodes.append(episode)
            logger.info(f"Episode went dormant: {episode.premise}")
    
    async def run_simulation_daemon(self):
        """Enhanced simulation daemon with dormant reactivation"""
        if not self.dynamic_mode:
            return
        
        # Run every 30 minutes
        if self._last_simulation_run:
            time_since = (datetime.now() - self._last_simulation_run).total_seconds() / 60
            if time_since < 30:
                return
        
        self._last_simulation_run = datetime.now()
        
        # Update Queen's schedule
        self.queen_current_state = await self._generate_queen_schedule()
        
        # Update neighborhood
        await self._run_neighborhood_pulse()
        
        # Progress active episodes
        for episode in self.active_episodes:
            if random.random() < 0.1:
                episode.progress = min(100, episode.progress + random.randint(5, 15))
        
        # Increment dormant reactivation scores
        for episode in self.dormant_episodes:
            episode.reactivation_score += random.randint(1, 5)
            
            # Chance to reactivate high-score episodes
            if episode.reactivation_score > 15 and random.random() < 0.3:
                episode.last_active = self.current_time
                episode.progress = max(episode.progress, 10)
                self.active_episodes.append(episode)
                self.dormant_episodes.remove(episode)
                logger.info(f"Dormant episode self-reactivated: {episode.premise}")
        
        # Run loose thread indexer daily
        await self._run_loose_thread_indexer()
        
        # Clean up
        await self._check_episode_lifecycle()
    
    async def _run_loose_thread_indexer(self):
        """Daily scan of dormant content for opportunities"""
        if not self.dynamic_mode:
            return
        
        # Only run daily
        if self._last_thread_index_run:
            time_since = (datetime.now() - self._last_thread_index_run).days
            if time_since < 1:
                return
        
        self._last_thread_index_run = datetime.now()
        
        system_prompt = """You index dormant story threads and memories for reentry opportunities.
Look for connections between old threads and current events.
Suggest which dormant episodes are ready to return with new urgency."""

        # Get recent memories with abandonment/unfinished tags
        relevant_memories = [
            mem for mem in self.memory_shards[-50:]
            if any(tag in ["abandonment", "unfinished", "promise", "threat"] for tag in mem.tags)
        ]

        user_prompt = f"""Index loose threads:
Dormant episodes: {[(ep.premise, ep.unresolved_stakes, ep.reactivation_score) for ep in self.dormant_episodes[:10]]}
Recent memories: {[(mem.text, mem.tags) for mem in relevant_memories[:5]]}
Current situation: Proximity={self.proximity.value}, Tension={self.relationship_tension}

Find high-weight threads and reentry hooks."""

        self._last_thread_index = await self.gpt_service.call_with_validation(
            model=_MEMORY_MODEL,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=LooseThreadIndex,
            temperature=0.6
        )
        
        # Boost recommended episodes
        for ep_id in self._last_thread_index.recommended_reactivations:
            episode = next((ep for ep in self.dormant_episodes if ep.id == ep_id), None)
            if episode:
                episode.reactivation_score += 10
    
    async def _update_slice_state(
        self, context: Dict[str, Any], response: Dict[str, Any]
    ):
        """Update state for slice-of-life progression"""
        
        # Update episode progress if one is spotlighted
        if self.spotlight_episode:
            # Progress based on engagement
            progress_gain = 10
            if "resolve" in context.get('player_input', '').lower():
                progress_gain = 25
            elif any(thread.lower() in context.get('player_input', '').lower() 
                    for thread in self.spotlight_episode.open_threads):
                progress_gain = 20
            
            self.spotlight_episode.progress = min(100, self.spotlight_episode.progress + progress_gain)
            self.spotlight_episode.last_active = self.current_time
        
        # Update network awareness if relevant
        if response.get('special_events', {}).get('network_revelation'):
            self.network_awareness = min(100, self.network_awareness + 10)
        
        # Update Queen goals based on interaction
        if self.dynamic_mode:
            memory_plan = await self._curate_slice_memories(context, response)
            if memory_plan.queen_goals_update:
                self._update_queen_goals(memory_plan.queen_goals_update)
        
        # Save state
        await self._save_slice_state()
    
    async def _curate_slice_memories(
        self, context: Dict[str, Any], response: Dict[str, Any]
    ) -> MemoryCuration:
        """Agent 9 adapted: Curate memories including shared goals"""
        system_prompt = """You curate memories for a slice-of-life experience.
Focus on: relationship moments, episode progress, Queen's reactions, shared experiences.
Also track updates to Lilith's goals - what she wants personally and with the player."""

        user_prompt = f"""Curate this interaction:
Player action: {context.get('player_input')}
Lilith's response: {response.get('dialogue', {}).get('script', [])}
Episode progress: {self.spotlight_episode.premise if self.spotlight_episode else 'No episode'}
Current goals: {json.dumps(self.queen_goals, default=str)}

Select important memories and any goal updates."""

        return await self.gpt_service.call_with_validation(
            model=_MEMORY_MODEL,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=MemoryCuration,
            temperature=0.5
        )
    
    def _update_queen_goals(self, updates: Dict[str, Any]):
        """Update Lilith's goals"""
        for category, new_goals in updates.items():
            if category in self.queen_goals:
                if isinstance(new_goals, list):
                    self.queen_goals[category] = new_goals
                else:
                    self.queen_goals[category].append(new_goals)
    
    async def _check_episode_resolution(self):
        """Check if any episodes should resolve"""
        resolved = []
        
        for episode in self.active_episodes:
            # Resolve if complete
            if episode.progress >= 100:
                resolved.append(episode)
            # Resolve if abandoned (not touched in 48 hours game time)
            elif (self.current_time - episode.last_active).days > 2:
                episode.progress = -1  # Mark as abandoned
                resolved.append(episode)
            # Resolve if all threads closed
            elif not episode.open_threads:
                episode.progress = 100
                resolved.append(episode)
        
        # Remove resolved episodes
        for episode in resolved:
            self.active_episodes.remove(episode)
            self.completed_episodes.append(episode.id)
            logger.info(f"Episode resolved: {episode.premise} (Progress: {episode.progress}%)")
    
    async def _save_slice_state(self):
        """Save current state to database"""
        story_flags = {
            'lilith_npc_id': self.lilith_npc_id,
            'network_awareness': self.network_awareness,
            'information_layer': self.information_layer,
            'player_rank': self.player_rank,
            'active_episodes': [ep.dict() for ep in self.active_episodes],
            'queen_goals': self.queen_goals,
            'queen_schedule': self.queen_current_state.dict() if self.queen_current_state else {},
            'completed_episodes': self.completed_episodes[-20:]  # Keep last 20
        }
        
        async with get_db_connection_context() as conn:
            await conn.execute(
                """
                INSERT INTO story_states (
                    user_id, conversation_id, story_id,
                    current_act, current_beat, story_flags
                )
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (user_id, conversation_id, story_id)
                DO UPDATE SET 
                    story_flags = EXCLUDED.story_flags,
                    updated_at = NOW()
                """,
                self.user_id, self.conversation_id, self.story_id,
                0, None, json.dumps(story_flags)  # No acts/beats
            )
    
    async def _generate_slice_description(self, context: Dict[str, Any]) -> str:
        """Generate scene description based on proximity"""
        location = context.get('current_location', 'somewhere')
        
        # Base description varies by proximity
        if self.proximity == Proximity.TOGETHER:
            base_desc = f"You're in {location} with Lilith."
            if self.queen_current_state:
                base_desc += f" She's {self.queen_current_state.activity}."
        elif self.proximity == Proximity.SAME_VENUE:
            base_desc = f"You're in {location}. Lilith is here too, but {self.queen_current_state.activity if self.queen_current_state else 'occupied with her own matters'}."
        elif self.proximity == Proximity.DIFFERENT_VENUE:
            base_desc = f"You're in {location}."
            if self.neighborhood_pulse:
                base_desc += f" {random.choice(self.neighborhood_pulse.local_events).description if self.neighborhood_pulse.local_events else ''}"
        else:  # OFFSCREEN
            base_desc = f"You're in {location}. The absence of a certain presence is notable."
        
        # Add time and weather
        if self.neighborhood_pulse:
            base_desc += f" {self.neighborhood_pulse.weather}."
        
        # Enhance with text generator
        enhanced = await self.text_generator.enhance_scene_description(
            base_desc,
            context.get('scene_type', 'slice-daily'),
            {
                'emotional_tone': context.get('lilith_emotion', 'neutral'),
                'proximity': self.proximity.value
            }
        )
        
        return enhanced
    
    # Helper methods
    def _get_time_period(self) -> str:
        """Get current time period"""
        hour = self.current_time.hour
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon" 
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"
    
    def _get_season(self) -> str:
        """Get current season"""
        month = self.current_time.month
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "fall"
    
    def _get_recent_locations(self) -> List[str]:
        """Get recently visited locations"""
        # This would pull from memory system in full implementation
        return ["SOMA", "Mission", "Marina"]
    
    def _determine_slice_scene_type(self, context: Dict[str, Any]) -> str:
        """Determine scene type from episodes and mood"""
        if self.spotlight_episode:
            # Use episode tags
            if "slice-thriller" in self.spotlight_episode.tags:
                return "thriller"
            elif "slice-romance" in self.spotlight_episode.tags:
                return "romantic"
            elif "slice-mystery" in self.spotlight_episode.tags:
                return "mysterious"
        
        # Default based on Queen's mood
        mood = context.get('lilith_emotion', 'neutral')
        mood_to_scene = {
            'dominant': 'power-dynamic',
            'vulnerable': 'intimate',
            'protective': 'tense',
            'contemplative': 'thoughtful',
            'playful': 'light'
        }
        
        return f"slice-{mood_to_scene.get(mood, 'daily')}"
    
    def _calculate_slice_intensity(
        self, context: Dict[str, Any], mechanics: Dict[str, Any]
    ) -> float:
        """Calculate emotional intensity for slice-of-life"""
        base = 0.3
        
        # Episode stakes add intensity
        if self.spotlight_episode:
            if "high-stakes" in self.spotlight_episode.tags:
                base += 0.3
            elif "medium-stakes" in self.spotlight_episode.tags:
                base += 0.2
            else:
                base += 0.1
        
        # Relationship tension
        trust = context.get('lilith_data', {}).get('trust', 50)
        if trust > 80:
            base += 0.2
        elif trust < 30:
            base += 0.1
        
        # Special mechanics
        if mechanics.get('mask_event'):
            base += 0.2
        if mechanics.get('three_words_moment'):
            base += 0.4
        
        return min(1.0, base)

    async def _get_reflection_lens(self, location: str) -> ReflectionLens:
        """Get ambient details when Lilith isn't present"""
        system_prompt = """You provide atmospheric details when the player is alone.
Include subtle reminders of Lilith's influence in the world.
Show how the network and Queen affect things even when not visible."""

        user_prompt = f"""Player is alone at {location}.
Queen is at: {self.queen_current_state.location if self.queen_current_state else 'unknown'}
Network awareness: {self.network_awareness}%
Recent episodes: {[ep.premise for ep in self.active_episodes[:3]]}

Generate ambient details and Queen echoes."""

        return await self.gpt_service.call_with_validation(
            model=_SCENE_MODEL,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=ReflectionLens,
            temperature=0.7
        )
    
    def _get_lilith_line(self, context: Dict[str, Any]) -> str:
        """Get a contextual Lilith line for non-dynamic mode"""
        mood = context.get('lilith_emotion', 'neutral')
        activity = self.queen_current_state.activity if self.queen_current_state else "with you"
        
        lines = {
            'dominant': f"Even while {activity}, I notice everything, darling.",
            'vulnerable': f"It's... nice, having you here while I'm {activity}.",
            'protective': f"Stay close. Even during {activity}, threats can emerge.",
            'contemplative': f"I was just thinking... but please, what did you want?",
            'neutral': f"Yes? I'm listening, even while {activity}."
        }
        
        return lines.get(mood, "Mmm?")
    
    def _get_background_detail(self, context: Dict[str, Any]) -> str:
        """Get atmospheric background detail"""
        location = context.get('current_location', 'somewhere')
        time_period = context.get('time_of_day', 'day')
        
        details = {
            'morning': f"Morning light filters through {location}",
            'afternoon': f"The afternoon crowd moves through {location}",
            'evening': f"Evening shadows lengthen across {location}",
            'night': f"Night wraps {location} in possibility"
        }
        
        return details.get(time_period, f"The atmosphere in {location} is charged")
    
    def _get_conversation_topics(self, context: Dict[str, Any]) -> List[str]:
        """Get available conversation topics"""
        topics = ["Her current activity", "The weather", "Plans for later"]
        
        if self.spotlight_episode:
            topics.append(f"The {self.spotlight_episode.tags[0].replace('slice-', '')} situation")
        
        if self.network_awareness > 30:
            topics.append("Network business")
        
        if context.get('lilith_data', {}).get('trust', 0) > 60:
            topics.extend(["Her past", "Your relationship", "Her feelings"])
        
        return topics[:5]  # Limit to 5
    
    # Import other needed methods from original class
    async def _get_location_data(self, location_name: str) -> Dict[str, Any]:
        """Get location details from database"""
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow(
                """
                SELECT location_name, description, metadata
                FROM Locations
                WHERE user_id = $1 AND conversation_id = $2 
                AND LOWER(location_name) = LOWER($3)
                """,
                self.user_id, self.conversation_id, location_name
            )
            
            if row:
                return {
                    'name': row['location_name'],
                    'description': row['description'],
                    'metadata': json.loads(row['metadata']) if row['metadata'] else {}
                }
            
            return {'name': location_name, 'description': '', 'metadata': {}}
    
    def _is_network_business(self, context: Dict[str, Any]) -> bool:
        """Check if current action involves network business"""
        indicators = [
            'network' in context.get('player_input', '').lower(),
            'garden' in context.get('player_input', '').lower(),
            'transform' in context.get('player_input', '').lower(),
            self.spotlight_episode and self.spotlight_episode.network_related,
            context.get('is_network_relevant', False)
        ]
        
        return any(indicators)
    
    def _is_private_location(self, location: str) -> bool:
        """Check if location is private"""
        location_lower = location.lower()
        private_keywords = ['private', 'chambers', 'bedroom', 'personal', 'hidden']
        
        return any(keyword in location_lower for keyword in private_keywords)
    
    def _determine_lilith_emotion(self, context: Dict[str, Any]) -> str:
        """Determine Lilith's current emotional state based on proximity"""
        # Check proximity first
        if self.proximity == Proximity.OFFSCREEN:
            return 'absent'
        elif self.proximity == Proximity.DIFFERENT_VENUE:
            return 'distant'
        
        # For SAME_VENUE, base it on what she's doing
        if self.proximity == Proximity.SAME_VENUE:
            if self.queen_current_state:
                base_mood = self.queen_current_state.mood
                # She's aware of player but focused on her own activities
                if self.relationship_tension < -20:
                    return 'avoiding'
                else:
                    return base_mood
            else:
                return "occupied"
        
        # TOGETHER - full emotional range
        if self.queen_current_state:
            base_mood = self.queen_current_state.mood
        else:
            base_mood = "neutral"
        
        # Modify based on player input
        player_input = context.get('player_input', '').lower()
        
        if any(word in player_input for word in ['disappear', 'leave', 'goodbye']):
            return 'fear'
        elif any(word in player_input for word in ['love', 'stay', 'promise']):
            trust = context.get('lilith_data', {}).get('trust', 0)
            return 'vulnerable' if trust > 60 else 'defensive'
        elif any(word in player_input for word in ['kneel', 'submit', 'obey']):
            return 'dominant'
        elif any(word in player_input for word in ['help', 'save', 'protect']):
            return 'protective'
        
        return base_mood
    
    # Keep other methods from original implementation as needed...
    async def _classify_intent(self, player_input: str) -> IntentClassification:
        """Agent 1: Enhanced intent classification for proximity awareness"""
        system_prompt = """You classify player intent in a dynamic story where characters can be together or apart.
    Pay special attention to proximity desires:
    - "come with me", "join me", "let's go together" = invite
    - "I'll go alone", "see you later", "I need space" = leave  
    - "where are you", "find Lilith", "look for her" = seek
    - "send her away", "leave me alone", "go away" = banish
    The network exists but this is about relationships and daily life."""
    
        current_proximity_context = f"""
    Current proximity: {self.proximity.value}
    Can invite Queen: {self._can_invite_queen()}
    Queen is: {self.queen_current_state.activity if self.queen_current_state and self.proximity != Proximity.OFFSCREEN else 'not in your story'}"""
    
        user_prompt = f"""Classify this player input:
    "{player_input}"
    {current_proximity_context}
    
    Identify any proximity intentions along with general intent."""
    
        return await self.gpt_service.call_with_validation(
            model=_INTENT_MODEL,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=IntentClassification,
            temperature=0.3,
            use_cache=True
        )
    
    # Network mechanics can stay mostly the same but less prominent
    async def _check_network_mechanics(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check network mechanics with proximity awareness"""
        results = {}
        
        # Only check if there's actual network relevance
        if not context.get('is_network_relevant'):
            return results
        
        # Recognition codes work regardless of proximity
        if self._check_recognition_code(context['player_input']):
            results['network_noticed'] = {
                'type': 'coded_language',
                'description': 'Your words carry network meaning'
            }
            self.network_awareness = min(100, self.network_awareness + 5)
        
        # Some mechanics require Queen's presence
        if self.proximity == Proximity.TOGETHER:
            # She can directly respond to network matters
            if self.network_awareness > 60 and "transform" in context['player_input'].lower():
                results['transformation_discussed'] = {
                    'description': 'Lilith\'s eyes gleam with interest at the mention of transformation'
                }
        elif self.proximity == Proximity.SAME_VENUE:
            # She might notice but won't engage
            if self.network_awareness > 40 and self._check_recognition_code(context['player_input']):
                results['queen_noticed'] = {
                    'description': 'You catch Lilith glancing your way at the coded words'
                }
        
        # Threat assessment always runs if network-active
        if self.dynamic_mode and self.network_awareness > 40:
            threat = await self._assess_threats(context)
            if threat.kozlov_activity > 60 or threat.federal_heat > 50:
                results['background_tension'] = {
                    'description': 'The atmosphere feels charged with hidden danger',
                    'queen_aware': self.proximity in [Proximity.TOGETHER, Proximity.SAME_VENUE]
                }
        
        return results
    
    def _check_recognition_code(self, player_input: str) -> bool:
        """Check if player used network recognition codes"""
        codes = [
            "interesting energy",
            "needs pruning", 
            "the garden",
            "roses and thorns"
        ]
        input_lower = player_input.lower()
        return any(code in input_lower for code in codes)
    
    async def _assess_threats(self, context: Dict[str, Any]) -> ThreatAssessment:
        """Agent 4: Assess network threats dynamically"""
        system_prompt = """You track background threats in a slice-of-life story.
The network exists but operates quietly. Threats should be subtle, not overwhelming.
This is about tension, not action scenes."""

        user_prompt = f"""Assess background threats:
Current episode: {self.spotlight_episode.premise if self.spotlight_episode else 'None'}
Network exposure: {self.information_layer}
Recent player actions: {context.get('player_input')}

Generate subtle threat levels."""

        return await self.gpt_service.call_with_validation(
            model=_THREAT_MODEL,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=ThreatAssessment,
            temperature=0.5
        )
    
    async def _check_lilith_special_mechanics(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check Lilith's mechanics only when she's actually present"""
        if not self.lilith_mechanics:
            return {}
        
        # No mechanics if she's offscreen
        if self.proximity == Proximity.OFFSCREEN:
            return {}
        
        results = {}
        
        # Base mechanics check only if together or same venue
        if self.proximity in [Proximity.TOGETHER, Proximity.SAME_VENUE]:
            base_checks = await self._check_base_lilith_mechanics(context)
            results.update(base_checks)
        
        # Inner voice only if together (can't hear thoughts from across the room)
        if self.dynamic_mode and self.proximity == Proximity.TOGETHER:
            inner_voice = await self._get_lilith_inner_voice(context)
            
            if inner_voice.vulnerability_level > 3:
                results['lilith_reaction'] = {
                    'type': 'emotional_response',
                    'level': inner_voice.vulnerability_level,
                    'state': inner_voice.emotional_state
                }
            
            if inner_voice.poetry_line:
                results['poetry_moment'] = {
                    'poetry_moment': True,
                    'line': inner_voice.poetry_line,
                    'context': 'intimate_moment'
                }
        
        return results

    
    async def _check_base_lilith_mechanics(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Base Lilith mechanics without AI"""
        results = {}
        
        # Since she's always present, check her state more frequently
        mask_check = await self.lilith_mechanics.check_mask_state(context)
        if mask_check.get('slipped'):
            results['mask_event'] = mask_check
        
        # Check for poetry moment
        poetry_check = await self.lilith_mechanics.check_poetry_moment(context)
        if poetry_check.get('poetry_moment'):
            results['poetry_moment'] = poetry_check
        
        return results
    
    async def _get_lilith_inner_voice(self, context: Dict[str, Any]) -> LilithInnerVoice:
        """Agent 5: Get Lilith's inner voice - only when close enough"""
        
        # No inner voice if not together
        if self.proximity != Proximity.TOGETHER:
            return LilithInnerVoice(
                mask_slip=False,
                mask_integrity=100,
                emotional_state="distant",
                vulnerability_level=0,
                is_offscreen=self.proximity == Proximity.OFFSCREEN,
                separation_feeling="focused elsewhere" if self.proximity == Proximity.SAME_VENUE else None
            )
        
        lilith_data = context.get('lilith_data', {})
        player_input = context.get('player_input', '')
        
        system_prompt = """You are Lilith Ravencroft's inner voice when she's intimately close to the player.
    She's physically with them, sharing the same space, able to be touched.
    React to both big and small things - a touch, a word, a glance.
    Her trauma and power are always there, but so is her humanity."""
    
        user_prompt = f"""Current intimate moment:
    Trust: {lilith_data.get('trust', 0)}/100
    Current activity: {context.get('queen_schedule', {}).get('activity', 'with you')}
    Player said: "{player_input}"
    Mood: {context.get('queen_schedule', {}).get('mood', 'neutral')}
    Episode context: {self.spotlight_episode.premise if self.spotlight_episode else 'Just being together'}
    
    Generate inner response - she's close enough to touch."""
    
        return await self.gpt_service.call_with_validation(
            model=_INNER_VOICE_MODEL,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=LilithInnerVoice,
            temperature=0.7
        )

    
    async def run_simulation_daemon(self):
        """Agent 11: Background simulation for living world"""
        if not self.dynamic_mode:
            return
        
        # Run more frequently for slice-of-life (every 30 min vs hourly)
        if self._last_simulation_run:
            time_since = (datetime.now() - self._last_simulation_run).total_seconds() / 60
            if time_since < 30:
                return
        
        self._last_simulation_run = datetime.now()
        
        # Update Queen's schedule
        self.queen_current_state = await self._generate_queen_schedule()
        
        # Update neighborhood
        await self._run_neighborhood_pulse()
        
        # Progress episodes in background
        for episode in self.active_episodes:
            if random.random() < 0.1:  # 10% chance of background progress
                episode.progress = min(100, episode.progress + random.randint(5, 15))
        
        # Clean up old episodes
        await self._check_episode_resolution()

    async def handle_queen_reunion(self) -> Dict[str, Any]:
        """Special handling for reuniting with Lilith after separation"""
        hours_apart = (self.current_time - self.last_proximity_change).seconds / 3600
        
        response = {
            'reunion': True,
            'hours_apart': hours_apart,
            'her_mood': self.queen_current_state.mood if self.queen_current_state else 'unknown'
        }
        
        # Generate reunion reaction based on tension and time
        if self.relationship_tension > 60:
            response['her_reaction'] = "warm" if hours_apart < 4 else "relieved"
        elif self.relationship_tension < -20:
            response['her_reaction'] = "cold" if hours_apart < 12 else "neutral"  
        else:
            response['her_reaction'] = "casual"
        
        # Update proximity
        self.proximity = Proximity.TOGETHER
        self.last_proximity_change = self.current_time
        
        return response
    
    async def get_story_status(self) -> Dict[str, Any]:
        """Get current slice-of-life status with proximity info"""
        if not self._initialized:
            await self.initialize()
        
        # Determine Queen's availability based on proximity
        queen_status = {
            Proximity.TOGETHER: "with_you",
            Proximity.SAME_VENUE: "nearby", 
            Proximity.DIFFERENT_VENUE: "elsewhere",
            Proximity.OFFSCREEN: "severed"
        }.get(self.proximity, "unknown")
        
        status = {
            'mode': 'unified_sandbox',
            'proximity': self.proximity.value,
            'queen_status': queen_status,
            'queen_state': {
                'location': self.queen_current_state.location if self.queen_current_state else "Unknown",
                'activity': self.queen_current_state.activity if self.queen_current_state else "Unknown", 
                'mood': self.queen_current_state.mood if self.queen_current_state and self.proximity != Proximity.OFFSCREEN else "Unknown",
                'available_for': self.queen_current_state.available_for if self.queen_current_state and self.proximity != Proximity.OFFSCREEN else [],
                'can_be_invited': self._can_invite_queen(),
                'can_be_restored': self._can_retcon_queen()
            },
            'active_episodes': [
                {
                    'premise': ep.premise,
                    'progress': ep.progress,
                    'stakes': ep.stakes,
                    'tags': ep.tags,
                    'lilith_involvement': ep.lilith_involvement
                }
                for ep in self.active_episodes
            ],
            'dormant_episodes_count': len(self.dormant_episodes),
            'current_spotlight_npc': await self._get_npc_name(self.current_spotlight_npc),
            'cast_scores': self._get_top_cast_scores(),
            'network_status': {
                'awareness': self.network_awareness,
                'layer': self.information_layer,
                'rank': self.player_rank
            },
            'neighborhood': {
                'mood': self.neighborhood_pulse.bay_mood if self.neighborhood_pulse else "normal",
                'weather': self.neighborhood_pulse.weather if self.neighborhood_pulse else "foggy",
                'events': [e.hook for e in self.neighborhood_pulse.local_events] if self.neighborhood_pulse else []
            },
            'time': self.current_time.strftime("%A, %I:%M %p"),
            'setting': 'San Francisco Bay Area, 2025'
        }
        
        return status
    
    # Memory shard methods
    def add_memory_shard(self, text: str, actors: List[str], tags: List[str], importance: int = 50):
        """Add a structured memory shard"""
        shard = MemoryShard(
            text=text,
            actors=actors,
            tags=tags,
            importance=importance
        )
        self.memory_shards.append(shard)
        
        if len(self.memory_shards) > 1000:
            self.memory_shards = self.memory_shards[-1000:]
    
    async def handle_special_choice(
        self, choice_type: str, player_choice: str
    ) -> Dict[str, Any]:
        """Handle special story choices with Agent 10 ripple planning"""
        
        # Base choice handling
        base_result = await self._handle_base_special_choice(choice_type, player_choice)
        
        # Agent 10: Ripple-Planner
        if self.dynamic_mode:
            ripple_plan = await self._plan_ripples(choice_type, player_choice, base_result)
            
            # Store consequences for later processing
            self._pending_consequences.extend(ripple_plan.consequences)
            
            base_result['ripple_effects'] = {
                'immediate': ripple_plan.immediate_effects,
                'delayed': ripple_plan.delayed_effects,
                'network_impact': ripple_plan.network_impact
            }
        
        return base_result
    
    async def _handle_base_special_choice(
        self, choice_type: str, player_choice: str
    ) -> Dict[str, Any]:
        """Handle base special choices without AI"""
        if choice_type == "rose_or_thorn":
            # Player choosing their path in the network
            if player_choice.lower() == "rose":
                self.story_flags['chosen_path'] = 'cultivator'
                role_desc = "You will help others grow and transform"
            else:
                self.story_flags['chosen_path'] = 'protector'
                role_desc = "You will protect the vulnerable and enforce justice"
            
            async with get_db_connection_context() as conn:
                await conn.execute(
                    """
                    UPDATE story_states
                    SET story_flags = story_flags || $4
                    WHERE user_id = $1 AND conversation_id = $2 AND story_id = $3
                    """,
                    self.user_id, self.conversation_id, self.story_id,
                    json.dumps({'chosen_path': self.story_flags['chosen_path']})
                )
            
            return {
                'choice_made': choice_type,
                'path_chosen': self.story_flags['chosen_path'],
                'description': role_desc,
                'new_responsibilities': True
            }
        
        return {"error": f"Unknown choice type: {choice_type}"}
    
    async def _plan_ripples(
        self, choice_type: str, player_choice: str, base_result: Dict[str, Any]
    ) -> RipplePlan:
        """Agent 10: Plan consequences of choices"""
        system_prompt = """You plan realistic consequences for player choices in Queen of Thorns.
Consider how choices ripple through the network and affect relationships.
Be creative but grounded in the established world."""

        user_prompt = f"""Plan consequences for this choice:
Choice type: {choice_type}
Player chose: {player_choice}
Result: {json.dumps(base_result, default=str)}
Network awareness: {self.network_awareness}%
Current rank: {self.player_rank}

Generate 2-3 consequences with timing and probability."""

        return await self.gpt_service.call_with_validation(
            model=_RIPPLE_MODEL,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=RipplePlan,
            temperature=0.7
        )
    
    async def _run_simulation_daemon(self):
        """Agent 11: Background simulation updates"""
        if not self.dynamic_mode:
            return
        
        # Check if enough time has passed
        if self._last_simulation_run:
            time_since = (datetime.now() - self._last_simulation_run).total_seconds() / 3600
            if time_since < 1:  # Run hourly at most
                return
        
        self._last_simulation_run = datetime.now()
        
        try:
            simulation = await self._simulate_world_changes()
            
            # Apply world changes through governance
            ctx = CanonicalContext(user_id=self.user_id, conversation_id=self.conversation_id)
            
            # Update character states
            for char_name, changes in simulation.character_changes.items():
                # Find character in canon
                async with get_db_connection_context() as conn:
                    char_row = await conn.fetchrow(
                        """
                        SELECT npc_id FROM NPCStats
                        WHERE LOWER(npc_name) = LOWER($1)
                        AND user_id = $2 AND conversation_id = $3
                        """,
                        char_name, self.user_id, self.conversation_id
                    )
                    
                    if char_row:
                        await propose_canonical_change(
                            ctx=ctx,
                            entity_type="NPCStats",
                            entity_identifier={"npc_id": char_row['npc_id']},
                            updates=changes,
                            reason=f"Simulation daemon: {char_name} progression",
                            agent_type=AgentType.NARRATIVE_CRAFTER,
                            agent_id="simulation_daemon"
                        )
            
            # Store network operations in state
            if simulation.network_operations:
                self.story_flags['active_operations'] = simulation.network_operations
                await self._save_story_flags()
            
        except Exception as e:
            logger.error(f"Simulation daemon error: {e}")
    
    async def _simulate_world_changes(self) -> SimulationUpdate:
        """Agent 11: Simulate world progression"""
        hours_passed = 4  # Simulate 4 hours of world time
        
        system_prompt = """You simulate background world progression for Queen of Thorns.
The network continues operating when the player isn't watching.
NPCs have their own lives and agendas. The world feels alive.
Be subtle - major events should be rare."""

        user_prompt = f"""Simulate {hours_passed} hours of world progression:
Network state: {self.information_layer} layer exposed to player
Active threats: Kozlov level {self._last_threat_assessment.kozlov_activity if self._last_threat_assessment else 50}
Player rank: {self.player_rank}
Recent events: {self.story_flags.get('recent_events', [])[-3:]}

Generate subtle world updates."""

        return await self.gpt_service.call_with_validation(
            model=_NARRATIVE_MODEL,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=SimulationUpdate,
            temperature=0.8
        )
    
    async def _save_story_flags(self):
        """Save story flags to database"""
        async with get_db_connection_context() as conn:
            await conn.execute(
                """
                UPDATE story_states
                SET story_flags = $4, updated_at = NOW()
                WHERE user_id = $1 AND conversation_id = $2 AND story_id = $3
                """,
                self.user_id, self.conversation_id, self.story_id,
                json.dumps(self.story_flags)
            )
    
    def _validate_beat_proposal(self, beat_id: str, context: Dict[str, Any]) -> bool:
        """Validate AI-proposed beat against deterministic rules"""
        # Add your validation logic here
        # For now, accept proposals with high network relevance
        return context.get('player_intent', {}).get('network_relevance', 0) > 5

# Export the unified runner
QueenOfThornsStoryRunner = QueenOfThornsSliceRunner
