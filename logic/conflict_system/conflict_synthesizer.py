# logic/conflict_system/conflict_synthesizer.py
"""
Conflict Synthesizer - THE Central Orchestration System

This module is the single orchestrator for all conflict operations.
All conflict subsystems register with and communicate through this synthesizer.
No other module should attempt to orchestrate - they only handle their specific domain.
"""

import logging
import json
import asyncio
from typing import Dict, List, Any, Optional, Set, Type, Callable, TypedDict, NotRequired
from pydantic import BaseModel, Field, ConfigDict  
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import weakref
from lore.core.canon import log_canonical_event, ensure_canonical_context
from lore.core.context import CanonicalContext

from agents import Agent, function_tool, ModelSettings, RunContextWrapper, Runner
from db.connection import get_db_connection_context

logger = logging.getLogger(__name__)

# ===============================================================================
# ORCHESTRATION TYPES
# ===============================================================================

class ConflictContext(BaseModel):
    """Context data for conflict operations."""
    model_config = ConfigDict(extra="ignore")

    # Scene-related fields
    scene_type: Optional[str] = None
    scene_description: Optional[str] = None
    activity: Optional[str] = None
    activity_type: Optional[str] = None
    location: Optional[str] = None
    location_id: Optional[int] = None
    timestamp: Optional[str] = None  # ISO-8601 preferred

    # Participant fields (various names used across modules)
    participants: Optional[List[int]] = None
    present_npcs: Optional[List[int]] = None
    npcs: Optional[List[int]] = None
    character_ids: Optional[List[int]] = None
    stakeholders: Optional[List[int]] = None

    # Conflict-specific fields
    conflict_type: Optional[str] = None
    intensity: Optional[str] = None  # e.g., "tension", "friction"
    intensity_level: Optional[float] = Field(None, ge=0.0, le=1.0)
    description: Optional[str] = None
    phase: Optional[str] = None

    # Context and history
    recent_events: Optional[List[str]] = None
    evidence: Optional[List[str]] = None
    tension_source: Optional[str] = None

    # Template and generation fields
    use_template: Optional[bool] = None
    template_id: Optional[int] = None
    generation_data: Optional[Dict[str, Any]] = None
    hooks: Optional[List[str]] = None
    complexity: Optional[float] = Field(None, ge=0.0, le=1.0)

    # Integration mode and processing flags
    integration_mode: Optional[str] = None
    integrating_conflicts: Optional[bool] = None
    boost_engagement: Optional[bool] = None

    # Flexible metadata for additional fields
    metadata: Optional[Dict[str, Any]] = None


class ConflictCreationResponse(BaseModel):
    """Response model for conflict creation."""
    model_config = ConfigDict(extra="ignore")

    conflict_id: int = Field(..., ge=0)
    conflict_type: Optional[str] = None
    conflict_name: Optional[str] = None
    status: str
    message: Optional[str] = None
    created_at: Optional[str] = None  # ISO-8601

    # Phase and flow information
    initial_phase: Optional[str] = None
    pacing_style: Optional[str] = None

    # Stakeholder information
    stakeholders_created: Optional[int] = None
    stakeholder_ids: Optional[List[int]] = None

    # Template information
    template_used: Optional[int] = None
    generated_conflict: Optional[int] = None
    narrative_hooks: Optional[List[str]] = None

    # Additional details
    conflict_details: Optional[Dict[str, Any]] = None
    subsystem_responses: Optional[Dict[str, Any]] = None


class SceneContext(BaseModel):
    """Context for scene processing."""
    model_config = ConfigDict(extra="ignore")

    # Identification
    scene_id: int
    scene_type: str

    # Participants & place
    characters_present: List[int]
    location_id: int

    # Optional timing & linkage
    timestamp: Optional[str] = None  # ISO-8601
    previous_scene_id: Optional[int] = None

    # Optional content
    action_sequence: Optional[List[str]] = None
    dialogue: Optional[List[Dict[str, str]]] = None


class SceneProcessingResponse(BaseModel):
    """Response from scene processing."""
    model_config = ConfigDict(extra="ignore")

    scene_id: Optional[int] = None
    processed: bool

    # Conflict manifestations
    conflicts_active: Optional[bool] = None
    conflicts_detected: Optional[List[int]] = None
    manifestations: Optional[List[str]] = None

    # Events and tensions
    events_triggered: Optional[List[str]] = None
    tensions_detected: Optional[int] = None
    slice_of_life_active: Optional[bool] = None

    # Suggestions and choices
    next_scene_suggestions: Optional[List[str]] = None
    player_choices: Optional[List[Dict[str, Any]]] = None
    choices: Optional[List[Dict[str, Any]]] = None

    # NPC behaviors and reactions
    npc_behaviors: Optional[Dict[str, Any]] = None
    npc_reactions: Optional[Dict[str, Any]] = None

    # Environmental and atmospheric elements
    atmospheric_elements: Optional[List[str]] = None
    atmosphere: Optional[List[str]] = None
    environmental_cues: Optional[List[str]] = None

    # State changes
    state_changes: Optional[Dict[str, Any]] = None

    # Subsystem data
    subsystem_data: Optional[Dict[str, Any]] = None

    # Experience quality metrics
    experience_quality: Optional[float] = None
    recommended_mode_change: Optional[str] = None


class ConflictResolutionResponse(BaseModel):
    """Response from conflict resolution."""
    model_config = ConfigDict(extra="ignore")

    conflict_id: int = Field(..., ge=0)
    resolved: bool
    resolution_type: str
    outcome: Optional[str] = None

    # Victory and achievement information
    victory_achieved: Optional[bool] = None
    achievements: Optional[List[Dict[str, Any]]] = None
    partial_victories: Optional[List[Dict[str, Any]]] = None

    # Impacts and consequences
    impacts: Optional[List[str]] = None
    consequences: Optional[Dict[str, Any]] = None
    immediate_consequences: Optional[Dict[str, Any]] = None
    long_term_consequences: Optional[Dict[str, Any]] = None

    # New conflicts or tensions
    new_conflicts_created: Optional[List[int]] = None

    # Narrative elements
    epilogue: Optional[str] = None
    resolution_narrative: Optional[str] = None
    legacy: Optional[str] = None

    # Canon and precedent
    became_canonical: Optional[bool] = None
    canonical_event: Optional[int] = None

    # Additional resolution data
    resolution_data: Optional[Dict[str, Any]] = None


# For complex nested structures, create specific models
class ConflictInfo(BaseModel):
    """Individual conflict information."""
    model_config = ConfigDict(extra="ignore")

    id: int
    type: str
    status: str
    created_at: str  # ISO-8601
    severity: Optional[float] = None


class EventInfo(BaseModel):
    """Individual event information."""
    model_config = ConfigDict(extra="ignore")

    id: int
    type: str
    status: str
    scheduled_at: Optional[str] = None  # ISO-8601


class SystemStateResponse(BaseModel):
    """Complete system state response."""
    model_config = ConfigDict(extra="ignore")

    active_conflicts: List[ConflictInfo]
    pending_events: List[EventInfo]
    system_health: str
    total_conflicts: int
    total_resolutions: int
    subsystem_states: Dict[str, str]
    last_update: str  # ISO-8601

class ConflictContextDTO(TypedDict, total=False):
    # Scene-related
    scene_type: str
    scene_description: str
    activity: str
    activity_type: str
    location: str
    location_id: int
    timestamp: str  # ISO-8601

    # Participants
    participants: List[int]
    present_npcs: List[int]
    npcs: List[int]
    character_ids: List[int]
    stakeholders: List[int]

    # Conflict specifics
    conflict_type: str
    intensity: str
    intensity_level: float
    description: str
    phase: str

    # Context & history
    recent_events: List[str]
    evidence: List[str]
    tension_source: str

    # Template / generation
    use_template: bool
    template_id: int
    generation_data: Dict[str, Any]
    hooks: List[str]
    complexity: float  # 0..1

    # Processing flags
    integration_mode: str
    integrating_conflicts: bool
    boost_engagement: bool

    # Flexible metadata
    metadata: Dict[str, Any]


class SceneContextDTO(TypedDict, total=False):
    scene_id: int
    scene_type: str
    characters_present: List[int]
    location_id: int
    timestamp: NotRequired[str]            # ISO-8601
    previous_scene_id: NotRequired[int]
    action_sequence: NotRequired[List[str]]
    dialogue: NotRequired[List[Dict[str, str]]]

class SubsystemType(Enum):
    TENSION = "tension"
    STAKEHOLDER = "stakeholder"
    FLOW = "flow"
    SOCIAL = "social"
    LEVERAGE = "leverage"
    MULTIPARTY = "multiparty"
    BACKGROUND = "background"
    VICTORY = "victory"
    CANON = "canon"
    TEMPLATE = "template"
    EDGE_HANDLER = "edge_handler"
    SLICE_OF_LIFE = "slice_of_life"
    DETECTION = "detection"
    RESOLUTION = "resolution"

class EventType(Enum):
    CONFLICT_CREATED = "conflict_created"
    CONFLICT_UPDATED = "conflict_updated"
    CONFLICT_RESOLVED = "conflict_resolved"
    TENSION_CHANGED = "tension_changed"
    PHASE_TRANSITION = "phase_transition"
    INTENSITY_CHANGED = "intensity_changed"
    STAKEHOLDER_ACTION = "stakeholder_action"
    PLAYER_CHOICE = "player_choice"
    NPC_REACTION = "npc_reaction"
    EDGE_CASE_DETECTED = "edge_case_detected"
    CANON_ESTABLISHED = "canon_established"
    TEMPLATE_GENERATED = "template_generated"
    HEALTH_CHECK = "health_check"
    STATE_SYNC = "state_sync"

@dataclass
class SystemEvent:
    event_id: str
    event_type: EventType
    source_subsystem: SubsystemType
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    target_subsystems: Optional[Set[SubsystemType]] = None
    requires_response: bool = False
    priority: int = 5  # 1-10, 1 is highest

@dataclass
class SubsystemResponse:
    subsystem: SubsystemType
    event_id: str
    success: bool
    data: Dict[str, Any]
    side_effects: List[SystemEvent] = field(default_factory=list)

# ===============================================================================
# SUBSYSTEM INTERFACE
# ===============================================================================

class ConflictSubsystem:
    """Base interface all subsystems must implement"""
    
    @property
    def subsystem_type(self) -> SubsystemType:
        """Return the subsystem type"""
        raise NotImplementedError
    
    @property
    def capabilities(self) -> Set[str]:
        """Return capabilities this subsystem provides"""
        raise NotImplementedError
    
    @property
    def dependencies(self) -> Set[SubsystemType]:
        """Return other subsystems this depends on"""
        return set()
    
    @property
    def event_subscriptions(self) -> Set[EventType]:
        """Return events this subsystem wants to receive"""
        return set()
    
    async def initialize(self, synthesizer: 'ConflictSynthesizer') -> bool:
        """Initialize the subsystem with synthesizer reference"""
        self.synthesizer = weakref.ref(synthesizer)
        return True
    
    async def handle_event(self, event: SystemEvent) -> SubsystemResponse:
        """Handle an event from the synthesizer"""
        return SubsystemResponse(
            subsystem=self.subsystem_type,
            event_id=event.event_id,
            success=True,
            data={}
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """Return health status of the subsystem"""
        return {'healthy': True}

# ===============================================================================
# THE MASTER CONFLICT SYNTHESIZER
# ===============================================================================

class ConflictSynthesizer:
    """
    THE central orchestrator for all conflict subsystems.
    All conflict operations flow through this synthesizer.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        
        # Subsystem registry
        self._subsystems: Dict[SubsystemType, ConflictSubsystem] = {}
        self._capabilities_map: Dict[str, List[SubsystemType]] = defaultdict(list)
        
        # Event system
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._event_handlers: Dict[EventType, List[SubsystemType]] = defaultdict(list)
        self._event_history: List[SystemEvent] = []
        self._pending_responses: Dict[str, List[SubsystemResponse]] = defaultdict(list)
        
        # State management
        self._conflict_states: Dict[int, Dict[str, Any]] = {}
        self._global_metrics = {
            'total_conflicts': 0,
            'active_conflicts': 0,
            'resolved_conflicts': 0,
            'system_health': 1.0,
            'complexity_score': 0.0
        }
        
        # Processing control
        self._processing = False
        self._shutdown = False
        
        # Orchestration agents (lazy loaded)
        self._orchestrator = None
        self._state_manager = None
        self._event_router = None
        self._health_monitor = None
    
    # ========== Subsystem Registration ==========
    
    async def register_subsystem(self, subsystem: ConflictSubsystem) -> bool:
        """Register a subsystem with the synthesizer"""
        try:
            # Initialize subsystem
            if not await subsystem.initialize(self):
                logger.error(f"Failed to initialize {subsystem.subsystem_type}")
                return False
            
            # Register in main registry
            self._subsystems[subsystem.subsystem_type] = subsystem
            
            # Register capabilities
            for capability in subsystem.capabilities:
                self._capabilities_map[capability].append(subsystem.subsystem_type)
            
            # Register event subscriptions
            for event_type in subsystem.event_subscriptions:
                self._event_handlers[event_type].append(subsystem.subsystem_type)
            
            logger.info(f"Registered subsystem: {subsystem.subsystem_type.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering subsystem: {e}")
            return False
    
    async def initialize_all_subsystems(self):
        """Initialize all standard subsystems"""
        from logic.conflict_system.tension import TensionSystemAdapter
        from logic.conflict_system.stakeholder_adapter import StakeholderSystemAdapter
        from logic.conflict_system.flow_adapter import FlowSystemAdapter
        from logic.conflict_system.social_adapter import SocialSystemAdapter
        from logic.conflict_system.leverage_adapter import LeverageSystemAdapter
        from logic.conflict_system.background_adapter import BackgroundSystemAdapter
        from logic.conflict_system.victory_adapter import VictorySystemAdapter
        from logic.conflict_system.canon_adapter import CanonSystemAdapter
        from logic.conflict_system.template_adapter import TemplateSystemAdapter
        from logic.conflict_system.edge_adapter import EdgeHandlerAdapter
        from logic.conflict_system.slice_adapter import SliceOfLifeAdapter
        
        # Create and register all subsystems
        subsystems = [
            TensionSystemAdapter(self.user_id, self.conversation_id),
            StakeholderSystemAdapter(self.user_id, self.conversation_id),
            FlowSystemAdapter(self.user_id, self.conversation_id),
            SocialSystemAdapter(self.user_id, self.conversation_id),
            LeverageSystemAdapter(self.user_id, self.conversation_id),
            BackgroundSystemAdapter(self.user_id, self.conversation_id),
            VictorySystemAdapter(self.user_id, self.conversation_id),
            CanonSystemAdapter(self.user_id, self.conversation_id),
            TemplateSystemAdapter(self.user_id, self.conversation_id),
            EdgeHandlerAdapter(self.user_id, self.conversation_id),
            SliceOfLifeAdapter(self.user_id, self.conversation_id),
        ]
        
        for subsystem in subsystems:
            await self.register_subsystem(subsystem)
        
        # Start event processing
        if not self._processing:
            asyncio.create_task(self._process_events())
    
    # ========== Event System ==========
    
    async def emit_event(self, event: SystemEvent) -> Optional[List[SubsystemResponse]]:
        """Emit an event to relevant subsystems"""
        # Add to history
        self._event_history.append(event)
        
        # Add to queue for processing
        await self._event_queue.put(event)
        
        # If synchronous response needed
        if event.requires_response:
            return await self._process_event_sync(event)
        
        return None
    
    async def _process_events(self):
        """Main event processing loop"""
        self._processing = True
        
        while not self._shutdown:
            try:
                # Get next event with timeout
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=1.0
                )
                
                # Process the event
                await self._route_event(event)
                
            except asyncio.TimeoutError:
                # Periodic health check
                await self._perform_health_check()
            except Exception as e:
                logger.error(f"Error processing event: {e}")
        
        self._processing = False
    
    async def _route_event(self, event: SystemEvent):
        """Route an event to appropriate subsystems"""
        # Determine target subsystems
        if event.target_subsystems:
            targets = event.target_subsystems
        else:
            targets = set(self._event_handlers.get(event.event_type, []))
        
        # Send to each target
        responses = []
        for subsystem_type in targets:
            if subsystem_type in self._subsystems:
                try:
                    response = await self._subsystems[subsystem_type].handle_event(event)
                    responses.append(response)
                    
                    # Process any side effects
                    for side_effect in response.side_effects:
                        await self.emit_event(side_effect)
                        
                except Exception as e:
                    logger.error(f"Error in {subsystem_type} handling event: {e}")
        
        # Store responses if needed
        if event.requires_response:
            self._pending_responses[event.event_id] = responses
    
    async def _process_event_sync(self, event: SystemEvent) -> List[SubsystemResponse]:
        """Process an event synchronously and return responses"""
        await self._route_event(event)
        
        # Wait for responses (with timeout)
        await asyncio.sleep(0.1)  # Allow processing
        
        return self._pending_responses.get(event.event_id, [])
    
    # ========== Conflict Operations ==========
    
    async def create_conflict(
        self,
        conflict_type: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a new conflict through the appropriate subsystems"""
        
        # Generate unique ID
        import uuid
        operation_id = str(uuid.uuid4())
        
        # Create canonical context for lore system
        from lore.core.context import CanonicalContext
        from lore.core.canon import log_canonical_event
        ctx = CanonicalContext(self.user_id, self.conversation_id)
        
        # Determine which subsystems to involve
        involved_subsystems = await self._determine_subsystems_for_operation(
            'create_conflict', conflict_type, context
        )
        
        # Create the conflict event
        event = SystemEvent(
            event_id=operation_id,
            event_type=EventType.CONFLICT_CREATED,
            source_subsystem=SubsystemType.SLICE_OF_LIFE,  # Default
            payload={
                'conflict_type': conflict_type,
                'context': context
            },
            target_subsystems=involved_subsystems,
            requires_response=True,
            priority=1
        )
        
        # Emit and wait for responses
        responses = await self.emit_event(event)
        
        # Aggregate responses
        conflict_data = self._aggregate_conflict_creation(responses)
        
        # Update state
        if conflict_data.get('conflict_id'):
            self._conflict_states[conflict_data['conflict_id']] = conflict_data
            self._global_metrics['total_conflicts'] += 1
            self._global_metrics['active_conflicts'] += 1
            
            # Log the conflict creation canonically using the core canon system
            async with get_db_connection_context() as conn:
                # Extract meaningful details from context
                participants = context.get('participants', [])
                location = context.get('location', 'Unknown location')
                description = context.get('description', f'A {conflict_type} conflict')
                intensity = context.get('intensity', 'moderate')
                
                # Build a descriptive event text
                event_text = f"New {conflict_type} conflict initiated"
                if participants:
                    if len(participants) == 1:
                        event_text += f" involving {participants[0]}"
                    elif len(participants) == 2:
                        event_text += f" between {participants[0]} and {participants[1]}"
                    else:
                        event_text += f" involving {len(participants)} parties"
                
                if location != 'Unknown location':
                    event_text += f" at {location}"
                
                event_text += f": {description}"
                
                # Determine significance based on conflict type and intensity
                significance_map = {
                    'political': 7,
                    'social': 5,
                    'personal': 4,
                    'faction': 8,
                    'power': 8,
                    'slice': 3,
                    'background': 3,
                    'multiparty': 7
                }
                base_significance = significance_map.get(conflict_type.lower(), 5)
                
                # Adjust for intensity
                intensity_modifiers = {
                    'subtle': -1,
                    'moderate': 0,
                    'high': 1,
                    'extreme': 2
                }
                significance = min(10, max(1, base_significance + intensity_modifiers.get(intensity, 0)))
                
                # Log the canonical event
                await log_canonical_event(
                    ctx, conn,
                    event_text,
                    tags=[
                        'conflict',
                        'creation',
                        conflict_type,
                        intensity,
                        f"conflict_id_{conflict_data['conflict_id']}"
                    ],
                    significance=significance
                )
                
                # If this conflict involves NPCs, log their involvement
                if 'npc_ids' in context:
                    for npc_id in context['npc_ids']:
                        # Get NPC name if possible
                        npc = await conn.fetchrow(
                            "SELECT npc_name FROM NPCStats WHERE npc_id = $1",
                            npc_id
                        )
                        npc_name = npc['npc_name'] if npc else f"NPC {npc_id}"
                        
                        await log_canonical_event(
                            ctx, conn,
                            f"{npc_name} became involved in {conflict_type} conflict",
                            tags=[
                                'npc_involvement',
                                'conflict',
                                f"npc_{npc_id}",
                                f"conflict_id_{conflict_data['conflict_id']}"
                            ],
                            significance=3
                        )
                
                # Check if this conflict should reference historical precedents
                if conflict_type in ['political', 'faction', 'power']:
                    # Look for similar past conflicts
                    similar_conflicts = await conn.fetch("""
                        SELECT event_text, tags, significance
                        FROM CanonicalEvents
                        WHERE user_id = $1 AND conversation_id = $2
                        AND 'conflict' = ANY(tags)
                        AND $3 = ANY(tags)
                        AND significance >= 7
                        ORDER BY timestamp DESC
                        LIMIT 3
                    """, self.user_id, self.conversation_id, conflict_type)
                    
                    if similar_conflicts:
                        # Store references to precedents
                        conflict_data['historical_precedents'] = [
                            dict(p) for p in similar_conflicts
                        ]
                        
                        # Update the conflict with precedent awareness
                        precedent_event = SystemEvent(
                            event_id=f"precedent_{operation_id}",
                            event_type=EventType.STATE_SYNC,
                            source_subsystem=SubsystemType.CANON,
                            payload={
                                'conflict_id': conflict_data['conflict_id'],
                                'precedents': conflict_data['historical_precedents'],
                                'message': 'Historical precedents identified for this conflict'
                            },
                            target_subsystems={SubsystemType.CANON},
                            requires_response=False,
                            priority=5
                        )
                        await self.emit_event(precedent_event)
            
            # Update complexity score based on conflict creation
            self._global_metrics['complexity_score'] = min(
                1.0,
                self._global_metrics['complexity_score'] + 0.1
            )
        
        return conflict_data
    
    async def resolve_conflict(
        self,
        conflict_id: int,
        resolution_type: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Resolve a conflict through appropriate subsystems"""
        
        # Get conflict state
        conflict_state = self._conflict_states.get(conflict_id, {})
        
        # Create canonical context
        ctx = CanonicalContext(self.user_id, self.conversation_id)
        
        # Create resolution event
        event = SystemEvent(
            event_id=f"resolve_{conflict_id}",
            event_type=EventType.CONFLICT_RESOLVED,
            source_subsystem=SubsystemType.RESOLUTION,
            payload={
                'conflict_id': conflict_id,
                'resolution_type': resolution_type,
                'context': context,
                'conflict_state': conflict_state
            },
            requires_response=True,
            priority=2
        )
        
        # Process resolution
        responses = await self.emit_event(event)
        result = self._aggregate_resolution_result(responses)
        
        # Log the resolution canonically
        if result['resolved']:
            async with get_db_connection_context() as conn:
                conflict_name = conflict_state.get('conflict_name', f'Conflict {conflict_id}')
                await log_canonical_event(
                    ctx, conn,
                    f"Conflict resolved: {conflict_name} through {resolution_type}",
                    tags=['conflict', 'resolution', resolution_type],
                    significance=7
                )
                
                # Log consequences as canonical events if significant
                for consequence in result.get('consequences', []):
                    if consequence.get('significance', 0) >= 5:
                        await log_canonical_event(
                            ctx, conn,
                            f"Consequence of {conflict_name}: {consequence.get('description', '')}",
                            tags=['conflict_consequence', 'ripple_effect'],
                            significance=consequence.get('significance', 5)
                        )
        
        # Update metrics
        self._global_metrics['active_conflicts'] -= 1
        self._global_metrics['resolved_conflicts'] += 1
        
        # Clean up state
        if conflict_id in self._conflict_states:
            del self._conflict_states[conflict_id]
        
        return result
    
    # ========== State Management ==========
    
    async def get_conflict_state(self, conflict_id: int) -> Dict[str, Any]:
        """Get comprehensive state of a conflict"""
        
        if conflict_id not in self._conflict_states:
            # Load from database if not in memory
            await self._load_conflict_state(conflict_id)
        
        state = self._conflict_states.get(conflict_id, {})
        
        # Enrich with subsystem data
        for subsystem_type, subsystem in self._subsystems.items():
            if hasattr(subsystem, 'get_conflict_data'):
                subsystem_data = await subsystem.get_conflict_data(conflict_id)
                state[subsystem_type.value] = subsystem_data
        
        return state
    
    async def get_system_state(self) -> Dict[str, Any]:
        """Get overall system state"""
        
        state = {
            'metrics': self._global_metrics.copy(),
            'active_conflicts': list(self._conflict_states.keys()),
            'subsystems': {},
            'health': {}
        }
        
        # Get state from each subsystem
        for subsystem_type, subsystem in self._subsystems.items():
            health = await subsystem.health_check()
            state['health'][subsystem_type.value] = health
            
            if hasattr(subsystem, 'get_state'):
                state['subsystems'][subsystem_type.value] = await subsystem.get_state()
        
        return state
    
    # ========== Health Monitoring ==========
    
    async def _perform_health_check(self):
        """Perform system-wide health check"""
        
        health_scores = []
        issues = []
        
        for subsystem_type, subsystem in self._subsystems.items():
            try:
                health = await subsystem.health_check()
                if health.get('healthy', True):
                    health_scores.append(1.0)
                else:
                    health_scores.append(0.5)
                    issues.append({
                        'subsystem': subsystem_type.value,
                        'issue': health.get('issue', 'Unknown')
                    })
            except Exception as e:
                health_scores.append(0.0)
                issues.append({
                    'subsystem': subsystem_type.value,
                    'issue': str(e)
                })
        
        # Calculate overall health
        if health_scores:
            self._global_metrics['system_health'] = sum(health_scores) / len(health_scores)
        
        # Handle critical issues
        if self._global_metrics['system_health'] < 0.5 and issues:
            await self._handle_health_issues(issues)
    
    async def _handle_health_issues(self, issues: List[Dict[str, Any]]):
        """Handle detected health issues"""
        
        # Emit edge case event
        event = SystemEvent(
            event_id=f"health_{datetime.now().timestamp()}",
            event_type=EventType.EDGE_CASE_DETECTED,
            source_subsystem=SubsystemType.EDGE_HANDLER,
            payload={'issues': issues},
            target_subsystems={SubsystemType.EDGE_HANDLER},
            priority=1
        )
        
        await self.emit_event(event)
    
    async def _emergency_recovery(self):
        """Perform emergency recovery when system health is critical"""
        
        logger.warning("Emergency recovery initiated")
        
        # Reduce active conflicts
        if self._global_metrics['active_conflicts'] > 5:
            # Auto-resolve some conflicts
            conflicts_to_resolve = list(self._conflict_states.keys())[:3]
            for conflict_id in conflicts_to_resolve:
                await self.resolve_conflict(
                    conflict_id,
                    'emergency_resolution',
                    {'reason': 'system_overload'}
                )
        
        # Reset complexity
        self._global_metrics['complexity_score'] = 0.5
        
        # Clear event queue if too large
        if self._event_queue.qsize() > 100:
            self._event_queue = asyncio.Queue()

    async def process_scene(self, scene_context: Dict[str, Any]) -> Dict[str, Any]:
        """Route a scene through relevant subsystems and synthesize a result."""
        active = await self._determine_active_subsystems(scene_context)

        # Build a routing event (STATE_SYNC is a reasonable generic event type)
        event = SystemEvent(
            event_id=f"scene_{datetime.now().timestamp()}",
            event_type=EventType.STATE_SYNC,
            source_subsystem=SubsystemType.SLICE_OF_LIFE,
            payload={"scene_context": scene_context},
            target_subsystems=active,
            requires_response=True,
            priority=5,
        )

        responses = await self.emit_event(event)
        return self._synthesize_scene_result(responses, scene_context)
    
    # ========== Helper Methods ==========
    
    async def _determine_subsystems_for_operation(
        self,
        operation: str,
        conflict_type: str,
        context: Dict[str, Any]
    ) -> Set[SubsystemType]:
        """Determine which subsystems should handle an operation"""
        
        # Use orchestrator agent if available
        if self.orchestrator:
            return await self._determine_with_llm(operation, conflict_type, context)
        
        # Fallback to rule-based
        subsystems = set()
        
        # Always include edge handler for safety
        subsystems.add(SubsystemType.EDGE_HANDLER)
        
        # Based on conflict type
        if 'slice' in conflict_type.lower():
            subsystems.add(SubsystemType.SLICE_OF_LIFE)
        if 'social' in conflict_type.lower():
            subsystems.add(SubsystemType.SOCIAL)
        if 'power' in conflict_type.lower():
            subsystems.add(SubsystemType.LEVERAGE)
        
        # Always include these for any conflict
        subsystems.update({
            SubsystemType.TENSION,
            SubsystemType.FLOW,
            SubsystemType.STAKEHOLDER
        })
        
        return subsystems
    
    async def _determine_active_subsystems(
        self,
        scene_context: Dict[str, Any]
    ) -> Set[SubsystemType]:
        """Determine which subsystems should process a scene"""
        
        active = set()
        
        # Check each subsystem's relevance
        for subsystem_type, subsystem in self._subsystems.items():
            if hasattr(subsystem, 'is_relevant_to_scene'):
                if await subsystem.is_relevant_to_scene(scene_context):
                    active.add(subsystem_type)
            else:
                # Default: include if has active conflicts
                if self._global_metrics['active_conflicts'] > 0:
                    active.add(subsystem_type)
        
        return active
    
    def _aggregate_conflict_creation(
        self,
        responses: List[SubsystemResponse]
    ) -> Dict[str, Any]:
        """Aggregate responses from conflict creation"""
        
        result = {
            'success': all(r.success for r in responses),
            'conflict_id': None,
            'subsystem_data': {}
        }
        
        for response in responses:
            # Look for conflict ID
            if 'conflict_id' in response.data:
                result['conflict_id'] = response.data['conflict_id']
            
            # Aggregate subsystem data
            result['subsystem_data'][response.subsystem.value] = response.data
        
        return result
    
    def _synthesize_scene_result(
        self,
        responses: List[SubsystemResponse],
        scene_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synthesize scene processing results"""
        
        result = {
            'scene_processed': True,
            'active_elements': [],
            'manifestations': [],
            'choices': [],
            'state_changes': {}
        }
        
        for response in responses:
            if 'manifestation' in response.data:
                result['manifestations'].append(response.data['manifestation'])
            if 'choices' in response.data:
                result['choices'].extend(response.data['choices'])
            if 'state_change' in response.data:
                result['state_changes'][response.subsystem.value] = response.data['state_change']
        
        return result
    
    def _aggregate_resolution_result(
        self,
        responses: List[SubsystemResponse]
    ) -> Dict[str, Any]:
        """Aggregate resolution responses"""
        
        result = {
            'resolved': all(r.success for r in responses),
            'resolution_data': {},
            'consequences': [],
            'new_patterns': []
        }
        
        for response in responses:
            result['resolution_data'][response.subsystem.value] = response.data
            
            if 'consequences' in response.data:
                result['consequences'].extend(response.data['consequences'])
            if 'patterns' in response.data:
                result['new_patterns'].extend(response.data['patterns'])
        
        return result
    
    async def _load_conflict_state(self, conflict_id: int):
        """Load conflict state from database"""
        
        async with get_db_connection_context() as conn:
            conflict = await conn.fetchrow("""
                SELECT * FROM Conflicts WHERE conflict_id = $1
            """, conflict_id)
            
            if conflict:
                self._conflict_states[conflict_id] = dict(conflict)
    
    # ========== Agent Properties ==========
    
    @property
    def orchestrator(self) -> Agent:
        """Main orchestration agent"""
        if self._orchestrator is None:
            self._orchestrator = Agent(
                name="Master Conflict Orchestrator",
                instructions="""
                You are the master orchestrator of a complex conflict system.
                
                Your role:
                - Determine which subsystems should handle each operation
                - Route events to appropriate subsystems
                - Coordinate responses from multiple subsystems
                - Maintain system coherence and narrative consistency
                - Optimize for performance and player experience
                
                Always consider dependencies between subsystems and ensure
                operations flow in the correct order.
                """,
                model="gpt-5-nano",
            )
        return self._orchestrator
    
    async def _determine_with_llm(
        self,
        operation: str,
        conflict_type: str,
        context: Dict[str, Any]
    ) -> Set[SubsystemType]:
        """Use LLM to determine subsystems"""
        
        prompt = f"""
        Determine which subsystems should handle this operation:
        
        Operation: {operation}
        Conflict Type: {conflict_type}
        Context: {json.dumps(context)}
        
        Available subsystems: {[s.value for s in SubsystemType]}
        
        Return a JSON list of subsystem names that should be involved.
        """
        
        response = await Runner.run(self.orchestrator, prompt)
        try:
            subsystem_names = json.loads(response.output)
            return {SubsystemType(name) for name in subsystem_names if name in SubsystemType._value2member_map_}
        except:
            # Fallback to default set
            return {SubsystemType.TENSION, SubsystemType.FLOW, SubsystemType.STAKEHOLDER}
    
    # ========== Cleanup ==========
    
    async def shutdown(self):
        """Gracefully shutdown the synthesizer"""
        self._shutdown = True
        
        # Wait for event processing to stop
        await asyncio.sleep(0.5)
        
        # Cleanup subsystems
        for subsystem in self._subsystems.values():
            if hasattr(subsystem, 'cleanup'):
                await subsystem.cleanup()
        
        logger.info("Conflict synthesizer shutdown complete")


# ===============================================================================
# PUBLIC API
# ===============================================================================

# Global synthesizer instance (per user/conversation)
_synthesizers: Dict[tuple, ConflictSynthesizer] = {}

async def get_synthesizer(user_id: int, conversation_id: int) -> ConflictSynthesizer:
    """Get or create synthesizer instance"""
    key = (user_id, conversation_id)
    if key not in _synthesizers:
        synthesizer = ConflictSynthesizer(user_id, conversation_id)
        await synthesizer.initialize_all_subsystems()
        _synthesizers[key] = synthesizer
    return _synthesizers[key]

@function_tool
async def orchestrate_conflict_creation(
    ctx: RunContextWrapper,
    conflict_type: str,
    context: ConflictContextDTO,   # <-- TypedDict input
) -> Dict[str, Any]:               # <-- plain dict output
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    synthesizer = await get_synthesizer(user_id, conversation_id)

    # context is already a dict-like
    result = await synthesizer.create_conflict(conflict_type, dict(context))

    # Build response dict (previously you constructed a Pydantic response here)
    response_data = {
        'conflict_id': result.get('conflict_id', 0),
        'status': result.get('status', 'created'),
        'conflict_type': result.get('conflict_type', conflict_type),
        'conflict_name': result.get('conflict_name'),
        'message': result.get('message'),
        'created_at': result.get('created_at'),
        'initial_phase': result.get('initial_phase'),
        'pacing_style': result.get('pacing_style'),
        'stakeholders_created': result.get('stakeholders_created'),
        'stakeholder_ids': result.get('stakeholder_ids'),
        'template_used': result.get('template_used'),
        'generated_conflict': result.get('generated_conflict'),
        'narrative_hooks': result.get('narrative_hooks'),
        'conflict_details': result.get('conflict_details'),
        'subsystem_responses': result.get('subsystem_responses'),
    }
    return {k: v for k, v in response_data.items() if v is not None}

@function_tool
async def orchestrate_scene_processing(
    ctx: RunContextWrapper,
    scene_context: SceneContextDTO,   # <-- TypedDict input
) -> Dict[str, Any]:                  # <-- plain dict output
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    synthesizer = await get_synthesizer(user_id, conversation_id)

    result = await synthesizer.process_scene(dict(scene_context))

    response_data = {
        'scene_id': result.get('scene_id'),
        'processed': result.get('processed', True),
        'conflicts_active': result.get('conflicts_active'),
        'conflicts_detected': result.get('conflicts_detected'),
        'manifestations': result.get('manifestations'),
        'events_triggered': result.get('events_triggered'),
        'tensions_detected': result.get('tensions_detected'),
        'slice_of_life_active': result.get('slice_of_life_active'),
        'next_scene_suggestions': result.get('next_scene_suggestions'),
        'player_choices': result.get('player_choices'),
        'choices': result.get('choices'),
        'npc_behaviors': result.get('npc_behaviors'),
        'npc_reactions': result.get('npc_reactions'),
        'atmospheric_elements': result.get('atmospheric_elements'),
        'atmosphere': result.get('atmosphere'),
        'environmental_cues': result.get('environmental_cues'),
        'state_changes': result.get('state_changes'),
        'subsystem_data': result.get('subsystem_data'),
        'experience_quality': result.get('experience_quality'),
        'recommended_mode_change': result.get('recommended_mode_change'),
    }
    return {k: v for k, v in response_data.items() if v is not None}

@function_tool
async def orchestrate_conflict_resolution(
    ctx: RunContextWrapper,
    conflict_id: int,
    resolution_type: str,
    context: ConflictContextDTO,    # <-- TypedDict input
) -> Dict[str, Any]:                # <-- plain dict output
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')
    synthesizer = await get_synthesizer(user_id, conversation_id)

    result = await synthesizer.resolve_conflict(conflict_id, resolution_type, dict(context))

    response_data = {
        'conflict_id': result.get('conflict_id', conflict_id),
        'resolved': result.get('resolved', True),
        'resolution_type': result.get('resolution_type', resolution_type),
        'outcome': result.get('outcome'),
        'victory_achieved': result.get('victory_achieved'),
        'achievements': result.get('achievements'),
        'partial_victories': result.get('partial_victories'),
        'impacts': result.get('impacts'),
        'consequences': result.get('consequences'),
        'immediate_consequences': result.get('immediate_consequences'),
        'long_term_consequences': result.get('long_term_consequences'),
        'new_conflicts_created': result.get('new_conflicts_created'),
        'epilogue': result.get('epilogue'),
        'resolution_narrative': result.get('resolution_narrative'),
        'legacy': result.get('legacy'),
        'became_canonical': result.get('became_canonical'),
        'canonical_event': result.get('canonical_event'),
        'resolution_data': result.get('resolution_data'),
    }
    return {k: v for k, v in response_data.items() if v is not None}

@function_tool
async def get_orchestrated_system_state(
    ctx: RunContextWrapper
) -> Dict[str, Any]:
    """Get complete system state from orchestrator (strict JSON shape)."""
    user_id = ctx.data.get('user_id')
    conversation_id = ctx.data.get('conversation_id')

    synthesizer = await get_synthesizer(user_id, conversation_id)
    raw = await synthesizer.get_system_state()

    # Derive a friendly health label from the numeric score
    health_score = float(raw.get("metrics", {}).get("system_health", 1.0))
    if health_score >= 0.8:
        health_label = "ok"
    elif health_score >= 0.5:
        health_label = "degraded"
    else:
        health_label = "critical"

    # Shape active_conflicts into ConflictInfo dicts
    now_iso = datetime.utcnow().isoformat()
    active_conflicts: List[Dict[str, Any]] = []
    for item in raw.get("active_conflicts", []):
        if isinstance(item, dict):
            # If the synthesizer later returns rich details, coerce to the expected keys
            active_conflicts.append({
                "id": item.get("id") or item.get("conflict_id"),
                "type": item.get("type") or item.get("conflict_type") or "unknown",
                "status": item.get("status") or "active",
                "created_at": item.get("created_at") or now_iso,
                "severity": item.get("severity"),
            })
        else:
            # Current implementation returns IDs; fabricate minimal record
            active_conflicts.append({
                "id": int(item),
                "type": "unknown",
                "status": "active",
                "created_at": now_iso,
                "severity": None,
            })

    # Subsystem states -> string labels
    subsystem_states = {
        name: ("healthy" if (info or {}).get("healthy", True) else "issue")
        for name, info in raw.get("health", {}).items()
    }

    shaped: Dict[str, Any] = {
        "active_conflicts": active_conflicts,
        "pending_events": [],  # populate when you track a queue externally
        "system_health": health_label,
        "total_conflicts": int(raw.get("metrics", {}).get("total_conflicts", 0)),
        "total_resolutions": int(raw.get("metrics", {}).get("resolved_conflicts", 0)),
        "subsystem_states": subsystem_states,
        "last_update": now_iso,
    }
    return shaped
