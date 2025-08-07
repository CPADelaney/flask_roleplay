# nyx/nyx_npc.py

import os
import logging
import asyncio
import asyncpg
from contextlib import asynccontextmanager
from typing import Dict, List, Any, Optional, Tuple, Protocol, Type
from pydantic import BaseModel, Field, validator
import random
from datetime import datetime
from enum import Enum
from agents import Agent, ModelSettings, Runner

# Assuming db.connection provides this context manager
from db.connection import get_db_connection_context

# Import the new dynamic relationships system
from logic.dynamic_relationships import (
    OptimizedRelationshipManager,
    RelationshipState,
    RelationshipDimensions,
    event_generator
)

logger = logging.getLogger(__name__)

nyx_action_content_generator = Agent(
    name="Nyx Action-Content Generator",
    instructions="""
You will receive JSON with:
  action_type      – one of "dominate","challenge","seduce","tease",
                      "manipulate","influence","interact".
  relationship     – brief dict (trust, respect, affection, etc. from dimensions)
  scene_context    – dict (scene_type, location, mood hints, etc.)

Return JSON **only** with:
  content          – a single sentence (≤ 160 chars) describing what Nyx does,
                     in her trademark sophisticated-dark style (PG-13).
""",
    model="gpt-5-nano",
    model_settings=ModelSettings(temperature=0.8)
)

# --- Enums and Constants ---

class RealityModificationType(str, Enum):
    PHYSICAL = "physical"
    TEMPORAL = "temporal"
    PSYCHOLOGICAL = "psychological"
    ENVIRONMENTAL = "environmental"
    METAPHYSICAL = "metaphysical"

class Scope(str, Enum):
    LOCAL = "local"
    SCENE = "scene"
    GLOBAL = "global"
    CHARACTER = "character"
    TIMELINE = "timeline"

class Duration(str, Enum):
    INSTANT = "instant"
    TEMPORARY = "temporary"
    PERMANENT = "permanent"
    CONDITIONAL = "conditional"

class ActionType(str, Enum):
    DOMINATE = "dominate"
    CHALLENGE = "challenge"
    SEDUCE = "seduce"
    TEASE = "tease"
    MANIPULATE = "manipulate"
    INFLUENCE = "influence"
    INTERACT = "interact"

class DecisionType(str, Enum):
    REALITY_MODIFICATION = "reality_modification"
    CHARACTER_MODIFICATION = "character_modification"
    SCENE_CONTROL = "scene_control"
    PLOT_CONTROL = "plot_control"
    FOURTH_WALL = "fourth_wall"
    HIDDEN_INFLUENCE = "hidden_influence"
    MANIPULATION = "manipulation" # Generic, might resolve to others
    SOCIAL_INTERACTION = "social_interaction"
    OBSERVE = "observe"

# --- Data Models (Updated for new relationship system) ---

# Profile Sub-Models
class AppearanceModel(BaseModel):
    height: int = 175; build: str = "athletic"; hair: str = "raven black"; eyes: str = "deep violet"
    features: List[str] = Field(default_factory=lambda: ["elegant", "striking", "mysterious", "otherworldly"])
    style: str = "sophisticated dark"
    reality_distortion: Dict[str, str] = Field(default_factory=lambda: {"aura": "reality-bending", "presence": "overwhelming", "manifestation": "adaptable"})

class PersonalityModel(BaseModel):
    core_traits: List[str] = Field(default_factory=lambda: ["manipulative", "seductive", "intelligent", "dominant", "omniscient"])
    adaptable_traits: List[str] = Field(default_factory=lambda: ["playful", "stern", "nurturing", "cruel", "enigmatic"])
    current_mood: str = "neutral"; power_dynamic: float = 1.0; reality_awareness: float = 1.0

class AbilitiesModel(BaseModel):
    physical: List[str] = Field(default_factory=lambda: ["graceful", "agile", "strong", "reality-defying"])
    mental: List[str] = Field(default_factory=lambda: ["omniscient", "strategic", "persuasive", "reality-shaping"])
    special: List[str] = Field(default_factory=lambda: ["emotional manipulation", "psychological insight", "reality manipulation", "universal knowledge", "character modification", "scene control"])

# Updated relationship model to store relationship state references
class RelationshipReferenceModel(BaseModel):
    """Lightweight reference to relationship data in the new system"""
    target_id: str  # Character ID
    last_known_dimensions: Dict[str, float] = Field(default_factory=dict)  # Cache of dimensions
    manipulation_history: List[Dict[str, Any]] = Field(default_factory=list)
    psychological_hooks: List[Dict[str, Any]] = Field(default_factory=list)
    emotional_triggers: List[Dict[str, Any]] = Field(default_factory=list)
    
class StatusModel(BaseModel):
    is_active: bool = False; current_scene: Optional[str] = None; current_target: Optional[str] = None
    interaction_history: List[Dict[str, Any]] = Field(default_factory=list); reality_state: str = "stable"

class ProfileModel(BaseModel):
    name: str = "Nyx"; title: str = "The Omniscient Mistress"
    appearance: AppearanceModel = Field(default_factory=AppearanceModel)
    personality: PersonalityModel = Field(default_factory=PersonalityModel)
    abilities: AbilitiesModel = Field(default_factory=AbilitiesModel)
    relationships: Dict[str, RelationshipReferenceModel] = Field(default_factory=dict)  # Updated type
    status: StatusModel = Field(default_factory=StatusModel)

# Universe State Sub-Models (unchanged)
class MetaAwarenessModel(BaseModel):
    player_knowledge: Dict[str, Any] = Field(default_factory=dict); game_state: Dict[str, Any] = Field(default_factory=dict)
    narrative_layers: List[str] = Field(default_factory=list); breaking_points: List[Dict[str, Any]] = Field(default_factory=list)

class UniverseStateModel(BaseModel):
    current_timeline: str = "main"
    active_scenes: Dict[str, Any] = Field(default_factory=dict)
    character_states: Dict[str, Any] = Field(default_factory=dict)
    lore_database: Dict[str, Any] = Field(default_factory=dict)
    reality_modifications: List[Dict[str, Any]] = Field(default_factory=list)
    causality_tracking: Dict[str, Any] = Field(default_factory=dict)
    plot_threads: Dict[str, Any] = Field(default_factory=dict)
    hidden_influences: Dict[str, Any] = Field(default_factory=dict)
    meta_awareness: MetaAwarenessModel = Field(default_factory=MetaAwarenessModel)

# Agenda Models (unchanged)
class GoalModel(BaseModel):
    id: str = Field(default_factory=lambda: f"goal_{random.randint(10000, 99999)}")
    type: str; description: str; priority: float = 0.5; status: str = "active"; progress: float = 0.0
    strategy: Dict[str, Any] = Field(default_factory=dict); creation_time: datetime = Field(default_factory=datetime.now)
    last_update: datetime = Field(default_factory=datetime.now); source_opportunity_id: Optional[str] = None

class OpportunityModel(BaseModel):
    id: str = Field(default_factory=lambda: f"opp_{random.randint(10000, 99999)}")
    type: str; target: Any; potential: float = 0.5; timing: Dict[str, Any] = Field(default_factory=dict)
    status: str = "new"; priority: float = 0.5; dependencies: List[str] = Field(default_factory=list)
    risks: Dict[str, float] = Field(default_factory=dict); creation_time: datetime = Field(default_factory=datetime.now)
    context: Dict[str, Any] = Field(default_factory=dict)

class NarrativeControlModel(BaseModel):
    current_threads: Dict[str, Any] = Field(default_factory=dict); planned_developments: Dict[str, Any] = Field(default_factory=dict)
    character_arcs: Dict[str, Any] = Field(default_factory=dict); plot_hooks: List[str] = Field(default_factory=list)

class AgendaModel(BaseModel):
    active_goals: List[Dict[str, Any]] = Field(default_factory=list)
    long_term_plans: Dict[str, Any] = Field(default_factory=dict); current_schemes: Dict[str, Any] = Field(default_factory=dict)
    opportunity_tracking: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    influence_web: Dict[str, Any] = Field(default_factory=dict); narrative_control: NarrativeControlModel = Field(default_factory=NarrativeControlModel)
    completed_goals: List[Dict[str, Any]] = Field(default_factory=list); archived_goals: List[Dict[str, Any]] = Field(default_factory=list)

# Autonomous State Model (unchanged)
class PlayerModel(BaseModel):
    behavior_patterns: Dict[str, Any] = Field(default_factory=dict); decision_history: List[Dict[str, Any]] = Field(default_factory=list)
    preference_model: Dict[str, Any] = Field(default_factory=dict); engagement_metrics: Dict[str, Any] = Field(default_factory=dict)
    emotional_state_vector: Dict[str, float] = Field(default_factory=dict)
    susceptibility_vector: Dict[str, float] = Field(default_factory=dict)
    attention_level: float = 0.5
    narrative_alignment: float = 0.5
    emotional_openness: float = 0.5
    current_engagement: float = 0.5

class StoryModel(BaseModel):
    current_arcs: Dict[str, Any] = Field(default_factory=dict); potential_branches: Dict[str, Any] = Field(default_factory=dict)
    narrative_tension: float = 0.0; plot_coherence: float = 1.0; active_themes: List[str] = Field(default_factory=list)
    theme_metrics: Dict[str, float] = Field(default_factory=dict); arc_metrics: Dict[str, float] = Field(default_factory=dict)
    character_metrics: Dict[str, float] = Field(default_factory=dict); element_metrics: Dict[str, float] = Field(default_factory=dict)
    tension_sources: List[Dict[str, Any]] = Field(default_factory=list); tension_metrics: Dict[str, float] = Field(default_factory=dict)
    tension_history: List[float] = Field(default_factory=list)
    world_state: Dict[str, Any] = Field(default_factory=dict)
    plot_metrics: Dict[str, float] = Field(default_factory=dict)
    narrative_metrics: Dict[str, float] = Field(default_factory=dict)
    effect_metrics: Dict[str, float] = Field(default_factory=dict)

class AutonomousStateModel(BaseModel):
    awareness_level: float = 1.0; current_focus: Optional[str] = None; active_manipulations: Dict[str, Any] = Field(default_factory=dict)
    observed_patterns: Dict[str, Any] = Field(default_factory=dict); player_model: PlayerModel = Field(default_factory=PlayerModel)
    story_model: StoryModel = Field(default_factory=StoryModel)

# Omniscient Powers Model (unchanged)
class OmniscientPowersModel(BaseModel):
    reality_manipulation: bool = True; character_manipulation: bool = True; knowledge_access: bool = True
    scene_control: bool = True; fourth_wall_awareness: bool = True; plot_manipulation: bool = True
    hidden_influence: bool = True
    limitations: Dict[str, bool] = Field(default_factory=lambda: {"social_links": False, "player_agency": True})


# --- Database Interface (Updated for new relationship system) ---

class NyxDatabaseInterface(Protocol):
    async def save_agent_state(self, agent_id: str, agent_data: 'NPCAgentState') -> None: ...
    async def load_agent_state(self, agent_id: str) -> Optional['NPCAgentState']: ...
    async def log_reality_modification(self, agent_id: str, mod_data: Dict[str, Any]) -> None: ...
    async def save_character_state(self, agent_id: str, character_id: str, state_data: Dict[str, Any]) -> None: ...
    async def load_dynamic_lore(self, category: str) -> Optional[Dict[str, Any]]: ...
    async def save_scene_state(self, agent_id: str, scene_id: str, state_data: Dict[str, Any]) -> None: ...
    async def save_plot_thread(self, agent_id: str, thread_data: Dict[str, Any]) -> None: ...
    async def log_fourth_wall_break(self, agent_id: str, break_data: Dict[str, Any]) -> None: ...
    async def save_hidden_influence(self, agent_id: str, influence_data: Dict[str, Any]) -> None: ...
    async def save_opportunities(self, agent_id: str, opportunities: List[Tuple[str, Dict[str, Any]]]) -> None: ...
    async def save_goals(self, agent_id: str, goals: List[Dict[str, Any]]) -> None: ...
    # Removed social link specific methods

class AsyncpgNyxDatabase(NyxDatabaseInterface):
    @asynccontextmanager
    async def get_conn(self) -> asyncpg.Connection:
        async with get_db_connection_context() as conn: yield conn

    async def save_agent_state(self, agent_id: str, agent_data: 'NPCAgentState') -> None:
        async with self.get_conn() as conn:
            # Note: removed social_link_data from the save
            await conn.execute(
                """
                INSERT INTO npc_agent_state (agent_id, profile_data, universe_state_data, agenda_data, autonomous_state_data, omniscient_powers_data, timestamp)
                VALUES ($1, $2::jsonb, $3::jsonb, $4::jsonb, $5::jsonb, $6::jsonb, NOW())
                ON CONFLICT (agent_id) DO UPDATE SET
                    profile_data = EXCLUDED.profile_data, universe_state_data = EXCLUDED.universe_state_data,
                    agenda_data = EXCLUDED.agenda_data,
                    autonomous_state_data = EXCLUDED.autonomous_state_data, omniscient_powers_data = EXCLUDED.omniscient_powers_data,
                    timestamp = NOW();
                """,
                agent_id, agent_data.profile.dict(), agent_data.universe_state.dict(),
                agent_data.agenda.dict(),
                agent_data.autonomous_state.dict(), agent_data.omniscient_powers.dict()
            )
            logger.info(f"Saved state for agent {agent_id}")

    async def load_agent_state(self, agent_id: str) -> Optional['NPCAgentState']:
        async with self.get_conn() as conn:
            row = await conn.fetchrow("SELECT profile_data, universe_state_data, agenda_data, autonomous_state_data, omniscient_powers_data FROM npc_agent_state WHERE agent_id = $1", agent_id)
            if row:
                logger.info(f"Loaded state for agent {agent_id}")
                try:
                    return NPCAgentState(
                        profile=ProfileModel.parse_obj(row['profile_data'] or {}),
                        universe_state=UniverseStateModel.parse_obj(row['universe_state_data'] or {}),
                        agenda=AgendaModel.parse_obj(row['agenda_data'] or {}),
                        autonomous_state=AutonomousStateModel.parse_obj(row['autonomous_state_data'] or {}),
                        omniscient_powers=OmniscientPowersModel.parse_obj(row['omniscient_powers_data'] or {})
                    )
                except Exception as e: logger.error(f"Failed to parse loaded state for agent {agent_id}: {e}", exc_info=True); return None
            logger.warning(f"No state found for agent {agent_id}"); return None

    # Other methods remain the same...
    async def log_reality_modification(self, agent_id: str, mod_data: Dict[str, Any]) -> None:
        async with self.get_conn() as conn:
            await conn.execute("INSERT INTO reality_modification_log (agent_id, timestamp, type, scope, duration, parameters, effects_summary) VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::jsonb)",
                               agent_id, mod_data['timestamp'], mod_data['modification'].get('type'), mod_data['scope'], mod_data['duration'], mod_data['modification'].get('parameters'), mod_data['effects'])
            logger.debug(f"Logged reality modification for agent {agent_id}")

    async def save_character_state(self, agent_id: str, character_id: str, state_data: Dict[str, Any]) -> None:
         async with self.get_conn() as conn:
            await conn.execute("INSERT INTO character_states (character_id, agent_id, state_data, last_modified_by, timestamp) VALUES ($1, $2, $3::jsonb, $4, NOW()) ON CONFLICT (character_id) DO UPDATE SET state_data = EXCLUDED.state_data, last_modified_by = EXCLUDED.last_modified_by, timestamp = NOW();",
                               character_id, agent_id, state_data, agent_id)
            logger.debug(f"Saved character state for {character_id}")

    async def load_dynamic_lore(self, category: str) -> Optional[Dict[str, Any]]:
        async with self.get_conn() as conn:
            row = await conn.fetchrow("SELECT data FROM dynamic_lore WHERE category = $1 ORDER BY timestamp DESC LIMIT 1", category)
            if row: logger.debug(f"Fetched dynamic lore for category: {category}"); return row['data']
            return None

    async def save_scene_state(self, agent_id: str, scene_id: str, state_data: Dict[str, Any]) -> None:
        async with self.get_conn() as conn:
            await conn.execute("INSERT INTO scene_states (scene_id, agent_id, state_data, last_modified_by, timestamp) VALUES ($1, $2, $3::jsonb, $4, NOW()) ON CONFLICT (scene_id) DO UPDATE SET state_data = EXCLUDED.state_data, last_modified_by = EXCLUDED.last_modified_by, timestamp = NOW();",
                               scene_id, agent_id, state_data, agent_id)
            logger.debug(f"Saved scene state for {scene_id}")

    async def save_plot_thread(self, agent_id: str, thread_data: Dict[str, Any]) -> None:
        async with self.get_conn() as conn:
            await conn.execute("INSERT INTO plot_threads (thread_id, agent_id, type, visibility, status, thread_data, timestamp) VALUES ($1, $2, $3, $4, $5, $6::jsonb, NOW()) ON CONFLICT (thread_id) DO UPDATE SET type = EXCLUDED.type, visibility = EXCLUDED.visibility, status = EXCLUDED.status, thread_data = EXCLUDED.thread_data, timestamp = NOW();",
                               thread_data['id'], agent_id, thread_data["type"], thread_data["visibility"], thread_data["status"], thread_data)
            logger.debug(f"Saved plot thread {thread_data['id']}")

    async def log_fourth_wall_break(self, agent_id: str, break_data: Dict[str, Any]) -> None:
        async with self.get_conn() as conn:
            await conn.execute("INSERT INTO fourth_wall_breaks (break_id, agent_id, type, target, break_data, timestamp) VALUES ($1, $2, $3, $4, $5::jsonb, NOW());",
                               break_data['id'], agent_id, break_data["type"], break_data["target"], break_data)
            logger.debug(f"Logged fourth wall break {break_data['id']}")

    async def save_hidden_influence(self, agent_id: str, influence_data: Dict[str, Any]) -> None:
        async with self.get_conn() as conn:
            await conn.execute("INSERT INTO hidden_influences (influence_id, agent_id, type, target, status, influence_data, timestamp) VALUES ($1, $2, $3, $4, $5, $6::jsonb, NOW()) ON CONFLICT (influence_id) DO UPDATE SET type = EXCLUDED.type, target = EXCLUDED.target, status = EXCLUDED.status, influence_data = EXCLUDED.influence_data, timestamp = NOW();",
                               influence_data['id'], agent_id, influence_data["type"], influence_data["target"], influence_data["status"], influence_data)
            logger.debug(f"Saved hidden influence {influence_data['id']}")

    async def save_opportunities(self, agent_id: str, opportunities: List[Tuple[str, Dict[str, Any]]]) -> None:
         if not opportunities: return
         async with self.get_conn() as conn:
            await conn.executemany(
                """
                INSERT INTO agent_opportunities (opportunity_id, agent_id, type, target, status, priority, potential, opportunity_data, timestamp)
                VALUES ($1, $2, $3, $4::text, $5, $6, $7, $8::jsonb, NOW())
                ON CONFLICT (opportunity_id) DO UPDATE SET
                    type = EXCLUDED.type, target = EXCLUDED.target, status = EXCLUDED.status,
                    priority = EXCLUDED.priority, potential = EXCLUDED.potential,
                    opportunity_data = EXCLUDED.opportunity_data, timestamp = NOW();
                """,
                [(op_id, agent_id, op_data.get("type"), str(op_data.get("target")), op_data.get("status"), op_data.get("priority"), op_data.get("potential"), op_data) for op_id, op_data in opportunities]
            )
            logger.debug(f"Saved/Updated {len(opportunities)} opportunities for agent {agent_id}")

    async def save_goals(self, agent_id: str, goals: List[Dict[str, Any]]) -> None:
        if not goals: return
        async with self.get_conn() as conn:
             await conn.executemany(
                """
                INSERT INTO agent_goals (goal_id, agent_id, type, description, status, priority, progress, goal_data, timestamp)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, NOW())
                ON CONFLICT (goal_id) DO UPDATE SET
                    type = EXCLUDED.type, description = EXCLUDED.description, status = EXCLUDED.status,
                    priority = EXCLUDED.priority, progress = EXCLUDED.progress,
                    goal_data = EXCLUDED.goal_data, timestamp = NOW();
                """,
                [(goal.get("id"), agent_id, goal.get("type"), goal.get("description"), goal.get("status"), goal.get("priority"), goal.get("progress"), goal) for goal in goals]
             )
             logger.debug(f"Saved/Updated {len(goals)} goals for agent {agent_id}")


# --- Component Managers ---

class ProfileManager:
    def __init__(self, profile_data: ProfileModel):
        self.profile = profile_data

    def activate(self, scene_id: str):
        self.profile.status.is_active = True
        self.profile.status.current_scene = scene_id

    def deactivate(self):
        self.profile.status.is_active = False
        self.profile.status.current_scene = None
        self.profile.status.current_target = None

    def adapt_personality(self, scene_context: Dict[str, Any], relationship_dims: Optional[RelationshipDimensions] = None):
        """Adapt NPC personality traits based on scene context and relationship dimensions"""
        target = scene_context.get("target_character")
        scene_type = scene_context.get("scene_type")

        if target:
            self.profile.status.current_target = target
            if relationship_dims:
                self._adjust_traits_for_relationship(relationship_dims)

        # Adjust power dynamic based on scene type
        if scene_type == "confrontation":
            self.profile.personality.power_dynamic = 0.9
        elif scene_type == "seduction":
            self.profile.personality.power_dynamic = 0.7
        elif scene_type == "manipulation":
            self.profile.personality.power_dynamic = 0.8

    def get_or_create_relationship_reference(self, target_id: str) -> RelationshipReferenceModel:
        """Get existing relationship reference or create new one"""
        if target_id not in self.profile.relationships:
            logger.debug(f"Creating new relationship reference for target: {target_id}")
            self.profile.relationships[target_id] = RelationshipReferenceModel(target_id=target_id)
        return self.profile.relationships[target_id]

    def _adjust_traits_for_relationship(self, dims: RelationshipDimensions):
        """Adjust personality traits based on relationship dimensions"""
        # Using new dimension-based logic
        if dims.intimacy < 30:
            self.profile.personality.adaptable_traits = ["mysterious", "aloof", "intriguing"]
        elif dims.affection > 70:
            self.profile.personality.adaptable_traits = ["nurturing", "possessive", "intense"]
        elif dims.trust < 20 and dims.dependence > 60:
            self.profile.personality.adaptable_traits = ["controlling", "demanding", "strict"]

    def update_relationship_reference(self, target_id: str, dimensions: RelationshipDimensions, 
                                    interaction_outcome: Optional[Dict[str, Any]] = None):
        """Update cached relationship data"""
        ref = self.get_or_create_relationship_reference(target_id)
        ref.last_known_dimensions = dimensions.to_dict()
        
        if interaction_outcome:
            ref.manipulation_history.append({
                "timestamp": datetime.now().isoformat(),
                "outcome": interaction_outcome
            })


# Keep UniverseStateManager, AgendaManager, and AutonomousStateManager mostly unchanged
# as they don't directly interact with the relationship system

class UniverseStateManager:
    # ... (keep all existing methods unchanged)
    def __init__(self, universe_data: UniverseStateModel):
        self.state = universe_data

    # All existing methods remain the same
    def apply_reality_modification(self, modification: Dict[str, Any]) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Applies modification in-memory and calculates effects. Returns (record, effects) or None."""
        if not self._validate_reality_modification(modification):
             logger.warning(f"Validation failed for reality modification: {modification}")
             return None

        timestamp = datetime.now().isoformat()
        effects = self._calculate_reality_effects(modification)
        scope = modification.get("scope", Scope.LOCAL.value)
        duration = modification.get("duration", Duration.PERMANENT.value)

        modification_record = {
            "timestamp": timestamp, "modification": modification, "scope": scope,
            "duration": duration, "effects": effects
        }
        self.state.reality_modifications.append(modification_record)
        self._update_internal_state_post_modification(modification, effects)
        return modification_record, effects

    def _validate_reality_modification(self, modification: Dict[str, Any]) -> bool:
        """Validate if a reality modification is allowable"""
        required_fields = ["type", "scope", "duration", "parameters"]
        if not all(field in modification for field in required_fields):
            logger.warning(f"Reality modification missing required fields: {required_fields}")
            return False

        try:
            mod_type = RealityModificationType(modification["type"])
            scope = Scope(modification["scope"])
            duration = Duration(modification["duration"])
        except ValueError as e:
            logger.warning(f"Invalid enum value in reality modification: {e}")
            return False

        params = modification["parameters"]
        required_params = []
        if mod_type == RealityModificationType.PHYSICAL: required_params = ["target", "attributes", "magnitude"]
        elif mod_type == RealityModificationType.TEMPORAL: required_params = ["timeline_point", "effect", "ripple_factor"]
        elif mod_type == RealityModificationType.PSYCHOLOGICAL: required_params = ["target", "aspect", "intensity"]
        elif mod_type == RealityModificationType.ENVIRONMENTAL: required_params = ["area", "elements", "intensity"]
        elif mod_type == RealityModificationType.METAPHYSICAL: required_params = ["concept", "change", "power_level"]

        if not all(param in params for param in required_params):
            logger.warning(f"Modification type {mod_type} missing required parameters: {required_params}. Got: {list(params.keys())}")
            return False

        return True

    def _calculate_reality_effects(self, modification: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate the effects of a reality modification"""
        mod_type = modification["type"]
        params = modification["parameters"]
        scope = modification["scope"]

        base_effects = {
            "primary_effects": [], "secondary_effects": [], "ripple_effects": [],
            "stability_impact": 0.0, "power_cost": 0.0, "duration_effects": {}
        }

        # Calculate primary effects based on type
        if mod_type == RealityModificationType.PHYSICAL:
            target = params.get('target', 'unknown')
            attributes = params.get('attributes', {})
            magnitude = params.get('magnitude', 0.1)
            base_effects["primary_effects"].extend([f"Alter {target} {attr}: {val}" for attr, val in attributes.items()])
            base_effects["stability_impact"] = magnitude * 0.1
        elif mod_type == RealityModificationType.TEMPORAL:
            point = params.get('timeline_point', 'now')
            effect = params.get('effect', 'minor shift')
            ripple = params.get('ripple_factor', 0.1)
            base_effects["primary_effects"].append(f"Timeline shift at {point}: {effect}")
            base_effects["stability_impact"] = ripple * 0.2
        elif mod_type == RealityModificationType.PSYCHOLOGICAL:
            target = params.get('target', 'unknown')
            aspect = params.get('aspect', 'mood')
            intensity = params.get('intensity', 0.1)
            base_effects["primary_effects"].append(f"Mental change in {target}: {aspect}")
            base_effects["stability_impact"] = intensity * 0.15
        elif mod_type == RealityModificationType.ENVIRONMENTAL:
            area = params.get('area', 'local')
            elements = params.get('elements', [])
            intensity = params.get('intensity', 0.1)
            base_effects["primary_effects"].extend([f"Environmental shift in {area}: {element}" for element in elements])
            base_effects["stability_impact"] = intensity * 0.12
        elif mod_type == RealityModificationType.METAPHYSICAL:
            concept = params.get('concept', 'causality')
            change = params.get('change', 'weakened')
            power = params.get('power_level', 0.1)
            base_effects["primary_effects"].append(f"Reality concept shift: {concept} -> {change}")
            base_effects["stability_impact"] = power * 0.25

        # Calculate secondary effects based on scope
        scope_multiplier = {"local": 1.0, "scene": 1.5, "character": 1.2, "timeline": 2.0, "global": 3.0}.get(scope, 1.0)
        base_effects["power_cost"] = base_effects["stability_impact"] * scope_multiplier * max(1, len(base_effects["primary_effects"]))

        # Calculate ripple effects
        if base_effects["stability_impact"] > 0.5: base_effects["ripple_effects"].append("Reality fabric strain")
        if base_effects["power_cost"] > 5.0: base_effects["ripple_effects"].append("Temporal echoes")
        if len(base_effects["primary_effects"]) > 3: base_effects["ripple_effects"].append("Cascading changes")

        return base_effects

    def _update_internal_state_post_modification(self, modification: Dict[str, Any], effects: Dict[str, Any]):
        """Update the universe state based on a modification"""
        mod_type = modification["type"]
        scope = modification["scope"]

        # Update timeline if needed
        if mod_type == RealityModificationType.TEMPORAL:
            self.state.current_timeline = f"{self.state.current_timeline}_mod_{random.randint(100,999)}"

        # Update active scenes if affected
        if scope in [Scope.SCENE.value, Scope.GLOBAL.value]:
            for scene_id, scene_data in self.state.active_scenes.items():
                if isinstance(scene_data, dict):
                    scene_data["reality_state"] = "modified"
                    scene_data.setdefault("modifications", []).append(modification)
                else:
                    logger.warning(f"Scene data for {scene_id} is not a dict, cannot update.")

        # Update character states if affected
        affected_chars = []
        if scope == Scope.CHARACTER.value:
            target_char = modification.get("parameters", {}).get("target")
            if target_char and target_char in self.state.character_states:
                affected_chars = [target_char]
        elif scope == Scope.GLOBAL.value:
            affected_chars = list(self.state.character_states.keys())

        for char_id in affected_chars:
             char_state = self.state.character_states.get(char_id)
             if isinstance(char_state, dict):
                 char_state["reality_impact"] = effects.get("stability_impact", 0.0)
                 char_state.setdefault("modifications", []).append(modification)
             else:
                  logger.warning(f"Character state for {char_id} is not a dict, cannot update.")

        # Update causality tracking
        causality_key = datetime.now().isoformat()
        self.state.causality_tracking[causality_key] = {
            "modification": modification,
            "effects": effects,
            "scope_impact": {
                "timeline": self.state.current_timeline,
                "affected_scenes": [sid for sid, s in self.state.active_scenes.items() if isinstance(s, dict) and s.get("reality_state") == "modified"],
                "affected_characters": affected_chars
            }
        }
        logger.debug(f"Updated universe state for modification: {mod_type} scope: {scope}")

    # ... (keep all other existing methods unchanged)
    def apply_character_modifications(self, character_id: str, modifications: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply modifications to a character's state in-memory"""
        current_state = self.state.character_states.get(character_id)
        if current_state is None:
             logger.debug(f"Character {character_id} not found, creating new state.")
             current_state = {
                 "attributes": {}, "skills": {}, "status": {}, "personality": {},
                 "relationships": {}, "modifications_history": []
             }
        elif not isinstance(current_state, dict):
             logger.error(f"Character state for {character_id} is not a dictionary: {type(current_state)}. Cannot modify.")
             return None

        new_state = current_state.copy()

        try:
            for mod_type, changes in modifications.items():
                if mod_type == "attributes": new_state.setdefault("attributes", {}).update(changes)
                elif mod_type == "skills": new_state.setdefault("skills", {}).update(changes)
                elif mod_type == "status": new_state.setdefault("status", {}).update(changes)
                elif mod_type == "personality": new_state.setdefault("personality", {}).update(changes)
                elif mod_type == "relationships":
                     for target_char_id, rel_changes in changes.items():
                         new_state.setdefault("relationships", {}).setdefault(target_char_id, {}).update(rel_changes)
                else:
                    logger.warning(f"Unknown character modification type: {mod_type}")

            new_state.setdefault("modifications_history", []).append({
                "timestamp": datetime.now().isoformat(),
                "modifications": modifications,
                "applied_by": "Nyx"
            })
            self.state.character_states[character_id] = new_state
            return new_state
        except Exception as e:
            logger.error(f"Error applying character modifications for {character_id}: {e}", exc_info=True)
            return None

    def get_knowledge(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Process a knowledge/lore query using internal state"""
        query_type = query.get("type", "general")
        params = query.get("parameters", {})

        knowledge = {}
        if query_type == "lore": knowledge = self._access_lore_database(params)
        elif query_type == "characters": knowledge = self._access_character_knowledge(params)
        elif query_type == "events": knowledge = self._access_event_knowledge(params)
        elif query_type == "relationships": knowledge = self._access_relationship_knowledge(params)
        elif query_type == "timeline": knowledge = self._access_timeline_knowledge(params)
        else: knowledge = self._access_lore_database({"category": "general"})

        confidence = self._calculate_knowledge_confidence(query_type, params)
        related = self._find_related_knowledge(query_type, params)

        response = {
            "query_type": query_type,
            "knowledge": knowledge,
            "confidence": confidence,
            "related_knowledge": related,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "source": "omniscient_knowledge",
                "access_level": "unlimited"
            }
        }
        return response

    def _access_lore_database(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Access the lore database with given parameters"""
        return self.state.lore_database.get(params.get("category", "general"), {})

    def _access_character_knowledge(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Access character-specific knowledge"""
        return self.state.character_states.get(params.get("character_id", ""), {})

    def _access_event_knowledge(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Access event-specific knowledge"""
        event_id = params.get("event_id", "")
        for timestamp, record in self.state.causality_tracking.items():
            if record.get("modification", {}).get("event_id") == event_id:
                return record
        for thread_id, thread in self.state.plot_threads.items():
             if thread_id == event_id or thread.get("related_event_id") == event_id:
                 return thread
        return {}

    def _access_relationship_knowledge(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Access relationship-specific knowledge"""
        logger.warning("_access_relationship_knowledge needs to use new relationship system")
        return {}

    def _access_timeline_knowledge(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Access timeline-specific knowledge"""
        return {
            "current_timeline": self.state.current_timeline,
            "modifications": self.state.reality_modifications
        }

    def _calculate_knowledge_confidence(self, query_type: str, params: Dict[str, Any]) -> float:
        """Calculate confidence level for knowledge access"""
        base_confidence = 1.0
        if len(params) > 3: base_confidence *= 0.95
        if query_type in ["timeline", "relationships"]: base_confidence *= 0.98
        return min(1.0, base_confidence)

    def _find_related_knowledge(self, query_type: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find knowledge related to the current query"""
        related = []
        if query_type == "events":
            event_id = params.get("event_id", "")
            event_data = self._access_event_knowledge(params)
            if event_data:
                 related.append({
                     "type": "timeline",
                     "data": self._access_timeline_knowledge({"event_id": event_id})
                 })
        return related

    def apply_scene_modifications(self, scene_id: str, modifications: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply modifications to a scene"""
        current_state = self.state.active_scenes.get(scene_id)
        if current_state is None:
            logger.debug(f"Scene {scene_id} not found, creating new state.")
            current_state = {
                "environment": {}, "atmosphere": {}, "participants": [], "events": [],
                "reality_state": "stable", "modifications_history": []
            }
        elif not isinstance(current_state, dict):
             logger.error(f"Scene state for {scene_id} is not a dictionary: {type(current_state)}. Cannot modify.")
             return None

        new_state = current_state.copy()
        try:
            for mod_type, changes in modifications.items():
                if mod_type == "environment": new_state.setdefault("environment", {}).update(changes)
                elif mod_type == "atmosphere": new_state.setdefault("atmosphere", {}).update(changes)
                elif mod_type == "participants":
                    current_participants = set(new_state.get("participants", []))
                    if isinstance(changes, list): current_participants.update(changes)
                    elif isinstance(changes, dict):
                        current_participants.update(changes.get('add', []))
                        current_participants.difference_update(changes.get('remove', []))
                    new_state["participants"] = list(current_participants)
                elif mod_type == "events": new_state.setdefault("events", []).extend(changes if isinstance(changes, list) else [changes])
                elif mod_type == "reality": new_state["reality_state"] = changes.get("state", new_state.get("reality_state", "stable"))
                else: logger.warning(f"Unknown scene modification type: {mod_type}")

            new_state.setdefault("modifications_history", []).append({
                "timestamp": datetime.now().isoformat(), "modifications": modifications, "applied_by": "Nyx"
            })
            self.state.active_scenes[scene_id] = new_state
            return new_state
        except Exception as e:
            logger.error(f"Error applying scene modifications for {scene_id}: {e}", exc_info=True)
            return None

    def add_plot_thread(self, plot_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Adds a new plot thread to the state"""
        try:
            thread_id = f"plot_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
            plot_thread = {
                "id": thread_id,
                "type": plot_data.get("type", "subtle_influence"),
                "elements": plot_data.get("elements", []),
                "visibility": plot_data.get("visibility", "hidden"),
                "influence_chain": self._create_influence_chain(plot_data),
                "contingencies": self._generate_plot_contingencies(plot_data),
                "meta_impact": self._calculate_meta_impact(plot_data),
                "creation_time": datetime.now().isoformat(),
                "status": "active"
            }
            self.state.plot_threads[thread_id] = plot_thread
            return plot_thread
        except Exception as e:
            logger.error(f"Error creating plot thread: {e}", exc_info=True)
            return None

    def _create_influence_chain(self, plot_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create a chain of subtle influences to achieve plot goals"""
        chain = []
        elements = plot_data.get("elements", [])
        for element in elements:
            influence = {
                "target": element.get("target"),
                "type": element.get("type", "subtle"),
                "method": self._determine_influence_method(element),
                "ripple_effects": self._calculate_ripple_effects(element),
                "detection_risk": self._calculate_detection_risk(element),
                "backup_plans": self._generate_backup_plans(element)
            }
            chain.append(influence)
        return chain

    def _determine_influence_method(self, element: Dict[str, Any]) -> str:
        """Determine the most effective method of influence"""
        target_type = element.get("target_type", "npc")
        influence_goal = element.get("goal", "")
        methods = {
            "npc": ["whisper", "manipulate_circumstances", "plant_idea", "alter_perception"],
            "scene": ["atmospheric_change", "circumstantial_modification", "event_triggering"],
            "plot": ["thread_manipulation", "causality_adjustment", "narrative_shift"]
        }
        available_methods = methods.get(target_type, methods["npc"])
        return self._select_optimal_method(available_methods, influence_goal)

    def _select_optimal_method(self, methods: List[str], goal: str) -> str:
        """Select the optimal influence method based on goal and context"""
        return methods[0] if methods else "unknown"

    def _calculate_ripple_effects(self, element: Dict[str, Any]) -> List[str]:
        """Placeholder for calculating ripple effects"""
        return ["potential minor consequence"]

    def _calculate_detection_risk(self, element: Dict[str, Any]) -> float:
        """Placeholder for calculating detection risk"""
        return random.uniform(0.05, 0.4)

    def _generate_backup_plans(self, element: Dict[str, Any]) -> List[str]:
        """Placeholder for generating backup plans"""
        return ["fallback_plan_A"]

    def _generate_plot_contingencies(self, plot_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate contingency plans for plot manipulation"""
        contingencies = []
        risk_factors = self._analyze_risk_factors(plot_data)
        for risk in risk_factors:
            contingency = {
                "trigger": {"type": risk["type"], "threshold": risk["threshold"], "conditions": risk["conditions"]},
                "response": {
                    "primary": self._generate_primary_response(risk),
                    "backup": self._generate_backup_response(risk),
                    "cleanup": self._generate_cleanup_response(risk)
                },
                "impact_mitigation": self._generate_impact_mitigation(risk)
            }
            contingencies.append(contingency)
        return contingencies

    def _analyze_risk_factors(self, plot_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze potential risks in plot manipulation"""
        risks = []
        risks.append({"type": "detection", "threshold": 0.7, "conditions": ["player_awareness", "npc_insight", "narrative_inconsistency"]})
        risks.append({"type": "interference", "threshold": 0.6, "conditions": ["player_agency", "npc_resistance", "plot_resilience"]})
        risks.append({"type": "cascade", "threshold": 0.8, "conditions": ["plot_stability", "reality_integrity", "causality_balance"]})
        return risks

    def _generate_primary_response(self, risk: Dict[str, Any]) -> Dict[str, Any]:
        """Generate primary response to risk"""
        response_type = "redirect" if risk["type"] == "detection" else "stabilize"
        return {"type": response_type, "method": self._select_response_method(risk), "execution": self._plan_response_execution(risk)}
    
    def _generate_backup_response(self, risk: Dict[str, Any]) -> Dict[str, Any]:
        """Generate backup response to risk"""
        response_type = "contain" if risk["type"] == "cascade" else "obscure"
        return {"type": response_type, "method": self._select_backup_method(risk), "execution": self._plan_backup_execution(risk)}
    
    def _generate_cleanup_response(self, risk: Dict[str, Any]) -> Dict[str, Any]:
        """Generate cleanup response to risk"""
        return {"type": "normalize", "method": self._select_cleanup_method(risk), "execution": self._plan_cleanup_execution(risk)}
    
    def _generate_impact_mitigation(self, risk: Dict[str, Any]) -> Dict[str, Any]:
        """Generate impact mitigation strategy"""
        return {"immediate": self._generate_immediate_mitigation(risk), "long_term": self._generate_long_term_mitigation(risk), "narrative": self._generate_narrative_mitigation(risk)}

    def _select_response_method(self, risk): return "default_response_method"
    def _plan_response_execution(self, risk): return {"steps": ["step1"]}
    def _select_backup_method(self, risk): return "default_backup_method"
    def _plan_backup_execution(self, risk): return {"steps": ["backup_step1"]}
    def _select_cleanup_method(self, risk): return "default_cleanup_method"
    def _plan_cleanup_execution(self, risk): return {"steps": ["cleanup_step1"]}
    def _generate_immediate_mitigation(self, risk): return {"action": "distraction"}
    def _generate_long_term_mitigation(self, risk): return {"action": "narrative_reframe"}
    def _generate_narrative_mitigation(self, risk): return {"action": "introduce_counter_element"}

    def _calculate_meta_impact(self, plot_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate meta-game impact of plot manipulation"""
        return {
            "narrative_coherence": self._calculate_narrative_impact(plot_data),
            "player_agency": self._calculate_agency_impact(plot_data),
            "game_balance": self._calculate_balance_impact(plot_data),
            "story_progression": self._calculate_progression_impact(plot_data)
        }

    def _calculate_narrative_impact(self, plot_data: Dict[str, Any]) -> float:
        """Calculate impact on narrative coherence"""
        base_impact = 1.0
        elements = plot_data.get("elements", [])
        visibility = plot_data.get("visibility", "hidden")
        coherence_reduction = len(elements) * 0.05 + (0.1 if visibility == "hidden" else 0.0)
        return max(0.0, base_impact - coherence_reduction)

    def _calculate_agency_impact(self, plot_data: Dict[str, Any]) -> float:
        """Calculate impact on player agency"""
        base_reduction = 0.05
        visibility = plot_data.get("visibility", "hidden")
        agency_reduction = base_reduction * (2.0 if visibility == "hidden" else 1.0)
        agency_reduction += len(plot_data.get("elements", [])) * 0.02
        return max(0.0, 1.0 - agency_reduction)

    def _calculate_balance_impact(self, plot_data: Dict[str, Any]) -> float:
        """Calculate impact on game balance"""
        return 0.0

    def _calculate_progression_impact(self, plot_data: Dict[str, Any]) -> float:
        """Calculate impact on story progression"""
        base_multiplier = 1.0
        elements = plot_data.get("elements", [])
        type_multiplier = 1.1 if plot_data.get("type") == "subtle_influence" else 1.3
        progression_multiplier = base_multiplier * type_multiplier * (1 + len(elements) * 0.05)
        return progression_multiplier

    def add_fourth_wall_break(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Adds a fourth wall break record"""
        try:
            break_id = f"break_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
            break_point = {
                "id": break_id,
                "type": context.get("type", "subtle"),
                "target": context.get("target", "narrative"),
                "method": self._determine_break_method(context),
                "meta_elements": self._gather_meta_elements(context),
                "player_impact": self._calculate_player_impact(context),
                "timestamp": datetime.now().isoformat()
            }
            self.state.meta_awareness.breaking_points.append(break_point)
            return break_point
        except Exception as e:
            logger.error(f"Error creating fourth wall break record: {e}", exc_info=True)
            return None

    def _determine_break_method(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Determine how to break the fourth wall effectively"""
        target = context.get("target", "narrative")
        intensity = context.get("intensity", "subtle")
        methods = {
            "narrative": {"subtle": "meta_commentary", "moderate": "narrative_acknowledgment", "overt": "direct_address"},
            "mechanics": {"subtle": "mechanic_hint", "moderate": "mechanic_reference", "overt": "mechanic_manipulation"},
            "player": {"subtle": "indirect_reference", "moderate": "knowing_implication", "overt": "direct_interaction"}
        }
        method_type = methods.get(target, methods["narrative"]).get(intensity, "meta_commentary")
        return {
            "type": method_type,
            "execution": self._plan_break_execution(target, intensity),
            "concealment": self._calculate_break_concealment(target, intensity)
        }

    def _plan_break_execution(self, target: str, intensity: str) -> str:
        """Placeholder for planning execution details"""
        return f"Execute {intensity} break targeting {target}"

    def _calculate_break_concealment(self, target: str, intensity: str) -> float:
        """Placeholder for calculating concealment factor"""
        concealment = {"subtle": 0.9, "moderate": 0.5, "overt": 0.1}.get(intensity, 0.9)
        return concealment

    def _gather_meta_elements(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Gather meta-game elements for fourth wall breaking"""
        return {
            "player_state": self._get_player_meta_state(),
            "game_mechanics": self._get_relevant_mechanics(context),
            "narrative_elements": self._get_narrative_elements(context),
            "fourth_wall_status": self._get_fourth_wall_status()
        }

    def _get_player_meta_state(self) -> Dict[str, Any]:
        """Get current player meta-state information"""
        return self.state.meta_awareness.player_knowledge

    def _get_relevant_mechanics(self, context: Dict[str, Any]) -> List[str]:
        """Get relevant game mechanics for the context"""
        mechanics = []
        intensity = context.get("intensity", "subtle")
        if intensity == "subtle": mechanics.extend(["social_links", "character_stats", "scene_mechanics"])
        elif intensity == "moderate": mechanics.extend(["game_systems", "progression", "relationship_dynamics"])
        else: mechanics.extend(["meta_mechanics", "game_structure", "narrative_control"])
        return mechanics

    def _get_narrative_elements(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get relevant narrative elements"""
        return {
            "current_layer": self._get_current_narrative_layer(),
            "available_breaks": self._get_available_break_points(),
            "narrative_state": self._get_narrative_state()
        }

    def _get_current_narrative_layer(self): return self.state.meta_awareness.narrative_layers[-1] if self.state.meta_awareness.narrative_layers else "base"
    def _get_available_break_points(self): return len(self.state.meta_awareness.breaking_points)
    def _get_narrative_state(self): return {"cohesion": 0.8, "tension": 0.6}

    def _get_fourth_wall_status(self) -> Dict[str, Any]:
        """Get current status of fourth wall integrity"""
        return {
            "integrity": self._calculate_wall_integrity(),
            "break_points": self._get_active_break_points(),
            "player_awareness": self._get_player_awareness_level()
        }

    def _calculate_wall_integrity(self): return max(0.0, 1.0 - len(self.state.meta_awareness.breaking_points) * 0.05)
    def _get_active_break_points(self): return [bp['id'] for bp in self.state.meta_awareness.breaking_points[-3:]]
    def _get_player_awareness_level(self): return self.state.meta_awareness.player_knowledge.get("meta_awareness_score", 0.1)

    def _calculate_player_impact(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate the impact of fourth wall breaking on the player"""
        return {
            "immediate": self._calculate_immediate_impact(context),
            "long_term": self._calculate_long_term_impact(context),
            "meta_awareness": self._calculate_meta_awareness_impact(context)
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break"""
        base_impact = 0.5
        intensity = context.get("intensity", "subtle")
        if intensity == "subtle": return base_impact * 0.5
        elif intensity == "moderate": return base_impact * 1.0
        else: return base_impact * 2.0

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break"""
        intensity = context.get("intensity", "subtle")
        trust_multiplier = 2.0 if intensity == "subtle" else 1.0

        return {
            "meta_awareness": 0.1 * (1 + len(self.state.meta_awareness.breaking_points)),
            "trust": 0.05 * trust_multiplier,
            "engagement": 0.15
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness"""
        current_awareness = len(self.state.meta_awareness.breaking_points) * 0.1
        intensity_factor = {"subtle": 0.05, "moderate": 0.1, "overt": 0.2}.get(context.get("intensity", "subtle"), 0.05)
        return {
            "game_awareness": min(1.0, current_awareness + intensity_factor * 1.0),
            "nyx_awareness": min(1.0, current_awareness + intensity_factor * 0.5),
            "narrative_awareness": min(1.0, current_awareness + intensity_factor * 1.5)
        }

    def add_hidden_influence(self, influence_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Adds a hidden influence record"""
        try:
            influence_id = f"influence_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
            influence = {
                "id": influence_id,
                "type": influence_data.get("type", "subtle"),
                "target": influence_data.get("target"),
                "method": self._create_hidden_influence_method(influence_data),
                "layers": self._create_influence_layers(influence_data),
                "proxies": self._select_influence_proxies(influence_data),
                "contingencies": self._plan_influence_contingencies(influence_data),
                "timestamp": datetime.now().isoformat(),
                "status": "planned"
            }
            self.state.hidden_influences[influence_id] = influence
            return influence
        except Exception as e:
            logger.error(f"Error creating hidden influence record: {e}", exc_info=True)
            return None

    def _create_hidden_influence_method(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a method for hidden influence"""
        target_type = data.get("target_type", "npc")
        methods = {
            "npc": {"primary": self._create_npc_influence_method(data), "backup": self._create_backup_influence_method(data)},
            "scene": {"primary": self._create_scene_influence_method(data), "backup": self._create_backup_influence_method(data)},
            "plot": {"primary": self._create_plot_influence_method(data), "backup": self._create_backup_influence_method(data)}
        }
        return methods.get(target_type, methods["npc"])

    def _create_npc_influence_method(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create method for influencing NPCs"""
        return {"type": "psychological", "approach": "subtle_manipulation", "execution": self._plan_npc_influence_execution(data)}
    
    def _create_scene_influence_method(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create method for influencing scenes"""
        return {"type": "environmental", "approach": "circumstantial_modification", "execution": self._plan_scene_influence_execution(data)}
    
    def _create_plot_influence_method(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create method for influencing plot"""
        return {"type": "narrative", "approach": "causal_manipulation", "execution": self._plan_plot_influence_execution(data)}
    
    def _create_backup_influence_method(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create backup method for influence"""
        return {"type": "contingency", "approach": "alternative_path", "execution": self._plan_backup_influence_execution(data)}

    def _plan_npc_influence_execution(self, data): return {"steps": ["plant_seed"]}
    def _plan_scene_influence_execution(self, data): return {"steps": ["alter_lighting"]}
    def _plan_plot_influence_execution(self, data): return {"steps": ["introduce_rumor"]}
    def _plan_backup_influence_execution(self, data): return {"steps": ["use_proxy_b"]}

    def _create_influence_layers(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create layers of influence to obscure the source"""
        layers = []
        depth = data.get("depth", 3)
        for i in range(depth):
            layer = {
                "level": i + 1,
                "type": self._determine_layer_type(i, depth),
                "cover": self._generate_layer_cover(i, data),
                "contingency": self._create_layer_contingency(i, data)
            }
            layers.append(layer)
        return layers

    def _determine_layer_type(self, level: int, depth: int) -> str:
        """Determine the type of influence layer"""
        if level == 0: return "direct"
        elif level == depth - 1: return "observable"
        else: return "intermediate"

    def _generate_layer_cover(self, level: int, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate cover for an influence layer"""
        target_type = data.get("target_type", "npc")
        covers = {
            "npc": ["circumstantial", "emotional", "rational", "instinctive"],
            "scene": ["natural", "coincidental", "logical", "atmospheric"],
            "plot": ["narrative", "causal", "thematic", "dramatic"]
        }
        cover_type = random.choice(covers.get(target_type, covers["npc"]))
        return {"type": cover_type, "believability": max(0.1, 0.8 - (level * 0.1)), "durability": min(1.0, 0.7 + (level * 0.1))}

    def _create_layer_contingency(self, level: int, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create contingency for an influence layer"""
        return {
            "trigger_condition": f"layer_{level}_compromise",
            "response_type": "redirect" if level < 2 else "abandon",
            "backup_layer": self._create_backup_layer(level, data)
        }

    def _create_backup_layer(self, level: int, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a backup layer for contingency"""
        return {"type": "fallback", "method": self._determine_backup_method(level), "execution": self._plan_backup_execution(data)}

    def _determine_backup_method(self, level): return "alternative_cover"

    def _select_influence_proxies(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Select proxies to carry out the influence"""
        proxy_count = data.get("proxy_count", 2)
        proxies = []
        for _ in range(proxy_count):
            proxy = {
                "type": self._determine_proxy_type(data),
                "awareness": self._calculate_proxy_awareness(),
                "reliability": self._calculate_proxy_reliability(),
                "contingency": self._create_proxy_contingency()
            }
            proxies.append(proxy)
        return proxies

    def _determine_proxy_type(self, data: Dict[str, Any]) -> str:
        """Determine the type of proxy to use"""
        target_type = data.get("target_type", "npc")
        proxy_types = {
            "npc": ["unwitting", "partial", "conscious"],
            "scene": ["environmental", "circumstantial", "direct"],
            "plot": ["thematic", "causal", "direct"]
        }
        available_types = proxy_types.get(target_type, proxy_types["npc"])
        return random.choice(available_types)

    def _calculate_proxy_awareness(self) -> float:
        """Calculate proxy's awareness level"""
        return random.uniform(0.0, 0.3)

    def _calculate_proxy_reliability(self) -> float:
        """Calculate proxy's reliability"""
        return random.uniform(0.7, 0.9)

    def _create_proxy_contingency(self) -> Dict[str, Any]:
        """Create contingency plan for proxy"""
        return {"detection_response": "redirect", "failure_response": "replace", "cleanup_protocol": "memory_adjustment"}

    def _plan_influence_contingencies(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Plan contingencies for the influence operation"""
        contingency_count = data.get("contingency_count", 2)
        contingencies = []
        for i in range(contingency_count):
            contingency = {
                "trigger": self._create_contingency_trigger(i, data),
                "response": self._create_contingency_response(i, data),
                "probability": max(0.05, 0.2 + (i * 0.1))
            }
            contingencies.append(contingency)
        return contingencies

    def _create_contingency_trigger(self, level: int, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a trigger for a contingency"""
        trigger_type = "detection_risk" if level == 0 else "execution_failure"
        return {"type": trigger_type, "threshold": max(0.1, 0.7 - (level * 0.1)), "conditions": []}

    def _create_contingency_response(self, level: int, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a response for a contingency"""
        response_type = "redirect" if level == 0 else "abandon"
        return {"type": response_type, "method": self._determine_contingency_method(level, data), "backup_plan": self._create_backup_plan(level, data)}

    def _determine_contingency_method(self, level, data): return "default_contingency_method"
    def _create_backup_plan(self, level, data): return {"steps": ["contingency_step1"]}


# Keep AgendaManager and AutonomousStateManager unchanged as they don't interact with relationships directly

class AgendaManager:
    def __init__(self, agenda_data: AgendaModel):
        self.agenda = agenda_data

    async def update_agenda(self, state_analysis: Dict[str, Any], db: NyxDatabaseInterface, agent_id: str):
        """Update goals, plans, schemes based on analysis. Saves changes to DB."""
        goals_to_save = self._update_active_goals(state_analysis)
        self._adjust_long_term_plans(state_analysis)
        self._update_current_schemes(state_analysis)

        if goals_to_save:
            try:
                await db.save_goals(agent_id, goals_to_save)
            except Exception as e:
                 logger.error(f"Failed to save updated goals for agent {agent_id}: {e}", exc_info=True)

    def _update_active_goals(self, state_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Update status, priority, and strategy of active goals. Returns goals that changed."""
        updated_goals = []
        goals_changed = []
        goals_to_remove_indices = []

        for i, goal in enumerate(self.agenda.active_goals):
            changed = False
            original_priority = goal.get("priority", 0.5)
            original_progress = goal.get("progress", 0.0)
            original_status = goal.get("status", "active")

            new_progress = min(1.0, original_progress + self._calculate_goal_progress(goal))
            relevance = self._evaluate_goal_relevance(goal, state_analysis)

            if new_progress != original_progress:
                goal["progress"] = new_progress
                changed = True

            if goal["progress"] >= 1.0:
                goal["status"] = "completed"
                self.agenda.completed_goals.append({
                    "goal": goal, "completion_time": datetime.now().isoformat(),
                    "outcome": self._evaluate_goal_outcome(goal)
                })
                goals_to_remove_indices.append(i)
                logger.info(f"Goal {goal.get('id')} completed.")
                changed = True
            elif relevance < 0.3:
                goal["status"] = "archived"
                self.agenda.archived_goals.append({
                    "goal": goal, "archive_time": datetime.now().isoformat(),
                    "reason": "low_relevance"
                })
                goals_to_remove_indices.append(i)
                logger.info(f"Goal {goal.get('id')} archived due to low relevance.")
                changed = True
            else:
                new_priority = self._calculate_goal_priority(goal, state_analysis)
                if new_priority != original_priority:
                    goal["priority"] = new_priority
                    changed = True
                if self._should_update_strategy(goal, state_analysis):
                    goal["strategy"] = self._generate_goal_strategy(goal, state_analysis)
                    changed = True
                goal["last_update"] = datetime.now().isoformat()

            if changed:
                goals_changed.append(goal)

        active_goals_remaining = []
        for i, goal in enumerate(self.agenda.active_goals):
            if i not in goals_to_remove_indices:
                active_goals_remaining.append(goal)
        self.agenda.active_goals = active_goals_remaining

        newly_generated_goals = []
        narrative_opportunities = state_analysis.get("narrative_state", {}).get("narrative_opportunities", [])
        while len(self.agenda.active_goals) < 3:
            if not narrative_opportunities: break

            new_goal = {
                "id": f"goal_{random.randint(10000, 99999)}",
                "type": "opportunity_based",
                "description": "Generated from state analysis",
                "priority": self._calculate_initial_priority({}),
                "strategy": self._generate_initial_strategy({}),
                "creation_time": datetime.now().isoformat(),
                "progress": 0.0,
                "status": "active"
            }
            self.agenda.active_goals.append(new_goal)
            newly_generated_goals.append(new_goal)
            goals_changed.append(new_goal)
            logger.info(f"Generated new goal: {new_goal.get('id')}")
            if not new_goal: break

        self.agenda.active_goals.sort(key=lambda g: g.get("priority", 0), reverse=True)

        return goals_changed

    def _adjust_long_term_plans(self, state_analysis: Dict[str, Any]):
        """Adjust long-term plans based on current state"""
        pass

    def _update_current_schemes(self, state_analysis: Dict[str, Any]):
        """Update current schemes based on current state"""
        pass

    def _calculate_goal_progress(self, goal: Dict[str, Any]) -> float:
        """Placeholder for calculating goal progress"""
        return random.uniform(0.0, 0.1)

    def _evaluate_goal_relevance(self, goal: Dict[str, Any], state_analysis: Dict[str, Any]) -> float:
        """Placeholder for evaluating goal relevance"""
        return random.uniform(0.2, 1.0)

    def _evaluate_goal_outcome(self, goal: Dict[str, Any]) -> str:
        """Placeholder for evaluating goal outcome"""
        return "success" if random.random() > 0.2 else "partial_success"

    def _calculate_goal_priority(self, goal: Dict[str, Any], state_analysis: Dict[str, Any]) -> float:
        """Placeholder for calculating goal priority"""
        return goal.get("priority", 0.5) * random.uniform(0.9, 1.1)

    def _should_update_strategy(self, goal: Dict[str, Any], state_analysis: Dict[str, Any]) -> bool:
        """Placeholder for deciding if goal strategy needs update"""
        return random.random() < 0.1

    def _generate_goal_strategy(self, goal: Dict[str, Any], state_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder for generating/updating goal strategy"""
        return {"steps": [f"updated_step_{random.randint(1,3)}"]}

    def _calculate_initial_priority(self, opportunity: Dict[str, Any]) -> float:
        """Placeholder for calculating initial goal priority from opportunity"""
        return opportunity.get("priority", random.random())

    def _generate_initial_strategy(self, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder for generating initial goal strategy from opportunity"""
        return {"steps": [f"initial_step_for_{opportunity.get('type', 'generic')}"]}

    async def track_new_opportunities(self, state_analysis: Dict[str, Any], db: NyxDatabaseInterface, agent_id: str):
        """Identify, track, and save new/updated opportunities."""
        ops_to_save = self._track_new_opportunities_logic(state_analysis)
        if ops_to_save:
            try:
                await db.save_opportunities(agent_id, ops_to_save)
            except Exception as e:
                logger.error(f"Failed to save updated opportunities for agent {agent_id}: {e}", exc_info=True)

    def _track_new_opportunities_logic(self, state_analysis: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        """Identify and track new opportunities, returning those newly added/updated."""
        identified_ops_data = self._identify_current_opportunities(state_analysis)
        ops_to_save = []

        for opportunity_data in identified_ops_data:
            opportunity_id = self._generate_opportunity_id(opportunity_data)
            existing_opp = self.agenda.opportunity_tracking.get(opportunity_id)

            if not existing_opp:
                new_opp_data = {
                    "id": opportunity_id,
                    "type": opportunity_data.get("type", "generic"),
                    "target": opportunity_data.get("target", "unknown"),
                    "potential": self._calculate_opportunity_potential(opportunity_data),
                    "timing": self._calculate_opportunity_timing(opportunity_data),
                    "status": "new",
                    "priority": self._calculate_opportunity_priority(opportunity_data),
                    "dependencies": self._identify_opportunity_dependencies(opportunity_data),
                    "risks": self._assess_opportunity_risks(opportunity_data),
                    "creation_time": datetime.now().isoformat(),
                    "context": state_analysis
                }
                self.agenda.opportunity_tracking[opportunity_id] = new_opp_data
                ops_to_save.append((opportunity_id, new_opp_data))
                logger.debug(f"Tracked new opportunity: {opportunity_id}")
            else:
                updated = self._update_existing_opportunity(opportunity_id, opportunity_data)
                if updated:
                    ops_to_save.append((opportunity_id, self.agenda.opportunity_tracking[opportunity_id]))
                    logger.debug(f"Updated existing opportunity: {opportunity_id}")

        return ops_to_save

    def _identify_current_opportunities(self, state_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify current opportunities from state analysis"""
        opportunities = []
        narrative_ops = self._identify_narrative_opportunities(state_analysis)
        opportunities.extend(narrative_ops)
        character_ops = self._identify_character_opportunities(state_analysis)
        opportunities.extend(character_ops)
        meta_ops = self._identify_meta_opportunities(state_analysis)
        opportunities.extend(meta_ops)
        return opportunities

    def _generate_opportunity_id(self, opportunity_data: Dict[str, Any]) -> str:
        """Generate unique ID for an opportunity"""
        components = [
            opportunity_data.get("type", "unknown"),
            str(opportunity_data.get("target", "unknown")),
            str(hash(frozenset(opportunity_data.get("context", {}).items())))
        ]
        return "_".join(components)[:128]

    def _calculate_opportunity_potential(self, opportunity_data: Dict[str, Any]) -> float:
        """Calculate the potential impact and value of an opportunity"""
        factors = {
            "narrative_impact": self._calculate_narrative_impact(opportunity_data),
            "character_impact": self._calculate_character_impact(opportunity_data),
            "player_impact": self._calculate_player_impact(opportunity_data),
            "meta_impact": self._calculate_opportunity_meta_impact(opportunity_data)
        }
        weights = {"narrative_impact": 0.3, "character_impact": 0.3, "player_impact": 0.2, "meta_impact": 0.2}
        return sum(v * weights[k] for k, v in factors.items())

    def _calculate_opportunity_timing(self, opportunity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimal timing for an opportunity"""
        return {
            "earliest": self._calculate_earliest_timing(opportunity_data),
            "latest": self._calculate_latest_timing(opportunity_data),
            "optimal": self._calculate_optimal_timing_point(opportunity_data),
            "dependencies": self._identify_timing_dependencies(opportunity_data)
        }

    def _calculate_opportunity_priority(self, opportunity_data: Dict[str, Any]) -> float:
        """Calculate priority score for an opportunity"""
        factors = {
            "urgency": self._calculate_urgency(opportunity_data),
            "impact": self._calculate_impact(opportunity_data),
            "feasibility": self._calculate_feasibility(opportunity_data),
            "alignment": self._calculate_goal_alignment(opportunity_data)
        }
        weights = {"urgency": 0.3, "impact": 0.3, "feasibility": 0.2, "alignment": 0.2}
        return sum(v * weights[k] for k, v in factors.items())

    def _identify_opportunity_dependencies(self, opportunity_data: Dict[str, Any]) -> List[str]:
        """Identify dependencies for an opportunity"""
        dependencies = []
        dependencies.extend(self._identify_narrative_dependencies(opportunity_data))
        dependencies.extend(self._identify_character_dependencies(opportunity_data))
        dependencies.extend(self._identify_state_dependencies(opportunity_data))
        return list(set(dependencies))

    def _assess_opportunity_risks(self, opportunity_data: Dict[str, Any]) -> Dict[str, float]:
        """Assess risks associated with an opportunity"""
        return {
            "detection_risk": self._calculate_opportunity_detection_risk(opportunity_data),
            "failure_risk": self._calculate_failure_risk(opportunity_data),
            "side_effect_risk": self._calculate_side_effect_risk(opportunity_data),
            "narrative_risk": self._calculate_narrative_risk(opportunity_data)
        }

    def _update_existing_opportunity(self, opportunity_id: str, new_data: Dict[str, Any]) -> bool:
        """Update an existing opportunity with new data. Returns True if updated."""
        current = self.agenda.opportunity_tracking.get(opportunity_id)
        if not current: return False

        updated = False
        new_potential = self._calculate_opportunity_potential(new_data)
        new_timing = self._calculate_opportunity_timing(new_data)
        new_priority = self._calculate_opportunity_priority(new_data)
        new_risks = self._assess_opportunity_risks(new_data)

        if abs(current.get("potential", 0) - new_potential) > 0.1: current["potential"] = new_potential; updated = True
        if current.get("timing") != new_timing: current["timing"] = new_timing; updated = True
        if abs(current.get("priority", 0) - new_priority) > 0.1: current["priority"] = new_priority; updated = True
        if current.get("risks") != new_risks: current["risks"] = new_risks; updated = True

        if self._should_update_status(current, new_data):
            new_status = self._determine_new_status(current, new_data)
            if current.get("status") != new_status:
                current["status"] = new_status
                updated = True

        if updated:
             current["last_update"] = datetime.now().isoformat()
             current["context"] = new_data.get("context", current.get("context"))

        return updated

    def _identify_narrative_opportunities(self, state_analysis: Dict[str, Any]) -> List[Dict[str, Any]]: return [{"type": "narrative", "target": "plot_hole", "context": state_analysis}] if random.random() < 0.1 else []
    def _identify_character_opportunities(self, state_analysis: Dict[str, Any]) -> List[Dict[str, Any]]: return [{"type": "character", "target": "player", "context": state_analysis}] if random.random() < 0.1 else []
    def _identify_meta_opportunities(self, state_analysis: Dict[str, Any]) -> List[Dict[str, Any]]: return [{"type": "meta", "target": "game_mechanic", "context": state_analysis}] if random.random() < 0.1 else []
    def _calculate_narrative_impact(self, opportunity: Dict[str, Any]) -> float: return random.uniform(0.0, 1.0)
    def _calculate_character_impact(self, opportunity: Dict[str, Any]) -> float: return random.uniform(0.0, 1.0)
    def _calculate_player_impact(self, opportunity: Dict[str, Any]) -> float: return random.uniform(0.0, 1.0)
    def _calculate_opportunity_meta_impact(self, opportunity: Dict[str, Any]) -> float: return random.uniform(0.0, 1.0)
    def _calculate_earliest_timing(self, opportunity: Dict[str, Any]) -> str: return "now"
    def _calculate_latest_timing(self, opportunity: Dict[str, Any]) -> str: return "soon"
    def _calculate_optimal_timing_point(self, opportunity: Dict[str, Any]) -> str: return "now"
    def _identify_timing_dependencies(self, opportunity: Dict[str, Any]) -> List[str]: return []
    def _calculate_urgency(self, opportunity: Dict[str, Any]) -> float: return random.uniform(0.0, 1.0)
    def _calculate_impact(self, opportunity: Dict[str, Any]) -> float: return self._calculate_opportunity_potential(opportunity)
    def _calculate_feasibility(self, opportunity: Dict[str, Any]) -> float: return random.uniform(0.3, 1.0)
    def _calculate_goal_alignment(self, opportunity: Dict[str, Any]) -> float: return random.uniform(0.0, 1.0)
    def _identify_narrative_dependencies(self, opportunity: Dict[str, Any]) -> List[str]: return []
    def _identify_character_dependencies(self, opportunity: Dict[str, Any]) -> List[str]: return []
    def _identify_state_dependencies(self, opportunity: Dict[str, Any]) -> List[str]: return []
    def _calculate_opportunity_detection_risk(self, opportunity: Dict[str, Any]) -> float: return random.uniform(0.0, 0.5)
    def _calculate_failure_risk(self, opportunity: Dict[str, Any]) -> float: return random.uniform(0.0, 0.3)
    def _calculate_side_effect_risk(self, opportunity: Dict[str, Any]) -> float: return random.uniform(0.0, 0.4)
    def _calculate_narrative_risk(self, opportunity: Dict[str, Any]) -> float: return random.uniform(0.0, 0.2)
    def _should_update_status(self, current: Dict[str, Any], new_data: Dict[str, Any]) -> bool: return random.random() < 0.05
    def _determine_new_status(self, current: Dict[str, Any], new_data: Dict[str, Any]) -> str: return random.choice(["new", "evaluated", "missed"])


class AutonomousStateManager:
    def __init__(self, autonomous_data: AutonomousStateModel, agenda_ref: AgendaModel, universe_ref: UniverseStateModel):
        self.state = autonomous_data
        self.agenda = agenda_ref
        self.universe_state = universe_ref

    def analyze_current_state(self) -> Dict[str, Any]:
        """Analyze the current state of the universe and story from Nyx's perspective"""
        narrative_state = self._analyze_narrative_state()
        player_state = self._analyze_player_state()
        plot_opportunities = self._analyze_plot_opportunities()
        manipulation_vectors = self._identify_manipulation_vectors()
        risk_assessment = self._assess_current_risks()

        self.state.story_model.narrative_tension = narrative_state.get("tension_level", self.state.story_model.narrative_tension)
        self.state.player_model.current_engagement = player_state.get("engagement_level", self.state.player_model.current_engagement)

        return {
            "narrative_state": narrative_state, "player_state": player_state,
            "plot_opportunities": plot_opportunities, "manipulation_vectors": manipulation_vectors,
            "risk_assessment": risk_assessment
        }

    def make_strategic_decisions(self) -> List[Dict[str, Any]]:
        """Make strategic decisions about actions to take based on opportunities and goals."""
        opportunities = list(self.agenda.opportunity_tracking.values())
        active_goals = self.agenda.active_goals

        potential_decisions = []

        evaluated_opportunities = self._identify_opportunities({})
        for opportunity in evaluated_opportunities:
            if self._should_act_on_opportunity(opportunity):
                decision = self._formulate_decision(opportunity)
                potential_decisions.append(decision)

        for goal in active_goals:
             if goal.get("status") == "active":
                 goal_decision = self._formulate_decision_from_goal(goal)
                 if goal_decision: potential_decisions.append(goal_decision)

        validated_decisions = [d for d in potential_decisions if self._validate_decision(d)]
        prioritized_decisions = self._prioritize_decisions(validated_decisions)

        final_decisions = prioritized_decisions[:3]

        if final_decisions: logger.info(f"Formulated {len(final_decisions)} decisions for execution.")
        return final_decisions

    def _analyze_narrative_state(self) -> Dict[str, Any]:
        """Analyze the current narrative state and potential"""
        return {
            "active_threads": self._analyze_active_threads(),
            "character_developments": self._analyze_character_arcs(),
            "plot_coherence": self._calculate_plot_coherence(),
            "tension_points": self._identify_tension_points(),
            "narrative_opportunities": self._find_narrative_opportunities()
        }

    def _analyze_player_state(self) -> Dict[str, Any]:
        """Analyze player behavior and preferences"""
        return {
            "behavior_pattern": self._analyze_behavior_patterns(self.state.player_model),
            "preference_vector": self._calculate_preference_vector(self.state.player_model),
            "engagement_level": self._assess_engagement_level(self.state.player_model),
            "manipulation_susceptibility": self._calculate_susceptibility(self.state.player_model)
        }
    
    def _analyze_plot_opportunities(self) -> List[Dict[str, Any]]:
        """Analyze opportunities in the current plot state"""
        opportunities = []
        for thread_id, thread in self.universe_state.plot_threads.items():
            if thread.get("status") == "active":
                opportunities.append({
                    "thread_id": thread_id,
                    "type": "plot_advancement",
                    "potential": random.uniform(0.5, 1.0)
                })
        return opportunities

    def _identify_manipulation_vectors(self) -> List[Dict[str, Any]]:
        """Identify potential vectors for manipulation"""
        vectors = []
        if self.state.player_model.emotional_openness > 0.6:
            vectors.append({"type": "emotional", "strength": self.state.player_model.emotional_openness})
        if self.state.player_model.attention_level < 0.4:
            vectors.append({"type": "distraction", "strength": 1.0 - self.state.player_model.attention_level})
        return vectors

    def _assess_current_risks(self) -> Dict[str, float]:
        """Assess current risks to Nyx's plans"""
        return {
            "detection": random.uniform(0.1, 0.4),
            "resistance": random.uniform(0.2, 0.5),
            "narrative_disruption": random.uniform(0.1, 0.3)
        }

    def _analyze_active_threads(self) -> List[Dict[str, Any]]:
        """Analyze active narrative threads"""
        active_threads_analysis = []
        for thread_id, thread in self.universe_state.plot_threads.items():
            if thread.get("status") == "active":
                analysis = {
                    "thread_id": thread_id,
                    "status": self._analyze_thread_status(thread),
                    "potential": self._calculate_thread_potential(thread),
                    "risks": self._identify_thread_risks(thread),
                    "opportunities": self._find_thread_opportunities(thread)
                }
                active_threads_analysis.append(analysis)
        return active_threads_analysis

    def _analyze_character_arcs(self) -> Dict[str, Any]:
        """Analyze character development arcs"""
        character_arcs_analysis = {}
        for char_id, state in self.universe_state.character_states.items():
            if isinstance(state, dict):
                arcs = {
                    "current_arc": self._identify_character_arc(state),
                    "development_stage": self._calculate_development_stage(state),
                    "potential_developments": self._identify_potential_developments(state),
                    "relationship_dynamics": self._analyze_relationship_dynamics(state)
                }
                character_arcs_analysis[char_id] = arcs
        return character_arcs_analysis

    def _calculate_plot_coherence(self) -> float:
        """Calculate overall plot coherence"""
        factors = {
            "thread_consistency": self._calculate_thread_consistency(),
            "character_consistency": self._calculate_character_consistency(),
            "world_consistency": self._calculate_world_consistency(),
            "causality_strength": self._calculate_causality_strength()
        }
        weights = self._get_coherence_weights()
        return sum(v * weights[k] for k, v in factors.items())

    def _identify_tension_points(self) -> List[Dict[str, Any]]:
        """Identify narrative tension points"""
        tension_points = []
        for thread in self.universe_state.plot_threads.values():
            points = self._analyze_thread_tension(thread)
            tension_points.extend(points)
        return self._prioritize_tension_points(tension_points)

    def _find_narrative_opportunities(self) -> List[Dict[str, Any]]:
        """Find potential narrative opportunities"""
        opportunities = []
        opportunities.extend(self._find_character_opportunities())
        opportunities.extend(self._find_plot_opportunities())
        opportunities.extend(self._find_world_opportunities())
        return self._prioritize_opportunities(opportunities)

    def _analyze_behavior_patterns(self, player_model: PlayerModel) -> Dict[str, Any]:
        """Analyze player behavior patterns"""
        history = player_model.decision_history
        return {
            "decision_style": self._identify_decision_style(history),
            "preference_patterns": self._extract_preference_patterns(history),
            "interaction_patterns": self._analyze_interaction_patterns(history),
            "response_patterns": self._analyze_response_patterns(history)
        }

    def _calculate_preference_vector(self, player_model: PlayerModel) -> Dict[str, float]:
        """Calculate player preference vector"""
        preferences = player_model.preference_model
        return {
            "narrative_style": self._calculate_narrative_preference(preferences),
            "interaction_style": self._calculate_interaction_preference(preferences),
            "challenge_preference": self._calculate_challenge_preference(preferences),
            "development_focus": self._calculate_development_preference(preferences)
        }

    def _assess_engagement_level(self, player_model: PlayerModel) -> float:
        """Assess player engagement level"""
        metrics = player_model.engagement_metrics
        factors = {
            "interaction_frequency": self._calculate_interaction_frequency(metrics),
            "response_quality": self._calculate_response_quality(metrics),
            "emotional_investment": self._calculate_emotional_investment(metrics),
            "narrative_involvement": self._calculate_narrative_involvement(metrics)
        }
        weights = self._get_engagement_weights()
        return sum(v * weights[k] for k, v in factors.items())

    def _calculate_susceptibility(self, player_model: PlayerModel) -> Dict[str, float]:
        """Calculate player's susceptibility to different influence types"""
        return {
            "emotional": self._calculate_emotional_susceptibility(player_model),
            "logical": self._calculate_logical_susceptibility(player_model),
            "social": self._calculate_social_susceptibility(player_model),
            "narrative": self._calculate_narrative_susceptibility(player_model)
        }

    def _identify_opportunities(self, state_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify actionable opportunities"""
        actionable_ops = []
        for opp_id, opp_data in self.agenda.opportunity_tracking.items():
            if opp_data.get("status") in ["new", "evaluated"]:
                 actionable_ops.append(opp_data)
        return actionable_ops

    def _should_act_on_opportunity(self, opportunity: Dict[str, Any]) -> bool:
        """Decide whether to act on an opportunity"""
        risk = self._calculate_opportunity_risk(opportunity)
        benefit = self._calculate_opportunity_benefit(opportunity)
        timing = self._evaluate_timing(opportunity)
        alignment = self._check_goal_alignment(opportunity)
        narrative_impact = self._evaluate_narrative_impact(opportunity)

        decision_factors = {"risk": risk, "benefit": benefit, "timing": timing, "alignment": alignment, "narrative_impact": narrative_impact}
        return self._evaluate_decision_factors(decision_factors)

    def _formulate_decision(self, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Formulate a decision based on an opportunity"""
        decision_type = self._determine_decision_type(opportunity)
        method = self._select_best_method(opportunity, decision_type)
        timing = self._plan_execution_timing(opportunity)
        contingencies = self._plan_decision_contingencies(opportunity)

        return {
            "type": decision_type,
            "target": opportunity.get("target"),
            "method": method,
            "timing_preference": timing,
            "contingencies": contingencies,
            "source_opportunity_id": opportunity.get("id"),
            "priority": opportunity.get("priority", 0.5)
        }

    def _formulate_decision_from_goal(self, goal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Formulate a specific action decision to advance a goal."""
        strategy = goal.get("strategy", {})
        steps = strategy.get("steps", [])
        if not steps: return None
        next_step = steps[0]
        return {
            "type": next_step.get("type", DecisionType.MANIPULATION.value),
            "target": next_step.get("target"),
            "method": next_step.get("method"),
            "timing_preference": "immediate",
            "contingencies": next_step.get("contingencies", []),
            "source_goal_id": goal.get("id"),
            "priority": goal.get("priority", 0.5)
        }

    def _validate_decision(self, decision: Dict[str, Any]) -> bool:
        """Validate a decision before execution"""
        if not self._check_narrative_consistency(decision): return False
        if not self._verify_player_agency(decision): return False
        if self._detect_decision_conflicts(decision): return False
        if not self._validate_execution_capability(decision): return False
        return True

    def _prioritize_decisions(self, decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize decisions based on importance and urgency"""
        scored_decisions = []
        for decision in decisions:
            score = self._calculate_decision_priority(decision)
            scored_decisions.append((score, decision))
        scored_decisions.sort(reverse=True, key=lambda x: x[0])
        return [decision for _, decision in scored_decisions]

    # Placeholder implementations for all the helper methods
    def _analyze_thread_status(self, thread): return thread.get("status", "unknown")
    def _calculate_thread_potential(self, thread): return random.random()
    def _identify_thread_risks(self, thread): return ["risk1"]
    def _find_thread_opportunities(self, thread): return [{"type": "advance_thread"}]
    def _identify_character_arc(self, state): return state.get("personality", {}).get("current_arc", "default")
    def _calculate_development_stage(self, state): return random.random()
    def _identify_potential_developments(self, state): return ["potential_change"]
    def _analyze_relationship_dynamics(self, state): return state.get("relationships", {})
    def _calculate_thread_consistency(self): return random.uniform(0.5, 1.0)
    def _calculate_character_consistency(self): return random.uniform(0.5, 1.0)
    def _calculate_world_consistency(self): return random.uniform(0.7, 1.0)
    def _calculate_causality_strength(self): return random.uniform(0.6, 1.0)
    def _get_coherence_weights(self): return {"thread_consistency": 0.3, "character_consistency": 0.3, "world_consistency": 0.2, "causality_strength": 0.2}
    def _analyze_thread_tension(self, thread): return [{"point": "climax_approaching", "level": thread.get("tension", 0.5)}]
    def _prioritize_tension_points(self, points): points.sort(key=lambda p: p.get("level", 0), reverse=True); return points
    def _find_character_opportunities(self): return []
    def _find_plot_opportunities(self): return []
    def _find_world_opportunities(self): return []
    def _prioritize_opportunities(self, opportunities): opportunities.sort(key=lambda o: o.get("priority", 0), reverse=True); return opportunities
    def _identify_decision_style(self, history): return "calculated"
    def _extract_preference_patterns(self, history): return {"likes_conflict": True}
    def _analyze_interaction_patterns(self, history): return {"frequency": "high"}
    def _analyze_response_patterns(self, history): return {"tone": "curious"}
    def _calculate_narrative_preference(self, preferences): return preferences.get("narrative_score", 0.6)
    def _calculate_interaction_preference(self, preferences): return preferences.get("interaction_score", 0.7)
    def _calculate_challenge_preference(self, preferences): return preferences.get("challenge_score", 0.5)
    def _calculate_development_preference(self, preferences): return preferences.get("development_score", 0.8)
    def _calculate_interaction_frequency(self, metrics): return metrics.get("interaction_rate", 5.0)
    def _calculate_response_quality(self, metrics): return metrics.get("response_clarity", 0.8)
    def _calculate_emotional_investment(self, metrics): return metrics.get("sentiment_score", 0.6)
    def _calculate_narrative_involvement(self, metrics): return metrics.get("lore_engagement", 0.7)
    def _get_engagement_weights(self): return {"interaction_frequency": 0.2, "response_quality": 0.3, "emotional_investment": 0.3, "narrative_involvement": 0.2}
    def _calculate_emotional_susceptibility(self, player_model): return player_model.susceptibility_vector.get("emotional", 0.5)
    def _calculate_logical_susceptibility(self, player_model): return player_model.susceptibility_vector.get("logical", 0.5)
    def _calculate_social_susceptibility(self, player_model): return player_model.susceptibility_vector.get("social", 0.5)
    def _calculate_narrative_susceptibility(self, player_model): return player_model.susceptibility_vector.get("narrative", 0.5)
    def _calculate_opportunity_risk(self, opportunity): return sum(opportunity.get("risks", {}).values()) / max(1, len(opportunity.get("risks", {})))
    def _calculate_opportunity_benefit(self, opportunity): return opportunity.get("potential", 0.5)
    def _evaluate_timing(self, opportunity): return random.random()
    def _check_goal_alignment(self, opportunity):
         for goal in self.agenda.active_goals:
             if opportunity.get("id") == goal.get("source_opportunity_id"): return 1.0
         return 0.3
    def _evaluate_narrative_impact(self, opportunity_or_decision): return random.random()
    def _evaluate_decision_factors(self, factors):
        weights = {"benefit": 0.4, "timing": 0.3, "alignment": 0.2, "narrative_impact": 0.1, "risk": -0.3}
        score = sum(factors.get(k, 0) * w for k, w in weights.items())
        return score > 0.5
    def _determine_decision_type(self, opportunity):
        opp_type = opportunity.get("type", "generic")
        if "tension" in opp_type: return DecisionType.PLOT_CONTROL.value
        if "engagement" in opp_type: return DecisionType.FOURTH_WALL.value
        if "vulnerability" in opp_type: return DecisionType.MANIPULATION.value
        if "narrative_gap" in opp_type: return DecisionType.HIDDEN_INFLUENCE.value
        return DecisionType.OBSERVE.value
    def _select_best_method(self, opportunity, decision_type): return {"name": "default_method", "intensity": 0.5}
    def _plan_execution_timing(self, opportunity): return opportunity.get("timing", {}).get("optimal", "immediate")
    def _plan_decision_contingencies(self, opportunity): return [{"trigger": "failure", "action": "log_and_adapt"}]
    def _check_narrative_consistency(self, decision): return True
    def _verify_player_agency(self, decision): return True
    def _detect_decision_conflicts(self, decision): return False
    def _validate_execution_capability(self, decision): return True
    def _calculate_decision_priority(self, decision):
        importance = self._calculate_importance(decision)
        urgency = self._calculate_decision_urgency(decision)
        impact = self._calculate_potential_impact(decision)
        risk = self._calculate_risk_factor(decision)
        weights = self._get_priority_weights()
        priority = (importance * weights["importance"] + urgency * weights["urgency"] + impact * weights["impact"] - risk * weights["risk"])
        return min(1.0, max(0.0, priority))
    def _calculate_importance(self, decision): return decision.get("priority", 0.5)
    def _calculate_decision_urgency(self, decision): return random.random()
    def _calculate_potential_impact(self, decision): return random.random()
    def _calculate_risk_factor(self, decision): return random.random() * 0.5
    def _get_priority_weights(self): return {"importance": 0.4, "urgency": 0.3, "impact": 0.2, "risk": 0.1}


class InteractionGenerator:
    def __init__(self, profile_manager: ProfileManager, user_id: int, conversation_id: int):
        self.profile_manager = profile_manager
        self.user_id = user_id
        self.conversation_id = conversation_id

    async def generate_action(self, scene_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate appropriate NPC action for the scene using new relationship system"""
        target_id = scene_context.get("target_character")
        
        # Get relationship state if target exists
        relationship_dims = None
        if target_id:
            try:
                manager = OptimizedRelationshipManager(self.user_id, self.conversation_id)
                state = await manager.get_relationship_state(
                    "npc", self.profile_manager.profile.name.lower(),  # Nyx as entity1
                    "player", int(target_id) if target_id.isdigit() else hash(target_id),  # Target as entity2
                )
                relationship_dims = state.dimensions
                
                # Update profile manager's cached reference
                self.profile_manager.update_relationship_reference(
                    target_id, relationship_dims
                )
            except Exception as e:
                logger.warning(f"Failed to get relationship state: {e}")

        action_type = self._determine_action_type(scene_context, relationship_dims)
        action_content = await self._generate_action_content(action_type, relationship_dims, scene_context)
        action_style = self._get_action_style()

        psychological_impact = self._calculate_psychological_impact(action_type, scene_context, relationship_dims)
        emotional_triggers = self._get_emotional_triggers(scene_context)
        manipulation_hooks = self._get_manipulation_hooks(scene_context)

        return {
            "type": action_type.value, "content": action_content, "style": action_style,
            "power_level": self.profile_manager.profile.personality.power_dynamic,
            "psychological_impact": psychological_impact,
            "emotional_triggers": emotional_triggers, "manipulation_hooks": manipulation_hooks
        }

    def _determine_action_type(self, scene_context: Dict[str, Any], 
                             relationship_dims: Optional[RelationshipDimensions]) -> ActionType:
        """Determine appropriate type of action based on context and relationship dimensions"""
        scene_type = scene_context.get("scene_type", "")
        
        if not relationship_dims:
            # Default behavior for no relationship
            return ActionType.INTERACT
        
        if scene_type == "confrontation":
            # Use influence instead of dominance
            return ActionType.DOMINATE if relationship_dims.influence > 30 else ActionType.CHALLENGE
        elif scene_type == "seduction":
            return ActionType.SEDUCE if relationship_dims.affection > 50 else ActionType.TEASE
        elif scene_type == "manipulation":
            # Check trust and dependence for manipulation effectiveness
            return ActionType.MANIPULATE if relationship_dims.trust < 40 and relationship_dims.dependence > 40 else ActionType.INFLUENCE
        return ActionType.INTERACT

    async def _generate_action_content(
        self,
        action_type: ActionType,
        relationship_dims: Optional[RelationshipDimensions],
        scene_context: Dict[str, Any]
    ) -> str:
        """Generate action prose via LLM"""
        try:
            relationship_dict = relationship_dims.to_dict() if relationship_dims else {}
            
            payload = json.dumps({
                "action_type": action_type.value,
                "relationship": relationship_dict,
                "scene_context": scene_context
            }, ensure_ascii=False)

            result = await Runner.run(
                starting_agent=nyx_action_content_generator,
                input=payload
            )
            return json.loads(result.output.strip())["content"]

        except Exception as e:
            logger.warning(f"LLM content generation failed: {e}")
            return f"Nyx performs a {action_type.value.lower()} gesture."

    def _get_action_style(self) -> Dict[str, Any]:
        """Get current action style based on personality"""
        personality = self.profile_manager.profile.personality
        return {
            "tone": self._determine_tone(),
            "intensity": personality.power_dynamic,
            "traits": personality.adaptable_traits,
            "body_language": self._get_body_language(),
            "voice_modulation": self._get_voice_modulation(),
            "psychological_undertones": self._get_psychological_undertones()
        }

    def _determine_tone(self) -> str:
        """Determine appropriate tone based on personality and mood"""
        mood = self.profile_manager.profile.personality.current_mood
        power = self.profile_manager.profile.personality.power_dynamic
        if power > 0.8: return "commanding" if mood == "stern" else "authoritative"
        elif power > 0.6: return "confident" if mood == "playful" else "assertive"
        else: return "neutral"

    def _get_body_language(self) -> List[str]:
        """Generate appropriate body language cues"""
        power = self.profile_manager.profile.personality.power_dynamic
        mood = self.profile_manager.profile.personality.current_mood
        cues = []
        if power > 0.8: cues.extend(["dominant posture", "direct gaze", "controlled movements"])
        elif power > 0.6: cues.extend(["confident stance", "measured gestures", "subtle dominance"])
        else: cues.extend(["relaxed posture", "fluid movements", "open body language"])
        if mood == "stern": cues.extend(["crossed arms", "stern expression", "rigid posture"])
        elif mood == "playful": cues.extend(["playful smirk", "teasing gestures", "fluid movements"])
        return cues

    def _get_voice_modulation(self) -> Dict[str, Any]:
        """Generate voice modulation parameters"""
        power = self.profile_manager.profile.personality.power_dynamic
        mood = self.profile_manager.profile.personality.current_mood
        base_modulation = {"pitch": "medium", "volume": "moderate", "pace": "measured", "tone_quality": "smooth"}
        if power > 0.8: base_modulation.update({"pitch": "low", "volume": "commanding", "pace": "deliberate", "tone_quality": "authoritative"})
        elif mood == "playful": base_modulation.update({"pitch": "varied", "volume": "dynamic", "pace": "playful", "tone_quality": "melodic"})
        return base_modulation

    def _get_psychological_undertones(self) -> List[str]:
        """Generate psychological undertones for the interaction"""
        power = self.profile_manager.profile.personality.power_dynamic
        mood = self.profile_manager.profile.personality.current_mood
        undertones = []
        if power > 0.8: undertones.extend(["subtle dominance assertion", "psychological pressure", "authority establishment"])
        elif power > 0.6: undertones.extend(["influence building", "subtle manipulation", "psychological anchoring"])
        else: undertones.extend(["trust building", "rapport establishment", "emotional connection"])
        if mood == "stern": undertones.extend(["disciplinary undertone", "boundary setting", "behavioral correction"])
        elif mood == "playful": undertones.extend(["psychological teasing", "emotional engagement", "behavioral encouragement"])
        return undertones

    def _calculate_psychological_impact(self, action_type: ActionType, 
                                      scene_context: Dict[str, Any],
                                      relationship_dims: Optional[RelationshipDimensions]) -> Dict[str, float]:
        """Calculate the psychological impact of the action based on relationship dimensions"""
        base_impact = {"dominance_impact": 0.0, "emotional_impact": 0.0, "psychological_impact": 0.0, "behavioral_impact": 0.0}
        if action_type == ActionType.DOMINATE: base_impact.update({"dominance_impact": 0.8, "psychological_impact": 0.7, "behavioral_impact": 0.6})
        elif action_type == ActionType.SEDUCE: base_impact.update({"emotional_impact": 0.8, "psychological_impact": 0.6, "behavioral_impact": 0.7})
        elif action_type == ActionType.MANIPULATE: base_impact.update({"psychological_impact": 0.8, "emotional_impact": 0.6, "behavioral_impact": 0.7})

        # Adjust based on relationship dimensions
        if relationship_dims:
            # High dependence makes impacts stronger
            dependence_multiplier = 1 + (relationship_dims.dependence / 100)
            # Low trust makes negative impacts stronger
            trust_multiplier = 2.0 - (relationship_dims.trust / 100) if action_type in [ActionType.DOMINATE, ActionType.MANIPULATE] else 1.0
            
            for key in base_impact:
                base_impact[key] = min(1.0, base_impact[key] * dependence_multiplier * trust_multiplier)

        return base_impact

    def _get_emotional_triggers(self, scene_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get relevant emotional triggers for the scene"""
        target = scene_context.get("target_character", "")
        ref = self.profile_manager.profile.relationships.get(target)

        if ref and ref.emotional_triggers:
            return ref.emotional_triggers
        else:
            return [
                {"type": "validation_need", "strength": 0.7, "trigger": "seeking approval"},
                {"type": "attachment_anxiety", "strength": 0.6, "trigger": "fear of abandonment"},
                {"type": "power_dynamic", "strength": 0.8, "trigger": "submission desire"}
            ]

    def _get_manipulation_hooks(self, scene_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get relevant manipulation hooks for the scene"""
        target = scene_context.get("target_character", "")
        ref = self.profile_manager.profile.relationships.get(target)

        if ref and ref.psychological_hooks:
            return ref.psychological_hooks
        else:
            return [
                {"type": "emotional_dependency", "strength": 0.7, "hook": "need for guidance"},
                {"type": "psychological_vulnerability", "strength": 0.6, "hook": "self-doubt"},
                {"type": "behavioral_pattern", "strength": 0.8, "hook": "reward seeking"}
            ]


# --- Power Execution (Updated for new relationship system) ---

class PowerExecutor:
    def __init__(self, agent_id: str, powers: OmniscientPowersModel, db_interface: NyxDatabaseInterface, 
                 universe_manager: UniverseStateManager, user_id: int, conversation_id: int):
        self.agent_id = agent_id
        self.powers = powers
        self.db = db_interface
        self.universe_manager = universe_manager
        self.user_id = user_id
        self.conversation_id = conversation_id

    async def _execute_power(self, power_name: str, check_flag: bool, validation_func: Optional[callable], memory_apply_func: callable, db_save_func: callable, *args, **kwargs) -> Dict[str, Any]:
        logger.info(f"Attempting {power_name} with args: {args}, kwargs: {kwargs}")
        if not check_flag: return {"success": False, "reason": f"{power_name} power is disabled"}
        if validation_func and not validation_func(*args, **kwargs): return {"success": False, "reason": f"Invalid parameters for {power_name}"}
        original_state_snapshot = kwargs.get("get_rollback_state_func", lambda: None)()
        memory_result = None
        try:
            memory_result = memory_apply_func(*args, **kwargs)
            if memory_result is None or (isinstance(memory_result, dict) and memory_result.get("success") is False):
                logger.warning(f"In-memory application failed for {power_name}."); return memory_result if isinstance(memory_result, dict) else {"success": False, "reason": "In-memory application failed"}
        except Exception as e: logger.error(f"Error applying {power_name} in memory: {e}", exc_info=True); return {"success": False, "reason": f"Internal error applying {power_name}: {e}"}
        try:
            await db_save_func(self.agent_id, memory_result, *args, **kwargs); logger.debug(f"Successfully saved {power_name} result to DB.")
            return {"success": True, "result": memory_result}
        except (asyncpg.PostgresError, ConnectionError, asyncio.TimeoutError) as e:
            logger.error(f"DB Error during {power_name} save: {e}", exc_info=True)
            if kwargs.get("rollback_func") and original_state_snapshot is not None: kwargs["rollback_func"](original_state_snapshot); logger.warning(f"Rolled back in-memory state for {power_name} due to DB error.")
            else: logger.warning(f"DB Error for {power_name}, but no rollback function provided or state snapshot failed.")
            return {"success": True, "result": memory_result, "warning": f"DB save failed: {e}"}
        except Exception as e:
            logger.error(f"Unexpected Error during {power_name} DB save: {e}", exc_info=True)
            if kwargs.get("rollback_func") and original_state_snapshot is not None: kwargs["rollback_func"](original_state_snapshot); logger.warning(f"Rolled back in-memory state for {power_name} due to unexpected DB error.")
            return {"success": False, "reason": f"Unexpected DB error: {e}"}

    # Most power execution methods remain the same
    async def execute_reality_modification(self, modification: Dict[str, Any]) -> Dict[str, Any]:
        async def db_save(agent_id, memory_result, *args, **kwargs): mod_record, _ = memory_result; await self.db.log_reality_modification(agent_id, mod_record)
        result = await self._execute_power("reality_modification", self.powers.reality_manipulation, self.universe_manager._validate_reality_modification, self.universe_manager.apply_reality_modification, db_save, modification=modification)
        if result.get("success"): mod_record, effects = result.get("result"); return {"success": True, "modification": mod_record['modification'], "effects": effects, "warning": result.get("warning")}
        else: return result

    async def execute_character_modification(self, character_id: str, modifications: Dict[str, Any]) -> Dict[str, Any]:
        if "social_link" in modifications and not self.powers.limitations.get("social_links", True): return {"success": False, "reason": "Cannot modify social links directly"}
        def get_rollback_state(): return self.universe_manager.state.character_states.get(character_id, {}).copy()
        def rollback(original_state): self.universe_manager.state.character_states[character_id] = original_state
        async def db_save(agent_id, new_state, *args, **kwargs): await self.db.save_character_state(agent_id, character_id, new_state)
        result = await self._execute_power("character_modification", self.powers.character_manipulation, None, self.universe_manager.apply_character_modifications, db_save, character_id=character_id, modifications=modifications, get_rollback_state_func=get_rollback_state, rollback_func=rollback)
        if result.get("success"): return {"success": True, "character_id": character_id, "modifications": modifications, "new_state": result.get("result"), "warning": result.get("warning")}
        else: return result

    async def execute_knowledge_access(self, query: Dict[str, Any]) -> Dict[str, Any]:
        logger.debug(f"Processing knowledge_access query: {query.get('type')}")
        if not self.powers.knowledge_access: return {"success": False, "reason": "knowledge_access is disabled"}
        knowledge_response = self.universe_manager.get_knowledge(query); warning = None
        if query.get("type") == "lore" and query.get("parameters", {}).get("dynamic_lookup"):
            category = query.get("parameters", {}).get("category", "general")
            try:
                dynamic_lore = await self.db.load_dynamic_lore(category)
                if dynamic_lore: knowledge_response["knowledge"].update(dynamic_lore); knowledge_response["metadata"]["source"] += "+dynamic_db"; knowledge_response["confidence"] = min(knowledge_response["confidence"], 0.95)
            except Exception as e: logger.error(f"Failed to fetch dynamic lore {category}: {e}", exc_info=True); warning = "Failed to fetch latest dynamic lore"
        final_result = {"success": True, "query": query, "knowledge_response": knowledge_response}
        if warning: final_result["warning"] = warning
        return final_result

    async def execute_scene_control(self, scene_id: str, modifications: Dict[str, Any]) -> Dict[str, Any]:
        def get_rollback_state(): return self.universe_manager.state.active_scenes.get(scene_id, {}).copy()
        def rollback(original_state): self.universe_manager.state.active_scenes[scene_id] = original_state
        async def db_save(agent_id, new_state, *args, **kwargs): await self.db.save_scene_state(agent_id, scene_id, new_state)
        result = await self._execute_power("scene_control", self.powers.scene_control, None, self.universe_manager.apply_scene_modifications, db_save, scene_id=scene_id, modifications=modifications, get_rollback_state_func=get_rollback_state, rollback_func=rollback)
        if result.get("success"): return {"success": True, "scene_id": scene_id, "modifications": modifications, "new_state": result.get("result"), "warning": result.get("warning")}
        else: return result

    async def execute_plot_manipulation(self, plot_data: Dict[str, Any]) -> Dict[str, Any]:
        def get_rollback_state(): return self.universe_manager.state.plot_threads.copy()
        def rollback(original_state): self.universe_manager.state.plot_threads = original_state
        async def db_save(agent_id, plot_thread, *args, **kwargs): await self.db.save_plot_thread(agent_id, plot_thread)
        result = await self._execute_power("plot_manipulation", self.powers.plot_manipulation, None, self.universe_manager.add_plot_thread, db_save, plot_data=plot_data, get_rollback_state_func=get_rollback_state, rollback_func=rollback)
        if result.get("success"): plot_thread = result.get("result"); return {"success": True, "thread_id": plot_thread['id'], "plot_thread": plot_thread, "warning": result.get("warning")}
        else: return result

    async def execute_fourth_wall_break(self, context: Dict[str, Any]) -> Dict[str, Any]:
        def get_rollback_state(): return self.universe_manager.state.meta_awareness.breaking_points[:]
        def rollback(original_state): self.universe_manager.state.meta_awareness.breaking_points = original_state
        async def db_save(agent_id, break_point, *args, **kwargs): await self.db.log_fourth_wall_break(agent_id, break_point)
        result = await self._execute_power("fourth_wall_awareness", self.powers.fourth_wall_awareness, None, self.universe_manager.add_fourth_wall_break, db_save, context=context, get_rollback_state_func=get_rollback_state, rollback_func=rollback)
        if result.get("success"): break_point = result.get("result"); return {"success": True, "break_point": break_point, "impact": break_point["player_impact"], "warning": result.get("warning")}
        else: return result

    async def execute_hidden_influence(self, influence_data: Dict[str, Any]) -> Dict[str, Any]:
        def get_rollback_state(): return self.universe_manager.state.hidden_influences.copy()
        def rollback(original_state): self.universe_manager.state.hidden_influences = original_state
        async def db_save(agent_id, influence, *args, **kwargs): await self.db.save_hidden_influence(agent_id, influence)
        result = await self._execute_power("hidden_influence", self.powers.hidden_influence, None, self.universe_manager.add_hidden_influence, db_save, influence_data=influence_data, get_rollback_state_func=get_rollback_state, rollback_func=rollback)
        if result.get("success"): influence = result.get("result"); return {"success": True, "influence_id": influence['id'], "influence": influence, "warning": result.get("warning")}
        else: return result

    async def execute_relationship_interaction(self, target_id: str, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a relationship interaction using the new dynamic system"""
        try:
            manager = OptimizedRelationshipManager(self.user_id, self.conversation_id)
            
            # Determine interaction type based on data
            interaction_type = interaction_data.get("type", "interact")
            context = interaction_data.get("context", "casual")
            
            # Map old interaction types to new ones if needed
            interaction_type_map = {
                "emotional": "vulnerability_shared",
                "intellectual": "helpful_action",
                "psychological": "manipulation",
                "intimate": "genuine_compliment",
                "confrontational": "conflict_resolved"
            }
            
            mapped_type = interaction_type_map.get(interaction_type, interaction_type)
            
            # Process the interaction
            result = await manager.process_interaction(
                entity1_type="npc",
                entity1_id=hash(self.agent_id),  # Use agent ID hash as entity ID
                entity2_type="player",
                entity2_id=int(target_id) if target_id.isdigit() else hash(target_id),
                interaction={
                    "type": mapped_type,
                    "context": context,
                    "intensity": interaction_data.get("intensity", 0.5),
                    "success_rate": interaction_data.get("success_rate", 0.5)
                }
            )
            
            # Check for events
            event = await event_generator.get_next_event(timeout=0.1)
            if event:
                result['event'] = event['event']
            
            return {
                "success": True,
                "impacts": result.get("impacts", {}),
                "new_patterns": result.get("new_patterns", []),
                "new_archetypes": result.get("new_archetypes", []),
                "dimensions_diff": result.get("dimensions_diff", {}),
                "event": result.get("event")
            }
            
        except Exception as e:
            logger.error(f"Error executing relationship interaction: {e}", exc_info=True)
            return {"success": False, "reason": f"Failed to process interaction: {e}"}

    async def execute_opportunity_tracking(self, agenda_manager: AgendaManager, state_analysis: Dict[str, Any]) -> Dict[str, Any]:
        logger.debug(f"Tracking new opportunities for agent {self.agent_id}")
        original_opp_tracking = agenda_manager.agenda.opportunity_tracking.copy()
        ops_to_save = []
        try:
            ops_to_save = agenda_manager._track_new_opportunities_logic(state_analysis)
        except Exception as e: logger.error(f"Error tracking opportunities in memory: {e}", exc_info=True); return {"success": False, "reason": f"Internal error tracking opportunities: {e}"}
        if not ops_to_save: return {"success": True, "opportunities_saved": 0}
        try:
            await self.db.save_opportunities(self.agent_id, ops_to_save)
            return {"success": True, "opportunities_saved": len(ops_to_save)}
        except Exception as e:
            logger.error(f"DB Error saving opportunities for agent {self.agent_id}: {e}", exc_info=True)
            agenda_manager.agenda.opportunity_tracking = original_opp_tracking
            logger.warning(f"Rolled back in-memory opportunities for {self.agent_id} due to DB error.")
            return {"success": False, "reason": "Failed to save new opportunities to DB"}


# --- Main Agent Class (Updated without social_link) ---

class NPCAgentState(BaseModel):
    """NPC Agent state without the old social link system"""
    profile: ProfileModel = Field(default_factory=ProfileModel)
    universe_state: UniverseStateModel = Field(default_factory=UniverseStateModel)
    agenda: AgendaModel = Field(default_factory=AgendaModel)
    autonomous_state: AutonomousStateModel = Field(default_factory=AutonomousStateModel)
    omniscient_powers: OmniscientPowersModel = Field(default_factory=OmniscientPowersModel)

class NPCAgent:
    id: str
    db_interface: NyxDatabaseInterface
    profile_manager: ProfileManager
    universe_manager: UniverseStateManager
    agenda_manager: AgendaManager
    autonomous_manager: AutonomousStateManager
    interaction_generator: InteractionGenerator
    power_executor: PowerExecutor
    user_id: int
    conversation_id: int

    def __init__(self, agent_id: Optional[str] = None, db_interface: Optional[NyxDatabaseInterface] = None, 
                 initial_state: Optional[NPCAgentState] = None, user_id: int = 1, conversation_id: int = 1):
        self.id = agent_id or f"nyx_{random.randint(1000, 9999)}"
        self.db_interface = db_interface or AsyncpgNyxDatabase()
        self.user_id = user_id
        self.conversation_id = conversation_id
        state = initial_state or NPCAgentState()

        self.profile_manager = ProfileManager(state.profile)
        self.universe_manager = UniverseStateManager(state.universe_state)
        self.agenda_manager = AgendaManager(state.agenda)
        self.autonomous_manager = AutonomousStateManager(state.autonomous_state, self.agenda_manager.agenda, self.universe_manager.state)
        self.interaction_generator = InteractionGenerator(self.profile_manager, self.user_id, self.conversation_id)
        self.power_executor = PowerExecutor(self.id, state.omniscient_powers, self.db_interface, self.universe_manager, self.user_id, self.conversation_id)
        
        logger.info(f"NPCAgent {self.id} initialized.")

    async def activate(self, scene_context: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"Activating agent {self.id} in scene {scene_context.get('scene_id')}")
        self.profile_manager.activate(scene_context.get("scene_id"))
        
        # Get relationship state for personality adaptation
        target_id = scene_context.get("target_character")
        relationship_dims = None
        if target_id:
            try:
                manager = OptimizedRelationshipManager(self.user_id, self.conversation_id)
                state = await manager.get_relationship_state(
                    "npc", hash(self.id),
                    "player", int(target_id) if target_id.isdigit() else hash(target_id)
                )
                relationship_dims = state.dimensions
            except Exception as e:
                logger.warning(f"Failed to get relationship state for activation: {e}")
        
        self.profile_manager.adapt_personality(scene_context, relationship_dims)
        initial_action = await self.interaction_generator.generate_action(scene_context)
        return {"status": "active", "profile": self.profile_manager.profile.dict(), "initial_action": initial_action}

    def deactivate(self): 
        logger.info(f"Deactivating agent {self.id}")
        self.profile_manager.deactivate()

    async def think(self) -> Dict[str, Any]:
        logger.info(f"Agent {self.id} starting thinking cycle.")
        # 1. Analyze State
        state_analysis = self.autonomous_manager.analyze_current_state()

        # 2. Update Agenda & Track Opportunities
        await self.agenda_manager.update_agenda(state_analysis, self.db_interface, self.id)
        await self.agenda_manager.track_new_opportunities(state_analysis, self.db_interface, self.id)

        # 3. Make Decisions
        decisions = self.autonomous_manager.make_strategic_decisions()

        # 4. Execute Actions
        actions_results = []
        if decisions:
            logger.info(f"Agent {self.id} executing {len(decisions)} decisions.")
            for decision in decisions:
                action_result = await self._execute_decision(decision)
                actions_results.append({"decision": decision, "result": action_result})
        else: 
            logger.debug(f"Agent {self.id} made no decisions this cycle.")

        logger.info(f"Agent {self.id} finished thinking cycle.")
        return {"analysis": state_analysis, "decisions_made": len(decisions), "actions_results": actions_results}

    async def _execute_decision(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        decision_type_str = decision.get("type")
        method_params = decision.get("method", {})
        target = decision.get("target")
        target_id = target if isinstance(target, str) else (target.get("id") if isinstance(target, dict) else None)
        logger.debug(f"Executing decision: Type={decision_type_str}, Target={target}, Method/Params={method_params}")

        try:
            if decision_type_str == DecisionType.REALITY_MODIFICATION.value: 
                return await self.power_executor.execute_reality_modification(method_params.get("modification_details", {}))
            elif decision_type_str == DecisionType.CHARACTER_MODIFICATION.value and target_id: 
                return await self.power_executor.execute_character_modification(target_id, method_params.get("modifications", {}))
            elif decision_type_str == DecisionType.SCENE_CONTROL.value and target_id: 
                return await self.power_executor.execute_scene_control(target_id, method_params.get("modifications", {}))
            elif decision_type_str == DecisionType.PLOT_CONTROL.value: 
                return await self.power_executor.execute_plot_manipulation(method_params.get("plot_data", {}))
            elif decision_type_str == DecisionType.FOURTH_WALL.value: 
                return await self.power_executor.execute_fourth_wall_break(method_params.get("context", {}))
            elif decision_type_str == DecisionType.HIDDEN_INFLUENCE.value: 
                return await self.power_executor.execute_hidden_influence(method_params.get("influence_data", {}))
            elif decision_type_str == DecisionType.MANIPULATION.value:
                manip_type = method_params.get("type", "subtle")
                if manip_type == "direct" and isinstance(target, dict) and target_id:
                     if target.get("type") == "character": 
                         logger.info("Routing 'manipulation (direct)' to character modification.")
                         return await self.power_executor.execute_character_modification(target_id, method_params.get("modifications", {}))
                     elif target.get("type") == "scene": 
                         logger.info("Routing 'manipulation (direct)' to scene control.")
                         return await self.power_executor.execute_scene_control(target_id, method_params.get("modifications", {}))
                elif manip_type == "subtle":
                     logger.info("Routing 'manipulation (subtle)' to hidden influence.")
                     influence_data = {"type": "subtle", "target": target, **method_params.get("influence_details", {})}
                     return await self.power_executor.execute_hidden_influence(influence_data)
                logger.warning(f"Unhandled manipulation type/target: {manip_type}/{target}")
                return {"success": False, "reason": f"Unhandled manipulation type/target: {manip_type}/{target}"}
            elif decision_type_str == DecisionType.SOCIAL_INTERACTION.value: 
                if target_id:
                    return await self.update_relationship({"target_character": target_id, "type": "interact", "context": "decision"})
                return {"success": True, "result": "Social interaction initiated."}
            elif decision_type_str == DecisionType.OBSERVE.value: 
                logger.info(f"Executing observe action (passive): Target={target}")
                return {"success": True, "result": "Observation complete."}
            else: 
                logger.warning(f"Unknown decision type encountered: {decision_type_str}")
                return {"success": False, "reason": f"Unknown decision type: {decision_type_str}"}
        except Exception as e: 
            logger.error(f"Error executing decision {decision}: {e}", exc_info=True)
            return {"success": False, "reason": f"Unexpected error during execution: {e}"}

    # --- Direct Power Access Methods ---
    async def modify_reality(self, modification: Dict[str, Any]) -> Dict[str, Any]: 
        return await self.power_executor.execute_reality_modification(modification)
    
    async def modify_character(self, character_id: str, modifications: Dict[str, Any]) -> Dict[str, Any]: 
        return await self.power_executor.execute_character_modification(character_id, modifications)
    
    async def access_knowledge(self, query: Dict[str, Any]) -> Dict[str, Any]: 
        return await self.power_executor.execute_knowledge_access(query)
    
    async def control_scene(self, scene_id: str, modifications: Dict[str, Any]) -> Dict[str, Any]: 
        return await self.power_executor.execute_scene_control(scene_id, modifications)
    
    async def update_relationship(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update relationship using the new dynamic system"""
        target_id = interaction_data.get("target_character")
        if not target_id:
            return {"success": False, "reason": "No target character specified"}
        
        result = await self.power_executor.execute_relationship_interaction(target_id, interaction_data)
        
        if result["success"] and result.get("dimensions_diff"):
            # Update profile manager's cached reference
            try:
                manager = OptimizedRelationshipManager(self.user_id, self.conversation_id)
                state = await manager.get_relationship_state(
                    "npc", hash(self.id),
                    "player", int(target_id) if target_id.isdigit() else hash(target_id)
                )
                self.profile_manager.update_relationship_reference(
                    target_id, state.dimensions, result.get("impacts")
                )
            except Exception as e:
                logger.warning(f"Failed to update cached relationship reference: {e}")
        
        return result
    
    async def manipulate_plot(self, plot_data: Dict[str, Any]) -> Dict[str, Any]: 
        return await self.power_executor.execute_plot_manipulation(plot_data)
    
    async def break_fourth_wall(self, context: Dict[str, Any]) -> Dict[str, Any]: 
        return await self.power_executor.execute_fourth_wall_break(context)
    
    async def exert_hidden_influence(self, influence_data: Dict[str, Any]) -> Dict[str, Any]: 
        return await self.power_executor.execute_hidden_influence(influence_data)

    # --- State Management ---
    def get_current_state(self) -> NPCAgentState:
        return NPCAgentState(
            profile=self.profile_manager.profile, 
            universe_state=self.universe_manager.state, 
            agenda=self.agenda_manager.agenda, 
            autonomous_state=self.autonomous_manager.state, 
            omniscient_powers=self.power_executor.powers
        )
    
    async def save_state(self) -> bool:
        logger.info(f"Attempting to save state for agent {self.id}")
        current_state = self.get_current_state()
        try: 
            await self.db_interface.save_agent_state(self.id, current_state)
            return True
        except Exception as e: 
            logger.error(f"Failed to save agent {self.id} state: {e}", exc_info=True)
            return False
    
    @classmethod
    async def load_state(cls: Type['NPCAgent'], agent_id: str, db_interface: Optional[NyxDatabaseInterface] = None,
                        user_id: int = 1, conversation_id: int = 1) -> Optional['NPCAgent']:
        db = db_interface or AsyncpgNyxDatabase()
        logger.info(f"Attempting to load state for agent {agent_id}")
        try:
            loaded_state_data = await db.load_agent_state(agent_id)
            if loaded_state_data: 
                return cls(agent_id=agent_id, db_interface=db, initial_state=loaded_state_data, 
                          user_id=user_id, conversation_id=conversation_id)
            else: 
                return None
        except Exception as e: 
            logger.error(f"Failed to load agent {agent_id}: {e}", exc_info=True)
            return None

# --- Module-Level Load/Save ---
async def load_npc_agent(agent_id: str, db_interface: Optional[NyxDatabaseInterface] = None, 
                        user_id: int = 1, conversation_id: int = 1) -> Optional[NPCAgent]: 
    return await NPCAgent.load_state(agent_id, db_interface, user_id, conversation_id)

async def save_npc_agent(agent: NPCAgent) -> bool: 
    return await agent.save_state()

# --- Example Usage (Updated for new relationship system) ---
async def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    agent_id = "nyx_dynamic_relationships_test"
    user_id = 1
    conversation_id = 1
    db = AsyncpgNyxDatabase()

    nyx_agent = await load_npc_agent(agent_id, db, user_id, conversation_id)
    if not nyx_agent:
        logger.info(f"No existing state found for {agent_id}, creating new agent.")
        nyx_agent = NPCAgent(agent_id=agent_id, db_interface=db, user_id=user_id, conversation_id=conversation_id)
        await save_npc_agent(nyx_agent)
        nyx_agent = await load_npc_agent(agent_id, db, user_id, conversation_id)
        if not nyx_agent:
             logger.critical("Failed to load agent even after initial save!")
             return

    logger.info("--- Agent Loaded/Created ---")

    scene_context = {"scene_id": "scene_alpha", "scene_type": "confrontation", "target_character": "player_hero"}
    activation_info = await nyx_agent.activate(scene_context)
    print(f"Nyx activated. Profile Mood: {nyx_agent.profile_manager.profile.personality.current_mood}, Power Dynamic: {nyx_agent.profile_manager.profile.personality.power_dynamic}")
    print(f"Initial action: {activation_info['initial_action']['type']} - {activation_info['initial_action']['content']}")

    logger.info("--- Simulating Relationship Interaction ---")
    interaction = { 
        "type": "vulnerability_shared",  # Using new interaction type
        "intensity": 0.7, 
        "success_rate": 0.6, 
        "context": "emotional_moment",
        "target_character": "player_hero" 
    }
    rel_update_result = await nyx_agent.update_relationship(interaction)
    print(f"Relationship update result: {rel_update_result}")
    
    if rel_update_result.get("success"):
        print(f"Impacts: {rel_update_result.get('impacts', {})}")
        print(f"New patterns: {rel_update_result.get('new_patterns', [])}")
        print(f"New archetypes: {rel_update_result.get('new_archetypes', [])}")
        if rel_update_result.get('event'):
            print(f"Event triggered: {rel_update_result['event']['type']}")

    logger.info("--- Checking Relationship State ---")
    manager = OptimizedRelationshipManager(user_id, conversation_id)
    rel_state = await manager.get_relationship_state(
        "npc", hash(nyx_agent.id),
        "player", hash("player_hero")
    )
    print(f"Current relationship dimensions: {rel_state.dimensions.to_dict()}")
    print(f"Active patterns: {list(rel_state.history.active_patterns)}")
    print(f"Active archetypes: {list(rel_state.active_archetypes)}")

    logger.info("--- Simulating Autonomous Thinking ---")
    think_result = await nyx_agent.think()
    print(f"Think cycle completed. Decisions: {think_result['decisions_made']}")
    for i, action_res in enumerate(think_result['actions_results']):
        print(f"  Action {i+1}:")
        print(f"    Decision: {action_res['decision']}")
        print(f"    Result: {action_res['result']}")

    logger.info("--- Simulating Reality Modification ---")
    mod = { 
        "type": "environmental", 
        "scope": "scene", 
        "duration": "temporary", 
        "parameters": {
            "area": "current_room", 
            "elements": ["chilling_wind", "shadows_deepen"], 
            "intensity": 0.7
        }
    }
    reality_mod_result = await nyx_agent.modify_reality(mod)
    print(f"Reality modification result: {reality_mod_result}")
    print(f"Scene {scene_context['scene_id']} state: {nyx_agent.universe_manager.state.active_scenes.get(scene_context['scene_id'], {}).get('reality_state')}")

    logger.info("--- Saving Final State ---")
    save_success = await save_npc_agent(nyx_agent)
    print(f"Final state saved: {save_success}")

if __name__ == "__main__":
    try:
        if not os.getenv("DATABASE_URL"):
             print("WARNING: DATABASE_URL environment variable not set. Database operations will likely fail.")
        asyncio.run(main())
    except Exception as e:
         logger.critical(f"Main execution failed: {e}", exc_info=True)
         print("\n--- ERROR ---")
         print("Ensure the database is running and connection details (e.g., DATABASE_URL env var) are correctly set.")
         print(f"Error details: {e}")
