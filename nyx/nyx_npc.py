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

logger = logging.getLogger(__name__)

nyx_action_content_generator = Agent(
    name="Nyx Action-Content Generator",
    instructions="""
You will receive JSON with:
  action_type      – one of "dominate","challenge","seduce","tease",
                      "manipulate","influence","interact".
  relationship     – brief dict (dominance, bond, familiarity, etc.)
  scene_context    – dict (scene_type, location, mood hints, etc.)

Return JSON **only** with:
  content          – a single sentence (≤ 160 chars) describing what Nyx does,
                     in her trademark sophisticated-dark style (PG-13).
""",
    model="gpt-4.1-nano",
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

# --- Data Models (Reusing from previous refactor) ---

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

class RelationshipModel(BaseModel):
    familiarity: float = 0.0; dominance: float = 0.8; emotional_bond: float = 0.0; manipulation_success: float = 0.0
    interaction_count: int = 0
    psychological_hooks: List[Dict[str, Any]] = Field(default_factory=list)
    emotional_triggers: List[Dict[str, Any]] = Field(default_factory=list)
    behavioral_patterns: List[str] = Field(default_factory=list)
    vulnerability_points: List[str] = Field(default_factory=list)
    power_dynamics: Dict[str, float] = Field(default_factory=lambda: {"submission_level": 0.0, "control_level": 0.8, "influence_strength": 0.0})

class StatusModel(BaseModel):
    is_active: bool = False; current_scene: Optional[str] = None; current_target: Optional[str] = None
    interaction_history: List[Dict[str, Any]] = Field(default_factory=list); reality_state: str = "stable"

class ProfileModel(BaseModel):
    name: str = "Nyx"; title: str = "The Omniscient Mistress"
    appearance: AppearanceModel = Field(default_factory=AppearanceModel)
    personality: PersonalityModel = Field(default_factory=PersonalityModel)
    abilities: AbilitiesModel = Field(default_factory=AbilitiesModel)
    relationships: Dict[str, RelationshipModel] = Field(default_factory=dict)
    status: StatusModel = Field(default_factory=StatusModel)

# Universe State Sub-Models
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

# Social Link Model
class SocialLinkModel(BaseModel):
    level: int = 0; experience: float = 0.0; milestones: List[Dict[str, Any]] = Field(default_factory=list)
    relationship_type: str = "complex"; interactions: List[Dict[str, Any]] = Field(default_factory=list); influence: float = 0.0

# Agenda Models
class GoalModel(BaseModel): # Simplified from previous - use dicts for now as in original
    id: str = Field(default_factory=lambda: f"goal_{random.randint(10000, 99999)}")
    type: str; description: str; priority: float = 0.5; status: str = "active"; progress: float = 0.0
    strategy: Dict[str, Any] = Field(default_factory=dict); creation_time: datetime = Field(default_factory=datetime.now)
    last_update: datetime = Field(default_factory=datetime.now); source_opportunity_id: Optional[str] = None

class OpportunityModel(BaseModel): # Simplified from previous
    id: str = Field(default_factory=lambda: f"opp_{random.randint(10000, 99999)}")
    type: str; target: Any; potential: float = 0.5; timing: Dict[str, Any] = Field(default_factory=dict)
    status: str = "new"; priority: float = 0.5; dependencies: List[str] = Field(default_factory=list)
    risks: Dict[str, float] = Field(default_factory=dict); creation_time: datetime = Field(default_factory=datetime.now)
    context: Dict[str, Any] = Field(default_factory=dict)

class NarrativeControlModel(BaseModel):
    current_threads: Dict[str, Any] = Field(default_factory=dict); planned_developments: Dict[str, Any] = Field(default_factory=dict)
    character_arcs: Dict[str, Any] = Field(default_factory=dict); plot_hooks: List[str] = Field(default_factory=list)

class AgendaModel(BaseModel):
    active_goals: List[Dict[str, Any]] = Field(default_factory=list) # Using Dict like original for goals
    long_term_plans: Dict[str, Any] = Field(default_factory=dict); current_schemes: Dict[str, Any] = Field(default_factory=dict)
    opportunity_tracking: Dict[str, Dict[str, Any]] = Field(default_factory=dict) # Using Dict like original for opportunities
    influence_web: Dict[str, Any] = Field(default_factory=dict); narrative_control: NarrativeControlModel = Field(default_factory=NarrativeControlModel)
    completed_goals: List[Dict[str, Any]] = Field(default_factory=list); archived_goals: List[Dict[str, Any]] = Field(default_factory=list)

# Autonomous State Model
class PlayerModel(BaseModel):
    behavior_patterns: Dict[str, Any] = Field(default_factory=dict); decision_history: List[Dict[str, Any]] = Field(default_factory=list)
    preference_model: Dict[str, Any] = Field(default_factory=dict); engagement_metrics: Dict[str, Any] = Field(default_factory=dict)
    # Adding fields from original calculations if needed later
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
    # Adding fields from original calculations if needed later
    world_state: Dict[str, Any] = Field(default_factory=dict) # For _identify_world_elements
    plot_metrics: Dict[str, float] = Field(default_factory=dict) # For _calculate_plot_readiness
    narrative_metrics: Dict[str, float] = Field(default_factory=dict) # For _calculate_narrative_momentum
    effect_metrics: Dict[str, float] = Field(default_factory=dict) # For _calculate_immediate_effect etc.

class AutonomousStateModel(BaseModel):
    awareness_level: float = 1.0; current_focus: Optional[str] = None; active_manipulations: Dict[str, Any] = Field(default_factory=dict)
    observed_patterns: Dict[str, Any] = Field(default_factory=dict); player_model: PlayerModel = Field(default_factory=PlayerModel)
    story_model: StoryModel = Field(default_factory=StoryModel)

# Omniscient Powers Model
class OmniscientPowersModel(BaseModel):
    reality_manipulation: bool = True; character_manipulation: bool = True; knowledge_access: bool = True
    scene_control: bool = True; fourth_wall_awareness: bool = True; plot_manipulation: bool = True
    hidden_influence: bool = True
    limitations: Dict[str, bool] = Field(default_factory=lambda: {"social_links": False, "player_agency": True})


# --- Database Interface (Reusing from previous refactor, adjusted save types) ---

class NyxDatabaseInterface(Protocol):
    async def save_agent_state(self, agent_id: str, agent_data: 'NPCAgentState') -> None: ...
    async def load_agent_state(self, agent_id: str) -> Optional['NPCAgentState']: ...
    async def log_reality_modification(self, agent_id: str, mod_data: Dict[str, Any]) -> None: ...
    async def save_character_state(self, agent_id: str, character_id: str, state_data: Dict[str, Any]) -> None: ...
    async def load_dynamic_lore(self, category: str) -> Optional[Dict[str, Any]]: ...
    async def save_scene_state(self, agent_id: str, scene_id: str, state_data: Dict[str, Any]) -> None: ...
    async def save_social_link_update(self, agent_id: str, link_data: SocialLinkModel, interaction_record: Dict[str, Any]) -> None: ...
    async def save_plot_thread(self, agent_id: str, thread_data: Dict[str, Any]) -> None: ...
    async def log_fourth_wall_break(self, agent_id: str, break_data: Dict[str, Any]) -> None: ...
    async def save_hidden_influence(self, agent_id: str, influence_data: Dict[str, Any]) -> None: ...
    async def save_opportunities(self, agent_id: str, opportunities: List[Tuple[str, Dict[str, Any]]]) -> None: ... # Saving dicts now
    async def save_goals(self, agent_id: str, goals: List[Dict[str, Any]]) -> None: ... # Saving dicts now

class AsyncpgNyxDatabase(NyxDatabaseInterface):
    @asynccontextmanager
    async def get_conn(self) -> asyncpg.Connection:
        async with get_db_connection_context() as conn: yield conn

    async def save_agent_state(self, agent_id: str, agent_data: 'NPCAgentState') -> None:
        async with self.get_conn() as conn:
            await conn.execute(
                """
                INSERT INTO npc_agent_state (agent_id, profile_data, universe_state_data, social_link_data, agenda_data, autonomous_state_data, omniscient_powers_data, timestamp)
                VALUES ($1, $2::jsonb, $3::jsonb, $4::jsonb, $5::jsonb, $6::jsonb, $7::jsonb, NOW())
                ON CONFLICT (agent_id) DO UPDATE SET
                    profile_data = EXCLUDED.profile_data, universe_state_data = EXCLUDED.universe_state_data,
                    social_link_data = EXCLUDED.social_link_data, agenda_data = EXCLUDED.agenda_data,
                    autonomous_state_data = EXCLUDED.autonomous_state_data, omniscient_powers_data = EXCLUDED.omniscient_powers_data,
                    timestamp = NOW();
                """,
                agent_id, agent_data.profile.dict(), agent_data.universe_state.dict(),
                agent_data.social_link.dict(), agent_data.agenda.dict(),
                agent_data.autonomous_state.dict(), agent_data.omniscient_powers.dict()
            )
            logger.info(f"Saved state for agent {agent_id}")

    async def load_agent_state(self, agent_id: str) -> Optional['NPCAgentState']:
        async with self.get_conn() as conn:
            row = await conn.fetchrow("SELECT profile_data, universe_state_data, social_link_data, agenda_data, autonomous_state_data, omniscient_powers_data FROM npc_agent_state WHERE agent_id = $1", agent_id)
            if row:
                logger.info(f"Loaded state for agent {agent_id}")
                try:
                    # Use parse_obj for robust parsing from dict
                    return NPCAgentState(
                        profile=ProfileModel.parse_obj(row['profile_data'] or {}),
                        universe_state=UniverseStateModel.parse_obj(row['universe_state_data'] or {}),
                        social_link=SocialLinkModel.parse_obj(row['social_link_data'] or {}),
                        agenda=AgendaModel.parse_obj(row['agenda_data'] or {}),
                        autonomous_state=AutonomousStateModel.parse_obj(row['autonomous_state_data'] or {}),
                        omniscient_powers=OmniscientPowersModel.parse_obj(row['omniscient_powers_data'] or {})
                    )
                except Exception as e: logger.error(f"Failed to parse loaded state for agent {agent_id}: {e}", exc_info=True); return None
            logger.warning(f"No state found for agent {agent_id}"); return None

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

    async def save_social_link_update(self, agent_id: str, link_data: SocialLinkModel, interaction_record: Dict[str, Any]) -> None:
        async with self.get_conn() as conn:
            async with conn.transaction():
                # Convert milestones back to plain dicts if they became models
                milestones_json = [m.dict() if isinstance(m, BaseModel) else m for m in link_data.milestones]
                await conn.execute("UPDATE agent_social_links SET level = $1, experience = $2, influence = $3, milestones = $4::jsonb WHERE agent_id = $5;",
                                   link_data.level, link_data.experience, link_data.influence, milestones_json, agent_id)
                await conn.execute("INSERT INTO social_link_interactions (agent_id, timestamp, type, impact, context, leveled_up) VALUES ($1, $2, $3, $4, $5::jsonb, $6);",
                                   agent_id, interaction_record["timestamp"], interaction_record["type"], interaction_record["impact"], interaction_record["context"], interaction_record["leveled_up"])
            logger.debug(f"Saved social link update for agent {agent_id}")

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

    def adapt_personality(self, scene_context: Dict[str, Any]):
        """Adapt NPC personality traits based on scene context (Original Logic)"""
        target = scene_context.get("target_character")
        scene_type = scene_context.get("scene_type")

        if target:
            self.profile.status.current_target = target
            relationship = self.get_or_create_relationship(target) # Use manager method
            self._adjust_traits_for_relationship(relationship) # Use manager method

        # Adjust power dynamic based on scene type (Original Logic)
        if scene_type == "confrontation":
            self.profile.personality.power_dynamic = 0.9
        elif scene_type == "seduction":
            self.profile.personality.power_dynamic = 0.7
        elif scene_type == "manipulation":
            self.profile.personality.power_dynamic = 0.8
        # Note: Original didn't have an else, might default to 1.0 from model

    def get_or_create_relationship(self, target_id: str) -> RelationshipModel:
        """Get existing relationship or create new one (Original Logic)"""
        if target_id not in self.profile.relationships:
            logger.debug(f"Creating new relationship profile for target: {target_id}")
            self.profile.relationships[target_id] = RelationshipModel(
                # Explicitly setting defaults from original _get_or_create_relationship
                familiarity=0.0, dominance=0.8, emotional_bond=0.0, manipulation_success=0.0,
                interaction_count=0, psychological_hooks=[], emotional_triggers=[],
                behavioral_patterns=[], vulnerability_points=[],
                power_dynamics={"submission_level": 0.0, "control_level": 0.8, "influence_strength": 0.0}
            )
        return self.profile.relationships[target_id]

    def _adjust_traits_for_relationship(self, relationship: RelationshipModel):
        """Adjust personality traits based on relationship (Original Logic)"""
        # Using the exact logic from the original _adjust_traits_for_relationship
        if relationship.familiarity < 0.3:
            self.profile.personality.adaptable_traits = ["mysterious", "aloof", "intriguing"]
        elif relationship.emotional_bond > 0.7:
            self.profile.personality.adaptable_traits = ["nurturing", "possessive", "intense"]
        elif relationship.manipulation_success > 0.8:
            self.profile.personality.adaptable_traits = ["controlling", "demanding", "strict"]
        # Note: Original didn't have an else to reset traits

    def update_relationship_on_interaction(self, target_id: str, interaction_outcome: Dict[str, Any]):
        """Updates relationship based on interaction results."""
        # This logic wasn't explicitly in the original but is needed
        relationship = self.get_or_create_relationship(target_id)
        relationship.interaction_count += 1
        # Example updates - refine based on actual game design needs
        relationship.familiarity = min(1.0, relationship.familiarity + 0.05 * interaction_outcome.get("rapport_gain", 0))
        relationship.emotional_bond = min(1.0, relationship.emotional_bond + 0.03 * interaction_outcome.get("emotional_impact", 0))
        relationship.manipulation_success = min(1.0, relationship.manipulation_success + 0.02 * interaction_outcome.get("manipulation_success", 0))
        # TODO: Add logic to update hooks, triggers based on outcome details


class UniverseStateManager:
    def __init__(self, universe_data: UniverseStateModel):
        self.state = universe_data
        # Add profile reference if needed by calculations moved here
        # self.profile_manager = profile_manager # If passed during init

    # --- Reality Modification ---
    def apply_reality_modification(self, modification: Dict[str, Any]) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Applies modification in-memory and calculates effects. Returns (record, effects) or None."""
        if not self._validate_reality_modification(modification):
             logger.warning(f"Validation failed for reality modification: {modification}")
             return None

        timestamp = datetime.now().isoformat()
        effects = self._calculate_reality_effects(modification) # Original logic
        scope = modification.get("scope", Scope.LOCAL.value)
        duration = modification.get("duration", Duration.PERMANENT.value)

        modification_record = {
            "timestamp": timestamp, "modification": modification, "scope": scope,
            "duration": duration, "effects": effects
        }
        self.state.reality_modifications.append(modification_record)
        self._update_internal_state_post_modification(modification, effects) # Original logic
        return modification_record, effects

    def _validate_reality_modification(self, modification: Dict[str, Any]) -> bool:
        """Validate if a reality modification is allowable (Original Logic)"""
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
        """Calculate the effects of a reality modification (Original Logic)"""
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

        # Calculate ripple effects (simplified from original)
        if base_effects["stability_impact"] > 0.5: base_effects["ripple_effects"].append("Reality fabric strain")
        if base_effects["power_cost"] > 5.0: base_effects["ripple_effects"].append("Temporal echoes")
        if len(base_effects["primary_effects"]) > 3: base_effects["ripple_effects"].append("Cascading changes")

        # TODO: Add secondary effects calculation logic if it existed in original
        # TODO: Add duration effects calculation logic if it existed in original

        return base_effects

    def _update_internal_state_post_modification(self, modification: Dict[str, Any], effects: Dict[str, Any]):
        """Update the universe state based on a modification (Original Logic)"""
        mod_type = modification["type"]
        scope = modification["scope"]

        # Update timeline if needed
        if mod_type == RealityModificationType.TEMPORAL:
            self.state.current_timeline = f"{self.state.current_timeline}_mod_{random.randint(100,999)}" # Simplified ID generation

        # Update active scenes if affected
        if scope in [Scope.SCENE.value, Scope.GLOBAL.value]:
            for scene_id, scene_data in self.state.active_scenes.items():
                # Ensure scene_data is a dict
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
             if isinstance(char_state, dict): # Ensure it's a dict
                 char_state["reality_impact"] = effects.get("stability_impact", 0.0) # Apply effect
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


    # --- Character Modification ---
    def apply_character_modifications(self, character_id: str, modifications: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply modifications to a character's state in-memory (Original Logic)"""
        # Get current state or initialize if doesn't exist (policy decision)
        current_state = self.state.character_states.get(character_id)
        if current_state is None:
            # Option 1: Fail if character doesn't exist
            # logger.warning(f"Character {character_id} not found for modification.")
            # return None
            # Option 2: Create a new character state (as in original _apply_character_modifications)
             logger.debug(f"Character {character_id} not found, creating new state.")
             current_state = {
                 "attributes": {}, "skills": {}, "status": {}, "personality": {},
                 "relationships": {}, "modifications_history": []
             }
        elif not isinstance(current_state, dict):
             logger.error(f"Character state for {character_id} is not a dictionary: {type(current_state)}. Cannot modify.")
             return None


        new_state = current_state.copy() # Work on a copy

        try:
            # Process each modification type (Original Logic)
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

            # Record modification in history (Original Logic)
            new_state.setdefault("modifications_history", []).append({
                "timestamp": datetime.now().isoformat(),
                "modifications": modifications,
                "applied_by": "Nyx" # Assuming Nyx is the agent
            })
            self.state.character_states[character_id] = new_state # Commit changes
            return new_state
        except Exception as e:
            logger.error(f"Error applying character modifications for {character_id}: {e}", exc_info=True)
            # State might be partially modified in 'new_state', but not committed to self.state
            return None # Indicate failure

    # --- Knowledge Access ---
    def get_knowledge(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Process a knowledge/lore query using internal state (Original Logic)"""
        query_type = query.get("type", "general")
        params = query.get("parameters", {})

        # Determine knowledge source based on query type
        knowledge = {}
        if query_type == "lore": knowledge = self._access_lore_database(params)
        elif query_type == "characters": knowledge = self._access_character_knowledge(params)
        elif query_type == "events": knowledge = self._access_event_knowledge(params)
        elif query_type == "relationships": knowledge = self._access_relationship_knowledge(params) # Needs profile access
        elif query_type == "timeline": knowledge = self._access_timeline_knowledge(params)
        else: knowledge = self._access_lore_database({"category": "general"}) # Default to general lore

        confidence = self._calculate_knowledge_confidence(query_type, params)
        related = self._find_related_knowledge(query_type, params) # Needs profile access

        response = {
            "query_type": query_type,
            "knowledge": knowledge,
            "confidence": confidence,
            "related_knowledge": related,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "source": "omniscient_knowledge", # As per original
                "access_level": "unlimited"
            }
        }
        return response

    # --- Knowledge Access Helpers (Original Logic) ---
    def _access_lore_database(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Access the lore database with given parameters"""
        return self.state.lore_database.get(params.get("category", "general"), {})

    def _access_character_knowledge(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Access character-specific knowledge"""
        return self.state.character_states.get(params.get("character_id", ""), {})

    def _access_event_knowledge(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Access event-specific knowledge"""
        # Assuming events might be tracked in causality or plot threads
        event_id = params.get("event_id", "")
        # Search causality tracking first
        for timestamp, record in self.state.causality_tracking.items():
            if record.get("modification", {}).get("event_id") == event_id: # Example check
                return record
        # Search plot threads
        for thread_id, thread in self.state.plot_threads.items():
             if thread_id == event_id or thread.get("related_event_id") == event_id: # Example check
                 return thread
        return {} # Not found

    def _access_relationship_knowledge(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Access relationship-specific knowledge"""
        # !! This requires access to the ProfileManager's state !!
        # This indicates a potential structural issue or the need for dependency injection here.
        # For now, we'll assume access is somehow provided or return empty.
        logger.warning("_access_relationship_knowledge needs profile access, returning empty.")
        # profile = self.profile_manager.profile # If profile_manager was injected
        # char_id = params.get("character_id", "")
        # relationship_model = profile.relationships.get(char_id)
        # return relationship_model.dict() if relationship_model else {}
        return {}


    def _access_timeline_knowledge(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Access timeline-specific knowledge"""
        return {
            "current_timeline": self.state.current_timeline,
            "modifications": self.state.reality_modifications # Return full list as per original
        }

    def _calculate_knowledge_confidence(self, query_type: str, params: Dict[str, Any]) -> float:
        """Calculate confidence level for knowledge access (Original Logic)"""
        base_confidence = 1.0
        if len(params) > 3: base_confidence *= 0.95
        if query_type in ["timeline", "relationships"]: base_confidence *= 0.98
        return min(1.0, base_confidence)

    def _find_related_knowledge(self, query_type: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find knowledge related to the current query (Original Logic)"""
        # !! Also requires profile access for relationships !!
        related = []
        if query_type == "characters":
            char_id = params.get("character_id", "")
            # relationship_data = self._access_relationship_knowledge({"character_id": char_id}) # Requires profile access
            # if relationship_data:
            #     related.append({"type": "relationship", "data": relationship_data})
            pass # Skip due to profile access need

        elif query_type == "events":
            event_id = params.get("event_id", "")
            # Add timeline knowledge if event is found?
            event_data = self._access_event_knowledge(params)
            if event_data:
                 related.append({
                     "type": "timeline",
                     "data": self._access_timeline_knowledge({"event_id": event_id})
                 })
        # TODO: Add more sophisticated related knowledge finding
        return related

    # --- Scene Control ---
    def apply_scene_modifications(self, scene_id: str, modifications: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply modifications to a scene (Original Logic)"""
        current_state = self.state.active_scenes.get(scene_id)
        if current_state is None:
            logger.debug(f"Scene {scene_id} not found, creating new state.")
            current_state = { # Default structure from original _apply_scene_modifications
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
                    # Original just extended, let's refine slightly for add/remove idea
                    current_participants = set(new_state.get("participants", []))
                    if isinstance(changes, list): current_participants.update(changes) # Add list
                    elif isinstance(changes, dict): # Handle add/remove dict
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

    # --- Plot Manipulation ---
    def add_plot_thread(self, plot_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Adds a new plot thread to the state (Original Logic for generation)"""
        try:
            thread_id = f"plot_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
            plot_thread = {
                "id": thread_id,
                "type": plot_data.get("type", "subtle_influence"),
                "elements": plot_data.get("elements", []),
                "visibility": plot_data.get("visibility", "hidden"),
                "influence_chain": self._create_influence_chain(plot_data), # Original logic
                "contingencies": self._generate_plot_contingencies(plot_data), # Original logic
                "meta_impact": self._calculate_meta_impact(plot_data), # Original logic
                "creation_time": datetime.now().isoformat(),
                "status": "active" # Default status
            }
            self.state.plot_threads[thread_id] = plot_thread
            # self._apply_plot_influences(plot_thread) # Original called this - apply separately if needed
            return plot_thread
        except Exception as e:
            logger.error(f"Error creating plot thread: {e}", exc_info=True)
            return None

    # --- Plot Manipulation Helpers (Original Logic) ---
    def _create_influence_chain(self, plot_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create a chain of subtle influences to achieve plot goals (Original Logic)"""
        chain = []
        elements = plot_data.get("elements", [])
        for element in elements:
            influence = {
                "target": element.get("target"),
                "type": element.get("type", "subtle"),
                "method": self._determine_influence_method(element), # Original helper
                "ripple_effects": self._calculate_ripple_effects(element), # Original helper (needs implementation)
                "detection_risk": self._calculate_detection_risk(element), # Original helper (needs implementation)
                "backup_plans": self._generate_backup_plans(element) # Original helper (needs implementation)
            }
            chain.append(influence)
        return chain

    def _determine_influence_method(self, element: Dict[str, Any]) -> str:
        """Determine the most effective method of influence (Original Logic)"""
        target_type = element.get("target_type", "npc")
        influence_goal = element.get("goal", "")
        methods = {
            "npc": ["whisper", "manipulate_circumstances", "plant_idea", "alter_perception"],
            "scene": ["atmospheric_change", "circumstantial_modification", "event_triggering"],
            "plot": ["thread_manipulation", "causality_adjustment", "narrative_shift"]
        }
        available_methods = methods.get(target_type, methods["npc"])
        # Original selected optimally, here just taking first as placeholder for selection logic
        return self._select_optimal_method(available_methods, influence_goal) # Original helper

    def _select_optimal_method(self, methods: List[str], goal: str) -> str:
        """Select the optimal influence method based on goal and context (Original Placeholder)"""
        # TODO: Implement actual selection logic based on effectiveness, risk, goal
        return methods[0] if methods else "unknown"

    def _calculate_ripple_effects(self, element: Dict[str, Any]) -> List[str]:
        """Placeholder for calculating ripple effects"""
        # TODO: Implement logic based on element type, target, magnitude
        return ["potential minor consequence"]

    def _calculate_detection_risk(self, element: Dict[str, Any]) -> float:
        """Placeholder for calculating detection risk"""
        # TODO: Implement logic based on visibility, method, target awareness
        return random.uniform(0.05, 0.4)

    def _generate_backup_plans(self, element: Dict[str, Any]) -> List[str]:
        """Placeholder for generating backup plans"""
        # TODO: Implement logic based on element goals and potential failures
        return ["fallback_plan_A"]

    def _generate_plot_contingencies(self, plot_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate contingency plans for plot manipulation (Original Logic)"""
        contingencies = []
        risk_factors = self._analyze_risk_factors(plot_data) # Original helper
        for risk in risk_factors:
            contingency = {
                "trigger": {"type": risk["type"], "threshold": risk["threshold"], "conditions": risk["conditions"]},
                "response": {
                    "primary": self._generate_primary_response(risk), # Original helper
                    "backup": self._generate_backup_response(risk), # Original helper
                    "cleanup": self._generate_cleanup_response(risk) # Original helper
                },
                "impact_mitigation": self._generate_impact_mitigation(risk) # Original helper
            }
            contingencies.append(contingency)
        return contingencies

    def _analyze_risk_factors(self, plot_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze potential risks in plot manipulation (Original Logic)"""
        risks = []
        risks.append({"type": "detection", "threshold": 0.7, "conditions": ["player_awareness", "npc_insight", "narrative_inconsistency"]})
        risks.append({"type": "interference", "threshold": 0.6, "conditions": ["player_agency", "npc_resistance", "plot_resilience"]})
        risks.append({"type": "cascade", "threshold": 0.8, "conditions": ["plot_stability", "reality_integrity", "causality_balance"]})
        # TODO: Add more sophisticated risk analysis based on plot_data specifics
        return risks

    def _generate_primary_response(self, risk: Dict[str, Any]) -> Dict[str, Any]:
        """Generate primary response to risk (Original Logic Structure)"""
        response_type = "redirect" if risk["type"] == "detection" else "stabilize"
        return {"type": response_type, "method": self._select_response_method(risk), "execution": self._plan_response_execution(risk)}
    def _generate_backup_response(self, risk: Dict[str, Any]) -> Dict[str, Any]:
        """Generate backup response to risk (Original Logic Structure)"""
        response_type = "contain" if risk["type"] == "cascade" else "obscure"
        return {"type": response_type, "method": self._select_backup_method(risk), "execution": self._plan_backup_execution(risk)}
    def _generate_cleanup_response(self, risk: Dict[str, Any]) -> Dict[str, Any]:
        """Generate cleanup response to risk (Original Logic Structure)"""
        return {"type": "normalize", "method": self._select_cleanup_method(risk), "execution": self._plan_cleanup_execution(risk)}
    def _generate_impact_mitigation(self, risk: Dict[str, Any]) -> Dict[str, Any]:
        """Generate impact mitigation strategy (Original Logic Structure)"""
        return {"immediate": self._generate_immediate_mitigation(risk), "long_term": self._generate_long_term_mitigation(risk), "narrative": self._generate_narrative_mitigation(risk)}

    # --- Placeholders for response/mitigation planning ---
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
        """Calculate meta-game impact of plot manipulation (Original Logic)"""
        return {
            "narrative_coherence": self._calculate_narrative_impact(plot_data), # Original helper
            "player_agency": self._calculate_agency_impact(plot_data), # Original helper
            "game_balance": self._calculate_balance_impact(plot_data), # Original helper
            "story_progression": self._calculate_progression_impact(plot_data) # Original helper
        }

    def _calculate_narrative_impact(self, plot_data: Dict[str, Any]) -> float:
        """Calculate impact on narrative coherence (Original Logic - simplified)"""
        # Original was just base_impact * (1 + len(elements) * 0.1)
        # Let's make it slightly more nuanced: coherence decreases with more elements or hidden visibility
        base_impact = 1.0 # Start with perfect coherence
        elements = plot_data.get("elements", [])
        visibility = plot_data.get("visibility", "hidden")
        coherence_reduction = len(elements) * 0.05 + (0.1 if visibility == "hidden" else 0.0)
        return max(0.0, base_impact - coherence_reduction)

    def _calculate_agency_impact(self, plot_data: Dict[str, Any]) -> float:
        """Calculate impact on player agency (Original Logic - negative impact)"""
        # Original was base_impact * (0.5 if visibility == "hidden" else 1.0)
        # Let's represent impact as reduction from 1.0 (full agency)
        base_reduction = 0.05 # Minimal impact even if visible
        visibility = plot_data.get("visibility", "hidden")
        agency_reduction = base_reduction * (2.0 if visibility == "hidden" else 1.0) # Hidden manipulation impacts agency perception more? Or less? Let's say less direct impact.
        agency_reduction += len(plot_data.get("elements", [])) * 0.02 # More elements = more manipulation
        return max(0.0, 1.0 - agency_reduction) # Return remaining agency factor

    def _calculate_balance_impact(self, plot_data: Dict[str, Any]) -> float:
        """Calculate impact on game balance (Original Logic - neutral impact)"""
        # Original was base_impact * type_multiplier
        # This needs game-specific logic. Returning neutral impact placeholder.
        return 0.0 # 0 means no change from baseline balance

    def _calculate_progression_impact(self, plot_data: Dict[str, Any]) -> float:
        """Calculate impact on story progression (Original Logic - positive impact)"""
        # Original was base_impact * (1 + len(elements) * 0.15)
        # Let's represent as a multiplier on base progression speed
        base_multiplier = 1.0
        elements = plot_data.get("elements", [])
        type_multiplier = 1.1 if plot_data.get("type") == "subtle_influence" else 1.3 # More direct manipulation speeds up more
        progression_multiplier = base_multiplier * type_multiplier * (1 + len(elements) * 0.05)
        return progression_multiplier


    # --- Fourth Wall Breaking ---
    def add_fourth_wall_break(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Adds a fourth wall break record (Original Logic for generation)"""
        try:
            break_id = f"break_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
            break_point = {
                "id": break_id,
                "type": context.get("type", "subtle"),
                "target": context.get("target", "narrative"),
                "method": self._determine_break_method(context), # Original logic
                "meta_elements": self._gather_meta_elements(context), # Original logic
                "player_impact": self._calculate_player_impact(context), # Original logic
                "timestamp": datetime.now().isoformat()
            }
            # Append to the list within the Pydantic model
            self.state.meta_awareness.breaking_points.append(break_point)
            return break_point
        except Exception as e:
            logger.error(f"Error creating fourth wall break record: {e}", exc_info=True)
            return None

    # --- Fourth Wall Helpers (Original Logic) ---
    def _determine_break_method(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Determine how to break the fourth wall effectively (Original Logic)"""
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
            "execution": self._plan_break_execution(target, intensity), # Original helper
            "concealment": self._calculate_break_concealment(target, intensity) # Original helper
        }

    def _plan_break_execution(self, target: str, intensity: str) -> str:
        """Placeholder for planning execution details"""
        return f"Execute {intensity} break targeting {target}"

    def _calculate_break_concealment(self, target: str, intensity: str) -> float:
        """Placeholder for calculating concealment factor"""
        concealment = {"subtle": 0.9, "moderate": 0.5, "overt": 0.1}.get(intensity, 0.9)
        return concealment

    def _gather_meta_elements(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Gather meta-game elements for fourth wall breaking (Original Logic)"""
        return {
            "player_state": self._get_player_meta_state(), # Original helper
            "game_mechanics": self._get_relevant_mechanics(context), # Original helper
            "narrative_elements": self._get_narrative_elements(context), # Original helper
            "fourth_wall_status": self._get_fourth_wall_status() # Original helper
        }

    def _get_player_meta_state(self) -> Dict[str, Any]:
        """Get current player meta-state information (Original Logic)"""
        # Accessing state directly - assumes this manager has access or it's passed
        return self.state.meta_awareness.player_knowledge

    def _get_relevant_mechanics(self, context: Dict[str, Any]) -> List[str]:
        """Get relevant game mechanics for the context (Original Logic)"""
        mechanics = []
        intensity = context.get("intensity", "subtle") # Use intensity instead of type here based on original _determine_break_method usage
        if intensity == "subtle": mechanics.extend(["social_links", "character_stats", "scene_mechanics"])
        elif intensity == "moderate": mechanics.extend(["game_systems", "progression", "relationship_dynamics"])
        else: mechanics.extend(["meta_mechanics", "game_structure", "narrative_control"])
        return mechanics

    def _get_narrative_elements(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get relevant narrative elements (Original Logic Structure)"""
        return {
            "current_layer": self._get_current_narrative_layer(), # Original helper
            "available_breaks": self._get_available_break_points(), # Original helper
            "narrative_state": self._get_narrative_state() # Original helper
        }

    # --- Placeholders for narrative element helpers ---
    def _get_current_narrative_layer(self): return self.state.meta_awareness.narrative_layers[-1] if self.state.meta_awareness.narrative_layers else "base"
    def _get_available_break_points(self): return len(self.state.meta_awareness.breaking_points) # Simplified
    def _get_narrative_state(self): return {"cohesion": 0.8, "tension": 0.6} # Placeholder state

    def _get_fourth_wall_status(self) -> Dict[str, Any]:
        """Get current status of fourth wall integrity (Original Logic Structure)"""
        return {
            "integrity": self._calculate_wall_integrity(), # Original helper
            "break_points": self._get_active_break_points(), # Original helper
            "player_awareness": self._get_player_awareness_level() # Original helper
        }

    # --- Placeholders for fourth wall status helpers ---
    def _calculate_wall_integrity(self): return max(0.0, 1.0 - len(self.state.meta_awareness.breaking_points) * 0.05)
    def _get_active_break_points(self): return [bp['id'] for bp in self.state.meta_awareness.breaking_points[-3:]] # Last 3
    def _get_player_awareness_level(self): return self.state.meta_awareness.player_knowledge.get("meta_awareness_score", 0.1) # Example


    def _calculate_player_impact(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate the impact of fourth wall breaking on the player (Original Logic)"""
        return {
            "immediate": self._calculate_immediate_impact(context), # Original helper
            "long_term": self._calculate_long_term_impact(context), # Original helper
            "meta_awareness": self._calculate_meta_awareness_impact(context) # Original helper
        }

    def _calculate_immediate_impact(self, context: Dict[str, Any]) -> float:
        """Calculate immediate impact of fourth wall break (Original Logic)"""
        base_impact = 0.5
        break_type = context.get("type", "subtle") # Should likely use intensity based on _determine_break_method
        intensity = context.get("intensity", break_type) # Use intensity if present, else type
        if intensity == "subtle": return base_impact * 0.5
        elif intensity == "moderate": return base_impact * 1.0
        else: return base_impact * 2.0 # overt

    def _calculate_long_term_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate long-term impact of fourth wall break (Original Logic)"""
        # !! Needs access to social link !!
        # Assuming access for now
        social_link_influence = 0.0 # Placeholder if access fails
        # if self.social_link_manager: social_link_influence = self.social_link_manager.link.influence

        intensity = context.get("intensity", "subtle")
        trust_multiplier = 2.0 if intensity == "subtle" else 1.0 # Subtle breaks might build trust more? Original logic.

        return {
            "meta_awareness": 0.1 * (1 + len(self.state.meta_awareness.breaking_points)),
            "trust": 0.05 * trust_multiplier,
            "engagement": 0.15 * (1 + social_link_influence)
        }

    def _calculate_meta_awareness_impact(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate impact on player's meta-game awareness (Original Logic)"""
        current_awareness = len(self.state.meta_awareness.breaking_points) * 0.1
        intensity_factor = {"subtle": 0.05, "moderate": 0.1, "overt": 0.2}.get(context.get("intensity", "subtle"), 0.05)
        return {
            "game_awareness": min(1.0, current_awareness + intensity_factor * 1.0),
            "nyx_awareness": min(1.0, current_awareness + intensity_factor * 0.5), # Less impact on Nyx awareness?
            "narrative_awareness": min(1.0, current_awareness + intensity_factor * 1.5) # More impact on narrative awareness?
        }

    # --- Hidden Influence ---
    def add_hidden_influence(self, influence_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Adds a hidden influence record (Original Logic for generation)"""
        try:
            influence_id = f"influence_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
            influence = {
                "id": influence_id,
                "type": influence_data.get("type", "subtle"),
                "target": influence_data.get("target"),
                "method": self._create_hidden_influence_method(influence_data), # Original logic
                "layers": self._create_influence_layers(influence_data), # Original logic
                "proxies": self._select_influence_proxies(influence_data), # Original logic
                "contingencies": self._plan_influence_contingencies(influence_data), # Original logic
                "timestamp": datetime.now().isoformat(),
                "status": "planned" # Default status
            }
            self.state.hidden_influences[influence_id] = influence
            # self._apply_hidden_influence(influence) # Apply separately if needed
            return influence
        except Exception as e:
            logger.error(f"Error creating hidden influence record: {e}", exc_info=True)
            return None

    # --- Hidden Influence Helpers (Original Logic) ---
    def _create_hidden_influence_method(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a method for hidden influence (Original Logic)"""
        target_type = data.get("target_type", "npc")
        methods = {
            "npc": {"primary": self._create_npc_influence_method(data), "backup": self._create_backup_influence_method(data)},
            "scene": {"primary": self._create_scene_influence_method(data), "backup": self._create_backup_influence_method(data)},
            "plot": {"primary": self._create_plot_influence_method(data), "backup": self._create_backup_influence_method(data)}
        }
        return methods.get(target_type, methods["npc"])

    def _create_npc_influence_method(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create method for influencing NPCs (Original Logic Structure)"""
        return {"type": "psychological", "approach": "subtle_manipulation", "execution": self._plan_npc_influence_execution(data)}
    def _create_scene_influence_method(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create method for influencing scenes (Original Logic Structure)"""
        return {"type": "environmental", "approach": "circumstantial_modification", "execution": self._plan_scene_influence_execution(data)}
    def _create_plot_influence_method(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create method for influencing plot (Original Logic Structure)"""
        return {"type": "narrative", "approach": "causal_manipulation", "execution": self._plan_plot_influence_execution(data)}
    def _create_backup_influence_method(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create backup method for influence (Original Logic Structure)"""
        return {"type": "contingency", "approach": "alternative_path", "execution": self._plan_backup_influence_execution(data)}

    # --- Placeholders for influence execution planning ---
    def _plan_npc_influence_execution(self, data): return {"steps": ["plant_seed"]}
    def _plan_scene_influence_execution(self, data): return {"steps": ["alter_lighting"]}
    def _plan_plot_influence_execution(self, data): return {"steps": ["introduce_rumor"]}
    def _plan_backup_influence_execution(self, data): return {"steps": ["use_proxy_b"]}


    def _create_influence_layers(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create layers of influence to obscure the source (Original Logic)"""
        layers = []
        depth = data.get("depth", 3)
        for i in range(depth):
            layer = {
                "level": i + 1,
                "type": self._determine_layer_type(i, depth), # Original helper
                "cover": self._generate_layer_cover(i, data), # Original helper
                "contingency": self._create_layer_contingency(i, data) # Original helper
            }
            layers.append(layer)
        return layers

    def _determine_layer_type(self, level: int, depth: int) -> str:
        """Determine the type of influence layer (Original Logic)"""
        if level == 0: return "direct" # Closest to source
        elif level == depth - 1: return "observable" # Furthest, most likely seen
        else: return "intermediate"

    def _generate_layer_cover(self, level: int, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate cover for an influence layer (Original Logic)"""
        target_type = data.get("target_type", "npc")
        covers = {
            "npc": ["circumstantial", "emotional", "rational", "instinctive"],
            "scene": ["natural", "coincidental", "logical", "atmospheric"],
            "plot": ["narrative", "causal", "thematic", "dramatic"]
        }
        cover_type = random.choice(covers.get(target_type, covers["npc"]))
        return {"type": cover_type, "believability": max(0.1, 0.8 - (level * 0.1)), "durability": min(1.0, 0.7 + (level * 0.1))}

    def _create_layer_contingency(self, level: int, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create contingency for an influence layer (Original Logic Structure)"""
        return {
            "trigger_condition": f"layer_{level}_compromise",
            "response_type": "redirect" if level < 2 else "abandon", # Example logic
            "backup_layer": self._create_backup_layer(level, data) # Original helper
        }

    def _create_backup_layer(self, level: int, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a backup layer for contingency (Original Logic Structure)"""
        return {"type": "fallback", "method": self._determine_backup_method(level), "execution": self._plan_backup_execution(data)}

    # --- Placeholders for backup planning ---
    def _determine_backup_method(self, level): return "alternative_cover"
    def _plan_backup_execution(self, data): return {"steps": ["execute_backup"]}


    def _select_influence_proxies(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Select proxies to carry out the influence (Original Logic)"""
        proxy_count = data.get("proxy_count", 2)
        proxies = []
        for _ in range(proxy_count):
            proxy = {
                "type": self._determine_proxy_type(data), # Original helper
                "awareness": self._calculate_proxy_awareness(), # Original helper
                "reliability": self._calculate_proxy_reliability(), # Original helper
                "contingency": self._create_proxy_contingency() # Original helper
            }
            proxies.append(proxy)
        return proxies

    def _determine_proxy_type(self, data: Dict[str, Any]) -> str:
        """Determine the type of proxy to use (Original Logic)"""
        target_type = data.get("target_type", "npc")
        # influence_type = data.get("type", "subtle") # Not used in original random choice
        proxy_types = {
            "npc": ["unwitting", "partial", "conscious"],
            "scene": ["environmental", "circumstantial", "direct"],
            "plot": ["thematic", "causal", "direct"]
        }
        available_types = proxy_types.get(target_type, proxy_types["npc"])
        return random.choice(available_types)

    def _calculate_proxy_awareness(self) -> float:
        """Calculate proxy's awareness level (Original Logic)"""
        return random.uniform(0.0, 0.3)

    def _calculate_proxy_reliability(self) -> float:
        """Calculate proxy's reliability (Original Logic)"""
        return random.uniform(0.7, 0.9)

    def _create_proxy_contingency(self) -> Dict[str, Any]:
        """Create contingency plan for proxy (Original Logic)"""
        return {"detection_response": "redirect", "failure_response": "replace", "cleanup_protocol": "memory_adjustment"}


    def _plan_influence_contingencies(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Plan contingencies for the influence operation (Original Logic)"""
        contingency_count = data.get("contingency_count", 2)
        contingencies = []
        for i in range(contingency_count):
            contingency = {
                "trigger": self._create_contingency_trigger(i, data), # Original helper
                "response": self._create_contingency_response(i, data), # Original helper
                "probability": max(0.05, 0.2 + (i * 0.1)) # Ensure non-zero probability
            }
            contingencies.append(contingency)
        return contingencies

    def _create_contingency_trigger(self, level: int, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a trigger for a contingency (Original Logic)"""
        trigger_type = "detection_risk" if level == 0 else "execution_failure"
        return {"type": trigger_type, "threshold": max(0.1, 0.7 - (level * 0.1)), "conditions": []} # TODO: Populate conditions

    def _create_contingency_response(self, level: int, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a response for a contingency (Original Logic Structure)"""
        response_type = "redirect" if level == 0 else "abandon"
        return {"type": response_type, "method": self._determine_contingency_method(level, data), "backup_plan": self._create_backup_plan(level, data)}

    # --- Placeholders for contingency response planning ---
    def _determine_contingency_method(self, level, data): return "default_contingency_method"
    def _create_backup_plan(self, level, data): return {"steps": ["contingency_step1"]}


class SocialLinkManager:
    def __init__(self, social_link_data: SocialLinkModel):
        self.link = social_link_data

    def update_on_interaction(self, interaction_data: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Calculates XP gain, updates link state, and returns gain + record."""
        exp_gain = self._calculate_social_link_experience(interaction_data) # Original logic method
        original_exp = self.link.experience
        original_level = self.link.level

        self.link.experience += exp_gain
        leveled_up = self._check_social_link_level_up() # Original logic method

        interaction_record = {
            "timestamp": datetime.now().isoformat(),
            "type": interaction_data.get("type", "unknown"),
            "impact": exp_gain,
            "context": interaction_data.get("context"),
            "leveled_up": leveled_up
        }
        self.link.interactions.append(interaction_record)

        logger.debug(f"Social link updated. XP Gain: {exp_gain}, New XP: {self.link.experience}, Level: {self.link.level}")
        return exp_gain, interaction_record

    def _calculate_social_link_experience(self, interaction_data: Dict[str, Any]) -> float:
        """Calculate experience gain for social link (Original Logic)"""
        base_exp = 10.0
        interaction_type = interaction_data.get("type", "basic")
        intensity = interaction_data.get("intensity", 1.0)
        success_rate = interaction_data.get("success_rate", 0.5)
        depth = interaction_data.get("depth", 1.0)

        type_multipliers = {"basic": 1.0, "emotional": 1.5, "intellectual": 1.3, "psychological": 1.8, "intimate": 2.0, "confrontational": 1.6}
        type_multiplier = type_multipliers.get(interaction_type, 1.0)

        experience = base_exp * type_multiplier * intensity * success_rate * depth
        level_scaling = 1.0 + (self.link.level * 0.1)
        experience /= max(1.0, level_scaling) # Avoid division by zero or negative scaling
        influence_modifier = 1.0 + (self.link.influence * 0.2)
        experience *= influence_modifier

        return round(max(0.0, experience), 2)

    def _check_social_link_level_up(self) -> bool:
        """Check and process social link level ups (Original Logic)"""
        leveled_up = False
        base_exp_required = 100
        # Loop in case of multiple level ups from one interaction
        while True:
            current_level = self.link.level
            exp_required = base_exp_required * (1.5 ** current_level)

            if self.link.experience >= exp_required:
                self.link.level += 1
                self.link.experience -= exp_required
                leveled_up = True

                new_abilities = self._generate_level_up_abilities(self.link.level) # Original helper
                self.link.milestones.append({
                    "type": "level_up", "from_level": current_level, "to_level": self.link.level,
                    "timestamp": datetime.now().isoformat(), "exp_required": exp_required, "new_abilities": new_abilities
                })
                self.link.influence = min(1.0, self.link.influence + 0.05)
                logger.info(f"Social Link Leveled Up! Level {self.link.level}. Abilities: {new_abilities}")
            else:
                break # Exit loop if not enough XP for next level

        self.link.experience = max(0.0, self.link.experience) # Ensure non-negative XP
        return leveled_up

    def _generate_level_up_abilities(self, new_level: int) -> List[str]:
        """Generate new abilities unlocked at level up (Original Logic)"""
        ability_pools = {
            "psychological": ["enhanced_insight", "emotional_resonance", "mental_fortitude", "psychological_manipulation"],
            "reality": ["local_reality_bend", "temporal_glimpse", "environmental_control", "metaphysical_touch"],
            "relationship": ["deeper_understanding", "emotional_bond", "trust_foundation", "influence_growth"]
        }
        new_abilities = []
        if new_level % 3 == 0: new_abilities.append(random.choice(ability_pools["psychological"]))
        if new_level % 4 == 0: new_abilities.append(random.choice(ability_pools["reality"]))
        if new_level % 5 == 0: new_abilities.append(random.choice(ability_pools["relationship"]))
        return new_abilities if new_abilities else ["minor_influence_increase"]


class AgendaManager:
    def __init__(self, agenda_data: AgendaModel):
        self.agenda = agenda_data

    async def update_agenda(self, state_analysis: Dict[str, Any], db: NyxDatabaseInterface, agent_id: str):
        """Update goals, plans, schemes based on analysis. Saves changes to DB."""
        goals_to_save = self._update_active_goals(state_analysis) # Original logic moved here
        self._adjust_long_term_plans(state_analysis) # Original logic moved here
        self._update_current_schemes(state_analysis) # Original logic moved here

        if goals_to_save:
            try:
                await db.save_goals(agent_id, goals_to_save)
            except Exception as e:
                 logger.error(f"Failed to save updated goals for agent {agent_id}: {e}", exc_info=True)

    def _update_active_goals(self, state_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Update status, priority, and strategy of active goals. Returns goals that changed. (Original Logic)"""
        updated_goals = []
        goals_changed = []
        goals_to_remove_indices = []

        for i, goal in enumerate(self.agenda.active_goals):
            changed = False
            original_priority = goal.get("priority", 0.5)
            original_progress = goal.get("progress", 0.0)
            original_status = goal.get("status", "active")

            # Calculate progress and relevance (Original Logic - placeholders inside helpers)
            new_progress = min(1.0, original_progress + self._calculate_goal_progress(goal)) # Original helper
            relevance = self._evaluate_goal_relevance(goal, state_analysis) # Original helper

            if new_progress != original_progress:
                goal["progress"] = new_progress
                changed = True

            if goal["progress"] >= 1.0:
                goal["status"] = "completed"
                self.agenda.completed_goals.append({ # Using dict as per original model
                    "goal": goal, "completion_time": datetime.now().isoformat(),
                    "outcome": self._evaluate_goal_outcome(goal) # Original helper
                })
                goals_to_remove_indices.append(i)
                logger.info(f"Goal {goal.get('id')} completed.")
                changed = True
            elif relevance < 0.3:
                goal["status"] = "archived"
                self.agenda.archived_goals.append({ # Using dict as per original model
                    "goal": goal, "archive_time": datetime.now().isoformat(),
                    "reason": "low_relevance"
                })
                goals_to_remove_indices.append(i)
                logger.info(f"Goal {goal.get('id')} archived due to low relevance.")
                changed = True
            else:
                # Update priority and strategy if still active
                new_priority = self._calculate_goal_priority(goal, state_analysis) # Original helper
                if new_priority != original_priority:
                    goal["priority"] = new_priority
                    changed = True
                if self._should_update_strategy(goal, state_analysis): # Original helper
                    goal["strategy"] = self._generate_goal_strategy(goal, state_analysis) # Original helper
                    changed = True
                goal["last_update"] = datetime.now().isoformat()
                # Keep in the list to be potentially re-saved if changed
                # updated_goals.append(goal) # Don't add here, add below after removal

            if changed: # Record any changed goal for saving
                goals_changed.append(goal)

        # Remove completed/archived goals from active list
        active_goals_remaining = []
        for i, goal in enumerate(self.agenda.active_goals):
            if i not in goals_to_remove_indices:
                active_goals_remaining.append(goal)
        self.agenda.active_goals = active_goals_remaining


        # Generate new goals if needed (Original Logic - placeholders inside helpers)
        newly_generated_goals = []
        # Original used opportunities from narrative state, let's mimic that structure
        narrative_opportunities = state_analysis.get("narrative_state", {}).get("narrative_opportunities", [])
        while len(self.agenda.active_goals) < 3: # Maintain minimum active goals
            if not narrative_opportunities: break # No opportunities found in analysis

            # Original selected based on 'value', let's use priority from opportunity tracking if available
            # This requires opportunities to be tracked first. Let's use a simpler approach based on original example.
            # best_opp = max(narrative_opportunities, key=lambda x: x.get("value", 0)) # Original approach

            # Let's generate a placeholder goal if opportunities aren't readily available in expected format
            new_goal = {
                "id": f"goal_{random.randint(10000, 99999)}",
                "type": "opportunity_based", # Original type
                "description": "Generated from state analysis", # Placeholder description
                # "source": best_opp, # Original source
                "priority": self._calculate_initial_priority({}), # Original helper (needs opp data)
                "strategy": self._generate_initial_strategy({}), # Original helper (needs opp data)
                "creation_time": datetime.now().isoformat(),
                "progress": 0.0,
                "status": "active"
            }
            self.agenda.active_goals.append(new_goal)
            newly_generated_goals.append(new_goal)
            goals_changed.append(new_goal) # Also needs saving
            logger.info(f"Generated new goal: {new_goal.get('id')}")
            # Avoid infinite loop if generation fails or no opps
            if not new_goal: break


        # Sort remaining active goals by priority
        self.agenda.active_goals.sort(key=lambda g: g.get("priority", 0), reverse=True)

        return goals_changed # Return all goals that need DB update/insert

    def _adjust_long_term_plans(self, state_analysis: Dict[str, Any]):
        """Adjust long-term plans based on current state (Original Placeholder)"""
        # TODO: Implement logic from original if it existed, or add new logic
        pass

    def _update_current_schemes(self, state_analysis: Dict[str, Any]):
        """Update current schemes based on current state (Original Placeholder)"""
        # TODO: Implement logic from original if it existed, or add new logic
        pass

    # --- Goal Helpers (Original Logic) ---
    def _calculate_goal_progress(self, goal: Dict[str, Any]) -> float:
        """Placeholder for calculating goal progress"""
        # TODO: Implement actual progress calculation based on goal type and state
        return random.uniform(0.0, 0.1) # Simulate random progress

    def _evaluate_goal_relevance(self, goal: Dict[str, Any], state_analysis: Dict[str, Any]) -> float:
        """Placeholder for evaluating goal relevance"""
        # TODO: Implement actual relevance calculation based on goal and current state
        return random.uniform(0.2, 1.0)

    def _evaluate_goal_outcome(self, goal: Dict[str, Any]) -> str:
        """Placeholder for evaluating goal outcome"""
        # TODO: Implement outcome evaluation based on goal type and final state
        return "success" if random.random() > 0.2 else "partial_success"

    def _calculate_goal_priority(self, goal: Dict[str, Any], state_analysis: Dict[str, Any]) -> float:
        """Placeholder for calculating goal priority"""
        # TODO: Implement priority calculation based on goal type, urgency, impact, state
        return goal.get("priority", 0.5) * random.uniform(0.9, 1.1) # Slight random adjustment

    def _should_update_strategy(self, goal: Dict[str, Any], state_analysis: Dict[str, Any]) -> bool:
        """Placeholder for deciding if goal strategy needs update"""
        # TODO: Implement logic based on state changes, strategy effectiveness etc.
        return random.random() < 0.1 # Rarely update strategy

    def _generate_goal_strategy(self, goal: Dict[str, Any], state_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder for generating/updating goal strategy"""
        # TODO: Implement strategy generation based on goal type and state
        return {"steps": [f"updated_step_{random.randint(1,3)}"]}

    def _calculate_initial_priority(self, opportunity: Dict[str, Any]) -> float:
        """Placeholder for calculating initial goal priority from opportunity"""
        return opportunity.get("priority", random.random())

    def _generate_initial_strategy(self, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder for generating initial goal strategy from opportunity"""
        return {"steps": [f"initial_step_for_{opportunity.get('type', 'generic')}"]}


    # --- Opportunity Tracking (Original Logic) ---
    async def track_new_opportunities(self, state_analysis: Dict[str, Any], db: NyxDatabaseInterface, agent_id: str):
        """Identify, track, and save new/updated opportunities."""
        ops_to_save = self._track_new_opportunities_logic(state_analysis) # Perform in-memory logic
        if ops_to_save:
            try:
                # Pass list of tuples (id, dict) to save function
                await db.save_opportunities(agent_id, ops_to_save)
            except Exception as e:
                logger.error(f"Failed to save updated opportunities for agent {agent_id}: {e}", exc_info=True)
                # Consider rollback or flagging opportunities as unsaved

    def _track_new_opportunities_logic(self, state_analysis: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        """Identify and track new opportunities, returning those newly added/updated. (Original Logic)"""
        # This combines original _track_new_opportunities and its helpers
        identified_ops_data = self._identify_current_opportunities(state_analysis) # Original helper
        ops_to_save = []

        for opportunity_data in identified_ops_data: # Assuming identify returns list of dicts
            opportunity_id = self._generate_opportunity_id(opportunity_data) # Original helper
            existing_opp = self.agenda.opportunity_tracking.get(opportunity_id)

            if not existing_opp:
                # Create new opportunity dict (as original used dicts)
                new_opp_data = {
                    "id": opportunity_id,
                    "type": opportunity_data.get("type", "generic"),
                    "target": opportunity_data.get("target", "unknown"),
                    "potential": self._calculate_opportunity_potential(opportunity_data), # Original helper
                    "timing": self._calculate_opportunity_timing(opportunity_data), # Original helper
                    "status": "new",
                    "priority": self._calculate_opportunity_priority(opportunity_data), # Original helper
                    "dependencies": self._identify_opportunity_dependencies(opportunity_data), # Original helper
                    "risks": self._assess_opportunity_risks(opportunity_data), # Original helper
                    "creation_time": datetime.now().isoformat(), # Use ISO format string
                    "context": state_analysis # Store context when identified
                }
                self.agenda.opportunity_tracking[opportunity_id] = new_opp_data
                ops_to_save.append((opportunity_id, new_opp_data))
                logger.debug(f"Tracked new opportunity: {opportunity_id}")
            else:
                # Update existing opportunity (Original Logic)
                updated = self._update_existing_opportunity(opportunity_id, opportunity_data) # Original helper returns bool
                if updated:
                    # Append the updated dict for saving
                    ops_to_save.append((opportunity_id, self.agenda.opportunity_tracking[opportunity_id]))
                    logger.debug(f"Updated existing opportunity: {opportunity_id}")

        return ops_to_save


    # --- Opportunity Helpers (Original Logic) ---
    def _identify_current_opportunities(self, state_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify current opportunities from state analysis (Original Logic Structure)"""
        opportunities = []
        # Check narrative opportunities
        narrative_ops = self._identify_narrative_opportunities(state_analysis) # Original helper
        opportunities.extend(narrative_ops)
        # Check character opportunities
        character_ops = self._identify_character_opportunities(state_analysis) # Original helper
        opportunities.extend(character_ops)
        # Check meta opportunities
        meta_ops = self._identify_meta_opportunities(state_analysis) # Original helper
        opportunities.extend(meta_ops)
        # TODO: Implement the actual identification logic within the helpers
        return opportunities

    def _generate_opportunity_id(self, opportunity_data: Dict[str, Any]) -> str:
        """Generate unique ID for an opportunity (Original Logic)"""
        # Using original hash-based approach - potentially unstable if context changes slightly
        components = [
            opportunity_data.get("type", "unknown"),
            str(opportunity_data.get("target", "unknown")), # Ensure target is stringified
            str(hash(frozenset(opportunity_data.get("context", {}).items()))) # Hash context items
        ]
        return "_".join(components)[:128] # Limit length

    def _calculate_opportunity_potential(self, opportunity_data: Dict[str, Any]) -> float:
        """Calculate the potential impact and value of an opportunity (Original Logic Structure)"""
        factors = {
            "narrative_impact": self._calculate_narrative_impact(opportunity_data), # Original helper
            "character_impact": self._calculate_character_impact(opportunity_data), # Original helper
            "player_impact": self._calculate_player_impact(opportunity_data), # Original helper (different from 4th wall one)
            "meta_impact": self._calculate_meta_impact(opportunity_data) # Original helper (different from plot one)
        }
        weights = {"narrative_impact": 0.3, "character_impact": 0.3, "player_impact": 0.2, "meta_impact": 0.2}
        return sum(v * weights[k] for k, v in factors.items())

    def _calculate_opportunity_timing(self, opportunity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimal timing for an opportunity (Original Logic Structure)"""
        return {
            "earliest": self._calculate_earliest_timing(opportunity_data), # Original helper
            "latest": self._calculate_latest_timing(opportunity_data), # Original helper
            "optimal": self._calculate_optimal_timing_point(opportunity_data), # Original helper
            "dependencies": self._identify_timing_dependencies(opportunity_data) # Original helper
        }

    def _calculate_opportunity_priority(self, opportunity_data: Dict[str, Any]) -> float:
        """Calculate priority score for an opportunity (Original Logic Structure)"""
        factors = {
            "urgency": self._calculate_urgency(opportunity_data), # Original helper
            "impact": self._calculate_impact(opportunity_data), # Original helper (potential?)
            "feasibility": self._calculate_feasibility(opportunity_data), # Original helper
            "alignment": self._calculate_goal_alignment(opportunity_data) # Original helper
        }
        weights = {"urgency": 0.3, "impact": 0.3, "feasibility": 0.2, "alignment": 0.2}
        return sum(v * weights[k] for k, v in factors.items())

    def _identify_opportunity_dependencies(self, opportunity_data: Dict[str, Any]) -> List[str]:
        """Identify dependencies for an opportunity (Original Logic Structure)"""
        dependencies = []
        dependencies.extend(self._identify_narrative_dependencies(opportunity_data)) # Original helper
        dependencies.extend(self._identify_character_dependencies(opportunity_data)) # Original helper
        dependencies.extend(self._identify_state_dependencies(opportunity_data)) # Original helper
        return list(set(dependencies))

    def _assess_opportunity_risks(self, opportunity_data: Dict[str, Any]) -> Dict[str, float]:
        """Assess risks associated with an opportunity (Original Logic Structure)"""
        return {
            "detection_risk": self._calculate_detection_risk(opportunity_data), # Original helper (different from plot one)
            "failure_risk": self._calculate_failure_risk(opportunity_data), # Original helper
            "side_effect_risk": self._calculate_side_effect_risk(opportunity_data), # Original helper
            "narrative_risk": self._calculate_narrative_risk(opportunity_data) # Original helper
        }

    def _update_existing_opportunity(self, opportunity_id: str, new_data: Dict[str, Any]) -> bool:
        """Update an existing opportunity with new data (Original Logic). Returns True if updated."""
        current = self.agenda.opportunity_tracking.get(opportunity_id)
        if not current: return False

        updated = False
        # Recalculate fields based on new_data context
        new_potential = self._calculate_opportunity_potential(new_data)
        new_timing = self._calculate_opportunity_timing(new_data)
        new_priority = self._calculate_opportunity_priority(new_data)
        new_risks = self._assess_opportunity_risks(new_data)

        # Update if changed significantly (thresholds are arbitrary)
        if abs(current.get("potential", 0) - new_potential) > 0.1: current["potential"] = new_potential; updated = True
        if current.get("timing") != new_timing: current["timing"] = new_timing; updated = True # Simple dict comparison
        if abs(current.get("priority", 0) - new_priority) > 0.1: current["priority"] = new_priority; updated = True
        if current.get("risks") != new_risks: current["risks"] = new_risks; updated = True # Simple dict comparison

        # Update status if needed (Original Logic)
        if self._should_update_status(current, new_data): # Original helper
            new_status = self._determine_new_status(current, new_data) # Original helper
            if current.get("status") != new_status:
                current["status"] = new_status
                updated = True

        if updated:
             current["last_update"] = datetime.now().isoformat()
             current["context"] = new_data.get("context", current.get("context")) # Update context

        return updated

    # --- Placeholders for Opportunity Helpers (Original Logic - Random/Basic) ---
    # These were mostly random in the original, keeping that spirit
    def _identify_narrative_opportunities(self, state_analysis: Dict[str, Any]) -> List[Dict[str, Any]]: return [{"type": "narrative", "target": "plot_hole", "context": state_analysis}] if random.random() < 0.1 else []
    def _identify_character_opportunities(self, state_analysis: Dict[str, Any]) -> List[Dict[str, Any]]: return [{"type": "character", "target": "player", "context": state_analysis}] if random.random() < 0.1 else []
    def _identify_meta_opportunities(self, state_analysis: Dict[str, Any]) -> List[Dict[str, Any]]: return [{"type": "meta", "target": "game_mechanic", "context": state_analysis}] if random.random() < 0.1 else []
    def _calculate_character_impact(self, opportunity: Dict[str, Any]) -> float: return random.uniform(0.0, 1.0)
    def _calculate_player_impact(self, opportunity: Dict[str, Any]) -> float: return random.uniform(0.0, 1.0) # Different from 4th wall one
    # def _calculate_meta_impact(self, opportunity: Dict[str, Any]) -> float: return random.uniform(0.0, 1.0) # Defined in UniverseState for plot, use that? No, this is opp potential.
    def _calculate_opportunity_meta_impact(self, opportunity: Dict[str, Any]) -> float: return random.uniform(0.0, 1.0) # Renamed for clarity
    def _calculate_earliest_timing(self, opportunity: Dict[str, Any]) -> str: return "now"
    def _calculate_latest_timing(self, opportunity: Dict[str, Any]) -> str: return "soon"
    def _calculate_optimal_timing_point(self, opportunity: Dict[str, Any]) -> str: return "now"
    def _identify_timing_dependencies(self, opportunity: Dict[str, Any]) -> List[str]: return []
    def _calculate_urgency(self, opportunity: Dict[str, Any]) -> float: return random.uniform(0.0, 1.0)
    def _calculate_impact(self, opportunity: Dict[str, Any]) -> float: return self._calculate_opportunity_potential(opportunity) # Impact is potential?
    def _calculate_feasibility(self, opportunity: Dict[str, Any]) -> float: return random.uniform(0.3, 1.0) # Assume Nyx finds most things feasible
    def _calculate_goal_alignment(self, opportunity: Dict[str, Any]) -> float: return random.uniform(0.0, 1.0) # Needs access to goals
    def _identify_narrative_dependencies(self, opportunity: Dict[str, Any]) -> List[str]: return []
    def _identify_character_dependencies(self, opportunity: Dict[str, Any]) -> List[str]: return []
    def _identify_state_dependencies(self, opportunity: Dict[str, Any]) -> List[str]: return []
    # def _calculate_detection_risk(self, opportunity: Dict[str, Any]) -> float: return random.uniform(0.0, 1.0) # Defined in UniverseState for plot, use that? No, this is opp risk.
    def _calculate_opportunity_detection_risk(self, opportunity: Dict[str, Any]) -> float: return random.uniform(0.0, 0.5) # Renamed
    def _calculate_failure_risk(self, opportunity: Dict[str, Any]) -> float: return random.uniform(0.0, 0.3) # Nyx has low failure risk
    def _calculate_side_effect_risk(self, opportunity: Dict[str, Any]) -> float: return random.uniform(0.0, 0.4)
    def _calculate_narrative_risk(self, opportunity: Dict[str, Any]) -> float: return random.uniform(0.0, 0.2) # Nyx is confident in narrative control
    def _should_update_status(self, current: Dict[str, Any], new_data: Dict[str, Any]) -> bool: return random.random() < 0.05 # Rarely update status automatically
    def _determine_new_status(self, current: Dict[str, Any], new_data: Dict[str, Any]) -> str: return random.choice(["new", "evaluated", "missed"])


class AutonomousStateManager:
    def __init__(self, autonomous_data: AutonomousStateModel, agenda_ref: AgendaModel, universe_ref: UniverseStateModel):
        self.state = autonomous_data
        self.agenda = agenda_ref # Need access to goals/opportunities
        self.universe_state = universe_ref # Need access to universe state for analysis

    def analyze_current_state(self) -> Dict[str, Any]:
        """Analyze the current state of the universe and story from Nyx's perspective (Original Logic Structure)"""
        narrative_state = self._analyze_narrative_state() # Original helper
        player_state = self._analyze_player_state() # Original helper
        plot_opportunities = self._analyze_plot_opportunities() # Original helper (now internal)
        manipulation_vectors = self._identify_manipulation_vectors() # Original helper (now internal)
        risk_assessment = self._assess_current_risks() # Original helper (now internal)

        # Update internal models based on analysis (example)
        self.state.story_model.narrative_tension = narrative_state.get("tension_level", self.state.story_model.narrative_tension)
        self.state.player_model.current_engagement = player_state.get("engagement_level", self.state.player_model.current_engagement)

        return {
            "narrative_state": narrative_state, "player_state": player_state,
            "plot_opportunities": plot_opportunities, "manipulation_vectors": manipulation_vectors,
            "risk_assessment": risk_assessment
        }

    def make_strategic_decisions(self) -> List[Dict[str, Any]]:
        """Make strategic decisions about actions to take based on opportunities and goals. (Original Logic Structure)"""
        # Use opportunities and goals from the referenced AgendaModel
        opportunities = list(self.agenda.opportunity_tracking.values())
        active_goals = self.agenda.active_goals

        potential_decisions = []

        # Consider opportunities (Original Logic)
        evaluated_opportunities = self._identify_opportunities({}) # Pass state analysis if needed by original logic
        for opportunity in evaluated_opportunities: # Assuming identify returns actionable opp dicts
            if self._should_act_on_opportunity(opportunity): # Original helper
                decision = self._formulate_decision(opportunity) # Original helper
                potential_decisions.append(decision)

        # Consider goals (Added logic based on structure)
        for goal in active_goals:
             if goal.get("status") == "active":
                 goal_decision = self._formulate_decision_from_goal(goal)
                 if goal_decision: potential_decisions.append(goal_decision)


        # Validate and Prioritize (Original Logic)
        validated_decisions = [d for d in potential_decisions if self._validate_decision(d)] # Original helper
        prioritized_decisions = self._prioritize_decisions(validated_decisions) # Original helper

        # Select final decisions (e.g., limit number of actions per cycle)
        final_decisions = prioritized_decisions[:3] # Example: Max 3 decisions per cycle

        if final_decisions: logger.info(f"Formulated {len(final_decisions)} decisions for execution.")
        return final_decisions

    # --- Analysis Helpers (Original Logic) ---
    def _analyze_narrative_state(self) -> Dict[str, Any]:
        """Analyze the current narrative state and potential (Original Logic Structure)"""
        # Access universe state directly via self.universe_state
        return {
            "active_threads": self._analyze_active_threads(), # Original helper
            "character_developments": self._analyze_character_arcs(), # Original helper
            "plot_coherence": self._calculate_plot_coherence(), # Original helper
            "tension_points": self._identify_tension_points(), # Original helper
            "narrative_opportunities": self._find_narrative_opportunities() # Original helper
        }

    def _analyze_player_state(self) -> Dict[str, Any]:
        """Analyze player behavior and preferences (Original Logic Structure)"""
        # Access player model directly via self.state.player_model
        return {
            "behavior_pattern": self._analyze_behavior_patterns(self.state.player_model), # Original helper
            "preference_vector": self._calculate_preference_vector(self.state.player_model), # Original helper
            "engagement_level": self._assess_engagement_level(self.state.player_model), # Original helper
            "manipulation_susceptibility": self._calculate_susceptibility(self.state.player_model) # Original helper
        }

    # --- Narrative Analysis Helpers (Original Logic) ---
    def _analyze_active_threads(self) -> List[Dict[str, Any]]:
        """Analyze active narrative threads (Original Logic Structure)"""
        active_threads_analysis = []
        for thread_id, thread in self.universe_state.plot_threads.items():
            if thread.get("status") == "active":
                analysis = {
                    "thread_id": thread_id,
                    "status": self._analyze_thread_status(thread), # Original helper
                    "potential": self._calculate_thread_potential(thread), # Original helper
                    "risks": self._identify_thread_risks(thread), # Original helper
                    "opportunities": self._find_thread_opportunities(thread) # Original helper
                }
                active_threads_analysis.append(analysis)
        return active_threads_analysis

    def _analyze_character_arcs(self) -> Dict[str, Any]:
        """Analyze character development arcs (Original Logic Structure)"""
        character_arcs_analysis = {}
        for char_id, state in self.universe_state.character_states.items():
            if isinstance(state, dict): # Ensure state is dict
                arcs = {
                    "current_arc": self._identify_character_arc(state), # Original helper
                    "development_stage": self._calculate_development_stage(state), # Original helper
                    "potential_developments": self._identify_potential_developments(state), # Original helper
                    "relationship_dynamics": self._analyze_relationship_dynamics(state) # Original helper
                }
                character_arcs_analysis[char_id] = arcs
        return character_arcs_analysis

    def _calculate_plot_coherence(self) -> float:
        """Calculate overall plot coherence (Original Logic Structure)"""
        factors = {
            "thread_consistency": self._calculate_thread_consistency(), # Original helper
            "character_consistency": self._calculate_character_consistency(), # Original helper
            "world_consistency": self._calculate_world_consistency(), # Original helper
            "causality_strength": self._calculate_causality_strength() # Original helper
        }
        weights = self._get_coherence_weights() # Original helper
        return sum(v * weights[k] for k, v in factors.items())

    def _identify_tension_points(self) -> List[Dict[str, Any]]:
        """Identify narrative tension points (Original Logic Structure)"""
        tension_points = []
        for thread in self.universe_state.plot_threads.values():
            points = self._analyze_thread_tension(thread) # Original helper
            tension_points.extend(points)
        return self._prioritize_tension_points(tension_points) # Original helper

    def _find_narrative_opportunities(self) -> List[Dict[str, Any]]:
        """Find potential narrative opportunities (Original Logic Structure)"""
        opportunities = []
        opportunities.extend(self._find_character_opportunities()) # Original helper
        opportunities.extend(self._find_plot_opportunities()) # Original helper
        opportunities.extend(self._find_world_opportunities()) # Original helper
        return self._prioritize_opportunities(opportunities) # Original helper

    # --- Player Analysis Helpers (Original Logic) ---
    def _analyze_behavior_patterns(self, player_model: PlayerModel) -> Dict[str, Any]:
        """Analyze player behavior patterns (Original Logic Structure)"""
        history = player_model.decision_history
        return {
            "decision_style": self._identify_decision_style(history), # Original helper
            "preference_patterns": self._extract_preference_patterns(history), # Original helper
            "interaction_patterns": self._analyze_interaction_patterns(history), # Original helper
            "response_patterns": self._analyze_response_patterns(history) # Original helper (duplicate name?)
        }

    def _calculate_preference_vector(self, player_model: PlayerModel) -> Dict[str, float]:
        """Calculate player preference vector (Original Logic Structure)"""
        preferences = player_model.preference_model
        return {
            "narrative_style": self._calculate_narrative_preference(preferences), # Original helper
            "interaction_style": self._calculate_interaction_preference(preferences), # Original helper
            "challenge_preference": self._calculate_challenge_preference(preferences), # Original helper
            "development_focus": self._calculate_development_preference(preferences) # Original helper
        }

    def _assess_engagement_level(self, player_model: PlayerModel) -> float:
        """Assess player engagement level (Original Logic Structure)"""
        metrics = player_model.engagement_metrics
        factors = {
            "interaction_frequency": self._calculate_interaction_frequency(metrics), # Original helper
            "response_quality": self._calculate_response_quality(metrics), # Original helper
            "emotional_investment": self._calculate_emotional_investment(metrics), # Original helper (duplicate name?)
            "narrative_involvement": self._calculate_narrative_involvement(metrics) # Original helper
        }
        weights = self._get_engagement_weights() # Original helper
        return sum(v * weights[k] for k, v in factors.items())

    def _calculate_susceptibility(self, player_model: PlayerModel) -> Dict[str, float]:
        """Calculate player's susceptibility to different influence types (Original Logic Structure)"""
        return {
            "emotional": self._calculate_emotional_susceptibility(player_model), # Original helper
            "logical": self._calculate_logical_susceptibility(player_model), # Original helper
            "social": self._calculate_social_susceptibility(player_model), # Original helper
            "narrative": self._calculate_narrative_susceptibility(player_model) # Original helper
        }

    # --- Decision Making Helpers (Original Logic) ---
    def _identify_opportunities(self, state_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify actionable opportunities (Placeholder implementation)."""
        # This should use the opportunities tracked in the AgendaManager
        # For now, just return a filtered list from the agenda
        actionable_ops = []
        for opp_id, opp_data in self.agenda.opportunity_tracking.items():
            if opp_data.get("status") in ["new", "evaluated"]:
                 actionable_ops.append(opp_data)
        return actionable_ops

    def _should_act_on_opportunity(self, opportunity: Dict[str, Any]) -> bool:
        """Decide whether to act on an opportunity (Original Logic Structure)"""
        risk = self._calculate_opportunity_risk(opportunity) # Original helper
        benefit = self._calculate_opportunity_benefit(opportunity) # Original helper
        timing = self._evaluate_timing(opportunity) # Original helper (different from opp timing calc)
        alignment = self._check_goal_alignment(opportunity) # Original helper (needs goals)
        narrative_impact = self._evaluate_narrative_impact(opportunity) # Original helper

        decision_factors = {"risk": risk, "benefit": benefit, "timing": timing, "alignment": alignment, "narrative_impact": narrative_impact}
        return self._evaluate_decision_factors(decision_factors) # Original helper

    def _formulate_decision(self, opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Formulate a decision based on an opportunity (Original Logic Structure)"""
        # This is similar to _formulate_decision_from_opportunity, using original helpers
        decision_type = self._determine_decision_type(opportunity) # Original helper
        method = self._select_best_method(opportunity, decision_type) # Use matched helper
        timing = self._plan_execution_timing(opportunity) # Original helper
        contingencies = self._plan_decision_contingencies(opportunity) # Original helper (different from plot one)

        return {
            "type": decision_type, # Should map to DecisionType enum
            "target": opportunity.get("target"),
            "method": method,
            "timing_preference": timing, # Original returned dict, adapt if needed
            "contingencies": contingencies,
            "source_opportunity_id": opportunity.get("id"),
            "priority": opportunity.get("priority", 0.5)
        }

    def _formulate_decision_from_goal(self, goal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Formulate a specific action decision to advance a goal."""
        # Placeholder: Select next step from goal strategy
        strategy = goal.get("strategy", {})
        steps = strategy.get("steps", [])
        if not steps: return None
        next_step = steps[0] # Example: take the first step
        # TODO: Implement logic to select appropriate step based on progress/context
        return {
            "type": next_step.get("type", DecisionType.MANIPULATION.value), # Use enum value
            "target": next_step.get("target"),
            "method": next_step.get("method"),
            "timing_preference": "immediate",
            "contingencies": next_step.get("contingencies", []),
            "source_goal_id": goal.get("id"),
            "priority": goal.get("priority", 0.5)
        }

    def _validate_decision(self, decision: Dict[str, Any]) -> bool:
        """Validate a decision before execution (Original Logic Structure)"""
        if not self._check_narrative_consistency(decision): return False # Original helper
        if not self._verify_player_agency(decision): return False # Original helper
        if self._detect_decision_conflicts(decision): return False # Original helper
        if not self._validate_execution_capability(decision): return False # Original helper
        return True

    def _prioritize_decisions(self, decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize decisions based on importance and urgency (Original Logic Structure)"""
        scored_decisions = []
        for decision in decisions:
            score = self._calculate_decision_priority(decision) # Original helper
            scored_decisions.append((score, decision))
        scored_decisions.sort(reverse=True, key=lambda x: x[0])
        return [decision for _, decision in scored_decisions]

    # --- Placeholders/Original Logic for numerous helpers called above ---
    # These need to be implemented based on the original file's logic,
    # accessing self.state (AutonomousStateModel) and self.universe_state as needed.
    # Many were random or basic calculations in the original.

    # --- Narrative Analysis Helpers ---
    def _analyze_thread_status(self, thread): return thread.get("status", "unknown") # Placeholder
    def _calculate_thread_potential(self, thread): return random.random() # Placeholder
    def _identify_thread_risks(self, thread): return ["risk1"] # Placeholder
    def _find_thread_opportunities(self, thread): return [{"type": "advance_thread"}] # Placeholder
    def _identify_character_arc(self, state): return state.get("personality", {}).get("current_arc", "default") # Placeholder
    def _calculate_development_stage(self, state): return random.random() # Placeholder
    def _identify_potential_developments(self, state): return ["potential_change"] # Placeholder
    def _analyze_relationship_dynamics(self, state): return state.get("relationships", {}) # Placeholder
    def _calculate_thread_consistency(self): return random.uniform(0.5, 1.0) # Placeholder
    def _calculate_character_consistency(self): return random.uniform(0.5, 1.0) # Placeholder
    def _calculate_world_consistency(self): return random.uniform(0.7, 1.0) # Placeholder
    def _calculate_causality_strength(self): return random.uniform(0.6, 1.0) # Placeholder
    def _get_coherence_weights(self): return {"thread_consistency": 0.3, "character_consistency": 0.3, "world_consistency": 0.2, "causality_strength": 0.2} # Placeholder
    def _analyze_thread_tension(self, thread): return [{"point": "climax_approaching", "level": thread.get("tension", 0.5)}] # Placeholder
    def _prioritize_tension_points(self, points): points.sort(key=lambda p: p.get("level", 0), reverse=True); return points # Placeholder
    def _find_character_opportunities(self): return [] # Placeholder
    def _find_plot_opportunities(self): return [] # Placeholder
    def _find_world_opportunities(self): return [] # Placeholder
    def _prioritize_opportunities(self, opportunities): opportunities.sort(key=lambda o: o.get("priority", 0), reverse=True); return opportunities # Placeholder

    # --- Player Analysis Helpers ---
    def _identify_decision_style(self, history): return "calculated" # Placeholder
    def _extract_preference_patterns(self, history): return {"likes_conflict": True} # Placeholder
    def _analyze_interaction_patterns(self, history): return {"frequency": "high"} # Placeholder
    def _analyze_response_patterns(self, history): return {"tone": "curious"} # Placeholder
    def _calculate_narrative_preference(self, preferences): return preferences.get("narrative_score", 0.6) # Placeholder
    def _calculate_interaction_preference(self, preferences): return preferences.get("interaction_score", 0.7) # Placeholder
    def _calculate_challenge_preference(self, preferences): return preferences.get("challenge_score", 0.5) # Placeholder
    def _calculate_development_preference(self, preferences): return preferences.get("development_score", 0.8) # Placeholder
    def _calculate_interaction_frequency(self, metrics): return metrics.get("interaction_rate", 5.0) # Placeholder
    def _calculate_response_quality(self, metrics): return metrics.get("response_clarity", 0.8) # Placeholder
    def _calculate_emotional_investment(self, metrics): return metrics.get("sentiment_score", 0.6) # Placeholder
    def _calculate_narrative_involvement(self, metrics): return metrics.get("lore_engagement", 0.7) # Placeholder
    def _get_engagement_weights(self): return {"interaction_frequency": 0.2, "response_quality": 0.3, "emotional_investment": 0.3, "narrative_involvement": 0.2} # Placeholder
    def _calculate_emotional_susceptibility(self, player_model): return player_model.susceptibility_vector.get("emotional", 0.5) # Placeholder
    def _calculate_logical_susceptibility(self, player_model): return player_model.susceptibility_vector.get("logical", 0.5) # Placeholder
    def _calculate_social_susceptibility(self, player_model): return player_model.susceptibility_vector.get("social", 0.5) # Placeholder
    def _calculate_narrative_susceptibility(self, player_model): return player_model.susceptibility_vector.get("narrative", 0.5) # Placeholder

     # --- Decision Making Helpers ---
    def _calculate_opportunity_risk(self, opportunity): return sum(opportunity.get("risks", {}).values()) / max(1, len(opportunity.get("risks", {}))) # Average risk
    def _calculate_opportunity_benefit(self, opportunity): return opportunity.get("potential", 0.5)
    def _evaluate_timing(self, opportunity): return random.random() # Placeholder for complex timing evaluation
    def _check_goal_alignment(self, opportunity):
         # Check against self.agenda.active_goals
         for goal in self.agenda.active_goals:
             if opportunity.get("id") == goal.get("source_opportunity_id"): return 1.0
         return 0.3 # Default low alignment
    def _evaluate_narrative_impact(self, opportunity_or_decision): return random.random() # Placeholder
    def _evaluate_decision_factors(self, factors):
        # Original logic wasn't specified, implement simple weighted sum
        weights = {"benefit": 0.4, "timing": 0.3, "alignment": 0.2, "narrative_impact": 0.1, "risk": -0.3}
        score = sum(factors.get(k, 0) * w for k, w in weights.items())
        return score > 0.5 # Example threshold

    def _determine_decision_type(self, opportunity):
        # Map opportunity type to decision type (more robust)
        opp_type = opportunity.get("type", "generic")
        if "tension" in opp_type: return DecisionType.PLOT_CONTROL.value
        if "engagement" in opp_type: return DecisionType.FOURTH_WALL.value
        if "vulnerability" in opp_type: return DecisionType.MANIPULATION.value
        if "narrative_gap" in opp_type: return DecisionType.HIDDEN_INFLUENCE.value
        return DecisionType.OBSERVE.value # Default

    # def _select_best_method(self, opportunity, decision_type): return {"name": "default_method", "intensity": 0.5} # Defined above
    def _plan_execution_timing(self, opportunity): return opportunity.get("timing", {}).get("optimal", "immediate")
    # def _plan_decision_contingencies(self, opportunity): return [{"trigger": "failure", "action": "log_and_adapt"}] # Defined above

    # def _check_narrative_consistency(self, decision): return True # Defined above
    # def _verify_player_agency(self, decision): return True # Defined above
    def _detect_decision_conflicts(self, decision): return False # Placeholder - check against other planned decisions
    def _validate_execution_capability(self, decision): return True # Placeholder - check if Nyx *can* perform the action

    def _calculate_decision_priority(self, decision):
        # Original logic structure
        importance = self._calculate_importance(decision) # Placeholder helper
        urgency = self._calculate_urgency(decision) # Placeholder helper (different from opp urgency)
        impact = self._calculate_potential_impact(decision) # Placeholder helper
        risk = self._calculate_risk_factor(decision) # Placeholder helper
        weights = self._get_priority_weights() # Original helper
        priority = (importance * weights["importance"] + urgency * weights["urgency"] + impact * weights["impact"] - risk * weights["risk"])
        return min(1.0, max(0.0, priority))

    # --- Placeholders for priority calculation helpers ---
    def _calculate_importance(self, decision): return decision.get("priority", 0.5) # Inherit base priority
    def _calculate_decision_urgency(self, decision): return random.random() # Urgency based on timing?
    def _calculate_potential_impact(self, decision): return random.random() # Impact based on method/target?
    def _calculate_risk_factor(self, decision): return random.random() * 0.5 # Risk based on method/contingencies?
    def _get_priority_weights(self): return {"importance": 0.4, "urgency": 0.3, "impact": 0.2, "risk": 0.1} # Original weights


class InteractionGenerator:
    def __init__(self, profile_manager: ProfileManager):
        self.profile_manager = profile_manager

    def generate_action(self, scene_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate appropriate NPC action for the scene (Original Logic)"""
        target_id = scene_context.get("target_character")
        relationship = self.profile_manager.get_or_create_relationship(target_id) if target_id else None

        action_type = self._determine_action_type(scene_context, relationship) # Original helper
        action_content = await self._generate_action_content(action_type, relationship, scene_context) # Original helper
        action_style = self._get_action_style() # Original helper (doesn't use relationship in original signature)

        psychological_impact = self._calculate_psychological_impact(action_type, scene_context) # Original helper
        emotional_triggers = self._get_emotional_triggers(scene_context) # Original helper
        manipulation_hooks = self._get_manipulation_hooks(scene_context) # Original helper

        return {
            "type": action_type.value, "content": action_content, "style": action_style,
            "power_level": self.profile_manager.profile.personality.power_dynamic,
            "psychological_impact": psychological_impact,
            "emotional_triggers": emotional_triggers, "manipulation_hooks": manipulation_hooks
        }

    def _determine_action_type(self, scene_context: Dict[str, Any], relationship: Optional[RelationshipModel]) -> ActionType:
        """Determine appropriate type of action based on context (Original Logic)"""
        scene_type = scene_context.get("scene_type", "")
        # Original used _get_or_create_relationship internally, we use the passed one
        rel = relationship or RelationshipModel() # Use default if no relationship

        if scene_type == "confrontation":
            return ActionType.DOMINATE if rel.dominance > 0.7 else ActionType.CHALLENGE
        elif scene_type == "seduction":
            return ActionType.SEDUCE if rel.emotional_bond > 0.5 else ActionType.TEASE
        elif scene_type == "manipulation":
            return ActionType.MANIPULATE if rel.manipulation_success > 0.6 else ActionType.INFLUENCE
        return ActionType.INTERACT

    async def _generate_action_content(
        self,
        action_type: ActionType,
        relationship: Optional[RelationshipModel],
        scene_context: Dict[str, Any]
    ) -> str:
        """Generate action prose via LLM instead of hard-coded strings."""
        try:
            payload = json.dumps({
                "action_type": action_type.value,
                "relationship": (relationship.dict() if relationship else {}),
                "scene_context": scene_context
            }, ensure_ascii=False)

            result = await Runner.run(
                starting_agent=nyx_action_content_generator,
                input=payload
            )
            return json.loads(result.output.strip())["content"]

        except Exception as e:
            logger.warning(f"LLM content generation failed: {e}")
            # Minimal fallback so the game never crashes
            return f"Nyx performs a {action_type.value.lower()} gesture."


    def _get_action_style(self) -> Dict[str, Any]:
        """Get current action style based on personality (Original Logic - Uses internal helpers)"""
        # Access profile via the manager
        personality = self.profile_manager.profile.personality
        return {
            "tone": self._determine_tone(), # Original helper
            "intensity": personality.power_dynamic,
            "traits": personality.adaptable_traits, # Assumes already adapted
            "body_language": self._get_body_language(), # Original helper
            "voice_modulation": self._get_voice_modulation(), # Original helper
            "psychological_undertones": self._get_psychological_undertones() # Original helper
        }

    # --- Original Style Helpers ---
    def _determine_tone(self) -> str:
        """Determine appropriate tone based on personality and mood (Original Logic)"""
        mood = self.profile_manager.profile.personality.current_mood
        power = self.profile_manager.profile.personality.power_dynamic
        if power > 0.8: return "commanding" if mood == "stern" else "authoritative"
        elif power > 0.6: return "confident" if mood == "playful" else "assertive"
        else: return "neutral"

    def _get_body_language(self) -> List[str]:
        """Generate appropriate body language cues (Original Logic)"""
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
        """Generate voice modulation parameters (Original Logic)"""
        power = self.profile_manager.profile.personality.power_dynamic
        mood = self.profile_manager.profile.personality.current_mood
        base_modulation = {"pitch": "medium", "volume": "moderate", "pace": "measured", "tone_quality": "smooth"}
        if power > 0.8: base_modulation.update({"pitch": "low", "volume": "commanding", "pace": "deliberate", "tone_quality": "authoritative"})
        elif mood == "playful": base_modulation.update({"pitch": "varied", "volume": "dynamic", "pace": "playful", "tone_quality": "melodic"})
        return base_modulation

    def _get_psychological_undertones(self) -> List[str]:
        """Generate psychological undertones for the interaction (Original Logic)"""
        power = self.profile_manager.profile.personality.power_dynamic
        mood = self.profile_manager.profile.personality.current_mood
        undertones = []
        if power > 0.8: undertones.extend(["subtle dominance assertion", "psychological pressure", "authority establishment"])
        elif power > 0.6: undertones.extend(["influence building", "subtle manipulation", "psychological anchoring"])
        else: undertones.extend(["trust building", "rapport establishment", "emotional connection"])
        if mood == "stern": undertones.extend(["disciplinary undertone", "boundary setting", "behavioral correction"])
        elif mood == "playful": undertones.extend(["psychological teasing", "emotional engagement", "behavioral encouragement"])
        return undertones

    # --- Impact/Trigger Helpers (Original Logic) ---
    def _calculate_psychological_impact(self, action_type: ActionType, scene_context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate the psychological impact of the action (Original Logic)"""
        target = scene_context.get("target_character", "")
        # Need relationship data for calculation
        relationship = self.profile_manager.get_or_create_relationship(target) if target else RelationshipModel() # Use default if no target

        base_impact = {"dominance_impact": 0.0, "emotional_impact": 0.0, "psychological_impact": 0.0, "behavioral_impact": 0.0}
        if action_type == ActionType.DOMINATE: base_impact.update({"dominance_impact": 0.8, "psychological_impact": 0.7, "behavioral_impact": 0.6})
        elif action_type == ActionType.SEDUCE: base_impact.update({"emotional_impact": 0.8, "psychological_impact": 0.6, "behavioral_impact": 0.7})
        elif action_type == ActionType.MANIPULATE: base_impact.update({"psychological_impact": 0.8, "emotional_impact": 0.6, "behavioral_impact": 0.7})

        # Adjust based on relationship
        final_impact = {}
        for key, value in base_impact.items():
            adjusted_value = value * (1 + relationship.manipulation_success) # Use relationship data
            final_impact[key] = min(1.0, adjusted_value)
        return final_impact

    def _get_emotional_triggers(self, scene_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get relevant emotional triggers for the scene (Original Logic)"""
        target = scene_context.get("target_character", "")
        relationship = self.profile_manager.get_or_create_relationship(target) if target else None

        if relationship and relationship.emotional_triggers:
            return relationship.emotional_triggers
        else:
            # Generate new potential triggers (Original default)
            return [
                {"type": "validation_need", "strength": 0.7, "trigger": "seeking approval"},
                {"type": "attachment_anxiety", "strength": 0.6, "trigger": "fear of abandonment"},
                {"type": "power_dynamic", "strength": 0.8, "trigger": "submission desire"}
            ]

    def _get_manipulation_hooks(self, scene_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get relevant manipulation hooks for the scene (Original Logic)"""
        target = scene_context.get("target_character", "")
        relationship = self.profile_manager.get_or_create_relationship(target) if target else None

        if relationship and relationship.psychological_hooks:
            return relationship.psychological_hooks
        else:
             # Generate new potential hooks (Original default)
            return [
                {"type": "emotional_dependency", "strength": 0.7, "hook": "need for guidance"},
                {"type": "psychological_vulnerability", "strength": 0.6, "hook": "self-doubt"},
                {"type": "behavioral_pattern", "strength": 0.8, "hook": "reward seeking"}
            ]


# --- Power Execution (Reusing previous refactor's template) ---

class PowerExecutor:
    def __init__(self, agent_id: str, powers: OmniscientPowersModel, db_interface: NyxDatabaseInterface, universe_manager: UniverseStateManager):
        self.agent_id = agent_id; self.powers = powers; self.db = db_interface; self.universe_manager = universe_manager

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
            return {"success": True, "result": memory_result, "warning": f"DB save failed: {e}"} # Success=True because memory op succeeded
        except Exception as e:
            logger.error(f"Unexpected Error during {power_name} DB save: {e}", exc_info=True)
            if kwargs.get("rollback_func") and original_state_snapshot is not None: kwargs["rollback_func"](original_state_snapshot); logger.warning(f"Rolled back in-memory state for {power_name} due to unexpected DB error.")
            return {"success": False, "reason": f"Unexpected DB error: {e}"}

    # --- Specific Power Execution Methods ---
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
        # !! Accessing relationship knowledge requires profile access, handled as warning in UniverseStateManager !!
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

    async def execute_social_link_update(self, social_link_manager: SocialLinkManager, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        original_link_state = social_link_manager.link.copy(deep=True); memory_result = None
        try:
            exp_gain, interaction_record = social_link_manager.update_on_interaction(interaction_data)
            memory_result = {"exp_gain": exp_gain, "interaction_record": interaction_record, "new_experience": social_link_manager.link.experience, "level": social_link_manager.link.level, "leveled_up": interaction_record["leveled_up"]}
        except Exception as e: logger.error(f"Error updating social link in memory: {e}", exc_info=True); return {"success": False, "reason": f"Internal error updating social link: {e}"}
        def rollback(original_state): social_link_manager.link = original_state
        async def db_save(agent_id, mem_res, *args, **kwargs): await self.db.save_social_link_update(agent_id, social_link_manager.link, mem_res["interaction_record"])
        result = await self._execute_power("social_link_update", True, None, lambda *a, **kw: memory_result, db_save, interaction_data=interaction_data, get_rollback_state_func=lambda: original_link_state, rollback_func=rollback)
        if result.get("success"): final_res = {"success": True, **memory_result}; del final_res["interaction_record"]; final_res["warning"] = result.get("warning"); return final_res
        else: return result

    async def execute_opportunity_tracking(self, agenda_manager: AgendaManager, state_analysis: Dict[str, Any]) -> Dict[str, Any]:
        # Note: This doesn't fit the _execute_power template well as memory_apply returns *what* to save.
        logger.debug(f"Tracking new opportunities for agent {self.agent_id}")
        original_opp_tracking = agenda_manager.agenda.opportunity_tracking.copy()
        ops_to_save = []
        try:
            ops_to_save = agenda_manager._track_new_opportunities_logic(state_analysis) # Sync logic
        except Exception as e: logger.error(f"Error tracking opportunities in memory: {e}", exc_info=True); return {"success": False, "reason": f"Internal error tracking opportunities: {e}"}
        if not ops_to_save: return {"success": True, "opportunities_saved": 0}
        try:
            await self.db.save_opportunities(self.agent_id, ops_to_save)
            return {"success": True, "opportunities_saved": len(ops_to_save)}
        except Exception as e:
            logger.error(f"DB Error saving opportunities for agent {self.agent_id}: {e}", exc_info=True)
            agenda_manager.agenda.opportunity_tracking = original_opp_tracking # Rollback
            logger.warning(f"Rolled back in-memory opportunities for {self.agent_id} due to DB error.")
            return {"success": False, "reason": "Failed to save new opportunities to DB"}


# --- Main Agent Class ---

class NPCAgentState(BaseModel): # Reusing previous definition
    profile: ProfileModel = Field(default_factory=ProfileModel); universe_state: UniverseStateModel = Field(default_factory=UniverseStateModel)
    social_link: SocialLinkModel = Field(default_factory=SocialLinkModel); agenda: AgendaModel = Field(default_factory=AgendaModel)
    autonomous_state: AutonomousStateModel = Field(default_factory=AutonomousStateModel); omniscient_powers: OmniscientPowersModel = Field(default_factory=OmniscientPowersModel)

class NPCAgent:
    id: str; db_interface: NyxDatabaseInterface; profile_manager: ProfileManager; universe_manager: UniverseStateManager
    social_link_manager: SocialLinkManager; agenda_manager: AgendaManager; autonomous_manager: AutonomousStateManager
    interaction_generator: InteractionGenerator; power_executor: PowerExecutor

    def __init__(self, agent_id: Optional[str] = None, db_interface: Optional[NyxDatabaseInterface] = None, initial_state: Optional[NPCAgentState] = None):
        self.id = agent_id or f"nyx_{random.randint(1000, 9999)}"
        self.db_interface = db_interface or AsyncpgNyxDatabase()
        state = initial_state or NPCAgentState()

        self.profile_manager = ProfileManager(state.profile)
        self.universe_manager = UniverseStateManager(state.universe_state)
        self.social_link_manager = SocialLinkManager(state.social_link)
        self.agenda_manager = AgendaManager(state.agenda)
        # Pass necessary refs to AutonomousStateManager
        self.autonomous_manager = AutonomousStateManager(state.autonomous_state, self.agenda_manager.agenda, self.universe_manager.state)
        self.interaction_generator = InteractionGenerator(self.profile_manager)
        self.power_executor = PowerExecutor(self.id, state.omniscient_powers, self.db_interface, self.universe_manager)
        # Inject profile manager dependency if needed by universe manager's knowledge access
        # self.universe_manager.profile_manager = self.profile_manager # Example injection if needed
        logger.info(f"NPCAgent {self.id} initialized.")

    def activate(self, scene_context: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"Activating agent {self.id} in scene {scene_context.get('scene_id')}")
        self.profile_manager.activate(scene_context.get("scene_id"))
        self.profile_manager.adapt_personality(scene_context) # Uses original logic
        initial_action = self.interaction_generator.generate_action(scene_context) # Uses original logic
        return {"status": "active", "profile": self.profile_manager.profile.dict(), "initial_action": initial_action}

    def deactivate(self): logger.info(f"Deactivating agent {self.id}"); self.profile_manager.deactivate()

    async def think(self) -> Dict[str, Any]:
        logger.info(f"Agent {self.id} starting thinking cycle.")
        # 1. Analyze State (using original logic structure)
        state_analysis = self.autonomous_manager.analyze_current_state()

        # 2. Update Agenda & Track Opportunities (includes DB saves via managers)
        await self.agenda_manager.update_agenda(state_analysis, self.db_interface, self.id)
        await self.agenda_manager.track_new_opportunities(state_analysis, self.db_interface, self.id) # Now handles its own saving

        # 3. Make Decisions (using original logic structure)
        decisions = self.autonomous_manager.make_strategic_decisions() # Uses internal refs to agenda/opps

        # 4. Execute Actions
        actions_results = []
        if decisions:
            logger.info(f"Agent {self.id} executing {len(decisions)} decisions.")
            for decision in decisions:
                action_result = await self._execute_decision(decision)
                actions_results.append({"decision": decision, "result": action_result})
        else: logger.debug(f"Agent {self.id} made no decisions this cycle.")

        logger.info(f"Agent {self.id} finished thinking cycle.")
        return {"analysis": state_analysis, "decisions_made": len(decisions), "actions_results": actions_results}

    async def _execute_decision(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        decision_type_str = decision.get("type")
        method_params = decision.get("method", {})
        target = decision.get("target")
        target_id = target if isinstance(target, str) else (target.get("id") if isinstance(target, dict) else None)
        logger.debug(f"Executing decision: Type={decision_type_str}, Target={target}, Method/Params={method_params}")

        try: # Map decision type string/enum to executor methods
            if decision_type_str == DecisionType.REALITY_MODIFICATION.value: return await self.power_executor.execute_reality_modification(method_params.get("modification_details", {}))
            elif decision_type_str == DecisionType.CHARACTER_MODIFICATION.value and target_id: return await self.power_executor.execute_character_modification(target_id, method_params.get("modifications", {}))
            elif decision_type_str == DecisionType.SCENE_CONTROL.value and target_id: return await self.power_executor.execute_scene_control(target_id, method_params.get("modifications", {}))
            elif decision_type_str == DecisionType.PLOT_CONTROL.value: return await self.power_executor.execute_plot_manipulation(method_params.get("plot_data", {}))
            elif decision_type_str == DecisionType.FOURTH_WALL.value: return await self.power_executor.execute_fourth_wall_break(method_params.get("context", {}))
            elif decision_type_str == DecisionType.HIDDEN_INFLUENCE.value: return await self.power_executor.execute_hidden_influence(method_params.get("influence_data", {}))
            elif decision_type_str == DecisionType.MANIPULATION.value:
                manip_type = method_params.get("type", "subtle")
                if manip_type == "direct" and isinstance(target, dict) and target_id:
                     if target.get("type") == "character": logger.info("Routing 'manipulation (direct)' to character modification."); return await self.power_executor.execute_character_modification(target_id, method_params.get("modifications", {}))
                     elif target.get("type") == "scene": logger.info("Routing 'manipulation (direct)' to scene control."); return await self.power_executor.execute_scene_control(target_id, method_params.get("modifications", {}))
                elif manip_type == "subtle":
                     logger.info("Routing 'manipulation (subtle)' to hidden influence.")
                     influence_data = {"type": "subtle", "target": target, **method_params.get("influence_details", {})}
                     return await self.power_executor.execute_hidden_influence(influence_data)
                logger.warning(f"Unhandled manipulation type/target: {manip_type}/{target}"); return {"success": False, "reason": f"Unhandled manipulation type/target: {manip_type}/{target}"}
            elif decision_type_str == DecisionType.SOCIAL_INTERACTION.value: logger.info(f"Executing social interaction (placeholder): Target={target}"); return {"success": True, "result": "Social interaction initiated."}
            elif decision_type_str == DecisionType.OBSERVE.value: logger.info(f"Executing observe action (passive): Target={target}"); return {"success": True, "result": "Observation complete."}
            else: logger.warning(f"Unknown decision type encountered: {decision_type_str}"); return {"success": False, "reason": f"Unknown decision type: {decision_type_str}"}
        except Exception as e: logger.error(f"Error executing decision {decision}: {e}", exc_info=True); return {"success": False, "reason": f"Unexpected error during execution: {e}"}

    # --- Direct Power Access Methods ---
    async def modify_reality(self, modification: Dict[str, Any]) -> Dict[str, Any]: return await self.power_executor.execute_reality_modification(modification)
    async def modify_character(self, character_id: str, modifications: Dict[str, Any]) -> Dict[str, Any]: return await self.power_executor.execute_character_modification(character_id, modifications)
    async def access_knowledge(self, query: Dict[str, Any]) -> Dict[str, Any]: return await self.power_executor.execute_knowledge_access(query)
    async def control_scene(self, scene_id: str, modifications: Dict[str, Any]) -> Dict[str, Any]: return await self.power_executor.execute_scene_control(scene_id, modifications)
    async def update_social_link(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        result = await self.power_executor.execute_social_link_update(self.social_link_manager, interaction_data)
        if result["success"] and interaction_data.get("target_character"):
             outcome_details = {"rapport_gain": result.get("exp_gain", 0) / 10.0, "emotional_impact": interaction_data.get("intensity", 0.5), "manipulation_success": 1.0 if result.get("exp_gain", 0) > 15 else 0.0 }
             self.profile_manager.update_relationship_on_interaction(interaction_data["target_character"], outcome_details)
        return result
    async def manipulate_plot(self, plot_data: Dict[str, Any]) -> Dict[str, Any]: return await self.power_executor.execute_plot_manipulation(plot_data)
    async def break_fourth_wall(self, context: Dict[str, Any]) -> Dict[str, Any]: return await self.power_executor.execute_fourth_wall_break(context)
    async def exert_hidden_influence(self, influence_data: Dict[str, Any]) -> Dict[str, Any]: return await self.power_executor.execute_hidden_influence(influence_data)

    # --- State Management ---
    def get_current_state(self) -> NPCAgentState:
        return NPCAgentState(profile=self.profile_manager.profile, universe_state=self.universe_manager.state, social_link=self.social_link_manager.link, agenda=self.agenda_manager.agenda, autonomous_state=self.autonomous_manager.state, omniscient_powers=self.power_executor.powers)
    async def save_state(self) -> bool:
        logger.info(f"Attempting to save state for agent {self.id}"); current_state = self.get_current_state()
        try: await self.db_interface.save_agent_state(self.id, current_state); return True
        except Exception as e: logger.error(f"Failed to save agent {self.id} state: {e}", exc_info=True); return False
    @classmethod
    async def load_state(cls: Type['NPCAgent'], agent_id: str, db_interface: Optional[NyxDatabaseInterface] = None) -> Optional['NPCAgent']:
        db = db_interface or AsyncpgNyxDatabase(); logger.info(f"Attempting to load state for agent {agent_id}")
        try:
            loaded_state_data = await db.load_agent_state(agent_id)
            if loaded_state_data: return cls(agent_id=agent_id, db_interface=db, initial_state=loaded_state_data)
            else: return None
        except Exception as e: logger.error(f"Failed to load agent {agent_id}: {e}", exc_info=True); return None

# --- Module-Level Load/Save ---
async def load_npc_agent(agent_id: str, db_interface: Optional[NyxDatabaseInterface] = None) -> Optional[NPCAgent]: return await NPCAgent.load_state(agent_id, db_interface)
async def save_npc_agent(agent: NPCAgent) -> bool: return await agent.save_state()

# --- Example Usage (Conceptual) ---
async def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    agent_id = "nyx_full_logic_test"
    db = AsyncpgNyxDatabase()

    nyx_agent = await load_npc_agent(agent_id, db)
    if not nyx_agent:
        logger.info(f"No existing state found for {agent_id}, creating new agent.")
        nyx_agent = NPCAgent(agent_id=agent_id, db_interface=db)
        # Ensure initial save includes all default substructures correctly
        await save_npc_agent(nyx_agent)
        # Reload to ensure parsing works
        nyx_agent = await load_npc_agent(agent_id, db)
        if not nyx_agent:
             logger.critical("Failed to load agent even after initial save!")
             return

    logger.info("--- Agent Loaded/Created ---")

    scene_context = {"scene_id": "scene_alpha", "scene_type": "confrontation", "target_character": "player_hero"}
    activation_info = nyx_agent.activate(scene_context)
    print(f"Nyx activated. Profile Mood: {nyx_agent.profile_manager.profile.personality.current_mood}, Power Dynamic: {nyx_agent.profile_manager.profile.personality.power_dynamic}")
    print(f"Initial action: {activation_info['initial_action']['type']} - {activation_info['initial_action']['content']}")

    logger.info("--- Simulating Interaction ---")
    interaction = { "type": "emotional", "intensity": 0.7, "success_rate": 0.6, "depth": 0.5, "context": {"topic": "player_motivation"}, "target_character": "player_hero" }
    sl_update_result = await nyx_agent.update_social_link(interaction)
    print(f"Social link update result: {sl_update_result}")
    print(f"Current SL Level: {nyx_agent.social_link_manager.link.level}, XP: {nyx_agent.social_link_manager.link.experience:.2f}")
    print(f"Relationship with player_hero familiarity: {nyx_agent.profile_manager.profile.relationships.get('player_hero', {}).familiarity:.2f}")


    logger.info("--- Simulating Autonomous Thinking ---")
    think_result = await nyx_agent.think()
    print(f"Think cycle completed. Decisions: {think_result['decisions_made']}")
    for i, action_res in enumerate(think_result['actions_results']):
        print(f"  Action {i+1}:")
        print(f"    Decision: {action_res['decision']}")
        print(f"    Result: {action_res['result']}")


    logger.info("--- Simulating Reality Modification ---")
    mod = { "type": "environmental", "scope": "scene", "duration": "temporary", "parameters": {"area": "current_room", "elements": ["chilling_wind", "shadows_deepen"], "intensity": 0.7}}
    reality_mod_result = await nyx_agent.modify_reality(mod)
    print(f"Reality modification result: {reality_mod_result}")
    print(f"Scene {scene_context['scene_id']} state: {nyx_agent.universe_manager.state.active_scenes.get(scene_context['scene_id'], {}).get('reality_state')}")


    logger.info("--- Saving Final State ---")
    save_success = await save_npc_agent(nyx_agent)
    print(f"Final state saved: {save_success}")

if __name__ == "__main__":
    try:
        # Ensure DB connection details are set (e.g., via DATABASE_URL env var)
        if not os.getenv("DATABASE_URL"):
             print("WARNING: DATABASE_URL environment variable not set. Database operations will likely fail.")
        asyncio.run(main())
    except Exception as e:
         logger.critical(f"Main execution failed: {e}", exc_info=True)
         print("\n--- ERROR ---")
         print("Ensure the database is running and connection details (e.g., DATABASE_URL env var) are correctly set.")
         print(f"Error details: {e}")
