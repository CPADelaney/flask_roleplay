# new_game_agent.py
#
# REFACTORED TO FIX RACE CONDITIONS:
# 1. Deterministic pipeline - tools called directly in sequence, not via LLM decision
# 2. Optional environment_data with auto-loading from DB if missing
# 3. Completeness check before marking conversation as ready
# 4. Background processing delayed until after setup complete
# 5. StoryDirector initialization deferred to avoid heavy ops during setup
# 6. Proper parameter passing for PostgreSQL queries (no raw Python lists in ANY())
# 7. Pydantic V2 compatible field validators with mode="after"

import logging
import json
import asyncio
import os
import functools
from datetime import datetime
from typing import Optional, List, Dict, Any

from agents import Agent, Runner, function_tool, GuardrailFunctionOutput, InputGuardrail, RunContextWrapper, input_guardrail, output_guardrail, OutputGuardrail
from pydantic import BaseModel, Field, ConfigDict, field_validator

from memory.wrapper import MemorySystem
from logic.stats_logic import insert_default_player_stats_chase, apply_stat_change

# Import your existing modules
from logic.calendar import update_calendar_names
from lore.core import canon
from logic.aggregator_sdk import get_aggregated_roleplay_context
from npcs.new_npc_creation import NPCCreationHandler  # Must properly await all async operations
from routes.ai_image_generator import generate_roleplay_image_from_gpt
from lore.lore_generator import DynamicLoreGenerator

# Import database connections
from db.connection import get_db_connection_context

# Import Nyx governance integration
from nyx.governance_helpers import with_governance, with_governance_permission, with_action_reporting
from nyx.directive_handler import DirectiveHandler
from nyx.nyx_governance import AgentType, DirectiveType, DirectivePriority

# Configuration
DB_DSN = os.getenv("DB_DSN")

logger = logging.getLogger(__name__)

NEW_GAME_AGENT_NYX_TYPE = AgentType.UNIVERSAL_UPDATER
NEW_GAME_AGENT_NYX_ID = "new_game_director_agent"  # Use consistent ID throughout

# Concrete models to replace generic dicts
class QuestData(BaseModel):
    quest_name: str = ""
    quest_description: str = ""
    model_config = ConfigDict(extra="forbid")

class Event(BaseModel):
    name: str
    description: str
    start_time: str
    end_time: str
    location: str
    year: int = 1
    month: int = 1
    day: int = 1
    time_of_day: str = "Morning"
    model_config = ConfigDict(extra="forbid")

class Location(BaseModel):
    location_name: str
    description: str
    type: str = "settlement"
    features: List[str] = Field(default_factory=list)
    open_hours_json: str = "{}"  # Store as JSON string instead of dict
    model_config = ConfigDict(extra="forbid")
    
    @field_validator("open_hours_json", mode="after")
    @classmethod
    def validate_open_hours_json(cls, v: str) -> str:
        """Validate that open_hours_json is valid JSON"""
        try:
            json.loads(v)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON for open_hours: {e}")
        return v

class DaySchedule(BaseModel):
    morning: str = ""
    afternoon: str = ""
    evening: str = ""
    night: str = ""
    model_config = ConfigDict(extra="forbid")

class WeekSchedule(BaseModel):
    monday: Optional[DaySchedule] = None
    tuesday: Optional[DaySchedule] = None
    wednesday: Optional[DaySchedule] = None
    thursday: Optional[DaySchedule] = None
    friday: Optional[DaySchedule] = None
    saturday: Optional[DaySchedule] = None
    sunday: Optional[DaySchedule] = None
    # For custom day names, store as JSON string
    custom_schedule_json: Optional[str] = None
    model_config = ConfigDict(extra="forbid")
    
    @field_validator("custom_schedule_json", mode="after")
    @classmethod
    def validate_custom_schedule_json(cls, v: Optional[str]) -> Optional[str]:
        """Validate that custom_schedule_json is valid JSON if provided"""
        if v is not None:
            try:
                json.loads(v)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON for custom_schedule: {e}")
        return v

class StatModifier(BaseModel):
    stat_name: str
    modifier_value: float
    model_config = ConfigDict(extra="forbid")

class EnvironmentInfo(BaseModel):
    setting_name: str = ""
    environment_desc: str = ""
    environment_history: str = ""
    scenario_name: str = ""
    model_config = ConfigDict(extra="forbid")

class CalendarData(BaseModel):
    days: List[str] = Field(default_factory=list)
    months: List[str] = Field(default_factory=list)
    seasons: List[str] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid")

# Output models for structured data
class EnvironmentData(BaseModel):
    setting_name: str
    environment_desc: str
    environment_history: str
    events: List[Event] = Field(default_factory=list)
    locations: List[Location] = Field(default_factory=list)
    scenario_name: str
    quest_data: QuestData = Field(default_factory=QuestData)
    model_config = ConfigDict(extra="forbid")

class NPCData(BaseModel):
    npc_name: str
    introduced: bool = False
    archetypes: List[str] = Field(default_factory=list)
    physical_description: str = ""
    hobbies: List[str] = Field(default_factory=list)
    personality_traits: List[str] = Field(default_factory=list)
    likes: List[str] = Field(default_factory=list)
    dislikes: List[str] = Field(default_factory=list)
    schedule: WeekSchedule = Field(default_factory=WeekSchedule)
    model_config = ConfigDict(extra="forbid")

class GameContext(BaseModel):
    user_id: int
    conversation_id: int
    db_dsn: str = DB_DSN
    model_config = ConfigDict(extra="forbid")

# Pydantic models for function parameters (excluding ctx)
class CalendarToolParams(BaseModel):
    environment_desc: str
    setting_name: Optional[str] = None
    environment_data: Optional[EnvironmentInfo] = None
    model_config = ConfigDict(extra="forbid")

class CreateCalendarParams(BaseModel):
    environment_desc: str
    model_config = ConfigDict(extra="forbid")

class GenerateEnvironmentParams(BaseModel):
    mega_name: str
    mega_desc: str
    env_components: Optional[List[str]] = None
    enhanced_features: Optional[List[str]] = None
    stat_modifiers: Optional[List[StatModifier]] = None
    model_config = ConfigDict(extra="forbid")

# Split into required vs optional fields for spawn NPCs
class SpawnNPCsRequiredParams(BaseModel):
    count: int = 5
    model_config = ConfigDict(extra="forbid")

class SpawnNPCsParams(BaseModel):
    count: int = 5
    environment_desc: Optional[str] = None
    day_names: Optional[List[str]] = None
    model_config = ConfigDict(extra="forbid")

class CreateChaseScheduleParams(BaseModel):
    environment_desc: str
    day_names: List[str]
    model_config = ConfigDict(extra="forbid")

# Make environment_data optional as suggested in fixes
class CreateNPCsAndSchedulesParams(BaseModel):
    environment_data: Optional[EnvironmentInfo] = None
    model_config = ConfigDict(extra="forbid")

class NPCScheduleData(BaseModel):
    npc_ids: List[str] = Field(default_factory=list)
    chase_schedule_json: str = "{}"  # Store schedule as JSON string
    model_config = ConfigDict(extra="forbid")
    
    @field_validator("chase_schedule_json", mode="after")
    @classmethod
    def validate_chase_schedule_json(cls, v: str) -> str:
        """Validate that chase_schedule_json is valid JSON"""
        try:
            json.loads(v)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON for chase_schedule: {e}")
        return v

class CreateOpeningNarrativeParams(BaseModel):
    environment_data: EnvironmentInfo
    npc_schedule_data: NPCScheduleData
    model_config = ConfigDict(extra="forbid")

class FinalizeGameSetupParams(BaseModel):
    opening_narrative: str
    model_config = ConfigDict(extra="forbid")

class GenerateLoreParams(BaseModel):
    environment_desc: str
    model_config = ConfigDict(extra="forbid")

class LoreResult(BaseModel):
    lore_summary: str
    factions_count: int
    cultural_elements_count: int
    locations_count: int
    historical_events_count: int
    error: Optional[str] = None
    model_config = ConfigDict(extra="forbid")

class FinalizeResult(BaseModel):
    status: str
    welcome_image_url: Optional[str]
    lore_summary: str
    initial_conflict: str
    currency_system: str
    model_config = ConfigDict(extra="forbid")

class ProcessNewGameResult(BaseModel):
    message: str
    scenario_name: str
    environment_name: str
    environment_desc: str
    lore_summary: str
    conversation_id: int
    welcome_image_url: Optional[str]
    status: str
    model_config = ConfigDict(extra="forbid")

class GameCreationResult(BaseModel):
    """Result from the main game creation agent"""
    setting_name: str
    scenario_name: str
    environment_desc: str
    environment_history: str = ""
    lore_summary: str = "Standard lore generated"
    welcome_image_url: Optional[str] = None
    initial_conflict: str = ""
    currency_system: str = "Standard currency"
    npc_count: int = 0
    quest_name: str = ""
    status: str = "ready"
    model_config = ConfigDict(extra="forbid")

# Module-level wrapper functions to avoid re-registration on instantiation
@function_tool
async def _calendar_tool_wrapper(
    ctx: RunContextWrapper[GameContext],
    params: CalendarToolParams,
) -> CalendarData:
    """Create calendar for the game world."""
    agent = ctx.context.get("agent_instance") or NewGameAgent()
    cal_params = CreateCalendarParams(environment_desc=params.environment_desc)
    return await agent.create_calendar(ctx, cal_params)

@function_tool
async def _spawn_npcs_tool_wrapper(ctx: RunContextWrapper[GameContext], count: int = 5) -> List[str]:
    """Spawn NPCs for the game world"""
    # Get the agent instance from context or create a temporary one
    agent = ctx.context.get("agent_instance")
    if not agent:
        agent = NewGameAgent()
    # Use the simplified params model that only has required fields
    params = SpawnNPCsRequiredParams(count=count)
    # Call spawn_npcs with full params for compatibility
    full_params = SpawnNPCsParams(
        environment_desc="",  # Not used internally
        day_names=[],  # Not used internally  
        count=count
    )
    return await agent.spawn_npcs(ctx, full_params)

@function_tool
async def _create_chase_schedule_tool_wrapper(ctx: RunContextWrapper[GameContext], environment_desc: str, day_names: List[str]) -> str:
    """Create Chase's schedule"""
    # Get the agent instance from context or create a temporary one
    agent = ctx.context.get("agent_instance")
    if not agent:
        agent = NewGameAgent()
    params = CreateChaseScheduleParams(environment_desc=environment_desc, day_names=day_names)
    return await agent.create_chase_schedule(ctx, params)

@function_tool
async def generate_environment_tool(
    ctx: RunContextWrapper[GameContext],
    params: GenerateEnvironmentParams,
) -> EnvironmentData:
    """Module‑level wrapper around NewGameAgent.generate_environment()."""
    agent = ctx.context.get("agent_instance") or NewGameAgent()
    return await agent.generate_environment(ctx, params)

class NewGameAgent:
    """Agent for handling new game creation process with Nyx governance integration"""
    
    def __init__(self):
        # Wrap methods as function tools
        self.environment_agent = Agent(
            name="EnvironmentCreator",
            instructions="""
            You are setting up a new daily-life sim environment with subtle, hidden layers of femdom and intrigue.
            
            Create a strictly valid JSON object with these keys:
            1. "setting_name" (string; a short, creative name for the environment)
            2. "environment_desc" (string; 1–3 paragraphs painting a vivid, cozy daily-life setting with faint, unsettling undertones)
            3. "environment_history" (string; a short paragraph on past events)
            4. "events" (array of objects with name, description, start_time, end_time, location, year, month, day, time_of_day)
            5. "locations" (array of objects with location_name, description, open_hours_json)
            6. "scenario_name" (string; a catchy title)
            7. "quest_data" (object with quest_name and quest_description)
            
            Focus on creating a mundane yet charming atmosphere with subtle undertones of control and influence.
            """,
            output_type=EnvironmentData,
            tools=[
                _calendar_tool_wrapper  # Use module-level wrapper
            ],
            model="gpt-4.1-nano"
        )
        
        self.npc_creator = Agent(
            name="NPCCreator",
            instructions="""
            You are crafting schedules for NPCs and the player "Chase" in a daily-life sim environment.
            
            Create NPCs with friendly facades that mask subtle control dynamics. Each NPC should have:
            - Detailed personality traits, likes, dislikes, and hobbies
            - A schedule that fits the setting
            - Subtle hints of dominance or control hidden behind routine activities
            
            Chase's schedule should feel normal yet guided, with subtle overlaps with the NPCs.
            """,
            tools=[
                _spawn_npcs_tool_wrapper,  # Use module-level wrapper
                _create_chase_schedule_tool_wrapper  # Use module-level wrapper
            ],
            model="gpt-4.1-nano"
        )
        
        self.narrative_agent = Agent(
            name="OpeningNarrator",
            instructions="""
            As Nyx, craft the opening narrative for a new game with subtle femdom undertones.
            
            Your voice should drape over Chase like a warm shroud—each word a silken thread stitching him into your world.
            Use guile and quiet lures to veil the control beneath. Create an immersive, atmospheric introduction
            that feels like a gentle descent into a comfortable routine while hinting at deeper layers.
            
            Address Chase as 'you,' drawing him through the veil with no whisper of retreat.
            """,
            model="gpt-4.1-nano"
        )
        
        # Main coordinating agent
        self.agent = Agent(
            name="NewGameDirector",
            instructions="""
            You are directing the creation of a new game world with subtle layers of femdom and intrigue,
            under the governance of Nyx.
            
            Coordinate the creation of the environment, lore, NPCs, and opening narrative.
            
            The game world should have:
            1. A detailed environment with locations and events
            2. Rich lore including factions, cultural elements, and history
            3. Multiple NPCs with schedules, personalities, and subtle control dynamics
            4. A player schedule that overlaps with NPCs
            5. An immersive opening narrative
            
            The lore should provide depth and context to the world, creating a rich backdrop for player interactions.
            All NPCs should be integrated with the lore, so they have knowledge about the world appropriate to their role.
            
            Maintain a balance between mundane daily life and subtle power dynamics.
            
            All actions must be approved by Nyx's governance system.
            """,
            tools=[
                function_tool(self.generate_environment),  # Wrap with function_tool
                function_tool(self.create_npcs_and_schedules),  # Wrap with function_tool
                function_tool(self.create_opening_narrative),  # Wrap with function_tool
                function_tool(self.finalize_game_setup),  # Wrap with function_tool
                function_tool(self.generate_lore)  # Wrap with function_tool
            ],
            output_type=GameCreationResult, 
            model="gpt-4.1-nano"
        )
        
        # Directive handler for processing Nyx directives
        self.directive_handler = None

    async def initialize_directive_handler(self, user_id: int, conversation_id: int):
        """Initialize the directive handler for this agent"""
        from nyx.integrate import get_central_governance
        governance = await get_central_governance(user_id, conversation_id)
        self.directive_handler = DirectiveHandler(
            user_id=user_id,
            conversation_id=conversation_id,
            agent_type=NEW_GAME_AGENT_NYX_TYPE, 
            agent_id=NEW_GAME_AGENT_NYX_ID, 
            governance=governance  # pass the object here
        )
        
        # Register handlers for different directive types
        self.directive_handler.register_handler(
            DirectiveType.ACTION, 
            self.handle_action_directive
        )
        self.directive_handler.register_handler(
            DirectiveType.OVERRIDE,
            self.handle_override_directive
        )
        
        # Don't start background processing here - do it after game setup is complete

    async def handle_action_directive(self, directive: dict) -> dict:
        """Handle an action directive from Nyx"""
        instruction = directive.get("instruction", "")
        logging.info(f"[NewGameAgent] Processing action directive: {instruction}")
        
        # Handle different instructions
        if "create new environment" in instruction.lower():
            # Extract parameters if provided
            params = directive.get("parameters", {})
            mega_name = params.get("mega_name", "New Setting")
            mega_desc = params.get("mega_desc", "A cozy town with hidden layers")
            
            # Simulate context with proper type structure
            ctx = type('RunContextWrapper', (object,), {
                'context': {
                    'user_id': self.directive_handler.user_id, 
                    'conversation_id': self.directive_handler.conversation_id,
                    'agent_instance': self
                }
            })()
            
            # Generate environment
            env_params = GenerateEnvironmentParams(
                mega_name=mega_name,
                mega_desc=mega_desc
            )
            result = await self.generate_environment(ctx, env_params)
            return {"result": "environment_generated", "data": result.dict()}
        
        return {"result": "action_not_recognized"}
    
    async def handle_override_directive(self, directive: dict) -> dict:
        """Handle an override directive from Nyx"""
        logging.info(f"[NewGameAgent] Processing override directive")
        
        # Extract override details
        override_action = directive.get("override_action", {})
        
        # Apply the override for future operations
        return {"result": "override_applied"}

    @with_governance_permission(AgentType.UNIVERSAL_UPDATER, "create_calendar")
    async def create_calendar(self, ctx: RunContextWrapper[GameContext], params: CreateCalendarParams) -> Dict[str, Any]:
        """
        Create an immersive calendar system for the game world.
        
        Args:
            params: CreateCalendarParams containing environment_desc
            
        Returns:
            Dictionary with calendar details
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        calendar_data = await update_calendar_names(user_id, conversation_id, params.environment_desc)
        return calendar_data

    @with_governance(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        action_type="generate_environment",
        action_description="Generated game environment for new game",
    )
    async def generate_environment(
        self,
        ctx: RunContextWrapper[GameContext],
        params: GenerateEnvironmentParams,
    ) -> EnvironmentData:
        """
        Generate a game environment and guarantee every location.open_hours_json
        is valid JSON, even if the LLM slips and outputs a plain string.
        """
        user_id         = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
    
        # ------------------------------------------------------------------ #
        # 1. Build the prompt with the strict open_hours_json instruction    #
        # ------------------------------------------------------------------ #
        env_comp_text = "\n".join(params.env_components) if params.env_components else "No components provided"
        enh_feat_text = ", ".join(params.enhanced_features) if params.enhanced_features else "No enhanced features"
        stat_mod_text = (
            ", ".join(f"{sm.stat_name}: {sm.modifier_value}" for sm in params.stat_modifiers)
            if params.stat_modifiers else "No stat modifiers"
        )
    
        prompt = f"""
        You are setting up a new daily-life sim environment with subtle, hidden layers of femdom and intrigue.
    
        Below is a merged environment concept:
        Mega Setting Name: {params.mega_name}
        Mega Description:
        {params.mega_desc}
    
        Using this as inspiration, create a cohesive environment with:
        - A creative setting name
        - A vivid environment description (1-3 paragraphs)
        - A brief history
        - Community events throughout the year
        - **At least 10 distinct locations.  
          Each location MUST include "open_hours_json" that is VALID JSON, e.g.  
          {{ "Mon-Sun": "24/7" }} or {{ "open": "18:00", "close": "23:59" }}.  
          Plain strings like "24/7" are NOT allowed.**
        - A scenario name
        - Initial quest data
    
        Reference these details:
        Environment components: {env_comp_text}
        Enhanced features: {enh_feat_text}
        Stat modifiers: {stat_mod_text}
        """
    
        # ------------------------------------------------------------------ #
        # 2. Run the EnvironmentCreator **without** a schema                 #
        # ------------------------------------------------------------------ #
        ctx.context["agent_instance"] = self
    
        # Re-use the same instructions/tools but drop the output_type
        env_agent_raw = Agent(
            name="EnvironmentCreatorRaw",
            instructions=self.environment_agent.instructions,
            tools=[_calendar_tool_wrapper],
            model="gpt-4.1-nano",
            output_type=None,          # <-- no schema validation yet
        )
    
        raw_run = await Runner.run(env_agent_raw, prompt, context=ctx.context)
    
        # raw_run.final_output is either str or dict; ensure we have a dict
        raw_data: Dict[str, Any] = (
            raw_run.final_output
            if isinstance(raw_run.final_output, dict)
            else json.loads(raw_run.final_output)
        )
    
        # ------------------------------------------------------------------ #
        # 3. Sanity-wrap any bad open_hours_json literals                     #
        # ------------------------------------------------------------------ #
        for loc in raw_data.get("locations", []):
            oh = loc.get("open_hours_json")
            if isinstance(oh, dict):
                continue                       # already good
            if isinstance(oh, str):
                try:
                    json.loads(oh)             # does it parse?
                except json.JSONDecodeError:
                    loc["open_hours_json"] = {"fallback": oh}
    
        # ------------------------------------------------------------------ #
        # 4. Validate with EnvironmentData (now guaranteed clean)             #
        # ------------------------------------------------------------------ #
        env_obj: EnvironmentData = EnvironmentData.model_validate(raw_data)
    
        # ------------------------------------------------------------------ #
        # 5. Calendar + canonical DB writes (same as your old code)           #
        # ------------------------------------------------------------------ #
        calendar_params = CreateCalendarParams(environment_desc=env_obj.environment_desc)
        calendar_data   = await self.create_calendar(ctx, calendar_params)
    
        from lore.core import canon
        async with get_db_connection_context() as conn, conn.transaction():
            canon_ctx = type("Ctx", (), {"user_id": user_id, "conversation_id": conversation_id})
    
            await canon.create_game_setting(
                canon_ctx, conn,
                env_obj.setting_name,
                environment_desc   = env_obj.environment_desc,
                environment_history= env_obj.environment_history,
                calendar_data      = calendar_data,
                scenario_name      = env_obj.scenario_name,
            )
    
            for ev in env_obj.events:
                await canon.find_or_create_event(
                    canon_ctx, conn,
                    ev.name, description=ev.description,
                    start_time=ev.start_time, end_time=ev.end_time,
                    location=ev.location, year=ev.year,
                    month=ev.month, day=ev.day, time_of_day=ev.time_of_day,
                )
    
            for loc in env_obj.locations:
                await canon.find_or_create_location(
                    canon_ctx, conn,
                    loc.location_name,
                    description      = loc.description,
                    location_type    = loc.type,
                    notable_features = loc.features,
                    open_hours       = loc.open_hours_json,   # dict — ready to store
                )
    
            qd = env_obj.quest_data
            await canon.find_or_create_quest(
                canon_ctx, conn,
                qd.quest_name,
                progress_detail=qd.quest_description,
                status="In Progress",
            )
    
        return env_obj


    async def _apply_setting_stat_modifiers(
        self,
        user_id: int,
        conversation_id: int,
        stat_mods: list["StatModifier"],
    ) -> None:
        """
        Persist the 'setting-based' stat modifiers and apply them once at game start.

        * `stat_mods` is the list you built earlier (List[StatModifier]).
        * Each modifier value is assumed to be an **absolute point delta**
          (-10 … +10).  If you want percentage, convert here.
        """
        # --- normalise to a {stat_name: delta} dict -------------------------
        clean: dict[str, float] = {}
        for sm in stat_mods:
            try:
                delta = float(sm.modifier_value)
                if delta == 0:
                    continue
                clean[sm.stat_name.lower()] = delta
            except (ValueError, TypeError):
                logging.warning(
                    "[NewGameAgent] Ignored non-numeric modifier %s=%s",
                    sm.stat_name, sm.modifier_value
                )

        if not clean:
            logging.info("[NewGameAgent] No valid setting modifiers to apply.")
            return

        # --- store a canonical copy for later systems ----------------------
        canon_ctx = type(
            "Ctx", (),
            {"user_id": user_id, "conversation_id": conversation_id}
        )()

        async with get_db_connection_context() as conn:
            await canon.update_current_roleplay(
                canon_ctx, conn,
                "SettingStatModifiers",
                json.dumps(clean),
            )

        # --- apply the changes to PlayerStats -----------------------------
        logging.info(
            "[NewGameAgent] Applying initial setting modifiers: %s", clean
        )
        result = await apply_stat_change(
            user_id,
            conversation_id,
            changes=clean,
            cause="initial_setting_modifier",
        )
        if not result.get("success"):
            logging.error(
                "[NewGameAgent] Failed to apply setting modifiers: %s", result
            )
        else:
            logging.info(
                "[NewGameAgent] Setting modifiers applied and recorded."
            )

    @with_governance_permission(AgentType.UNIVERSAL_UPDATER, "spawn_npcs")
    async def spawn_npcs(self, ctx: RunContextWrapper[GameContext], params: SpawnNPCsParams) -> List[str]:
        """
        Spawn multiple NPCs for the game world.
        
        Args:
            params: SpawnNPCsParams with environment_desc, day_names, and count
            
        Returns:
            List of NPC IDs
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        # Create an instance of NPCCreationHandler
        npc_handler = NPCCreationHandler()
        
        # Use the class method directly - ensure we await properly
        # This MUST await all internal async operations to avoid race conditions
        # Per fix #3: NPCCreationHandler.spawn_multiple_npcs must use:
        #   npc_ids = await asyncio.gather(*coros)  # MUST await
        #   await self._commit_all(npc_ids)         # transaction/flush
        npc_ids = await npc_handler.spawn_multiple_npcs(
            ctx=ctx,
            count=params.count
        )
        
        # Verify NPCs were actually created and committed to DB
        async with get_db_connection_context() as conn:
            actual_count = await conn.fetchval("""
                SELECT COUNT(*) FROM NPCStats
                WHERE user_id = $1 AND conversation_id = $2
            """, user_id, conversation_id)
            
            if actual_count < params.count:
                logger.warning(f"Expected {params.count} NPCs but only found {actual_count}")
        
        return npc_ids

    @with_governance_permission(AgentType.UNIVERSAL_UPDATER, "create_chase_schedule")
    async def create_chase_schedule(self, ctx: RunContextWrapper[GameContext], params: CreateChaseScheduleParams) -> str:
        """
        Create a schedule for the player "Chase".
        
        Args:
            params: CreateChaseScheduleParams with environment_desc and day_names
            
        Returns:
            JSON string with Chase's schedule
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        # Create a basic schedule structure
        default_schedule = {}
        for day in params.day_names:
            default_schedule[day] = {
                "Morning": f"Chase wakes up and prepares for the day",
                "Afternoon": f"Chase attends to their responsibilities",
                "Evening": f"Chase spends time on personal activities",
                "Night": f"Chase returns home and rests"
            }
        
        # Store in database
        async with get_db_connection_context() as conn:
            await conn.execute("""
                INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                VALUES($1, $2, 'ChaseSchedule', $3)
                ON CONFLICT (user_id, conversation_id, key)
                DO UPDATE SET value=EXCLUDED.value
            """, user_id, conversation_id, json.dumps(default_schedule))
        
        # Store as player memory using the new memory system
        try:
            memory_system = await MemorySystem.get_instance(user_id, conversation_id)
            
            # Create a journal entry for the schedule
            schedule_summary = "My typical schedule for the week: "
            for day, activities in default_schedule.items():
                day_summary = f"\n{day}: "
                for period, activity in activities.items():
                    day_summary += f"{period.lower()}: {activity}; "
                schedule_summary += day_summary
                
            await memory_system.add_journal_entry(
                player_name="Chase",
                entry_text=schedule_summary,
                entry_type="schedule"
            )
        except Exception as e:
            logging.error(f"Error storing player schedule in memory system: {e}")
        
        # Return as JSON string to avoid dict issues
        return json.dumps(default_schedule)

    async def _load_environment_from_db(self, user_id: int, conversation_id: int) -> EnvironmentInfo:
        """Load environment data from database if not provided"""
        async with get_db_connection_context() as conn:
            # Get setting info
            setting_row = await conn.fetchrow("""
                SELECT value FROM CurrentRoleplay
                WHERE user_id = $1 AND conversation_id = $2 AND key = 'CurrentSetting'
            """, user_id, conversation_id)
            
            desc_row = await conn.fetchrow("""
                SELECT value FROM CurrentRoleplay
                WHERE user_id = $1 AND conversation_id = $2 AND key = 'EnvironmentDesc'
            """, user_id, conversation_id)
            
            history_row = await conn.fetchrow("""
                SELECT value FROM CurrentRoleplay
                WHERE user_id = $1 AND conversation_id = $2 AND key = 'EnvironmentHistory'
            """, user_id, conversation_id)
            
            scenario_row = await conn.fetchrow("""
                SELECT value FROM CurrentRoleplay
                WHERE user_id = $1 AND conversation_id = $2 AND key = 'ScenarioName'
            """, user_id, conversation_id)
            
            return EnvironmentInfo(
                setting_name=setting_row["value"] if setting_row else "Unknown Setting",
                environment_desc=desc_row["value"] if desc_row else "A mysterious environment",
                environment_history=history_row["value"] if history_row else "History unknown",
                scenario_name=scenario_row["value"] if scenario_row else "New Adventure"
            )

    @with_governance(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        action_type="create_npcs_and_schedules",
        action_description="Created NPCs and schedules for new game"
    )
    async def create_npcs_and_schedules(self, ctx: RunContextWrapper[GameContext], params: CreateNPCsAndSchedulesParams) -> NPCScheduleData:
        """
        Create NPCs and schedules for the game world using canonical functions.
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        from lore.core import canon
        
        # If environment_data is None, load it from DB
        if params.environment_data is None:
            params.environment_data = await self._load_environment_from_db(user_id, conversation_id)
        
        # Get calendar data for day names
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow("""
                SELECT value FROM CurrentRoleplay
                WHERE user_id=$1 AND conversation_id=$2 AND key='CalendarNames'
                LIMIT 1
            """, user_id, conversation_id)
            
            calendar_data = json.loads(row["value"]) if row else {}
            day_names = calendar_data.get("days", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        
        # Create NPCs (they should already use canonical creation through NPCCreationHandler)
        environment_desc = params.environment_data.environment_desc + "\n\n" + params.environment_data.environment_history
        
        # Create params for spawn_npcs
        spawn_params = SpawnNPCsParams(
            environment_desc=environment_desc,
            day_names=day_names,
            count=5
        )
        
        # Add agent instance to context for sub-tools
        ctx.context["agent_instance"] = self
        
        npc_ids = await self.spawn_npcs(ctx, spawn_params)
        
        # Create Chase's schedule
        chase_params = CreateChaseScheduleParams(
            environment_desc=environment_desc,
            day_names=day_names
        )
        chase_schedule_json = await self.create_chase_schedule(ctx, chase_params)
        
        # Parse the schedule for canonical storage
        chase_schedule = json.loads(chase_schedule_json)
        
        # Store Chase's schedule canonically
        async with get_db_connection_context() as conn:
            canon_ctx = type('obj', (object,), {
                'user_id': user_id, 
                'conversation_id': conversation_id
            })
            
            await canon.store_player_schedule(
                canon_ctx, conn,
                "Chase",
                chase_schedule
            )
        
        # Create result with proper model
        result = NPCScheduleData(
            npc_ids=npc_ids,
            chase_schedule_json=chase_schedule_json
        )
        
        return result

    @with_governance(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        action_type="create_opening_narrative",
        action_description="Created opening narrative for new game"
    )
    async def create_opening_narrative(self, ctx: RunContextWrapper[GameContext], params: CreateOpeningNarrativeParams) -> str:
        """
        Create the opening narrative for the game using canonical functions.
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        from lore.core import canon
        from routes.story_routes import build_aggregator_text 
        
        # Get aggregator data
        aggregator_data = await get_aggregated_roleplay_context(user_id, conversation_id, "Chase")
        aggregator_text = build_aggregator_text(aggregator_data)
        
        # Get calendar data for first day name
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow("""
                SELECT value FROM CurrentRoleplay
                WHERE user_id=$1 AND conversation_id=$2 AND key='CalendarNames'
                LIMIT 1
            """, user_id, conversation_id)
            
            calendar_data = json.loads(row["value"]) if row else {}
            day_names = calendar_data.get("days", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
            first_day_name = day_names[0] if day_names else "the first day"
        
        # Create prompt for the narrative agent
        prompt = f"""
        As Nyx, let your voice drape over Chase like a warm shroud—each word a silken thread stitching him into my world, soft yet relentless. 
        Cloak my grip in whispers and half-smiles, using guile and quiet lures to veil the femdom storm beneath. 
        Ease him into a false haven of routine—a familiar hum so natural he won't feel the snare until it's locked, the truth dawning only as he treads deeper. 
        Your prose is a slow graze across skin—intimate, lingering, stirring shivers he can't place, each phrase a pull into my shadowed embrace. 
        Structure this descent as a gentle drift into dusk, transitions seamless as a held breath, folding him into the tale without a crack. 
        Address Chase as 'you,' drawing him through the veil with no whisper of retreat:
    
        {aggregator_text}

        As {first_day_name} unfurls like a soft tide across the expanse, unveil Chase's world through a haze of everyday ease—a place where the ordinary cloaks a deeper pulse. 
        Drench it in tender detail—the light spilling gentle over your frame, a scent of home laced with an edge you'd swear you know, the rhythm of your surroundings swaying to a beat you've always followed. 
        Paint the souls threading your path not as strangers but as anchors of your days—companions, perhaps, or echoes of habit, their words and nearness wrapping you in a comfort too easy to trust. 
        Stitch your hours into the weave as a string of quiet moments—your day already humming with a shape you feel more than plan, nudging you toward a familiar haunt by morning's rise, a task to tend soon after, then a place to be as shadows stretch, each step simple, each one seen. 
        Let a shadow ripple in the stillness—a voice that lingers a touch too long, a look that pins you soft and sure, a passing touch that rests with unspoken weight, all so woven into the day you'd call it nothing. 
        End with you turning toward that first call—maybe a face waiting where you're due, maybe a pull to somewhere you've been before—the air humming thicker, a flicker of promise brushing your senses, a step that feels your own but sings with my intent. 
        Hold it gentle—my sway lives in the unsaid, the softest hums hiding the deepest hooks, every line a lure of safety veiling a pull he'll fall into blind. 
        No mechanics, no tells—just a cocoon of ease and shadow, immersive, teasing, where every plain moment cradles a depth he won't see coming, guiding him where I want him to go.
        """
        
        # Run the narrative agent
        result = await Runner.run(
            self.narrative_agent,
            prompt,
            context=ctx.context
        )
        
        opening_narrative = result.final_output
        
        # Store the opening narrative canonically
        async with get_db_connection_context() as conn:
            canon_ctx = type('obj', (object,), {
                'user_id': user_id, 
                'conversation_id': conversation_id
            })
            
            await canon.create_opening_message(
                canon_ctx, conn,
                "Nyx",
                opening_narrative
            )
        
        return opening_narrative

    async def _is_setup_complete(self, user_id: int, conversation_id: int) -> bool:
        """Check if the game setup is complete before marking as ready"""
        async with get_db_connection_context() as conn:
            # Check NPCs
            npc_count = await conn.fetchval("""
                SELECT COUNT(*) FROM NPCStats 
                WHERE user_id = $1 AND conversation_id = $2
            """, user_id, conversation_id)
            
            # Check locations
            location_count = await conn.fetchval("""
                SELECT COUNT(*) FROM Locations 
                WHERE user_id = $1 AND conversation_id = $2
            """, user_id, conversation_id)
            
            # Check key roleplay data - loop through keys to avoid array type issues
            roleplay_keys = ['CurrentSetting', 'EnvironmentDesc', 'ChaseSchedule', 'LoreSummary']
            roleplay_count = 0
            for key in roleplay_keys:
                exists = await conn.fetchval("""
                    SELECT EXISTS(
                        SELECT 1 FROM CurrentRoleplay 
                        WHERE user_id = $1 AND conversation_id = $2 AND key = $3
                    )
                """, user_id, conversation_id, key)  # Parameters: $1=user_id, $2=conversation_id, $3=key
                if exists:
                    roleplay_count += 1
            
            logger.info(f"Setup check - NPCs: {npc_count}, Locations: {location_count}, Roleplay keys: {roleplay_count}/{len(roleplay_keys)}")
            
            # Require at least 5 NPCs, 10 locations, and all key roleplay data
            return npc_count >= 5 and location_count >= 10 and roleplay_count >= len(roleplay_keys)

    @with_governance(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        action_type="finalize_game_setup",
        action_description="Finalized game setup including lore, conflict, currency and image"
    )
    async def finalize_game_setup(self, ctx: RunContextWrapper[GameContext], params: FinalizeGameSetupParams) -> FinalizeResult:
        """
        Finalize game setup including lore generation, conflict generation and image generation.
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        from lore.core import canon
        
        # Get the environment description for lore generation
        async with get_db_connection_context() as conn:
            canon_ctx = type('obj', (object,), {
                'user_id': user_id, 
                'conversation_id': conversation_id
            })
            
            row = await conn.fetchrow("""
                SELECT value FROM CurrentRoleplay
                WHERE user_id = $1 AND conversation_id = $2 AND key = 'EnvironmentDesc'
            """, user_id, conversation_id)
            
            environment_desc = row["value"] if row else "A mysterious environment with hidden layers of complexity."
            
            # Get NPC IDs for lore integration
            rows = await conn.fetch("""
                SELECT npc_id FROM NPCStats
                WHERE user_id = $1 AND conversation_id = $2
            """, user_id, conversation_id)
            
            npc_ids = [row["npc_id"] for row in rows] if rows else []
            
        # Initialize and generate lore
        try:
            # Get the lore system instance and initialize it
            lore_system = DynamicLoreGenerator.get_instance(user_id, conversation_id)
            await lore_system.initialize()
            
            # Generate comprehensive lore based on the environment
            logging.info(f"Generating lore for new game (user_id={user_id}, conversation_id={conversation_id})")
            lore_result = await lore_system.generate_complete_lore(environment_desc)

            
            # Integrate lore with NPCs if we have any
            if npc_ids:
                logging.info(f"Integrating lore with {len(npc_ids)} NPCs")
                await lore_system.integrate_lore_with_npcs(npc_ids)
                
            lore_summary = f"Generated {len(lore_result.get('factions', []))} factions, {len(lore_result.get('cultural_elements', []))} cultural elements, and {len(lore_result.get('locations', []))} locations"
            
            # Store lore summary canonically
            async with get_db_connection_context() as conn:
                await canon.update_current_roleplay(
                    canon_ctx, conn,
                    user_id, conversation_id,
                    'LoreSummary', lore_summary
                )
                
            logging.info(f"Lore generation complete: {lore_summary}")
        except Exception as e:
            logging.error(f"Error generating lore: {e}", exc_info=True)
            lore_summary = "Failed to generate lore"
        
        # Generate initial conflict
        try:
            from logic.conflict_system.conflict_integration import ConflictSystemIntegration
            
            conflict_integration = ConflictSystemIntegration(user_id, conversation_id)
            initial_conflict = await conflict_integration.generate_conflict({
                "conflict_type": "major",
                "intensity": "medium",
                "player_involvement": "indirect"
            })
            conflict_name = initial_conflict.get("conflict_details", {}).get("name", "Unnamed Conflict")
        except Exception as e:
            logging.error(f"Error generating initial conflict: {e}")
            conflict_name = "Failed to generate conflict"
        
        # Generate currency system canonically
        try:
            from logic.currency_generator import CurrencyGenerator
            currency_gen = CurrencyGenerator(user_id, conversation_id)
            currency_system = await currency_gen.get_currency_system()
            
            # Create currency canonically
            async with get_db_connection_context() as conn:
                await canon.find_or_create_currency_system(
                    canon_ctx, conn,
                    currency_name=currency_system['currency_name'],
                    currency_plural=currency_system['currency_plural'],
                    minor_currency_name=currency_system.get('minor_currency_name'),
                    minor_currency_plural=currency_system.get('minor_currency_plural'),
                    exchange_rate=currency_system.get('exchange_rate', 100),
                    currency_symbol=currency_system.get('currency_symbol', '$'),
                    description=currency_system.get('description', ''),
                    setting_context=currency_system.get('setting_context', '')
                )
                
            currency_name = f"{currency_system['currency_name']} / {currency_system['currency_plural']}"
        except Exception as e:
            logging.error(f"Error generating currency system: {e}")
            currency_name = "Standard currency"
            
        # Generate welcome image
        scene_data_json = json.dumps({
            "scene_data": {
                "npc_names": [],
                "setting": await self._get_setting_name(ctx),
                "actions": ["introduction", "welcome"],
                "mood": "atmospheric",
                "expressions": {},
                "npc_positions": {},
                "visibility_triggers": {
                    "character_introduction": True,
                    "significant_location": True,
                    "emotional_intensity": 50,
                    "intimacy_level": 20,
                    "appearance_change": False
                }
            },
            "image_generation": {
                "generate": True,
                "priority": "high",
                "focus": "setting",
                "framing": "wide_shot",
                "reason": "Initial scene visualization"
            }
        })
        
        # Parse back to dict for the function call
        scene_data = json.loads(scene_data_json)
        
        image_result = await generate_roleplay_image_from_gpt(scene_data, user_id, conversation_id)
        welcome_image_url = None
        if image_result and "image_urls" in image_result and image_result["image_urls"]:
            welcome_image_url = image_result["image_urls"][0]
            
            # Store the image URL canonically
            async with get_db_connection_context() as conn:
                await canon.update_current_roleplay(
                    canon_ctx, conn,
                    user_id, conversation_id,
                    'WelcomeImageUrl', welcome_image_url
                )
        
        # Return structured result - but DON'T mark as ready yet
        return FinalizeResult(
            status="finalized",  # Not "ready" yet
            welcome_image_url=welcome_image_url,
            lore_summary=lore_summary,
            initial_conflict=conflict_name,
            currency_system=currency_name
        )

    async def _get_setting_name(self, ctx: RunContextWrapper[GameContext]) -> str:
        """Helper method to get the setting name from the database"""
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow("""
                SELECT value FROM CurrentRoleplay
                WHERE user_id=$1 AND conversation_id=$2 AND key='CurrentSetting'
                LIMIT 1
            """, user_id, conversation_id)
            
            return row["value"] if row else "Unknown Setting"

    @with_governance(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        action_type="generate_lore",
        action_description="Generated lore for new game environment"
    )
    async def generate_lore(self, ctx: RunContextWrapper[GameContext], params: GenerateLoreParams) -> LoreResult:
        """
        Generate comprehensive lore for the game environment.
        
        Args:
            params: GenerateLoreParams with environment_desc
            
        Returns:
            LoreResult with generated lore data
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        try:
            # Get the lore system instance and initialize it
            lore_system = DynamicLoreGenerator.get_instance(user_id, conversation_id)
            await lore_system.initialize()
            
            # Generate comprehensive lore based on the environment
            logging.info(f"Generating lore for environment: {params.environment_desc[:100]}...")
            lore_result = await lore_system.generate_complete_lore(params.environment_desc)

            
            # Get NPC IDs for lore integration
            async with get_db_connection_context() as conn:
                rows = await conn.fetch("""
                    SELECT npc_id FROM NPCStats
                    WHERE user_id = $1 AND conversation_id = $2
                """, user_id, conversation_id)
                
                npc_ids = [row["npc_id"] for row in rows] if rows else []
                
            # Integrate lore with NPCs if we have any
            if npc_ids:
                logging.info(f"Integrating lore with {len(npc_ids)} NPCs")
                await lore_system.integrate_lore_with_npcs(npc_ids)
            
            # Create summary for easy reference
            factions_count = len(lore_result.get('factions', []))
            cultural_count = len(lore_result.get('cultural_elements', []))
            locations_count = len(lore_result.get('locations', []))
            events_count = len(lore_result.get('historical_events', []))
            
            lore_summary = f"Generated {factions_count} factions, {cultural_count} cultural elements, {locations_count} locations, and {events_count} historical events"
            
            # Store lore summary in the database
            async with get_db_connection_context() as conn:
                await conn.execute("""
                    INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                    VALUES($1, $2, 'LoreSummary', $3)
                    ON CONFLICT (user_id, conversation_id, key)
                    DO UPDATE SET value=EXCLUDED.value
                """, user_id, conversation_id, lore_summary)
            
            return LoreResult(
                lore_summary=lore_summary,
                factions_count=factions_count,
                cultural_elements_count=cultural_count,
                locations_count=locations_count,
                historical_events_count=events_count
            )
            
        except Exception as e:
            logging.error(f"Error generating lore: {e}", exc_info=True)
            return LoreResult(
                lore_summary="Failed to generate lore",
                factions_count=0,
                cultural_elements_count=0,
                locations_count=0,
                historical_events_count=0,
                error=f"Failed to generate lore: {str(e)}"
            )

    # Update the process_new_game method to be deterministic
    @with_governance(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        action_type="process_new_game",
        action_description="Processed complete new game creation workflow"
    )
    async def process_new_game(self, ctx, conversation_data: Dict[str, Any]) -> ProcessNewGameResult:
        # Extract user_id from context
        user_id = ctx.user_id
        conversation_id = None
        
        try:
            provided_convo_id = conversation_data.get("conversation_id")
            from nyx.integrate import get_central_governance
        
            async with get_db_connection_context() as conn:
                if not provided_convo_id:
                    row = await conn.fetchrow("""
                        INSERT INTO conversations (user_id, conversation_name, status)
                        VALUES ($1, 'New Game - Initializing', 'processing')
                        RETURNING id
                    """, user_id)
                    conversation_id = row["id"]
                else:
                    conversation_id = provided_convo_id
                    row = await conn.fetchrow("""
                        SELECT id FROM conversations WHERE id=$1 AND user_id=$2
                    """, conversation_id, user_id)
                    if not row:
                        raise Exception(f"Conversation {conversation_id} not found or unauthorized")
                    
                    # Update status to processing
                    await conn.execute("""
                        UPDATE conversations 
                        SET status='processing', conversation_name='New Game - Initializing'
                        WHERE id=$1 AND user_id=$2
                    """, conversation_id, user_id)
        
                # Clear old data
                logger.info(f"Clearing old data for conversation {conversation_id}")
                tables = ["Events", "PlannedEvents", "PlayerInventory", "Quests",
                          "NPCStats", "Locations", "SocialLinks", "CurrentRoleplay"]
                for t in tables:
                    await conn.execute(f"DELETE FROM {t} WHERE user_id=$1 AND conversation_id=$2",
                                       user_id, conversation_id)
                                       
            # Initialize player stats
            await insert_default_player_stats_chase(user_id, conversation_id)
            logger.info(f"Default player stats for Chase inserted")
    
            # Update status - environment generation
            async with get_db_connection_context() as conn:
                await conn.execute("""
                    UPDATE conversations 
                    SET conversation_name='New Game - Creating Environment'
                    WHERE id=$1 AND user_id=$2
                """, conversation_id, user_id)
        
            # Get governance ONCE - this will handle all initialization
            governance = await get_central_governance(user_id, conversation_id)
            
            # Initialize directive handler WITHOUT starting background processing yet
            await self.initialize_directive_handler(user_id, conversation_id)
            
            # Register this agent with governance
            await governance.register_agent(
                agent_type=NEW_GAME_AGENT_NYX_TYPE,
                agent_instance=self,
                agent_id=NEW_GAME_AGENT_NYX_ID
            )
            
            # Note: StoryDirector initialization moved to after game setup is complete
            
            # Process any pending directives
            if self.directive_handler:
                await self.directive_handler.process_directives(force_check=True)
            
            # Gather environment components
            from routes.settings_routes import generate_mega_setting_logic
            mega_data = await generate_mega_setting_logic()
            env_comps = mega_data.get("selected_settings") or mega_data.get("unique_environments") or []
            if not env_comps:
                env_comps = [
                    "A sprawling cyberpunk metropolis under siege by monstrous clans",
                    "Floating archaic ruins steeped in ancient rituals",
                    "Futuristic tech hubs that blend magic and machinery"
                ]
            enh_feats         = mega_data.get("enhanced_features", [])
            stat_mods_raw     = mega_data.get("stat_modifiers", {})   # already numeric!
            
            # Build pydantic StatModifier objects
            stat_mods = [
                StatModifier(stat_name=k, modifier_value=v)
                for k, v in stat_mods_raw.items()
            ]
            
            # Apply them to the DB / player stats
            await self._apply_setting_stat_modifiers(
                user_id,
                conversation_id,
                stat_mods,
            )
            
            mega_name = mega_data.get("mega_name", "Untitled Mega Setting")
            mega_desc = mega_data.get("mega_description", "No environment generated")
            
            # Set up context wrapper for the agent methods
            ctx_wrap = type('RunContextWrapper', (object,), {
                'context': {
                    'user_id': user_id,
                    'conversation_id': conversation_id,
                    'db_dsn': DB_DSN,
                    'agent_instance': self
                }
            })()
            
            # Update status - running agent
            async with get_db_connection_context() as conn:
                await conn.execute("""
                    UPDATE conversations 
                    SET conversation_name='New Game - Building World'
                    WHERE id=$1 AND user_id=$2
                """, conversation_id, user_id)
            
            # DETERMINISTIC PIPELINE - Call tools directly instead of using Runner.run
            logger.info(f"Starting deterministic game creation pipeline for conversation {conversation_id}")
            
            # 1. Generate Environment
            logger.info("Step 1: Generating environment...")
            env_params = GenerateEnvironmentParams(
                mega_name=mega_name,
                mega_desc=mega_desc,
                env_components=env_comps,
                enhanced_features=enh_feats,
                stat_modifiers=stat_mods
            )
            env = await self.generate_environment(ctx_wrap, env_params)
            logger.info(f"Environment generated: {env.setting_name}")
            
            # Update status
            async with get_db_connection_context() as conn:
                await conn.execute("""
                    UPDATE conversations 
                    SET conversation_name='New Game - Creating NPCs'
                    WHERE id=$1 AND user_id=$2
                """, conversation_id, user_id)
            
            # 2. Create NPCs & schedules
            logger.info("Step 2: Creating NPCs and schedules...")
            npc_params = CreateNPCsAndSchedulesParams(
                environment_data=EnvironmentInfo(
                    setting_name=env.setting_name,
                    environment_desc=env.environment_desc,
                    environment_history=env.environment_history,
                    scenario_name=env.scenario_name
                )
            )
            npc_sched = await self.create_npcs_and_schedules(ctx_wrap, npc_params)
            logger.info(f"Created {len(npc_sched.npc_ids)} NPCs")
            
            # Update status
            async with get_db_connection_context() as conn:
                await conn.execute("""
                    UPDATE conversations 
                    SET conversation_name='New Game - Writing Narrative'
                    WHERE id=$1 AND user_id=$2
                """, conversation_id, user_id)
            
            # 3. Create opening narrative
            logger.info("Step 3: Creating opening narrative...")
            narrative_params = CreateOpeningNarrativeParams(
                environment_data=EnvironmentInfo(
                    setting_name=env.setting_name,
                    environment_desc=env.environment_desc,
                    environment_history=env.environment_history,
                    scenario_name=env.scenario_name
                ),
                npc_schedule_data=npc_sched
            )
            opening = await self.create_opening_narrative(ctx_wrap, narrative_params)
            logger.info("Opening narrative created")
            
            # Update status
            async with get_db_connection_context() as conn:
                await conn.execute("""
                    UPDATE conversations 
                    SET conversation_name='New Game - Finalizing'
                    WHERE id=$1 AND user_id=$2
                """, conversation_id, user_id)
            
            # 4. Finalize game setup
            logger.info("Step 4: Finalizing game setup...")
            finalize_params = FinalizeGameSetupParams(
                opening_narrative=opening
            )
            final = await self.finalize_game_setup(ctx_wrap, finalize_params)
            logger.info(f"Game setup finalized: {final.lore_summary}")
            
            # 5. Verify setup is complete before marking ready
            logger.info("Step 5: Verifying game setup completeness...")
            if not await self._is_setup_complete(user_id, conversation_id):
                raise Exception("Game setup incomplete - missing required components")
            
            # Update conversation with final name and ready status
            async with get_db_connection_context() as conn:
                await conn.execute("""
                    UPDATE conversations 
                    SET status='ready', 
                        conversation_name=$3
                    WHERE id=$1 AND user_id=$2
                """, conversation_id, user_id, f"Game: {env.setting_name}")
            
            # Initialize StoryDirector AFTER game setup is complete
            logger.info("Initializing StoryDirector")
            try:
                from story_agent.story_director_agent import StoryDirector
                story_director = StoryDirector(user_id, conversation_id)
                # Register it with the EXISTING governance
                await governance.register_agent(
                    agent_type=AgentType.STORY_DIRECTOR,
                    agent_instance=story_director,
                    agent_id="story_director"
                )
                logger.info(f"StoryDirector initialized and registered for {conversation_id}")
            except Exception as e:
                logger.error(f"StoryDirector init failed: {e}", exc_info=True)
                # Don't fail the entire game creation if StoryDirector fails
            
            # NOW start background directive processing after everything is set up
            if self.directive_handler:
                await self.directive_handler.start_background_processing()
                logger.info("Started background directive processing")
            
            logger.info(f"New game creation completed for conversation {conversation_id}")
            
            # Return structured result
            return ProcessNewGameResult(
                message=f"New game started. environment={env.setting_name}, conversation_id={conversation_id}",
                scenario_name=env.scenario_name,
                environment_name=env.setting_name,
                environment_desc=env.environment_desc,
                lore_summary=final.lore_summary,
                conversation_id=conversation_id,
                welcome_image_url=final.welcome_image_url,
                status="ready"
            )
            
        except Exception as e:
            logger.error(f"Error in process_new_game: {e}", exc_info=True)
            
            # Update conversation status to failed
            if conversation_id:
                try:
                    async with get_db_connection_context() as conn:
                        await conn.execute("""
                            UPDATE conversations 
                            SET status='failed', 
                                conversation_name='New Game - Creation Failed'
                            WHERE id=$1 AND user_id=$2
                        """, conversation_id, user_id)
                except:
                    pass
                    
            raise  # Re-raise the exception to be handled by Celery
                    
# Register with governance system
async def register_with_governance(user_id: int, conversation_id: int) -> None:
    """
    Register the NewGameAgent with the Nyx governance system.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
    """
    try:
        # Import here to avoid circular import
        from nyx.integrate import get_central_governance
        
        # Get the governance system
        governance = await get_central_governance(user_id, conversation_id)
        
        # Create the agent
        agent = NewGameAgent()
        
        # Register with governance (using consistent ID)
        await governance.register_agent(
            agent_type=AgentType.UNIVERSAL_UPDATER,
            agent_instance=agent,
            agent_id=NEW_GAME_AGENT_NYX_ID
        )
        
        # Issue directive to be ready to create new games
        await governance.issue_directive(
            agent_type=AgentType.UNIVERSAL_UPDATER,
            agent_id=NEW_GAME_AGENT_NYX_ID, 
            directive_type=DirectiveType.ACTION,
            directive_data={
                "instruction": "Initialize and stand ready to create new game environments",
                "scope": "initialization"
            },
            priority=DirectivePriority.MEDIUM,
            duration_minutes=24*60  # 24 hours
        )
        
        logging.info(f"NewGameAgent registered with Nyx governance system for user {user_id}, conversation {conversation_id}")
    except Exception as e:
        logging.error(f"Error registering NewGameAgent with governance: {e}")
