# new_game_agent.py

import logging
import json
import asyncio
import uuid
import os
import functools
import random
from datetime import datetime
from typing import Optional, List, Dict, Any

from logic.setting_rules import synthesize_setting_rules

from agents import Agent, Runner, function_tool, GuardrailFunctionOutput, InputGuardrail, RunContextWrapper, input_guardrail, output_guardrail, OutputGuardrail
from pydantic import BaseModel, Field, ConfigDict, field_validator

from memory.wrapper import MemorySystem
from logic.stats_logic import insert_default_player_stats_chase, apply_stat_change
from lore.core.context import CanonicalContext

# Import your existing modules
from logic.calendar import update_calendar_names, load_calendar_names
from logic.time_cycle import set_current_time, TIME_PHASES
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
    npc_ids: List[int] = Field(default_factory=list)  # Changed from List[str] to List[int]
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
    npc_schedule_data: Optional[NPCScheduleData] = None
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
    opening_narrative: str  # Add this field
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

@function_tool
async def _calendar_tool_wrapper(
    ctx: RunContextWrapper[GameContext],
    params: CalendarToolParams,
) -> str:
    """Create calendar for the game world."""
    call_id = str(uuid.uuid4())[:8]
    
    # Log the call with unique ID
    logger.info(f"[{call_id}] _calendar_tool_wrapper called with env_desc length: {len(params.environment_desc) if params.environment_desc else 0}")
    
    # Validate environment description
    if not params.environment_desc or params.environment_desc.strip() == "":
        logger.warning(f"[{call_id}] Empty environment description received, using fallback")
        params.environment_desc = "A mysterious environment with hidden layers of intrigue and control"
    
    agent = ctx.context.get("agent_instance") or NewGameAgent()
    cal_params = CreateCalendarParams(environment_desc=params.environment_desc)
    
    # Prevent duplicate calls by checking if calendar already exists
    user_id = ctx.context["user_id"]
    conversation_id = ctx.context["conversation_id"]
    
    async with get_db_connection_context() as conn:
        existing = await conn.fetchrow("""
            SELECT value FROM CurrentRoleplay
            WHERE user_id=$1 AND conversation_id=$2 AND key='CalendarNames'
        """, user_id, conversation_id)
        
        if existing:
            logger.info(f"[{call_id}] Calendar already exists, returning cached data")
            try:
                cal_data = json.loads(existing["value"])
                return CalendarData(
                    days=cal_data.get("days", []),
                    months=cal_data.get("months", []),
                    seasons=cal_data.get("seasons", [])
                ).model_dump_json()
            except:
                pass
    
    logger.info(f"[{call_id}] Creating new calendar")
    result = await agent.create_calendar(ctx, cal_params)
    return result.model_dump_json()


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
) -> str:
    """Module‑level wrapper around NewGameAgent.generate_environment()."""
    agent = ctx.context.get("agent_instance") or NewGameAgent()
    result = await agent.generate_environment(ctx, params)
    return result.model_dump_json()

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
            model="gpt-5-nano"
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
            model="gpt-5-nano"
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
            model="gpt-5-nano"
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
            model="gpt-5-nano"
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

    async def _require_day_names(self, user_id: int, conversation_id: int,
                                 timeout: float = 15.0) -> List[str]:
        start = asyncio.get_running_loop().time()
        while True:
            async with get_db_connection_context() as conn:
                row = await conn.fetchrow("""
                    SELECT value
                    FROM   CurrentRoleplay
                    WHERE  user_id=$1 AND conversation_id=$2
                           AND key='CalendarNames'
                """, user_id, conversation_id)
                if row:
                    days = json.loads(row["value"]).get("days") or []
                    if days:
                        return days              # ✅ got them
            if asyncio.get_running_loop().time() - start > timeout:
                raise RuntimeError(
                    "Calendar day names not generated in time."
                )
            await asyncio.sleep(0.5)

    # --------------------------------------------------------------------- #
    #  main function                                                        #
    # --------------------------------------------------------------------- #
    
    # Also update the generate_environment method to better handle the agent tools
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
        Create the environment with better error handling and tool isolation.
        """
        user_id, conversation_id = ctx.context["user_id"], ctx.context["conversation_id"]
        
        # Build the prompt
        env_comp_text = "\n".join(params.env_components) if params.env_components else "No components provided"
        enh_feat_text = ", ".join(params.enhanced_features) if params.enhanced_features else "No enhanced features"
        stat_mod_text = (
            ", ".join(f"{sm.stat_name}: {sm.modifier_value}" for sm in params.stat_modifiers)
            if params.stat_modifiers else "No stat modifiers"
        )
    
        prompt = f"""
        Create a new daily-life sim environment with subtle, hidden layers of femdom and intrigue.
    
        Below is a merged environment concept:
        Mega Setting Name: {params.mega_name}
        Mega Description:
        {params.mega_desc}
    
        Using this as inspiration, create a VALID JSON object with these exact keys:
        1. "setting_name": A creative setting name (string)
        2. "environment_desc": A vivid environment description of 1-3 paragraphs (string)
        3. "environment_history": A brief history (string)
        4. "events": An array of event objects, each with: name, description, start_time, end_time, location, year, month, day, time_of_day
        5. "locations": An array of at least 10 location objects, each with: location_name, description, type, features (array), open_hours_json (JSON string)
        6. "scenario_name": A catchy scenario name (string)
        7. "quest_data": An object with quest_name and quest_description
    
        Reference details:
        Environment components: {env_comp_text}
        Enhanced features: {enh_feat_text}
        Stat modifiers: {stat_mod_text}
    
        IMPORTANT: 
        - Return ONLY valid JSON, no other text
        - Each location's open_hours_json must be a JSON STRING like "{{\\"Mon-Sun\\": \\"24/7\\"}}"
        - Do NOT call any tools or functions - just return the JSON
        """
    
        # Create a separate agent instance to avoid tool conflicts
        ctx.context["agent_instance"] = self
        
        # Use a modified agent that won't auto-call tools
        env_agent_isolated = Agent(
            name="EnvironmentCreatorIsolated",
            instructions="""
            You create game environments. When given a prompt, return ONLY valid JSON data.
            Do NOT use any tools or functions - just generate the requested JSON structure.
            """,
            tools=[],  # No tools to prevent unwanted calls
            model="gpt-5-nano",
            output_type=None,
        )
        
        # Run with retry logic
        max_retries = 3
        retry_count = 0
        raw_data = None
        
        while retry_count < max_retries:
            try:
                logger.info(f"Generating environment (attempt {retry_count + 1}/{max_retries})")
                raw_run = await Runner.run(env_agent_isolated, prompt, context=ctx.context)
                
                if not raw_run.final_output:
                    raise ValueError("Empty response from GPT")
                
                # Parse response
                if isinstance(raw_run.final_output, dict):
                    raw_data = raw_run.final_output
                else:
                    output_str = str(raw_run.final_output).strip()
                    if not output_str:
                        raise ValueError("Empty string response")
                    
                    # Find JSON in response
                    json_start = output_str.find('{')
                    json_end = output_str.rfind('}') + 1
                    
                    if json_start != -1 and json_end > json_start:
                        json_str = output_str[json_start:json_end]
                        raw_data = json.loads(json_str)
                    else:
                        raise ValueError("No JSON found in response")
                
                break
                
            except Exception as e:
                retry_count += 1
                logger.warning(f"Environment generation failed (attempt {retry_count}/{max_retries}): {e}")
                if retry_count >= max_retries:
                    logger.error("Borked")
                else:
                    await asyncio.sleep(1)
        
        # Fix open_hours_json format
        for loc in raw_data.get("locations", []):
            oh = loc.get("open_hours_json")
            if isinstance(oh, dict):
                loc["open_hours_json"] = json.dumps(oh)
            elif isinstance(oh, str):
                try:
                    json.loads(oh)
                except:
                    loc["open_hours_json"] = json.dumps({"hours": oh})
            else:
                loc["open_hours_json"] = json.dumps({"Mon-Sun": "09:00-17:00"})
        
        # Validate
        env_obj = EnvironmentData.model_validate(raw_data)
        
        # Create calendar after environment is ready
        cal_data = await self.create_calendar(
            ctx, CreateCalendarParams(environment_desc=env_obj.environment_desc)
        )
        
        # Persist calendar
        async with get_db_connection_context() as conn:
            await conn.execute("""
                INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                VALUES ($1,$2,'CalendarNames',$3)
                ON CONFLICT (user_id,conversation_id,key)
                DO UPDATE SET value = EXCLUDED.value
            """, user_id, conversation_id, json.dumps(cal_data))
        
        # Wait for calendar to be ready
        await self._require_day_names(user_id, conversation_id)
        
        # Store everything in database (single connection and transaction)
        async with get_db_connection_context() as conn, conn.transaction():
            cctx = RunContextWrapper(context={
                'user_id': user_id,
                'conversation_id': conversation_id
            })
            
            await canon.create_game_setting(
                cctx, conn,
                env_obj.setting_name,
                environment_desc=env_obj.environment_desc,
                environment_history=env_obj.environment_history,
                calendar_data=cal_data,
                scenario_name=env_obj.scenario_name,
            )
            
            for ev in env_obj.events:
                await canon.find_or_create_event(
                    cctx, conn,
                    ev.name, description=ev.description,
                    start_time=ev.start_time, end_time=ev.end_time,
                    location=ev.location, year=ev.year,
                    month=ev.month, day=ev.day, time_of_day=ev.time_of_day,
                )
                
            for loc in env_obj.locations:
                open_hours = json.loads(loc.open_hours_json)
                await canon.find_or_create_location(
                    cctx, conn,
                    loc.location_name,
                    description=loc.description,
                    location_type=loc.type,
                    notable_features=loc.features,
                    open_hours=open_hours,
                )
            
            qd = env_obj.quest_data
            await canon.find_or_create_quest(
                cctx, conn,
                qd.quest_name,
                progress_detail=qd.quest_description,
                status="In Progress",
            )
    
            # Synthesize and persist setting rules, capabilities, kind, reality context
            try:
                rules = await synthesize_setting_rules(env_obj.environment_desc, env_obj.setting_name)
    
                # Store capabilities + setting kind for fast lookup
                await canon.update_current_roleplay(
                    cctx, conn, "SettingCapabilities", json.dumps(rules.get("capabilities", {}))
                )
                await canon.update_current_roleplay(
                    cctx, conn, "SettingKind", rules.get("setting_kind", "modern_realistic")
                )
                await canon.update_current_roleplay(
                    cctx, conn, "RealityContext", rules.get("_reality_context","normal")
                )
    
                # Persist rules (scoped per conversation)
                for r in rules.get("hard_rules", []):
                    await conn.execute("""
                      INSERT INTO GameRules (user_id, conversation_id, rule_name, condition, effect)
                      VALUES ($1, $2, $3, $4, $5)
                      ON CONFLICT (user_id, conversation_id, rule_name)
                      DO UPDATE SET condition = EXCLUDED.condition, effect = EXCLUDED.effect
                    """, user_id, conversation_id, r.get("rule_name"), r.get("condition"), r.get("effect"))
                for r in rules.get("soft_rules", []):
                    await conn.execute("""
                      INSERT INTO GameRules (user_id, conversation_id, rule_name, condition, effect)
                      VALUES ($1, $2, $3, $4, $5)
                      ON CONFLICT (user_id, conversation_id, rule_name)
                      DO UPDATE SET condition = EXCLUDED.condition, effect = EXCLUDED.effect
                    """, user_id, conversation_id, r.get("rule_name"), r.get("condition"), r.get("effect"))
            except Exception as e:
                logger.warning(f"Setting rule synthesis failed: {e}")
        
        return env_obj
        
    
    async def process_new_game_with_preset(self, ctx, conversation_data: Dict[str, Any], preset_story_id: str) -> ProcessNewGameResult:
        """Process new game creation with a preset story (LLM environment generation)."""
        user_id = ctx.user_id
        conversation_id = None
        
        try:
            # Load the preset story
            async with get_db_connection_context() as conn:
                story_row = await conn.fetchrow(
                    "SELECT story_data FROM PresetStories WHERE story_id = $1",
                    preset_story_id
                )
            if not story_row:
                raise ValueError(f"Preset story {preset_story_id} not found")
            preset_story_data = json.loads(story_row['story_data'])
            
            # Create conversation
            async with get_db_connection_context() as conn:
                row = await conn.fetchrow("""
                    INSERT INTO conversations (user_id, conversation_name, status)
                    VALUES ($1, $2, 'processing')
                    RETURNING id
                """, user_id, f"New Game - {preset_story_data['name']}")
                conversation_id = row["id"]
            
            # Initialize player stats
            await insert_default_player_stats_chase(user_id, conversation_id)
            
            # Build context wrapper
            ctx_wrap = RunContextWrapper(context={
                'user_id': user_id,
                'conversation_id': conversation_id,
                'db_dsn': DB_DSN,
                'agent_instance': self
            })
            
            # Generate environment based on preset story
            env_params = GenerateEnvironmentParams(
                mega_name=preset_story_data['name'],
                mega_desc=preset_story_data['theme'] + "\n\n" + preset_story_data['synopsis'],
                env_components=[],  # Can be filled from story data
                enhanced_features=[],
                stat_modifiers=[]
            )
            env = await self.generate_environment(ctx_wrap, env_params)
    
            # Immediately synthesize and persist rules/capabilities for this conversation
            try:
                env_desc = preset_story_data['synopsis']
                setting_name = preset_story_data['name']
                rules = await synthesize_setting_rules(env_desc, setting_name)
                async with get_db_connection_context() as conn:
                    await canon.update_current_roleplay(
                        ctx_wrap, conn, "SettingCapabilities", json.dumps(rules.get("capabilities", {}))
                    )
                    await canon.update_current_roleplay(
                        ctx_wrap, conn, "SettingKind", rules.get("setting_kind", "modern_realistic")
                    )
                    await canon.update_current_roleplay(
                        ctx_wrap, conn, "RealityContext", rules.get("_reality_context","normal")
                    )
                    for r in rules.get("hard_rules", []):
                        await conn.execute("""
                          INSERT INTO GameRules (user_id, conversation_id, rule_name, condition, effect)
                          VALUES ($1, $2, $3, $4, $5)
                          ON CONFLICT (user_id, conversation_id, rule_name)
                          DO UPDATE SET condition = EXCLUDED.condition, effect = EXCLUDED.effect
                        """, user_id, conversation_id, r.get("rule_name"), r.get("condition"), r.get("effect"))
                    for r in rules.get("soft_rules", []):
                        await conn.execute("""
                          INSERT INTO GameRules (user_id, conversation_id, rule_name, condition, effect)
                          VALUES ($1, $2, $3, $4, $5)
                          ON CONFLICT (user_id, conversation_id, rule_name)
                          DO UPDATE SET condition = EXCLUDED.condition, effect = EXCLUDED.effect
                        """, user_id, conversation_id, r.get("rule_name"), r.get("condition"), r.get("effect"))
            except Exception as e:
                logger.warning(f"Preset rules synthesis failed: {e}")
            
            # Create required locations from preset
            async with get_db_connection_context() as conn:
                for location_data in preset_story_data.get('required_locations', []):
                    await canon.find_or_create_location(
                        ctx_wrap, conn,
                        location_data['name'],
                        description=location_data.get('description', ''),
                        location_type=location_data.get('type', 'building')
                    )
            
            # Create required NPCs from preset
            npc_handler = NPCCreationHandler()
            npc_ids = []
            for npc_data in preset_story_data.get('required_npcs', []):
                npc_id = await npc_handler.create_preset_npc(
                    ctx=ctx_wrap,
                    npc_data=npc_data,
                    environment_context=env.environment_desc
                )
                npc_ids.append(npc_id)
            
            # Initialize preset story tracking
            async with get_db_connection_context() as conn:
                await conn.execute("""
                    INSERT INTO PresetStoryProgress (
                        user_id, conversation_id, story_id, 
                        current_act, completed_beats, story_variables
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (user_id, conversation_id) 
                    DO UPDATE SET story_id = $3
                """, user_id, conversation_id, preset_story_id, 
                    1, json.dumps([]), json.dumps({}))
            
            # Create opening narrative
            opening = await self._create_preset_opening(ctx_wrap, preset_story_data)
            
            # Store opening message
            async with get_db_connection_context() as conn:
                await conn.execute("""
                    INSERT INTO messages (conversation_id, sender, content, created_at)
                    VALUES ($1, 'Nyx', $2, NOW())
                """, conversation_id, opening)
            
            # Finalize
            await self.finalize_game_setup(ctx_wrap, FinalizeGameSetupParams(
                opening_narrative=opening
            ))
            
            # Update conversation status
            async with get_db_connection_context() as conn:
                await conn.execute("""
                    UPDATE conversations 
                    SET status='ready', conversation_name=$3
                    WHERE id=$1 AND user_id=$2
                """, conversation_id, user_id, preset_story_data['name'])
            
            return ProcessNewGameResult(
                message=f"Started preset story: {preset_story_data['name']}",
                scenario_name=preset_story_data['name'],
                environment_name=env.setting_name,
                environment_desc=env.environment_desc,
                lore_summary="Preset story loaded",
                conversation_id=conversation_id,
                welcome_image_url=None,
                status="ready",
                opening_narrative=opening
            )
            
        except Exception as e:
            logger.error(f"Error in process_new_game_with_preset: {e}", exc_info=True)
            if conversation_id:
                async with get_db_connection_context() as conn:
                    await conn.execute("""
                        UPDATE conversations 
                        SET status='failed'
                        WHERE id=$1 AND user_id=$2
                    """, conversation_id, user_id)
            raise

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
        canon_ctx = RunContextWrapper(context={
            'user_id': user_id,
            'conversation_id': conversation_id
        })


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
    async def spawn_npcs(self, ctx: RunContextWrapper[GameContext], params: SpawnNPCsParams) -> List[int]:
        """
        Spawn multiple NPCs for the game world.
        """
        # Handle different context types
        if hasattr(ctx, 'context') and isinstance(ctx.context, dict):
            # RunContextWrapper style
            user_id = ctx.context["user_id"]
            conversation_id = ctx.context["conversation_id"]
        elif hasattr(ctx, 'user_id') and hasattr(ctx, 'conversation_id'):
            # Direct attribute style (CanonicalContext)
            user_id = ctx.user_id
            conversation_id = ctx.conversation_id
        else:
            raise ValueError("Invalid context object - missing user_id/conversation_id")
        
        # Create an instance of NPCCreationHandler
        npc_handler = NPCCreationHandler()
        
        # Create a proper context for the handler
        handler_ctx = RunContextWrapper(context={
            "user_id": user_id,
            "conversation_id": conversation_id,
            "agent_instance": self
        })
        
        # Use the class method directly with proper context
        npc_ids = await npc_handler.spawn_multiple_npcs(
            ctx=handler_ctx,  # Pass the proper context
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

    def _image_gen_available(self) -> bool:
        """
        Return True only when it makes sense to call `generate_roleplay_image_from_gpt`.
        Current rule: we must be able to import routes.ai_image_generator *and*
        its SINKIN_ACCESS_TOKEN must be non-empty.
        """
        try:
            from routes.ai_image_generator import SINKIN_ACCESS_TOKEN
            return bool(SINKIN_ACCESS_TOKEN)
        except Exception:
            return False

    @with_governance_permission(AgentType.UNIVERSAL_UPDATER, "create_chase_schedule")
    async def create_chase_schedule(self, ctx: RunContextWrapper[GameContext], params: CreateChaseScheduleParams) -> str:
        """
        Create a schedule for the player "Chase".
        """
        # Handle different context types
        if hasattr(ctx, 'context') and isinstance(ctx.context, dict):
            user_id = ctx.context["user_id"]
            conversation_id = ctx.context["conversation_id"]
        elif hasattr(ctx, 'user_id') and hasattr(ctx, 'conversation_id'):
            user_id = ctx.user_id
            conversation_id = ctx.conversation_id
        else:
            raise ValueError("Invalid context object")
        
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
        # Handle different context types
        if hasattr(ctx, 'context') and isinstance(ctx.context, dict):
            user_id = ctx.context["user_id"]
            conversation_id = ctx.context["conversation_id"]
        elif hasattr(ctx, 'user_id') and hasattr(ctx, 'conversation_id'):
            user_id = ctx.user_id
            conversation_id = ctx.conversation_id
        else:
            raise ValueError("Invalid context object")
        
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

    async def _create_player_schedule_data(
        self,
        ctx: RunContextWrapper[GameContext],
        environment_desc: str
    ) -> NPCScheduleData:
        """Create Chase's default schedule and persist it canonically."""

        if hasattr(ctx, 'context') and isinstance(ctx.context, dict):
            user_id = ctx.context["user_id"]
            conversation_id = ctx.context["conversation_id"]
        elif hasattr(ctx, 'user_id') and hasattr(ctx, 'conversation_id'):
            user_id = ctx.user_id
            conversation_id = ctx.conversation_id
        else:
            raise ValueError("Invalid context object - missing user_id/conversation_id")

        day_names = await self._require_day_names(user_id, conversation_id)

        chase_params = CreateChaseScheduleParams(
            environment_desc=environment_desc,
            day_names=day_names
        )

        chase_schedule_json = await self.create_chase_schedule(ctx, chase_params)
        chase_schedule = json.loads(chase_schedule_json)

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

        return NPCScheduleData(
            npc_ids=[],
            chase_schedule_json=chase_schedule_json
        )

    async def _queue_npc_pool_fill(
        self,
        user_id: int,
        conversation_id: int,
        target_count: int = 5
    ) -> None:
        """Queue a background task to ensure the NPC pool reaches the target size."""

        timestamp = datetime.utcnow().isoformat() + "Z"
        status_payload = {
            "status": "queued",
            "target": target_count,
            "queued_at": timestamp,
            "source": "new_game_bootstrap"
        }

        async with get_db_connection_context() as conn:
            await conn.execute(
                """
                INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                VALUES ($1, $2, 'NPCPoolStatus', $3)
                ON CONFLICT (user_id, conversation_id, key)
                DO UPDATE SET value = EXCLUDED.value
                """,
                user_id,
                conversation_id,
                json.dumps(status_payload)
            )

        try:
            from celery_config import celery_app

            celery_app.send_task(
                'tasks.ensure_npc_pool_task',
                args=[user_id, conversation_id, target_count],
                kwargs={'source': 'new_game_bootstrap'}
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error(
                "Failed to queue NPC pool fill for conversation %s: %s",
                conversation_id,
                exc,
                exc_info=True
            )

            failure_payload = {
                **status_payload,
                "status": "failed",
                "error": str(exc),
                "updated_at": datetime.utcnow().isoformat() + "Z"
            }

            async with get_db_connection_context() as conn:
                await conn.execute(
                    """
                    INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                    VALUES ($1, $2, 'NPCPoolStatus', $3)
                    ON CONFLICT (user_id, conversation_id, key)
                    DO UPDATE SET value = EXCLUDED.value
                    """,
                    user_id,
                    conversation_id,
                    json.dumps(failure_payload)
                )

            raise

    async def _queue_lore_generation(self, user_id: int, conversation_id: int) -> str:
        """Schedule background lore generation and record status."""

        placeholder = "Lore generation pending (background task queued)"
        timestamp = datetime.utcnow().isoformat() + "Z"
        status_payload = {
            "status": "queued",
            "queued_at": timestamp
        }

        try:
            async with get_db_connection_context() as conn:
                await conn.execute(
                    """
                    INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                    VALUES ($1, $2, 'LoreSummary', $3)
                    ON CONFLICT (user_id, conversation_id, key)
                    DO UPDATE SET value = EXCLUDED.value
                    """,
                    user_id,
                    conversation_id,
                    placeholder
                )

                await conn.execute(
                    """
                    INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                    VALUES ($1, $2, 'LoreGenerationStatus', $3)
                    ON CONFLICT (user_id, conversation_id, key)
                    DO UPDATE SET value = EXCLUDED.value
                    """,
                    user_id,
                    conversation_id,
                    json.dumps(status_payload)
                )
        except Exception as exc:
            logging.error(
                "Failed to record lore generation status for conversation %s: %s",
                conversation_id,
                exc,
                exc_info=True
            )
            return "Lore generation skipped - unable to record status"

        try:
            from celery_config import celery_app

            celery_app.send_task(
                'tasks.generate_lore_background_task',
                args=[user_id, conversation_id]
            )
        except Exception as exc:
            logging.error(
                "Failed to queue lore generation task for conversation %s: %s",
                conversation_id,
                exc,
                exc_info=True
            )

            failure_message = f"Lore generation failed to queue: {exc}"
            failure_payload = {
                **status_payload,
                "status": "failed",
                "failed_at": datetime.utcnow().isoformat() + "Z",
                "error": str(exc)
            }

            async with get_db_connection_context() as conn:
                await conn.execute(
                    """
                    INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                    VALUES ($1, $2, 'LoreSummary', $3)
                    ON CONFLICT (user_id, conversation_id, key)
                    DO UPDATE SET value = EXCLUDED.value
                    """,
                    user_id,
                    conversation_id,
                    failure_message
                )

                await conn.execute(
                    """
                    INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                    VALUES ($1, $2, 'LoreGenerationStatus', $3)
                    ON CONFLICT (user_id, conversation_id, key)
                    DO UPDATE SET value = EXCLUDED.value
                    """,
                    user_id,
                    conversation_id,
                    json.dumps(failure_payload)
                )

            return failure_message

        return placeholder

    async def _queue_conflict_generation(self, user_id: int, conversation_id: int) -> str:
        """Schedule background conflict generation if prerequisites are met."""

        timestamp = datetime.utcnow().isoformat() + "Z"

        async with get_db_connection_context() as conn:
            npc_count = await conn.fetchval(
                """
                SELECT COUNT(*) FROM NPCStats
                WHERE user_id = $1 AND conversation_id = $2
                """,
                user_id,
                conversation_id
            )

        npc_count = int(npc_count or 0)

        if npc_count < 3:
            summary = "No initial conflict - insufficient NPCs"
            status_payload = {
                "status": "skipped",
                "reason": "insufficient_npcs",
                "npc_count": npc_count,
                "updated_at": timestamp
            }

            async with get_db_connection_context() as conn:
                await conn.execute(
                    """
                    INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                    VALUES ($1, $2, 'InitialConflictSummary', $3)
                    ON CONFLICT (user_id, conversation_id, key)
                    DO UPDATE SET value = EXCLUDED.value
                    """,
                    user_id,
                    conversation_id,
                    summary
                )

                await conn.execute(
                    """
                    INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                    VALUES ($1, $2, 'InitialConflictStatus', $3)
                    ON CONFLICT (user_id, conversation_id, key)
                    DO UPDATE SET value = EXCLUDED.value
                    """,
                    user_id,
                    conversation_id,
                    json.dumps(status_payload)
                )

            return summary

        placeholder = "Initial conflict generation pending (background task queued)"
        status_payload = {
            "status": "queued",
            "queued_at": timestamp,
            "npc_count": npc_count
        }

        async with get_db_connection_context() as conn:
            await conn.execute(
                """
                INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                VALUES ($1, $2, 'InitialConflictSummary', $3)
                ON CONFLICT (user_id, conversation_id, key)
                DO UPDATE SET value = EXCLUDED.value
                """,
                user_id,
                conversation_id,
                placeholder
            )

            await conn.execute(
                """
                INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                VALUES ($1, $2, 'InitialConflictStatus', $3)
                ON CONFLICT (user_id, conversation_id, key)
                DO UPDATE SET value = EXCLUDED.value
                """,
                user_id,
                conversation_id,
                json.dumps(status_payload)
            )

        try:
            from celery_config import celery_app

            celery_app.send_task(
                'tasks.generate_initial_conflict_task',
                args=[user_id, conversation_id]
            )
        except Exception as exc:
            logging.error(
                "Failed to queue conflict generation task for conversation %s: %s",
                conversation_id,
                exc,
                exc_info=True
            )

            failure_message = f"No initial conflict - failed to queue background task: {exc}"
            failure_payload = {
                **status_payload,
                "status": "failed",
                "failed_at": datetime.utcnow().isoformat() + "Z",
                "error": str(exc)
            }

            async with get_db_connection_context() as conn:
                await conn.execute(
                    """
                    INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                    VALUES ($1, $2, 'InitialConflictSummary', $3)
                    ON CONFLICT (user_id, conversation_id, key)
                    DO UPDATE SET value = EXCLUDED.value
                    """,
                    user_id,
                    conversation_id,
                    failure_message
                )

                await conn.execute(
                    """
                    INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                    VALUES ($1, $2, 'InitialConflictStatus', $3)
                    ON CONFLICT (user_id, conversation_id, key)
                    DO UPDATE SET value = EXCLUDED.value
                    """,
                    user_id,
                    conversation_id,
                    json.dumps(failure_payload)
                )

            return failure_message

        return placeholder

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
        
        async with get_db_connection_context() as conn:
            canon_ctx = RunContextWrapper(context={
                'user_id': user_id,
                'conversation_id': conversation_id
            })
    
            await canon.create_opening_message(
                canon_ctx, conn,
                "Nyx",
                opening_narrative
            )
        
            # Also store in messages table for easy retrieval
            await conn.execute(
                """INSERT INTO messages (conversation_id, sender, content, created_at)
                   VALUES ($1, $2, $3, NOW())""",
                conversation_id, "Nyx", opening_narrative
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

            pool_row = await conn.fetchrow("""
                SELECT value FROM CurrentRoleplay
                WHERE user_id = $1 AND conversation_id = $2 AND key = 'NPCPoolStatus'
            """, user_id, conversation_id)

            pool_status = {}
            if pool_row and pool_row["value"]:
                try:
                    pool_status = json.loads(pool_row["value"])
                except json.JSONDecodeError:
                    logger.warning("Invalid NPCPoolStatus JSON for conversation %s", conversation_id)

            # Check locations
            location_count = await conn.fetchval("""
                SELECT COUNT(*) FROM Locations
                WHERE user_id = $1 AND conversation_id = $2
            """, user_id, conversation_id)

            # Check key roleplay data - loop through keys to avoid array type issues
            roleplay_keys = ['CurrentSetting', 'EnvironmentDesc', 'ChaseSchedule', 'LoreSummary', 'NPCPoolStatus']
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
            
            npc_ready = npc_count >= 5
            if not npc_ready:
                status = str(pool_status.get("status", "")).lower()
                target = int(pool_status.get("target", 0) or 0)
                if status in {"queued", "in_progress", "ready"} and target >= 5:
                    npc_ready = True

            logger.info(
                "Setup check - NPCs: %s (ready=%s), Locations: %s, Roleplay keys: %s/%s",
                npc_count,
                npc_ready,
                location_count,
                roleplay_count,
                len(roleplay_keys)
            )

            # Require populated NPC pool plan, 10 locations, and all key roleplay data
            return npc_ready and location_count >= 10 and roleplay_count >= len(roleplay_keys)

    async def process_preset_game_direct(self, ctx, conversation_data: Dict[str, Any], preset_story_id: str) -> ProcessNewGameResult:
        """
        Process preset game creation WITHOUT using LLM generation.
        All data comes directly from the preset story definition.
        """
        user_id = ctx.user_id
        conversation_id = None
        
        try:
            # Load the preset story
            async with get_db_connection_context() as conn:
                story_row = await conn.fetchrow(
                    "SELECT story_data FROM PresetStories WHERE story_id = $1",
                    preset_story_id
                )
                
                if not story_row:
                    raise ValueError(f"Preset story {preset_story_id} not found")
                
                preset_story_data = json.loads(story_row['story_data'])
            
            # Create or reuse conversation
            provided_convo_id = conversation_data.get("conversation_id")
            async with get_db_connection_context() as conn:
                if provided_convo_id:
                    row = await conn.fetchrow(
                        "SELECT id FROM conversations WHERE id=$1 AND user_id=$2",
                        provided_convo_id,
                        user_id,
                    )
                    if not row:
                        raise ValueError(
                            f"Conversation {provided_convo_id} not found or unauthorized"
                        )

                    conversation_id = row["id"]
                    await conn.execute(
                        """
                        UPDATE conversations
                           SET status='processing', conversation_name=$3
                         WHERE id=$1 AND user_id=$2
                        """,
                        conversation_id,
                        user_id,
                        preset_story_data['name'],
                    )

                    tables = [
                        "Events",
                        "PlannedEvents",
                        "PlayerInventory",
                        "Quests",
                        "NPCStats",
                        "Locations",
                        "SocialLinks",
                        "CurrentRoleplay",
                    ]
                    for t in tables:
                        await conn.execute(
                            f"DELETE FROM {t} WHERE user_id=$1 AND conversation_id=$2",
                            user_id,
                            conversation_id,
                        )
                else:
                    row = await conn.fetchrow(
                        """
                        INSERT INTO conversations (user_id, conversation_name, status)
                        VALUES ($1, $2, 'processing')
                        RETURNING id
                        """,
                        user_id,
                        f"{preset_story_data['name']}",
                    )
                    conversation_id = row["id"]

            conversation_data["conversation_id"] = conversation_id
            
            # Initialize player stats
            await insert_default_player_stats_chase(user_id, conversation_id)
            
            # Create context wrapper
            ctx_wrap = RunContextWrapper(context={
                'user_id': user_id,
                'conversation_id': conversation_id,
                'db_dsn': DB_DSN,
                'agent_instance': self
            })
            
            # 1. Set up environment directly (NO LLM)
            await self._setup_preset_environment(ctx_wrap, preset_story_data)

            try:
                env_desc = preset_story_data['synopsis']
                setting_name = preset_story_data['name']
                rules = await synthesize_setting_rules(env_desc, setting_name)
                async with get_db_connection_context() as conn:
                    await canon.update_current_roleplay(
                        ctx_wrap, conn, "SettingCapabilities", json.dumps(rules.get("capabilities", {}))
                    )
                    await canon.update_current_roleplay(
                        ctx_wrap, conn, "SettingKind", rules.get("setting_kind", "modern_realistic")
                    )
                    await canon.update_current_roleplay(
                        ctx_wrap, conn, "RealityContext", rules.get("_reality_context","normal")
                    )
                    for r in rules.get("hard_rules", []):
                        # INSERT with per-conversation scope
                        await conn.execute("""
                          INSERT INTO GameRules (user_id, conversation_id, rule_name, condition, effect)
                          VALUES ($1, $2, $3, $4, $5)
                          ON CONFLICT (user_id, conversation_id, rule_name)
                          DO UPDATE SET condition = EXCLUDED.condition, effect = EXCLUDED.effect
                        """, user_id, conversation_id, r.get("rule_name"), r.get("condition"), r.get("effect"))
                    for r in rules.get("soft_rules", []):
                        # INSERT with per-conversation scope
                        await conn.execute("""
                          INSERT INTO GameRules (user_id, conversation_id, rule_name, condition, effect)
                          VALUES ($1, $2, $3, $4, $5)
                          ON CONFLICT (user_id, conversation_id, rule_name)
                          DO UPDATE SET condition = EXCLUDED.condition, effect = EXCLUDED.effect
                        """, user_id, conversation_id, r.get("rule_name"), r.get("condition"), r.get("effect"))
            except Exception as e:
                logger.warning(f"Preset rules synthesis failed: {e}")
            
            # 2. Set up standard calendar (NO LLM)
            await self._setup_standard_calendar(ctx_wrap)
            
            # 3. Create all required locations directly
            location_ids = await self._create_preset_locations(ctx_wrap, preset_story_data)
            
            # 4. Create all required NPCs directly  
            npc_ids = await self._create_preset_npcs(ctx_wrap, preset_story_data)
            
            # 5. Initialize story-specific mechanics
            if preset_story_id == "the_moth_and_flame":
                from story_templates.moth.story_initializer import MothFlameStoryInitializer
                await MothFlameStoryInitializer._initialize_story_state(
                    ctx_wrap, user_id, conversation_id, npc_ids[0]  # Lilith is first
                )
                await MothFlameStoryInitializer._setup_special_mechanics(
                    ctx_wrap, user_id, conversation_id, npc_ids[0]
                )
            
            # 6. Create opening narrative (can still use story-specific generators)
            opening = await self._create_preset_opening(ctx_wrap, preset_story_data)
            
            # 7. Store opening message
#            async with get_db_connection_context() as conn:
#                await conn.execute("""
#                    INSERT INTO messages (conversation_id, sender, content, created_at)
#                    VALUES ($1, 'Nyx', $2, NOW())
#                """, conversation_id, opening)
          
            # 8. Mark as ready
            async with get_db_connection_context() as conn:
                await conn.execute("""
                    UPDATE conversations 
                    SET status='ready', conversation_name=$3
                    WHERE id=$1 AND user_id=$2
                """, conversation_id, user_id, preset_story_data['name'])
            
            return ProcessNewGameResult(
                message=f"Started preset story: {preset_story_data['name']}",
                scenario_name=preset_story_data['name'],
                environment_name=preset_story_data['name'],
                environment_desc=preset_story_data['synopsis'],
                lore_summary="Preset story loaded",
                conversation_id=conversation_id,
                welcome_image_url=None,
                status="ready",
                opening_narrative=opening
            )
            
        except Exception as e:
            logger.error(f"Error in process_preset_game_direct: {e}", exc_info=True)
            if conversation_id:
                async with get_db_connection_context() as conn:
                    await conn.execute("""
                        UPDATE conversations 
                        SET status='failed'
                        WHERE id=$1 AND user_id=$2
                    """, conversation_id, user_id)
            raise

    async def _setup_preset_environment(self, ctx: RunContextWrapper[GameContext], preset_data: Dict[str, Any]):
        """Set up environment directly from preset data without LLM"""
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        from lore.core import canon
        
        # Extract key data from preset
        setting_name = preset_data['name']
        environment_desc = preset_data['synopsis']
        environment_history = f"The world of {preset_data['name']} - {preset_data['theme']}"
        scenario_name = preset_data['name']
        
        # Store in database
        async with get_db_connection_context() as conn:
            # Setting info
            await conn.execute("""
                INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                VALUES ($1, $2, 'CurrentSetting', $3)
                ON CONFLICT (user_id, conversation_id, key)
                DO UPDATE SET value = EXCLUDED.value
            """, user_id, conversation_id, setting_name)
            
            await conn.execute("""
                INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                VALUES ($1, $2, 'EnvironmentDesc', $3)
                ON CONFLICT (user_id, conversation_id, key)
                DO UPDATE SET value = EXCLUDED.value
            """, user_id, conversation_id, environment_desc)
            
            await conn.execute("""
                INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                VALUES ($1, $2, 'EnvironmentHistory', $3)
                ON CONFLICT (user_id, conversation_id, key)
                DO UPDATE SET value = EXCLUDED.value
            """, user_id, conversation_id, environment_history)
            
            await conn.execute("""
                INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                VALUES ($1, $2, 'ScenarioName', $3)
                ON CONFLICT (user_id, conversation_id, key)
                DO UPDATE SET value = EXCLUDED.value
            """, user_id, conversation_id, scenario_name)
    
    async def _setup_standard_calendar(self, ctx: RunContextWrapper[GameContext]):
        """Set up a standard 12-month calendar without LLM"""
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        # Standard calendar
        calendar_data = {
            "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            "months": ["January", "February", "March", "April", "May", "June", 
                       "July", "August", "September", "October", "November", "December"],
            "seasons": ["Spring", "Summer", "Autumn", "Winter"]
        }
        
        # Store in database
        async with get_db_connection_context() as conn:
            await conn.execute("""
                INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                VALUES ($1, $2, 'CalendarNames', $3)
                ON CONFLICT (user_id, conversation_id, key)
                DO UPDATE SET value = EXCLUDED.value
            """, user_id, conversation_id, json.dumps(calendar_data))
        
        return calendar_data
    
    async def _create_preset_locations(self, ctx: RunContextWrapper[GameContext], preset_data: Dict[str, Any]) -> List[str]:
        """Create locations directly from preset data"""
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        location_ids = []
        
        from lore.core import canon
        
        async with get_db_connection_context() as conn:
            for loc_data in preset_data.get('required_locations', []):
                # Handle different data formats
                if isinstance(loc_data, dict):
                    location_name = loc_data.get('name', 'Unknown Location')
                    description = loc_data.get('description', '')
                    location_type = loc_data.get('type', 'general')
                    
                    # Extract additional data
                    areas = loc_data.get('areas', {})
                    schedule = loc_data.get('schedule', {})
                    atmosphere = loc_data.get('atmosphere', '')
                    
                    metadata = {
                        'location_type': location_type,
                        'areas': areas,
                        'schedule': schedule,
                        'atmosphere': atmosphere
                    }
                else:
                    # Handle simple string format
                    location_name = str(loc_data)
                    description = f"A location in {preset_data['name']}"
                    metadata = {}
                
                # Create location
                location_id = await canon.find_or_create_location(
                    ctx, conn,
                    location_name,
                    description=description,
                    metadata=metadata
                )
                location_ids.append(location_id)
        
        return location_ids
    
    async def _create_preset_npcs(self, ctx: RunContextWrapper[GameContext], preset_data: Dict[str, Any]) -> List[int]:
        """Create NPCs directly from preset data using PresetNPCHandler"""
        from npcs.preset_npc_handler import PresetNPCHandler
        
        npc_ids = []
        story_context = {
            'story_name': preset_data['name'],
            'story_id': preset_data['id'],
            'theme': preset_data['theme']
        }
        
        for npc_data in preset_data.get('required_npcs', []):
            try:
                # Use PresetNPCHandler for ALL NPCs including Lilith
                npc_id = await PresetNPCHandler.create_detailed_npc(
                    ctx=ctx,
                    npc_data=npc_data,
                    story_context=story_context
                )
                npc_ids.append(npc_id)
                        
            except Exception as e:
                logger.error(f"Error creating preset NPC {npc_data.get('name', 'Unknown')}: {e}", exc_info=True)
                # Continue with other NPCs even if one fails
        
        logger.info(f"Created {len(npc_ids)} preset NPCs")
        return npc_ids
        
    
    async def _create_preset_opening(self, ctx: RunContextWrapper[GameContext], preset_data: Dict[str, Any]) -> str:
        """Create dynamic opening narrative for preset story"""
        
        # Extract user_id and conversation_id from context
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        # For The Moth and Flame / Queen of Thorns
        if preset_data['id'] in ['the_moth_and_flame', 'queen_of_thorns']:
            try:
                # Try to import GPTService and related classes
                from story_templates.moth.story_initializer import GPTService
                from pydantic import BaseModel
                
                # Define a simple model for the atmosphere response
                class AtmosphereData(BaseModel):
                    introduction: str
                
                service = GPTService()
                
                # Get current time/season for atmosphere
                current_time = datetime.now()
                hour = current_time.hour
                time_period = "midnight" if 0 <= hour < 4 else "late evening" if 20 <= hour else "dusk"
                
                # Determine moon phase for gothic atmosphere
                moon_phases = ["new moon", "waxing crescent", "first quarter", "waxing gibbous", 
                              "full moon", "waning gibbous", "last quarter", "waning crescent"]
                current_moon = moon_phases[current_time.day % 8]
                
                system_prompt = """You are Nyx, the omniscient narrator of dark stories. Your voice is:
    - Sultry and seductive, dripping with dark honey
    - Ominous and threatening, like silk hiding razor wire  
    - Intimate and knowing, as if you've been watching the player forever
    - Poetic and metaphorical, using imagery of moths, flames, thorns, and transformation
    
    You're introducing a player named Chase to the world of the Queen of Thorns - a story of:
    - A mysterious dominatrix who rules San Francisco's underground
    - The shadow network that transforms predators and saves victims
    - Masks that hide broken souls
    - Power dynamics that blur consent and control
    - The inability to speak three simple words: "I love you"
    
    Reference these gothic poems' imagery:
    - Moths drawn helplessly to flames
    - Porcelain masks hiding rough geographies of breaks
    - Roses with thorns that draw blood
    - Binary stars locked in gravitational pull
    - The taste of burning stars on tongues
    
    NEVER reveal the Queen's true identity or that she leads the network.
    Make the player feel like prey that thinks it's the predator."""
    
                user_prompt = f"""Create an immersive opening for Chase entering this world.
    Setting: San Francisco, {time_period}, {current_moon}
    Starting location: Outside the Velvet Sanctum in SoMa
    Atmosphere: Fog rolling in from the bay, neon bleeding into shadows
    
    Include:
    1. Direct address to Chase as "you" - make it personal and invasive
    2. Sensory details that feel like caresses and threats
    3. Hint at the Queen without naming her - "she who holds court below"
    4. Foreshadow the network - "roses grow in unexpected places"
    5. Reference transformation - "predators learning to kneel"
    6. End with an invitation that feels like a trap closing
    
    Write 3-4 atmospheric paragraphs that make Chase feel they're already caught in a web.
    Make it sultry, ominous, and unforgettable.
    
    Return your response as JSON with an "introduction" field containing the narrative text."""
    
                try:
                    # Use the service to generate the opening
                    result = await service.call_with_validation(
                        model="gpt-5-nano",  # Use a default model
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        response_model=AtmosphereData,
                        temperature=0.85  # Higher for more creative variation
                    )
                    
                    opening = result.introduction
                    
                    # Add a personalized touch based on current state
                    if current_moon == "full moon":
                        opening += "\n\nTonight, the full moon watches. Even the Queen's mask cannot hide what hunger becomes under such light."
                    elif current_moon == "new moon":  
                        opening += "\n\nTonight, the new moon offers perfect darkness. In the absence of light, all masks become meaningless."
                    
                    # Store as the opening message
                    async with get_db_connection_context() as conn:
                        await conn.execute("""
                            INSERT INTO messages (conversation_id, sender, content, created_at)
                            VALUES ($1, 'Nyx', $2, NOW())
                        """, conversation_id, opening)  # Use conversation_id from context
                    
                    return opening
                    
                except Exception as e:
                    logger.error(f"Error calling GPTService: {e}")
                    # Fall through to static version
                    
            except ImportError as e:
                logger.warning(f"GPTService or dependencies not available: {e}")
                # Fall through to static version
            except Exception as e:
                logger.error(f"Error generating dynamic opening: {e}")
                # Fall through to static version
        
        # Enhanced static fallback that's still better than the current one
        fallback_openings = [
            """The city breathes differently after midnight, Chase. I should know—I've been watching you navigate these streets, thinking you understand the shadows. But you don't. Not yet.
    
    Below your feet, in a place where neon bleeds into darkness and desire takes corporeal form, she waits. They call her space the Velvet Sanctum, though 'sanctuary' is perhaps too kind a word for what happens there. She holds court—beautiful, terrible, offering transcendence through submission to those brave enough to kneel.
    
    You think you're here by choice, following whispers of a Queen who cannot speak love but makes grown CEOs weep with need. You think you're the moth, conscious of the flame. But darling, in my story, sometimes the flame hunts the moth. Sometimes the rose grows thorns specifically for your blood.
    
    The door below doesn't have a sign. It doesn't need one. It's already opening for you—it always was. After all, I've been preparing you for this moment far longer than you realize. Welcome to your beautiful destruction.""",
    
            """Listen closely, Chase. That sound beneath the city's pulse? That's power changing hands in rooms you're about to enter. That taste in the air, like copper and roses? That's what transformation smells like when it's still fresh.
    
    She's down there now, perhaps adjusting a porcelain mask that hides everything and nothing, perhaps tracing names in red ink—those who disappointed, those who fled, those who learned to worship properly. Which list will you join? The choice was never really yours.
    
    They say the network has eyes everywhere, that roses grow in the strangest places, that predators come here to be devoured and reborn. They're right about all of it, and yet they know nothing. The real truth is written in the space between heartbeats, in the moment when submission becomes salvation.
    
    The Velvet Sanctum's door is unmarked, but you'll find it. Moths always find their flames. And when you descend those stairs, when you first glimpse her throne, remember: I brought you here. I've been guiding you all along. Now dance for me, pretty moth. Dance until your wings catch fire.""",
    
            """Oh, Chase. Sweet, oblivious Chase. You stand at the threshold thinking you're about to enter a BDSM club, maybe meet a mysterious dominatrix, perhaps explore some dark desires. If only it were that simple.
    
    What waits below is so much more. She—and I won't tell you her name, you'll hear it gasped in reverence soon enough—is architect of agonies and ecstasies you can't imagine. Behind her masks (and she has so many) lies a woman who saves the broken by breaking the proud. The network she tends grows through the city like morning glory through a corpse.
    
    You'll think you see her vulnerability. You'll think those moments when her mask slips are real. Some of them are. Some of them aren't. The tragedy is you'll never know which, and the not knowing will consume you like flame consumes paper—slowly, inevitably, completely.
    
    But here's my gift to you, my newest plaything: you're already hers. You became hers the moment you heard her name. Now all that remains is the delicious descent. The door is opening. Can you hear it? That's the sound of your old life ending. Welcome to the garden where every rose draws blood."""
        ]
        
        import random
        selected = random.choice(fallback_openings)
        
        # For other stories, keep simple
        if preset_data['id'] not in ['the_moth_and_flame', 'queen_of_thorns']:
            return f"Welcome to {preset_data['name']}. {preset_data['synopsis']}\n\nYour story begins..."
        
        return selected
    
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

        canon_ctx = RunContextWrapper(context={
            'user_id': user_id,
            'conversation_id': conversation_id
        })

        logging.info(
            "Queueing background lore generation for user_id=%s conversation_id=%s",
            user_id,
            conversation_id,
        )
        lore_summary = await self._queue_lore_generation(user_id, conversation_id)

        logging.info(
            "Queueing background conflict generation for user_id=%s conversation_id=%s",
            user_id,
            conversation_id,
        )
        conflict_name = await self._queue_conflict_generation(user_id, conversation_id)

        # Generate currency system canonically
        currency_name = "Standard currency"  # Default value
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
            
        # Try to generate welcome image, but don't fail if it's not available
        welcome_image_url = None
        try:
            if self._image_gen_available():
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
                if image_result and "image_urls" in image_result and image_result["image_urls"]:
                    welcome_image_url = image_result["image_urls"][0]
                    
                    # Store the image URL canonically
                    async with get_db_connection_context() as conn:
                        await canon.update_current_roleplay(
                            canon_ctx, conn,
                            'WelcomeImageUrl', welcome_image_url
                        )
                    logging.info("Welcome image generated successfully")
            else:
                logging.info("Image generation skipped - no API key configured")
                
        except Exception as e:
            # Log the error but don't fail the entire setup
            logging.warning(f"Failed to generate welcome image: {e}")
            logging.info("Continuing without welcome image")

        # Establish initial player context (location and time)
        try:
            async with get_db_connection_context() as conn:
                rows = await conn.fetch(
                    """
                    SELECT location_name FROM Locations
                    WHERE user_id=$1 AND conversation_id=$2
                    """,
                    user_id,
                    conversation_id,
                )
                if rows:
                    start_location = random.choice([row["location_name"] for row in rows])
                else:
                    start_location = "Unknown"
                await canon.update_current_roleplay(
                    canon_ctx, conn, "CurrentLocation", start_location
                )

            calendar_names = await load_calendar_names(user_id, conversation_id)
            year = random.randint(1, 100)
            month_idx = random.randint(1, len(calendar_names.get("months", [])) or 1)
            day_num = random.randint(1, 30)
            phase = random.choice(TIME_PHASES)
            month_name = calendar_names.get("months", ["Month"])[month_idx - 1]
            day_name = calendar_names.get("days", ["Day"])[(day_num - 1) % len(calendar_names.get("days", ["Day"]))]

            await set_current_time(user_id, conversation_id, year, month_idx, day_num, phase)

            async with get_db_connection_context() as conn:
                await canon.update_current_roleplay(
                    canon_ctx,
                    conn,
                    "CurrentTime",
                    f"Year {year} {month_name} {day_name} {phase}",
                )
        except Exception as e:
            logging.warning(f"Failed to initialize player context: {e}")

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
            lore_system = await DynamicLoreGenerator.get_instance(user_id, conversation_id)
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
        # Check if this is a preset story request
        preset_story_id = conversation_data.get("preset_story_id")
        if preset_story_id:
            logger.info(f"Detected preset story request: {preset_story_id}")
            return await self.process_preset_game_direct(ctx, conversation_data, preset_story_id)
        
        user_id = ctx.user_id  # Direct attribute access
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
            
            # Note: WorldDirector initialization moved to after game setup is complete
            
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
            ctx_wrap = RunContextWrapper(context={
                'user_id': user_id,
                'conversation_id': conversation_id,
                'db_dsn': DB_DSN,
                'agent_instance': self
            })
            
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

            # 2. Prepare player schedule and queue NPC creation
            logger.info("Step 2: Preparing player schedule and queueing NPC generation in the background...")
            schedule_context_desc = env.environment_desc + "\n\n" + env.environment_history
            npc_sched = await self._create_player_schedule_data(ctx_wrap, schedule_context_desc)
            await self._queue_npc_pool_fill(user_id, conversation_id, target_count=5)
            logger.info("Background NPC generation task dispatched")

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
            
            async with get_db_connection_context() as conn:
                # First verify the opening narrative exists
                msg_check = await conn.fetchval("""
                    SELECT COUNT(*) FROM messages 
                    WHERE conversation_id=$1 AND sender='Nyx'
                """, conversation_id)
                
                if msg_check == 0:
                    # If no message exists, store the opening narrative now
                    await conn.execute("""
                        INSERT INTO messages (conversation_id, sender, content, created_at)
                        VALUES ($1, 'Nyx', $2, NOW())
                    """, conversation_id, opening)
                
                await conn.execute("""
                    UPDATE conversations 
                    SET status='ready', 
                        conversation_name=$3
                    WHERE id=$1 AND user_id=$2
                """, conversation_id, user_id, f"Game: {env.setting_name}")
            
            # Initialize WorldDirector AFTER game setup is complete
            logger.info("Initializing WorldDirector")
            try:
                from story_agent.world_director_agent import CompleteWorldDirector
                world_director = CompleteWorldDirector(user_id, conversation_id)
                await world_director.initialize()
                # Register it with the existing governance
                await governance.register_agent(
                    agent_type=AgentType.WORLD_DIRECTOR,
                    agent_instance=world_director,
                    agent_id="world_director"
                )
                logger.info(f"WorldDirector initialized and registered for {conversation_id}")
            except Exception as e:
                logger.error(f"WorldDirector init failed: {e}", exc_info=True)
                # Don't fail the entire game creation if WorldDirector fails
            
            # NOW start background directive processing after everything is set up
            if self.directive_handler:
                await self.directive_handler.start_background_processing()
                logger.info("Started background directive processing")
            
            logger.info(f"New game creation completed for conversation {conversation_id}")
                        
            opening_narrative_text = opening  # Store the actual text
            
            # Then in the return statement at the end:
            return ProcessNewGameResult(
                message=f"New game started. environment={env.setting_name}, conversation_id={conversation_id}",
                scenario_name=env.scenario_name,
                environment_name=env.setting_name,
                environment_desc=env.environment_desc,
                lore_summary=final.lore_summary,
                conversation_id=conversation_id,
                welcome_image_url=final.welcome_image_url,
                status="ready",
                opening_narrative=opening_narrative_text  # Add this
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
