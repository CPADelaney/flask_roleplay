# new_game_agent.py

import logging
import json
import asyncio
import os
import functools
from datetime import datetime
from typing import Dict, Any, Optional, List

from agents import Agent, Runner, function_tool, GuardrailFunctionOutput, InputGuardrail, RunContextWrapper
from pydantic import BaseModel, Field

from memory.wrapper import MemorySystem
from logic.stats_logic import insert_default_player_stats_chase

# Import your existing modules
from logic.calendar import update_calendar_names
from logic.aggregator_sdk import get_aggregated_roleplay_context
from npcs.new_npc_creation import NPCCreationHandler
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

NEW_GAME_AGENT_NYX_TYPE = AgentType.UNIVERSAL_UPDATER # Or AgentType.UNIVERSAL_UPDATER.value if methods expect string
NEW_GAME_AGENT_NYX_ID = "new_game_director_agent" 

# Output models for structured data
class EnvironmentData(BaseModel):
    setting_name: str
    environment_desc: str
    environment_history: str
    events: list = Field(default_factory=list)
    locations: list = Field(default_factory=list)
    scenario_name: str
    quest_data: dict = Field(default_factory=dict)

class NPCData(BaseModel):
    npc_name: str
    introduced: bool = False
    archetypes: list = Field(default_factory=list)
    physical_description: str = ""
    hobbies: list = Field(default_factory=list)
    personality_traits: list = Field(default_factory=list)
    likes: list = Field(default_factory=list)
    dislikes: list = Field(default_factory=list)
    schedule: dict = Field(default_factory=dict)

class GameContext(BaseModel):
    user_id: int
    conversation_id: int
    db_dsn: str = DB_DSN

# Pydantic models for function parameters (excluding ctx)
class CalendarToolParams(BaseModel):
    environment_desc: str
    setting_name: Optional[str] = None
    environment_data: Optional[Dict[str, Any]] = None

class CreateCalendarParams(BaseModel):
    environment_desc: str

class GenerateEnvironmentParams(BaseModel):
    mega_name: str
    mega_desc: str
    env_components: Optional[List[str]] = None
    enhanced_features: Optional[List[str]] = None
    stat_modifiers: Optional[Dict[str, Any]] = None

class SpawnNPCsParams(BaseModel):
    environment_desc: str
    day_names: List[str]
    count: int = 5

class CreateChaseScheduleParams(BaseModel):
    environment_desc: str
    day_names: List[str]

class CreateNPCsAndSchedulesParams(BaseModel):
    environment_data: Dict[str, Any]

class CreateOpeningNarrativeParams(BaseModel):
    environment_data: Dict[str, Any]
    npc_schedule_data: Dict[str, Any]

class FinalizeGameSetupParams(BaseModel):
    opening_narrative: str

class GenerateLoreParams(BaseModel):
    environment_desc: str

class NewGameAgent:
    """Agent for handling new game creation process with Nyx governance integration"""
    
    def __init__(self):
        # Create wrapper functions for sub-agent tools
        @function_tool
        async def _calendar_tool(ctx: RunContextWrapper[Any], params: CalendarToolParams):
            """Create calendar for the game world"""
            calendar_params = CreateCalendarParams(environment_desc=params.environment_desc)
            return await self.create_calendar(ctx, calendar_params)
        
        @function_tool
        async def _spawn_npcs_tool(ctx: RunContextWrapper[Any], count: int = 5):
            """Spawn NPCs for the game world"""
            # Note: environment_desc and day_names are not used by spawn_npcs, 
            # so we create minimal params
            params = SpawnNPCsParams(
                environment_desc="",  # Not used internally
                day_names=[],  # Not used internally  
                count=count
            )
            return await self.spawn_npcs(ctx, params)
        
        @function_tool
        async def _create_chase_schedule_tool(ctx: RunContextWrapper[Any], environment_desc: str, day_names: List[str]):
            """Create Chase's schedule"""
            params = CreateChaseScheduleParams(environment_desc=environment_desc, day_names=day_names)
            return await self.create_chase_schedule(ctx, params)
        
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
            5. "locations" (array of objects with location_name, description, open_hours)
            6. "scenario_name" (string; a catchy title)
            7. "quest_data" (object with quest_name and quest_description)
            
            Focus on creating a mundane yet charming atmosphere with subtle undertones of control and influence.
            """,
            output_type=EnvironmentData,
            tools=[
                _calendar_tool  # Use wrapper instead of direct method
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
                _spawn_npcs_tool,  # Use wrapper instead of direct method
                _create_chase_schedule_tool  # Use wrapper instead of direct method
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
        
        # Start background processing of directives
        await self.directive_handler.start_background_processing()

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
                    'conversation_id': self.directive_handler.conversation_id
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
    async def create_calendar(self, ctx: RunContextWrapper[GameContext], params: CreateCalendarParams):
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
        action_description="Generated game environment for new game"
    )
    async def generate_environment(self, ctx: RunContextWrapper[GameContext], params: GenerateEnvironmentParams):
        """
        Generate the game environment data using canonical functions.
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        # Import canon functions
        from lore.core import canon
        
        env_comp_text = "\n".join(params.env_components) if params.env_components else "No components provided"
        enh_feat_text = ", ".join(params.enhanced_features) if params.enhanced_features else "No enhanced features"
        stat_mod_text = ", ".join([f"{k}: {v}" for k, v in params.stat_modifiers.items()]) if params.stat_modifiers else "No stat modifiers"
        
        # Create prompt for the environment agent
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
        - At least 10 distinct locations
        - A scenario name
        - Initial quest data
    
        Reference these details:
        Environment components: {env_comp_text}
        Enhanced features: {enh_feat_text}
        Stat modifiers: {stat_mod_text}
        """
        
        # Run the environment agent
        result = await Runner.run(
            self.environment_agent,
            prompt,
            context=ctx.context
        )
        
        # Get calendar data - need to use params object
        calendar_params = CreateCalendarParams(environment_desc=result.final_output.environment_desc)
        calendar_data = await self.create_calendar(ctx, calendar_params)
        
        # Store environment data canonically
        async with get_db_connection_context() as conn:
            async with conn.transaction():
                # Create context object for canon functions
                canon_ctx = type('obj', (object,), {
                    'user_id': user_id, 
                    'conversation_id': conversation_id
                })
                
                # Store game setting
                await canon.create_game_setting(
                    canon_ctx, conn, 
                    result.final_output.setting_name,
                    environment_desc=result.final_output.environment_desc,
                    environment_history=result.final_output.environment_history,
                    calendar_data=calendar_data,
                    scenario_name=result.final_output.scenario_name
                )
                
                # Create events canonically
                for event in result.final_output.events:
                    # Handle both lowercase and capitalized keys
                    event_name = event.get("name") or event.get("Name") or "Unnamed Event"
                    event_desc = event.get("description") or event.get("Description") or ""
                    event_start = event.get("start_time") or event.get("Start_time") or "TBD"
                    event_end = event.get("end_time") or event.get("End_time") or "TBD"
                    event_location = event.get("location") or event.get("Location") or "Unknown"
                    event_year = event.get("year") or event.get("Year") or 1
                    event_month = event.get("month") or event.get("Month") or 1
                    event_day = event.get("day") or event.get("Day") or 1
                    event_time = event.get("time_of_day") or event.get("Time_of_day") or "Morning"
                    
                    await canon.find_or_create_event(
                        canon_ctx, conn,
                        event_name,
                        description=event_desc,
                        start_time=event_start,
                        end_time=event_end,
                        location=event_location,
                        year=event_year,
                        month=event_month,
                        day=event_day,
                        time_of_day=event_time
                    )
                
                # Create locations canonically
                for location in result.final_output.locations:
                    # Handle both lowercase and capitalized keys
                    loc_name = location.get("location_name") or location.get("Location_name") or "Unnamed"
                    loc_desc = location.get("description") or location.get("Description") or ""
                    loc_type = location.get("type") or location.get("Type") or "settlement"
                    loc_features = location.get("features") or location.get("Features") or []
                    loc_hours = location.get("open_hours") or location.get("Open_hours") or {}
                    
                    await canon.find_or_create_location(
                        canon_ctx, conn,
                        loc_name,
                        description=loc_desc,
                        location_type=loc_type,
                        notable_features=loc_features,
                        open_hours=loc_hours
                    )
                
                # Create main quest canonically
                quest_data = result.final_output.quest_data
                quest_name = quest_data.get("quest_name") or quest_data.get("Quest_name") or "Unnamed Quest"
                quest_desc = quest_data.get("quest_description") or quest_data.get("Quest_description") or "Quest summary"
                
                await canon.find_or_create_quest(
                    canon_ctx, conn,
                    quest_name,
                    progress_detail=quest_desc,
                    status="In Progress"
                )
        
        return result.final_output

    @with_governance_permission(AgentType.UNIVERSAL_UPDATER, "spawn_npcs")
    async def spawn_npcs(self, ctx: RunContextWrapper[GameContext], params: SpawnNPCsParams):
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
        
        # Use the class method directly
        npc_ids = await npc_handler.spawn_multiple_npcs(
            ctx=ctx,
            count=params.count
        )
        
        return npc_ids

    @with_governance_permission(AgentType.UNIVERSAL_UPDATER, "create_chase_schedule")
    async def create_chase_schedule(self, ctx: RunContextWrapper[GameContext], params: CreateChaseScheduleParams):
        """
        Create a schedule for the player "Chase".
        
        Args:
            params: CreateChaseScheduleParams with environment_desc and day_names
            
        Returns:
            Dictionary with Chase's schedule
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
        
        return default_schedule

    @with_governance(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        action_type="create_npcs_and_schedules",
        action_description="Created NPCs and schedules for new game"
    )
    async def create_npcs_and_schedules(self, ctx: RunContextWrapper[GameContext], params: CreateNPCsAndSchedulesParams):
        """
        Create NPCs and schedules for the game world using canonical functions.
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        from lore.core import canon
        
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
        environment_desc = params.environment_data.get("environment_desc", "") + "\n\n" + params.environment_data.get("environment_history", "")
        
        # Create params for spawn_npcs
        spawn_params = SpawnNPCsParams(
            environment_desc=environment_desc,
            day_names=day_names,
            count=5
        )
        npc_ids = await self.spawn_npcs(ctx, spawn_params)
        
        # Create Chase's schedule
        chase_params = CreateChaseScheduleParams(
            environment_desc=environment_desc,
            day_names=day_names
        )
        chase_schedule = await self.create_chase_schedule(ctx, chase_params)
        
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
        
        return {
            "npc_ids": npc_ids,
            "chase_schedule": chase_schedule
        }

    @with_governance(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        action_type="create_opening_narrative",
        action_description="Created opening narrative for new game"
    )
    async def create_opening_narrative(self, ctx: RunContextWrapper[GameContext], params: CreateOpeningNarrativeParams):
        """
        Create the opening narrative for the game using canonical functions.
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        from lore.core import canon
        from routes.story_routes import build_aggregator_text 
        
        # Get aggregator data
        aggregator_data = get_aggregated_roleplay_context(user_id, conversation_id, "Chase")
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

    @with_governance(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        action_type="finalize_game_setup",
        action_description="Finalized game setup including lore, conflict, currency and image"
    )
    async def finalize_game_setup(self, ctx: RunContextWrapper[GameContext], params: FinalizeGameSetupParams):
        """
        Finalize game setup including lore generation, conflict generation and image generation.
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        from lore.core import canon
        
        # Mark conversation as ready
        async with get_db_connection_context() as conn:
            canon_ctx = type('obj', (object,), {
                'user_id': user_id, 
                'conversation_id': conversation_id
            })
            
            # Get the environment description for lore generation
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
            
            await conn.execute("""
                UPDATE conversations
                SET status='ready'
                WHERE id=$1 AND user_id=$2
            """, conversation_id, user_id)
            
        # Initialize and generate lore
        try:
            # Get the lore system instance and initialize it
            # Note: LoreSystem is now properly initialized through governance,
            # so this should not cause circular dependency issues
            lore_system = DynamicLoreGenerator.get_instance(user_id, conversation_id)
            await lore_system.initialize()
            
            # Generate comprehensive lore based on the environment
            logging.info(f"Generating lore for new game (user_id={user_id}, conversation_id={conversation_id})")
            lore_result = await lore_system.generate_world_lore(environment_desc)
            
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
        scene_data = {
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
        }
        
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
        
        return {
            "status": "ready",
            "welcome_image_url": welcome_image_url,
            "lore_summary": lore_summary,
            "initial_conflict": conflict_name,
            "currency_system": currency_name
        }

    async def _get_setting_name(self, ctx: RunContextWrapper[GameContext]):
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
    async def generate_lore(self, ctx: RunContextWrapper[GameContext], params: GenerateLoreParams) -> Dict[str, Any]:
        """
        Generate comprehensive lore for the game environment.
        
        Args:
            params: GenerateLoreParams with environment_desc
            
        Returns:
            Dictionary with generated lore data
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        try:
            # Get the lore system instance and initialize it
            # Note: LoreSystem is now properly initialized through governance,
            # so this should not cause circular dependency issues
            lore_system = DynamicLoreGenerator.get_instance(user_id, conversation_id)
            await lore_system.initialize()
            
            # Generate comprehensive lore based on the environment
            logging.info(f"Generating lore for environment: {params.environment_desc[:100]}...")
            lore_result = await lore_system.generate_world_lore(params.environment_desc)
            
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
            
            return {
                "lore_summary": lore_summary,
                "factions_count": factions_count,
                "cultural_elements_count": cultural_count,
                "locations_count": locations_count,
                "historical_events_count": events_count
            }
            
        except Exception as e:
            logging.error(f"Error generating lore: {e}", exc_info=True)
            return {
                "error": f"Failed to generate lore: {str(e)}",
                "lore_summary": "Failed to generate lore"
            }

    @with_governance(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        action_type="process_new_game",
        action_description="Processed complete new game creation workflow"
    )
    async def process_new_game(self, user_id, conversation_data):
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
        
            # Initialize StoryDirector asynchronously
            logger.info("Scheduling StoryDirector initialization")
            async def _init_story_director():
                try:
                    # Import here to avoid circular import
                    from story_agent.story_director_agent import initialize_story_director, register_with_governance as register_sd_with_gov
                    
                    await initialize_story_director(user_id, conversation_id)
                    await register_sd_with_gov(user_id, conversation_id)
                    logger.info(f"StoryDirector initialized for {conversation_id}")
                except Exception as e:
                    logger.error(f"StoryDirector init failed: {e}", exc_info=True)
                    
            asyncio.create_task(_init_story_director())
        
            # Initialize directive handler and register with governance
            await self.initialize_directive_handler(user_id, conversation_id)
            
            # Get governance (which will properly initialize LoreSystem internally)
            governance = await get_central_governance(user_id, conversation_id)
            
            # Register this agent with governance
            await governance.register_agent(
                agent_type=NEW_GAME_AGENT_NYX_TYPE,
                agent_instance=self,
                agent_id=NEW_GAME_AGENT_NYX_ID
            )
            
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
            enh_feats = mega_data.get("enhanced_features", [])
            stat_mods = mega_data.get("stat_modifiers", {})
            mega_name = mega_data.get("mega_name", "Untitled Mega Setting")
            mega_desc = mega_data.get("mega_description", "No environment generated")
            
            # Set up context for the agent
            context = GameContext(
                user_id=user_id,
                conversation_id=conversation_id,
                db_dsn=DB_DSN
            )
            
            # Update status - running agent
            async with get_db_connection_context() as conn:
                await conn.execute("""
                    UPDATE conversations 
                    SET conversation_name='New Game - Building World'
                    WHERE id=$1 AND user_id=$2
                """, conversation_id, user_id)
            
            # Run the game creation process
            prompt = f"""
            Create a new game world with the following components:
            
            Mega Setting Name: {mega_name}
            Mega Description: {mega_desc}
            
            Environment Components: {env_comps}
            Enhanced Features: {enh_feats}
            Stat Modifiers: {stat_mods}
            
            Create a complete game world with environment, NPCs, and an opening narrative.
            """
            
            result = await Runner.run(
                self.agent,
                prompt,
                context=context.dict()
            )
            
            # Extract data from result
            final_output = result.final_output if hasattr(result, 'final_output') else {}
            setting_name = final_output.get('setting_name') or final_output.get('Setting_name') or 'Unknown Setting'
            
            # Update conversation with final name and ready status
            async with get_db_connection_context() as conn:
                await conn.execute("""
                    UPDATE conversations 
                    SET status='ready', 
                        conversation_name=$3
                    WHERE id=$1 AND user_id=$2
                """, conversation_id, user_id, f"Game: {setting_name}")
            
            logger.info(f"New game creation completed for conversation {conversation_id}")
            
            # Return success message and data
            return {
                "message": f"New game started. environment={setting_name}, conversation_id={conversation_id}",
                "scenario_name": final_output.get('scenario_name') or final_output.get('Scenario_name') or 'New Game',
                "environment_name": setting_name,
                "environment_desc": final_output.get('environment_desc') or final_output.get('Environment_desc') or '',
                "lore_summary": final_output.get('lore_summary') or final_output.get('Lore_summary') or 'Standard lore generated',
                "conversation_id": conversation_id,
                "welcome_image_url": final_output.get('welcome_image_url') or final_output.get('Welcome_image_url') or None,
                "status": "ready"
            }
            
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
        
        # Register with governance
        await governance.register_agent(
            agent_type=AgentType.UNIVERSAL_UPDATER,
            agent_instance=agent,
            agent_id="new_game"
        )
        
        # Issue directive to be ready to create new games
        await governance.issue_directive(
            agent_type=AgentType.UNIVERSAL_UPDATER,
            agent_id="new_game", 
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
