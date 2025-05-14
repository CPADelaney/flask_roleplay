# new_game_agent.py

import logging
import json
import asyncio
import os
import functools
from datetime import datetime
from typing import Dict, Any

from agents import Agent, Runner, function_tool, GuardrailFunctionOutput, InputGuardrail
from pydantic import BaseModel, Field

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

from nyx.nyx_governance import AgentType # Make sure this is imported

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

class NewGameAgent:
    """Agent for handling new game creation process with Nyx governance integration"""
    
    def __init__(self):
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
                function_tool(self.create_calendar)
            ]
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
                function_tool(self.spawn_npcs),
                function_tool(self.create_chase_schedule)
            ]
        )
        
        self.narrative_agent = Agent(
            name="OpeningNarrator",
            instructions="""
            As Nyx, craft the opening narrative for a new game with subtle femdom undertones.
            
            Your voice should drape over Chase like a warm shroud—each word a silken thread stitching him into your world.
            Use guile and quiet lures to veil the control beneath. Create an immersive, atmospheric introduction
            that feels like a gentle descent into a comfortable routine while hinting at deeper layers.
            
            Address Chase as 'you,' drawing him through the veil with no whisper of retreat.
            """
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
                function_tool(self.generate_environment),
                function_tool(self.create_npcs_and_schedules),
                function_tool(self.create_opening_narrative),
                function_tool(self.finalize_game_setup),
                function_tool(self.generate_lore)
            ]
        )
        
        # Directive handler for processing Nyx directives
        self.directive_handler = None
    
    async def initialize_directive_handler(self, user_id: int, conversation_id: int):
        """Initialize the directive handler for this agent"""
        from nyx.integrate import get_central_governance, register_with_governance
        governance = await get_central_governance(user_id, conversation_id)
        handler = DirectiveHandler(
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

    logger.info("Scheduling StoryDirector initialization after startup")
    
    async def _init_story_director():
        try:
            logger.info("Deferred StoryDirector async init starts")
            await initialize_story_director(story_user_id, story_conversation_id)
            await register_with_governance(story_user_id, story_conversation_id)
            logger.info("StoryDirector initialized+registered (deferred)")
        except Exception as e:
            logger.error(f"Deferred StoryDirector init failed: {e}", exc_info=True)
    
    asyncio.create_task(_init_story_director())
    
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
            
            # Simulate context
            ctx = type('obj', (object,), {'context': {'user_id': self.directive_handler.user_id, 'conversation_id': self.directive_handler.conversation_id}})
            
            # Generate environment
            result = await self.generate_environment(ctx, mega_name, mega_desc)
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
    async def create_calendar(self, ctx, environment_desc):
        """
        Create an immersive calendar system for the game world.
        
        Args:
            environment_desc: Description of the environment
            
        Returns:
            Dictionary with calendar details
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        calendar_data = await update_calendar_names(user_id, conversation_id, environment_desc)
        return calendar_data
    
    @with_governance(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        action_type="generate_environment",
        action_description="Generated game environment for new game"
    )
    async def generate_environment(self, ctx, mega_name, mega_desc, env_components=None, enhanced_features=None, stat_modifiers=None):
        """
        Generate the game environment data.
        
        Args:
            mega_name: Name of the merged setting
            mega_desc: Description of the merged setting
            env_components: List of environment components
            enhanced_features: List of enhanced features
            stat_modifiers: Dictionary of stat modifiers
            
        Returns:
            EnvironmentData object
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        env_comp_text = "\n".join(env_components) if env_components else "No components provided"
        enh_feat_text = ", ".join(enhanced_features) if enhanced_features else "No enhanced features"
        stat_mod_text = ", ".join([f"{k}: {v}" for k, v in stat_modifiers.items()]) if stat_modifiers else "No stat modifiers"
        
        # Create prompt for the environment agent
        prompt = f"""
        You are setting up a new daily-life sim environment with subtle, hidden layers of femdom and intrigue.

        Below is a merged environment concept:
        Mega Setting Name: {mega_name}
        Mega Description:
        {mega_desc}

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
        
        # Get calendar data
        calendar_data = await self.create_calendar(ctx, result.final_output.environment_desc)
        
        # Store environment data in database
        async with get_db_connection_context() as conn:
            # Store environment description
            await conn.execute("""
                INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                VALUES($1, $2, 'EnvironmentDesc', $3)
                ON CONFLICT (user_id, conversation_id, key)
                DO UPDATE SET value=EXCLUDED.value
            """, user_id, conversation_id, result.final_output.environment_desc + "\n\nHistory: " + result.final_output.environment_history)
            
            # Store setting name
            await conn.execute("""
                INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                VALUES($1, $2, 'CurrentSetting', $3)
                ON CONFLICT (user_id, conversation_id, key)
                DO UPDATE SET value=EXCLUDED.value
            """, user_id, conversation_id, result.final_output.setting_name)
            
            # Insert events
            for event in result.final_output.events:
                await conn.execute("""
                    INSERT INTO Events (
                        user_id, conversation_id,
                        event_name, description, start_time, end_time, location,
                        year, month, day, time_of_day
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """,
                    user_id, conversation_id,
                    event.get("name", "Unnamed Event"),
                    event.get("description", ""),
                    event.get("start_time", "TBD Start"),
                    event.get("end_time", "TBD End"),
                    event.get("location", "Unknown"),
                    event.get("year", 1),
                    event.get("month", 1),
                    event.get("day", 1),
                    event.get("time_of_day", "Morning")
                )
            
            # Insert locations
            for location in result.final_output.locations:
                await conn.execute("""
                    INSERT INTO Locations (user_id, conversation_id, location_name, description, open_hours)
                    VALUES($1, $2, $3, $4, $5)
                """, 
                    user_id, 
                    conversation_id, 
                    location.get("location_name", "Unnamed"),
                    location.get("description", ""),
                    json.dumps(location.get("open_hours", []))
                )
            
            # Insert main quest
            quest_data = result.final_output.quest_data
            await conn.execute("""
                INSERT INTO Quests (user_id, conversation_id, quest_name, status, progress_detail, quest_giver, reward)
                VALUES($1, $2, $3, 'In Progress', $4, '', '')
            """, 
                user_id, 
                conversation_id, 
                quest_data.get("quest_name", "Unnamed Quest"),
                quest_data.get("quest_description", "Quest summary")
            )
            
            # Update conversation name
            await conn.execute("""
                UPDATE conversations
                SET conversation_name=$1
                WHERE id=$2 AND user_id=$3
            """, result.final_output.scenario_name, conversation_id, user_id)
            
            # Store calendar names
            await conn.execute("""
                INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                VALUES($1, $2, 'CalendarNames', $3)
                ON CONFLICT (user_id, conversation_id, key)
                DO UPDATE SET value=EXCLUDED.value
            """, user_id, conversation_id, json.dumps(calendar_data))
        
        return result.final_output
    
    @with_governance_permission(AgentType.UNIVERSAL_UPDATER, "spawn_npcs")
    async def spawn_npcs(self, ctx, environment_desc, day_names, count=5):
        """
        Spawn multiple NPCs for the game world.
        
        Args:
            environment_desc: Description of the environment
            day_names: List of day names for scheduling
            count: Number of NPCs to spawn
            
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
            count=count
        )
        
        return npc_ids
    
    @with_governance_permission(AgentType.UNIVERSAL_UPDATER, "create_chase_schedule")
    async def create_chase_schedule(self, ctx, environment_desc, day_names):
        """
        Create a schedule for the player "Chase".
        
        Args:
            environment_desc: Description of the environment
            day_names: List of day names for scheduling
            
        Returns:
            Dictionary with Chase's schedule
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        # Create a basic schedule structure
        default_schedule = {}
        for day in day_names:
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
    async def create_npcs_and_schedules(self, ctx, environment_data):
        """
        Create NPCs and schedules for the game world.
        
        Args:
            environment_data: Environment data from generate_environment
            
        Returns:
            Dictionary with NPC IDs and Chase's schedule
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        # Get calendar data for day names
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow("""
                SELECT value FROM CurrentRoleplay
                WHERE user_id=$1 AND conversation_id=$2 AND key='CalendarNames'
                LIMIT 1
            """, user_id, conversation_id)
            
            calendar_data = json.loads(row["value"]) if row else {}
            day_names = calendar_data.get("days", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        
        # Create NPCs
        environment_desc = environment_data.get("environment_desc", "") + "\n\n" + environment_data.get("environment_history", "")
        npc_ids = await self.spawn_npcs(ctx, environment_desc, day_names)
        
        # Create Chase's schedule
        chase_schedule = await self.create_chase_schedule(ctx, environment_desc, day_names)
        
        # Store Chase's schedule
        async with get_db_connection_context() as conn:
            await conn.execute("""
                INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                VALUES($1, $2, 'ChaseSchedule', $3)
                ON CONFLICT (user_id, conversation_id, key)
                DO UPDATE SET value=EXCLUDED.value
            """, user_id, conversation_id, json.dumps(chase_schedule))
        
        return {
            "npc_ids": npc_ids,
            "chase_schedule": chase_schedule
        }
    
    @with_governance(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        action_type="create_opening_narrative",
        action_description="Created opening narrative for new game"
    )
    async def create_opening_narrative(self, ctx, environment_data, npc_schedule_data):
        """
        Create the opening narrative for the game.
        
        Args:
            environment_data: Environment data from generate_environment
            npc_schedule_data: NPC and schedule data from create_npcs_and_schedules
            
        Returns:
            String with the opening narrative
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
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
        
        # Store the opening narrative
        async with get_db_connection_context() as conn:
            await conn.execute("""
                INSERT INTO messages (conversation_id, sender, content)
                VALUES($1, $2, $3)
            """, conversation_id, "Nyx", opening_narrative)
        
        return opening_narrative
    
    @with_governance(
        agent_type=AgentType.UNIVERSAL_UPDATER,
        action_type="finalize_game_setup",
        action_description="Finalized game setup including lore, conflict, currency and image"
    )
    async def finalize_game_setup(self, ctx, opening_narrative):
        """
        Finalize game setup including lore generation, conflict generation and image generation.
        
        Args:
            opening_narrative: The opening narrative
            
        Returns:
            Dictionary with finalization results
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        # Mark conversation as ready
        async with get_db_connection_context() as conn:
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
            logging.info(f"Lore generation complete: {lore_summary}")
        except Exception as e:
            logging.error(f"Error generating lore: {e}", exc_info=True)
            lore_summary = "Failed to generate lore"
        
        # Generate initial conflict
        try:
            # MODIFIED: Import ConflictSystemIntegration only when needed
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
        
        # Generate currency system
        try:
            from logic.currency_generator import CurrencyGenerator
            currency_gen = CurrencyGenerator(user_id, conversation_id)
            currency_system = await currency_gen.get_currency_system()
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
            
            # Store the image URL
            async with get_db_connection_context() as conn:
                await conn.execute("""
                    INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                    VALUES($1, $2, 'WelcomeImageUrl', $3)
                    ON CONFLICT (user_id, conversation_id, key)
                    DO UPDATE SET value=EXCLUDED.value
                """, user_id, conversation_id, welcome_image_url)
        
        return {
            "status": "ready",
            "welcome_image_url": welcome_image_url,
            "lore_summary": lore_summary,
            "initial_conflict": conflict_name,
            "currency_system": currency_name
        }
    
    async def _get_setting_name(self, ctx):
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
    async def generate_lore(self, ctx, environment_desc: str) -> Dict[str, Any]:
        """
        Generate comprehensive lore for the game environment.
        
        Args:
            environment_desc: Description of the environment
            
        Returns:
            Dictionary with generated lore data
        """
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        try:
            # Get the lore system instance and initialize it
            lore_system = DynamicLoreGenerator.get_instance(user_id, conversation_id)
            await lore_system.initialize()
            
            # Generate comprehensive lore based on the environment
            logging.info(f"Generating lore for environment: {environment_desc[:100]}...")
            lore_result = await lore_system.generate_world_lore(environment_desc)
            
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
        provided_convo_id = conversation_data.get("conversation_id")
        from nyx.integrate import get_central_governance, register_with_governance
    
        async with get_db_connection_context() as conn:
            if not provided_convo_id:
                row = await conn.fetchrow("""
                    INSERT INTO conversations (user_id, conversation_name)
                    VALUES ($1, 'New Game')
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
    
            # Clear old data, etc...
            tables = ["Events", "PlannedEvents", "PlayerInventory", "Quests",
                      "NPCStats", "Locations", "SocialLinks", "CurrentRoleplay"]
            for t in tables:
                # Use f-string carefully, ensure t is from a controlled list (which it is here)
                await conn.execute(f"DELETE FROM {t} WHERE user_id=$1 AND conversation_id=$2",
                                   user_id, conversation_id)
        
        # NOW we have conversation_id, so let's insert default "Chase" stats
        # Since insert_default_player_stats_chase is async, await it directly:
        await insert_default_player_stats_chase(user_id, conversation_id) # <<< CORRECTED CALL
        logging.info(f"Default player stats for Chase inserted for user_id={user_id}, conv_id={conversation_id}")
        
        # Initialize directive handler for this session
        await self.initialize_directive_handler(user_id, conversation_id)
        
        # Register with the governance system
        governance = await get_central_governance(user_id, conversation_id)
        await governance.register_agent(
            agent_type=NEW_GAME_AGENT_NYX_TYPE,
            agent_instance=self,
            agent_id=NEW_GAME_AGENT_NYX_ID
        )
        
        # Process directives (to pick up any pending directives for this game)
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
        
        # Return success message and data
        return {
            "message": f"New game started. environment={result.final_output.get('setting_name', 'Unknown')}, lore={result.final_output.get('lore_summary', 'Standard lore')}, conversation_id={conversation_id}",
            "scenario_name": result.final_output.get('scenario_name', 'New Game'),
            "environment_name": result.final_output.get('setting_name', 'Unknown Setting'),
            "environment_desc": result.final_output.get('environment_desc', ''),
            "lore_summary": result.final_output.get('lore_summary', 'Standard lore generated'),
            "conversation_id": conversation_id,
            "welcome_image_url": result.final_output.get('welcome_image_url', None)
        }

# Register with governance system
async def register_with_governance(user_id: int, conversation_id: int) -> None:
    """
    Register the NewGameAgent with the Nyx governance system.
    
    Args:
        user_id: User ID
        conversation_id: Conversation ID
    """
    from nyx.integrate import get_central_governance, register_with_governance
    try:
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
