# lore/managers/local_lore.py

import logging
import json
import random
from typing import Dict, List, Any, Optional

# Agents SDK imports
from agents import Agent, function_tool, Runner
from agents.run_context import RunContextWrapper
from agents.run import RunConfig
from agents.models import ModelSettings

# Governance
from nyx.nyx_governance import AgentType, DirectivePriority
from nyx.governance_helpers import with_governance

# Project imports
from embedding.vector_store import generate_embedding
from lore.core.base_manager import BaseLoreManager
from lore.utils.theming import MatriarchalThemingUtils

logger = logging.getLogger(__name__)


class LocalLoreManager(BaseLoreManager):
    """
    Consolidated manager for local lore elements including urban myths, 
    local histories, landmarks, and other location-specific narratives.
    """

    def __init__(self, user_id: int, conversation_id: int):
        super().__init__(user_id, conversation_id)
        self.cache_namespace = "locallore"

    async def initialize_tables(self):
        """Ensure all local lore tables exist."""
        table_definitions = {
            "UrbanMyths": """
                CREATE TABLE UrbanMyths (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    origin_location TEXT,
                    origin_event TEXT,
                    believability INTEGER CHECK (believability BETWEEN 1 AND 10),
                    spread_rate INTEGER CHECK (spread_rate BETWEEN 1 AND 10),
                    regions_known TEXT[],
                    variations TEXT[],
                    creation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    embedding VECTOR(1536)
                );
                
                CREATE INDEX IF NOT EXISTS idx_urbanmyths_embedding 
                ON UrbanMyths USING ivfflat (embedding vector_cosine_ops);
            """,
            "LocalHistories": """
                CREATE TABLE LocalHistories (
                    id SERIAL PRIMARY KEY,
                    location_id INTEGER NOT NULL,
                    event_name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    date_description TEXT,
                    significance INTEGER CHECK (significance BETWEEN 1 AND 10),
                    impact_type TEXT,
                    notable_figures TEXT[],
                    current_relevance TEXT,
                    commemoration TEXT,
                    embedding VECTOR(1536),
                    FOREIGN KEY (location_id) REFERENCES Locations(id) ON DELETE CASCADE
                );
                
                CREATE INDEX IF NOT EXISTS idx_localhistories_embedding 
                ON LocalHistories USING ivfflat (embedding vector_cosine_ops);
                
                CREATE INDEX IF NOT EXISTS idx_localhistories_location
                ON LocalHistories(location_id);
            """,
            "Landmarks": """
                CREATE TABLE Landmarks (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    location_id INTEGER NOT NULL,
                    landmark_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    historical_significance TEXT,
                    current_use TEXT,
                    controlled_by TEXT,
                    legends TEXT[],
                    embedding VECTOR(1536),
                    FOREIGN KEY (location_id) REFERENCES Locations(id) ON DELETE CASCADE
                );
                
                CREATE INDEX IF NOT EXISTS idx_landmarks_embedding 
                ON Landmarks USING ivfflat (embedding vector_cosine_ops);
                
                CREATE INDEX IF NOT EXISTS idx_landmarks_location
                ON Landmarks(location_id);
            """
        }
        await self.initialize_tables_for_class(table_definitions)

    # ------------------------------------------------------------------------
    # 1) Add an urban myth
    # ------------------------------------------------------------------------
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_urban_myth",
        action_description="Adding urban myth: {name}",
        id_from_context=lambda ctx: "local_lore_manager"
    )
    @function_tool
    async def add_urban_myth(
        self,
        ctx,
        name: str,
        description: str,
        origin_location: Optional[str] = None,
        origin_event: Optional[str] = None,
        believability: int = 6,
        spread_rate: int = 5,
        regions_known: List[str] = None
    ) -> int:
        """
        Add a new urban myth to the database (now a function tool).
        """

        await self.ensure_initialized()
        regions_known = regions_known or ["local area"]

        # Theming
        description = MatriarchalThemingUtils.apply_matriarchal_theme("myth", description)

        # Embedding
        embedding_text = f"{name} {description}"
        embedding = await generate_embedding(embedding_text)

        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                myth_id = await conn.fetchval("""
                    INSERT INTO UrbanMyths (
                        name, description, origin_location, origin_event,
                        believability, spread_rate, regions_known, embedding
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    RETURNING id
                """,
                name, description, origin_location, origin_event,
                believability, spread_rate, regions_known, embedding)

                return myth_id

    # ------------------------------------------------------------------------
    # 2) Add local history
    # ------------------------------------------------------------------------
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_local_history",
        action_description="Adding local history event: {event_name}",
        id_from_context=lambda ctx: "local_lore_manager"
    )
    @function_tool
    async def add_local_history(
        self,
        ctx,
        location_id: int,
        event_name: str,
        description: str,
        date_description: str = "Some time ago",
        significance: int = 5,
        impact_type: str = "cultural",
        notable_figures: List[str] = None,
        current_relevance: str = None,
        commemoration: str = None
    ) -> int:
        """
        Add a local historical event to the database (function tool).
        """

        await self.ensure_initialized()
        notable_figures = notable_figures or []

        description = MatriarchalThemingUtils.apply_matriarchal_theme("history", description)

        embedding_text = f"{event_name} {description} {date_description}"
        embedding = await generate_embedding(embedding_text)

        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                event_id = await conn.fetchval("""
                    INSERT INTO LocalHistories (
                        location_id, event_name, description, date_description,
                        significance, impact_type, notable_figures,
                        current_relevance, commemoration, embedding
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    RETURNING id
                """,
                location_id, event_name, description, date_description,
                significance, impact_type, notable_figures,
                current_relevance, commemoration, embedding)

                # Invalidate relevant cache
                self.invalidate_cache_pattern(f"local_history_{location_id}")
                return event_id

    # ------------------------------------------------------------------------
    # 3) Add landmark
    # ------------------------------------------------------------------------
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_landmark",
        action_description="Adding landmark: {name}",
        id_from_context=lambda ctx: "local_lore_manager"
    )
    @function_tool
    async def add_landmark(
        self,
        ctx,
        name: str,
        location_id: int,
        landmark_type: str,
        description: str,
        historical_significance: str = None,
        current_use: str = None,
        controlled_by: str = None,
        legends: List[str] = None
    ) -> int:
        """
        Add a landmark to the database (function tool).
        """

        await self.ensure_initialized()
        legends = legends or []

        description = MatriarchalThemingUtils.apply_matriarchal_theme("landmark", description)

        embedding_text = f"{name} {landmark_type} {description}"
        embedding = await generate_embedding(embedding_text)

        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                landmark_id = await conn.fetchval("""
                    INSERT INTO Landmarks (
                        name, location_id, landmark_type, description,
                        historical_significance, current_use, controlled_by,
                        legends, embedding
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    RETURNING id
                """,
                name, location_id, landmark_type, description,
                historical_significance, current_use, controlled_by,
                legends, embedding)

                self.invalidate_cache_pattern(f"landmarks_{location_id}")
                return landmark_id

    # ------------------------------------------------------------------------
    # 4) Get location lore
    # ------------------------------------------------------------------------
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="get_location_lore",
        action_description="Getting all lore for location: {location_id}",
        id_from_context=lambda ctx: "local_lore_manager"
    )
    async def get_location_lore(self, ctx, location_id: int) -> Dict[str, Any]:
        """
        Get all lore associated with a location (myths, history, landmarks).
        """

        cache_key = f"location_lore_{location_id}"
        cached = self.get_cache(cache_key)
        if cached:
            return cached

        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                location = await conn.fetchrow("""
                    SELECT id, location_name
                    FROM Locations
                    WHERE id = $1
                """, location_id)
                if not location:
                    return {"error": "Location not found"}

                location_name = location["location_name"]

                # Histories
                histories = await conn.fetch("""
                    SELECT id, event_name, description, date_description,
                           significance, impact_type, notable_figures,
                           current_relevance, commemoration
                    FROM LocalHistories
                    WHERE location_id = $1
                    ORDER BY significance DESC
                """, location_id)

                # Landmarks
                landmarks = await conn.fetch("""
                    SELECT id, name, landmark_type, description,
                           historical_significance, current_use,
                           controlled_by, legends
                    FROM Landmarks
                    WHERE location_id = $1
                """, location_id)

                # Myths
                myths = await conn.fetch("""
                    SELECT id, name, description, believability, spread_rate
                    FROM UrbanMyths
                    WHERE origin_location = $1 OR $1 = ANY(regions_known)
                """, location_name)

                result = {
                    "location": dict(location),
                    "histories": [dict(h) for h in histories],
                    "landmarks": [dict(l) for l in landmarks],
                    "myths": [dict(m) for m in myths]
                }
                self.set_cache(cache_key, result)
                return result

    # ------------------------------------------------------------------------
    # 5) Generate location lore
    # ------------------------------------------------------------------------
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_location_lore",
        action_description="Generating lore for location: {location_data['id']}",
        id_from_context=lambda ctx: "local_lore_manager"
    )
    async def generate_location_lore(self, ctx, location_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive lore for a location using LLM calls 
        for myths, local histories, and landmarks.
        """

        run_ctx = self.create_run_context(ctx)

        if not location_data or "id" not in location_data:
            return {"error": "Invalid location data"}

        location_id = location_data["id"]

        # Generate myths
        myths = await self._generate_myths_for_location(run_ctx, location_data)
        # Generate local histories
        histories = await self._generate_local_history(run_ctx, location_data)
        # Generate landmarks
        landmarks = await self._generate_landmarks(run_ctx, location_data)

        self.invalidate_cache(f"location_lore_{location_id}")

        return {
            "location": location_data,
            "generated_myths": myths,
            "generated_histories": histories,
            "generated_landmarks": landmarks
        }

    # ------------------------------------------------------------------------
    # Private generation methods for location lore
    # ------------------------------------------------------------------------
    async def _generate_myths_for_location(self, ctx, location_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate urban myths for a location."""
        location_id = location_data.get('id')
        location_name = location_data.get('location_name', 'Unknown Location')
        location_type = location_data.get('location_type', 'place')
        desc = location_data.get('description', '')

        prompt = f"""
        Generate 2-3 urban myths or local legends associated with this location:

        LOCATION: {location_name} ({location_type})
        DESCRIPTION: {desc}

        Requirements:
        1. Somewhat believable but with fantastical elements
        2. Reflect local concerns or history
        3. Ties to matriarchal power structures

        Return JSON array with:
        - name
        - description
        - believability (1-10)
        - spread_rate (1-10)
        - origin
        """

        myth_agent = Agent(
            name="UrbanMythAgent",
            instructions="You create urban myths and local legends for locations.",
            model="o3-mini",
            model_settings=ModelSettings(temperature=0.8)
        )
        run_config = RunConfig(workflow_name="GenerateMyths")
        result = await Runner.run(myth_agent, prompt, context=ctx.context, run_config=run_config)

        saved_myths = []
        try:
            myths = json.loads(result.final_output)
            if not isinstance(myths, list):
                myths = [myths] if isinstance(myths, dict) else []

            for myth in myths:
                name = myth.get('name')
                description = myth.get('description')
                believability = myth.get('believability', random.randint(4,8))
                spread_rate = myth.get('spread_rate', random.randint(3,7))

                if not name or not description:
                    continue

                try:
                    myth_id = await self.add_urban_myth(
                        ctx,
                        name=name,
                        description=description,
                        origin_location=location_name,
                        believability=believability,
                        spread_rate=spread_rate,
                        regions_known=[location_name]
                    )
                    myth['id'] = myth_id
                    saved_myths.append(myth)
                except Exception as e:
                    logger.error(f"Error saving urban myth '{name}': {e}")

        except json.JSONDecodeError:
            logger.error(f"Failed to parse myths JSON: {result.final_output}")

        return saved_myths

    async def _generate_local_history(self, ctx, location_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate local historical events for a location."""
        location_id = location_data.get('id')
        location_name = location_data.get('location_name', 'Unknown Location')
        location_type = location_data.get('location_type', 'place')
        desc = location_data.get('description', '')

        prompt = f"""
        Generate 2-3 local historical events for this location:

        LOCATION: {location_name} ({location_type})
        DESCRIPTION: {desc}

        Requirements:
        - Reflect local development, conflict, or cultural shifts
        - At least one event about matriarchal power
        - Different time frames

        Return JSON array with:
        - event_name
        - description
        - date_description
        - significance (1-10)
        - impact_type
        - notable_figures
        - current_relevance
        """

        history_agent = Agent(
            name="LocalHistoryAgent",
            instructions="You create local historical events for locations.",
            model="o3-mini",
            model_settings=ModelSettings(temperature=0.8)
        )
        run_config = RunConfig(workflow_name="GenerateLocalHistory")
        result = await Runner.run(history_agent, prompt, context=ctx.context, run_config=run_config)

        saved_events = []
        try:
            events = json.loads(result.final_output)
            if not isinstance(events, list):
                events = [events] if isinstance(events, dict) else []

            for evt in events:
                event_name = evt.get('event_name')
                description = evt.get('description')
                date_description = evt.get('date_description', 'Some time ago')
                significance = evt.get('significance', 5)
                impact_type = evt.get('impact_type', 'cultural')
                notable_figures = evt.get('notable_figures', [])
                current_relevance = evt.get('current_relevance')

                if not event_name or not description:
                    continue

                try:
                    event_id = await self.add_local_history(
                        ctx,
                        location_id=location_id,
                        event_name=event_name,
                        description=description,
                        date_description=date_description,
                        significance=significance,
                        impact_type=impact_type,
                        notable_figures=notable_figures,
                        current_relevance=current_relevance
                    )
                    evt['id'] = event_id
                    saved_events.append(evt)
                except Exception as e:
                    logger.error(f"Error saving local history '{event_name}': {e}")

        except json.JSONDecodeError:
            logger.error(f"Failed to parse local history JSON: {result.final_output}")

        return saved_events

    async def _generate_landmarks(self, ctx, location_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate landmarks for a location."""
        location_id = location_data.get('id')
        location_name = location_data.get('location_name', 'Unknown Location')
        location_type = location_data.get('location_type', 'place')
        desc = location_data.get('description', '')

        prompt = f"""
        Generate 2-3 landmarks for this location:

        LOCATION: {location_name} ({location_type})
        DESCRIPTION: {desc}

        Requirements:
        1. At least one natural landmark (if relevant)
        2. At least one architectural/built landmark
        3. At least one linked to matriarchal power

        Return JSON array with:
        - name
        - landmark_type
        - description
        - historical_significance
        - current_use
        - controlled_by
        - legends (array)
        """

        landmark_agent = Agent(
            name="LandmarkAgent",
            instructions="You create landmarks for specific locations.",
            model="o3-mini",
            model_settings=ModelSettings(temperature=0.8)
        )
        run_config = RunConfig(workflow_name="GenerateLandmarks")
        result = await Runner.run(landmark_agent, prompt, context=ctx.context, run_config=run_config)

        saved_landmarks = []
        try:
            landmarks = json.loads(result.final_output)
            if not isinstance(landmarks, list):
                landmarks = [landmarks] if isinstance(landmarks, dict) else []

            for lm in landmarks:
                name = lm.get('name')
                lm_type = lm.get('landmark_type', 'building')
                description = lm.get('description')
                hist_signif = lm.get('historical_significance')
                current_use = lm.get('current_use')
                controlled_by = lm.get('controlled_by')
                legends = lm.get('legends', [])

                if not name or not description:
                    continue

                try:
                    landmark_id = await self.add_landmark(
                        ctx,
                        name=name,
                        location_id=location_id,
                        landmark_type=lm_type,
                        description=description,
                        historical_significance=hist_signif,
                        current_use=current_use,
                        controlled_by=controlled_by,
                        legends=legends
                    )
                    lm['id'] = landmark_id
                    saved_landmarks.append(lm)
                except Exception as e:
                    logger.error(f"Error saving landmark '{name}': {e}")

        except json.JSONDecodeError:
            logger.error(f"Failed to parse landmark JSON: {result.final_output}")

        return saved_landmarks

    # ------------------------------------------------------------------------
    # 6) Evolve location lore
    # ------------------------------------------------------------------------
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="evolve_location_lore",
        action_description="Evolving lore for location: {location_id}",
        id_from_context=lambda ctx: "local_lore_manager"
    )
    async def evolve_location_lore(self, ctx, location_id: int, event_description: str) -> Dict[str, Any]:
        """
        Evolve the lore of a location based on an event, 
        using agent calls to produce new or updated content.
        """

        run_ctx = self.create_run_context(ctx)
        location_lore = await self.get_location_lore(ctx, location_id)
        if "error" in location_lore:
            return location_lore

        # Theming the event
        themed_event = MatriarchalThemingUtils.apply_matriarchal_theme("event", event_description, emphasis_level=1)

        location_name = location_lore.get('location', {}).get('location_name', 'Unknown')
        evolution_agent = Agent(
            name="LoreEvolutionAgent",
            instructions="You evolve location lore based on an event that occurs.",
            model="o3-mini",
            model_settings=ModelSettings(temperature=0.8)
        )

        # We'll do a triple-prompt approach as before
        history_prompt = f"""
        The location is: {location_name}
        EVENT: {themed_event}
        
        Create one new historical entry in JSON:
        "new_history": {{
          "event_name": "...",
          "description": "...",
          "date_description": "Recently",
          "significance": int (1-10),
          "impact_type": "...",
          "notable_figures": [...],
          "current_relevance": "..."
        }}
        """

        landmark_prompt = f"""
        The location is: {location_name}
        EVENT: {themed_event}

        CURRENT LANDMARKS:
        {json.dumps(location_lore.get('landmarks', [])[:2], indent=2)}

        Decide if we add or modify a landmark. Return JSON:
        - "new_landmark": ... (same structure as add_landmark call)
        - "modified_landmark_id": ...
        - "landmark_update": "New description if modifying"
        """

        myth_prompt = f"""
        The location is: {location_name}
        EVENT: {themed_event}

        Possibly create a new myth. Return JSON with "new_myth": {{
          "name": "...",
          "description": "...",
          "believability": int,
          "spread_rate": int
        }}
        """

        # Run them
        history_result = await Runner.run(evolution_agent, history_prompt, context=run_ctx.context)
        landmark_result = await Runner.run(evolution_agent, landmark_prompt, context=run_ctx.context)
        myth_result = await Runner.run(evolution_agent, myth_prompt, context=run_ctx.context)

        # Process results
        new_history = None
        new_landmark = None
        updated_landmark = None
        new_myth = None

        try:
            # 1) Add new history
            history_data = json.loads(history_result.final_output)
            if "new_history" in history_data:
                h = history_data["new_history"]
                try:
                    hist_id = await self.add_local_history(
                        run_ctx,
                        location_id=location_id,
                        event_name=h.get("event_name","Recent Event"),
                        description=h.get("description",""),
                        date_description=h.get("date_description","Recently"),
                        significance=h.get("significance",5),
                        impact_type=h.get("impact_type","cultural"),
                        notable_figures=h.get("notable_figures",[]),
                        current_relevance=h.get("current_relevance")
                    )
                    h["id"] = hist_id
                    new_history = h
                except Exception as e:
                    logger.error(f"Error adding new history: {e}")
        except json.JSONDecodeError:
            logger.error(f"Failed parsing new_history: {history_result.final_output}")

        try:
            # 2) Landmark changes
            landmark_data = json.loads(landmark_result.final_output)
            if "new_landmark" in landmark_data and landmark_data["new_landmark"]:
                nl = landmark_data["new_landmark"]
                try:
                    lm_id = await self.add_landmark(
                        run_ctx,
                        name=nl.get("name","New Landmark"),
                        location_id=location_id,
                        landmark_type=nl.get("landmark_type","structure"),
                        description=nl.get("description",""),
                        historical_significance=nl.get("historical_significance"),
                        current_use=nl.get("current_use"),
                        controlled_by=nl.get("controlled_by"),
                        legends=nl.get("legends",[])
                    )
                    nl["id"] = lm_id
                    new_landmark = nl
                except Exception as e:
                    logger.error(f"Error adding new landmark: {e}")

            if "modified_landmark_id" in landmark_data and "landmark_update" in landmark_data:
                mod_id = landmark_data["modified_landmark_id"]
                mod_desc = landmark_data["landmark_update"]
                if mod_id and mod_desc:
                    try:
                        async with self.get_connection_pool() as pool:
                            async with pool.acquire() as conn:
                                # verify
                                existing = await conn.fetchrow("""
                                    SELECT * FROM Landmarks WHERE id=$1 AND location_id=$2
                                """, mod_id, location_id)
                                if existing:
                                    await conn.execute("""
                                        UPDATE Landmarks SET description=$1
                                        WHERE id=$2
                                    """, mod_desc, mod_id)
                                    updated_landmark = {
                                        "id": mod_id,
                                        "name": existing["name"],
                                        "old_description": existing["description"],
                                        "new_description": mod_desc
                                    }
                    except Exception as e:
                        logger.error(f"Error updating landmark {mod_id}: {e}")
        except json.JSONDecodeError:
            logger.error(f"Failed parsing new_landmark or modification: {landmark_result.final_output}")

        try:
            # 3) Myth changes
            myth_data = json.loads(myth_result.final_output)
            if "new_myth" in myth_data and myth_data["new_myth"]:
                nm = myth_data["new_myth"]
                try:
                    myth_id = await self.add_urban_myth(
                        run_ctx,
                        name=nm.get("name","New Myth"),
                        description=nm.get("description",""),
                        origin_location=location_lore["location"].get("location_name"),
                        origin_event=themed_event,
                        believability=nm.get("believability",5),
                        spread_rate=nm.get("spread_rate",3),
                        regions_known=[location_lore["location"].get("location_name")]
                    )
                    nm["id"] = myth_id
                    new_myth = nm
                except Exception as e:
                    logger.error(f"Error adding new myth: {e}")
        except json.JSONDecodeError:
            logger.error(f"Failed parsing new_myth: {myth_result.final_output}")

        self.invalidate_cache(f"location_lore_{location_id}")
        return {
            "event": themed_event,
            "location_id": location_id,
            "location_name": location_lore["location"].get("location_name","Unknown"),
            "new_history": new_history,
            "new_landmark": new_landmark,
            "updated_landmark": updated_landmark,
            "new_myth": new_myth
        }

    async def register_with_governance(self):
        """Register with Nyx governance system."""
        await super().register_with_governance(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="local_lore_manager",
            directive_text="Create and manage local lore, myths, and histories with matriarchal influences.",
            scope="world_building",
            priority=DirectivePriority.MEDIUM
        )
