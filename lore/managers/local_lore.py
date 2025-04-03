# lore/managers/local_lore.py

import logging
import json
from typing import Dict, List, Any, Optional
import random

from agents import Agent, Runner
from agents.run_context import RunContextWrapper

from nyx.nyx_governance import AgentType, DirectivePriority
from nyx.governance_helpers import with_governance

from embedding.vector_store import generate_embedding

from lore.core.base_manager import BaseLoreManager
from lore.utils.theming import MatriarchalThemingUtils

class LocalLoreManager(BaseLoreManager):
    """
    Consolidated manager for local lore elements including urban myths, local histories,
    landmarks, and other location-specific narratives.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        super().__init__(user_id, conversation_id)
        self.cache_namespace = "locallore"
    
    async def initialize_tables(self):
        """Ensure all local lore tables exist"""
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
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_urban_myth",
        action_description="Adding urban myth: {name}",
        id_from_context=lambda ctx: "local_lore_manager"
    )
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
        Add a new urban myth to the database
        
        Args:
            name: Name of the urban myth
            description: Full description of the myth
            origin_location: Where the myth originated
            origin_event: What event spawned the myth
            believability: How believable the myth is (1-10)
            spread_rate: How quickly the myth is spreading (1-10)
            regions_known: List of regions where the myth is known
            
        Returns:
            ID of the created urban myth
        """
        # Ensure tables exist
        await self.ensure_initialized()
        
        # Set defaults
        regions_known = regions_known or ["local area"]
        
        # Apply matriarchal theming
        description = MatriarchalThemingUtils.apply_matriarchal_theme("myth", description)
        
        # Generate embedding for the myth
        embedding_text = f"{name} {description}"
        embedding = await generate_embedding(embedding_text)
        
        # Store in database
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                myth_id = await conn.fetchval("""
                    INSERT INTO UrbanMyths (
                        name, description, origin_location, origin_event,
                        believability, spread_rate, regions_known, embedding
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    RETURNING id
                """, name, description, origin_location, origin_event,
                     believability, spread_rate, regions_known, embedding)
                
                return myth_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_local_history",
        action_description="Adding local history event: {event_name}",
        id_from_context=lambda ctx: "local_lore_manager"
    )
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
        Add a local historical event to the database
        
        Args:
            location_id: ID of the associated location
            event_name: Name of the historical event
            description: Description of the event
            date_description: When it occurred
            significance: Importance from 1-10
            impact_type: Type of impact (political, cultural, etc.)
            notable_figures: People involved
            current_relevance: How it affects the present
            commemoration: How it's remembered/celebrated
            
        Returns:
            ID of the created local history event
        """
        # Ensure tables exist
        await self.ensure_initialized()
        
        # Set defaults
        notable_figures = notable_figures or []
        
        # Apply matriarchal theming
        description = MatriarchalThemingUtils.apply_matriarchal_theme("history", description)
        
        # Generate embedding
        embedding_text = f"{event_name} {description} {date_description}"
        embedding = await generate_embedding(embedding_text)
        
        # Store in database
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
                """, location_id, event_name, description, date_description,
                     significance, impact_type, notable_figures,
                     current_relevance, commemoration, embedding)
                
                # Invalidate relevant cache
                self.invalidate_cache_pattern(f"local_history_{location_id}")
                
                return event_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="add_landmark",
        action_description="Adding landmark: {name}",
        id_from_context=lambda ctx: "local_lore_manager"
    )
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
        Add a landmark to the database
        
        Args:
            name: Name of the landmark
            location_id: ID of the associated location
            landmark_type: Type of landmark (monument, building, natural feature, etc.)
            description: Description of the landmark
            historical_significance: Historical importance
            current_use: How it's used today
            controlled_by: Who controls/owns it
            legends: Associated legends or stories
            
        Returns:
            ID of the created landmark
        """
        # Ensure tables exist
        await self.ensure_initialized()
        
        # Set defaults
        legends = legends or []
        
        # Apply matriarchal theming
        description = MatriarchalThemingUtils.apply_matriarchal_theme("landmark", description)
        
        # Generate embedding
        embedding_text = f"{name} {landmark_type} {description}"
        embedding = await generate_embedding(embedding_text)
        
        # Store in database
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
                """, name, location_id, landmark_type, description,
                     historical_significance, current_use, controlled_by,
                     legends, embedding)
                
                # Invalidate relevant cache
                self.invalidate_cache_pattern(f"landmarks_{location_id}")
                
                return landmark_id
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="get_location_lore",
        action_description="Getting all lore for location: {location_id}",
        id_from_context=lambda ctx: "local_lore_manager"
    )
    async def get_location_lore(self, ctx, location_id: int) -> Dict[str, Any]:
        """
        Get all lore associated with a location (myths, history, landmarks)
        
        Args:
            location_id: ID of the location
            
        Returns:
            Dictionary with all lore for the location
        """
        # Check cache first
        cache_key = f"location_lore_{location_id}"
        cached = self.get_cache(cache_key)
        if cached:
            return cached
        
        # Get location details
        async with self.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get location name
                location = await conn.fetchrow("""
                    SELECT id, location_name
                    FROM Locations
                    WHERE id = $1
                """, location_id)
                
                if not location:
                    return {"error": "Location not found"}
                
                location_name = location["location_name"]
                
                # Get all local histories
                histories = await conn.fetch("""
                    SELECT id, event_name, description, date_description,
                           significance, impact_type, notable_figures,
                           current_relevance, commemoration
                    FROM LocalHistories
                    WHERE location_id = $1
                    ORDER BY significance DESC
                """, location_id)
                
                # Get all landmarks
                landmarks = await conn.fetch("""
                    SELECT id, name, landmark_type, description,
                           historical_significance, current_use,
                           controlled_by, legends
                    FROM Landmarks
                    WHERE location_id = $1
                """, location_id)
                
                # Get all myths
                myths = await conn.fetch("""
                    SELECT id, name, description, believability, spread_rate
                    FROM UrbanMyths
                    WHERE origin_location = $1 OR $1 = ANY(regions_known)
                """, location_name)
                
                # Compile result
                result = {
                    "location": dict(location),
                    "histories": [dict(hist) for hist in histories],
                    "landmarks": [dict(landmark) for landmark in landmarks],
                    "myths": [dict(myth) for myth in myths]
                }
                
                # Cache result
                self.set_cache(cache_key, result)
                
                return result
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_location_lore",
        action_description="Generating lore for location: {location_data['id']}",
        id_from_context=lambda ctx: "local_lore_manager"
    )
    async def generate_location_lore(self, ctx, location_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive lore for a location
        
        Args:
            location_data: Dictionary with location details
            
        Returns:
            Dictionary with generated lore
        """
        # Create run context
        run_ctx = self.create_run_context(ctx)
        
        # Ensure location data is valid
        if not location_data or "id" not in location_data:
            return {"error": "Invalid location data"}
        
        location_id = location_data["id"]
        
        # Generate myths
        myths = await self._generate_myths_for_location(run_ctx, location_data)
        
        # Generate local histories
        histories = await self._generate_local_history(run_ctx, location_data)
        
        # Generate landmarks
        landmarks = await self._generate_landmarks(run_ctx, location_data)
        
        # Invalidate location lore cache
        self.invalidate_cache(f"location_lore_{location_id}")
        
        return {
            "location": location_data,
            "generated_myths": myths,
            "generated_histories": histories,
            "generated_landmarks": landmarks
        }
    
    async def _generate_myths_for_location(self, ctx, location_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate urban myths for a location"""
        # Extract relevant details
        location_id = location_data.get('id')
        location_name = location_data.get('location_name', 'Unknown Location')
        location_type = location_data.get('location_type', 'place')
        description = location_data.get('description', '')
        
        # Create a prompt for the LLM
        prompt = f"""
        Generate 2-3 urban myths or local legends associated with this location:
        
        LOCATION: {location_name} ({location_type})
        DESCRIPTION: {description}
        
        Create urban myths that feel authentic to this location. Each myth should:
        1. Be somewhat believable but with fantastical elements
        2. Reflect local concerns, features, or history
        3. Have some connection to matriarchal power structures
        
        Format your response as a JSON array where each object has:
        - "name": The name/title of the myth
        - "description": A detailed description of the myth
        - "believability": Number from 1-10 indicating how believable it is
        - "spread_rate": Number from 1-10 indicating how widely it has spread
        - "origin": Brief statement of how the myth originated
        """
        
        # Create an agent for myth generation
        myth_agent = Agent(
            name="UrbanMythAgent",
            instructions="You create urban myths and local legends for locations.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(myth_agent, prompt, context=ctx.context)
        
        try:
            # Parse the JSON response
            myths = json.loads(result.final_output)
            
            # Ensure we got a list
            if not isinstance(myths, list):
                if isinstance(myths, dict):
                    myths = [myths]
                else:
                    myths = []
            
            # Store each myth
            saved_myths = []
            for myth in myths:
                # Extract myth details
                name = myth.get('name')
                description = myth.get('description')
                believability = myth.get('believability', random.randint(4, 8))
                spread_rate = myth.get('spread_rate', random.randint(3, 7))
                
                if not name or not description:
                    continue
                
                # Save the myth
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
                    
                    # Add to results
                    myth['id'] = myth_id
                    saved_myths.append(myth)
                except Exception as e:
                    logging.error(f"Error saving urban myth '{name}': {e}")
            
            return saved_myths
        except json.JSONDecodeError:
            logging.error(f"Failed to parse LLM response for urban myths: {result.final_output}")
            return []
    
    async def _generate_local_history(self, ctx, location_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate local historical events for a location"""
        # Extract relevant details
        location_id = location_data.get('id')
        location_name = location_data.get('location_name', 'Unknown Location')
        location_type = location_data.get('location_type', 'place')
        description = location_data.get('description', '')
        
        # Create a prompt for the LLM
        prompt = f"""
        Generate 2-3 local historical events specific to this location:
        
        LOCATION: {location_name} ({location_type})
        DESCRIPTION: {description}
        
        Create local historical events that feel authentic to this location. Each event should:
        1. Be specific to this location rather than world-changing
        2. Reflect local development, conflicts, or cultural shifts
        3. Include at least one event related to matriarchal power structures
        4. Include a range of timeframes (some recent, some older)
        
        Format your response as a JSON array where each object has:
        - "event_name": The name of the historical event
        - "description": A detailed description of what happened
        - "date_description": When it occurred (e.g., "50 years ago", "during the reign of...")
        - "significance": Number from 1-10 indicating historical importance
        - "impact_type": Type of impact (political, cultural, economic, religious, etc.)
        - "notable_figures": Array of names of people involved
        - "current_relevance": How it still affects the location today
        """
        
        # Create an agent for history generation
        history_agent = Agent(
            name="LocalHistoryAgent",
            instructions="You create local historical events for specific locations.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(history_agent, prompt, context=ctx.context)
        
        try:
            # Parse the JSON response
            events = json.loads(result.final_output)
            
            # Ensure we got a list
            if not isinstance(events, list):
                if isinstance(events, dict):
                    events = [events]
                else:
                    events = []
            
            # Store each event
            saved_events = []
            for event in events:
                # Extract event details
                event_name = event.get('event_name')
                description = event.get('description')
                date_description = event.get('date_description', 'Some time ago')
                significance = event.get('significance', 5)
                impact_type = event.get('impact_type', 'cultural')
                notable_figures = event.get('notable_figures', [])
                current_relevance = event.get('current_relevance')
                
                if not event_name or not description:
                    continue
                
                # Save the event
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
                    
                    # Add to results
                    event['id'] = event_id
                    saved_events.append(event)
                except Exception as e:
                    logging.error(f"Error saving local historical event '{event_name}': {e}")
            
            return saved_events
        except json.JSONDecodeError:
            logging.error(f"Failed to parse LLM response for local history: {result.final_output}")
            return []
    
    async def _generate_landmarks(self, ctx, location_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate landmarks for a location"""
        # Extract relevant details
        location_id = location_data.get('id')
        location_name = location_data.get('location_name', 'Unknown Location')
        location_type = location_data.get('location_type', 'place')
        description = location_data.get('description', '')
        
        # Create a prompt for the LLM
        prompt = f"""
        Generate 2-3 landmarks found in this location:
        
        LOCATION: {location_name} ({location_type})
        DESCRIPTION: {description}
        
        Create landmarks that feel authentic to this location. Include:
        1. At least one natural landmark (if appropriate)
        2. At least one architectural/built landmark
        3. At least one landmark related to matriarchal power structures
        
        Format your response as a JSON array where each object has:
        - "name": The name of the landmark
        - "landmark_type": Type of landmark (monument, building, natural feature, temple, etc.)
        - "description": A detailed physical description
        - "historical_significance": Its importance to local history
        - "current_use": How it's used today (ceremonial, practical, tourist attraction, etc.)
        - "controlled_by": Which faction or group controls it
        - "legends": Array of brief legends or stories associated with it
        """
        
        # Create an agent for landmark generation
        landmark_agent = Agent(
            name="LandmarkAgent",
            instructions="You create landmarks for specific locations.",
            model="o3-mini"
        )
        
        # Get the response
        result = await Runner.run(landmark_agent, prompt, context=ctx.context)
        
        try:
            # Parse the JSON response
            landmarks = json.loads(result.final_output)
            
            # Ensure we got a list
            if not isinstance(landmarks, list):
                if isinstance(landmarks, dict):
                    landmarks = [landmarks]
                else:
                    landmarks = []
            
            # Store each landmark
            saved_landmarks = []
            for landmark in landmarks:
                # Extract landmark details
                name = landmark.get('name')
                landmark_type = landmark.get('landmark_type', 'building')
                description = landmark.get('description')
                historical_significance = landmark.get('historical_significance')
                current_use = landmark.get('current_use')
                controlled_by = landmark.get('controlled_by')
                legends = landmark.get('legends', [])
                
                if not name or not description:
                    continue
                
                # Save the landmark
                try:
                    landmark_id = await self.add_landmark(
                        ctx,
                        name=name,
                        location_id=location_id,
                        landmark_type=landmark_type,
                        description=description,
                        historical_significance=historical_significance,
                        current_use=current_use,
                        controlled_by=controlled_by,
                        legends=legends
                    )
                    
                    # Add to results
                    landmark['id'] = landmark_id
                    saved_landmarks.append(landmark)
                except Exception as e:
                    logging.error(f"Error saving landmark '{name}': {e}")
            
            return saved_landmarks
        except json.JSONDecodeError:
            logging.error(f"Failed to parse LLM response for landmarks: {result.final_output}")
            return []
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="evolve_location_lore",
        action_description="Evolving lore for location: {location_id}",
        id_from_context=lambda ctx: "local_lore_manager"
    )
    async def evolve_location_lore(self, ctx, location_id: int, event_description: str) -> Dict[str, Any]:
        """
        Evolve the lore of a location based on an event
        
        Args:
            location_id: ID of the location
            event_description: Description of the event affecting the location
            
        Returns:
            Dictionary with evolution results
        """
        # Create run context
        run_ctx = self.create_run_context(ctx)
        
        # Get current location lore
        location_lore = await self.get_location_lore(ctx, location_id)
        
        if "error" in location_lore:
            return location_lore
        
        # Theming the event
        themed_event = MatriarchalThemingUtils.apply_matriarchal_theme("event", event_description, emphasis_level=1)
        
        # Get location name
        location_name = location_lore.get('location', {}).get('location_name', 'Unknown Location')
        
        # Create an agent for lore evolution
        evolution_agent = Agent(
            name="LoreEvolutionAgent",
            instructions="You evolve location lore based on events that affect the location.",
            model="o3-mini"
        )
        
        # Process each type of lore and generate updates
        
        # 1. Generate a new historical entry for this event
        history_prompt = f"""
        Based on this event that occurred at {location_name}, create a new historical entry:

        EVENT:
        {themed_event}

        Create a new historical entry for this event.

        Format your response as a JSON object with:
        "new_history": {{
            "event_name": "Name for this historical event",
            "description": "Detailed description of what happened",
            "date_description": "Recently",
            "significance": Number from 1-10 indicating historical importance,
            "impact_type": Type of impact (political, cultural, etc.),
            "notable_figures": Array of people involved,
            "current_relevance": How it affects the location now
        }}
        """
        
        # 2. Maybe add a new landmark or modify existing one
        landmark_prompt = f"""
        Based on this event that occurred at {location_name}, determine if it would create a new landmark 
        or significantly modify an existing one:

        EVENT:
        {themed_event}

        CURRENT LANDMARKS:
        {json.dumps(location_lore.get('landmarks', [])[:2], indent=2)}

        Format your response as a JSON object with:
        - "new_landmark": Optional details for a new landmark if the event creates one
        - "modified_landmark_id": Optional ID of a landmark to modify
        - "landmark_update": Optional new description for the modified landmark
        """
        
        # 3. Maybe add a new urban myth
        myth_prompt = f"""
        Based on this event that occurred at {location_name}, determine if it would spawn a new urban myth:

        EVENT:
        {themed_event}

        Format your response as a JSON object with:
        "new_myth": {{
            "name": "Name of the new myth",
            "description": "Detailed description of the myth",
            "believability": Number from 1-10,
            "spread_rate": Number from 1-10
        }}
        """
        
        # Execute all three prompts
        history_result = await Runner.run(evolution_agent, history_prompt, context=run_ctx.context)
        landmark_result = await Runner.run(evolution_agent, landmark_prompt, context=run_ctx.context)
        myth_result = await Runner.run(evolution_agent, myth_prompt, context=run_ctx.context)
        
        # Process the results and apply updates
        try:
            # Add new history entry
            history_changes = json.loads(history_result.final_output)
            new_history = None
            
            if "new_history" in history_changes:
                history_entry = history_changes["new_history"]
                
                # Add the history entry
                try:
                    history_id = await self.add_local_history(
                        run_ctx,
                        location_id=location_id,
                        event_name=history_entry.get("event_name", "Recent Event"),
                        description=history_entry.get("description", ""),
                        date_description=history_entry.get("date_description", "Recently"),
                        significance=history_entry.get("significance", 5),
                        impact_type=history_entry.get("impact_type", "event"),
                        notable_figures=history_entry.get("notable_figures", []),
                        current_relevance=history_entry.get("current_relevance")
                    )
                    
                    history_entry["id"] = history_id
                    new_history = history_entry
                except Exception as e:
                    logging.error(f"Error adding new history entry: {e}")
            
            # Process landmark changes
            landmark_changes = json.loads(landmark_result.final_output)
            new_landmark = None
            updated_landmark = None
            
            # Add new landmark if suggested
            if "new_landmark" in landmark_changes and landmark_changes["new_landmark"]:
                landmark_info = landmark_changes["new_landmark"]
                try:
                    landmark_id = await self.add_landmark(
                        run_ctx,
                        name=landmark_info.get("name", "New Landmark"),
                        location_id=location_id,
                        landmark_type=landmark_info.get("landmark_type", "structure"),
                        description=landmark_info.get("description", ""),
                        historical_significance=landmark_info.get("historical_significance", f"Created during the {themed_event}"),
                        current_use=landmark_info.get("current_use"),
                        controlled_by=landmark_info.get("controlled_by")
                    )
                    
                    landmark_info["id"] = landmark_id
                    new_landmark = landmark_info
                except Exception as e:
                    logging.error(f"Error adding new landmark: {e}")
            
            # Update existing landmark if suggested
            if "modified_landmark_id" in landmark_changes and landmark_changes["modified_landmark_id"] and "landmark_update" in landmark_changes:
                landmark_id = landmark_changes["modified_landmark_id"]
                new_description = landmark_changes["landmark_update"]
                
                try:
                    async with self.get_connection_pool() as pool:
                        async with pool.acquire() as conn:
                            # Get current landmark to verify it exists
                            landmark = await conn.fetchrow("""
                                SELECT * FROM Landmarks WHERE id = $1 AND location_id = $2
                            """, landmark_id, location_id)
                            
                            if landmark:
                                # Apply update
                                await conn.execute("""
                                    UPDATE Landmarks 
                                    SET description = $1
                                    WHERE id = $2
                                """, new_description, landmark_id)
                                
                                updated_landmark = {
                                    "id": landmark_id,
                                    "name": landmark["name"],
                                    "old_description": landmark["description"],
                                    "new_description": new_description
                                }
                except Exception as e:
                    logging.error(f"Error updating landmark {landmark_id}: {e}")
            
            # Process myth changes
            myth_changes = json.loads(myth_result.final_output)
            new_myth = None
            
            if "new_myth" in myth_changes and myth_changes["new_myth"]:
                myth_info = myth_changes["new_myth"]
                try:
                    myth_id = await self.add_urban_myth(
                        run_ctx,
                        name=myth_info.get("name", "New Myth"),
                        description=myth_info.get("description", ""),
                        origin_location=location_name,
                        origin_event=themed_event,
                        believability=myth_info.get("believability", 5),
                        spread_rate=myth_info.get("spread_rate", 3),
                        regions_known=[location_name]
                    )
                    
                    myth_info["id"] = myth_id
                    new_myth = myth_info
                except Exception as e:
                    logging.error(f"Error adding new myth: {e}")
            
            # Invalidate cache for this location
            self.invalidate_cache(f"location_lore_{location_id}")
            
            # Return results
            return {
                "event": themed_event,
                "location_id": location_id,
                "location_name": location_name,
                "new_history": new_history,
                "new_landmark": new_landmark,
                "updated_landmark": updated_landmark,
                "new_myth": new_myth
            }
            
        except Exception as e:
            logging.error(f"Error processing location lore evolution: {e}")
            return {"error": f"Failed to evolve location lore: {str(e)}"}
    
    async def register_with_governance(self):
        """Register with Nyx governance system."""
        await super().register_with_governance(
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="local_lore_manager",
            directive_text="Create and manage local lore, myths, and histories with matriarchal influences.",
            scope="world_building",
            priority=DirectivePriority.MEDIUM
        )
