# lore/matriarchal_lore_system.py

import logging
import random
import json
import re
import asyncio
import datetime
from typing import Dict, List, Any, Optional, Set, Tuple

# Agents SDK imports
from agents import Agent, ModelSettings, function_tool, Runner
from agents.run_context import RunContextWrapper

# Nyx governance integration
from nyx.integrate import get_central_governance
from nyx.nyx_governance import AgentType, DirectiveType
from nyx.governance_helpers import with_governance, with_governance_permission

# Database functionality
from db.connection import get_db_connection
from embedding.vector_store import generate_embedding
from utils.caching import LoreCache, FAITH_CACHE

# Import existing modules
from lore.lore_manager import LoreManager
from lore.enhanced_lore import GeopoliticalSystemManager, EmergentLoreSystem, FaithSystem

# Initialize cache for various lore elements
LORE_CACHE = LoreCache(max_size=200, ttl=3600)  # 1 hour TTL
CONFLICT_CACHE = LoreCache(max_size=100, ttl=3600)  # 1 hour TTL
CULTURE_CACHE = LoreCache(max_size=100, ttl=7200)  # 2 hour TTL

# -------------------------------------------------
# INTEGRATED MATRIARCHAL LORE SYSTEM
# -------------------------------------------------

class MatriarchalLoreSystem:
    """
    Master class that integrates all lore systems with a matriarchal theme focus.
    Acts as the primary interface for all lore generation and updates.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.lore_manager = LoreManager(user_id, conversation_id)
        
        # Initialize sub-systems
        self.culture_system = RegionalCultureSystem(user_id, conversation_id)
        self.conflict_system = NationalConflictSystem(user_id, conversation_id)
        self.religion_system = ReligiousDistributionSystem(user_id, conversation_id)
        self.update_system = LoreUpdateSystem(user_id, conversation_id)
        self.geopolitical_manager = GeopoliticalSystemManager(user_id, conversation_id)
        self.governor = None
        
    async def initialize_governance(self):
        """Initialize Nyx governance connection"""
        if not self.governor:
            self.governor = await get_central_governance(self.user_id, self.conversation_id)
        return self.governor
    
    async def initialize_all_systems(self):
        """Initialize all subsystems and their tables"""
        await self.initialize_governance()
        
        # Initialize all sub-systems
        await self.culture_system.initialize_tables()
        await self.conflict_system.initialize_tables()
        await self.religion_system.initialize_tables()
        
        # Register systems with governance
        await self.culture_system.register_with_governance()
        await self.conflict_system.register_with_governance()
        await self.religion_system.register_with_governance()
        
        logging.info(f"All matriarchal lore systems initialized for user {self.user_id}, conversation {self.conversation_id}")
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_complete_world",
        action_description="Generating complete matriarchal world lore",
        id_from_context=lambda ctx: "matriarchal_lore_system"
    )
    async def generate_complete_world(self, ctx, environment_desc: str) -> Dict[str, Any]:
        """
        Generate a complete world with matriarchal theming
        
        Args:
            environment_desc: Description of the environment/setting
            
        Returns:
            Dictionary containing the complete world lore
        """
        # Create run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # 1. First generate base lore through DynamicLoreGenerator
        dynamic_lore = DynamicLoreGenerator(self.user_id, self.conversation_id)
        
        # Check for directives before proceeding
        await dynamic_lore._check_and_process_directives()
        
        # Generate foundation lore
        foundation_data = await dynamic_lore.initialize_world_lore(environment_desc)
        
        # Apply matriarchal theming to foundation lore
        for key, content in foundation_data.items():
            foundation_data[key] = MatriarchalThemingUtils.apply_matriarchal_theme(key, content)
        
        # 2. Generate factions with matriarchal power structures
        factions_data = await dynamic_lore.generate_factions(environment_desc, foundation_data)
        
        # 3. Generate cultural elements
        cultural_data = await dynamic_lore.generate_cultural_elements(environment_desc, factions_data)
        
        # 4. Generate historical events emphasizing matriarchal history
        historical_data = await dynamic_lore.generate_historical_events(
            environment_desc, foundation_data, factions_data
        )
        
        # 5. Generate locations
        locations_data = await dynamic_lore.generate_locations(environment_desc, factions_data)
        
        # 6. Generate quest hooks
        quests_data = await dynamic_lore.generate_quest_hooks(factions_data, locations_data)
        
        # 7. Generate languages through our regional culture system
        languages = await self.culture_system.generate_languages(run_ctx, count=5)
        
        # 8. Generate religious pantheons and distribute them
        await self.religion_system.initialize_tables()
        religious_data = await self.religion_system.distribute_religions(run_ctx)
        
        # 9. Generate conflicts and domestic issues
        await self.conflict_system.initialize_tables()
        conflicts = await self.conflict_system.generate_initial_conflicts(run_ctx, count=3)
        
        # 10. For each nation, generate cultural norms and etiquette
        nations = await self.geopolitical_manager.get_all_nations(run_ctx)
        nation_cultures = []
        
        for nation in nations:
            # Generate cultural norms
            norms = await self.culture_system.generate_cultural_norms(run_ctx, nation["id"])
            
            # Generate etiquette
            etiquette = await self.culture_system.generate_etiquette(run_ctx, nation["id"])
            
            # Generate domestic issues
            issues = await self.conflict_system.generate_domestic_issues(run_ctx, nation["id"])
            
            nation_cultures.append({
                "nation_id": nation["id"],
                "name": nation["name"],
                "cultural_norms": norms,
                "etiquette": etiquette,
                "domestic_issues": issues
            })
        
        # Combine all results
        complete_lore = {
            "world_lore": foundation_data,
            "factions": factions_data,
            "cultural_elements": cultural_data,
            "historical_events": historical_data,
            "locations": locations_data,
            "quests": quests_data,
            "languages": languages,
            "religions": religious_data,
            "conflicts": conflicts,
            "nation_cultures": nation_cultures
        }
        
        return complete_lore
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="handle_narrative_event",
        action_description="Handling narrative event impacts",
        id_from_context=lambda ctx: "matriarchal_lore_system"
    )
    async def handle_narrative_event(
        self, 
        ctx,
        event_description: str,
        affected_lore_ids: List[str] = None,
        player_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Handle impacts of a narrative event on the world
        
        Args:
            event_description: Description of the event that occurred
            affected_lore_ids: Optional list of specifically affected lore IDs
            player_data: Optional player character data
            
        Returns:
            Dictionary with all updates applied
        """
        # Create run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # If no specific lore IDs provided, determine affected elements automatically
        if not affected_lore_ids:
            affected_lore_ids = await self._determine_affected_elements(event_description)
        
        # Fetch the lore elements
        affected_elements = await self._fetch_elements_by_ids(affected_lore_ids)
        
        # Generate updates for these elements
        updates = await self.update_system.generate_lore_updates(
            run_ctx,
            affected_elements=affected_elements,
            event_description=event_description,
            player_character=player_data
        )
        
        # Apply the updates to the database
        await self._apply_lore_updates(updates)
        
        # Check if the event should affect conflicts or domestic issues
        event_impact = await self._calculate_event_impact(event_description)
        
        # If significant impact, evolve conflicts and issues
        if event_impact > 6:
            # Evolve conflicts and domestic issues
            evolution_results = await self.conflict_system.evolve_all_conflicts(run_ctx, days_passed=7)
            
            # Add these results to the updates
            updates.append({
                "type": "conflict_evolution",
                "results": evolution_results
            })
        
        return {
            "event": event_description,
            "updates": updates,
            "update_count": len(updates),
            "event_impact": event_impact
        }
    
    async def _determine_affected_elements(self, event_description: str) -> List[str]:
        """Determine which elements would be affected by this event"""
        # This would use NLP or keyword matching to find relevant elements
        # For now, we'll use a placeholder implementation
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Simple keyword based search
                words = re.findall(r'\b\w+\b', event_description.lower())
                significant_words = [w for w in words if len(w) > 3]
                
                if not significant_words:
                    # Fallback to get some random important elements
                    elements = await conn.fetch("""
                        SELECT lore_id FROM LoreElements
                        WHERE importance > 7
                        LIMIT 3
                    """)
                    return [e['lore_id'] for e in elements]
                
                # Search for elements matching keywords
                placeholders = ', '.join(f'${i+1}' for i in range(len(significant_words)))
                query = f"""
                    SELECT DISTINCT lore_id FROM LoreElements
                    WHERE (name ILIKE ANY(ARRAY[{placeholders}]) 
                        OR description ILIKE ANY(ARRAY[{placeholders}]))
                    LIMIT 5
                """
                
                search_terms = [f'%{word}%' for word in significant_words] * 2
                elements = await conn.fetch(query, *search_terms)
                
                return [e['lore_id'] for e in elements]
    
    async def _fetch_elements_by_ids(self, element_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch lore elements by their IDs"""
        if not element_ids:
            return []
            
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                elements = await conn.fetch("""
                    SELECT lore_id, name, lore_type, description
                    FROM LoreElements
                    WHERE lore_id = ANY($1)
                """, element_ids)
                
                return [dict(elem) for elem in elements]
    
    async def _apply_lore_updates(self, updates: List[Dict[str, Any]]) -> None:
        """Apply updates to the lore database"""
        if not updates:
            return
            
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                async with conn.transaction():
                    for update in updates:
                        if 'is_cascade_update' in update or 'lore_id' not in update:
                            continue  # Skip cascade updates or invalid updates
                            
                        # Update the element description
                        await conn.execute("""
                            UPDATE LoreElements 
                            SET description = $1
                            WHERE lore_id = $2
                        """, update['new_description'], update['lore_id'])
                        
                        # Add update record
                        await conn.execute("""
                            INSERT INTO LoreUpdates (
                                lore_id, old_description, new_description, 
                                update_reason, impact_level, timestamp
                            ) VALUES ($1, $2, $3, $4, $5, $6)
                        """, 
                        update['lore_id'],
                        update['old_description'],
                        update['new_description'],
                        update['update_reason'],
                        update['impact_level'],
                        update['timestamp'])
                        
                        # Update any type-specific tables
                        if update['lore_type'] == 'character':
                            # Example of character-specific updates
                            char_dev = update.get('character_development', {})
                            if char_dev:
                                await conn.execute("""
                                    UPDATE Characters
                                    SET confidence = $1, resolve = $2, ambition = $3
                                    WHERE character_id = $4
                                """,
                                char_dev.get('confidence', 5),
                                char_dev.get('resolve', 5),
                                char_dev.get('ambition', 5),
                                update['lore_id'])
    
    async def _calculate_event_impact(self, event_description: str) -> int:
        """Calculate the general impact level of an event"""
        # This is a simple keyword-based analysis
        # In a full implementation, this would use NLP
        
        high_impact = ['war', 'death', 'revolution', 'disaster', 'coronation', 'marriage', 'birth', 'conquest']
        medium_impact = ['conflict', 'dispute', 'challenge', 'treaty', 'alliance', 'ceremony', 'festival']
        low_impact = ['meeting', 'conversation', 'journey', 'meal', 'performance', 'minor']
        
        words = set(event_description.lower().split())
        
        high_count = sum(1 for word in high_impact if word in words)
        medium_count = sum(1 for word in medium_impact if word in words)
        low_count = sum(1 for word in low_impact if word in words)
        
        # Calculate base impact
        if high_count > 0:
            return 8 + min(high_count, 2)  # Max 10
        elif medium_count > 0:
            return 5 + min(medium_count, 2)  # Max 7
        elif low_count > 0:
            return 2 + min(low_count, 2)  # Max 4
        else:
            return 5  # Default medium impact


# -------------------------------------------------
# MATRIARCHAL THEMING UTILITIES
# -------------------------------------------------

# A dictionary of regex patterns to replacement strings for basic feminization
# \b ensures "god" won't become "Goddess" inside words like "good"
_FEMDOM_WORD_MAP = {
    r"\bgod\b": "Goddess",
    r"\bgods\b": "Goddesses",
    r"\bgodhood\b": "Goddesshood",
    r"\bking\b": "Queen",
    r"\bkings\b": "Queens",
    r"\bkingdom\b": "Queendom",
    r"\bprince\b": "princess",
    r"\bprinces\b": "princesses",
    r"\bfather\b": "mother",
    r"\bfathers\b": "mothers",
    r"\bhim\b": "her",
    r"\bhis\b": "her",
    r"\bhe\b": "she",
    r"\blord\b": "lady",
    r"\blords\b": "ladies",
    r"\bman\b": "woman",
    r"\bmen\b": "women",
}

# Random synonyms for a supreme feminine figure
_GODDESS_SYNONYMS = [
    "Supreme Goddess",
    "High Empress",
    "Great Matriarch",
    "Divine Mother",
    "Infinite Mistress of Creation",
]

# Default emphasis level for matriarchal theming (1=low, 3=very high)
_DEFAULT_EMPHASIS_LEVEL = 2


def _apply_basic_replacements(text: str) -> str:
    """
    Runs a set of regex-based replacements to feminize words/phrases.
    Respects case; if the original word is capitalized, keep it capitalized.
    """
    result = text

    for pattern_str, replacement_str in _FEMDOM_WORD_MAP.items():
        pattern = re.compile(pattern_str, re.IGNORECASE)

        def _replacement_func(match):
            original = match.group(0)
            # If the original word starts with uppercase, we uppercase the replacement's first letter
            if original and original[0].isupper():
                return replacement_str.capitalize()
            return replacement_str

        result = pattern.sub(_replacement_func, result)

    return result


def _ensure_goddess_reference(text: str) -> str:
    """
    If there's no mention of 'Goddess' or a similar figure, insert a default reference
    to a supreme feminine force at the end of the text.
    """
    if not re.search(r"(goddess|divine mother|matriarch|empress of creation)", text, re.IGNORECASE):
        chosen_title = random.choice(_GODDESS_SYNONYMS)
        insertion = (
            f"\n\nAt the cosmic center stands {chosen_title}, "
            "the eternal wellspring of existence. Her dominion weaves reality itself."
        )
        text += insertion

    return text


def _inject_contextual_lore(text: str, insertion: str, label: str = "", append_if_not_found: bool = True) -> str:
    """
    Insert 'insertion' text after a specific label or heading (case-insensitive).
    If label not found and append_if_not_found is True, just append at the end.
    """
    if not label:
        # No label provided, always append at the end
        return text.strip() + "\n\n" + insertion

    pattern = re.compile(rf"({re.escape(label)})", re.IGNORECASE)
    match = pattern.search(text)

    if match:
        # Insert after the matched label
        idx = match.end()
        # Insert a blank line before the snippet
        return text[:idx] + "\n\n" + insertion + text[idx:]
    else:
        if append_if_not_found:
            return text.strip() + "\n\n" + insertion
        else:
            # Return unchanged if not found and we don't want to append
            return text


def _emphasize_feminine_power(text: str, emphasis_level: int = _DEFAULT_EMPHASIS_LEVEL) -> str:
    """
    Add extra lines emphasizing matriarchal/feminine power, depending on emphasis_level.
    1 = minimal emphasis
    2 = moderate emphasis
    3 = strong emphasis
    """
    if emphasis_level >= 2:
        text += (
            "\n\nThroughout every realm, the ascendancy of womanhood underpins all aspects of life, "
            "unquestioned and unassailable."
        )
    if emphasis_level >= 3:
        text += (
            "\n\nNo force—mortal or divine—can transcend the supremacy of the feminine essence, "
            "the primal wellspring of creation itself."
        )
    return text


class MatriarchalThemingUtils:
    """
    Utility class for applying matriarchal theming to different types of lore content.
    """

    @staticmethod
    def feminize_cosmology(cosmology: str) -> str:
        """
        Create a comprehensive, feminized version of the cosmology, ensuring references
        to gods, powers, and origins are dominated by feminine authority.
        """
        # 1) Basic replacements (god->Goddess, king->Queen, etc.)
        result = _apply_basic_replacements(cosmology)

        # 2) Ensure we have at least one reference to a goddess or matriarchal figure
        result = _ensure_goddess_reference(result)

        # 3) Insert text about the 'Feminine Principle' after a "COSMOLOGY" heading, if present
        feminine_principle = (
            "At the heart of all creation is the Feminine Principle, the source of all life and power. "
            "The cosmos itself is understood as fundamentally feminine in nature, "
            "with any masculine elements serving and supporting the greater feminine whole."
        )
        result = _inject_contextual_lore(result, feminine_principle, label="COSMOLOGY")

        # 4) Optionally emphasize the matriarchal tone
        result = _emphasize_feminine_power(result, _DEFAULT_EMPHASIS_LEVEL)

        return result

    @staticmethod
    def gender_magic_system(magic_system: str) -> str:
        """
        Apply gendered dynamics to the magic system, making feminine energies
        paramount and male magic supportive or secondary.
        """
        # 1) Feminize references
        result = _apply_basic_replacements(magic_system)

        # 2) Ensure mention of a goddess figure for continuity
        result = _ensure_goddess_reference(result)

        # 3) Insert advanced lore about women's superior magical authority
        gendered_magic = (
            "The flow and expression of magical energies reflect the natural order of feminine dominance. "
            "Women typically possess greater innate magical potential and exclusive rights to the highest mysteries. "
            "Men specializing in arcane arts often excel in supportive, protective, or enhancing magics, "
            "operating in service to more powerful feminine traditions. "
            "Only when guided by a woman's touch do the most potent rituals fully manifest."
        )
        result = _inject_contextual_lore(result, gendered_magic, label="MAGIC")

        # 4) Emphasize
        result = _emphasize_feminine_power(result, _DEFAULT_EMPHASIS_LEVEL)

        return result

    @staticmethod
    def matriarchalize_history(history: str) -> str:
        """
        Overhaul historical accounts so that women have always held power,
        shaping the course of civilization through matriarchal leadership.
        """
        # 1) Feminize references
        result = _apply_basic_replacements(history)

        # 2) Ensure mention of goddess figure
        result = _ensure_goddess_reference(result)

        # 3) Insert matriarchal historical note
        matriarchal_history = (
            "Throughout recorded chronicles, women have held the reins of power. "
            "Great Empresses, Matriarchs, and female rulers have guided civilizations toward prosperity. "
            "Though conflicts and rebellions against this natural order have arisen, "
            "the unshakable principle of feminine dominance remains the bedrock of history."
        )
        result = _inject_contextual_lore(result, matriarchal_history, label="HISTORY")

        # 4) Emphasize
        result = _emphasize_feminine_power(result, _DEFAULT_EMPHASIS_LEVEL)

        return result

    @staticmethod
    def feminize_calendar(calendar_system: str) -> str:
        """
        Make the calendar reflect significant feminine milestones, lunar cycles,
        and holidays honoring matriarchal power and achievements.
        """
        # 1) Feminize references
        result = _apply_basic_replacements(calendar_system)

        # 2) Ensure mention of a goddess figure
        result = _ensure_goddess_reference(result)

        # 3) Insert note about matriarchal calendar features
        feminine_calendar = (
            "The calendar marks vital dates in feminine history, aligning festivals and holy days "
            "with lunar cycles and the reigns of legendary Empresses. Major celebrations honor "
            "the cyclical power of womanhood, reflecting its role in birth, renewal, and creation."
        )
        result = _inject_contextual_lore(result, feminine_calendar, label="CALENDAR")

        # 4) Emphasize
        result = _emphasize_feminine_power(result, _DEFAULT_EMPHASIS_LEVEL)

        return result

    @staticmethod
    def apply_matriarchal_theme(lore_type: str, content: str) -> str:
        """
        Apply appropriate matriarchal theming based on lore type.
        
        Args:
            lore_type: Type of lore content ('cosmology', 'magic_system', 'history', etc.)
            content: Original content to modify
            
        Returns:
            Modified content with matriarchal theming
        """
        if lore_type == 'cosmology':
            return MatriarchalThemingUtils.feminize_cosmology(content)
        elif lore_type == 'magic_system':
            return MatriarchalThemingUtils.gender_magic_system(content)
        elif lore_type == 'history' or lore_type == 'world_history':
            return MatriarchalThemingUtils.matriarchalize_history(content)
        elif lore_type == 'calendar':
            return MatriarchalThemingUtils.feminize_calendar(content)
        else:
            # For other types, just do basic replacements and add some emphasis
            result = _apply_basic_replacements(content)
            return _emphasize_feminine_power(result, 1)  # Light emphasis for misc content


# -------------------------------------------------
# REGIONAL CULTURE SYSTEM
# -------------------------------------------------

class RegionalCultureSystem:
    """
    Manages culturally specific norms, customs, manners, and languages
    across different regions and nations.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.lore_manager = LoreManager(user_id, conversation_id)
        self.geopolitical_manager = GeopoliticalSystemManager(user_id, conversation_id)
        self.governor = None
        
    async def initialize_governance(self):
        """Initialize Nyx governance connection"""
        if not self.governor:
            self.governor = await get_central_governance(self.user_id, self.conversation_id)
        return self.governor
        
    async def initialize_tables(self):
        """Ensure regional culture tables exist"""
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Check if Languages table exists
                languages_exist = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'languages'
                    );
                """)
                
                if not languages_exist:
                    # Create the table
                    await conn.execute("""
                        CREATE TABLE Languages (
                            id SERIAL PRIMARY KEY,
                            name TEXT NOT NULL,
                            language_family TEXT,
                            description TEXT NOT NULL,
                            writing_system TEXT,
                            primary_regions INTEGER[], -- Nation IDs where primarily spoken
                            minority_regions INTEGER[], -- Nation IDs where spoken by minorities
                            formality_levels TEXT[], -- Different levels of formality
                            common_phrases JSONB, -- Basic phrases in this language
                            difficulty INTEGER CHECK (difficulty BETWEEN 1 AND 10),
                            relation_to_power TEXT, -- How language relates to power structures
                            dialects JSONB, -- Regional variations
                            embedding VECTOR(1536)
                        );
                    """)
                    
                    # Create index
                    await conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_languages_embedding 
                        ON Languages USING ivfflat (embedding vector_cosine_ops);
                    """)
                    
                    logging.info("Languages table created")
                
                # Check if CulturalNorms table exists
                norms_exist = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'culturalnorms'
                    );
                """)
                
                if not norms_exist:
                    # Create the table
                    await conn.execute("""
                        CREATE TABLE CulturalNorms (
                            id SERIAL PRIMARY KEY,
                            nation_id INTEGER NOT NULL, -- Nation this applies to
                            category TEXT NOT NULL, -- greeting, dining, authority, gift, etc.
                            description TEXT NOT NULL, -- Detailed description
                            formality_level TEXT, -- casual, formal, ceremonial
                            gender_specific BOOLEAN DEFAULT FALSE, -- If norm differs by gender
                            female_variation TEXT, -- Female-specific version if applicable
                            male_variation TEXT, -- Male-specific version if applicable
                            taboo_level INTEGER CHECK (taboo_level BETWEEN 0 AND 10), -- How taboo breaking this is
                            consequence TEXT, -- Consequence of breaking norm
                            regional_variations JSONB, -- Variations within the nation
                            embedding VECTOR(1536),
                            FOREIGN KEY (nation_id) REFERENCES Nations(id) ON DELETE CASCADE
                        );
                    """)
                    
                    # Create index
                    await conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_culturalnorms_embedding 
                        ON CulturalNorms USING ivfflat (embedding vector_cosine_ops);
                        
                        CREATE INDEX IF NOT EXISTS idx_culturalnorms_nation
                        ON CulturalNorms(nation_id);
                    """)
                    
                    logging.info("CulturalNorms table created")
                
                # Check if Etiquette table exists
                etiquette_exist = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'etiquette'
                    );
                """)
                
                if not etiquette_exist:
                    # Create the table
                    await conn.execute("""
                        CREATE TABLE Etiquette (
                            id SERIAL PRIMARY KEY,
                            nation_id INTEGER NOT NULL, -- Nation this applies to
                            context TEXT NOT NULL, -- Context (court, public, private, etc.)
                            title_system TEXT, -- How titles work
                            greeting_ritual TEXT, -- How people greet each other
                            body_language TEXT, -- Expected body language
                            eye_contact TEXT, -- Eye contact norms
                            distance_norms TEXT, -- Personal space norms
                            gift_giving TEXT, -- Gift-giving norms
                            dining_etiquette TEXT, -- Table manners
                            power_display TEXT, -- How power is displayed
                            respect_indicators TEXT, -- How respect is shown
                            gender_distinctions TEXT, -- How gender impacts etiquette
                            taboos TEXT[], -- Things never to do
                            embedding VECTOR(1536),
                            FOREIGN KEY (nation_id) REFERENCES Nations(id) ON DELETE CASCADE
                        );
                    """)
                    
                    # Create index
                    await conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_etiquette_embedding 
                        ON Etiquette USING ivfflat (embedding vector_cosine_ops);
                        
                        CREATE INDEX IF NOT EXISTS idx_etiquette_nation
                        ON Etiquette(nation_id);
                    """)
                    
                    logging.info("Etiquette table created")
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_languages",
        action_description="Generating languages for the world",
        id_from_context=lambda ctx: "regional_culture_system"
    )
    async def generate_languages(self, ctx, count: int = 5) -> List[Dict[str, Any]]:
        """
        Generate languages for the world with governance oversight.
        
        Args:
            count: Number of languages to generate
            
        Returns:
            List of generated languages
        """
        # Create the run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Get nations for context
        nations = await self.geopolitical_manager.get_all_nations(run_ctx)
        
        if not nations:
            return []
        
        # Create agent for language generation
        language_agent = Agent(
            name="LanguageGenerationAgent",
            instructions="You create realistic languages for fantasy worlds with matriarchal power structures.",
            model="gpt-4.1-nano"
        )
        
        languages = []
        for i in range(count):
            # Determine language distribution
            # For simplicity: create some major languages and some minor ones
            is_major = i < count // 2
            
            if is_major:
                # Major language used by multiple nations
                primary_nations = random.sample(nations, min(3, len(nations)))
                minority_nations = random.sample([n for n in nations if n not in primary_nations], 
                                               min(2, len(nations) - len(primary_nations)))
            else:
                # More localized language
                primary_nations = random.sample(nations, 1)
                minority_nations = random.sample([n for n in nations if n not in primary_nations], 
                                               min(2, len(nations) - 1))
            
            # Create prompt for language generation
            prompt = f"""
            Generate a detailed language for a fantasy world with matriarchal power structures.
            
            PRIMARY NATIONS:
            {json.dumps([{
                "name": n.get("name", "Unknown"),
                "government_type": n.get("government_type", "Unknown"),
                "matriarchy_level": n.get("matriarchy_level", 5)
            } for n in primary_nations], indent=2)}
            
            MINORITY NATIONS:
            {json.dumps([{
                "name": n.get("name", "Unknown"),
                "government_type": n.get("government_type", "Unknown"),
                "matriarchy_level": n.get("matriarchy_level", 5)
            } for n in minority_nations], indent=2)}
            
            Create a {'major regional' if is_major else 'localized'} language that:
            1. Reflects the matriarchal power structures of the world
            2. Has realistic features and complexity
            3. Includes information about how formality and power are expressed
            4. Includes feminine-dominant linguistic features
            5. Has some common phrases or expressions
            
            Return a JSON object with:
            - name: Name of the language
            - language_family: Linguistic family it belongs to
            - description: Detailed description of the language
            - writing_system: How it's written (if at all)
            - formality_levels: Array of formality levels (from casual to formal)
            - common_phrases: Object with key phrases (greeting, farewell, etc.)
            - difficulty: How hard it is to learn (1-10)
            - relation_to_power: How the language reflects power dynamics
            - dialects: Object with different regional dialects
            """
            
            # Get response from agent
            result = await Runner.run(language_agent, prompt, context=run_ctx.context)
            
            try:
                # Parse JSON response
                language_data = json.loads(result.final_output)
                
                # Ensure required fields
                if not all(k in language_data for k in ["name", "description"]):
                    continue
                
                # Add nation IDs
                language_data["primary_regions"] = [n["id"] for n in primary_nations]
                language_data["minority_regions"] = [n["id"] for n in minority_nations]
                
                # Generate embedding
                embedding_text = f"{language_data['name']} {language_data['description']}"
                embedding = await generate_embedding(embedding_text)
                
                # Store in database
                async with self.lore_manager.get_connection_pool() as pool:
                    async with pool.acquire() as conn:
                        language_id = await conn.fetchval("""
                            INSERT INTO Languages (
                                name, language_family, description, writing_system,
                                primary_regions, minority_regions, formality_levels,
                                common_phrases, difficulty, relation_to_power, dialects, embedding
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                            RETURNING id
                        """,
                        language_data.get("name"),
                        language_data.get("language_family"),
                        language_data.get("description"),
                        language_data.get("writing_system"),
                        language_data.get("primary_regions"),
                        language_data.get("minority_regions"),
                        language_data.get("formality_levels"),
                        json.dumps(language_data.get("common_phrases", {})),
                        language_data.get("difficulty", 5),
                        language_data.get("relation_to_power"),
                        json.dumps(language_data.get("dialects", {})),
                        embedding)
                        
                        language_data["id"] = language_id
                        languages.append(language_data)
            
            except Exception as e:
                logging.error(f"Error generating language: {e}")
        
        return languages
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_cultural_norms",
        action_description="Generating cultural norms for nation {nation_id}",
        id_from_context=lambda ctx: "regional_culture_system"
    )
    async def generate_cultural_norms(self, ctx, nation_id: int) -> List[Dict[str, Any]]:
        """
        Generate cultural norms for a specific nation with governance oversight.
        
        Args:
            nation_id: ID of the nation
            
        Returns:
            List of generated cultural norms
        """
        # Create the run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Get nation details
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                nation = await conn.fetchrow("""
                    SELECT id, name, government_type, matriarchy_level, cultural_traits
                    FROM Nations
                    WHERE id = $1
                """, nation_id)
                
                if not nation:
                    return []
                
                nation_data = dict(nation)
        
        # Create agent for cultural norm generation
        norm_agent = Agent(
            name="CulturalNormAgent",
            instructions="You create cultural norms for fantasy nations with matriarchal power structures.",
            model="gpt-4.1-nano"
        )
        
        # Categories of norms to generate
        categories = [
            "greeting", "dining", "authority", "gift_giving", "personal_space", 
            "gender_relations", "age_relations", "public_behavior", "private_behavior", 
            "business_conduct", "religious_practice"
        ]
        
        norms = []
        for category in categories:
            # Create prompt for norm generation
            prompt = f"""
            Generate cultural norms about {category} for this nation:
            
            NATION:
            {json.dumps(nation_data, indent=2)}
            
            Create detailed cultural norms that:
            1. Reflect the nation's matriarchy level ({nation_data.get("matriarchy_level", 5)}/10)
            2. Feel authentic and consistent with the nation's traits
            3. Include both dos and don'ts
            4. Specify if norms differ by gender
            
            Return a JSON object with:
            - category: "{category}"
            - description: Detailed description of the norm
            - formality_level: Level of formality (casual, formal, ceremonial)
            - gender_specific: Boolean - whether norm differs by gender
            - female_variation: Female-specific version if applicable
            - male_variation: Male-specific version if applicable
            - taboo_level: How taboo breaking this is (0-10)
            - consequence: Consequence of breaking norm
            - regional_variations: Object with any variations within the nation
            """
            
            # Get response from agent
            result = await Runner.run(norm_agent, prompt, context=run_ctx.context)
            
            try:
                # Parse JSON response
                norm_data = json.loads(result.final_output)
                
                # Ensure required fields
                if not all(k in norm_data for k in ["category", "description"]):
                    continue
                
                # Generate embedding
                embedding_text = f"{category} {norm_data['description']}"
                embedding = await generate_embedding(embedding_text)
                
                # Store in database
                async with self.lore_manager.get_connection_pool() as pool:
                    async with pool.acquire() as conn:
                        norm_id = await conn.fetchval("""
                            INSERT INTO CulturalNorms (
                                nation_id, category, description, formality_level,
                                gender_specific, female_variation, male_variation,
                                taboo_level, consequence, regional_variations, embedding
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                            RETURNING id
                        """,
                        nation_id,
                        norm_data.get("category"),
                        norm_data.get("description"),
                        norm_data.get("formality_level"),
                        norm_data.get("gender_specific", False),
                        norm_data.get("female_variation"),
                        norm_data.get("male_variation"),
                        norm_data.get("taboo_level", 5),
                        norm_data.get("consequence"),
                        json.dumps(norm_data.get("regional_variations", {})),
                        embedding)
                        
                        norm_data["id"] = norm_id
                        norm_data["nation_id"] = nation_id
                        norms.append(norm_data)
            
            except Exception as e:
                logging.error(f"Error generating cultural norm for category {category}: {e}")
        
        return norms

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_etiquette",
        action_description="Generating etiquette for nation {nation_id}",
        id_from_context=lambda ctx: "regional_culture_system"
    )
    async def generate_etiquette(self, ctx, nation_id: int) -> List[Dict[str, Any]]:
        """
        Generate etiquette systems for a specific nation with governance oversight.
        
        Args:
            nation_id: ID of the nation
            
        Returns:
            List of generated etiquette systems
        """
        # Create the run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Get nation details
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                nation = await conn.fetchrow("""
                    SELECT id, name, government_type, matriarchy_level, cultural_traits
                    FROM Nations
                    WHERE id = $1
                """, nation_id)
                
                if not nation:
                    return []
                
                nation_data = dict(nation)
        
        # Create agent for etiquette generation
        etiquette_agent = Agent(
            name="EtiquetteAgent",
            instructions="You create etiquette systems for fantasy nations with matriarchal power structures.",
            model="gpt-4.1-nano"
        )
        
        # Contexts for etiquette
        contexts = ["court", "noble", "public", "private", "religious", "business"]
        
        etiquette_systems = []
        for context in contexts:
            # Create prompt for etiquette generation
            prompt = f"""
            Generate an etiquette system for {context} contexts in this nation:
            
            NATION:
            {json.dumps(nation_data, indent=2)}
            
            Create a detailed etiquette system that:
            1. Reflects the nation's matriarchy level ({nation_data.get("matriarchy_level", 5)}/10)
            2. Provides clear rules for behavior in {context} settings
            3. Specifies how power and respect are shown
            4. Includes gender-specific elements that reflect feminine authority
            
            Return a JSON object with:
            - context: "{context}"
            - title_system: How titles and forms of address work
            - greeting_ritual: How people greet each other
            - body_language: Expected body language
            - eye_contact: Eye contact norms
            - distance_norms: Personal space norms
            - gift_giving: Gift-giving norms
            - dining_etiquette: Table manners
            - power_display: How power is displayed
            - respect_indicators: How respect is shown
            - gender_distinctions: How gender impacts etiquette
            - taboos: Array of things never to do
            """
            
            # Get response from agent
            result = await Runner.run(etiquette_agent, prompt, context=run_ctx.context)
            
            try:
                # Parse JSON response
                etiquette_data = json.loads(result.final_output)
                
                # Ensure required fields
                if not all(k in etiquette_data for k in ["context", "greeting_ritual"]):
                    continue
                
                # Generate embedding
                embedding_text = f"{context} etiquette {etiquette_data['greeting_ritual']} {etiquette_data.get('respect_indicators', '')}"
                embedding = await generate_embedding(embedding_text)
                
                # Store in database
                async with self.lore_manager.get_connection_pool() as pool:
                    async with pool.acquire() as conn:
                        etiquette_id = await conn.fetchval("""
                            INSERT INTO Etiquette (
                                nation_id, context, title_system, greeting_ritual,
                                body_language, eye_contact, distance_norms, gift_giving,
                                dining_etiquette, power_display, respect_indicators,
                                gender_distinctions, taboos, embedding
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                            RETURNING id
                        """,
                        nation_id,
                        etiquette_data.get("context"),
                        etiquette_data.get("title_system"),
                        etiquette_data.get("greeting_ritual"),
                        etiquette_data.get("body_language"),
                        etiquette_data.get("eye_contact"),
                        etiquette_data.get("distance_norms"),
                        etiquette_data.get("gift_giving"),
                        etiquette_data.get("dining_etiquette"),
                        etiquette_data.get("power_display"),
                        etiquette_data.get("respect_indicators"),
                        etiquette_data.get("gender_distinctions"),
                        etiquette_data.get("taboos", []),
                        embedding)
                        
                        etiquette_data["id"] = etiquette_id
                        etiquette_data["nation_id"] = nation_id
                        etiquette_systems.append(etiquette_data)
            
            except Exception as e:
                logging.error(f"Error generating etiquette for context {context}: {e}")
        
        return etiquette_systems

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="get_nation_culture",
        action_description="Getting cultural information for nation {nation_id}",
        id_from_context=lambda ctx: "regional_culture_system"
    )
    async def get_nation_culture(self, ctx, nation_id: int) -> Dict[str, Any]:
        """
        Get comprehensive cultural information about a nation with governance oversight.
        
        Args:
            nation_id: ID of the nation
            
        Returns:
            Dictionary with nation's cultural information
        """
        # Check cache first
        cache_key = f"nation_culture_{nation_id}_{self.user_id}_{self.conversation_id}"
        cached = CULTURE_CACHE.get(cache_key)
        if cached:
            return cached
        
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get nation details
                nation = await conn.fetchrow("""
                    SELECT id, name, government_type, matriarchy_level, cultural_traits
                    FROM Nations
                    WHERE id = $1
                """, nation_id)
                
                if not nation:
                    return {"error": "Nation not found"}
                
                # Get languages
                languages = await conn.fetch("""
                    SELECT id, name, description, writing_system, formality_levels
                    FROM Languages
                    WHERE $1 = ANY(primary_regions) OR $1 = ANY(minority_regions)
                """, nation_id)
                
                # Get cultural norms
                norms = await conn.fetch("""
                    SELECT id, category, description, formality_level, gender_specific,
                           female_variation, male_variation, taboo_level, consequence
                    FROM CulturalNorms
                    WHERE nation_id = $1
                """, nation_id)
                
                # Get etiquette
                etiquette = await conn.fetch("""
                    SELECT id, context, title_system, greeting_ritual, power_display,
                           respect_indicators, gender_distinctions, taboos
                    FROM Etiquette
                    WHERE nation_id = $1
                """, nation_id)
                
                # Compile result
                result = {
                    "nation": dict(nation),
                    "languages": {
                        "primary": [dict(lang) for lang in languages if nation_id in lang["primary_regions"]],
                        "minority": [dict(lang) for lang in languages if nation_id in lang["minority_regions"]]
                    },
                    "cultural_norms": [dict(norm) for norm in norms],
                    "etiquette": [dict(etiq) for etiq in etiquette]
                }
                
                # Cache the result
                CULTURE_CACHE.set(cache_key, result)
                
                return result


# -------------------------------------------------
# NATIONAL CONFLICT SYSTEM
# -------------------------------------------------

class NationalConflictSystem:
    """
    System for managing, generating, and evolving national and international
    conflicts that serve as background elements in the world.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.lore_manager = LoreManager(user_id, conversation_id)
        self.geopolitical_manager = GeopoliticalSystemManager(user_id, conversation_id)
        self.governor = None
        
    async def initialize_governance(self):
        """Initialize Nyx governance connection"""
        if not self.governor:
            self.governor = await get_central_governance(self.user_id, self.conversation_id)
        return self.governor
        
    async def initialize_tables(self):
        """Ensure conflict system tables exist"""
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Check if Conflicts table exists
                conflicts_exist = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'nationalconflicts'
                    );
                """)
                
                if not conflicts_exist:
                    # Create the table
                    await conn.execute("""
                        CREATE TABLE NationalConflicts (
                            id SERIAL PRIMARY KEY,
                            name TEXT NOT NULL,
                            conflict_type TEXT NOT NULL, -- war, trade_dispute, diplomatic_tension, etc.
                            description TEXT NOT NULL,
                            severity INTEGER CHECK (severity BETWEEN 1 AND 10),
                            status TEXT NOT NULL, -- active, resolved, escalating, de-escalating
                            start_date TEXT NOT NULL,
                            end_date TEXT, -- NULL if ongoing
                            involved_nations INTEGER[], -- IDs of nations involved
                            primary_aggressor INTEGER, -- Nation ID of aggressor
                            primary_defender INTEGER, -- Nation ID of defender
                            current_casualties TEXT, -- Description of casualties so far
                            economic_impact TEXT, -- Description of economic impact
                            diplomatic_consequences TEXT, -- Description of diplomatic fallout
                            public_opinion JSONB, -- Public opinion in different nations
                            recent_developments TEXT[], -- Recent events in the conflict
                            potential_resolution TEXT, -- Potential ways it might end
                            embedding VECTOR(1536)
                        );
                    """)
                    
                    # Create index
                    await conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_nationalconflicts_embedding 
                        ON NationalConflicts USING ivfflat (embedding vector_cosine_ops);
                    """)
                    
                    logging.info("NationalConflicts table created")
                
                # Check if ConflictNews table exists
                news_exist = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'conflictnews'
                    );
                """)
                
                if not news_exist:
                    # Create the table
                    await conn.execute("""
                        CREATE TABLE ConflictNews (
                            id SERIAL PRIMARY KEY,
                            conflict_id INTEGER NOT NULL,
                            headline TEXT NOT NULL,
                            content TEXT NOT NULL,
                            publication_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            source_nation INTEGER, -- Nation ID where this news originated
                            bias TEXT, -- pro_aggressor, pro_defender, neutral
                            embedding VECTOR(1536),
                            FOREIGN KEY (conflict_id) REFERENCES NationalConflicts(id) ON DELETE CASCADE
                        );
                    """)
                    
                    # Create index
                    await conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_conflictnews_embedding 
                        ON ConflictNews USING ivfflat (embedding vector_cosine_ops);
                    """)
                    
                    logging.info("ConflictNews table created")

                # Check if DomesticIssues table exists
                domestic_exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'domesticissues'
                    );
                """)
                
                if not domestic_exists:
                    # Create the table
                    await conn.execute("""
                        CREATE TABLE DomesticIssues (
                            id SERIAL PRIMARY KEY,
                            nation_id INTEGER NOT NULL,
                            name TEXT NOT NULL,
                            issue_type TEXT NOT NULL, -- civil_rights, political_controversy, economic_crisis, etc.
                            description TEXT NOT NULL,
                            severity INTEGER CHECK (severity BETWEEN 1 AND 10),
                            status TEXT NOT NULL, -- emerging, active, waning, resolved
                            start_date TEXT NOT NULL,
                            end_date TEXT, -- NULL if ongoing
                            supporting_factions TEXT[], -- Groups supporting one side
                            opposing_factions TEXT[], -- Groups opposing
                            neutral_factions TEXT[], -- Groups remaining neutral
                            affected_demographics TEXT[], -- Demographics most affected
                            public_opinion JSONB, -- Opinion distribution
                            government_response TEXT, -- How the government is responding
                            recent_developments TEXT[], -- Recent events in this issue
                            political_impact TEXT, -- Impact on political landscape
                            social_impact TEXT, -- Impact on society
                            economic_impact TEXT, -- Economic consequences
                            potential_resolution TEXT, -- Potential ways it might resolve
                            embedding VECTOR(1536),
                            FOREIGN KEY (nation_id) REFERENCES Nations(id) ON DELETE CASCADE
                        );
                    """)
                    
                    # Create index
                    await conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_domesticissues_embedding 
                        ON DomesticIssues USING ivfflat (embedding vector_cosine_ops);
                        
                        CREATE INDEX IF NOT EXISTS idx_domesticissues_nation
                        ON DomesticIssues(nation_id);
                    """)
                    
                    logging.info("DomesticIssues table created")
                
                    # Check if DomesticNews table exists
                    domestic_news_exist = await conn.fetchval("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = 'domesticnews'
                        );
                    """)
                    
                    if not domestic_news_exist:
                        # Create the table
                        await conn.execute("""
                            CREATE TABLE DomesticNews (
                                id SERIAL PRIMARY KEY,
                                issue_id INTEGER NOT NULL,
                                headline TEXT NOT NULL,
                                content TEXT NOT NULL,
                                publication_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                source_faction TEXT, -- Faction perspective
                                bias TEXT, -- supporting, opposing, neutral
                                embedding VECTOR(1536),
                                FOREIGN KEY (issue_id) REFERENCES DomesticIssues(id) ON DELETE CASCADE
                            );
                        """)
                        
                        # Create index
                        await conn.execute("""
                            CREATE INDEX IF NOT EXISTS idx_domesticnews_embedding 
                            ON DomesticNews USING ivfflat (embedding vector_cosine_ops);
                        """)
                        
                        logging.info("DomesticNews table created")

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_domestic_issues",
        action_description="Generating domestic issues for nation {nation_id}",
        id_from_context=lambda ctx: "national_conflict_system"
    )
    async def generate_domestic_issues(self, ctx, nation_id: int, count: int = 2) -> List[Dict[str, Any]]:
        """
        Generate domestic issues for a specific nation with governance oversight.
        
        Args:
            nation_id: ID of the nation
            count: Number of issues to generate
            
        Returns:
            List of generated domestic issues
        """
        # Create the run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Get nation details
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                nation = await conn.fetchrow("""
                    SELECT id, name, government_type, matriarchy_level, cultural_traits
                    FROM Nations
                    WHERE id = $1
                """, nation_id)
                
                if not nation:
                    return []
                
                nation_data = dict(nation)
                
                # Get factions in this nation
                factions = await conn.fetch("""
                    SELECT id, name, type, description, values
                    FROM Factions
                    WHERE $1 = ANY(territory)
                """, nation_data.get("name"))
                
                faction_data = [dict(faction) for faction in factions]
        
        # Create agent for domestic issue generation
        issue_agent = Agent(
            name="DomesticIssueAgent",
            instructions="You create realistic domestic political and social issues for fantasy nations with matriarchal power structures.",
            model="gpt-4.1-nano"
        )
        
        # Determine issue types based on nation characteristics
        issue_types = []
        
        # Higher matriarchy has different issues than lower
        matriarchy_level = nation_data.get("matriarchy_level", 5)
        
        if matriarchy_level >= 8:
            # High matriarchy issues
            issue_types.extend([
                "male_rights_movement", "traditionalist_opposition", "matriarchy_reform", 
                "male_separatism", "gender_hierarchy_legislation"
            ])
        elif matriarchy_level <= 3:
            # Low matriarchy issues
            issue_types.extend([
                "feminist_movement", "equality_legislation", "patriarchal_opposition",
                "female_leadership_controversy", "gender_role_debates"
            ])
        else:
            # Balanced matriarchy issues
            issue_types.extend([
                "gender_balance_debate", "power_sharing_reform", "traditionalist_vs_progressive"
            ])
        
        # Universal issue types
        universal_issues = [
            "economic_crisis", "environmental_disaster", "disease_outbreak",
            "succession_dispute", "religious_controversy", "tax_reform",
            "military_service_debate", "trade_regulation", "education_policy",
            "infrastructure_development", "foreign_policy_shift", "corruption_scandal",
            "resource_scarcity", "technological_change", "constitutional_crisis",
            "land_rights_dispute", "criminal_justice_reform", "public_safety_concerns",
            "media_censorship", "social_services_funding"
        ]
        
        issue_types.extend(universal_issues)
        
        # Generate issues
        issues = []
        selected_types = random.sample(issue_types, min(count, len(issue_types)))
        
        for issue_type in selected_types:
            # Create prompt for the agent
            prompt = f"""
            Generate a domestic political or social issue for this nation:
            
            NATION:
            {json.dumps(nation_data, indent=2)}
            
            FACTIONS:
            {json.dumps(faction_data, indent=2)}
            
            Create a {issue_type} issue that:
            1. Makes sense given the nation's characteristics
            2. Creates realistic societal tension and debate
            3. Involves multiple factions or groups
            4. Considers the matriarchal level of the society ({matriarchy_level}/10)
            
            Return a JSON object with:
            - name: Name of the issue/controversy
            - issue_type: "{issue_type}"
            - description: Detailed description
            - severity: Severity level (1-10)
            - status: Current status (emerging, active, waning, resolved)
            - start_date: When it started (narrative date)
            - supporting_factions: Groups supporting one side
            - opposing_factions: Groups opposing
            - neutral_factions: Groups remaining neutral
            - affected_demographics: Demographics most affected
            - public_opinion: Object describing opinion distribution
            - government_response: How the government is responding
            - recent_developments: Array of recent events in this issue
            - political_impact: Impact on political landscape
            - social_impact: Impact on society
            - economic_impact: Economic consequences
            - potential_resolution: Potential ways it might resolve
            """
            
            # Get response from agent
            result = await Runner.run(issue_agent, prompt, context=run_ctx.context)
            
            try:
                # Parse response
                issue_data = json.loads(result.final_output)
                
                # Ensure required fields exist
                if not all(k in issue_data for k in ["name", "description", "issue_type"]):
                    continue
                
                # Add nation_id
                issue_data["nation_id"] = nation_id
                
                # Generate embedding
                embedding_text = f"{issue_data['name']} {issue_data['description']} {issue_data['issue_type']}"
                embedding = await generate_embedding(embedding_text)
                
                # Store in database
                async with self.lore_manager.get_connection_pool() as pool:
                    async with pool.acquire() as conn:
                        issue_id = await conn.fetchval("""
                            INSERT INTO DomesticIssues (
                                nation_id, name, issue_type, description, severity,
                                status, start_date, supporting_factions, opposing_factions,
                                neutral_factions, affected_demographics, public_opinion,
                                government_response, recent_developments, political_impact,
                                social_impact, economic_impact, potential_resolution, embedding
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)
                            RETURNING id
                        """, 
                        nation_id,
                        issue_data.get("name"), 
                        issue_data.get("issue_type"),
                        issue_data.get("description"),
                        issue_data.get("severity", 5),
                        issue_data.get("status", "active"),
                        issue_data.get("start_date", "Recently"),
                        issue_data.get("supporting_factions", []),
                        issue_data.get("opposing_factions", []),
                        issue_data.get("neutral_factions", []),
                        issue_data.get("affected_demographics", []),
                        json.dumps(issue_data.get("public_opinion", {})),
                        issue_data.get("government_response", ""),
                        issue_data.get("recent_developments", []),
                        issue_data.get("political_impact", ""),
                        issue_data.get("social_impact", ""),
                        issue_data.get("economic_impact", ""),
                        issue_data.get("potential_resolution", ""),
                        embedding)
                        
                        # Generate initial news about this issue
                        await self._generate_domestic_news(run_ctx, issue_id, issue_data, nation_data)
                        
                        # Add to result
                        issue_data["id"] = issue_id
                        issues.append(issue_data)
                        
            except Exception as e:
                logging.error(f"Error generating domestic issue: {e}")
        
        return issues
    
    async def _generate_domestic_news(
        self, 
        ctx, 
        issue_id: int, 
        issue_data: Dict[str, Any],
        nation_data: Dict[str, Any]
    ) -> None:
        """Generate initial news articles about a domestic issue"""
        # Create agent for news generation
        news_agent = Agent(
            name="DomesticNewsAgent",
            instructions="You create realistic news articles about domestic political issues in a matriarchal society.",
            model="gpt-4.1-nano"
        )
        
        # Generate news articles from different perspectives
        biases = ["supporting", "opposing", "neutral"]
        
        for bias in biases:
            # Create prompt for the agent
            prompt = f"""
            Generate a news article about this domestic issue from a {bias} perspective:
            
            ISSUE:
            {json.dumps(issue_data, indent=2)}
            
            NATION:
            {json.dumps(nation_data, indent=2)}
            
            Create a news article that:
            1. Has a clear {bias} bias toward the issue
            2. Includes quotes from relevant figures
            3. Covers the key facts but with the appropriate spin
            4. Has a catchy headline
            5. Reflects the matriarchal power structures of society
            
            Return a JSON object with:
            - headline: The article headline
            - content: The full article content (300-500 words)
            - source_faction: The faction or institution publishing this
            """
            
            # Get response from agent
            result = await Runner.run(news_agent, prompt, context=ctx.context)
            
            try:
                # Parse response
                news_data = json.loads(result.final_output)
                
                # Ensure required fields exist
                if not all(k in news_data for k in ["headline", "content"]):
                    continue
                
                # Generate embedding
                embedding_text = f"{news_data['headline']} {news_data['content'][:200]}"
                embedding = await generate_embedding(embedding_text)
                
                # Store in database
                async with self.lore_manager.get_connection_pool() as pool:
                    async with pool.acquire() as conn:
                        await conn.execute("""
                            INSERT INTO DomesticNews (
                                issue_id, headline, content, source_faction, bias, embedding
                            )
                            VALUES ($1, $2, $3, $4, $5, $6)
                        """, 
                        issue_id,
                        news_data.get("headline"), 
                        news_data.get("content"),
                        news_data.get("source_faction", "Unknown Source"),
                        bias,
                        embedding)
                        
            except Exception as e:
                logging.error(f"Error generating domestic news: {e}")

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_initial_conflicts",
        action_description="Generating initial national conflicts",
        id_from_context=lambda ctx: "national_conflict_system"
    )
    async def generate_initial_conflicts(self, ctx, count: int = 3) -> List[Dict[str, Any]]:
        """
        Generate initial conflicts between nations with governance oversight.
        
        Args:
            count: Number of conflicts to generate
            
        Returns:
            List of generated conflicts
        """
        # Create the run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Get nations for context
        nations = await self.geopolitical_manager.get_all_nations(run_ctx)
        
        if len(nations) < 2:
            return []
        
        conflicts = []
        
        # Create agent for conflict generation
        conflict_agent = Agent(
            name="NationalConflictAgent",
            instructions="You create realistic international conflicts for a fantasy world with matriarchal power structures.",
            model="gpt-4.1-nano"
        )
        
        for i in range(count):
            # Select random nations that aren't already in major conflicts
            available_nations = [n for n in nations if not any(
                n["id"] in c.get("involved_nations", []) for c in conflicts
            )]
            
            if len(available_nations) < 2:
                available_nations = nations  # Fallback if needed
            
            # Choose two random nations
            nation_pair = random.sample(available_nations, 2)
            
            # Determine conflict type based on nations' characteristics
            matriarchy_diff = abs(
                nation_pair[0].get("matriarchy_level", 5) - 
                nation_pair[1].get("matriarchy_level", 5)
            )
            
            # Higher difference makes ideological conflicts more likely
            if matriarchy_diff > 4:
                conflict_types = ["ideological_dispute", "cultural_tension", "religious_conflict", "proxy_war"]
            elif matriarchy_diff > 2:
                conflict_types = ["diplomatic_tension", "border_dispute", "trade_dispute", "resource_conflict"]
            else:
                conflict_types = ["territorial_dispute", "trade_war", "succession_crisis", "alliance_dispute"]
                
            # Randomly select conflict type
            conflict_type = random.choice(conflict_types)
            
            # Create prompt for the agent
            prompt = f"""
            Generate a detailed international conflict between these two nations:
            
            NATION 1:
            {json.dumps(nation_pair[0], indent=2)}
            
            NATION 2:
            {json.dumps(nation_pair[1], indent=2)}
            
            Create a {conflict_type} that:
            1. Makes sense given the nations' characteristics
            2. Has appropriate severity and clear causes
            3. Includes realistic consequences and casualties
            4. Considers the matriarchal nature of the world
            5. Reflects how the differing matriarchy levels ({matriarchy_diff} point difference) might cause tension
            
            Return a JSON object with:
            - name: Name of the conflict
            - conflict_type: "{conflict_type}"
            - description: Detailed description
            - severity: Severity level (1-10)
            - status: Current status (active, escalating, etc.)
            - start_date: When it started (narrative date)
            - involved_nations: IDs of involved nations
            - primary_aggressor: ID of the primary aggressor
            - primary_defender: ID of the primary defender
            - current_casualties: Description of casualties so far
            - economic_impact: Description of economic impact
            - diplomatic_consequences: Description of diplomatic fallout
            - public_opinion: Object with nation IDs as keys and opinion descriptions as values
            - recent_developments: Array of recent events in the conflict
            - potential_resolution: Potential ways it might end
            """
            
            # Get response from agent
            result = await Runner.run(conflict_agent, prompt, context=run_ctx.context)
            
            try:
                # Parse response
                conflict_data = json.loads(result.final_output)
                
                # Ensure required fields exist
                if not all(k in conflict_data for k in ["name", "description", "conflict_type", "severity", "status"]):
                    continue
                
                # Generate embedding
                embedding_text = f"{conflict_data['name']} {conflict_data['description']} {conflict_data['conflict_type']}"
                embedding = await generate_embedding(embedding_text)
                
                # Store in database
                async with self.lore_manager.get_connection_pool() as pool:
                    async with pool.acquire() as conn:
                        conflict_id = await conn.fetchval("""
                            INSERT INTO NationalConflicts (
                                name, conflict_type, description, severity, status,
                                start_date, involved_nations, primary_aggressor, primary_defender,
                                current_casualties, economic_impact, diplomatic_consequences,
                                public_opinion, recent_developments, potential_resolution, embedding
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                            RETURNING id
                        """, 
                        conflict_data.get("name"), 
                        conflict_data.get("conflict_type"),
                        conflict_data.get("description"),
                        conflict_data.get("severity", 5),
                        conflict_data.get("status", "active"),
                        conflict_data.get("start_date", "Recently"),
                        conflict_data.get("involved_nations", [nation_pair[0]["id"], nation_pair[1]["id"]]),
                        conflict_data.get("primary_aggressor", nation_pair[0]["id"]),
                        conflict_data.get("primary_defender", nation_pair[1]["id"]),
                        conflict_data.get("current_casualties", "Unknown"),
                        conflict_data.get("economic_impact", "Unknown"),
                        conflict_data.get("diplomatic_consequences", "Unknown"),
                        json.dumps(conflict_data.get("public_opinion", {})),
                        conflict_data.get("recent_developments", []),
                        conflict_data.get("potential_resolution", "Unknown"),
                        embedding)
                        
                        # Generate initial news about this conflict
                        await self._generate_conflict_news(run_ctx, conflict_id, conflict_data, nation_pair)
                        
                        # Add to result
                        conflict_data["id"] = conflict_id
                        conflicts.append(conflict_data)
                        
            except Exception as e:
                logging.error(f"Error generating conflict: {e}")
        
        return conflicts
    
    async def _generate_conflict_news(
        self, 
        ctx, 
        conflict_id: int, 
        conflict_data: Dict[str, Any],
        nations: List[Dict[str, Any]]
    ) -> None:
        """Generate initial news articles about a conflict"""
        # Create agent for news generation
        news_agent = Agent(
            name="ConflictNewsAgent",
            instructions="You create realistic news articles about international conflicts in a matriarchal world.",
            model="gpt-4.1-nano"
        )
        
        # Generate one news article from each nation's perspective
        for i, nation in enumerate(nations[:2]):
            bias = "pro_defender" if nation["id"] == conflict_data.get("primary_defender") else "pro_aggressor"
            
            # Create prompt for the agent
            prompt = f"""
            Generate a news article about this conflict from the perspective of {nation["name"]}:
            
            CONFLICT:
            {json.dumps(conflict_data, indent=2)}
            
            REPORTING NATION:
            {json.dumps(nation, indent=2)}
            
            Create a news article that:
            1. Has a clear {bias} bias
            2. Includes quotes from officials (primarily women in positions of power)
            3. Covers the key facts but with the nation's spin
            4. Has a catchy headline
            5. Reflects matriarchal power structures in its language and reporting style
            
            Return a JSON object with:
            - headline: The article headline
            - content: The full article content (300-500 words)
            """
            
            # Get response from agent
            result = await Runner.run(news_agent, prompt, context=ctx.context)
            
            try:
                # Parse response
                news_data = json.loads(result.final_output)
                
                # Ensure required fields exist
                if not all(k in news_data for k in ["headline", "content"]):
                    continue
                
                # Apply matriarchal theming to content
                news_data["content"] = _apply_basic_replacements(news_data["content"])
                
                # Generate embedding
                embedding_text = f"{news_data['headline']} {news_data['content'][:200]}"
                embedding = await generate_embedding(embedding_text)
                
                # Store in database
                async with self.lore_manager.get_connection_pool() as pool:
                    async with pool.acquire() as conn:
                        await conn.execute("""
                            INSERT INTO ConflictNews (
                                conflict_id, headline, content, source_nation, bias, embedding
                            )
                            VALUES ($1, $2, $3, $4, $5, $6)
                        """, 
                        conflict_id,
                        news_data.get("headline"), 
                        news_data.get("content"),
                        nation["id"],
                        bias,
                        embedding)
                        
            except Exception as e:
                logging.error(f"Error generating conflict news: {e}")

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="get_active_conflicts",
        action_description="Getting active national conflicts",
        id_from_context=lambda ctx: "national_conflict_system"
    )
    async def get_active_conflicts(self, ctx) -> List[Dict[str, Any]]:
        """
        Get all active conflicts with governance oversight.
        
        Returns:
            List of active conflicts
        """
        # Check cache first
        cache_key = f"active_conflicts_{self.user_id}_{self.conversation_id}"
        cached = CONFLICT_CACHE.get(cache_key)
        if cached:
            return cached
        
        # Query database for active conflicts
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                conflicts = await conn.fetch("""
                    SELECT * FROM NationalConflicts
                    WHERE status != 'resolved'
                    ORDER BY severity DESC
                """)
                
                # Convert to list of dicts
                result = [dict(conflict) for conflict in conflicts]
                
                # Cache result
                CONFLICT_CACHE.set(cache_key, result)
                
                return result

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="get_nation_issues",
        action_description="Getting domestic issues for nation {nation_id}",
        id_from_context=lambda ctx: "national_conflict_system"
    )
    async def get_nation_issues(self, ctx, nation_id: int) -> List[Dict[str, Any]]:
        """
        Get all domestic issues for a nation with governance oversight.
        
        Args:
            nation_id: ID of the nation
            
        Returns:
            List of domestic issues
        """
        # Check cache first
        cache_key = f"nation_domestic_issues_{nation_id}_{self.user_id}_{self.conversation_id}"
        cached = CONFLICT_CACHE.get(cache_key)
        if cached:
            return cached
        
        # Query database for domestic issues
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                issues = await conn.fetch("""
                    SELECT * FROM DomesticIssues
                    WHERE nation_id = $1
                    ORDER BY severity DESC
                """, nation_id)
                
                # Convert to list of dicts
                result = [dict(issue) for issue in issues]
                
                # Parse JSON fields
                for issue in result:
                    if "public_opinion" in issue and issue["public_opinion"]:
                        try:
                            issue["public_opinion"] = json.loads(issue["public_opinion"])
                        except:
                            pass
                
                # Cache result
                CONFLICT_CACHE.set(cache_key, result)
                
                return result


# -------------------------------------------------
# RELIGIOUS DISTRIBUTION SYSTEM
# -------------------------------------------------

class ReligiousDistributionSystem:
    """
    Extends the FaithSystem to handle religious distribution across regions,
    including religious diversity, state religions, and religious laws.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.lore_manager = LoreManager(user_id, conversation_id)
        self.geopolitical_manager = GeopoliticalSystemManager(user_id, conversation_id)
        self.faith_system = FaithSystem(user_id, conversation_id)
        self.governor = None
        
    async def initialize_governance(self):
        """Initialize Nyx governance connection"""
        if not self.governor:
            self.governor = await get_central_governance(self.user_id, self.conversation_id)
        return self.governor
        
    async def initialize_tables(self):
        """Ensure religious distribution tables exist"""
        # First initialize base faith system tables
        await self.faith_system.initialize_tables()
        
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Check if NationReligion table exists
                nation_religion_exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'nationreligion'
                    );
                """)
                
                if not nation_religion_exists:
                    # Create the table
                    await conn.execute("""
                        CREATE TABLE NationReligion (
                            id SERIAL PRIMARY KEY,
                            nation_id INTEGER NOT NULL,
                            state_religion BOOLEAN DEFAULT FALSE,
                            primary_pantheon_id INTEGER, -- Main pantheon if any
                            pantheon_distribution JSONB, -- Distribution of pantheons by percentage
                            religiosity_level INTEGER CHECK (religiosity_level BETWEEN 1 AND 10),
                            religious_tolerance INTEGER CHECK (religious_tolerance BETWEEN 1 AND 10),
                            religious_leadership TEXT, -- Who leads religion nationally
                            religious_laws JSONB, -- Religious laws in effect
                            religious_holidays TEXT[], -- Major religious holidays
                            religious_conflicts TEXT[], -- Current religious tensions
                            religious_minorities TEXT[], -- Description of minority faiths
                            embedding VECTOR(1536),
                            FOREIGN KEY (nation_id) REFERENCES Nations(id) ON DELETE CASCADE,
                            FOREIGN KEY (primary_pantheon_id) REFERENCES Pantheons(id) ON DELETE SET NULL
                        );
                    """)
                    
                    # Create index
                    await conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_nationreligion_embedding 
                        ON NationReligion USING ivfflat (embedding vector_cosine_ops);
                        
                        CREATE INDEX IF NOT EXISTS idx_nationreligion_nation
                        ON NationReligion(nation_id);
                    """)
                    
                    logging.info("NationReligion table created")
                
                # Check if RegionalReligiousPractice table exists
                regional_practices_exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'regionalreligiouspractice'
                    );
                """)
                
                if not regional_practices_exists:
                    # Create the table
                    await conn.execute("""
                        CREATE TABLE RegionalReligiousPractice (
                            id SERIAL PRIMARY KEY,
                            nation_id INTEGER NOT NULL,
                            practice_id INTEGER NOT NULL, -- Reference to ReligiousPractices
                            regional_variation TEXT, -- How practice differs in this region
                            importance INTEGER CHECK (importance BETWEEN 1 AND 10),
                            frequency TEXT, -- How often practiced locally
                            local_additions TEXT, -- Any local additions to the practice
                            gender_differences TEXT, -- Any local gender differences
                            embedding VECTOR(1536),
                            FOREIGN KEY (nation_id) REFERENCES Nations(id) ON DELETE CASCADE,
                            FOREIGN KEY (practice_id) REFERENCES ReligiousPractices(id) ON DELETE CASCADE
                        );
                    """)
                    
                    # Create index
                    await conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_regionalreligiouspractice_embedding 
                        ON RegionalReligiousPractice USING ivfflat (embedding vector_cosine_ops);
                        
                        CREATE INDEX IF NOT EXISTS idx_regionalreligiouspractice_nation
                        ON RegionalReligiousPractice(nation_id);
                    """)
                    
                    logging.info("RegionalReligiousPractice table created")
    
    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="distribute_religions",
        action_description="Distributing religions across nations",
        id_from_context=lambda ctx: "religious_distribution_system"
    )
    async def distribute_religions(self, ctx) -> List[Dict[str, Any]]:
        """
        Distribute religions across nations with governance oversight.
        
        Returns:
            List of national religion distributions
        """
        # Create the run context
        run_ctx = RunContextWrapper(context=ctx.context)
        
        # Get nations and pantheons
        nations = await self.geopolitical_manager.get_all_nations(run_ctx)
        
        # Get pantheons through the faith system
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                pantheons = await conn.fetch("""
                    SELECT id, name, description, matriarchal_elements
                    FROM Pantheons
                """)
                
                # Convert to list of dicts
                pantheon_data = [dict(pantheon) for pantheon in pantheons]
        
        if not nations or not pantheon_data:
            return []
        
        # Create agent for religious distribution
        distribution_agent = Agent(
            name="ReligiousDistributionAgent",
            instructions="You distribute religious pantheons across fantasy nations in a matriarchal world.",
            model="gpt-4.1-nano"
        )
        
        distributions = []
        for nation in nations:
            # Create prompt for distribution
            prompt = f"""
            Determine religious distribution for this nation:
            
            NATION:
            {json.dumps(nation, indent=2)}
            
            AVAILABLE PANTHEONS:
            {json.dumps(pantheon_data, indent=2)}
            
            Create a realistic religious distribution that:
            1. Considers the nation's matriarchy level ({nation.get("matriarchy_level", 5)}/10)
            2. Determines whether it has a state religion
            3. Distributes pantheons in percentages
            4. Establishes religious laws and practices
            5. Emphasizes matriarchal and feminine aspects of religion
            
            Return a JSON object with:
            - nation_id: The nation ID
            - state_religion: Boolean indicating if there's a state religion
            - primary_pantheon_id: ID of main pantheon (or null if none)
            - pantheon_distribution: Object mapping pantheon IDs to percentage of population
            - religiosity_level: Overall religiosity (1-10)
            - religious_tolerance: Tolerance level (1-10)
            - religious_leadership: Who leads religion nationally (favor matriarchal leadership)
            - religious_laws: Object describing religious laws in effect
            - religious_holidays: Array of major religious holidays
            - religious_conflicts: Array of current religious tensions
            - religious_minorities: Array of minority faith descriptions
            """
            
            # Get response from agent
            result = await Runner.run(distribution_agent, prompt, context=run_ctx.context)
            
            try:
                # Parse JSON response
                distribution_data = json.loads(result.final_output)
                
                # Ensure required fields
                if not all(k in distribution_data for k in ["nation_id", "religiosity_level"]):
                    continue
                
                # Generate embedding
                embedding_text = f"religion {nation['name']} {distribution_data.get('religious_leadership', '')} {distribution_data.get('religious_tolerance', 5)}"
                embedding = await generate_embedding(embedding_text)
                
                # Store in database
                async with self.lore_manager.get_connection_pool() as pool:
                    async with pool.acquire() as conn:
                        distribution_id = await conn.fetchval("""
                            INSERT INTO NationReligion (
                                nation_id, state_religion, primary_pantheon_id, pantheon_distribution,
                                religiosity_level, religious_tolerance, religious_leadership,
                                religious_laws, religious_holidays, religious_conflicts,
                                religious_minorities, embedding
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                            RETURNING id
                        """,
                        distribution_data.get("nation_id"),
                        distribution_data.get("state_religion", False),
                        distribution_data.get("primary_pantheon_id"),
                        json.dumps(distribution_data.get("pantheon_distribution", {})),
                        distribution_data.get("religiosity_level", 5),
                        distribution_data.get("religious_tolerance", 5),
                        distribution_data.get("religious_leadership"),
                        json.dumps(distribution_data.get("religious_laws", {})),
                        distribution_data.get("religious_holidays", []),
                        distribution_data.get("religious_conflicts", []),
                        distribution_data.get("religious_minorities", []),
                        embedding)
                        
                        distribution_data["id"] = distribution_id
                        distributions.append(distribution_data)
                        
                        # Now generate regional religious practices
                        await self._generate_regional_practices(run_ctx, distribution_data)
            
            except Exception as e:
                logging.error(f"Error distributing religion for nation {nation['id']}: {e}")
        
        return distributions
    
    async def _generate_regional_practices(self, ctx, distribution_data: Dict[str, Any]) -> None:
        """Generate regional variations of religious practices"""
        # Get pantheons and practices
        nation_id = distribution_data.get("nation_id")
        primary_pantheon_id = distribution_data.get("primary_pantheon_id")
        
        if not primary_pantheon_id:
            return
        
        # Get religious practices for this pantheon
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                practices = await conn.fetch("""
                    SELECT id, name, practice_type, description, purpose
                    FROM ReligiousPractices
                    WHERE pantheon_id = $1
                """, primary_pantheon_id)
                
                # Convert to list of dicts
                practice_data = [dict(practice) for practice in practices]
                
                # Get nation details
                nation = await conn.fetchrow("""
                    SELECT name, government_type, matriarchy_level, cultural_traits
                    FROM Nations
                    WHERE id = $1
                """, nation_id)
                
                nation_data = dict(nation) if nation else {}
        
        if not practice_data or not nation_data:
            return
        
        # Create agent for regional practice generation
        practice_agent = Agent(
            name="RegionalPracticeAgent",
            instructions="You create regional variations of religious practices for a matriarchal society.",
            model="gpt-4.1-nano"
        )
        
        for practice in practice_data:
            # Create prompt for practice variation
            prompt = f"""
            Create a regional variation of this religious practice for this nation:
            
            NATION:
            {json.dumps(nation_data, indent=2)}
            
            RELIGIOUS PRACTICE:
            {json.dumps(practice, indent=2)}
            
            RELIGIOUS CONTEXT:
            Religiosity level: {distribution_data.get("religiosity_level", 5)}
            Religious tolerance: {distribution_data.get("religious_tolerance", 5)}
            
            Create a regional variation that:
            1. Adapts the practice to local culture
            2. Considers the nation's matriarchy level ({nation_data.get("matriarchy_level", 5)}/10)
            3. Emphasizes feminine power and authority
            4. Feels authentic to both the practice and the nation
            
            Return a JSON object with:
            - practice_id: ID of the original practice
            - regional_variation: How the practice is modified regionally
            - importance: Importance in this region (1-10)
            - frequency: How often practiced locally
            - local_additions: Any local additions to the practice
            - gender_differences: Any local gender differences
            """
            
            # Get response from agent
            result = await Runner.run(practice_agent, prompt, context=ctx.context)
            
            try:
                # Parse JSON response
                variation_data = json.loads(result.final_output)
                
                # Ensure required fields
                if not all(k in variation_data for k in ["practice_id", "regional_variation"]):
                    continue
                
                # Generate embedding
                embedding_text = f"practice {practice['name']} {variation_data['regional_variation']}"
                embedding = await generate_embedding(embedding_text)
                
                # Store in database
                async with self.lore_manager.get_connection_pool() as pool:
                    async with pool.acquire() as conn:
                        await conn.execute("""
                            INSERT INTO RegionalReligiousPractice (
                                nation_id, practice_id, regional_variation,
                                importance, frequency, local_additions,
                                gender_differences, embedding
                            )
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        """,
                        nation_id,
                        variation_data.get("practice_id"),
                        variation_data.get("regional_variation"),
                        variation_data.get("importance", 5),
                        variation_data.get("frequency"),
                        variation_data.get("local_additions"),
                        variation_data.get("gender_differences"),
                        embedding)
            
            except Exception as e:
                logging.error(f"Error generating regional practice variation: {e}")

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="get_nation_religion",
        action_description="Getting religious information for nation {nation_id}",
        id_from_context=lambda ctx: "religious_distribution_system"
    )
    async def get_nation_religion(self, ctx, nation_id: int) -> Dict[str, Any]:
        """
        Get comprehensive religious information about a nation with governance oversight.
        
        Args:
            nation_id: ID of the nation
            
        Returns:
            Dictionary with nation's religious information
        """
        # Check cache first
        cache_key = f"nation_religion_{nation_id}_{self.user_id}_{self.conversation_id}"
        cached = FAITH_CACHE.get(cache_key)
        if cached:
            return cached
        
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get nation details
                nation = await conn.fetchrow("""
                    SELECT id, name, government_type, matriarchy_level
                    FROM Nations
                    WHERE id = $1
                """, nation_id)
                
                if not nation:
                    return {"error": "Nation not found"}
                
                # Get religious distribution
                religion = await conn.fetchrow("""
                    SELECT * FROM NationReligion
                    WHERE nation_id = $1
                """, nation_id)
                
                if not religion:
                    return {"error": "No religious data for this nation"}
                
                # Get primary pantheon
                primary_pantheon_id = religion["primary_pantheon_id"]
                primary_pantheon = None
                if primary_pantheon_id:
                    pantheon = await conn.fetchrow("""
                        SELECT id, name, description, matriarchal_elements
                        FROM Pantheons
                        WHERE id = $1
                    """, primary_pantheon_id)
                    
                    if pantheon:
                        primary_pantheon = dict(pantheon)
                
                # Get regional practices
                practices = await conn.fetch("""
                    SELECT r.*, p.name as practice_name, p.practice_type, p.purpose
                    FROM RegionalReligiousPractice r
                    JOIN ReligiousPractices p ON r.practice_id = p.id
                    WHERE r.nation_id = $1
                """, nation_id)
                
                # Get holy sites in this nation
                holy_sites = await conn.fetch("""
                    SELECT h.* 
                    FROM HolySites h
                    JOIN Locations l ON h.location_id = l.id
                    JOIN LoreConnections lc ON l.id = lc.target_id
                    JOIN Nations n ON lc.source_id = n.id
                    WHERE n.id = $1 AND lc.source_type = 'Nations' AND lc.target_type = 'Locations'
                """, nation_id)
                
                # Compile result
                result = {
                    "nation": dict(nation),
                    "religion": dict(religion),
                    "primary_pantheon": primary_pantheon,
                    "regional_practices": [dict(practice) for practice in practices],
                    "holy_sites": [dict(site) for site in holy_sites]
                }
                
                # Parse JSON fields
                if "pantheon_distribution" in result["religion"] and result["religion"]["pantheon_distribution"]:
                    try:
                        result["religion"]["pantheon_distribution"] = json.loads(result["religion"]["pantheon_distribution"])
                    except:
                        pass
                
                if "religious_laws" in result["religion"] and result["religion"]["religious_laws"]:
                    try:
                        result["religion"]["religious_laws"] = json.loads(result["religion"]["religious_laws"])
                    except:
                        pass
                
                # Cache the result
                FAITH_CACHE.set(cache_key, result)
                
                return result


# -------------------------------------------------
# LORE UPDATE SYSTEM
# -------------------------------------------------

class LoreUpdateSystem:
    """
    System for updating lore elements based on events with sophisticated
    cascade effects and societal impact calculations.
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.lore_manager = LoreManager(user_id, conversation_id)
        self.governor = None
        
    async def initialize_governance(self):
        """Initialize Nyx governance connection"""
        if not self.governor:
            self.governor = await get_central_governance(self.user_id, self.conversation_id)
        return self.governor

    async def _build_enhanced_lore_prompt(
        self,
        element: Dict[str, Any],
        event_description: str,
        societal_impact: Dict[str, Any],
        related_elements: List[Dict[str, Any]],
        hierarchy_position: int,
        update_history: List[Dict[str, Any]],
        player_character: Dict[str, Any] = None,
        dominant_npcs: List[Dict[str, Any]] = None
    ) -> str:
        """
        Build a sophisticated prompt for lore updates with rich context
        
        Args:
            element: The lore element to update
            event_description: Description of the event
            societal_impact: Impact assessment on society
            related_elements: Elements connected to this one
            hierarchy_position: Position in power hierarchy (lower is more powerful)
            update_history: Recent update history for this element
            player_character: Optional player character data
            dominant_npcs: Optional list of ruling NPCs
            
        Returns:
            A detailed prompt for the LLM
        """
        # Format update history as context
        history_context = ""
        if update_history:
            history_items = []
            for update in update_history:
                history_items.append(f"- {update['timestamp']}: {update['update_reason']}")
            history_context = "UPDATE HISTORY:\n" + "\n".join(history_items)
        
        # Format related elements as context
        relationships_context = ""
        if related_elements:
            rel_items = []
            for rel in related_elements:
                rel_items.append(f"- {rel['name']} ({rel['lore_type']}): {rel['relationship_type']} - {rel['relationship_strength']}/10")
            relationships_context = "RELATIONSHIPS:\n" + "\n".join(rel_items)
        
        # Format player character context if available
        player_context = ""
        if player_character:
            player_context = f"""
            PLAYER CHARACTER CONTEXT:
            Name: {player_character['name']}
            Status: {player_character['status']}
            Recent Actions: {player_character['recent_actions']}
            Position in Hierarchy: {player_character.get('hierarchy_position', 'subordinate')}
            """
        
        # Format dominant NPCs context if available
        dominant_context = ""
        if dominant_npcs:
            dom_items = []
            for npc in dominant_npcs:
                dom_items.append(f"- {npc['name']}: {npc['position']} - {npc['attitude']} toward situation")
            dominant_context = "RELEVANT AUTHORITY FIGURES:\n" + "\n".join(dom_items)
        
        # Hierarchy-appropriate directive
        hierarchy_directive = await self._get_hierarchy_directive(hierarchy_position)
        
        # Build the complete prompt
        prompt = f"""
        The following lore element in our matriarchal-themed RPG world requires updating based on recent events:
        
        LORE ELEMENT:
        Type: {element['lore_type']}
        Name: {element['name']}
        Current Description: {element['description']}
        Position in Hierarchy: {hierarchy_position}/10 (lower number = higher authority)
        
        {relationships_context}
        
        {history_context}
        
        EVENT THAT OCCURRED:
        {event_description}
        
        SOCIETAL IMPACT ASSESSMENT:
        Stability Impact: {societal_impact['stability_impact']}/10
        Power Structure Change: {societal_impact['power_structure_change']}
        Public Perception Shift: {societal_impact['public_perception']}
        
        {player_context}
        
        {dominant_context}
        
        {hierarchy_directive}
        
        Generate a sophisticated update for this lore element that incorporates the impact of this event.
        The update should maintain narrative consistency while allowing for meaningful development.
        
        Return your response as a JSON object with:
        {{
            "new_description": "The updated description that reflects event impact",
            "update_reason": "Detailed explanation of why this update makes sense",
            "impact_level": A number from 1-10 indicating how significantly this event affects this element,
            "narrative_themes": ["List", "of", "relevant", "themes"],
            "adaptation_quality": A number from 1-10 indicating how well the element adapts to change,
            "hierarchy_changes": {{"entity_id": change_value, ...}},
            "mood_shift": "emotional tone shift (e.g. 'neutral', 'tense', 'relieved')"
        }}
        
        ADDITIONAL TYPE-SPECIFIC FIELDS:
        """
        
        # Add type-specific fields to the prompt
        if element['lore_type'] == 'character':
            prompt += """
            "character_development": {
                "confidence": 1-10,
                "resolve": 1-10,
                "ambition": 1-10
            },
            "motivation_shift": "description of how character's motivations might change"
            """
        elif element['lore_type'] == 'location':
            prompt += """
            "atmosphere_change": "how the location's feel or atmosphere changes",
            "accessibility_change": "how access to this location may have changed"
            """
        elif element['lore_type'] == 'faction':
            prompt += """
            "internal_stability": "how stable the faction is after events",
            "external_influence": "how the faction's influence has changed"
            """
        
        return prompt
    
    async def _get_hierarchy_directive(self, hierarchy_position: int) -> str:
        """
        Get an appropriate directive based on the element's position in hierarchy
        
        Args:
            hierarchy_position: Position in the power hierarchy (1-10, lower is more powerful)
            
        Returns:
            A directive string appropriate to the hierarchy level
        """
        if hierarchy_position <= 2:
            return """
            DIRECTIVE: This element represents a highest-tier authority figure. 
            Their decisions significantly impact the world. 
            They rarely change their core principles but may adjust strategies.
            They maintain control and authority in all situations.
            """
        elif hierarchy_position <= 4:
            return """
            DIRECTIVE: This element represents high authority.
            They have significant influence but answer to the highest tier.
            They strongly maintain the established order while pursuing their ambitions.
            They assert dominance in their domain but show deference to higher authority.
            """
        elif hierarchy_position <= 7:
            return """
            DIRECTIVE: This element has mid-level authority.
            They implement the will of higher authorities while managing those below.
            They may have personal aspirations but function within established boundaries.
            They balance compliance with higher authority against control of subordinates.
            """
        else:
            return """
            DIRECTIVE: This element has low authority in the hierarchy.
            They follow directives from above and have limited autonomy.
            They may seek to improve their position but must navigate carefully.
            They show appropriate deference to those of higher status.
            """
    
    async def _parse_lore_update_response(self, response_text: str, element: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the LLM response with advanced error handling
        
        Args:
            response_text: Raw response from the LLM
            element: Original lore element
            
        Returns:
            Parsed update data
        """
        try:
            # First try to parse as JSON
            update_data = json.loads(response_text)
            
            # Validate required fields
            required_fields = ['new_description', 'update_reason', 'impact_level']
            for field in required_fields:
                if field not in update_data:
                    raise ValueError(f"Missing required field: {field}")
            
            return update_data
            
        except json.JSONDecodeError:
            logging.warning(f"Failed to parse JSON response for {element['name']}")
            
            # Try regex extraction for common patterns
            patterns = {
                'new_description': r'"new_description"\s*:\s*"([^"]+)"',
                'update_reason': r'"update_reason"\s*:\s*"([^"]+)"',
                'impact_level': r'"impact_level"\s*:\s*(\d+)',
                'narrative_themes': r'"narrative_themes"\s*:\s*\[(.*?)\]',
                'mood_shift': r'"mood_shift"\s*:\s*"([^"]+)"'
            }
            
            extracted_data = {}
            for key, pattern in patterns.items():
                match = re.search(pattern, response_text, re.DOTALL)
                if match:
                    if key == 'impact_level':
                        extracted_data[key] = int(match.group(1))
                    elif key == 'narrative_themes':
                        themes_str = match.group(1)
                        themes = [t.strip().strip('"\'') for t in themes_str.split(',')]
                        extracted_data[key] = themes
                    else:
                        extracted_data[key] = match.group(1)
            
            # Fill in missing required fields with defaults
            if 'new_description' not in extracted_data:
                # Find the longest paragraph as a fallback description
                paragraphs = re.split(r'\n\n+', response_text)
                paragraphs = [p for p in paragraphs if len(p) > 50]
                if paragraphs:
                    extracted_data['new_description'] = max(paragraphs, key=len)
                else:
                    extracted_data['new_description'] = element['description']
            
            if 'update_reason' not in extracted_data:
                extracted_data['update_reason'] = "Event impact (extracted from unstructured response)"
                
            if 'impact_level' not in extracted_data:
                # Look for numbers in text that might indicate impact level
                numbers = re.findall(r'\b([1-9]|10)\b', response_text)
                if numbers:
                    extracted_data['impact_level'] = int(numbers[0])
                else:
                    extracted_data['impact_level'] = 5
            
            return extracted_data
    
    async def _calculate_cascade_effects(
        self, 
        element: Dict[str, Any], 
        update_data: Dict[str, Any],
        related_elements: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate how updates to one element affect related elements
        
        Args:
            element: The updated lore element
            update_data: The update data for this element
            related_elements: Elements related to this one
            
        Returns:
            Dictionary of cascade effects
        """
        cascade_effects = {
            'relationship_changes': {},
            'power_shifts': {}
        }
        
        impact_level = update_data.get('impact_level', 5)
        
        # Calculate relationship changes based on impact
        for related in related_elements:
            rel_id = related['lore_id']
            rel_strength = related.get('relationship_strength', 5)
            rel_type = related.get('relationship_type', 'neutral')
            
            # Calculate relationship change based on event impact and current relationship
            # Higher impact events cause more relationship change
            if rel_type in ['subservient', 'loyal']:
                # Loyal/subservient relationships strengthen during impactful events
                change = (impact_level - 5) * 0.3
            elif rel_type in ['authority', 'dominant']:
                # Authority relationships may weaken slightly during high-impact events
                change = (5 - impact_level) * 0.2
            elif rel_type in ['rival', 'adversarial']:
                # Rivalries intensify during impactful events
                change = -abs(impact_level - 5) * 0.4
            else:
                # Neutral relationships shift based on impact direction
                change = (impact_level - 5) * 0.1
            
            # Adjust for hierarchy differences - larger gaps mean more significant changes
            hierarchy_diff = abs(
                element.get('hierarchy_position', 5) - 
                related.get('hierarchy_position', 5)
            )
            
            change *= (1 + (hierarchy_diff * 0.1))
            
            cascade_effects['relationship_changes'][rel_id] = round(change, 1)
        
        # Calculate power shifts for relevant factions
        if element['lore_type'] == 'faction':
            # Direct power shift for the affected faction
            faction_id = element['lore_id']
            power_shift = (impact_level - 5) * 0.5
            cascade_effects['power_shifts'][faction_id] = power_shift
        
        # Calculate power shifts for factions related to the element
        faction_relations = [r for r in related_elements if r['lore_type'] == 'faction']
        for faction in faction_relations:
            faction_id = faction['lore_id']
            rel_type = faction.get('relationship_type', 'neutral')
            
            # Calculate power shift based on relationship type
            if rel_type in ['allied', 'supportive']:
                # Allied factions shift in the same direction
                shift = (impact_level - 5) * 0.3
            elif rel_type in ['rival', 'opposed']:
                # Rival factions shift in the opposite direction
                shift = (5 - impact_level) * 0.3
            else:
                # Neutral factions have minimal shifts
                shift = (impact_level - 5) * 0.1
                
            cascade_effects['power_shifts'][faction_id] = round(shift, 1)
        
        return cascade_effects
    
    async def _calculate_societal_impact(
        self,
        event_description: str,
        stability_index: int,
        power_hierarchy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate the societal impact of an event
        
        Args:
            event_description: Description of the event
            stability_index: Current stability of society (1-10)
            power_hierarchy: Current power structure data
            
        Returns:
            Dictionary of societal impact metrics
        """
        # Analyze event text for impact keywords
        impact_keywords = {
            'high_impact': [
                'overthrown', 'revolution', 'usurped', 'conquered', 'rebellion',
                'assassination', 'coup', 'catastrophe', 'disaster'
            ],
            'medium_impact': [
                'challenge', 'conflict', 'dispute', 'tension', 'unrest',
                'scandal', 'controversy', 'uprising', 'demonstration'
            ],
            'low_impact': [
                'minor', 'small', 'limited', 'isolated', 'contained',
                'private', 'personal', 'individual', 'trivial'
            ]
        }
        
        # Count keyword occurrences
        high_count = sum(1 for word in impact_keywords['high_impact'] if word.lower() in event_description.lower())
        medium_count = sum(1 for word in impact_keywords['medium_impact'] if word.lower() in event_description.lower())
        low_count = sum(1 for word in impact_keywords['low_impact'] if word.lower() in event_description.lower())
        
        # Calculate base stability impact
        if high_count > 0:
            base_stability_impact = 7 + min(high_count, 3)
        elif medium_count > 0:
            base_stability_impact = 4 + min(medium_count, 3)
        elif low_count > 0:
            base_stability_impact = 2 + min(low_count, 2)
        else:
            base_stability_impact = 3  # Default moderate impact
        
        # Adjust for current stability
        # Higher stability means events have less impact
        stability_modifier = (10 - stability_index) / 10
        adjusted_impact = base_stability_impact * (0.5 + stability_modifier)
        
        # Determine power structure change
        if adjusted_impact >= 8:
            power_change = "significant realignment of authority"
        elif adjusted_impact >= 6:
            power_change = "moderate shift in power dynamics"
        elif adjusted_impact >= 4:
            power_change = "subtle adjustments to authority structures"
        else:
            power_change = "minimal change to established order"
        
        # Determine public perception
        if adjusted_impact >= 7:
            if "rebellion" in event_description.lower() or "uprising" in event_description.lower():
                perception = "widespread questioning of authority"
            else:
                perception = "significant public concern"
        elif adjusted_impact >= 5:
            perception = "notable public interest and discussion"
        else:
            perception = "limited public awareness or interest"
        
        return {
            'stability_impact': round(adjusted_impact),
            'power_structure_change': power_change,
            'public_perception': perception
        }
    
    async def _generate_cascade_updates(
        self,
        cascade_elements: List[Dict[str, Any]],
        event_description: str,
        relationship_changes: Dict[str, float],
        power_shifts: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Generate simplified updates for elements affected indirectly
        
        Args:
            cascade_elements: List of elements affected by cascade
            event_description: Original event description
            relationship_changes: Dictionary of relationship changes by ID
            power_shifts: Dictionary of power shifts by faction ID
            
        Returns:
            List of update records for cascade elements
        """
        cascade_updates = []
        
        for element in cascade_elements:
            element_id = element['lore_id']
            
            # Get the relationship change if any
            rel_change = relationship_changes.get(element_id, 0)
            
            # Get power shift if this is a faction
            power_shift = 0
            if element['lore_type'] == 'faction':
                power_shift = power_shifts.get(element_id, 0)
            
            # Calculate impact level based on relationship change and power shift
            impact_level = min(10, max(1, round(5 + abs(rel_change) * 2 + abs(power_shift) * 2)))
            
            # Generate a simplified update
            if abs(rel_change) > 1 or abs(power_shift) > 1:
                # Significant enough to warrant an update
                
                # Determine the nature of the update based on changes
                if element['lore_type'] == 'character':
                    if rel_change > 0:
                        update_reason = f"Strengthened position due to recent events"
                        description_modifier = "more confident and assured"
                    else:
                        update_reason = f"Position weakened by recent events"
                        description_modifier = "more cautious and reserved"
                
                elif element['lore_type'] == 'faction':
                    if power_shift > 0:
                        update_reason = f"Gained influence following recent events"
                        description_modifier = "increasing their authority and reach"
                    else:
                        update_reason = f"Lost influence due to recent events"
                        description_modifier = "adapting to their diminished standing"
                
                elif element['lore_type'] == 'location':
                    if rel_change > 0:
                        update_reason = f"Increased importance after recent events"
                        description_modifier = "now sees more activity and attention"
                    else:
                        update_reason = f"Decreased importance after recent events"
                        description_modifier = "now sees less activity and attention"
                
                else:
                    update_reason = f"Indirectly affected by recent events"
                    description_modifier = "subtly changed by recent developments"
                
                # Create a new description with the modifier
                new_description = element['description']
                if "." in new_description:
                    parts = new_description.split(".")
                    parts[-2] += f", {description_modifier}"
                    new_description = ".".join(parts)
                else:
                    new_description = f"{new_description} {description_modifier}."
                
                # Create update record
                cascade_updates.append({
                    'lore_type': element['lore_type'],
                    'lore_id': element['lore_id'],
                    'name': element['name'],
                    'old_description': element['description'],
                    'new_description': new_description,
                    'update_reason': update_reason,
                    'impact_level': impact_level,
                    'is_cascade_update': True,
                    'timestamp': datetime.datetime.now().isoformat()
                })
        
        return cascade_updates
    
    # Utility methods
    
    async def _fetch_world_state(self) -> Dict[str, Any]:
        """Fetch current world state from database"""
        # Implementation would query the database for world state
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Example query - adjust based on your schema
                world_state = await conn.fetchrow("""
                    SELECT * FROM WorldState 
                    WHERE user_id = $1 AND conversation_id = $2
                    LIMIT 1
                """, self.user_id, self.conversation_id)
                
                if world_state:
                    return dict(world_state)
                else:
                    # Return default values if no world state found
                    return {
                        'stability_index': 8,
                        'narrative_tone': 'dramatic',
                        'power_dynamics': 'strict_hierarchy',
                        'power_hierarchy': {}
                    }
    
    async def _fetch_related_elements(self, lore_id: str) -> List[Dict[str, Any]]:
        """Fetch elements related to the given lore ID"""
        # Implementation would query the database for relationships
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Example query - adjust based on your schema
                related = await conn.fetch("""
                    SELECT e.lore_id, e.name, e.lore_type, r.relationship_type, r.relationship_strength 
                    FROM LoreElements e
                    JOIN LoreRelationships r ON e.lore_id = r.target_id
                    WHERE r.source_id = $1
                """, lore_id)
                
                return [dict(rel) for rel in related]
    
    async def _get_hierarchy_position(self, element: Dict[str, Any]) -> int:
        """Determine element's position in the power hierarchy"""
        # Implementation would calculate or retrieve hierarchy position
        # For now, return a default based on lore_type
        if element['lore_type'] == 'character':
            # Check if character has a stored hierarchy value
            if 'hierarchy_position' in element:
                return element['hierarchy_position']
            else:
                # Default based on name keywords
                name = element['name'].lower()
                if any(title in name for title in ['queen', 'empress', 'matriarch', 'high', 'supreme']):
                    return 1
                elif any(title in name for title in ['princess', 'duchess', 'lady', 'noble']):
                    return 3
                elif any(title in name for title in ['advisor', 'minister', 'council']):
                    return 5
                else:
                    return 8
        elif element['lore_type'] == 'faction':
            # Check faction importance
            if 'importance' in element:
                return max(1, 10 - element['importance'])
            else:
                return 4  # Default for factions
        elif element['lore_type'] == 'location':
            # Check location significance
            if 'significance' in element:
                return max(1, 10 - element['significance'])
            else:
                return 6  # Default for locations
        else:
            return 5  # Default middle position
    
    async def _fetch_element_update_history(self, lore_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Fetch recent update history for an element"""
        # Implementation would query the database for update history
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Example query - adjust based on your schema
                history = await conn.fetch("""
                    SELECT timestamp, update_reason
                    FROM LoreUpdates
                    WHERE lore_id = $1
                    ORDER BY timestamp DESC
                    LIMIT $2
                """, lore_id, limit)
                
                return [dict(update) for update in history]
    
    async def _log_lore_update_error(self, lore_id: str, error: str, context: Dict[str, Any]) -> None:
        """Log an error that occurred during lore update"""
        # Implementation would log the error to database or file
        logging.error(f"Error updating lore element {lore_id}: {error}")
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Example query - adjust based on your schema
                await conn.execute("""
                    INSERT INTO LoreUpdateErrors (
                        lore_id, error_message, timestamp, context_data
                    ) VALUES ($1, $2, $3, $4)
                """, lore_id, error, datetime.datetime.now(), json.dumps(context))
    
    async def _fetch_elements_by_ids(self, element_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch multiple elements by their IDs"""
        # Implementation would query the database for multiple elements
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Example query - adjust based on your schema
                elements = await conn.fetch("""
                    SELECT lore_id, name, lore_type, description
                    FROM LoreElements
                    WHERE lore_id = ANY($1)
                """, element_ids)
                
                return [dict(elem) for elem in elements]
    
    async def _update_world_power_balance(self, power_shifts: Dict[str, float]) -> None:
        """Update the world state to reflect power shifts"""
        # Implementation would update the world state in the database
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Get current power hierarchy
                world_state = await self._fetch_world_state()
                power_hierarchy = world_state.get('power_hierarchy', {})
                
                # Update power values
                for faction_id, shift in power_shifts.items():
                    current_power = power_hierarchy.get(faction_id, 5)
                    new_power = max(1, min(10, current_power + shift))
                    power_hierarchy[faction_id] = new_power
                
                # Update the database
                await conn.execute("""
                    UPDATE WorldState
                    SET power_hierarchy = $1,
                        last_updated = $2
                    WHERE user_id = $3 AND conversation_id = $4
                """, json.dumps(power_hierarchy), datetime.datetime.now(), 
                self.user_id, self.conversation_id)
    
    async def _log_narrative_interactions(self, updates: List[Dict[str, Any]], societal_impact: Dict[str, Any]) -> None:
        """Log the narrative interactions for future reference"""
        # Implementation would log the interactions to database
        if not updates:
            return
            
        async with self.lore_manager.get_connection_pool() as pool:
            async with pool.acquire() as conn:
                # Example query - adjust based on your schema
                await conn.execute("""
                    INSERT INTO NarrativeInteractions (
                        timestamp, stability_impact, power_change, 
                        public_perception, affected_elements, update_count
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                """, 
                datetime.datetime.now(),
                societal_impact.get('stability_impact', 5),
                societal_impact.get('power_structure_change', 'minimal'),
                societal_impact.get('public_perception', 'limited'),
                json.dumps([u.get('lore_id') for u in updates]),
                len(updates))

    @with_governance(
        agent_type=AgentType.NARRATIVE_CRAFTER,
        action_type="generate_lore_updates",
        action_description="Generating lore updates for event",
        id_from_context=lambda ctx: "lore_update_system"
    )
    async def generate_lore_updates(
        self, 
        ctx,
        affected_elements: List[Dict[str, Any]], 
        event_description: str,
        player_character: Dict[str, Any] = None,
        dominant_npcs: List[Dict[str, Any]] = None,
        world_state: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate sophisticated updates for affected lore elements in a matriarchal-themed RPG
        
        Args:
            affected_elements: List of affected lore elements
            event_description: Description of the event
            player_character: Optional player character data to provide context
            dominant_npcs: Optional list of ruling NPCs relevant to the event
            world_state: Optional current world state data
            
        Returns:
            List of detailed updates to apply with cascading effects
        """
        updates = []
        relationship_changes = {}
        power_shifts = {}
        
        # Track elements that will need secondary updates due to relationships
        cascading_elements = set()
        
        # Retrieve world context from database if not provided
        if not world_state:
            world_state = await self._fetch_world_state()
        
        # Determine societal consequences of the event
        societal_impact = await self._calculate_societal_impact(
            event_description, 
            world_state.get('stability_index', 8),
            world_state.get('power_hierarchy', {})
        )
        
        # Use an LLM to generate detailed updates for each element
        for element in affected_elements:
            # Retrieve relationship network for this element
            related_elements = await self._fetch_related_elements(element['lore_id'])
            
            # Determine element's position in power hierarchy
            hierarchy_position = await self._get_hierarchy_position(element)
            
            # Build contextual history of recent updates to this element
            update_history = await self._fetch_element_update_history(
                element['lore_id'], 
                limit=5
            )
            
            # Create enhanced run context with detailed metadata
            run_ctx = RunContextWrapper(context={
                "user_id": self.user_id,
                "conversation_id": self.conversation_id,
                "world_theme": "matriarchal_society",
                "narrative_tone": world_state.get('narrative_tone', 'dramatic'),
                "power_dynamics": world_state.get('power_dynamics', 'strict_hierarchy'),
                "element_update_count": len(update_history)
            })
            
            # Create a sophisticated prompt for the LLM with richer context
            prompt = await self._build_enhanced_lore_prompt(
                element=element,
                event_description=event_description,
                societal_impact=societal_impact,
                related_elements=related_elements,
                hierarchy_position=hierarchy_position,
                update_history=update_history,
                player_character=player_character,
                dominant_npcs=dominant_npcs
            )
            
            # Select appropriate model based on element importance
            model_name = "o3-large" if hierarchy_position < 3 else "gpt-4.1-nano"
            
            # Create an advanced agent for sophisticated lore updates
            lore_update_agent = Agent(
                name="MatriarchalLoreAgent",
                instructions="""
                You update narrative elements in a matriarchal society RPG setting.
                Focus on power dynamics, authority shifts, and social consequences.
                Maintain internal consistency while allowing for character development.
                Ensure updates reflect the established hierarchy and social order.
                Consider how changes cascade through relationship networks.
                """,
                model=model_name,
                temperature=0.7
            )
            
            # Get the response with enhanced error handling
            try:
                result = await Runner.run(
                    lore_update_agent, 
                    prompt, 
                    context=run_ctx.context,
                    timeout=15
                )
                response_text = result.final_output
                
                # Parse the enhanced JSON response
                update_data = await self._parse_lore_update_response(response_text, element)
                
                # Calculate cascading effects on related elements
                cascade_effects = await self._calculate_cascade_effects(
                    element, 
                    update_data, 
                    related_elements
                )
                
                # Track relationship changes
                for rel_id, change in cascade_effects.get('relationship_changes', {}).items():
                    if rel_id in relationship_changes:
                        relationship_changes[rel_id] += change
                    else:
                        relationship_changes[rel_id] = change
                    
                    # Add affected relationships to cascade list
                    if abs(change) > 2:  # Only cascade significant changes
                        cascading_elements.add(rel_id)
                
                # Track power shifts
                for faction_id, shift in cascade_effects.get('power_shifts', {}).items():
                    if faction_id in power_shifts:
                        power_shifts[faction_id] += shift
                    else:
                        power_shifts[faction_id] = shift
                
                # Create enriched update record
                update_record = {
                    'lore_type': element['lore_type'],
                    'lore_id': element['lore_id'],
                    'name': element['name'],
                    'old_description': element['description'],
                    'new_description': update_data['new_description'],
                    'update_reason': update_data['update_reason'],
                    'impact_level': update_data['impact_level'],
                    'narrative_themes': update_data.get('narrative_themes', []),
                    'adaptation_quality': update_data.get('adaptation_quality', 7),
                    'hierarchy_changes': update_data.get('hierarchy_changes', {}),
                    'mood_shift': update_data.get('mood_shift', 'neutral'),
                    'timestamp': datetime.datetime.now().isoformat()
                }
                
                # Add lore-specific fields based on element type
                if element['lore_type'] == 'character':
                    update_record['character_development'] = update_data.get('character_development', {})
                    update_record['motivation_shift'] = update_data.get('motivation_shift', 'none')
                elif element['lore_type'] == 'location':
                    update_record['atmosphere_change'] = update_data.get('atmosphere_change', 'none')
                    update_record['accessibility_change'] = update_data.get('accessibility_change', 'unchanged')
                elif element['lore_type'] == 'faction':
                    update_record['internal_stability'] = update_data.get('internal_stability', 'stable')
                    update_record['external_influence'] = update_data.get('external_influence', 'unchanged')
                
                updates.append(update_record)
                
            except Exception as e:
                logging.error(f"Error generating update for {element['name']}: {str(e)}")
                # Log the error and continue with next element
                await self._log_lore_update_error(element['lore_id'], str(e), run_ctx.context)
        
        # Process cascading updates for indirectly affected elements
        if cascading_elements:
            # Fetch the cascading elements data
            cascade_element_data = await self._fetch_elements_by_ids(list(cascading_elements))
            
            # Generate less detailed updates for cascade elements
            cascade_updates = await self._generate_cascade_updates(
                cascade_element_data,
                event_description,
                relationship_changes,
                power_shifts
            )
            
            # Add cascade updates to the main updates list
            updates.extend(cascade_updates)
        
        # Update world state with power shifts
        if power_shifts:
            await self._update_world_power_balance(power_shifts)
        
        # Log complex interactions for narrative coherence tracking
        await self._log_narrative_interactions(updates, societal_impact)
        
        return updates
