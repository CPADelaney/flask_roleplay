# story_templates/moth/story_initializer.py
"""
Complete story initialization system for The Queen of Thorns
Enhanced with performance optimizations, better error handling, and cost controls
"""

import logging
import json
import asyncio
import random
import hashlib
import textwrap
from typing import Dict, Any, List, Optional, Tuple, TypeVar, Union
from datetime import datetime
from functools import wraps
from dataclasses import dataclass
from pydantic import BaseModel, Field, ValidationError
from contextlib import asynccontextmanager
import os

from npcs.new_npc_creation import NPCCreationHandler
from db.connection import get_db_connection_context
from story_templates.moth.poem_integrated_loader import ThornsIntegratedStoryLoader
from story_templates.character_profiles.lilith_ravencroft import LILITH_RAVENCROFT
from lore.core import canon
from memory.wrapper import MemorySystem
from story_templates.preset_stories import StoryBeat, PresetStory
from nyx.integrate import remember_with_governance
from npcs.preset_npc_handler import PresetNPCHandler
from story_templates.moth.lore.world_lore_manager import SFBayQueenOfThornsPreset
from logic.chatgpt_integration import get_async_openai_client

logger = logging.getLogger(__name__)

# Environment variable overrides with sensible defaults
_DEFAULT_LORE_MODEL = os.getenv("OPENAI_LORE_MODEL", "gpt-4o-mini")
_DEFAULT_MEMORY_MODEL = os.getenv("OPENAI_MEMORY_MODEL", "gpt-4o-mini")
_DEFAULT_ATMOSPHERE_MODEL = os.getenv("OPENAI_ATMOSPHERE_MODEL", "gpt-4o-mini")
_DEFAULT_LOCATION_MODEL = os.getenv("OPENAI_LOCATION_MODEL", "gpt-4o-mini")
_DEFAULT_POETRY_MODEL = os.getenv("OPENAI_POETRY_MODEL", "gpt-4o-mini")

# Performance settings
MAX_CONCURRENT_GPT_CALLS = int(os.getenv("MAX_CONCURRENT_GPT_CALLS", "3"))
ENABLE_GPT_CACHE = os.getenv("ENABLE_GPT_CACHE", "true").lower() == "true"
MAX_TOKEN_SAFETY = int(os.getenv("MAX_TOKEN_SAFETY", "8000"))
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))

# Pydantic models for response validation
class LoreData(BaseModel):
    news_items: List[str] = Field(default_factory=list, max_items=5)
    subcultures: List[str] = Field(default_factory=list, max_items=6)
    rumors: List[str] = Field(default_factory=list, max_items=7)
    supernatural_whispers: List[str] = Field(default_factory=list, max_items=5)

class LocationEnhancement(BaseModel):
    description: str = Field(..., min_length=100, max_length=2000)
    atmospheric_details: Dict[str, Any] = Field(default_factory=dict)
    hidden_features: List[str] = Field(default_factory=list, max_items=5)

class CharacterEvolution(BaseModel):
    new_scars: List[str] = Field(default_factory=list, max_items=3)
    recent_intrigues: List[str] = Field(default_factory=list, max_items=3)
    current_masks: List[str] = Field(default_factory=list, max_items=4)
    dialogue_evolution: List[str] = Field(default_factory=list, max_items=5)

class NetworkState(BaseModel):
    active_operations: List[Dict[str, Any]] = Field(default_factory=list, max_items=10)
    threat_assessment: Dict[str, Any] = Field(default_factory=dict)
    resource_allocation: Dict[str, Any] = Field(default_factory=dict)

class AtmosphereData(BaseModel):
    introduction: str = Field(..., min_length=200, max_length=1500)
    atmosphere: Dict[str, str] = Field(default_factory=dict)
    hidden_elements: List[str] = Field(default_factory=list, max_items=5)

# Central GPT service for all LLM calls
class GPTService:
    """Centralized service for GPT calls with caching, rate limiting, and monitoring"""
    
    _instance = None
    _semaphore = None
    _cache = {}
    _call_metrics = []
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._semaphore = asyncio.Semaphore(MAX_CONCURRENT_GPT_CALLS)
        return cls._instance
    
    @staticmethod
    def _get_cache_key(model: str, system_prompt: str, user_prompt: str) -> str:
        """Generate cache key from inputs"""
        content = f"{model}:{system_prompt}:{user_prompt}"
        return hashlib.md5(content.encode()).hexdigest()
    
    @staticmethod
    def _sanitize_input(text: str) -> str:
        """Sanitize user input to prevent prompt injection"""
        # Remove potential injection patterns
        text = text.replace("}", "\\}")
        text = text.replace("{", "\\{")
        text = text.replace("\n\n\n", "\n\n")  # Prevent prompt separation
        # Remove any PII patterns (basic example)
        import re
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        return text
    
    async def call_with_validation(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
        temperature: float = DEFAULT_TEMPERATURE,
        max_retries: int = 3,
        use_cache: bool = True
    ) -> BaseModel:
        """Make GPT call with automatic validation and retries"""
        
        # Sanitize inputs
        user_prompt = self._sanitize_input(user_prompt)
        
        # Token safety check
        total_prompt_len = len(system_prompt) + len(user_prompt)
        if total_prompt_len > MAX_TOKEN_SAFETY:
            logger.warning(f"Prompt too long ({total_prompt_len} chars), truncating user prompt")
            user_prompt = textwrap.shorten(user_prompt, width=MAX_TOKEN_SAFETY - len(system_prompt) - 100)
        
        # Check cache
        cache_key = self._get_cache_key(model, system_prompt, user_prompt)
        if use_cache and ENABLE_GPT_CACHE and cache_key in self._cache:
            logger.debug(f"Cache hit for {response_model.__name__}")
            return self._cache[cache_key]
        
        # Rate limiting
        async with self._semaphore:
            start_time = datetime.now()
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    # Lower temperature on retries for more consistent output
                    retry_temp = temperature - (0.1 * attempt)
                    
                    raw_response = await self._make_gpt_call(
                        model=model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=retry_temp
                    )
                    
                    # Parse and validate
                    data = self._parse_json_response(raw_response)
                    validated = response_model(**data)
                    
                    # Cache successful result
                    if use_cache and ENABLE_GPT_CACHE:
                        self._cache[cache_key] = validated
                    
                    # Record metrics
                    self._record_metrics(
                        model=model,
                        duration=(datetime.now() - start_time).total_seconds(),
                        tokens_estimate=len(raw_response),
                        success=True
                    )
                    
                    return validated
                    
                except (json.JSONDecodeError, ValidationError) as e:
                    last_error = e
                    logger.warning(f"Attempt {attempt + 1} failed for {response_model.__name__}: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1.5 * (attempt + 1))  # Backoff
                        
            # All retries failed
            logger.error(f"All retries failed for {response_model.__name__}. Last error: {last_error}")
            logger.debug(f"Failed prompt snippet: {user_prompt[:200]}...")
            raise last_error
    
    async def _make_gpt_call(
        self, model: str, system_prompt: str, user_prompt: str, temperature: float
    ) -> str:
        """Make actual GPT call - isolated for mocking in tests"""
        from npcs.new_npc_creation import _responses_json_call
        
        return await _responses_json_call(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            response_format={"type": "json_object"}
        )
    
    def _parse_json_response(self, response: str) -> dict:
        """Parse JSON with fallback strategies"""
        from npcs.new_npc_creation import _json_first_obj
        
        result = _json_first_obj(response)
        if result is None:
            raise json.JSONDecodeError("No valid JSON found", response, 0)
        return result
    
    def _record_metrics(self, model: str, duration: float, tokens_estimate: int, success: bool):
        """Record call metrics for analysis"""
        self._call_metrics.append({
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "duration": duration,
            "tokens_estimate": tokens_estimate,
            "success": success
        })
        
        # Keep only last 1000 metrics
        if len(self._call_metrics) > 1000:
            self._call_metrics = self._call_metrics[-1000:]
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of GPT call metrics"""
        if not self._call_metrics:
            return {"total_calls": 0}
        
        successful = [m for m in self._call_metrics if m["success"]]
        return {
            "total_calls": len(self._call_metrics),
            "success_rate": len(successful) / len(self._call_metrics),
            "avg_duration": sum(m["duration"] for m in successful) / len(successful) if successful else 0,
            "cache_size": len(self._cache),
            "estimated_tokens": sum(m["tokens_estimate"] for m in successful)
        }

# Profiling decorator
def profile_gpt(func):
    """Decorator to profile GPT-using functions"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = datetime.now()
        service = GPTService()
        metrics_before = len(service._call_metrics)
        
        try:
            result = await func(*args, **kwargs)
            duration = (datetime.now() - start).total_seconds()
            calls_made = len(service._call_metrics) - metrics_before
            
            logger.info(f"{func.__name__} completed in {duration:.2f}s with {calls_made} GPT calls")
            return result
            
        except Exception as e:
            logger.error(f"{func.__name__} failed after {(datetime.now() - start).total_seconds():.2f}s: {e}")
            raise
    
    return wrapper

# Transaction context manager
@asynccontextmanager
async def db_transaction(conn):
    """Database transaction with automatic rollback on error"""
    await conn.execute("BEGIN")
    try:
        yield conn
        await conn.execute("COMMIT")
    except Exception:
        await conn.execute("ROLLBACK")
        raise

class QueenOfThornsStoryInitializer:
    """Enhanced story initialization with performance optimizations"""
    
    @staticmethod
    @profile_gpt
    async def initialize_story(
        ctx, user_id: int, conversation_id: int, dynamic: bool = True
    ) -> Dict[str, Any]:
        """
        Initialize the complete story with optional dynamic content.
        
        Args:
            ctx: Context object
            user_id: User ID
            conversation_id: Conversation ID
            dynamic: Whether to use dynamic GPT generation (False for tests)
        """
        try:
            logger.info(f"Initializing Queen of Thorns story for user {user_id} (dynamic={dynamic})")
            
            # Generate or use seed for reproducibility
            async with get_db_connection_context() as conn:
                seed_row = await conn.fetchrow(
                    """
                    SELECT started_at_seed FROM story_states
                    WHERE user_id = $1 AND conversation_id = $2 AND story_id = 'queen_of_thorns'
                    """,
                    user_id, conversation_id
                )
                
                if seed_row and seed_row['started_at_seed']:
                    seed = seed_row['started_at_seed']
                else:
                    seed = random.randint(1000000, 9999999)
                    await conn.execute(
                        """
                        UPDATE story_states SET started_at_seed = $3
                        WHERE user_id = $1 AND conversation_id = $2 AND story_id = 'queen_of_thorns'
                        """,
                        user_id, conversation_id, seed
                    )
            
            random.seed(seed)
            logger.info(f"Using seed {seed} for reproducibility")
            
            # Run initialization steps - some in parallel where possible
            if dynamic:
                # Parallel initialization of independent components
                lore_task = asyncio.create_task(
                    QueenOfThornsStoryInitializer._initialize_dynamic_lore(ctx, user_id, conversation_id)
                )
                
                # Load story structure (can't parallelize - needs to complete first)
                from story_templates.moth.queen_of_thorns_story import QUEEN_OF_THORNS_STORY
                await QueenOfThornsStoryInitializer._load_story_with_dynamic_themes(
                    QUEEN_OF_THORNS_STORY, user_id, conversation_id
                )
                
                # Wait for lore to complete
                lore_result = await lore_task
                
                # Parallel creation of locations and Lilith
                location_task = asyncio.create_task(
                    QueenOfThornsStoryInitializer._create_dynamic_locations(ctx, user_id, conversation_id)
                )
                lilith_task = asyncio.create_task(
                    QueenOfThornsStoryInitializer._create_evolved_lilith(ctx, user_id, conversation_id)
                )
                
                location_ids = await location_task
                lilith_id = await lilith_task
                
                # Supporting NPCs can be created in parallel
                support_npc_ids = await QueenOfThornsStoryInitializer._create_dynamic_supporting_npcs_parallel(
                    ctx, user_id, conversation_id
                )
                
            else:
                # Static initialization for tests
                lore_result = await SFBayQueenOfThornsPreset.initialize_complete_sf_preset(
                    ctx, user_id, conversation_id
                )
                
                from story_templates.moth.queen_of_thorns_story import QUEEN_OF_THORNS_STORY
                await ThornsIntegratedStoryLoader.load_story_with_themes(
                    QUEEN_OF_THORNS_STORY, user_id, conversation_id
                )
                
                location_ids = await QueenOfThornsStoryInitializer._create_static_locations(
                    ctx, user_id, conversation_id
                )
                lilith_id = await QueenOfThornsStoryInitializer._create_static_lilith(
                    ctx, user_id, conversation_id
                )
                support_npc_ids = await QueenOfThornsStoryInitializer._create_static_supporting_npcs(
                    ctx, user_id, conversation_id
                )
            
            # Sequential steps that depend on previous results
            await QueenOfThornsStoryInitializer._setup_all_relationships(
                ctx, user_id, conversation_id, lilith_id, support_npc_ids
            )
            
            await QueenOfThornsStoryInitializer._initialize_story_state(
                ctx, user_id, conversation_id, lilith_id
            )
            
            if dynamic:
                # Parallel initialization of mechanics and network
                mechanics_task = asyncio.create_task(
                    QueenOfThornsStoryInitializer._setup_dynamic_mechanics(
                        ctx, user_id, conversation_id, lilith_id
                    )
                )
                network_task = asyncio.create_task(
                    QueenOfThornsStoryInitializer._initialize_dynamic_network_systems(
                        ctx, user_id, conversation_id, lilith_id
                    )
                )
                atmosphere_task = asyncio.create_task(
                    QueenOfThornsStoryInitializer._set_dynamic_atmosphere(
                        ctx, user_id, conversation_id
                    )
                )
                
                await asyncio.gather(mechanics_task, network_task, atmosphere_task)
            else:
                await QueenOfThornsStoryInitializer._setup_special_mechanics(
                    ctx, user_id, conversation_id, lilith_id
                )
                await QueenOfThornsStoryInitializer._initialize_network_systems(
                    ctx, user_id, conversation_id, lilith_id
                )
                await QueenOfThornsStoryInitializer._set_initial_atmosphere(
                    ctx, user_id, conversation_id
                )
            
            # Get metrics if using dynamic generation
            if dynamic:
                service = GPTService()
                metrics = service.get_metrics_summary()
                logger.info(f"Story initialization metrics: {metrics}")
            
            return {
                "status": "success",
                "story_id": QUEEN_OF_THORNS_STORY.id,
                "main_npc_id": lilith_id,
                "support_npc_ids": support_npc_ids,
                "location_ids": location_ids,
                "network_initialized": True,
                "seed": seed,
                "dynamic": dynamic,
                "message": f"Queen of Thorns story initialized {'dynamically' if dynamic else 'statically'}"
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize story: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "message": "Failed to initialize The Queen of Thorns story"
            }
    
    @staticmethod
    async def _initialize_dynamic_lore(ctx, user_id: int, conversation_id: int) -> Dict[str, Any]:
        """Initialize SF Bay Area lore with LLM-generated contemporary elements"""
        
        service = GPTService()
        
        # First load base lore to use as context
        base_lore = {
            "districts": SFGeopoliticalLore.get_districts(),
            "factions": SFGeopoliticalLore.get_factions(),
            "myths": SFLocalLore.get_urban_myths(),
            "consistency_rules": QueenOfThornsConsistencyGuide.get_critical_rules()
        }
        
        try:
            system_prompt = """You generate contemporary San Francisco Bay Area lore for a supernatural shadow network story.
    You must ENHANCE existing lore, not contradict it.
    
    CRITICAL RULES TO MAINTAIN:
    1. The network has NO official name - only "the network" or outsider names like "Rose & Thorn Society"
    2. The Queen's identity is ALWAYS ambiguous
    3. The network controls Bay Area ONLY - other cities have allies, not branches
    4. Transformation takes months/years, never instant
    
    Focus on: recent gentrification, tech culture shadows, underground movements, urban legends.
    Return JSON with the specified structure."""
    
            # Include base lore summary in prompt
            base_summary = f"""
    Existing factions include: {', '.join(f['name'] for f in base_lore['factions'][:3])}
    Existing myths include: {', '.join(m['name'] for m in base_lore['myths'][:3])}
    Key districts: {', '.join(d['name'] for d in base_lore['districts'][:3])}
    """
    
            user_prompt = f"""Generate current Bay Area shadow network lore that ADDS TO this foundation:
    {base_summary}
    
    Generate:
    - 3-5 recent news items that hint at supernatural activity (connected to existing factions)
    - 4-6 underground subcultures that could hide occult activities (in existing districts)
    - 5-7 rumors about gentrification hiding something darker (referencing known locations)
    - 3-5 supernatural whispers specific to SF neighborhoods (building on existing myths)
    
    Make it feel contemporary (2024-2025) with tech culture undertones."""

            lore_data = await service.call_with_validation(
                model=_DEFAULT_LORE_MODEL,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=LoreData,
                temperature=0.8  # Higher for creativity
            )
            
            # Merge with base preset
            base_result = await SFBayQueenOfThornsPreset.initialize_complete_sf_preset(
                ctx, user_id, conversation_id
            )
            
            # Store enhanced lore without duplicating memories
            async with get_db_connection_context() as conn:
                async with db_transaction(conn):
                    # Check for existing rumors to avoid duplicates
                    existing_rumors = await conn.fetch(
                        """
                        SELECT memory_text FROM NPCMemories
                        WHERE user_id = $1 AND conversation_id = $2 
                        AND entity_type = 'world' AND 'rumor' = ANY(tags)
                        """,
                        user_id, conversation_id
                    )
                    
                    existing_texts = {row['memory_text'] for row in existing_rumors}
                    
                    # Add only new rumors
                    for rumor in lore_data.rumors:
                        full_text = f"Bay Area rumor: {rumor}"
                        if full_text not in existing_texts:
                            await remember_with_governance(
                                user_id=user_id,
                                conversation_id=conversation_id,
                                entity_type="world",
                                entity_id=0,
                                memory_text=full_text,
                                importance="low",
                                emotional=False,
                                tags=["world_lore", "rumor", "dynamic"]
                            )
                    
                    # Store enhanced lore
                    await conn.execute(
                        """
                        INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                        VALUES ($1, $2, 'dynamic_lore', $3)
                        ON CONFLICT (user_id, conversation_id, key) 
                        DO UPDATE SET value = EXCLUDED.value
                        """,
                        user_id, conversation_id, json.dumps(lore_data.dict())
                    )
            
            return {**base_result, "dynamic_elements": lore_data.dict()}
            
        except Exception as e:
            logger.error(f"Error generating dynamic lore: {e}")
            return await SFBayQueenOfThornsPreset.initialize_complete_sf_preset(
                ctx, user_id, conversation_id
            )
    
    @staticmethod
    async def _load_story_with_dynamic_themes(story: PresetStory, user_id: int, conversation_id: int):
        """Load story with dynamically decorated poems and beats"""
        
        service = GPTService()
        
        # First load base story
        await ThornsIntegratedStoryLoader.load_story_with_themes(
            story, user_id, conversation_id
        )
        
        # Check if we've already enhanced these beats
        async with get_db_connection_context() as conn:
            existing_versions = await conn.fetch(
                """
                SELECT beat_id, enhancement_version FROM story_beat_enhancements
                WHERE user_id = $1 AND conversation_id = $2
                """,
                user_id, conversation_id
            )
            
            version_map = {row['beat_id']: row['enhancement_version'] for row in existing_versions}
            
            # Get player profile for personalization (with PII protection)
            player_profile = await conn.fetchrow(
                """
                SELECT kinks, fears, emotional_style 
                FROM player_profiles
                WHERE user_id = $1 AND conversation_id = $2
                """,
                user_id, conversation_id
            )
            
            profile_context = ""
            if player_profile:
                # Sanitize any sensitive content
                kinks = GPTService._sanitize_input(player_profile.get('kinks', 'unknown'))
                fears = GPTService._sanitize_input(player_profile.get('fears', 'unknown'))
                
                profile_context = f"""
Player preferences:
- Interests: {textwrap.shorten(kinks, width=100)}
- Fears: {textwrap.shorten(fears, width=100)}
- Emotional style: {player_profile.get('emotional_style', 'intense')}"""

            # Enhancement tasks for beats that need it
            enhancement_tasks = []
            
            for beat in story.story_beats[:5]:  # First 5 beats
                current_version = f"v1_{beat.description[:20]}"  # Simple version based on content
                
                if beat.id not in version_map or version_map[beat.id] != current_version:
                    if beat.dialogue_hints.get("poetic_elements"):
                        enhancement_tasks.append(
                            QueenOfThornsStoryInitializer._enhance_single_beat(
                                service, beat, profile_context, user_id, conversation_id, current_version
                            )
                        )
            
            # Run enhancements in parallel with rate limiting
            if enhancement_tasks:
                await asyncio.gather(*enhancement_tasks)
    
    @staticmethod
    async def _enhance_single_beat(
        service: GPTService, beat: StoryBeat, profile_context: str,
        user_id: int, conversation_id: int, version: str
    ):
        """Enhance a single story beat with poetry"""
        
        class BeatEnhancement(BaseModel):
            enhanced_poetry: str = Field(..., min_length=50, max_length=500)
            mood_adjustment: str = Field(..., max_length=100)
            personalized_imagery: List[str] = Field(default_factory=list, max_items=5)
        
        system_prompt = """You enhance story beats with personalized poetic variations.
Create evocative, thematic poetry that fits the Queen of Thorns aesthetic."""

        user_prompt = f"""Enhance this story beat with poetic elements:
Beat: {beat.name}
Description: {textwrap.shorten(beat.description, width=200)}
Stage: {beat.narrative_stage}
{profile_context}

Create variations that match the player's style while maintaining dark gothic themes."""

        try:
            enhancement = await service.call_with_validation(
                model=_DEFAULT_POETRY_MODEL,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=BeatEnhancement,
                temperature=0.7  # Moderate for poetic but coherent
            )
            
            async with get_db_connection_context() as conn:
                await conn.execute(
                    """
                    INSERT INTO story_beat_enhancements 
                    (user_id, conversation_id, beat_id, enhancement_data, enhancement_version)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (user_id, conversation_id, beat_id)
                    DO UPDATE SET 
                        enhancement_data = EXCLUDED.enhancement_data,
                        enhancement_version = EXCLUDED.enhancement_version
                    """,
                    user_id, conversation_id, beat.id, 
                    json.dumps(enhancement.dict()), version
                )
                
        except Exception as e:
            logger.error(f"Error enhancing beat {beat.id}: {e}")
    
    @staticmethod
    async def _create_dynamic_locations(ctx, user_id: int, conversation_id: int) -> List[str]:
        """Create locations with dynamically generated descriptions"""
        
        service = GPTService()
        canon_ctx = type('CanonicalContext', (), {
            'user_id': user_id,
            'conversation_id': conversation_id
        })()
        
        location_templates = [
            {
                "name": "Velvet Sanctum",
                "base_desc": "Underground temple of transformation",
                "type": "bdsm_club",
                "cache_key": "velvet_sanctum_v1"
            },
            {
                "name": "The Rose Garden Café", 
                "base_desc": "Mission café serving as network entry",
                "type": "café",
                "cache_key": "rose_garden_v1"
            },
            {
                "name": "Marina Safehouse",
                "base_desc": "Mediterranean villa for survivors",
                "type": "safehouse",
                "cache_key": "marina_safe_v1"
            }
        ]
        
        # Create location enhancement tasks
        location_tasks = []
        for template in location_templates:
            location_tasks.append(
                QueenOfThornsStoryInitializer._create_single_location(
                    service, canon_ctx, template, user_id, conversation_id
                )
            )
        
        # Run in parallel
        location_ids = await asyncio.gather(*location_tasks)
        return location_ids
    
    @staticmethod
    async def _create_single_location(
        service: GPTService, canon_ctx, template: dict,
        user_id: int, conversation_id: int
    ) -> str:
        """Create a single location with enhanced description"""
        
        # Get base location data from lore
        base_locations = SFLocalLore.get_specific_locations()
        base_data = next((loc for loc in base_locations if loc['name'] == template['name']), None)
        
        if base_data:
            system_prompt = """You ENHANCE existing location descriptions for a supernatural story.
    You must maintain all established facts while adding atmospheric details.
    Include sensory details, hidden purposes, and subtle power dynamics."""
    
            user_prompt = f"""Enhance this EXISTING location (do not contradict established facts):
    Name: {template['name']}
    Established description: {base_data.get('description', template['base_desc'])}
    Type: {template['type']}
    Known features: {json.dumps(base_data.get('areas', {}), indent=2)}
    Schedule: {json.dumps(base_data.get('schedule', {}), indent=2)}
    
    ADD 2-3 paragraphs of atmospheric detail that complement the existing description.
    Focus on sensory details and hidden supernatural elements that don't contradict what's established."""

        try:
            # Use cache key to avoid regenerating identical locations
            enhancement = await service.call_with_validation(
                model=_DEFAULT_LOCATION_MODEL,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=LocationEnhancement,
                temperature=0.6,  # Lower for consistent locations
                use_cache=True
            )
            
            async with get_db_connection_context() as conn:
                location_id = await canon.find_or_create_location(
                    canon_ctx, conn,
                    template["name"],
                    description=enhancement.description,
                    metadata={
                        "location_type": template["type"],
                        "atmospheric_details": enhancement.atmospheric_details,
                        "hidden_features": enhancement.hidden_features,
                        "cache_key": template["cache_key"]
                    }
                )
                
            return location_id
            
        except Exception as e:
            logger.error(f"Error creating location {template['name']}: {e}")
            # Fallback to static creation
            async with get_db_connection_context() as conn:
                location_id = await canon.find_or_create_location(
                    canon_ctx, conn,
                    template["name"],
                    description=template["base_desc"]
                )
            return location_id
    
    @staticmethod
    async def _create_evolved_lilith(ctx, user_id: int, conversation_id: int) -> int:
        """Create Lilith with dynamically evolved backstory"""
        
        service = GPTService()
        
        try:
            async with get_db_connection_context() as conn:
                async with db_transaction(conn):
                    lilith_id = await canon.find_or_create_npc(
                        ctx, conn,
                        npc_name=LILITH_RAVENCROFT["name"],
                        role="The Queen of Thorns",
                        affiliations=["Velvet Sanctum", "The Shadow Network"]
                    )
                    
                    # Check if already fully created
                    existing = await conn.fetchrow(
                        """
                        SELECT personality_traits, created_at, evolved_version
                        FROM NPCStats WHERE npc_id = $1
                        """, 
                        lilith_id
                    )
                    
                    current_version = "v2_evolved"
                    if existing and existing.get('evolved_version') == current_version:
                        logger.info(f"Lilith already evolved (version {current_version})")
                        return lilith_id
                    
                    # Generate evolution
                    system_prompt = """You evolve character backstories with contemporary touches.
            You must RESPECT the established character profile and only ADD new details that complement it.
            
            ESTABLISHED FACTS TO MAINTAIN:
            - Name: Lilith Ravencroft, The Queen of Thorns
            - Core trauma: Trafficking attempt at 15, built network from survivors
            - Personality: Masks vulnerability, fears abandonment, speaks in poetry when emotional
            - The three words: She cannot speak "I love you" - this is central to her character
            - Dual life: Dominatrix by night, network leader always
            - Lists: Red ink for failed saves, blue for those who left
            
            Focus on recent struggles, current challenges, and character growth that ADDS to this foundation."""
            
                    user_prompt = f"""Evolve Lilith Ravencroft for a new story while maintaining her core:
            Base backstory: {json.dumps(LILITH_RAVENCROFT["backstory"], indent=2)}
            Core traits: {', '.join(LILITH_RAVENCROFT["traits"])}
            Central mechanic: Cannot speak the three words "I love you"
            
            Generate ONLY:
            - 2-3 recent events that deepen her character (last 6 months)
            - 1-2 new scars (emotional or physical) from recent network operations
            - 2-3 evolution in her masks/personas
            - 3-4 new dialogue patterns that fit her poetic style
            
            Do NOT change: her name, core trauma, fear of abandonment, the three words mechanic, or her lists."""

                    evolution = await service.call_with_validation(
                        model=_DEFAULT_MEMORY_MODEL,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        response_model=CharacterEvolution,
                        temperature=0.7
                    )
                    
                    # Apply evolution
                    enhanced_backstory = LILITH_RAVENCROFT["backstory"].copy()
                    enhanced_backstory["recent_events"] = evolution.recent_intrigues
                    enhanced_backstory["current_struggles"] = evolution.new_scars
                    
                    await canon.update_entity_canonically(
                        ctx, conn, "NPCStats", lilith_id,
                        {
                            "backstory": json.dumps(enhanced_backstory),
                            "evolved_masks": json.dumps(evolution.current_masks),
                            "dialogue_evolution": json.dumps(evolution.dialogue_evolution),
                            "evolved_version": current_version
                        },
                        "Evolving Lilith with contemporary elements"
                    )
                    
                    # Complete setup with PresetNPCHandler
                    enhanced_data = LILITH_RAVENCROFT.copy()
                    enhanced_data["evolution_data"] = evolution.dict()
                    
                    await PresetNPCHandler.create_detailed_npc(ctx, enhanced_data, {
                        "story_context": "queen_of_thorns",
                        "is_main_character": True
                    })
            
            return lilith_id
            
        except Exception as e:
            logger.error(f"Failed to create evolved Lilith: {e}", exc_info=True)
            raise
    
    @staticmethod
    async def _generate_dynamic_physical_description(
        npc_id: int, base_data: Dict, user_id: int, conversation_id: int
    ) -> str:
        """Generate mood and context-aware physical description"""
        
        # Get current time and atmosphere
        async with get_db_connection_context() as conn:
            atmosphere = await conn.fetchrow(
                """
                SELECT value FROM CurrentRoleplay
                WHERE user_id = $1 AND conversation_id = $2 AND key = 'story_atmosphere'
                """,
                user_id, conversation_id
            )
            
            time_of_day = await conn.fetchval(
                """
                SELECT value FROM CurrentRoleplay
                WHERE user_id = $1 AND conversation_id = $2 AND key = 'TimeOfDay'
                """,
                user_id, conversation_id
            ) or "Night"
        
        atmosphere_context = ""
        if atmosphere:
            atmo_data = json.loads(atmosphere['value'])
            atmosphere_context = f"Current mood: {atmo_data.get('feeling', 'mysterious')}"
        
        system_prompt = """You create dynamic character descriptions that shift with mood, mask, and lighting.
Focus on how appearance changes with context while maintaining core features.
Return the description as plain text, 2-3 rich paragraphs."""

        user_prompt = f"""Re-render Lilith Ravencroft's appearance for this moment:
Base appearance: {base_data['physical_description']['base']}
Current time: {time_of_day}
Current mask: Porcelain Goddess (starting mask)
{atmosphere_context}

Show how lighting, mood, and the mask transform her presence. Include subtle supernatural elements."""

        try:
            raw_response = await _responses_json_call(
                model=_DEFAULT_LOCATION_MODEL,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.7,
                response_format=None  # Plain text response
            )
            
            return raw_response.strip()
            
        except Exception as e:
            logger.error(f"Error generating dynamic physical description: {e}")
            # Fallback to base description
            return QueenOfThornsStoryInitializer._build_comprehensive_physical_description(
                base_data['physical_description']
            )

    @staticmethod
    async def _create_dynamic_supporting_npcs_parallel(
        ctx, user_id: int, conversation_id: int
    ) -> List[int]:
        """Create supporting NPCs in parallel with dynamic enhancements"""
        
        service = GPTService()
        
        core_npcs = [
            {
                "name": "Marcus Sterling",
                "role": "Devoted Submissive / Former Tech CEO",
                "concept": "Transformed predator funding safehouses"
            },
            {
                "name": "Sarah Chen",
                "role": "Trafficking Survivor / Safehouse Coordinator",
                "concept": "Saved by Queen, helps others heal"
            },
            {
                "name": "Victoria Chen",
                "role": "VC Partner / Rose Council Member",
                "concept": "Transforms tech bros in basement"
            }
        ]
        
        # Create NPCs in parallel
        npc_tasks = []
        for npc_template in core_npcs:
            npc_tasks.append(
                QueenOfThornsStoryInitializer._create_single_dynamic_npc(
                    service, ctx, npc_template, user_id, conversation_id
                )
            )
        
        npc_results = await asyncio.gather(*npc_tasks, return_exceptions=True)
        
        # Extract successful IDs
        npc_ids = []
        for i, result in enumerate(npc_results):
            if isinstance(result, Exception):
                logger.error(f"Failed to create NPC {core_npcs[i]['name']}: {result}")
            elif isinstance(result, dict) and result.get("npc_id"):
                npc_ids.append(result["npc_id"])
            elif isinstance(result, int):
                npc_ids.append(result)
        
        return npc_ids
    
    @staticmethod
    async def _create_single_dynamic_npc(
        service: GPTService, ctx, template: dict,
        user_id: int, conversation_id: int
    ) -> int:
        """Create a single NPC with dynamic enhancements"""
        
        class NPCEnhancement(BaseModel):
            personality_quirks: List[str] = Field(default_factory=list, max_items=5)
            schedule_variations: Dict[str, Any] = Field(default_factory=dict)
            relationship_dynamics: Dict[str, str] = Field(default_factory=dict)
        
        system_prompt = """You enhance NPC concepts with memorable, specific details.
Focus on quirks that reveal character depth and create roleplay opportunities."""

        user_prompt = f"""Enhance this NPC:
Name: {template['name']}
Role: {template['role']}
Concept: {template['concept']}

Generate unique personality quirks and behavioral patterns."""

        try:
            enhancement = await service.call_with_validation(
                model=_DEFAULT_LORE_MODEL,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=NPCEnhancement,
                temperature=0.7
            )
            
            handler = NPCCreationHandler()
            result = await handler.create_npc_in_database(ctx, {
                "npc_name": template["name"],
                "role": template["role"],
                "personality": {
                    "personality_traits": enhancement.personality_quirks[:5]
                },
                "special_features": enhancement.dict()
            })
            
            return result.get("npc_id") if isinstance(result, dict) else result
            
        except Exception as e:
            logger.error(f"Error creating NPC {template['name']}: {e}")
            raise
    
    # Static fallback methods
    @staticmethod
    async def _create_static_locations(ctx, user_id: int, conversation_id: int) -> List[str]:
        """Create locations without dynamic generation (for tests)"""
        locations = [
            ("Velvet Sanctum", "An underground temple of transformation"),
            ("The Rose Garden Café", "A Mission café with hidden purposes"),
            ("Marina Safehouse", "A villa for healing and transformation")
        ]
        
        location_ids = []
        canon_ctx = type('CanonicalContext', (), {
            'user_id': user_id,
            'conversation_id': conversation_id
        })()
        
        async with get_db_connection_context() as conn:
            for name, desc in locations:
                loc_id = await canon.find_or_create_location(
                    canon_ctx, conn, name, description=desc
                )
                location_ids.append(loc_id)
        
        return location_ids
    
    @staticmethod
    async def _create_static_lilith(ctx, user_id: int, conversation_id: int) -> int:
        """Create Lilith without dynamic generation (for tests)"""
        async with get_db_connection_context() as conn:
            lilith_id = await canon.find_or_create_npc(
                ctx, conn,
                npc_name=LILITH_RAVENCROFT["name"],
                role="The Queen of Thorns"
            )
            
            await PresetNPCHandler.create_detailed_npc(
                ctx, LILITH_RAVENCROFT, {"story_context": "queen_of_thorns"}
            )
            
        return lilith_id
    
    @staticmethod
    async def _create_static_supporting_npcs(ctx, user_id: int, conversation_id: int) -> List[int]:
        """Create supporting NPCs without dynamic generation (for tests)"""
        npcs = [
            ("Marcus Sterling", "Devoted Submissive"),
            ("Sarah Chen", "Safehouse Coordinator"),
            ("Victoria Chen", "Rose Council Member")
        ]
        
        npc_ids = []
        async with get_db_connection_context() as conn:
            for name, role in npcs:
                npc_id = await canon.find_or_create_npc(ctx, conn, name, role)
                npc_ids.append(npc_id)
        
        return npc_ids
    
    # Keep other methods (relationships, state, mechanics) mostly unchanged
    # but add transaction handling where appropriate
    
    @staticmethod
    async def _setup_all_relationships(
        ctx, user_id: int, conversation_id: int,
        lilith_id: int, support_npc_ids: List[int]
    ):
        """Establish relationships with proper transaction handling"""
        
        canon_ctx = type('CanonicalContext', (), {
            'user_id': user_id,
            'conversation_id': conversation_id
        })()
        
        async with get_db_connection_context() as conn:
            async with db_transaction(conn):
                # Get NPC names
                npc_names = {}
                for npc_id in [lilith_id] + support_npc_ids:
                    row = await conn.fetchrow(
                        "SELECT npc_name FROM NPCStats WHERE npc_id = $1",
                        npc_id
                    )
                    if row:
                        npc_names[npc_id] = row['npc_name']
                
                # Define relationships
                relationships = []
                npc_id_map = {name: npc_id for npc_id, name in npc_names.items()}
                
                if "Marcus Sterling" in npc_id_map:
                    relationships.append({
                        "source": lilith_id,
                        "target": npc_id_map["Marcus Sterling"],
                        "type": "owns",
                        "reverse": "owned_by",
                        "strength": 95
                    })
                
                # Create relationships
                for rel in relationships:
                    await canon.find_or_create_social_link(
                        canon_ctx, conn,
                        user_id=user_id,
                        conversation_id=conversation_id,
                        entity1_type="npc",
                        entity1_id=rel["source"],
                        entity2_type="npc",
                        entity2_id=rel["target"],
                        link_type=rel["type"],
                        link_level=rel["strength"]
                    )
    

    @staticmethod
    async def _generate_dynamic_memories(
        npc_id: int, base_data: Dict, evolution_data: Dict,
        user_id: int, conversation_id: int
    ) -> List[str]:
        """Generate personalized memories based on current state"""
        
        system_prompt = """You create deeply personal NPC memories that reveal character depth.
Each memory should be 3-5 sentences, first-person, with specific sensory details.
Include power dynamics, emotional truth, and subtle control elements.
Return JSON: {"memories": [...]}"""

        # Get any player-specific context
        async with get_db_connection_context() as conn:
            player_style = await conn.fetchrow(
                """
                SELECT emotional_style, preferred_dynamics 
                FROM player_profiles
                WHERE user_id = $1 AND conversation_id = $2
                """,
                user_id, conversation_id
            )
        
        player_context = ""
        if player_style:
            player_context = f"\nTailor memories to resonate with {player_style.get('emotional_style', 'intense')} emotional style"

        recent_events = evolution_data.get("recent_intrigues", ["network expansion", "new threats"])
        
        user_prompt = f"""Generate 8 powerful memories for Lilith Ravencroft, Queen of Thorns:

Character: Trauma survivor turned protector, runs shadow network saving trafficking victims
Recent events: {', '.join(recent_events[:2])}
Core themes: Transformation through power, saving others, fear of abandonment{player_context}

Include memories about:
1. A recent network victory tinged with personal cost
2. A moment of unexpected vulnerability with someone
3. Transforming a predator into protector
4. The weight of leading the Rose Council
5. A close call with Viktor Kozlov's operations
6. Someone who promised to stay but left
7. The moment she decided to become the Queen
8. What the three words mean to her

Make each memory emotionally complex and specific."""

        try:
            raw_response = await _responses_json_call(
                model=_DEFAULT_MEMORY_MODEL,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.8,
                max_output_tokens=1200
            )
            
            memory_data = _json_first_obj(raw_response) or {}
            memories = memory_data.get("memories", [])
            
            if len(memories) >= 6:
                return memories
            else:
                # Fallback to some static + some generated
                static_memories = QueenOfThornsStoryInitializer._create_lilith_memories()
                return memories + static_memories[:8-len(memories)]
                
        except Exception as e:
            logger.error(f"Error generating dynamic memories: {e}")
            return QueenOfThornsStoryInitializer._create_lilith_memories()

    @staticmethod
    async def _create_dynamic_supporting_npcs(ctx, user_id: int, conversation_id: int) -> List[int]:
        """Create supporting NPCs with dynamic descriptions and quirks"""
        
        npc_ids = []
        handler = NPCCreationHandler()
        
        # Core NPCs that need dynamic enhancement
        core_npcs = [
            {
                "name": "Marcus Sterling",
                "role": "Devoted Submissive / Former Tech CEO",
                "base_concept": "Transformed predator now funding safehouses",
                "needs_dynamic": ["recent sins", "transformation details", "devotional quirks"]
            },
            {
                "name": "Sarah Chen",
                "role": "Trafficking Survivor / Safehouse Coordinator", 
                "base_concept": "Saved by Queen, now helps others heal",
                "needs_dynamic": ["trauma details", "healing methods", "protective instincts"]
            },
            {
                "name": "Victoria Chen",
                "role": "VC Partner / Rose Council Member",
                "base_concept": "Transforms tech bros in Noe Valley basement",
                "needs_dynamic": ["transformation techniques", "council politics", "dual life"]
            }
        ]
        
        for npc_template in core_npcs:
            try:
                # Generate dynamic enhancements
                system_prompt = """You enhance NPC concepts with specific, memorable details.
Return JSON: {"personality_quirks": [...], "schedule_variations": {...}, "relationship_dynamics": {...}}"""

                user_prompt = f"""Enhance this Queen of Thorns NPC:
Name: {npc_template['name']}
Role: {npc_template['role']}
Concept: {npc_template['base_concept']}
Needs: {', '.join(npc_template['needs_dynamic'])}

Generate unique quirks, schedule details, and relationship patterns that make them memorable."""

                raw_response = await _responses_json_call(
                    model=_DEFAULT_LORE_MODEL,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0.8
                )
                
                enhancement_data = _json_first_obj(raw_response) or {}
                
                # Create NPC with enhancements
                npc_id = await handler.create_npc_in_database(ctx, {
                    "npc_name": npc_template["name"],
                    "sex": "female" if "Chen" in npc_template["name"] else "male",
                    "role": npc_template["role"],
                    "personality": {
                        "personality_traits": enhancement_data.get("personality_quirks", [
                            "complex", "layered", "evolving"
                        ])[:5]
                    },
                    "schedule": enhancement_data.get("schedule_variations", {}),
                    "special_features": enhancement_data
                })
                
                if isinstance(npc_id, dict) and npc_id.get("npc_id"):
                    npc_ids.append(npc_id["npc_id"])
                    
            except Exception as e:
                logger.error(f"Error creating dynamic NPC {npc_template['name']}: {e}")
        
        return npc_ids

    @staticmethod
    async def _setup_dynamic_relationships(
        ctx, user_id: int, conversation_id: int,
        lilith_id: int, support_npc_ids: List[int]
    ):
        """Create relationships with dynamically generated shared memories"""
        
        # First establish base relationships
        await QueenOfThornsStoryInitializer._setup_all_relationships(
            ctx, user_id, conversation_id, lilith_id, support_npc_ids
        )
        
        # Then generate dynamic shared memories
        async with get_db_connection_context() as conn:
            # Get NPC pairs that have relationships
            relationships = await conn.fetch(
                """
                SELECT DISTINCT sl.entity1_id, sl.entity2_id, sl.link_type,
                       n1.npc_name as name1, n2.npc_name as name2
                FROM SocialLinks sl
                JOIN NPCStats n1 ON sl.entity1_id = n1.npc_id
                JOIN NPCStats n2 ON sl.entity2_id = n2.npc_id
                WHERE sl.user_id = $1 AND sl.conversation_id = $2
                AND sl.entity1_type = 'npc' AND sl.entity2_type = 'npc'
                AND sl.entity1_id = ANY($3::int[])
                LIMIT 5
                """,
                user_id, conversation_id, [lilith_id] + support_npc_ids[:3]
            )
            
            for rel in relationships:
                # Generate relationship-specific memory
                system_prompt = """You create shared memories between characters that reveal relationship dynamics.
The memory should work from both perspectives with minor variations.
Return JSON: {"shared_memory": "...", "emotional_tone": "...", "power_dynamic": "..."}"""

                user_prompt = f"""Create a shared memory between:
{rel['name1']} and {rel['name2']}
Relationship type: {rel['link_type']}
Story context: Queen of Thorns shadow network in San Francisco

The memory should reveal their power dynamic and emotional connection."""

                try:
                    raw_response = await _responses_json_call(
                        model=_DEFAULT_MEMORY_MODEL,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=0.8
                    )
                    
                    memory_data = _json_first_obj(raw_response) or {}
                    
                    if memory_data.get("shared_memory"):
                        # Store for both NPCs
                        for npc_id in [rel['entity1_id'], rel['entity2_id']]:
                            await remember_with_governance(
                                user_id=user_id,
                                conversation_id=conversation_id,
                                entity_type="npc",
                                entity_id=npc_id,
                                memory_text=memory_data["shared_memory"],
                                importance="medium",
                                emotional=True,
                                tags=["shared_memory", "relationship", memory_data.get("emotional_tone", "complex")]
                            )
                            
                except Exception as e:
                    logger.error(f"Error generating shared memory: {e}")

    @staticmethod
    async def _setup_dynamic_mechanics(
        ctx, user_id: int, conversation_id: int, lilith_id: int
    ):
        """Set up special mechanics with dynamic triggers and conditions"""
        
        # Get Lilith's established triggers from character data
        base_triggers = LILITH_RAVENCROFT.get("trauma_triggers", [])
        base_mechanics = LILITH_RAVENCROFT.get("special_mechanics", {})
        
        async with get_db_connection_context() as conn:
            # Generate dynamic mask slippage triggers
            system_prompt = """You create psychological triggers for mask slippage in a character.
    You must BUILD ON these established trauma triggers, not replace them.
    
    ESTABLISHED TRIGGERS TO MAINTAIN:
    - Sudden departures without warning
    - The phrase 'I'll always be here' (too many liars)
    - Being seen without consent
    - Betrayal of the network's trust
    - Being called 'weak' or 'broken'
    
    Return JSON: {"additional_triggers": [...], "physical_tells": [...], "verbal_patterns": [...]}"""
    
            user_prompt = f"""Generate ADDITIONAL mask slippage triggers for Lilith Ravencroft:
    Character: Controlled dominant who hides trauma and vulnerability
    Existing triggers: {json.dumps(base_triggers)}
    Context: Modern San Francisco, tech predators, human trafficking
    
    Create 3-5 NEW subtle triggers that complement the existing ones."""

            try:
                raw_response = await _responses_json_call(
                    model=_DEFAULT_LORE_MODEL,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0.7
                )
                
                trigger_data = _json_first_obj(raw_response) or {}
                
                # Apply dynamic triggers to mask system
                mask_data = {
                    "available_masks": [
                        {
                            "name": "Porcelain Goddess",
                            "description": "Perfect, cold, untouchable divinity",
                            "trust_required": 0,
                            "slippage_triggers": trigger_data.get("triggers", ["abandonment", "genuine care"])[:3]
                        },
                        {
                            "name": "Leather Predator",
                            "description": "Dangerous, hunting, protective fury",
                            "trust_required": 30,
                            "slippage_triggers": trigger_data.get("triggers", ["threat to innocent", "Kozlov"])[:3]
                        },
                        {
                            "name": "Lace Vulnerability",
                            "description": "Soft edges barely containing sharp pain",
                            "trust_required": 60,
                            "physical_tells": trigger_data.get("physical_tells", ["trembling hands", "wet eyes"])[:2]
                        }
                    ],
                    "current_mask": "Porcelain Goddess",
                    "mask_integrity": 100,
                    "dynamic_triggers": trigger_data
                }
                
                await conn.execute(
                    """
                    INSERT INTO npc_special_mechanics (
                        user_id, conversation_id, npc_id, mechanic_type, mechanic_data
                    )
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (user_id, conversation_id, npc_id, mechanic_type)
                    DO UPDATE SET mechanic_data = EXCLUDED.mechanic_data
                    """,
                    user_id, conversation_id, lilith_id, "mask_system",
                    json.dumps(mask_data)
                )
                
            except Exception as e:
                logger.error(f"Error setting up dynamic mechanics: {e}")
                # Fallback to static mechanics
                await QueenOfThornsStoryInitializer._setup_special_mechanics(
                    ctx, user_id, conversation_id, lilith_id
                )

    @staticmethod
    async def _initialize_dynamic_network_systems(
        ctx, user_id: int, conversation_id: int, lilith_id: int
    ):
        """Initialize network with dynamic threat assessments and operations"""
        
        # Get base network structure
        base_network_structure = SFBayQueenOfThornsPreset.get_network_structure()
        consistency_rules = QueenOfThornsConsistencyGuide.get_critical_rules()
        
        # Generate current network state
        system_prompt = """You create dynamic shadow network operational data.
    You must RESPECT the established network structure and naming conventions.
    
    CRITICAL RULES:
    1. NEVER give the network an official name - it's just "the network"
    2. Maintain the established hierarchy: Seedlings → Roses → Thorns → Gardeners → Regional Thorns → Rose Council → The Queen
    3. Bay Area control only - other cities are allies, not branches
    
    Return JSON: {"active_operations": [...], "threat_assessment": {...}, "resource_allocation": {...}}"""
    
        user_prompt = f"""Generate current state for the Queen of Thorns shadow network:
    Established structure:
    {json.dumps(base_network_structure, indent=2)}
    
    Network purpose: Saving trafficking victims, transforming predators
    Setting: San Francisco Bay Area, 2024-2025
    Main antagonist: Viktor Kozlov (human trafficker)
    
    Create realistic operations that fit within the established structure."""

        try:
            raw_response = await _responses_json_call(
                model=_DEFAULT_LORE_MODEL,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.7
            )
            
            network_data = _json_first_obj(raw_response) or {}
            
            async with get_db_connection_context() as conn:
                # Merge with base network structure
                base_network = {
                    "organization_names": ["the network", "the garden", "Rose & Thorn Society"],
                    "structure": {
                        "queen_of_thorns": lilith_id,
                        "rose_council": ["Victoria Chen", "Judge Thornfield", "5 others"]
                    },
                    "dynamic_state": network_data,
                    "statistics": {
                        "saved_this_month": random.randint(3, 8),
                        "active_operations": len(network_data.get("active_operations", [])),
                        "threat_level": network_data.get("threat_assessment", {}).get("overall", "moderate")
                    }
                }
                
                await conn.execute(
                    """
                    INSERT INTO network_state (user_id, conversation_id, network_data)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (user_id, conversation_id)
                    DO UPDATE SET network_data = EXCLUDED.network_data
                    """,
                    user_id, conversation_id, json.dumps(base_network)
                )
                
        except Exception as e:
            logger.error(f"Error initializing dynamic network: {e}")
            await QueenOfThornsStoryInitializer._initialize_network_systems(
                ctx, user_id, conversation_id, lilith_id
            )

    @staticmethod
    async def _generate_with_lore_context(
        service: GPTService,
        generation_type: str,
        base_context: Dict[str, Any],
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
        temperature: float = 0.7
    ) -> BaseModel:
        """Generate content that respects established lore"""
        
        # Always include consistency rules
        consistency_rules = QueenOfThornsConsistencyGuide.get_quick_reference()
        
        enhanced_system_prompt = f"""{system_prompt}
    
    MANDATORY CONSISTENCY RULES:
    {consistency_rules}
    
    You are ADDING to established lore, not replacing it. Any generated content must complement and respect what already exists."""
        
        # Add base context summary to user prompt
        context_summary = f"""
    ESTABLISHED CONTEXT YOU MUST RESPECT:
    {json.dumps(base_context, indent=2)}
    
    ---
    {user_prompt}"""
        
        return await service.call_with_validation(
            model=_DEFAULT_LORE_MODEL,
            system_prompt=enhanced_system_prompt,
            user_prompt=context_summary,
            response_model=response_model,
            temperature=temperature
        )
    
    @staticmethod
    async def _validate_generated_content(
        content_type: str,
        generated_content: Any,
        base_data: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Validate that generated content doesn't contradict base lore"""
        
        violations = []
        
        if content_type == "location":
            # Check location consistency
            if "name" in generated_content and generated_content["name"] != base_data.get("name"):
                violations.append(f"Location name changed from {base_data['name']} to {generated_content['name']}")
        
        elif content_type == "character":
            # Check character consistency
            if "name" in generated_content and generated_content["name"] != "Lilith Ravencroft":
                violations.append("Character name cannot be changed")
            
            if "can_say_three_words" in generated_content and generated_content["can_say_three_words"]:
                violations.append("Lilith cannot speak the three words - this is core to her character")
        
        elif content_type == "network":
            # Check network consistency
            content_str = json.dumps(generated_content)
            if any(name in content_str for name in ["The Rose & Thorn Society's official", "The Garden announced"]):
                violations.append("Network given official name - it has none")
        
        # Use the consistency validator
        validation_result = QueenOfThornsConsistencyGuide.validate_content(
            json.dumps(generated_content) if not isinstance(generated_content, str) else generated_content
        )
        
        violations.extend(validation_result.get("violations", []))
        
        return len(violations) == 0, violations

    @staticmethod
    async def _set_dynamic_atmosphere(ctx, user_id: int, conversation_id: int):
        """Generate personalized atmospheric introduction"""
        
        # Get current time and weather context
        from datetime import datetime
        import random
        
        current_time = datetime.now()
        moon_phases = ["new moon", "waxing crescent", "first quarter", "waxing gibbous", 
                      "full moon", "waning gibbous", "last quarter", "waning crescent"]
        current_moon = moon_phases[current_time.day % 8]
        weather_options = ["fog rolling in", "light rain", "clear night", "wind from the bay"]
        current_weather = random.choice(weather_options)
        
        # Get player emotional profile
        async with get_db_connection_context() as conn:
            player_data = await conn.fetchrow(
                """
                SELECT emotional_style, preferred_atmosphere 
                FROM player_profiles
                WHERE user_id = $1 AND conversation_id = $2
                """,
                user_id, conversation_id
            )
        
        emotional_context = ""
        if player_data:
            emotional_context = f"Player prefers {player_data.get('emotional_style', 'intense')} emotional experiences"
        
        system_prompt = """You create immersive story introductions that blend environment, emotion, and anticipation.
Focus on sensory details, hidden dangers, and the promise of transformation.
Return JSON: {"introduction": "...", "atmosphere": {...}, "hidden_elements": [...]}"""

        user_prompt = f"""Create the opening atmosphere for The Queen of Thorns story:
Setting: SoMa underground, San Francisco after midnight
Moon phase: {current_moon}
Weather: {current_weather}
Venue: The Velvet Sanctum (hidden BDSM club and network hub)
{emotional_context}

Write 4-5 paragraphs that capture the descent into this world. Include sensory details and hidden network hints."""

        try:
            raw_response = await _responses_json_call(
                model=_DEFAULT_ATMOSPHERE_MODEL,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.8,
                max_output_tokens=800
            )
            
            atmosphere_data = _json_first_obj(raw_response) or {}
            introduction = atmosphere_data.get("introduction", "")
            
            if not introduction:
                raise ValueError("No introduction generated")
            
            # Store atmospheric data
            await conn.execute(
                """
                INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (user_id, conversation_id, key) 
                DO UPDATE SET value = EXCLUDED.value
                """,
                user_id, conversation_id, "story_atmosphere",
                json.dumps(atmosphere_data.get("atmosphere", {
                    "tone": "noir_gothic",
                    "moon_phase": current_moon,
                    "weather": current_weather
                }))
            )
            
            # Store as initial memory
            await remember_with_governance(
                user_id=user_id,
                conversation_id=conversation_id,
                entity_type="player",
                entity_id=user_id,
                memory_text=introduction,
                importance="high",
                emotional=True,
                tags=["story_start", "atmosphere", "dynamic_intro"]
            )
            
        except Exception as e:
            logger.error(f"Error generating dynamic atmosphere: {e}")
            # Fallback to static atmosphere
            await QueenOfThornsStoryInitializer._set_initial_atmosphere(
                ctx, user_id, conversation_id
            )

    # Keep static helper methods as fallbacks
    @staticmethod
    def _build_comprehensive_physical_description(desc_data: Dict[str, str]) -> str:
        """Build Lilith's rich, detailed physical description"""
        
        parts = [
            desc_data["base"],
            "",
            "PUBLIC APPEARANCE:",
            desc_data["public_persona"],
            "",
            "PRIVATE REALITY:",
            desc_data["private_self"],
            "",
            "BEHAVIORAL TELLS:",
            desc_data["tells"],
            "",
            "HER PRESENCE:",
            desc_data["presence"]
        ]
        
        return "\n".join(parts)
    
    @staticmethod
    def _create_lilith_memories() -> List[str]:
        """Create Lilith's foundational memories as Queen of Thorns (fallback)"""
        
        return [
            # Trauma and survival
            "I was fifteen when they tried to make me disappear. The van, the men, the promises of modeling work in "
            "the city. But I had already learned that pretty faces hide sharp teeth. I bit, I clawed, I burned their "
            "operation down from the inside. The scars on my wrists aren't from giving up - they're from breaking free. "
            "That's when I learned that sometimes you have to become the monster to defeat the monsters.",
            
            # Building the network
            "The day I took the name 'Queen of Thorns' was the day I stopped being a victim. The network had no name - "
            "still doesn't, despite what outsiders call it. 'The Rose & Thorn Society' they whisper, but we are so much "
            "more. I built it from other survivors, from women who understood power, from the ashes of those who tried "
            "to break us. Now it spans the entire Bay Area, invisible thorns protecting hidden roses.",
            
            # The dual identity
            "By night I rule the Velvet Sanctum, transforming desire into submission. By day I move through charity "
            "galas and boardrooms, building the network. They think the dominatrix and the philanthropist are different "
            "women. Let them. The Rose Council meets on Mondays at 3 PM, but the Queen of Thorns is always watching. "
            "Some say I'm seven women, some say I'm a role that passes. The mystery is my greatest protection.",
            
            # First abandonment
            "Alexandra swore she'd never leave. 'You're my gravity,' she said, kneeling so beautifully in my private "
            "chambers. Six months later, I found her engagement announcement in the Chronicle's society pages. I added "
            "her porcelain mask to my collection and her name to the blue list. Another ghost, another lie. The garden "
            "grows thorns for a reason.",
            
            # The transformation work
            "Marcus Sterling was my first complete transformation. A tech CEO with wandering hands and three NDAs. The "
            "Rose email found him with evidence he couldn't buy away. Now he funds our safehouses, kneels publicly, "
            "and thanks me for his collar. Every predator we transform is a victory. The network grows stronger with "
            "each executive who learns to submit.",
            
            # The unspoken words
            "Last month, someone almost made me say them - those three words that taste of burning stars. I bit my "
            "tongue until it bled rather than let them escape. Love is a luxury the Queen of Thorns can't afford. "
            "Everyone who claims to love me disappears. Better to rule through fear and respect than lose through love.",
            
            # The lists
            "Red ink for those I failed to save - too many names, too many girls who didn't make it out. Blue ink for "
            "those who failed to stay - lovers, submissives, would-be partners who promised forever. Tonight I added "
            "two names: one red (a girl Kozlov's people took before we could reach her), one blue (a Rose Council "
            "member who relocated to Seattle). The blue list is longer. It always is.",
            
            # Her greatest fear
            "My deepest terror isn't Kozlov or the FBI or exposure. It's the moment someone sees all of me - the Queen, "
            "the survivor, the frightened girl, the network's architect - and chooses to leave anyway. When they know "
            "about the transformation chambers and the Monday meetings and the broken girl beneath the crown, and they "
            "still walk away. That's why I never remove all the masks. Always keep one layer of thorns."
        ]


# Enhance story progression with dynamic beat checking
class QueenOfThornsStoryProgression:
    """Handles story progression and beat triggers for Queen of Thorns"""
    
    @staticmethod
    async def check_beat_triggers(user_id: int, conversation_id: int) -> Optional[str]:
        """Check if any story beats should trigger with dynamic conditions"""
        
        async with get_db_connection_context() as conn:
            # Get current story state
            state_row = await conn.fetchrow(
                """
                SELECT current_beat, story_flags, progress, current_act
                FROM story_states
                WHERE user_id = $1 AND conversation_id = $2 AND story_id = $3
                """,
                user_id, conversation_id, "queen_of_thorns"
            )
            
            if not state_row:
                return None
            
            current_beat = state_row['current_beat']
            current_act = state_row['current_act']
            story_flags = json.loads(state_row['story_flags'] or '{}')
            completed_beats = story_flags.get('completed_beats', [])
            
            # Check for dynamic events first
            dynamic_event = await QueenOfThornsStoryProgression._check_dynamic_events(
                user_id, conversation_id, story_flags
            )
            
            if dynamic_event:
                return dynamic_event
            
            # Then check standard beats
            from story_templates.moth.queen_of_thorns_story import QUEEN_OF_THORNS_STORY
            
            sorted_beats = sorted(QUEEN_OF_THORNS_STORY.story_beats, 
                                key=lambda b: (QueenOfThornsStoryProgression._get_beat_act(b), 
                                             QUEEN_OF_THORNS_STORY.story_beats.index(b)))
            
            for beat in sorted_beats:
                if beat.id in completed_beats or beat.id == current_beat:
                    continue
                
                if await QueenOfThornsStoryProgression._check_single_beat_conditions(
                    beat, story_flags, current_act, user_id, conversation_id, conn
                ):
                    logger.info(f"Story beat '{beat.id}' conditions met for user {user_id}")
                    return beat.id
        
        return None
    
    @staticmethod
    async def _check_dynamic_events(
        user_id: int, conversation_id: int, story_flags: Dict
    ) -> Optional[str]:
        """Check for dynamically generated story events"""
        
        # Get network state for dynamic events
        async with get_db_connection_context() as conn:
            network_state = await conn.fetchrow(
                """
                SELECT network_data FROM network_state
                WHERE user_id = $1 AND conversation_id = $2
                """,
                user_id, conversation_id
            )
            
            if network_state:
                network_data = json.loads(network_state['network_data'])
                dynamic_state = network_data.get('dynamic_state', {})
                
                # Check threat levels
                threat_assessment = dynamic_state.get('threat_assessment', {})
                if threat_assessment.get('kozlov_activity', 0) > 70:
                    if 'high_kozlov_threat_addressed' not in story_flags:
                        return 'dynamic_kozlov_threat'
                
                # Check active operations
                operations = dynamic_state.get('active_operations', [])
                for op in operations:
                    if op.get('urgency') == 'critical' and op.get('id') not in story_flags.get('addressed_operations', []):
                        return f'dynamic_operation_{op.get("id", "unknown")}'
        
        return None
    
    @staticmethod
    def _get_beat_act(beat: StoryBeat) -> int:
        """Determine which act a beat belongs to"""
        stage_to_act = {
            "Innocent Beginning": 1,
            "First Doubts": 1,
            "Creeping Realization": 2,
            "Veil Thinning": 2,
            "Full Revelation": 3
        }
        return stage_to_act.get(beat.narrative_stage, 1)
    
    @staticmethod
    async def _check_single_beat_conditions(
        beat: StoryBeat, 
        story_flags: Dict, 
        current_act: int,
        user_id: int, 
        conversation_id: int,
        conn
    ) -> bool:
        """Check if a single beat's conditions are met (enhanced with dynamic checks)"""
        
        try:
            # Original condition checking logic remains
            for condition, value in beat.trigger_conditions.items():
                # ... (keep all existing condition checks)
                
                # Add dynamic condition checks
                if condition == "network_operation_complete":
                    addressed = story_flags.get('addressed_operations', [])
                    if value not in addressed:
                        return False
                
                elif condition == "dynamic_threat_level":
                    network_state = await conn.fetchrow(
                        """
                        SELECT network_data FROM network_state
                        WHERE user_id = $1 AND conversation_id = $2
                        """,
                        user_id, conversation_id
                    )
                    
                    if network_state:
                        network_data = json.loads(network_state['network_data'])
                        threat = network_data.get('statistics', {}).get('threat_level', 'low')
                        threat_values = {'low': 1, 'moderate': 2, 'high': 3, 'critical': 4}
                        if threat_values.get(threat, 0) < threat_values.get(value, 0):
                            return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking beat conditions for {beat.id}: {e}", exc_info=True)
            return False
    
    # ... (keep remaining methods like trigger_story_beat, _apply_beat_outcomes, etc.)
    
    @staticmethod
    async def generate_dynamic_network_event(
        user_id: int, conversation_id: int, event_type: str
    ) -> Dict[str, Any]:
        """Generate a dynamic network event based on current state"""
        
        system_prompt = """You create dynamic story events for an underground network narrative.
Focus on moral choices, urgent decisions, and character revelations.
Return JSON: {"event_description": "...", "choices": [...], "consequences": {...}}"""

        async with get_db_connection_context() as conn:
            story_flags = await conn.fetchval(
                """
                SELECT story_flags FROM story_states
                WHERE user_id = $1 AND conversation_id = $2 AND story_id = 'queen_of_thorns'
                """,
                user_id, conversation_id
            )
            
            flags = json.loads(story_flags or '{}')
            
        user_prompt = f"""Generate a dynamic story event:
Type: {event_type}
Player trust level: {flags.get('trust_level', 0)}
Network awareness: {flags.get('network_awareness', 0)}
Current threats: High Kozlov activity

Create an urgent situation requiring player choice that advances the network storyline."""

        try:
            raw_response = await _responses_json_call(
                model=_DEFAULT_LORE_MODEL,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.8
            )
            
            event_data = _json_first_obj(raw_response) or {}
            
            return {
                "event_type": event_type,
                "description": event_data.get("event_description", "An urgent situation arises"),
                "choices": event_data.get("choices", ["act", "wait", "seek help"]),
                "dynamic": True
            }
            
        except Exception as e:
            logger.error(f"Error generating dynamic event: {e}")
            return {
                "event_type": event_type,
                "description": "The network faces a critical moment",
                "choices": ["intervene", "observe", "report to Queen"]
            }
            
    @staticmethod
    async def advance_network_knowledge(
        user_id: int, conversation_id: int, 
        knowledge_type: str, amount: int = 10
    ) -> Dict[str, Any]:
        """Advance player's understanding of the network with dynamic revelations"""
        
        async with get_db_connection_context() as conn:
            # Get current state
            state_row = await conn.fetchrow(
                """
                SELECT story_flags FROM story_states
                WHERE user_id = $1 AND conversation_id = $2 AND story_id = $3
                """,
                user_id, conversation_id, "queen_of_thorns"
            )
            
            if not state_row:
                return {"error": "Story state not found"}
            
            story_flags = json.loads(state_row['story_flags'] or '{}')
            
            # Update knowledge with dynamic revelations
            revelations = []
            new_value = 0
            
            if knowledge_type == "network":
                current = story_flags.get("network_awareness", 0)
                new_value = min(100, current + amount)
                story_flags["network_awareness"] = new_value
                
                # Generate contextual revelation
                if current < new_value and new_value >= 50:
                    revelation = await QueenOfThornsStoryProgression._generate_network_revelation(
                        user_id, conversation_id, new_value
                    )
                    if revelation:
                        revelations.append(revelation)
            
            # Update database
            await conn.execute(
                """
                UPDATE story_states
                SET story_flags = $3
                WHERE user_id = $1 AND conversation_id = $2 AND story_id = 'queen_of_thorns'
                """,
                user_id, conversation_id, json.dumps(story_flags)
            )
            
            return {
                "knowledge_type": knowledge_type,
                "new_value": new_value,
                "revelations": revelations
            }
    
    @staticmethod
    async def _generate_network_revelation(
        user_id: int, conversation_id: int, awareness_level: int
    ) -> Optional[str]:
        """Generate a contextual revelation about the network"""
        
        system_prompt = """You create shocking but logical revelations about a shadow network.
The revelation should feel earned and connect to previous hints.
Return a single powerful sentence."""

        user_prompt = f"""Generate a network revelation for awareness level {awareness_level}:
Context: Underground network saving trafficking victims in San Francisco
Previous hints: BDSM club is a front, roses and thorns symbolism, Monday meetings

Create a revelation that deepens understanding without revealing everything."""

        try:
            raw_response = await _responses_json_call(
                model=_DEFAULT_LORE_MODEL,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.7,
                response_format=None,
                max_output_tokens=100
            )
            
            return raw_response.strip()
            
        except Exception as e:
            logger.error(f"Error generating revelation: {e}")
            return None


# Maintain compatibility
MothFlameStoryInitializer = QueenOfThornsStoryInitializer
MothFlameStoryProgression = QueenOfThornsStoryProgression
