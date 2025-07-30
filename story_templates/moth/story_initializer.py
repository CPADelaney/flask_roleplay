# story_templates/moth/story_initializer.py
"""
Complete story initialization system for The Queen of Thorns
Refactored with proper canon integration and governance approval
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
from lore.core.context import CanonicalContext
from memory.wrapper import MemorySystem
from story_templates.preset_stories import StoryBeat, PresetStory
from nyx.integrate import remember_with_governance
from nyx.governance_helpers import (
    propose_canonical_change, 
    create_canonical_entity,
    with_canon_governance
)
from nyx.governance import AgentType
from npcs.preset_npc_handler import PresetNPCHandler
from story_templates.moth.lore.world_lore_manager import (
    SFBayQueenOfThornsPreset,
    QueenOfThornsConsistencyGuide
)
from logic.chatgpt_integration import get_async_openai_client
from embedding.vector_store import generate_embedding

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

# Semantic similarity threshold for duplicate detection
SIMILARITY_THRESHOLD = 0.85

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

# Custom exceptions
class CanonError(Exception):
    """Raised when canon operations fail"""
    pass

class GovernanceError(Exception):
    """Raised when governance rejects an operation"""
    pass

class ConsistencyError(Exception):
    """Raised when content violates consistency rules"""
    pass

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
        text = text.replace("}", "\\}")
        text = text.replace("{", "\\{")
        text = text.replace("\n\n\n", "\n\n")
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
        
        user_prompt = self._sanitize_input(user_prompt)
        
        total_prompt_len = len(system_prompt) + len(user_prompt)
        if total_prompt_len > MAX_TOKEN_SAFETY:
            logger.warning(f"Prompt too long ({total_prompt_len} chars), truncating user prompt")
            user_prompt = textwrap.shorten(user_prompt, width=MAX_TOKEN_SAFETY - len(system_prompt) - 100)
        
        cache_key = self._get_cache_key(model, system_prompt, user_prompt)
        if use_cache and ENABLE_GPT_CACHE and cache_key in self._cache:
            logger.debug(f"Cache hit for {response_model.__name__}")
            return self._cache[cache_key]
        
        async with self._semaphore:
            start_time = datetime.now()
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    retry_temp = temperature - (0.1 * attempt)
                    
                    raw_response = await self._make_gpt_call(
                        model=model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=retry_temp
                    )
                    
                    data = self._parse_json_response(raw_response)
                    validated = response_model(**data)
                    
                    if use_cache and ENABLE_GPT_CACHE:
                        self._cache[cache_key] = validated
                    
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
                        await asyncio.sleep(1.5 * (attempt + 1))
                        
            logger.error(f"All retries failed for {response_model.__name__}. Last error: {last_error}")
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

# Canon integration service
class CanonIntegrationService:
    """Service for proper canon integration with duplicate detection"""
    
    @staticmethod
    async def check_semantic_duplicate(
        ctx: CanonicalContext,
        entity_type: str,
        content: str,
        metadata: Dict[str, Any] = None
    ) -> Optional[Dict[str, Any]]:
        """Check if semantically similar content already exists in canon"""
        
        embedding = await generate_embedding(content)
        
        async with get_db_connection_context() as conn:
            # Map entity type to actual table name
            table_map = {
                "location": "Locations",
                "npc": "NPCStats",
                "faction": "Factions",
                "myth": "UrbanMyths",
                "event": "Events"
            }
            
            table_name = table_map.get(entity_type.lower(), entity_type)
            
            # Different tables have different name fields
            name_field = "npc_name" if table_name == "NPCStats" else "name"
            id_field = "npc_id" if table_name == "NPCStats" else "id"
            
            query = f"""
                SELECT {id_field} as id, {name_field} as name, 
                       1 - (embedding <=> $1) AS similarity
                FROM {table_name}
                WHERE embedding IS NOT NULL
                AND 1 - (embedding <=> $1) > $2
                ORDER BY embedding <=> $1
                LIMIT 1
            """
            
            similar = await conn.fetchrow(query, embedding, SIMILARITY_THRESHOLD)
            
            if similar:
                logger.info(
                    f"Found semantically similar {entity_type}: '{similar['name']}' "
                    f"(similarity: {similar['similarity']:.2f})"
                )
                return dict(similar)
        
        return None
    
    @staticmethod
    async def validate_against_consistency(
        content: str,
        content_type: str
    ) -> Tuple[bool, List[str]]:
        """Validate content against consistency rules"""
        
        # Use the consistency guide
        validation_result = QueenOfThornsConsistencyGuide.validate_content(content)
        
        if validation_result.get("violations"):
            return False, validation_result["violations"]
        
        # Additional type-specific validations
        violations = []
        
        if content_type == "network" and "official name" in content.lower():
            violations.append("Network must not have an official name")
        
        if content_type == "character" and "lilith" in content.lower():
            if "can say i love you" in content.lower():
                violations.append("Lilith cannot speak the three words")
        
        return len(violations) == 0, violations
    
    @staticmethod
    async def propose_with_governance(
        ctx: CanonicalContext,
        entity_type: str,
        entity_name: str,
        entity_data: Dict[str, Any],
        reason: str,
        check_duplicate: bool = True
    ) -> Dict[str, Any]:
        """Propose entity creation/update through governance with duplicate checking"""
        
        # Check for semantic duplicates first
        if check_duplicate:
            existing = await CanonIntegrationService.check_semantic_duplicate(
                ctx, entity_type, entity_name, entity_data
            )
            
            if existing:
                return {
                    "status": "duplicate_found",
                    "existing_id": existing["id"],
                    "existing_name": existing["name"],
                    "similarity": existing.get("similarity", 1.0)
                }
        
        # Validate consistency
        content_str = json.dumps(entity_data)
        valid, violations = await CanonIntegrationService.validate_against_consistency(
            content_str, entity_type
        )
        
        if not valid:
            raise ConsistencyError(f"Content violates consistency rules: {violations}")
        
        # Create through governance
        return await create_canonical_entity(
            ctx=ctx,
            entity_type=entity_type,
            entity_name=entity_name,
            entity_data=entity_data,
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="story_initializer"
        )

class QueenOfThornsStoryInitializer:
    """Enhanced story initialization with proper canon and governance integration"""
    
    @staticmethod
    async def initialize_story(
        ctx, user_id: int, conversation_id: int, dynamic: bool = True
    ) -> Dict[str, Any]:
        """
        Initialize the complete story with proper canon integration.
        
        Args:
            ctx: Context object
            user_id: User ID
            conversation_id: Conversation ID
            dynamic: Whether to use dynamic GPT generation (False for tests)
        """
        try:
            logger.info(f"Initializing Queen of Thorns story for user {user_id} (dynamic={dynamic})")
            
            # Ensure canonical context
            canon_ctx = CanonicalContext.from_object(ctx)
            
            # Phase 1: Establish base canon from preset lore
            logger.info("Phase 1: Establishing base canon")
            base_result = await QueenOfThornsStoryInitializer._establish_base_canon(
                canon_ctx, user_id, conversation_id
            )
            
            # Phase 2: Load story structure
            logger.info("Phase 2: Loading story structure")
            from story_templates.moth.queen_of_thorns_story import QUEEN_OF_THORNS_STORY
            await ThornsIntegratedStoryLoader.load_story_with_themes(
                QUEEN_OF_THORNS_STORY, user_id, conversation_id
            )
            
            # Phase 3: Create core entities with enhancements
            logger.info("Phase 3: Creating core entities")
            
            # Create locations (static or dynamic)
            if dynamic:
                location_ids = await QueenOfThornsStoryInitializer._create_enhanced_locations(
                    canon_ctx, user_id, conversation_id
                )
            else:
                location_ids = await QueenOfThornsStoryInitializer._create_base_locations(
                    canon_ctx, user_id, conversation_id
                )
            
            # Create Lilith (with or without evolution)
            if dynamic:
                lilith_id = await QueenOfThornsStoryInitializer._create_evolved_lilith_with_governance(
                    canon_ctx, user_id, conversation_id
                )
            else:
                lilith_id = await QueenOfThornsStoryInitializer._create_base_lilith(
                    canon_ctx, user_id, conversation_id
                )
            
            # Create supporting NPCs
            if dynamic:
                support_npc_ids = await QueenOfThornsStoryInitializer._create_enhanced_supporting_npcs(
                    canon_ctx, user_id, conversation_id
                )
            else:
                support_npc_ids = await QueenOfThornsStoryInitializer._create_base_supporting_npcs(
                    canon_ctx, user_id, conversation_id
                )
            
            # Phase 4: Establish relationships
            logger.info("Phase 4: Establishing relationships")
            await QueenOfThornsStoryInitializer._setup_canonical_relationships(
                canon_ctx, user_id, conversation_id, lilith_id, support_npc_ids
            )
            
            # Phase 5: Initialize story systems
            logger.info("Phase 5: Initializing story systems")
            await QueenOfThornsStoryInitializer._initialize_story_state(
                canon_ctx, user_id, conversation_id, lilith_id
            )
            
            await QueenOfThornsStoryInitializer._setup_special_mechanics(
                canon_ctx, user_id, conversation_id, lilith_id
            )
            
            await QueenOfThornsStoryInitializer._initialize_network_systems(
                canon_ctx, user_id, conversation_id, lilith_id
            )
            
            # Phase 6: Set atmosphere
            logger.info("Phase 6: Setting atmosphere")
            if dynamic:
                await QueenOfThornsStoryInitializer._set_dynamic_atmosphere_with_governance(
                    canon_ctx, user_id, conversation_id
                )
            else:
                await QueenOfThornsStoryInitializer._set_base_atmosphere(
                    canon_ctx, user_id, conversation_id
                )
            
            # Get metrics if using dynamic generation
            metrics_summary = {}
            if dynamic:
                service = GPTService()
                metrics_summary = service.get_metrics_summary()
                logger.info(f"Story initialization metrics: {metrics_summary}")
            
            return {
                "status": "success",
                "story_id": QUEEN_OF_THORNS_STORY.id,
                "main_npc_id": lilith_id,
                "support_npc_ids": support_npc_ids,
                "location_ids": location_ids,
                "network_initialized": True,
                "base_canon_result": base_result,
                "dynamic": dynamic,
                "metrics": metrics_summary,
                "message": f"Queen of Thorns story initialized {'dynamically' if dynamic else 'statically'} with full canon integration"
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize story: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "message": "Failed to initialize The Queen of Thorns story"
            }
    
    @staticmethod
    async def _establish_base_canon(
        ctx: CanonicalContext, user_id: int, conversation_id: int
    ) -> Dict[str, Any]:
        """Load all established lore into canon system first"""
        
        logger.info("Establishing base canon from preset lore")
        
        # Get complete world state from preset
        world_data = SFBayQueenOfThornsPreset.get_complete_world_state()
        
        stats = {
            "districts": 0,
            "factions": 0,
            "myths": 0,
            "locations": 0,
            "events": 0
        }
        
        async with get_db_connection_context() as conn:
            # Create districts
            for district in world_data['locations']['districts']:
                try:
                    result = await create_canonical_entity(
                        ctx=ctx,
                        entity_type="location",
                        entity_name=district['name'],
                        entity_data={
                            'description': district.get('description', ''),
                            'location_type': 'district',
                            'demographics': district.get('demographics', {}),
                            'vibe': district.get('vibe', ''),
                            'network_presence': district.get('network_presence', 'minimal')
                        },
                        agent_type=AgentType.NARRATIVE_CRAFTER,
                        agent_id="preset_loader"
                    )
                    if result.get("status") != "duplicate_found":
                        stats["districts"] += 1
                except Exception as e:
                    logger.error(f"Error creating district {district['name']}: {e}")
            
            # Create factions with proper governance
            for faction in world_data['power_structures']['factions']:
                try:
                    # Special handling for the network
                    if 'Shadow Network' in faction['name']:
                        faction['name'] = 'The Shadow Network'  # Consistent reference
                    
                    result = await create_canonical_entity(
                        ctx=ctx,
                        entity_type="faction",
                        entity_name=faction['name'],
                        entity_data={
                            'type': faction.get('type', 'organization'),
                            'description': faction.get('description', ''),
                            'public_face': faction.get('public_face', ''),
                            'private_reality': faction.get('private_reality', ''),
                            'power_level': faction.get('power_level', 5),
                            'resources': faction.get('resources', []),
                            'territory': faction.get('territory', [])
                        },
                        agent_type=AgentType.NARRATIVE_CRAFTER,
                        agent_id="preset_loader"
                    )
                    if result.get("status") != "duplicate_found":
                        stats["factions"] += 1
                except Exception as e:
                    logger.error(f"Error creating faction {faction['name']}: {e}")
            
            # Create urban myths
            for myth in world_data['culture']['myths']:
                try:
                    result = await create_canonical_entity(
                        ctx=ctx,
                        entity_type="myth",
                        entity_name=myth['name'],
                        entity_data={
                            'description': myth.get('story', ''),
                            'origin_location': myth.get('origin', 'San Francisco'),
                            'believability': myth.get('believability', 6),
                            'spread_rate': myth.get('spread_rate', 5),
                            'truth_level': myth.get('truth_level', 'mixed'),
                            'network_connection': myth.get('network_connection', 'possible')
                        },
                        agent_type=AgentType.NARRATIVE_CRAFTER,
                        agent_id="preset_loader"
                    )
                    if result.get("status") != "duplicate_found":
                        stats["myths"] += 1
                except Exception as e:
                    logger.error(f"Error creating myth {myth['name']}: {e}")
            
            # Create specific locations
            for location in world_data['locations']['specific']:
                try:
                    result = await create_canonical_entity(
                        ctx=ctx,
                        entity_type="location",
                        entity_name=location['name'],
                        entity_data={
                            'description': location.get('description', ''),
                            'location_type': location.get('type', 'venue'),
                            'cover_business': location.get('cover', ''),
                            'hidden_purpose': location.get('hidden_purpose', ''),
                            'schedule': location.get('schedule', {}),
                            'areas': location.get('areas', {})
                        },
                        agent_type=AgentType.NARRATIVE_CRAFTER,
                        agent_id="preset_loader"
                    )
                    if result.get("status") != "duplicate_found":
                        stats["locations"] += 1
                except Exception as e:
                    logger.error(f"Error creating location {location['name']}: {e}")
        
        logger.info(f"Base canon established: {stats}")
        
        return {
            "status": "success",
            "stats": stats,
            "message": "Base canon established from preset lore"
        }
    
    @staticmethod
    async def _create_enhanced_locations(
        ctx: CanonicalContext, user_id: int, conversation_id: int
    ) -> List[str]:
        """Create locations with dynamic enhancements through governance"""
        
        service = GPTService()
        location_ids = []
        
        # Get base locations from preset
        base_locations = SFLocalLore.get_specific_locations()
        
        for base_loc in base_locations[:3]:  # Key locations only
            try:
                # Check if already exists
                existing = await CanonIntegrationService.check_semantic_duplicate(
                    ctx, "location", base_loc['name']
                )
                
                if existing:
                    location_ids.append(existing["id"])
                    
                    # Enhance existing location
                    await QueenOfThornsStoryInitializer._enhance_existing_location(
                        ctx, existing["id"], base_loc, service
                    )
                else:
                    # Create new location with enhancements
                    location_id = await QueenOfThornsStoryInitializer._create_new_enhanced_location(
                        ctx, base_loc, service
                    )
                    if location_id:
                        location_ids.append(location_id)
                        
            except Exception as e:
                logger.error(f"Error creating location {base_loc['name']}: {e}")
        
        return location_ids
    
    @staticmethod
    async def _enhance_existing_location(
        ctx: CanonicalContext,
        location_id: str,
        base_data: Dict[str, Any],
        service: GPTService
    ):
        """Enhance an existing location without overwriting canon data"""
        
        # Generate atmospheric enhancements
        system_prompt = """You ENHANCE existing location descriptions with atmospheric details.
You must ADD to what exists, never contradict or replace established facts.
Focus on sensory details, hidden elements, and subtle supernatural touches."""

        user_prompt = f"""Enhance this EXISTING location with additional atmosphere:
Name: {base_data['name']}
Established facts: {json.dumps(base_data, indent=2)}

Generate ONLY new atmospheric details that complement the existing description."""

        try:
            enhancement = await service.call_with_validation(
                model=_DEFAULT_LOCATION_MODEL,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=LocationEnhancement,
                temperature=0.6
            )
            
            # Propose enhancement through governance
            await propose_canonical_change(
                ctx=ctx,
                entity_type="Locations",
                entity_identifier={"id": location_id},
                updates={
                    "atmospheric_details": enhancement.atmospheric_details,
                    "hidden_features": enhancement.hidden_features
                },
                reason="Dynamic atmospheric enhancement for story immersion",
                agent_type=AgentType.NARRATIVE_CRAFTER,
                agent_id="story_initializer"
            )
            
        except Exception as e:
            logger.error(f"Error enhancing location {location_id}: {e}")
    
    @staticmethod
    async def _create_new_enhanced_location(
        ctx: CanonicalContext,
        base_data: Dict[str, Any],
        service: GPTService
    ) -> Optional[str]:
        """Create a new location with dynamic enhancements"""
        
        system_prompt = """You create rich location descriptions for a supernatural story.
Include all provided facts while adding atmospheric and sensory details.
The location should feel real, lived-in, and slightly mysterious."""

        user_prompt = f"""Create a full description for this location:
{json.dumps(base_data, indent=2)}

Expand the description with atmospheric details while keeping all established facts."""

        try:
            enhancement = await service.call_with_validation(
                model=_DEFAULT_LOCATION_MODEL,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=LocationEnhancement,
                temperature=0.6
            )
            
            # Merge base data with enhancement
            location_data = {
                **base_data,
                'description': enhancement.description,
                'atmospheric_details': enhancement.atmospheric_details,
                'hidden_features': enhancement.hidden_features
            }
            
            # Create through governance
            result = await CanonIntegrationService.propose_with_governance(
                ctx=ctx,
                entity_type="location",
                entity_name=base_data['name'],
                entity_data=location_data,
                reason="Creating story location with atmospheric enhancements"
            )
            
            if result.get("status") == "success":
                return result.get("entity_id")
            elif result.get("status") == "duplicate_found":
                return result.get("existing_id")
            else:
                raise CanonError(f"Failed to create location: {result}")
                
        except Exception as e:
            logger.error(f"Error creating location {base_data['name']}: {e}")
            # Fallback to base creation
            return await QueenOfThornsStoryInitializer._create_base_location(
                ctx, base_data['name'], base_data.get('description', '')
            )
    
    @staticmethod
    async def _create_evolved_lilith_with_governance(
        ctx: CanonicalContext, user_id: int, conversation_id: int
    ) -> int:
        """Create Lilith with dynamic evolution through governance"""
        
        service = GPTService()
        
        # First check if Lilith already exists
        existing = await CanonIntegrationService.check_semantic_duplicate(
            ctx, "npc", LILITH_RAVENCROFT["name"]
        )
        
        if existing:
            lilith_id = existing["id"]
            logger.info(f"Lilith already exists with ID {lilith_id}")
            
            # Check if she needs evolution
            async with get_db_connection_context() as conn:
                current_data = await conn.fetchrow(
                    "SELECT evolved_version FROM NPCStats WHERE npc_id = $1",
                    lilith_id
                )
                
                if current_data and current_data.get('evolved_version') == 'v2_evolved':
                    return lilith_id
        else:
            # Create base Lilith first
            result = await CanonIntegrationService.propose_with_governance(
                ctx=ctx,
                entity_type="npc",
                entity_name=LILITH_RAVENCROFT["name"],
                entity_data={
                    "role": "The Queen of Thorns",
                    "affiliations": ["Velvet Sanctum", "The Shadow Network"],
                    **LILITH_RAVENCROFT
                },
                reason="Creating main character: Lilith Ravencroft"
            )
            
            if result.get("status") != "success":
                raise CanonError(f"Failed to create Lilith: {result}")
            
            lilith_id = result.get("entity_id")
        
        # Generate evolution that respects established character
        system_prompt = """You evolve character backstories with contemporary touches.
You must RESPECT the established character profile and only ADD new details.

ESTABLISHED FACTS TO MAINTAIN:
- Name: Lilith Ravencroft, The Queen of Thorns
- Core trauma: Trafficking attempt at 15
- Cannot speak "I love you" - central mechanic
- Dual life: Dominatrix/network leader
- Lists: Red ink for failed saves, blue for those who left

Focus on RECENT events and growth, not changing the core."""

        user_prompt = f"""Add recent character development for Lilith:
{json.dumps(LILITH_RAVENCROFT["backstory"], indent=2)}

Generate ONLY recent additions (last 6 months):
- 2-3 recent challenges
- 1-2 new scars (emotional/physical)
- 2-3 mask evolution details
- 3-4 new dialogue patterns

Do NOT change core character elements."""

        try:
            evolution = await service.call_with_validation(
                model=_DEFAULT_MEMORY_MODEL,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=CharacterEvolution,
                temperature=0.7
            )
            
            # Prepare evolution updates
            evolution_updates = {
                "recent_events": json.dumps(evolution.recent_intrigues),
                "current_struggles": json.dumps(evolution.new_scars),
                "evolved_masks": json.dumps(evolution.current_masks),
                "dialogue_evolution": json.dumps(evolution.dialogue_evolution),
                "evolved_version": "v2_evolved"
            }
            
            # Propose evolution through governance
            evolution_result = await propose_canonical_change(
                ctx=ctx,
                entity_type="NPCStats",
                entity_identifier={"npc_id": lilith_id},
                updates=evolution_updates,
                reason="Evolving Lilith with recent character development",
                agent_type=AgentType.NARRATIVE_CRAFTER,
                agent_id="story_initializer"
            )
            
            if evolution_result.get("status") != "success":
                logger.warning(f"Evolution rejected by governance: {evolution_result}")
            
            # Complete setup with PresetNPCHandler
            enhanced_data = LILITH_RAVENCROFT.copy()
            enhanced_data["evolution_data"] = evolution.dict()
            
            await PresetNPCHandler.create_detailed_npc(ctx, enhanced_data, {
                "story_context": "queen_of_thorns",
                "is_main_character": True
            })
            
        except Exception as e:
            logger.error(f"Error evolving Lilith: {e}")
        
        return lilith_id
    
    @staticmethod
    async def _create_enhanced_supporting_npcs(
        ctx: CanonicalContext, user_id: int, conversation_id: int
    ) -> List[int]:
        """Create supporting NPCs with enhancements through governance"""
        
        service = GPTService()
        npc_ids = []
        
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
        
        for npc_template in core_npcs:
            try:
                # Check for existing NPC
                existing = await CanonIntegrationService.check_semantic_duplicate(
                    ctx, "npc", npc_template["name"]
                )
                
                if existing:
                    npc_ids.append(existing["id"])
                    continue
                
                # Generate enhancements
                class NPCEnhancement(BaseModel):
                    personality_quirks: List[str] = Field(default_factory=list, max_items=5)
                    backstory_details: str = Field(..., max_length=500)
                    relationship_hooks: List[str] = Field(default_factory=list, max_items=3)
                
                system_prompt = """You enhance NPC concepts with memorable, specific details.
Focus on quirks that reveal character depth and create roleplay opportunities."""

                user_prompt = f"""Enhance this Queen of Thorns NPC:
Name: {npc_template['name']}
Role: {npc_template['role']}
Concept: {npc_template['concept']}

Generate unique personality and backstory details."""

                enhancement = await service.call_with_validation(
                    model=_DEFAULT_LORE_MODEL,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response_model=NPCEnhancement,
                    temperature=0.7
                )
                
                # Create NPC through governance
                npc_data = {
                    "role": npc_template["role"],
                    "backstory": enhancement.backstory_details,
                    "personality_traits": enhancement.personality_quirks,
                    "relationship_hooks": enhancement.relationship_hooks,
                    "affiliations": ["The Shadow Network"],
                    "concept": npc_template["concept"]
                }
                
                result = await CanonIntegrationService.propose_with_governance(
                    ctx=ctx,
                    entity_type="npc",
                    entity_name=npc_template["name"],
                    entity_data=npc_data,
                    reason=f"Creating supporting character: {npc_template['name']}"
                )
                
                if result.get("status") == "success":
                    npc_ids.append(result.get("entity_id"))
                elif result.get("status") == "duplicate_found":
                    npc_ids.append(result.get("existing_id"))
                    
            except Exception as e:
                logger.error(f"Error creating NPC {npc_template['name']}: {e}")
        
        return npc_ids
    
    @staticmethod
    async def _setup_canonical_relationships(
        ctx: CanonicalContext, user_id: int, conversation_id: int,
        lilith_id: int, support_npc_ids: List[int]
    ):
        """Establish relationships through the canon system"""
        
        # Get NPC names for relationship creation
        npc_names = {}
        async with get_db_connection_context() as conn:
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
                "strength": 95,
                "description": "Complete ownership and transformation"
            })
        
        if "Sarah Chen" in npc_id_map:
            relationships.append({
                "source": lilith_id,
                "target": npc_id_map["Sarah Chen"],
                "type": "protects",
                "strength": 85,
                "description": "Saved her, now protects and guides"
            })
        
        if "Victoria Chen" in npc_id_map:
            relationships.append({
                "source": lilith_id,
                "target": npc_id_map["Victoria Chen"],
                "type": "commands",
                "strength": 75,
                "description": "Rose Council member under Queen's authority"
            })
        
        # Create relationships through canon
        for rel in relationships:
            try:
                await canon.find_or_create_social_link(
                    ctx, conn,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    entity1_type="npc",
                    entity1_id=rel["source"],
                    entity2_type="npc",
                    entity2_id=rel["target"],
                    link_type=rel["type"],
                    link_level=rel["strength"]
                )
                
                # Add shared memory through governance
                memory_text = f"{npc_names[rel['source']]} and {npc_names[rel['target']]}: {rel['description']}"
                
                for npc_id in [rel["source"], rel["target"]]:
                    await remember_with_governance(
                        user_id=user_id,
                        conversation_id=conversation_id,
                        entity_type="npc",
                        entity_id=npc_id,
                        memory_text=memory_text,
                        importance="medium",
                        emotional=True,
                        tags=["relationship", "shared_memory"]
                    )
                    
            except Exception as e:
                logger.error(f"Error creating relationship: {e}")
    
    @staticmethod
    async def _set_dynamic_atmosphere_with_governance(
        ctx: CanonicalContext, user_id: int, conversation_id: int
    ):
        """Generate and set atmospheric introduction through governance"""
        
        service = GPTService()
        
        # Get contextual information
        from datetime import datetime
        import random
        
        current_time = datetime.now()
        moon_phases = ["new moon", "waxing crescent", "first quarter", "waxing gibbous", 
                      "full moon", "waning gibbous", "last quarter", "waning crescent"]
        current_moon = moon_phases[current_time.day % 8]
        weather_options = ["fog rolling in", "light rain", "clear night", "wind from the bay"]
        current_weather = random.choice(weather_options)
        
        system_prompt = """You create immersive story atmospheres for dark supernatural narratives.
Focus on sensory details, hidden dangers, and the promise of transformation.
The atmosphere should feel noir, gothic, and slightly dangerous."""

        user_prompt = f"""Create the opening atmosphere for The Queen of Thorns:
Setting: SoMa underground, San Francisco after midnight
Moon phase: {current_moon}
Weather: {current_weather}
Venue: The Velvet Sanctum (hidden BDSM club)
Theme: Power, transformation, hidden networks

Write an atmospheric introduction (3-4 paragraphs)."""

        try:
            atmosphere = await service.call_with_validation(
                model=_DEFAULT_ATMOSPHERE_MODEL,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=AtmosphereData,
                temperature=0.8
            )
            
            # Store atmosphere through governance
            atmosphere_data = {
                "tone": "noir_gothic",
                "moon_phase": current_moon,
                "weather": current_weather,
                "feeling": atmosphere.atmosphere.get("feeling", "mysterious"),
                "hidden_elements": atmosphere.hidden_elements
            }
            
            await propose_canonical_change(
                ctx=ctx,
                entity_type="CurrentRoleplay",
                entity_identifier={"key": "story_atmosphere"},
                updates={"value": json.dumps(atmosphere_data)},
                reason="Setting dynamic story atmosphere",
                agent_type=AgentType.NARRATIVE_CRAFTER,
                agent_id="story_initializer"
            )
            
            # Store introduction as memory
            await remember_with_governance(
                user_id=user_id,
                conversation_id=conversation_id,
                entity_type="player",
                entity_id=user_id,
                memory_text=atmosphere.introduction,
                importance="high",
                emotional=True,
                tags=["story_start", "atmosphere", "dynamic"]
            )
            
        except Exception as e:
            logger.error(f"Error setting dynamic atmosphere: {e}")
            await QueenOfThornsStoryInitializer._set_base_atmosphere(
                ctx, user_id, conversation_id
            )
    
    # Base/static creation methods (fallbacks)
    
    @staticmethod
    async def _create_base_locations(
        ctx: CanonicalContext, user_id: int, conversation_id: int
    ) -> List[str]:
        """Create locations without dynamic generation"""
        
        locations = [
            ("Velvet Sanctum", "An underground temple of transformation"),
            ("The Rose Garden Café", "A Mission café with hidden purposes"),
            ("Marina Safehouse", "A villa for healing and transformation")
        ]
        
        location_ids = []
        
        for name, desc in locations:
            location_id = await QueenOfThornsStoryInitializer._create_base_location(
                ctx, name, desc
            )
            if location_id:
                location_ids.append(location_id)
        
        return location_ids
    
    @staticmethod
    async def _create_base_location(
        ctx: CanonicalContext, name: str, description: str
    ) -> Optional[str]:
        """Create a single base location"""
        
        result = await CanonIntegrationService.propose_with_governance(
            ctx=ctx,
            entity_type="location",
            entity_name=name,
            entity_data={"description": description},
            reason="Creating base story location"
        )
        
        if result.get("status") == "success":
            return result.get("entity_id")
        elif result.get("status") == "duplicate_found":
            return result.get("existing_id")
        
        return None
    
    @staticmethod
    async def _create_base_lilith(
        ctx: CanonicalContext, user_id: int, conversation_id: int
    ) -> int:
        """Create Lilith without dynamic generation"""
        
        result = await CanonIntegrationService.propose_with_governance(
            ctx=ctx,
            entity_type="npc",
            entity_name=LILITH_RAVENCROFT["name"],
            entity_data={
                "role": "The Queen of Thorns",
                "affiliations": ["Velvet Sanctum", "The Shadow Network"],
                **LILITH_RAVENCROFT
            },
            reason="Creating main character: Lilith Ravencroft"
        )
        
        if result.get("status") != "success":
            if result.get("status") == "duplicate_found":
                return result.get("existing_id")
            raise CanonError(f"Failed to create Lilith: {result}")
        
        lilith_id = result.get("entity_id")
        
        # Complete setup
        await PresetNPCHandler.create_detailed_npc(ctx, LILITH_RAVENCROFT, {
            "story_context": "queen_of_thorns",
            "is_main_character": True
        })
        
        return lilith_id
    
    @staticmethod
    async def _create_base_supporting_npcs(
        ctx: CanonicalContext, user_id: int, conversation_id: int
    ) -> List[int]:
        """Create supporting NPCs without dynamic generation"""
        
        npcs = [
            ("Marcus Sterling", "Devoted Submissive / Former Tech CEO"),
            ("Sarah Chen", "Trafficking Survivor / Safehouse Coordinator"),
            ("Victoria Chen", "VC Partner / Rose Council Member")
        ]
        
        npc_ids = []
        
        for name, role in npcs:
            result = await CanonIntegrationService.propose_with_governance(
                ctx=ctx,
                entity_type="npc",
                entity_name=name,
                entity_data={
                    "role": role,
                    "affiliations": ["The Shadow Network"]
                },
                reason=f"Creating supporting character: {name}"
            )
            
            if result.get("status") == "success":
                npc_ids.append(result.get("entity_id"))
            elif result.get("status") == "duplicate_found":
                npc_ids.append(result.get("existing_id"))
        
        return npc_ids
    
    @staticmethod
    async def _set_base_atmosphere(
        ctx: CanonicalContext, user_id: int, conversation_id: int
    ):
        """Set basic atmosphere without dynamic generation"""
        
        atmosphere_text = """
The fog rolls in from the bay, thick and consuming, as you descend the unmarked stairs in SoMa. 
Each step takes you further from the sanitized tech world above and deeper into something ancient 
wearing modern clothes. The Velvet Sanctum waits below, a temple where power is worshipped in 
its truest forms. Here, CEOs kneel and survivors become queens. The scent of leather and roses 
mingles with something darker - the perfume of secrets and transformation.

You've heard whispers of this place, of the woman who rules it. They say she saves the lost 
and breaks the proud. They say she cannot speak three simple words. They say many things, 
but tonight you'll learn the truth.

The door opens before you knock. It always does for those who are meant to enter.
"""
        
        # Store atmosphere
        await propose_canonical_change(
            ctx=ctx,
            entity_type="CurrentRoleplay",
            entity_identifier={"key": "story_atmosphere"},
            updates={"value": json.dumps({
                "tone": "noir_gothic",
                "feeling": "anticipation",
                "hidden_elements": ["The door knows who belongs"]
            })},
            reason="Setting initial story atmosphere",
            agent_type=AgentType.NARRATIVE_CRAFTER,
            agent_id="story_initializer"
        )
        
        # Store as memory
        await remember_with_governance(
            user_id=user_id,
            conversation_id=conversation_id,
            entity_type="player",
            entity_id=user_id,
            memory_text=atmosphere_text,
            importance="high",
            emotional=True,
            tags=["story_start", "atmosphere", "static"]
        )
    
    @staticmethod
    async def _initialize_story_state(
        ctx: CanonicalContext, user_id: int, conversation_id: int, lilith_id: int
    ):
        """Initialize story state tracking"""
        
        async with get_db_connection_context() as conn:
            await conn.execute("""
                INSERT INTO story_states (
                    user_id, conversation_id, story_id,
                    current_act, current_beat, progress,
                    story_flags, main_npc_id
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (user_id, conversation_id, story_id)
                DO UPDATE SET 
                    current_act = EXCLUDED.current_act,
                    main_npc_id = EXCLUDED.main_npc_id
            """, 
            user_id, conversation_id, "queen_of_thorns",
            1, None, 0,
            json.dumps({
                "trust_level": 0,
                "network_awareness": 0,
                "lilith_mask": "Porcelain Goddess",
                "completed_beats": []
            }),
            lilith_id)
    
    @staticmethod
    async def _setup_special_mechanics(
        ctx: CanonicalContext, user_id: int, conversation_id: int, lilith_id: int
    ):
        """Set up special story mechanics"""
        
        # Mask system
        mask_data = {
            "available_masks": [
                {
                    "name": "Porcelain Goddess",
                    "trust_required": 0,
                    "description": "Perfect, cold, untouchable divinity"
                },
                {
                    "name": "Leather Predator",
                    "trust_required": 30,
                    "description": "Dangerous, hunting, protective fury"
                },
                {
                    "name": "Lace Vulnerability",
                    "trust_required": 60,
                    "description": "Soft edges barely containing sharp pain"
                },
                {
                    "name": "No Mask",
                    "trust_required": 85,
                    "description": "The broken woman beneath"
                }
            ],
            "current_mask": "Porcelain Goddess",
            "mask_integrity": 100
        }
        
        async with get_db_connection_context() as conn:
            await conn.execute("""
                INSERT INTO npc_special_mechanics (
                    user_id, conversation_id, npc_id,
                    mechanic_type, mechanic_data
                )
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (user_id, conversation_id, npc_id, mechanic_type)
                DO UPDATE SET mechanic_data = EXCLUDED.mechanic_data
            """,
            user_id, conversation_id, lilith_id,
            "mask_system", json.dumps(mask_data))
    
    @staticmethod
    async def _initialize_network_systems(
        ctx: CanonicalContext, user_id: int, conversation_id: int, lilith_id: int
    ):
        """Initialize the shadow network systems"""
        
        network_data = {
            "organization_names": ["the network", "the garden"],
            "structure": {
                "queen_of_thorns": lilith_id,
                "rose_council": ["Victoria Chen", "Judge Thornfield", "5 others"],
                "ranks": ["Seedlings", "Roses", "Thorns", "Gardeners"]
            },
            "statistics": {
                "saved_this_month": random.randint(3, 8),
                "active_operations": 4,
                "threat_level": "moderate"
            },
            "current_operations": [
                {
                    "id": "op_kozlov_1",
                    "type": "surveillance",
                    "target": "Kozlov trafficking ring",
                    "status": "active"
                },
                {
                    "id": "op_safehouse_3",
                    "type": "protection",
                    "target": "Marina safehouse",
                    "status": "ongoing"
                }
            ]
        }
        
        async with get_db_connection_context() as conn:
            await conn.execute("""
                INSERT INTO network_state (user_id, conversation_id, network_data)
                VALUES ($1, $2, $3)
                ON CONFLICT (user_id, conversation_id)
                DO UPDATE SET network_data = EXCLUDED.network_data
            """,
            user_id, conversation_id, json.dumps(network_data))


# Additional helper functions for story progression

class QueenOfThornsStoryProgression:
    """Handles story progression with canon integration"""
    
    @staticmethod
    async def check_beat_triggers(user_id: int, conversation_id: int) -> Optional[str]:
        """Check if any story beats should trigger"""
        
        async with get_db_connection_context() as conn:
            state_row = await conn.fetchrow("""
                SELECT current_beat, story_flags, progress, current_act
                FROM story_states
                WHERE user_id = $1 AND conversation_id = $2 AND story_id = $3
            """, user_id, conversation_id, "queen_of_thorns")
            
            if not state_row:
                return None
            
            current_beat = state_row['current_beat']
            story_flags = json.loads(state_row['story_flags'] or '{}')
            completed_beats = story_flags.get('completed_beats', [])
            
            # Check for story beats
            from story_templates.moth.queen_of_thorns_story import QUEEN_OF_THORNS_STORY
            
            for beat in QUEEN_OF_THORNS_STORY.story_beats:
                if beat.id in completed_beats or beat.id == current_beat:
                    continue
                
                # Check conditions
                conditions_met = True
                for condition, value in beat.trigger_conditions.items():
                    if condition in story_flags:
                        if story_flags[condition] < value:
                            conditions_met = False
                            break
                
                if conditions_met:
                    return beat.id
        
        return None
    
    @staticmethod
    async def advance_story_element(
        user_id: int, conversation_id: int,
        element_type: str, amount: int = 10
    ) -> Dict[str, Any]:
        """Advance a story element through governance"""
        
        ctx = CanonicalContext(user_id=user_id, conversation_id=conversation_id)
        
        async with get_db_connection_context() as conn:
            state_row = await conn.fetchrow("""
                SELECT story_flags FROM story_states
                WHERE user_id = $1 AND conversation_id = $2 AND story_id = $3
            """, user_id, conversation_id, "queen_of_thorns")
            
            if not state_row:
                return {"error": "Story state not found"}
            
            story_flags = json.loads(state_row['story_flags'] or '{}')
            
            # Update element
            current_value = story_flags.get(element_type, 0)
            new_value = min(100, current_value + amount)
            story_flags[element_type] = new_value
            
            # Propose update through governance
            result = await propose_canonical_change(
                ctx=ctx,
                entity_type="story_states",
                entity_identifier={
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                    "story_id": "queen_of_thorns"
                },
                updates={"story_flags": json.dumps(story_flags)},
                reason=f"Advancing {element_type} by {amount}",
                agent_type=AgentType.STORY_DIRECTOR,
                agent_id="story_progression"
            )
            
            return {
                "element_type": element_type,
                "old_value": current_value,
                "new_value": new_value,
                "governance_result": result
            }


# Maintain compatibility
MothFlameStoryInitializer = QueenOfThornsStoryInitializer
MothFlameStoryProgression = QueenOfThornsStoryProgression
