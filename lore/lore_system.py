"""
Lore System - Main Entry Point

This module serves as the primary interface for all lore-related functionality.
It provides a clean, optimized API that uses the Data Access Layer for all database operations.
"""

import logging
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import re

# Import data access layer
from data import lore_data_access, npc_data_access, location_data_access, conflict_data_access

# Nyx governance integration
from nyx.integrate import get_central_governance
from nyx.nyx_governance import AgentType, DirectiveType, DirectivePriority
from nyx.governance_helpers import with_governance

from logic.conflict_system.conflict_integration import ConflictSystemIntegration

logger = logging.getLogger(__name__)

# Cache for LoreSystem instances to avoid creating multiple instances
LORE_SYSTEM_INSTANCES = {}

# Cache for language data to reduce database queries
LANGUAGE_CACHE = {}
DIALECT_CACHE = {}

class LoreSystem:
    """
    Unified interface for all lore-related functionality.
    
    This class consolidates functionality from multiple legacy lore-related classes
    and uses the standardized Data Access Layer for all database operations.
    """
    
    def __init__(self, user_id: Optional[int] = None, conversation_id: Optional[int] = None):
        """
        Initialize the LoreSystem with optional user and conversation context.
        
        Args:
            user_id: Optional user ID for filtering
            conversation_id: Optional conversation ID for filtering
        """
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.initialized = False
        self.languages = {}
        self._language_cache_time = None
        self._cache_ttl_seconds = 300  # 5 minutes cache TTL
        
    @classmethod
    def get_instance(cls, user_id: Optional[int] = None, conversation_id: Optional[int] = None) -> 'LoreSystem':
        """
        Get a singleton instance of LoreSystem for the given user and conversation.
        Reuses existing instances to avoid redundant initialization.
        
        Args:
            user_id: Optional user ID for filtering
            conversation_id: Optional conversation ID for filtering
            
        Returns:
            LoreSystem instance
        """
        key = f"{user_id or 'global'}:{conversation_id or 'global'}"
        
        if key not in LORE_SYSTEM_INSTANCES:
            LORE_SYSTEM_INSTANCES[key] = cls(user_id, conversation_id)
            
        return LORE_SYSTEM_INSTANCES[key]
    
    async def initialize(self, force: bool = False) -> bool:
        """
        Initialize the LoreSystem by loading languages and other common data.
        
        Args:
            force: If True, force reinitialization even if already initialized
            
        Returns:
            True if initialization successful, False otherwise
        """
        if self.initialized and not force:
            return True
            
        try:
            # Load languages
            await self._load_languages()
            
            self.initialized = True
            return True
        except Exception as e:
            logger.error(f"Error initializing LoreSystem: {e}")
            return False
    
    async def _load_languages(self) -> None:
        """
        Load all languages from the database.
        Cached to avoid redundant database queries.
        """
        # Check if we need to refresh the cache
        now = datetime.now()
        
        # Cache key based on user_id and conversation_id
        cache_key = f"languages:{self.user_id or 'global'}:{self.conversation_id or 'global'}"
        
        # Check if cached data exists and is not expired
        if (cache_key in LANGUAGE_CACHE and 
            (now - LANGUAGE_CACHE[cache_key]['timestamp']).total_seconds() < self._cache_ttl_seconds):
            
            self.languages = LANGUAGE_CACHE[cache_key]['data']
            logger.debug(f"Using cached languages for {cache_key}")
            return
            
        try:
            # Build query conditions based on user_id and conversation_id
            conditions = []
            params = []
            
            if self.user_id is not None:
                conditions.append(f"user_id = ${len(params) + 1}")
                params.append(self.user_id)
                
            if self.conversation_id is not None:
                conditions.append(f"conversation_id = ${len(params) + 1}")
                params.append(self.conversation_id)
                
            # Build the complete query
            query = """
                SELECT id, name, description, writing_system, formality_levels, dialects
                FROM Languages
            """
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
                
            # Execute the query
            rows = await fetch(query, *params)
            
            # Process the results
            languages = {}
            for row in rows:
                lang_data = dict(row)
                
                # Parse dialects if it's a JSON string
                if 'dialects' in lang_data and lang_data['dialects'] and isinstance(lang_data['dialects'], str):
                    try:
                        lang_data['dialects'] = json.loads(lang_data['dialects'])
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to decode dialects JSON for language {lang_data.get('id')}")
                        lang_data['dialects'] = {}
                elif 'dialects' not in lang_data or lang_data['dialects'] is None:
                    lang_data['dialects'] = {}
                
                # Parse formality_levels array
                if 'formality_levels' in lang_data and lang_data['formality_levels']:
                    if isinstance(lang_data['formality_levels'], str):
                        # Remove braces and split by comma
                        formality_str = lang_data['formality_levels'].strip('{}')
                        if formality_str:
                            lang_data['formality_levels'] = [level.strip('"') for level in formality_str.split(',')]
                        else:
                            lang_data['formality_levels'] = []
                else:
                    lang_data['formality_levels'] = []
                
                languages[lang_data['id']] = lang_data
            
            self.languages = languages
            
            # Update the cache
            LANGUAGE_CACHE[cache_key] = {
                'data': languages,
                'timestamp': now
            }
            
        except Exception as e:
            logger.error(f"Error loading languages: {e}")
            # Set to empty dict if error occurred
            self.languages = {}
    
    async def get_language_list(self, include_dialects: bool = True) -> List[Dict[str, Any]]:
        """
        Get a list of all languages.
        
        Args:
            include_dialects: Whether to include dialect information
            
        Returns:
            List of language dictionaries
        """
        # Ensure initialized
        if not self.initialized:
            await self.initialize()
            
        try:
            languages = list(self.languages.values())
            
            # If not including dialects, remove dialect information to reduce response size
            if not include_dialects:
                for lang in languages:
                    lang.pop('dialects', None)
                    
            return languages
        except Exception as e:
            logger.error(f"Error getting language list: {e}")
            return []
    
    async def get_language_by_id(self, language_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a language by its ID.
        
        Args:
            language_id: The language ID to retrieve
            
        Returns:
            Language dictionary or None if not found
        """
        # Ensure initialized
        if not self.initialized:
            await self.initialize()
            
        try:
            # First check in our loaded languages
            if language_id in self.languages:
                return self.languages[language_id]
                
            # If not found, try to fetch from database
            # This can happen if languages were added after initialization
            conditions = ["id = $1"]
            params = [language_id]
            
            if self.user_id is not None:
                conditions.append(f"user_id = ${len(params) + 1}")
                params.append(self.user_id)
                
            if self.conversation_id is not None:
                conditions.append(f"conversation_id = ${len(params) + 1}")
                params.append(self.conversation_id)
                
            query = """
                SELECT id, name, description, writing_system, formality_levels, dialects
                FROM Languages
                WHERE {}
                LIMIT 1
            """.format(" AND ".join(conditions))
            
            row = await fetchrow(query, *params)
            
            if not row:
                return None
                
            lang_data = dict(row)
            
            # Parse dialects if it's a JSON string
            if 'dialects' in lang_data and lang_data['dialects'] and isinstance(lang_data['dialects'], str):
                try:
                    lang_data['dialects'] = json.loads(lang_data['dialects'])
                except json.JSONDecodeError:
                    logger.warning(f"Failed to decode dialects JSON for language {lang_data.get('id')}")
                    lang_data['dialects'] = {}
            elif 'dialects' not in lang_data or lang_data['dialects'] is None:
                lang_data['dialects'] = {}
            
            # Parse formality_levels array
            if 'formality_levels' in lang_data and lang_data['formality_levels']:
                if isinstance(lang_data['formality_levels'], str):
                    # Remove braces and split by comma
                    formality_str = lang_data['formality_levels'].strip('{}')
                    if formality_str:
                        lang_data['formality_levels'] = [level.strip('"') for level in formality_str.split(',')]
                    else:
                        lang_data['formality_levels'] = []
            else:
                lang_data['formality_levels'] = []
            
            # Add to our cache for future requests
            self.languages[language_id] = lang_data
            
            return lang_data
        except Exception as e:
            logger.error(f"Error getting language by ID {language_id}: {e}")
            return None
    
    async def get_language_by_name(self, language_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a language by its name.
        
        Args:
            language_name: The language name to retrieve
            
        Returns:
            Language dictionary or None if not found
        """
        # Ensure initialized
        if not self.initialized:
            await self.initialize()
            
        try:
            # First check in our loaded languages
            for lang in self.languages.values():
                if lang['name'].lower() == language_name.lower():
                    return lang
                    
            # If not found, try to fetch from database
            conditions = ["LOWER(name) = LOWER($1)"]
            params = [language_name]
            
            if self.user_id is not None:
                conditions.append(f"user_id = ${len(params) + 1}")
                params.append(self.user_id)
                
            if self.conversation_id is not None:
                conditions.append(f"conversation_id = ${len(params) + 1}")
                params.append(self.conversation_id)
                
            query = """
                SELECT id, name, description, writing_system, formality_levels, dialects
                FROM Languages
                WHERE {}
                LIMIT 1
            """.format(" AND ".join(conditions))
            
            row = await fetchrow(query, *params)
            
            if not row:
                return None
                
            lang_data = dict(row)
            
            # Process the language data (same as in get_language_by_id)
            # Parse dialects if it's a JSON string
            if 'dialects' in lang_data and lang_data['dialects'] and isinstance(lang_data['dialects'], str):
                try:
                    lang_data['dialects'] = json.loads(lang_data['dialects'])
                except json.JSONDecodeError:
                    logger.warning(f"Failed to decode dialects JSON for language {lang_data.get('id')}")
                    lang_data['dialects'] = {}
            elif 'dialects' not in lang_data or lang_data['dialects'] is None:
                lang_data['dialects'] = {}
            
            # Parse formality_levels array
            if 'formality_levels' in lang_data and lang_data['formality_levels']:
                if isinstance(lang_data['formality_levels'], str):
                    # Remove braces and split by comma
                    formality_str = lang_data['formality_levels'].strip('{}')
                    if formality_str:
                        lang_data['formality_levels'] = [level.strip('"') for level in formality_str.split(',')]
                    else:
                        lang_data['formality_levels'] = []
            else:
                lang_data['formality_levels'] = []
            
            # Add to our cache for future requests
            self.languages[lang_data['id']] = lang_data
            
            return lang_data
        except Exception as e:
            logger.error(f"Error getting language by name '{language_name}': {e}")
            return None
    
    async def create_language(self, language_data: Dict[str, Any]) -> Optional[int]:
        """
        Create a new language.
        
        Args:
            language_data: Dictionary with language details
            
        Returns:
            New language ID if successful, None otherwise
        """
        # Validate required fields
        required_fields = ['name', 'description']
        for field in required_fields:
            if field not in language_data:
                logger.error(f"Missing required field '{field}' for language creation")
                return None
                
        try:
            # Prepare language data
            name = language_data['name']
            description = language_data['description']
            writing_system = language_data.get('writing_system', '')
            
            # Prepare formality levels
            formality_levels = language_data.get('formality_levels', [])
            if not formality_levels:
                formality_levels = ['informal', 'neutral', 'formal']
                
            # Prepare dialects - store as jsonb
            dialects = language_data.get('dialects', {})
            
            # Set user_id and conversation_id if not provided
            user_id = language_data.get('user_id', self.user_id)
            conversation_id = language_data.get('conversation_id', self.conversation_id)
            
            # Build the query
            query = """
                INSERT INTO Languages
                (name, description, writing_system, formality_levels, dialects, user_id, conversation_id)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING id
            """
            
            # For jsonb fields, ensure we're passing valid JSON
            dialects_json = dialects
            if not isinstance(dialects, str):
                dialects_json = json.dumps(dialects)
                
            # Execute the query
            language_id = await fetchval(
                query,
                name,
                description,
                writing_system,
                formality_levels,
                dialects_json,
                user_id,
                conversation_id
            )
            
            if language_id:
                # Invalidate cache to ensure we reload languages
                cache_key = f"languages:{self.user_id or 'global'}:{self.conversation_id or 'global'}"
                if cache_key in LANGUAGE_CACHE:
                    del LANGUAGE_CACHE[cache_key]
                
                # Force reload languages
                await self._load_languages()
                
            return language_id
        except Exception as e:
            logger.error(f"Error creating language: {e}")
            return None
    
    async def create_dialect_variant(
        self,
        language_id: int,
        dialect_name: str,
        dialect_data: Dict[str, Any]
    ) -> Optional[int]:
        """
        Create a dialect variant for a language.
        
        Args:
            language_id: ID of the parent language
            dialect_name: Name of the dialect
            dialect_data: Dictionary with dialect details
            
        Returns:
            ID of the new dialect if successful, None otherwise
        """
        try:
            # Get the language first
            language = await self.get_language_by_id(language_id)
            if not language:
                logger.error(f"Language with ID {language_id} not found")
                return None
                
            # Generate a new dialect ID
            dialect_id = int(datetime.now().timestamp())
            
            # Ensure we have a dialects dictionary
            if 'dialects' not in language or not language['dialects']:
                language['dialects'] = {}
            elif not isinstance(language['dialects'], dict):
                # Convert to dict if it's not already
                try:
                    if isinstance(language['dialects'], str):
                        language['dialects'] = json.loads(language['dialects'])
                    else:
                        language['dialects'] = {}
                        logger.warning(f"Invalid dialects format for language {language_id}, reset to empty dict")
                except json.JSONDecodeError:
                    language['dialects'] = {}
                    logger.warning(f"Failed to parse dialects for language {language_id}, reset to empty dict")
            
            # Make a copy of the dialect data to avoid modifying the input
            dialect_info = dialect_data.copy()
            dialect_info['id'] = dialect_id
            
            # Add required fields if missing
            if 'description' not in dialect_info:
                dialect_info['description'] = f"A dialect of {language['name']}"
            
            # Important dialect features to track if not provided
            feature_categories = ['accent', 'vocabulary', 'grammar', 'cultural_expressions']
            
            # Add empty dictionaries for missing categories
            for category in feature_categories:
                if category not in dialect_info:
                    dialect_info[category] = {}
                # Extract any features from description if provided with field naming patterns
                elif category + '_differences' in dialect_data:
                    # Extract relevant data from the *_differences field to structured format
                    differences = dialect_data[category + '_differences']
                    if isinstance(differences, str) and differences:
                        # Parse simple comma-separated items
                        items = [item.strip() for item in differences.split(',')]
                        if category == 'accent':
                            dialect_info[category] = {f"feature_{i}": item for i, item in enumerate(items)}
                        elif category == 'vocabulary':
                            # For vocabulary, try to extract word pairs
                            for i, item in enumerate(items):
                                if ':' in item or '->' in item:
                                    parts = item.replace('->', ':').split(':')
                                    if len(parts) == 2:
                                        standard, dialect = parts
                                        dialect_info[category][standard.strip()] = dialect.strip()
                                    else:
                                        dialect_info[category][f"term_{i}"] = item
                                else:
                                    dialect_info[category][f"term_{i}"] = item
                        elif category == 'grammar':
                            dialect_info[category] = {f"rule_{i}": item for i, item in enumerate(items)}
                        else:
                            dialect_info[category] = {f"item_{i}": item for i, item in enumerate(items)}
            
            # Add to the language's dialects
            language['dialects'][dialect_name] = dialect_info
            
            # Update the language in the database with proper JSON handling
            async with acquire() as conn:
                async with conn.transaction():
                    # Prepare dialects JSON directly as a serializable object
                    # This prevents double encoding or other JSON handling issues
                    query = """
                        UPDATE Languages
                        SET dialects = $1::jsonb
                        WHERE id = $2
                        RETURNING id
                    """
                    
                    result = await conn.fetchval(query, json.dumps(language['dialects']), language_id)
                    
                    if not result:
                        logger.error(f"Failed to update language {language_id} with new dialect")
                        return None
            
            # Get cache key
            cache_key = f"languages:{self.user_id or 'global'}:{self.conversation_id or 'global'}"
            
            # Update cache if it exists
            if cache_key in LANGUAGE_CACHE:
                LANGUAGE_CACHE[cache_key]['data'][language_id]['dialects'] = language['dialects']
            
            # Also add to dialect cache
            dialect_cache_key = f"dialect:{dialect_id}:{self.user_id or 'global'}:{self.conversation_id or 'global'}"
            DIALECT_CACHE[dialect_cache_key] = {
                'data': {
                    'id': dialect_id,
                    'name': dialect_name,
                    'language_id': language_id,
                    'language_name': language['name'],
                    **dialect_info
                },
                'timestamp': datetime.now()
            }
            
            logger.info(f"Successfully created dialect '{dialect_name}' (ID: {dialect_id}) for language {language_id}")
            return dialect_id
                
        except Exception as e:
            logger.error(f"Error creating dialect variant: {e}")
            return None
    
    async def get_dialect_by_id(self, dialect_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a dialect by its ID.
        
        Args:
            dialect_id: The dialect ID to retrieve
            
        Returns:
            Dialect dictionary or None if not found
        """
        # Check cache first
        dialect_cache_key = f"dialect:{dialect_id}:{self.user_id or 'global'}:{self.conversation_id or 'global'}"
        if dialect_cache_key in DIALECT_CACHE:
            cache_data = DIALECT_CACHE[dialect_cache_key]
            if (datetime.now() - cache_data['timestamp']).total_seconds() < self._cache_ttl_seconds:
                return cache_data['data']
        
        # Ensure initialized
        if not self.initialized:
            await self.initialize()
            
        try:
            # Search all languages for the dialect
            for language in self.languages.values():
                if 'dialects' in language and language['dialects']:
                    # Search dialects by ID
                    for dialect_name, dialect_info in language['dialects'].items():
                        if isinstance(dialect_info, dict) and dialect_info.get('id') == dialect_id:
                            # Prepare a complete dialect info object
                            result = {
                                'id': dialect_id,
                                'name': dialect_name,
                                'language_id': language['id'],
                                'language_name': language['name'],
                                **dialect_info
                            }
                            
                            # Cache the result
                            DIALECT_CACHE[dialect_cache_key] = {
                                'data': result,
                                'timestamp': datetime.now()
                            }
                            
                            return result
            
            # If not found, check if we missed any languages that might have dialects
            # Here we do a targeted query specifically for this dialect ID
            query = """
                SELECT l.id as language_id, l.name as language_name, l.dialects
                FROM Languages l
                WHERE l.dialects::text LIKE $1
            """
            
            if self.user_id is not None:
                query += f" AND l.user_id = ${len([f'%"id":{dialect_id}%']) + 1}"
                params = [f'%"id":{dialect_id}%', self.user_id]
            else:
                params = [f'%"id":{dialect_id}%']
                
            if self.conversation_id is not None:
                query += f" AND l.conversation_id = ${len(params) + 1}"
                params.append(self.conversation_id)
                
            # Execute the query
            language_rows = await fetch(query, *params)
            
            for row in language_rows:
                lang_data = dict(row)
                
                # Parse dialects
                dialects = lang_data.get('dialects', {})
                if isinstance(dialects, str):
                    try:
                        dialects = json.loads(dialects)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to decode dialects JSON for language {lang_data.get('language_id')}")
                        continue
                
                # Search for the dialect
                for dialect_name, dialect_info in dialects.items():
                    if isinstance(dialect_info, dict) and dialect_info.get('id') == dialect_id:
                        # Prepare a complete dialect info object
                        result = {
                            'id': dialect_id,
                            'name': dialect_name,
                            'language_id': lang_data['language_id'],
                            'language_name': lang_data['language_name'],
                            **dialect_info
                        }
                        
                        # Cache the result
                        DIALECT_CACHE[dialect_cache_key] = {
                            'data': result,
                            'timestamp': datetime.now()
                        }
                        
                        return result
            
            # Not found
            return None
        except Exception as e:
            logger.error(f"Error getting dialect by ID {dialect_id}: {e}")
            return None
            
    async def get_npc_speech_patterns(self, npc_id: int) -> Dict[str, Any]:
        """
        Get speech patterns for an NPC.
        
        Args:
            npc_id: ID of the NPC
            
        Returns:
            Dictionary with speech pattern details
        """
        try:
            # Check if we are initialized
            if not self.initialized:
                await self.initialize()
                
            # Default speech patterns
            speech_patterns = {
                'default_language': None,
                'languages': [],
                'primary_dialect': None,
                'formality_level': 'neutral',
                'speech_characteristics': {},
                'cultural_expressions': []
            }
            
            # Get NPC's cultural attributes including languages and dialect
            from data.npc_dal import NPCDataAccess
            
            cultural_attributes = await NPCDataAccess.get_npc_cultural_attributes(
                npc_id=npc_id,
                user_id=self.user_id,
                conversation_id=self.conversation_id
            )
            
            if not cultural_attributes:
                return speech_patterns
                
            # Process languages
            languages = cultural_attributes.get('languages', [])
            if languages:
                speech_patterns['languages'] = languages
                
                # Set default language as the one with highest fluency
                speech_patterns['default_language'] = languages[0]
                
            # Process dialect
            primary_dialect = cultural_attributes.get('primary_dialect')
            if primary_dialect:
                speech_patterns['primary_dialect'] = primary_dialect
                
                # Extract dialect features if available
                dialect_features = cultural_attributes.get('dialect_features', {})
                
                for feature, description in dialect_features.items():
                    if feature not in speech_patterns['speech_characteristics']:
                        speech_patterns['speech_characteristics'][feature] = []
                    
                    speech_patterns['speech_characteristics'][feature].append(description)
            
            # Get NPC personality for additional speech characteristics
            from lore.npc_lore_integration import NPCLoreIntegration
            
            # Use the same instance to ensure consistency
            npc_integration = NPCLoreIntegration(self.user_id, self.conversation_id)
            
            personality = await npc_integration._get_npc_personality(npc_id)
            
            # Map personality traits to speech characteristics
            personality_speech_map = {
                'formality': ['polite', 'proper', 'formal', 'respectful'],
                'dialect_strength': ['traditional', 'cultural'],
                'vocabulary': ['intelligent', 'educated', 'articulate', 'simple', 'direct'],
                'speech_rate': ['quick', 'slow', 'deliberate', 'thoughtful'],
                'confidence': ['confident', 'shy', 'assertive', 'hesitant'],
                'expression': ['expressive', 'reserved', 'animated', 'monotone']
            }
            
            # Initialize speech characteristics based on personality
            for speech_aspect, related_traits in personality_speech_map.items():
                matching_traits = []
                
                for trait_name, trait_value in personality.get('traits', {}).items():
                    # Check if this personality trait maps to this speech aspect
                    if any(related in trait_name.lower() for related in related_traits):
                        # Add to matching traits with strength
                        matching_traits.append((trait_name, trait_value))
                
                # If we found matching traits, add the strongest one to speech characteristics
                if matching_traits:
                    # Sort by strength (highest first)
                    matching_traits.sort(key=lambda x: x[1], reverse=True)
                    
                    # Add the strongest trait to speech characteristics
                    if speech_aspect not in speech_patterns['speech_characteristics']:
                        speech_patterns['speech_characteristics'][speech_aspect] = []
                        
                    speech_patterns['speech_characteristics'][speech_aspect].append(
                        f"{matching_traits[0][0]} ({matching_traits[0][1]})"
                    )
            
            # Determine formality level based on personality and cultural background
            if 'formality' in speech_patterns['speech_characteristics']:
                formality_traits = speech_patterns['speech_characteristics']['formality']
                
                if any('formal' in trait.lower() for trait in formality_traits):
                    speech_patterns['formality_level'] = 'formal'
                elif any('informal' in trait.lower() or 'casual' in trait.lower() for trait in formality_traits):
                    speech_patterns['formality_level'] = 'informal'
            
            # Get cultural expressions based on nationality and faith
            if cultural_attributes.get('nationality'):
                # Get common phrases for this nationality
                await self._add_cultural_expressions(
                    speech_patterns, 
                    'Nations', 
                    cultural_attributes['nationality']['id']
                )
                
            if cultural_attributes.get('faith'):
                # Get religious expressions
                await self._add_cultural_expressions(
                    speech_patterns, 
                    'CulturalElements',
                    cultural_attributes['faith']['id']
                )
            
            return speech_patterns
        except Exception as e:
            logger.error(f"Error getting NPC speech patterns for NPC {npc_id}: {e}")
            return {
                'default_language': None,
                'languages': [],
                'primary_dialect': None,
                'formality_level': 'neutral',
                'speech_characteristics': {},
                'cultural_expressions': []
            }
    
    async def _add_cultural_expressions(
        self, 
        speech_patterns: Dict[str, Any],
        source_type: str,
        source_id: int
    ) -> None:
        """
        Add cultural expressions from a source to speech patterns.
        
        Args:
            speech_patterns: Speech patterns dictionary to update
            source_type: Type of source (Nations, CulturalElements, etc)
            source_id: ID of the source
        """
        try:
            # Query for expressions linked to this source
            query = """
                SELECT e.phrase, e.meaning, e.context_of_use, e.formality_level
                FROM CulturalExpressions e
                WHERE e.source_type = $1 AND e.source_id = $2
            """
            
            params = [source_type, source_id]
            
            if self.user_id is not None:
                query += f" AND e.user_id = ${len(params) + 1}"
                params.append(self.user_id)
                
            if self.conversation_id is not None:
                query += f" AND e.conversation_id = ${len(params) + 1}"
                params.append(self.conversation_id)
                
            # Execute the query
            expressions = await fetch(query, *params)
            
            # Add expressions to speech patterns
            for expr in expressions:
                expr_data = dict(expr)
                
                # Only add expressions matching the formality level
                if (not expr_data.get('formality_level') or 
                    expr_data.get('formality_level') == speech_patterns['formality_level']):
                    
                    speech_patterns['cultural_expressions'].append(expr_data)
        except Exception as e:
            logger.error(f"Error adding cultural expressions from {source_type} {source_id}: {e}")
    
    async def apply_dialect_to_text(
        self, 
        text: str, 
        dialect_id: int, 
        intensity: str = 'medium',
        npc_id: Optional[int] = None
    ) -> str:
        """
        Apply dialect features to a text.
        Uses optimized regex matching and sentence boundary detection.
        
        Args:
            text: Original text
            dialect_id: ID of the dialect to apply
            intensity: Intensity of dialect application ('light', 'medium', 'strong')
            npc_id: Optional NPC ID for personalized dialect features
            
        Returns:
            Modified text with dialect features applied
        """
        if not text or not text.strip():
            return text
        
        try:
            # Check if we are initialized
            if not self.initialized:
                await self.initialize()
            
            # Get the dialect information
            dialect = await self.get_dialect_by_id(dialect_id)
            if not dialect:
                logger.warning(f"Dialect with ID {dialect_id} not found")
                return text
            
            # Determine modification probability based on intensity
            probabilities = {
                'light': 0.3,
                'medium': 0.6,
                'strong': 0.9
            }
            probability = probabilities.get(intensity.lower(), 0.5)
            
            # Get NPC-specific speech patterns if an NPC ID is provided
            npc_speech_patterns = None
            if npc_id:
                npc_speech_patterns = await self.get_npc_speech_patterns(npc_id)
            
            # Extract dialect features
            accent_features = dialect.get('accent', {})
            vocabulary = dialect.get('vocabulary', {})
            grammar_rules = dialect.get('grammar', {})
            
            # Precompile regex patterns for efficiency
            # Use word boundaries to ensure we're replacing whole words
            accent_patterns = {}
            for original, replacement in accent_features.items():
                if original and replacement:  # Skip empty entries
                    try:
                        accent_patterns[original] = (
                            re.compile(r'\b' + re.escape(original) + r'\b', re.IGNORECASE),
                            replacement
                        )
                    except re.error:
                        logger.warning(f"Invalid regex pattern for accent feature: {original}")
            
            vocab_patterns = {}
            for standard_word, dialect_word in vocabulary.items():
                if standard_word and dialect_word:  # Skip empty entries
                    try:
                        vocab_patterns[standard_word] = (
                            re.compile(r'\b' + re.escape(standard_word) + r'\b', re.IGNORECASE),
                            dialect_word
                        )
                    except re.error:
                        logger.warning(f"Invalid regex pattern for vocabulary: {standard_word}")
            
            grammar_patterns = {}
            for grammar_rule, replacement in grammar_rules.items():
                if grammar_rule and replacement and " " in grammar_rule:  # Skip empty entries and single words
                    try:
                        grammar_patterns[grammar_rule] = (
                            re.compile(r'\b' + re.escape(grammar_rule) + r'\b', re.IGNORECASE),
                            replacement
                        )
                    except re.error:
                        logger.warning(f"Invalid regex pattern for grammar rule: {grammar_rule}")
            
            # Better sentence boundary detection with support for common abbreviations
            # This improves the accuracy of dialect application at sentence boundaries
            abbreviations = r'(?<!\b(?:Mr|Mrs|Ms|Dr|Prof|Jr|Sr|etc|e\.g|i\.e)\.)(?<![A-Z]\.)(?<=\.|\?|\!)\s'
            sentence_boundary = re.compile(abbreviations)
            
            # Split text into sentences
            sentences = sentence_boundary.split(text)
            if not sentences:
                return text  # Nothing to process
            
            # Process each sentence
            import random
            modified_sentences = []
            
            for sentence in sentences:
                modified_sentence = sentence
                
                # Apply accent modifications based on intensity/probability
                # For strong intensity, apply more changes
                if intensity == 'strong':
                    # Apply more accent features for strong intensity
                    for _, (pattern, replacement) in accent_patterns.items():
                        modified_sentence = pattern.sub(replacement, modified_sentence)
                else:
                    # For medium/light intensity, apply fewer changes
                    for _, (pattern, replacement) in accent_patterns.items():
                        if random.random() < probability:
                            modified_sentence = pattern.sub(replacement, modified_sentence)
                
                # Apply vocabulary substitutions based on intensity/probability
                for _, (pattern, replacement) in vocab_patterns.items():
                    if random.random() < probability:
                        modified_sentence = pattern.sub(replacement, modified_sentence)
                
                # Apply grammar modifications for medium/strong intensity only
                if intensity != 'light':
                    for _, (pattern, replacement) in grammar_patterns.items():
                        if random.random() < probability:
                            modified_sentence = pattern.sub(replacement, modified_sentence)
                
                # Add to modified sentences
                modified_sentences.append(modified_sentence)
            
            # Join sentences back together
            modified_text = ' '.join(modified_sentences)
            
            # Add cultural expressions if available (for strong intensity only)
            if intensity == 'strong' and npc_speech_patterns and npc_speech_patterns.get('cultural_expressions'):
                # Get a random cultural expression
                if npc_speech_patterns['cultural_expressions']:
                    expression = random.choice(npc_speech_patterns['cultural_expressions'])
                    
                    # Add the expression at the beginning or end of the text
                    if random.random() > 0.5:
                        modified_text = f"{expression['phrase']} {modified_text}"
                    else:
                        modified_text = f"{modified_text} {expression['phrase']}"
            
            return modified_text
        except Exception as e:
            logger.error(f"Error applying dialect to text: {e}", exc_info=True)
            return text  # Return original text on error
    
    # Additional methods for working with lore elements will be added here
    
# Create a singleton instance for easy access
lore_system = LoreSystem() 