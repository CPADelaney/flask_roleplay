# lore/generators.py

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import random
import logging

logger = logging.getLogger(__name__)

@dataclass
class ComponentConfig:
    """Configuration for component generation"""
    min_length: int = 100
    max_length: int = 500
    style: str = "descriptive"
    tone: str = "neutral"
    include_metadata: bool = True

class ComponentGenerator:
    """Base class for all component generators"""
    def __init__(self, config: ComponentConfig):
        self.config = config
        self._cache = {}
    
    def generate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a component with the given context"""
        raise NotImplementedError
    
    def _get_cached(self, key: str) -> Optional[Dict[str, Any]]:
        """Get a cached component if available"""
        return self._cache.get(key)
    
    def _cache_component(self, key: str, component: Dict[str, Any]):
        """Cache a generated component"""
        self._cache[key] = component

class CharacterGenerator(ComponentGenerator):
    """Generator for character components"""
    def generate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        cache_key = f"character_{context.get('name', '')}"
        if cached := self._get_cached(cache_key):
            return cached
            
        component = {
            "type": "character",
            "name": context.get("name", "Unknown Character"),
            "description": self._generate_description(context),
            "traits": self._generate_traits(context),
            "background": self._generate_background(context),
            "relationships": self._generate_relationships(context),
            "metadata": {
                "created_at": datetime.utcnow().isoformat(),
                "version": "1.0"
            } if self.config.include_metadata else {}
        }
        
        self._cache_component(cache_key, component)
        return component
    
    def _generate_description(self, context: Dict[str, Any]) -> str:
        """Generate a detailed description for a character."""
        try:
            # Extract relevant context
            name = context.get("name", "Unknown Character")
            role = context.get("role", "Unknown Role")
            background = context.get("background", {})
            
            # Build description components
            physical_desc = self._generate_physical_description(context)
            personality_desc = self._generate_personality_description(context)
            background_desc = self._generate_background_description(background)
            
            # Combine descriptions
            description = f"{name} is {physical_desc}. {personality_desc} {background_desc}"
            
            return description
        except Exception as e:
            logger.error(f"Error generating character description: {str(e)}")
            return f"Description for {name}"
    
    def _generate_traits(self, context: Dict[str, Any]) -> List[str]:
        """Generate character traits based on context."""
        try:
            traits = []
            
            # Extract personality traits
            personality = context.get("personality", {})
            if personality:
                traits.extend(self._extract_personality_traits(personality))
            
            # Extract skills and abilities
            skills = context.get("skills", [])
            if skills:
                traits.extend(self._extract_skill_traits(skills))
            
            # Extract background traits
            background = context.get("background", {})
            if background:
                traits.extend(self._extract_background_traits(background))
            
            # Ensure unique traits
            traits = list(set(traits))
            
            return traits
        except Exception as e:
            logger.error(f"Error generating character traits: {str(e)}")
            return ["Trait 1", "Trait 2", "Trait 3"]
    
    def _generate_background(self, context: Dict[str, Any]) -> str:
        """Generate a detailed background story for a character."""
        try:
            # Extract background components
            origin = context.get("origin", "Unknown Origin")
            history = context.get("history", [])
            relationships = context.get("relationships", {})
            
            # Build background components
            origin_story = self._generate_origin_story(origin)
            history_story = self._generate_history_story(history)
            relationship_story = self._generate_relationship_story(relationships)
            
            # Combine background components
            background = f"{origin_story} {history_story} {relationship_story}"
            
            return background
        except Exception as e:
            logger.error(f"Error generating character background: {str(e)}")
            return f"Background for {context.get('name', 'Unknown Character')}"
    
    def _generate_relationships(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate relationship data for a character."""
        try:
            relationships = []
            
            # Extract relationship data
            family = context.get("family", {})
            friends = context.get("friends", [])
            enemies = context.get("enemies", [])
            allies = context.get("allies", [])
            
            # Generate family relationships
            if family:
                relationships.extend(self._generate_family_relationships(family))
            
            # Generate friend relationships
            if friends:
                relationships.extend(self._generate_friend_relationships(friends))
            
            # Generate enemy relationships
            if enemies:
                relationships.extend(self._generate_enemy_relationships(enemies))
            
            # Generate ally relationships
            if allies:
                relationships.extend(self._generate_ally_relationships(allies))
            
            return relationships
        except Exception as e:
            logger.error(f"Error generating character relationships: {str(e)}")
            return []

class LocationGenerator(ComponentGenerator):
    """Generator for location components"""
    def generate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        cache_key = f"location_{context.get('name', '')}"
        if cached := self._get_cached(cache_key):
            return cached
            
        component = {
            "type": "location",
            "name": context.get("name", "Unknown Location"),
            "description": self._generate_description(context),
            "climate": self._generate_climate(context),
            "geography": self._generate_geography(context),
            "culture": self._generate_culture(context),
            "metadata": {
                "created_at": datetime.utcnow().isoformat(),
                "version": "1.0"
            } if self.config.include_metadata else {}
        }
        
        self._cache_component(cache_key, component)
        return component
    
    def _generate_description(self, context: Dict[str, Any]) -> str:
        """Generate a detailed description for a location."""
        try:
            # Extract relevant context
            name = context.get("name", "Unknown Location")
            type = context.get("type", "Unknown Type")
            features = context.get("features", [])
            
            # Build description components
            physical_desc = self._generate_physical_description(context)
            atmosphere_desc = self._generate_atmosphere_description(context)
            feature_desc = self._generate_feature_description(features)
            
            # Combine descriptions
            description = f"{name} is {physical_desc}. {atmosphere_desc} {feature_desc}"
            
            return description
        except Exception as e:
            logger.error(f"Error generating location description: {str(e)}")
            return f"Description for {name}"
    
    def _generate_climate(self, context: Dict[str, Any]) -> str:
        """Generate climate information for a location."""
        try:
            # Extract climate data
            base_climate = context.get("climate", "Unknown")
            seasonal_variations = context.get("seasonal_variations", {})
            weather_patterns = context.get("weather_patterns", [])
            
            # Build climate description
            climate_desc = self._build_climate_description(
                base_climate,
                seasonal_variations,
                weather_patterns
            )
            
            return climate_desc
        except Exception as e:
            logger.error(f"Error generating climate: {str(e)}")
            return "Temperate"
    
    def _generate_geography(self, context: Dict[str, Any]) -> str:
        """Generate geographical information for a location."""
        try:
            # Extract geography data
            terrain = context.get("terrain", "Unknown")
            features = context.get("geographical_features", [])
            resources = context.get("natural_resources", [])
            
            # Build geography description
            geography_desc = self._build_geography_description(
                terrain,
                features,
                resources
            )
            
            return geography_desc
        except Exception as e:
            logger.error(f"Error generating geography: {str(e)}")
            return "Mountainous"
    
    def _generate_culture(self, context: Dict[str, Any]) -> str:
        """Generate cultural information for a location."""
        try:
            # Extract culture data
            inhabitants = context.get("inhabitants", [])
            customs = context.get("customs", [])
            traditions = context.get("traditions", [])
            
            # Build culture description
            culture_desc = self._build_culture_description(
                inhabitants,
                customs,
                traditions
            )
            
            return culture_desc
        except Exception as e:
            logger.error(f"Error generating culture: {str(e)}")
            return "Diverse"

class EventGenerator(ComponentGenerator):
    """Generator for event components"""
    def generate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        cache_key = f"event_{context.get('name', '')}"
        if cached := self._get_cached(cache_key):
            return cached
            
        component = {
            "type": "event",
            "name": context.get("name", "Unknown Event"),
            "description": self._generate_description(context),
            "date": self._generate_date(context),
            "participants": self._generate_participants(context),
            "consequences": self._generate_consequences(context),
            "metadata": {
                "created_at": datetime.utcnow().isoformat(),
                "version": "1.0"
            } if self.config.include_metadata else {}
        }
        
        self._cache_component(cache_key, component)
        return component
    
    def _generate_description(self, context: Dict[str, Any]) -> str:
        """Generate a detailed description for an event."""
        try:
            # Extract relevant context
            name = context.get("name", "Unknown Event")
            type = context.get("type", "Unknown Type")
            details = context.get("details", {})
            
            # Build description components
            event_desc = self._generate_event_details(details)
            impact_desc = self._generate_event_impact(details)
            consequence_desc = self._generate_event_consequences(details)
            
            # Combine descriptions
            description = f"{name} was {event_desc}. {impact_desc} {consequence_desc}"
            
            return description
        except Exception as e:
            logger.error(f"Error generating event description: {str(e)}")
            return f"Description for {name}"
    
    def _generate_date(self, context: Dict[str, Any]) -> str:
        """Generate date information for an event."""
        try:
            # Extract date data
            year = context.get("year")
            month = context.get("month")
            day = context.get("day")
            era = context.get("era", "CE")
            
            # Build date string
            date_str = self._build_date_string(year, month, day, era)
            
            return date_str
        except Exception as e:
            logger.error(f"Error generating event date: {str(e)}")
            return datetime.utcnow().isoformat()
    
    def _generate_participants(self, context: Dict[str, Any]) -> List[str]:
        """Generate participant information for an event."""
        try:
            participants = []
            
            # Extract participant data
            primary_participants = context.get("primary_participants", [])
            secondary_participants = context.get("secondary_participants", [])
            groups = context.get("participating_groups", [])
            
            # Add primary participants
            participants.extend(primary_participants)
            
            # Add secondary participants
            participants.extend(secondary_participants)
            
            # Add group participants
            for group in groups:
                participants.extend(self._expand_group_participants(group))
            
            # Ensure unique participants
            participants = list(set(participants))
            
            return participants
        except Exception as e:
            logger.error(f"Error generating event participants: {str(e)}")
            return []
    
    def _generate_consequences(self, context: Dict[str, Any]) -> List[str]:
        """Generate consequence information for an event."""
        try:
            consequences = []
            
            # Extract consequence data
            immediate_effects = context.get("immediate_effects", [])
            long_term_effects = context.get("long_term_effects", [])
            ripple_effects = context.get("ripple_effects", [])
            
            # Add immediate effects
            consequences.extend(immediate_effects)
            
            # Add long-term effects
            consequences.extend(long_term_effects)
            
            # Add ripple effects
            consequences.extend(ripple_effects)
            
            # Ensure unique consequences
            consequences = list(set(consequences))
            
            return consequences
        except Exception as e:
            logger.error(f"Error generating event consequences: {str(e)}")
            return []

class ComponentGeneratorFactory:
    """Factory for creating component generators"""
    @staticmethod
    def create_generator(component_type: str, config: ComponentConfig) -> ComponentGenerator:
        generators = {
            "character": CharacterGenerator,
            "location": LocationGenerator,
            "event": EventGenerator
        }
        
        generator_class = generators.get(component_type.lower())
        if not generator_class:
            raise ValueError(f"Unknown component type: {component_type}")
            
        return generator_class(config) 
