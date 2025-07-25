# story_templates/moth/lore/world_lore_manager.py
"""
Central entry point for SF Bay Area lore management
Maintains backward compatibility while organizing into modules
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

# Import all submodules
from .education import SFEducationLore
from .geopolitical import SFGeopoliticalLore
from .local_lore import SFLocalLore
from .politics import SFPoliticsLore
from .religion import SFReligionLore

logger = logging.getLogger(__name__)


class SFBayMothFlamePreset:
    """Complete preset lore for The Moth and Flame set in SF Bay Area"""
    
    @staticmethod
    def get_world_foundation() -> Dict[str, Any]:
        """Get the base world setting"""
        return {
            "world_setting": {
                "name": "San Francisco Bay Metropolitan Area",
                "type": "modern_gothic_city",
                "population": "7.7 million",
                "established": "1776 (Mission), 1835 (Yerba Buena)",
                "description": (
                    "A city of stark contrasts where tech billionaires step over the homeless, "
                    "where Victorian painted ladies cast shadows on underground dungeons. The fog "
                    "rolls in each evening like a living thing, obscuring sins and swallowing screams. "
                    "Beneath the progressive facade thrives an economy of desire, desperation, and "
                    "dark salvation. They say the city's heart beats strongest after midnight."
                ),
                "atmosphere": {
                    "physical": "Fog-shrouded hills, cold ocean wind, Victorian shadows",
                    "social": "Tech wealth, artistic rebellion, hidden desperation",
                    "spiritual": "Lost souls seeking meaning in sensation"
                }
            }
        }
    
    # Delegate to submodules
    @staticmethod
    def get_districts() -> List[Dict[str, Any]]:
        """Get the main districts/locations"""
        return SFGeopoliticalLore.get_districts()
    
    @staticmethod
    def get_specific_locations() -> List[Dict[str, Any]]:
        """Get story-specific detailed locations"""
        return SFLocalLore.get_specific_locations()
    
    @staticmethod
    def get_urban_myths() -> List[Dict[str, Any]]:
        """Get SF-specific urban myths for the story"""
        return SFLocalLore.get_urban_myths()
    
    @staticmethod
    def get_historical_events() -> List[Dict[str, Any]]:
        """Get historical events that shape the story world"""
        return SFLocalLore.get_historical_events()
    
    @staticmethod
    def get_factions() -> List[Dict[str, Any]]:
        """Get the power factions in the story"""
        return SFGeopoliticalLore.get_factions()
    
    @staticmethod
    def get_cultural_elements() -> List[Dict[str, Any]]:
        """Get unique cultural elements of the SF underground"""
        return SFLocalLore.get_cultural_elements()
    
    @staticmethod
    def get_landmarks() -> List[Dict[str, Any]]:
        """Get significant landmarks for the story"""
        return SFLocalLore.get_landmarks()
    
    @staticmethod
    def get_conflicts() -> List[Dict[str, Any]]:
        """Get ongoing conflicts in the story world"""
        return SFGeopoliticalLore.get_conflicts()
    
    @staticmethod
    def get_notable_figures() -> List[Dict[str, Any]]:
        """Get important NPCs beyond the main cast"""
        return SFGeopoliticalLore.get_notable_figures()
    
    @staticmethod
    def get_educational_systems() -> List[Dict[str, Any]]:
        """Get educational institutions for SF Bay Area"""
        return SFEducationLore.get_educational_systems()

    @staticmethod
    def get_religious_institutions() -> List[Dict[str, Any]]:
        """Get religious organizations in SF Bay Area"""
        return SFReligionLore.get_religious_institutions()

    @staticmethod
    def get_knowledge_traditions() -> List[Dict[str, Any]]:
        """Get knowledge transmission traditions"""
        return SFEducationLore.get_knowledge_traditions()

    @staticmethod
    def get_quest_hooks() -> List[Dict[str, Any]]:
        """Get story quests and missions"""
        return SFPoliticsLore.get_quest_hooks()

    @staticmethod
    def get_domestic_issues() -> List[Dict[str, Any]]:
        """Get local political conflicts"""
        return SFPoliticsLore.get_domestic_issues()
    
    @staticmethod
    def get_pantheons() -> List[Dict[str, Any]]:
        """Get belief systems that function as 'religions' in modern setting"""
        return SFReligionLore.get_pantheons()
    
    @staticmethod
    def get_deities() -> List[Dict[str, Any]]:
        """Get deity-like figures in the modern pantheon"""
        return SFReligionLore.get_deities()
    
    @staticmethod
    def get_religious_practices() -> List[Dict[str, Any]]:
        """Get ritual practices of the underground faiths"""
        return SFReligionLore.get_religious_practices()
    
    @staticmethod
    def get_holy_sites() -> List[Dict[str, Any]]:
        """Get sacred locations beyond the main venues"""
        return SFReligionLore.get_holy_sites()
    
    @staticmethod
    def get_religious_texts() -> List[Dict[str, Any]]:
        """Get sacred writings and teachings"""
        return SFReligionLore.get_religious_texts()
    
    @staticmethod
    def get_religious_orders() -> List[Dict[str, Any]]:
        """Get organized groups within the faiths"""
        return SFReligionLore.get_religious_orders()
    
    @staticmethod
    def get_religious_conflicts() -> List[Dict[str, Any]]:
        """Get theological and practical disputes"""
        return SFReligionLore.get_religious_conflicts()
    
    @staticmethod
    def get_faction_relationships() -> List[Dict[str, Any]]:
        """Get detailed faction relationships beyond basic conflicts"""
        return SFGeopoliticalLore.get_faction_relationships()
    
    @staticmethod
    def get_district_religious_distribution() -> List[Dict[str, Any]]:
        """Get religious distribution by district"""
        return SFReligionLore.get_district_religious_distribution()
    
    @staticmethod
    def get_regional_religious_practices() -> List[Dict[str, Any]]:
        """Get regional variations of religious practices"""
        return SFReligionLore.get_regional_religious_practices()
    
    @staticmethod
    def get_underground_etiquette() -> List[Dict[str, Any]]:
        """Get etiquette rules for different underground contexts"""
        return SFLocalLore.get_underground_etiquette()
    
    @staticmethod
    def get_district_cultural_norms() -> List[Dict[str, Any]]:
        """Get detailed cultural norms for each SF district"""
        return SFLocalLore.get_district_cultural_norms()
    
    @staticmethod
    def get_underground_languages() -> List[Dict[str, Any]]:
        """Get languages specific to the SF Bay underground culture"""
        return SFEducationLore.get_underground_languages()
    
    @staticmethod
    def get_mystical_phenomena() -> List[Dict[str, Any]]:
        """Get supernatural/psychological phenomena specific to setting"""
        return SFLocalLore.get_mystical_phenomena()
    
    @staticmethod
    def get_seasonal_events() -> List[Dict[str, Any]]:
        """Get regular events beyond daily operations"""
        return SFLocalLore.get_seasonal_events()
    
    @staticmethod
    def get_specialized_locations() -> List[Dict[str, Any]]:
        """Get additional specialized underground locations"""
        return SFLocalLore.get_specialized_locations()
    
    @staticmethod
    def get_communication_networks() -> List[Dict[str, Any]]:
        """Get how the underground communicates"""
        return SFLocalLore.get_communication_networks()
    
    @staticmethod
    def get_underground_economies() -> List[Dict[str, Any]]:
        """Get economic systems within the underground"""
        return SFLocalLore.get_underground_economies()
    
    @staticmethod
    def get_expanded_seasonal_events() -> List[Dict[str, Any]]:
        """More seasonal/cyclical events"""
        return SFLocalLore.get_expanded_seasonal_events()
    
    @staticmethod
    def get_myth_evolution_scenarios() -> List[Dict[str, Any]]:
        """How urban myths evolve over time in the SF underground"""
        return SFLocalLore.get_myth_evolution_scenarios()
    
    @staticmethod
    def get_cultural_evolution_scenarios() -> List[Dict[str, Any]]:
        """How underground cultural elements develop"""
        return SFLocalLore.get_cultural_evolution_scenarios()
    
    @staticmethod
    def get_geopolitical_shift_scenarios() -> List[Dict[str, Any]]:
        """Power dynamics evolution in the Bay Area underground"""
        return SFGeopoliticalLore.get_geopolitical_shift_scenarios()
    
    @staticmethod
    def get_language_evolution() -> List[Dict[str, Any]]:
        """How underground languages evolve"""
        return SFEducationLore.get_language_evolution()
    
    @staticmethod
    def get_underground_economy_evolution() -> List[Dict[str, Any]]:
        """How the shadow economy adapts"""
        return SFLocalLore.get_underground_economy_evolution()
    
    @staticmethod
    def get_communication_evolution() -> List[Dict[str, Any]]:
        """How communication networks adapt"""
        return SFLocalLore.get_communication_evolution()
    
    @staticmethod
    def get_figure_evolution_scenarios() -> List[Dict[str, Any]]:
        """How key figures' stories evolve"""
        return SFGeopoliticalLore.get_figure_evolution_scenarios()
    
    @staticmethod
    def get_world_evolution_scenarios() -> List[Dict[str, Any]]:
        """Major evolutionary scenarios for the entire setting"""
        return SFGeopoliticalLore.get_world_evolution_scenarios()
    
    @staticmethod
    def get_conflict_news_cycles() -> List[Dict[str, Any]]:
        """Get news articles for major conflicts"""
        return SFPoliticsLore.get_conflict_news_cycles()
    
    @staticmethod
    def apply_sf_matriarchal_themes(content: Dict[str, Any]) -> Dict[str, Any]:
        """Apply matriarchal themes to SF-specific content"""
        
        # Apply to myths
        for myth in content.get('myths', []):
            myth['matriarchal_elements'] = []
            
            if 'moth queen' in myth['name'].lower():
                myth['matriarchal_elements'].extend([
                    'female_savior_figure',
                    'protective_mother_archetype',
                    'feminine_power_over_predators'
                ])
                
            if 'fog mother' in myth['name'].lower():
                myth['matriarchal_elements'].extend([
                    'nature_as_feminine_protector',
                    'mother_earth_manifestation'
                ])
        
        # Apply to factions
        for faction in content.get('factions', []):
            if faction['type'] == 'underground_authority':
                faction['matriarchal_structure'] = {
                    'leadership': 'Absolute feminine authority',
                    'succession': 'Female to female',
                    'male_roles': 'Protectors and servants'
                }
        
        # Apply to religious elements
        for religion in content.get('religious_institutions', []):
            religion['feminine_divine_aspect'] = True
            religion['matriarchal_interpretation'] = (
                "Even traditional faiths acknowledge feminine divine power "
                "in this underground context"
            )
        
        return content

    @staticmethod
    async def initialize_complete_sf_preset(ctx, user_id: int, conversation_id: int):
        """Initialize ALL lore components for SF Bay preset"""
        logger.info("Initializing complete SF Bay Area preset for The Moth and Flame")
        
        from lore.core.lore_system import LoreSystem
        lore_system = await LoreSystem.get_instance(user_id, conversation_id)
        
        # Get all preset data - now includes everything
        preset_data = {
            # World foundation
            "world": SFBayMothFlamePreset.get_world_foundation(),
            
            # Geopolitical elements
            "districts": SFBayMothFlamePreset.get_districts(),
            "factions": SFBayMothFlamePreset.get_factions(),
            "faction_relationships": SFBayMothFlamePreset.get_faction_relationships(),
            "conflicts": SFBayMothFlamePreset.get_conflicts(),
            "figures": SFBayMothFlamePreset.get_notable_figures(),
            
            # Local lore elements
            "locations": SFBayMothFlamePreset.get_specific_locations(),
            "myths": SFBayMothFlamePreset.get_urban_myths(),
            "history": SFBayMothFlamePreset.get_historical_events(),
            "culture": SFBayMothFlamePreset.get_cultural_elements(),
            "landmarks": SFBayMothFlamePreset.get_landmarks(),
            "seasonal_events": SFBayMothFlamePreset.get_seasonal_events(),
            "specialized_locations": SFBayMothFlamePreset.get_specialized_locations(),
            "communication_networks": SFBayMothFlamePreset.get_communication_networks(),
            "underground_economies": SFBayMothFlamePreset.get_underground_economies(),
            "expanded_seasonal_events": SFBayMothFlamePreset.get_expanded_seasonal_events(),
            "cultural_norms": SFBayMothFlamePreset.get_district_cultural_norms(),
            "etiquette": SFBayMothFlamePreset.get_underground_etiquette(),
            "mystical_phenomena": SFBayMothFlamePreset.get_mystical_phenomena(),
            
            # Educational elements
            "education": SFBayMothFlamePreset.get_educational_systems(),
            "knowledge": SFBayMothFlamePreset.get_knowledge_traditions(),
            "languages": SFBayMothFlamePreset.get_underground_languages(),
            "language_evolution": SFBayMothFlamePreset.get_language_evolution(),
            
            # Political elements
            "domestic_issues": SFBayMothFlamePreset.get_domestic_issues(),
            "quests": SFBayMothFlamePreset.get_quest_hooks(),
            "news_cycles": SFBayMothFlamePreset.get_conflict_news_cycles(),
            
            # Religious elements
            "religion": SFBayMothFlamePreset.get_religious_institutions(),
            "pantheons": SFBayMothFlamePreset.get_pantheons(),
            "deities": SFBayMothFlamePreset.get_deities(),
            "religious_practices": SFBayMothFlamePreset.get_religious_practices(),
            "holy_sites": SFBayMothFlamePreset.get_holy_sites(),
            "religious_texts": SFBayMothFlamePreset.get_religious_texts(),
            "religious_orders": SFBayMothFlamePreset.get_religious_orders(),
            "religious_conflicts": SFBayMothFlamePreset.get_religious_conflicts(),
            "religious_distribution": SFBayMothFlamePreset.get_district_religious_distribution(),
            "regional_religious": SFBayMothFlamePreset.get_regional_religious_practices(),
            
            # Evolution scenarios
            "myth_evolution": SFBayMothFlamePreset.get_myth_evolution_scenarios(),
            "cultural_evolution": SFBayMothFlamePreset.get_cultural_evolution_scenarios(),
            "geopolitical_shifts": SFBayMothFlamePreset.get_geopolitical_shift_scenarios(),
            "economy_evolution": SFBayMothFlamePreset.get_underground_economy_evolution(),
            "communication_evolution": SFBayMothFlamePreset.get_communication_evolution(),
            "figure_evolution": SFBayMothFlamePreset.get_figure_evolution_scenarios(),
            "world_evolution": SFBayMothFlamePreset.get_world_evolution_scenarios()
        }
        
        # Apply matriarchal themes
        preset_data = SFBayMothFlamePreset.apply_sf_matriarchal_themes(preset_data)
        
        # Initialize through proper managers
        # Education
        from lore.managers.education import get_education_manager
        edu_mgr = await get_education_manager(user_id, conversation_id)
        
        for edu_system in preset_data['education']:
            await edu_mgr.add_educational_system(ctx, **edu_system)
        
        for tradition in preset_data['knowledge']:
            await edu_mgr.add_knowledge_tradition(ctx, **tradition)
        
        for language in preset_data['languages']:
            await edu_mgr.add_language(ctx, **language)
        
        # Religion
        from lore.managers.religion import get_religion_manager
        religion_mgr = await get_religion_manager(user_id, conversation_id)
        
        for pantheon in preset_data['pantheons']:
            await religion_mgr.add_pantheon(ctx, **pantheon)
        
        for deity in preset_data['deities']:
            await religion_mgr.add_deity(ctx, **deity)
        
        for practice in preset_data['religious_practices']:
            await religion_mgr.add_religious_practice(ctx, **practice)
        
        for site in preset_data['holy_sites']:
            await religion_mgr.add_holy_site(ctx, **site)
        
        for text in preset_data['religious_texts']:
            await religion_mgr.add_religious_text(ctx, **text)
        
        for order in preset_data['religious_orders']:
            await religion_mgr.add_religious_order(ctx, **order)
        
        # Politics
        from lore.managers.politics import WorldPoliticsManager
        politics_mgr = WorldPoliticsManager(user_id, conversation_id)
        await politics_mgr.ensure_initialized()
        
        # Create SF as a "nation" for the politics system
        sf_nation_id = await politics_mgr.add_nation(
            ctx,
            name="San Francisco Bay Area",
            government_type="Municipal Democracy",
            description="Progressive city with dark underbelly",
            relative_power=8,
            matriarchy_level=6,  # Underground is matriarchal
            population_scale="7.7 million",
            major_resources=["Tech industry", "Port trade", "Tourism"],
            major_cities=["San Francisco", "Oakland", "San Jose"],
            cultural_traits=["Progressive", "Tech-focused", "Diverse", "Underground culture"]
        )
        
        # Add domestic issues
        for issue in preset_data['domestic_issues']:
            await politics_mgr.generate_domestic_issues(ctx, sf_nation_id, issue)
        
        # Add factions
        for faction in preset_data['factions']:
            await politics_mgr.add_faction(ctx, sf_nation_id, **faction)
        
        # Local lore
        from lore.managers.local_lore import (
            add_urban_myth, add_local_history, add_landmark,
            generate_location_lore, LocationDataInput,
            MythCreationInput, HistoryCreationInput, LandmarkCreationInput
        )
        
        # Initialize districts as locations
        for district in preset_data['districts']:
            location_input = LocationDataInput(
                id=0,  # Will be assigned
                location_name=district['name'],
                location_type=district['type'],
                description=district['description']
            )
            
            # Generate location
            await generate_location_lore(ctx, location_input)
        
        # Add all urban myths
        for myth in preset_data['myths']:
            myth_input = MythCreationInput(
                name=myth['name'],
                description=myth['description'],
                origin_location=myth['origin_location'],
                believability=myth['believability'],
                spread_rate=8,  # High spread for well-known myths
                regions_known=myth['spread_regions'],
                themes=['gothic', 'protection', 'transformation'],
                matriarchal_elements=myth.get('matriarchal_elements', ['female_savior', 'moth_queen'])
            )
            
            await add_urban_myth(ctx, myth_input)
        
        # Add historical events
        for event in preset_data['history']:
            # First ensure location exists
            location_id = await _ensure_location_exists(ctx, event['location'])
            
            history_input = HistoryCreationInput(
                location_id=location_id,
                event_name=event['event_name'],
                description=event['description'],
                date_description=event['date'],
                significance=event['significance'],
                impact_type='transformative',
                notable_figures=['The Moth Queen lineage'],
                current_relevance=event.get('legacy', '')
            )
            
            await add_local_history(ctx, history_input)
        
        # Add specific landmarks
        for location in preset_data['locations']:
            # Get district ID
            district_name = _determine_district(location['name'])
            location_id = await _ensure_location_exists(ctx, district_name)
            
            landmark_input = LandmarkCreationInput(
                name=location['name'],
                location_id=location_id,
                landmark_type=location['type'],
                description=location['description'],
                current_use=location.get('schedule', {}).get('Monday', 'Active'),
                controlled_by='The Velvet Court',
                legends=[m['name'] for m in preset_data['myths'] if location['name'] in m.get('description', '')],
                matriarchal_significance='high'
            )
            
            await add_landmark(ctx, landmark_input)
        
        # Add cultural elements
        from lore.managers.culture import get_culture_manager
        culture_mgr = await get_culture_manager(user_id, conversation_id)
        
        for element in preset_data['culture']:
            await culture_mgr.add_cultural_element(ctx, **element)
        
        for norm in preset_data['cultural_norms']:
            await culture_mgr.add_cultural_norm(ctx, **norm)
        
        for etiquette in preset_data['etiquette']:
            await culture_mgr.add_etiquette_rule(ctx, **etiquette)
        
        logger.info("Complete SF Bay Area preset initialized successfully")
        
        return preset_data


# Enhanced story initializer that uses the preset
class EnhancedMothFlameInitializer:
    """Enhanced initializer that loads all preset lore"""
    
    @staticmethod
    async def initialize_with_sf_preset(ctx, user_id: int, conversation_id: int):
        """Initialize the story with full SF Bay Area preset"""
        
        # Use the centralized initialization
        preset_data = await SFBayMothFlamePreset.initialize_complete_sf_preset(
            ctx, user_id, conversation_id
        )
        
        logger.info("SF Bay Area preset lore initialized successfully via enhanced initializer")
        
        return preset_data


# Helper functions
async def _ensure_location_exists(ctx, location_name: str) -> int:
    """Ensure a location exists and return its ID"""
    from lore.core import canon
    from db.connection import get_db_connection_context
    
    async with get_db_connection_context() as conn:
        location_id = await canon.find_or_create_location(
            ctx, conn, location_name
        )
        
    return location_id


def _determine_district(location_name: str) -> str:
    """Determine which district a location belongs to"""
    location_lower = location_name.lower()
    
    if 'mission' in location_lower or 'valencia' in location_lower:
        return "The Mission Underground"
    elif 'soma' in location_lower or 'folsom' in location_lower:
        return "SoMa Shadow District"
    elif 'tenderloin' in location_lower:
        return "Tenderloin Shadows"
    elif 'financial' in location_lower or 'montgomery' in location_lower:
        return "Financial District After Hours"
    elif 'marina' in location_lower:
        return "The Marina Safehouse"
    else:
        return "The Mission Underground"  # Default
