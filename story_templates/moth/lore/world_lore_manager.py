# story_templates/moth/lore/world_lore_manager.py
"""
Central entry point for SF Bay Area lore management
Coordinates the hidden world of power beneath Silicon Valley's surface
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


class SFBayQueenOfThornsPreset:
    """Complete preset lore for The Queen of Thorns set in modern SF Bay Area"""
    
    @staticmethod
    def get_world_foundation() -> Dict[str, Any]:
        """Get the base world setting"""
        return {
            "world_setting": {
                "name": "San Francisco Bay Area",
                "type": "modern_metropolitan_region",
                "population": "7.7 million (101,000 per square mile in SF)",
                "established": "1776 (Mission), 1849 (Gold Rush), perpetually reinventing",
                "description": (
                    "The San Francisco Bay Area in 2025: where thousand-dollar phones order "
                    "$20 burritos while stepping over homeless encampments. Tech billionaires "
                    "meditate in Japanese gardens while their companies destroy privacy. "
                    "Progressive politics clash with capitalist extremes. The fog Karl rolls "
                    "in each evening, blurring the lines between what's real and what's "
                    "performed. Beneath the surface of kombucha bars and billion-dollar "
                    "valuations, older forms of power flow through new channels. They say "
                    "if you know where to look, you can find anything in San Francisco - "
                    "even submission from those who rule the world."
                ),
                "atmosphere": {
                    "physical": (
                        "Steep hills revealing sudden vistas, microclimates within blocks, "
                        "fog horns and cable car bells, earthquake retrofits and glass towers, "
                        "redwoods meeting ocean, bridges defining horizons"
                    ),
                    "social": (
                        "Extreme wealth beside extreme poverty, innovation and displacement, "
                        "diversity as brand and reality, everyone's from somewhere else, "
                        "performance of authenticity, seeking meaning in success"
                    ),
                    "hidden": (
                        "Power games in progressive clothing, submission needs among the "
                        "dominant, networks of women who understand control, roses that "
                        "bloom with thorns intact, the Queen who might be anyone"
                    )
                },
                "key_tensions": [
                    "Tech money vs. soul of the city",
                    "Progressive ideals vs. capitalist reality",
                    "Public personas vs. private needs",
                    "Visible power vs. hidden authority",
                    "Those who take vs. those who give properly"
                ]
            }
        }
    
    @staticmethod
    def get_districts() -> List[Dict[str, Any]]:
        """Get the main districts/locations"""
        return SFGeopoliticalLore.get_districts()
    
    @staticmethod
    def get_factions() -> List[Dict[str, Any]]:
        """Get the power factions in the story"""
        factions = SFGeopoliticalLore.get_factions()
        
        # Add some additional detail about how factions hide
        for faction in factions:
            if 'Rose' in faction['name'] and 'Thorn' in faction['name']:
                faction['name'] = 'The Shadow Network (various names)'
                faction['type'] = 'shadow_matriarchy'
                faction['description'] = (
                    "A powerful network of women operating in San Francisco's shadows. "
                    "Outsiders call it many names - The Rose & Thorn Society, The Garden, "
                    "The Thorn Network - but insiders know it has no formal name. What "
                    "everyone agrees on: it's led by someone called the Queen of Thorns, "
                    "and her reach extends wherever power needs checking. The Rose Council, "
                    "seven senior dominants, handles regional operations while the Queen "
                    "remains beautifully mysterious."
                )
                faction['naming_note'] = (
                    "The lack of official name is intentional - makes it harder to investigate "
                    "or infiltrate. Different cells might not even know they're part of the "
                    "same network."
                )
                
        return factions
    
    @staticmethod
    def get_specific_locations() -> List[Dict[str, Any]]:
        """Get story-specific detailed locations"""
        locations = SFLocalLore.get_specific_locations()
        
        # Enhance with operational details
        for loc in locations:
            if 'Rose Garden' in loc['name']:
                loc['operational_details'] = {
                    "staff_selection": "All employees personally vetted",
                    "customer_screening": "Subtle evaluation process",
                    "communication": "Orders that aren't on menu",
                    "back_room_access": "Invitation after trust established",
                    "cover_maintenance": "Legitimately excellent coffee"
                }
                
        return locations
    
    @staticmethod
    def get_urban_myths() -> List[Dict[str, Any]]:
        """Get SF-specific urban myths for the story"""
        return SFLocalLore.get_urban_myths()
    
    @staticmethod
    def get_religious_institutions() -> List[Dict[str, Any]]:
        """Get religious organizations in SF Bay Area"""
        return SFReligionLore.get_religious_institutions()
    
    @staticmethod
    def get_educational_systems() -> List[Dict[str, Any]]:
        """Get educational institutions for SF Bay Area"""
        return SFEducationLore.get_educational_systems()
    
    @staticmethod
    def get_complete_world_state() -> Dict[str, Any]:
        """Get a complete snapshot of the world state"""
        return {
            "setting": SFBayQueenOfThornsPreset.get_world_foundation(),
            "locations": {
                "districts": SFBayQueenOfThornsPreset.get_districts(),
                "specific": SFBayQueenOfThornsPreset.get_specific_locations(),
                "landmarks": SFLocalLore.get_landmarks()
            },
            "power_structures": {
                "factions": SFBayQueenOfThornsPreset.get_factions(),
                "relationships": SFGeopoliticalLore.get_faction_relationships(),
                "conflicts": SFGeopoliticalLore.get_conflicts()
            },
            "culture": {
                "myths": SFBayQueenOfThornsPreset.get_urban_myths(),
                "elements": SFLocalLore.get_cultural_elements(),
                "events": SFLocalLore.get_seasonal_events(),
                "underground": {
                    "communication": SFLocalLore.get_communication_networks(),
                    "economy": SFLocalLore.get_underground_economies(),
                    "etiquette": SFLocalLore.get_underground_etiquette()
                }
            },
            "institutions": {
                "religious": SFBayQueenOfThornsPreset.get_religious_institutions(),
                "educational": SFBayQueenOfThornsPreset.get_educational_systems(),
                "professional": SFEducationLore.get_professional_networks()
            },
            "knowledge": {
                "traditions": SFEducationLore.get_knowledge_traditions(),
                "languages": SFEducationLore.get_underground_languages(),
                "texts": SFReligionLore.get_sacred_texts()
            },
            "politics": {
                "issues": SFPoliticsLore.get_current_issues(),
                "events": SFPoliticsLore.get_political_events(),
                "media": SFPoliticsLore.get_media_landscape()
            }
        }
    
    @staticmethod
    def get_character_archetypes() -> List[Dict[str, Any]]:
        """Get character types that populate this world"""
        return [
            {
                "archetype": "The Tech Executive Submissive",
                "public_persona": "Disrupting industries, giving TED talks, ruling boardrooms",
                "private_truth": "Desperately needs to kneel to someone who understands real power",
                "common_journey": [
                    "Achieves everything, feels empty",
                    "Seeks meaning in extreme experiences",
                    "Finds workshop or therapist who sees deeper",
                    "Discovers submission as completion",
                    "Funds the network out of gratitude"
                ],
                "value_to_network": "Resources and connections"
            },
            {
                "archetype": "The Academic Gatekeeper",
                "public_persona": "Tenured professor in Gender Studies or Psychology",
                "private_truth": "Guardian of knowledge, identifier of potential",
                "responsibilities": [
                    "Teach theory with double meanings",
                    "Identify students ready for more",
                    "Provide academic legitimacy",
                    "Connect generations of practitioners"
                ],
                "typical_background": "Awakened during own graduate work"
            },
            {
                "archetype": "The Society Matron Dominant",
                "public_persona": "Philanthropist, museum board member, perfect hostess",
                "private_truth": "Decades of experience wielding hidden power",
                "methods": [
                    "Charity events as hunting grounds",
                    "Social connections as web of control",
                    "Mentoring next generation of dominants",
                    "Protecting the network's interests"
                ],
                "philosophy": "Power wielded silently lasts longest"
            },
            {
                "archetype": "The Healer Guide",
                "public_persona": "Therapist, yoga teacher, wellness coach",
                "private_truth": "Helps people discover their true nature",
                "specialties": [
                    "Trauma recovery through power exchange",
                    "Awakening dominant tendencies",
                    "Teaching ethical practice",
                    "Healing through controlled experiences"
                ],
                "ethical_stance": "Power must serve growth"
            },
            {
                "archetype": "The Professional Dominant",
                "public_persona": "Consultant, executive coach, entrepreneur",
                "private_truth": "Teaches dominance disguised as leadership",
                "client_types": [
                    "Women learning their power",
                    "Men learning to submit productively",
                    "Couples navigating power dynamics",
                    "Organizations needing hierarchy"
                ],
                "cover_business": "Perfectly legitimate coaching practice"
            },
            {
                "archetype": "The Broken Doll Warrior",
                "public_persona": "Social worker, victims' advocate, security consultant",
                "private_truth": "Survivor who became protector",
                "mission": [
                    "Identify those at risk",
                    "Extract victims from bad situations",
                    "Teach others to protect themselves",
                    "Sometimes darker justice"
                ],
                "network_role": "The Garden's thorns"
            },
            {
                "archetype": "The Innocent Initiate",
                "public_persona": "Graduate student, young professional, seeker",
                "private_truth": "Feeling the pull toward something unnamed",
                "typical_path": [
                    "Notices power dynamics everywhere",
                    "Finds mentors who see potential",
                    "Begins exploration carefully",
                    "Discovers true nature",
                    "Chooses role in network"
                ],
                "protection_needed": "Vulnerable to predators"
            }
        ]
    
    @staticmethod
    def get_initiation_paths() -> List[Dict[str, Any]]:
        """How people enter the hidden world"""
        return [
            {
                "path_name": "The Academic Route",
                "entry_point": "University gender studies or psychology program",
                "progression": [
                    "Takes class with certain professors",
                    "Shows unusual insight in discussions",
                    "Invited to advanced reading group",
                    "Introduced to practitioners",
                    "Begins practical education"
                ],
                "typical_duration": "2-4 years",
                "advantages": "Intellectual framework, peer support",
                "challenges": "Can remain too theoretical"
            },
            {
                "path_name": "The Therapy Journey",
                "entry_point": "Seeking help for relationships or trauma",
                "progression": [
                    "Finds therapist who sees deeper",
                    "Explores power dynamics in sessions",
                    "Recognizes patterns and desires",
                    "Guided to safe exploration",
                    "Integrated into community"
                ],
                "typical_duration": "1-3 years",
                "advantages": "Emotional support, ethical grounding",
                "challenges": "Dependent on therapist quality"
            },
            {
                "path_name": "The Professional Development",
                "entry_point": "Women's leadership workshop or executive coaching",
                "progression": [
                    "Attends public workshop",
                    "Responds to advanced concepts",
                    "Invited to intensive program",
                    "Discovers deeper teachings",
                    "Applies in real life"
                ],
                "typical_duration": "6 months - 2 years",
                "advantages": "Practical application, career integration",
                "challenges": "May focus on power over growth"
            },
            {
                "path_name": "The Social Introduction",
                "entry_point": "Meets someone at party, event, or through friends",
                "progression": [
                    "Notices something different about them",
                    "Careful conversations reveal alignment",
                    "Invited to private gathering",
                    "Observes and learns",
                    "Chooses involvement level"
                ],
                "typical_duration": "Variable",
                "advantages": "Personal connection, gradual exposure",
                "challenges": "Quality depends on introducer"
            },
            {
                "path_name": "The Crisis Awakening",
                "entry_point": "Life crisis forces examination",
                "progression": [
                    "Everything falls apart",
                    "Seeks new meaning",
                    "Finds unexpected resources",
                    "Discovers hidden strength",
                    "Rebuilds with new understanding"
                ],
                "typical_duration": "Intense and rapid",
                "advantages": "Deep motivation, quick transformation",
                "challenges": "Needs careful guidance"
            }
        ]
    
    @staticmethod
    def get_power_dynamics() -> Dict[str, Any]:
        """How power actually flows in this world"""
        return {
            "visible_power": {
                "tech_money": "Appears to run everything",
                "political_office": "Makes laws and speeches",
                "media_influence": "Shapes narrative",
                "academic_authority": "Defines truth"
            },
            "hidden_power": {
                "information_networks": "Who knows what about whom",
                "psychological_influence": "Ability to shape behavior",
                "sexual_dynamics": "Desires that create vulnerability",
                "protection_systems": "Safety in dangerous city",
                "Queen's_web": "Connections that transcend categories"
            },
            "power_exchange": {
                "public_form": "Mentorship, coaching, therapy",
                "private_reality": "Dominance and submission",
                "currency": "Trust, vulnerability, service",
                "returns": "Transformation, protection, purpose"
            },
            "leverage_points": [
                "Shame about desires",
                "Need for authentic connection",
                "Exhaustion from false power",
                "Hunger for meaning",
                "Fear of exposure"
            ]
        }
    
    @staticmethod
    def get_recognition_codes() -> Dict[str, Any]:
        """How the initiated recognize each other"""
        return {
            "visual_signals": {
                "jewelry": [
                    "Rose gold anything",
                    "Thorn motifs subtle",
                    "Vintage keys worn as pendants",
                    "Specific gemstones (garnet, obsidian)",
                    "Collar-like necklaces on vanilla occasions"
                ],
                "clothing": [
                    "Power colors in specific contexts",
                    "Leather accessories in corporate settings",
                    "Victorian influences modernized",
                    "Intentional covering/revealing"
                ],
                "body_language": [
                    "Posture that commands without aggression",
                    "Eye contact held precisely",
                    "Space taken or yielded consciously",
                    "Touch that asks permission subtly"
                ]
            },
            "verbal_codes": {
                "phrases": [
                    "'Interesting energy' about someone",
                    "'She has presence' (dominant)",
                    "'Very responsive' (submissive)",
                    "'Growth-oriented' (seeking)",
                    "'Experienced guide' (teacher)"
                ],
                "topics": [
                    "Power dynamics in literature",
                    "Consent philosophy",
                    "Energy work",
                    "Specific authors/thinkers",
                    "Garden metaphors"
                ],
                "responses": [
                    "Knowing pause before answering",
                    "Redirect vanilla questions",
                    "Layer meanings in replies",
                    "Test with subtle provocations"
                ]
            },
            "digital_markers": {
                "social_media": [
                    "Rose emoji patterns",
                    "Following specific accounts",
                    "Coded hashtags",
                    "Event attendance patterns"
                ],
                "professional": [
                    "LinkedIn endorsements from network",
                    "Volunteer work patterns",
                    "Conference attendance",
                    "Publication topics"
                ]
            },
            "behavioral_patterns": {
                "attendance": [
                    "Specific yoga studios",
                    "Certain therapists",
                    "Workshop series",
                    "Gallery openings",
                    "Charity events"
                ],
                "consumption": [
                    "Books ordered",
                    "Coffee shops frequented",
                    "Wine preferences",
                    "Art collected"
                ]
            }
        }
    
    @staticmethod
    def get_network_structure() -> Dict[str, Any]:
        """How the shadow network is organized"""
        return {
            "identity": {
                "formal_name": "None - the organization has no official name",
                "outsider_names": [
                    "The Rose & Thorn Society",
                    "The Thorn Garden", 
                    "The Rose Network",
                    "The Shadow Matriarchy",
                    "The Garden"
                ],
                "insider_reference": "Simply 'the network' or 'the garden'",
                "only_certainty": "Led by someone known as the Queen of Thorns",
                "mystery_level": "Even many members don't know if there's a formal name"
            },
            "leadership": {
                "The_Queen_of_Thorns": {
                    "nature": "Role/person/principle - deliberately unclear",
                    "known_facts": "Female, commands absolute loyalty, protects the vulnerable",
                    "unknown_facts": "Identity, selection process, true extent of power",
                    "succession": "Great mystery - unclear if role passes or person changes"
                },
                "The_Rose_Council": {
                    "composition": "Seven senior dominant women",
                    "selection": "Queen's choice with network approval", 
                    "responsibilities": "Regional coordination, major decisions, advisory role",
                    "knowledge_level": "Know more than most, but not everything",
                    "meeting_schedule": "Quarterly at changing locations"
                },
                "Regional_Thorns": {
                    "coverage": "Bay Area districts",
                    "duties": "Local network management",
                    "authority": "Handle all but major issues",
                    "reporting": "To Council quarterly"
                }
            },
            "geographic_scope": {
                "primary_territory": "San Francisco Bay Area - absolute control",
                "secondary_influence": "Major tech hubs - Seattle, Austin, NYC, LA",
                "mechanism": "Tech diaspora carries practices and connections",
                "international": "Allied networks, not direct subsidiaries",
                "growth_pattern": "Following tech money and cultural influence"
            },
            "membership_levels": {
                "Seedlings": {
                    "status": "Newly aware, exploring",
                    "access": "Public events, basic education",
                    "assessment": "Watched for potential",
                    "duration": "6 months to 2 years"
                },
                "Roses": {
                    "status": "Practicing, committed",
                    "access": "Private events, mentorship",
                    "responsibilities": "Support others, maintain standards",
                    "advancement": "Through service and skill"
                },
                "Thorns": {
                    "status": "Protectors and enforcers",
                    "access": "Security information, direct action",
                    "selection": "Proven loyalty and capability",
                    "duties": "Garden defense, problem solving"
                },
                "Gardeners": {
                    "status": "Teachers and guides",
                    "access": "Shape next generation",
                    "requirements": "Wisdom and experience",
                    "legacy": "Ensure continuity"
                }
            },
            "communication": {
                "routine": "Encrypted apps, coded social media",
                "urgent": "Phone trees, specific signals",
                "emergency": "Dead drops, safe houses",
                "ceremonial": "In-person at sacred times"
            },
            "resources": {
                "funding": [
                    "Member contributions",
                    "Business profits",
                    "Anonymous donations",
                    "Guilt payments from exposed"
                ],
                "properties": [
                    "Safe houses",
                    "Meeting venues",
                    "Business fronts",
                    "Retreat centers"
                ],
                "services": [
                    "Legal assistance",
                    "Medical care",
                    "Identity help",
                    "Protection details"
                ]
            }
        }
    
    @staticmethod
    def get_evolution_scenarios() -> List[Dict[str, Any]]:
        """How the world might develop"""
        return [
            {
                "scenario": "The Great Exposure",
                "trigger": "Major tech figure's involvement revealed",
                "progression": [
                    "Media frenzy about 'secret female supremacist cult'",
                    "Network goes deeper underground",
                    "Public witchhunt meets coordinated resistance",
                    "Narrative controlled through strategic leaks",
                    "Emerges stronger with myth magnified"
                ],
                "outcome_variations": [
                    "Mainstreaming of power exchange",
                    "Deeper entrenchment of secrecy",
                    "Schism between public and private factions"
                ]
            },
            {
                "scenario": "The Succession Crisis",
                "trigger": "Queen of Thorns retirement/disappearance",
                "factions": [
                    "Traditionalists wanting single Queen",
                    "Modernists proposing council rule",
                    "Radicals suggesting full democratization",
                    "Mystics believing Queen is eternal role"
                ],
                "resolution_paths": [
                    "Chosen successor emerges",
                    "Power sharing arrangement",
                    "Schism into multiple networks",
                    "Discovery Queen was always multiple"
                ]
            },
            {
                "scenario": "The Tech Integration",
                "description": "Silicon Valley fully discovers and adopts",
                "stages": [
                    "VCs fund 'power dynamics' startups",
                    "Apps for dominance training",
                    "Corporate programs normalized",
                    "Old guard resists commercialization",
                    "Two-tier system emerges"
                ],
                "challenges": [
                    "Maintaining authentic practice",
                    "Preventing exploitation",
                    "Preserving mystery",
                    "Protecting vulnerable"
                ]
            },
            {
                "scenario": "The Global Expansion",
                "description": "Network spreads beyond Bay Area",
                "mechanisms": [
                    "Tech diaspora carries practice",
                    "Online education spreads knowledge",
                    "Other cities develop chapters",
                    "International exchanges"
                ],
                "complications": [
                    "Cultural adaptation needs",
                    "Quality control issues",
                    "Communication security",
                    "Competing philosophies"
                ]
            }
        ]
    
    @staticmethod
    def apply_hidden_power_themes(content: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the hidden power dynamic themes to all content"""
        
        # Ensure all elements have both public face and hidden nature
        for category in ['myths', 'factions', 'locations', 'institutions']:
            if category in content:
                for item in content[category]:
                    if 'public_face' not in item:
                        item['public_face'] = item.get('description', '')[:100] + '...'
                    if 'hidden_nature' not in item:
                        item['hidden_nature'] = 'Deeper purposes known to initiates'
                    if 'recognition_signs' not in item:
                        item['recognition_signs'] = ['Subtle indicators', 'Known to those who know']
        
        # Add power dynamics to all relationships
        if 'relationships' in content:
            for rel in content['relationships']:
                rel['power_dynamics'] = 'Complex flows of influence and control'
                rel['hidden_exchanges'] = 'Services rendered in shadow'
        
        return content
    
    @staticmethod
    async def initialize_complete_sf_preset(ctx, user_id: int, conversation_id: int):
        """Initialize ALL lore components for SF Bay preset"""
        logger.info("Initializing Queen of Thorns SF Bay Area preset")
        
        # Get complete world state
        world_data = SFBayQueenOfThornsPreset.get_complete_world_state()
        
        # Apply hidden power themes
        world_data = SFBayQueenOfThornsPreset.apply_hidden_power_themes(world_data)
        
        # Initialize the base world
        from lore.core.lore_system import LoreSystem
        lore_system = await LoreSystem.get_instance(user_id, conversation_id)
        
        # Set the foundation
        environment_desc = world_data['setting']['world_setting']['description']
        await lore_system.generate_complete_lore(ctx, environment_desc)
        
        # Log what we're initializing
        logger.info(f"Initializing {len(world_data['locations']['districts'])} districts")
        logger.info(f"Initializing {len(world_data['power_structures']['factions'])} factions")
        logger.info(f"Initializing {len(world_data['institutions']['religious'])} religious institutions")
        logger.info(f"Initializing {len(world_data['institutions']['educational'])} educational institutions")
        
        return {
            "status": "success",
            "message": "Queen of Thorns SF Bay Area lore initialized",
            "stats": {
                "districts": len(world_data['locations']['districts']),
                "factions": len(world_data['power_structures']['factions']),
                "institutions": len(world_data['institutions']['religious']) + len(world_data['institutions']['educational']),
                "myths": len(world_data['culture']['myths']),
                "total_elements": sum(len(v) if isinstance(v, list) else 1 for v in world_data.values())
            }
        }


# Helper class for accessing lore
class QueenOfThornsLoreAccess:
    """Convenient access to Queen of Thorns lore elements"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.preset = SFBayQueenOfThornsPreset()
    
    def get_location_by_name(self, name: str) -> Dict[str, Any]:
        """Find a specific location by name"""
        all_locations = self.preset.get_specific_locations()
        for loc in all_locations:
            if name.lower() in loc['name'].lower():
                return loc
        return None
    
    def get_faction_by_name(self, name: str) -> Dict[str, Any]:
        """Find a specific faction by name"""
        all_factions = self.preset.get_factions()
        for faction in all_factions:
            if name.lower() in faction['name'].lower():
                return faction
        return None
    
    def get_random_entry_point(self) -> Dict[str, Any]:
        """Get a random way someone might discover the network"""
        import random
        paths = self.preset.get_initiation_paths()
        return random.choice(paths)
    
    def get_character_suggestions(self, archetype: str = None) -> List[Dict[str, Any]]:
        """Get character suggestions based on archetype"""
        all_archetypes = self.preset.get_character_archetypes()
        if archetype:
            return [a for a in all_archetypes if archetype.lower() in a['archetype'].lower()]
        return all_archetypes
    
    def check_recognition_code(self, detail: str) -> bool:
        """Check if a detail might be a recognition code"""
        codes = self.preset.get_recognition_codes()
        detail_lower = detail.lower()
        
        # Check all categories of codes
        for category in codes.values():
            if isinstance(category, dict):
                for code_list in category.values():
                    if isinstance(code_list, list):
                        for code in code_list:
                            if isinstance(code, str) and code.lower() in detail_lower:
                                return True
        return False


# Story-specific initializer
class EnhancedQueenOfThornsInitializer:
    """Enhanced initializer for the Queen of Thorns story"""
    
    @staticmethod
    async def initialize_with_sf_preset(ctx, user_id: int, conversation_id: int):
        """Initialize the story with full SF Bay Area preset"""
        
        # Initialize all lore
        result = await SFBayQueenOfThornsPreset.initialize_complete_sf_preset(
            ctx, user_id, conversation_id
        )
        
        # Add story-specific elements
        await EnhancedQueenOfThornsInitializer._add_story_hooks(ctx, user_id, conversation_id)
        
        logger.info("Queen of Thorns story initialized with full SF preset")
        
        return result
    
    @staticmethod
    async def _add_story_hooks(ctx, user_id: int, conversation_id: int):
        """Add specific story elements beyond base lore"""
        
        # Story-specific locations that need special attention
        key_locations = [
            {
                "name": "The Inner Garden",
                "description": (
                    "The Queen of Thorns' private sanctuary. Location unknown to all "
                    "but her inner circle. Said to be where she tends to both roses "
                    "and those who serve. Some say it's metaphorical, others have "
                    "seen the thorns."
                ),
                "access": "By Queen's invitation only",
                "rumors": [
                    "Hidden in plain sight",
                    "Moves with her needs",
                    "Contains every variety of rose",
                    "Where transformations complete"
                ]
            }
        ]
        
        # Key relationships to establish
        key_relationships = [
            {
                "character1": "The Queen of Thorns",
                "character2": "The Rose Council",
                "relationship": "Mysterious authority",
                "dynamic": "They serve but don't fully know her"
            },
            {
                "character1": "Tech Elite",
                "character2": "The Network",
                "relationship": "Mutual exploitation",
                "dynamic": "Money for silence, control for funding"
            }
        ]
        
        # Opening situation options
        opening_scenarios = [
            {
                "name": "The New Initiate",
                "description": "Player discovers the network exists",
                "entry_point": "Workshop, therapy, or social introduction",
                "first_challenge": "Proving worthy of deeper knowledge"
            },
            {
                "name": "The Crisis",
                "description": "Network under threat, player must help",
                "entry_point": "Already involved, called to action",
                "first_challenge": "Protecting identities while solving problem"
            },
            {
                "name": "The Investigation",
                "description": "Player investigating disappearances",
                "entry_point": "Detective, journalist, or concerned friend",
                "first_challenge": "Network is obstacle and solution"
            }
        ]
        
        logger.info("Story-specific hooks added")


# Main entry point for backward compatibility
SFBayMothFlamePreset = SFBayQueenOfThornsPreset  # Alias for compatibility

__all__ = [
    'SFBayQueenOfThornsPreset',
    'SFBayMothFlamePreset',  # For compatibility
    'QueenOfThornsLoreAccess',
    'EnhancedQueenOfThornsInitializer'
]
