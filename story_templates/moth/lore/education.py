# story_templates/moth/lore/education.py
"""
Educational systems and knowledge traditions for SF Bay Area
"""

from typing import Dict, Any, List

class SFEducationLore:
    """Educational and knowledge-related lore for SF Bay Area"""
    
    @staticmethod
    def get_educational_systems() -> List[Dict[str, Any]]:
        """Get educational institutions for SF Bay Area"""
        return [
            {
                "name": "The Conservatory of Shadows",
                "system_type": "alternative_education",
                "description": (
                    "Hidden beneath the California Academy of Sciences, this underground school "
                    "teaches the arts of dominance, submission, and power exchange. Run by former "
                    "Velvet Court members, it maintains the old traditions while preparing the next "
                    "generation. The Moth Queen occasionally guest lectures on consent and protection."
                ),
                "target_demographics": ["Adult practitioners", "Underground community", "Those seeking mastery"],
                "controlled_by": "The Velvet Court Educational Trust",
                "core_teachings": [
                    "Consent as sacred law",
                    "Power exchange dynamics",
                    "Safety and risk awareness",
                    "Underground history and traditions",
                    "Protection of the vulnerable"
                ],
                "teaching_methods": ["Mentorship", "Practical workshops", "Ritualized lessons"],
                "coming_of_age_rituals": "The First Mask Ceremony - earning your place in the underground",
                "knowledge_restrictions": "Outsiders forbidden, law enforcement especially",
                "female_leadership_roles": ["Headmistress", "Senior Dominatrixes", "Safety Wardens"],
                "male_roles": ["Assistant instructors", "Demonstration subjects", "Security"],
                "gender_specific_teachings": {
                    "female": ["Dominance arts", "Protective leadership", "Underground governance"],
                    "male": ["Service protocols", "Protective submission", "Support roles"]
                },
                "taboo_subjects": ["Trafficking methods", "Breaking consent", "Exposing the network"],
                "censorship_level": 8,
                "censorship_enforcement": "Immediate expulsion and blacklisting"
            },
            {
                "name": "St. Dominic's Reform School",
                "system_type": "religious_education",
                "description": (
                    "Catholic school in the Sunset District with a dark reputation. Officially for "
                    "'troubled' youth, it's known for producing either broken spirits or fierce rebels. "
                    "Several of the Moth Queen's rescued victims attended here. The rigid discipline "
                    "and shame-based teaching create perfect future submissives - or revolutionaries."
                ),
                "target_demographics": ["Troubled teens", "Court-mandated youth", "Parents' last resort"],
                "controlled_by": "Archdiocese of San Francisco",
                "core_teachings": [
                    "Strict Catholic doctrine",
                    "Corporal discipline",
                    "Shame and redemption",
                    "Gender role enforcement"
                ],
                "teaching_methods": ["Rote learning", "Physical punishment", "Isolation", "Prayer"],
                "coming_of_age_rituals": "Confirmation under duress",
                "knowledge_restrictions": "No questioning doctrine, no outside media",
                "connections_to_story": "Source of trauma for many underground members"
            },
            {
                "name": "Berkeley Underground Studies",
                "system_type": "informal_education",
                "description": (
                    "Anarchist collective near UC Berkeley teaching survival skills, legal rights, "
                    "and resistance tactics. Secretly funded by the Moth Queen, it helps at-risk "
                    "youth avoid trafficking. Classes held in rotating locations, communicated "
                    "through encrypted channels."
                ),
                "target_demographics": ["Runaways", "At-risk youth", "Sex workers", "Activists"],
                "controlled_by": "The Collective (anarchist structure)",
                "core_teachings": [
                    "Know your rights",
                    "Self-defense (physical and legal)",
                    "Recognizing predators",
                    "Safe communication",
                    "Mutual aid networks"
                ],
                "teaching_methods": ["Peer education", "Street workshops", "Practical exercises"],
                "connections_to_underground": "Direct pipeline to safehouse network"
            }
        ]

    @staticmethod
    def get_knowledge_traditions() -> List[Dict[str, Any]]:
        """Get knowledge transmission traditions"""
        return [
            {
                "name": "The Velvet Protocols",
                "tradition_type": "oral_tradition",
                "description": (
                    "The underground's unwritten rules passed down through mentorship. "
                    "Covers everything from scene negotiation to recognizing trafficking "
                    "signs. Each venue has a Protocol Keeper who trains newcomers. "
                    "Breaking protocol means exile from all underground spaces."
                ),
                "knowledge_domain": "Underground safety and ethics",
                "preservation_method": "Mentorship chains and practical demonstration",
                "access_requirements": "Vouching by established member, proven trustworthiness",
                "associated_group": "The Velvet Court",
                "examples": [
                    "The Three Taps (emergency signal)",
                    "Moth Sign (protection needed)",
                    "Red Night Rules (when cops raid)"
                ],
                "female_gatekeepers": True,
                "gendered_access": {
                    "female": "Full protocol training including leadership",
                    "male": "Service protocols and protection duties"
                },
                "matriarchal_reinforcement": "Female dominants as protocol authorities"
            },
            {
                "name": "Trauma Mapping",
                "tradition_type": "healing_practice",
                "description": (
                    "Body-based healing tradition developed by trafficking survivors. "
                    "Uses touch, movement, and power exchange to reclaim agency. The "
                    "Moth Queen learned this from the previous Queen. Practitioners "
                    "can 'read' trauma in body language and help release it."
                ),
                "knowledge_domain": "Somatic healing through kink",
                "preservation_method": "Direct body-to-body teaching",
                "access_requirements": "Must be survivor or proven ally",
                "associated_group": "The Fog Walkers",
                "female_gatekeepers": True,
                "matriarchal_reinforcement": "Female survivors as wisdom keepers"
            },
            {
                "name": "The Mask Makers' Art",
                "tradition_type": "craft_tradition",
                "description": (
                    "Creating personas and masks - both physical and psychological - "
                    "for protection and power. Master mask makers teach selected "
                    "students how to craft identities that can withstand scrutiny. "
                    "Used for both underground personas and new identities for rescued."
                ),
                "knowledge_domain": "Identity crafting and protection",
                "preservation_method": "Apprenticeship system",
                "access_requirements": "Artistic skill and absolute discretion",
                "examples": [
                    "Legal identity creation",
                    "Persona development",
                    "Physical mask crafting",
                    "Digital identity management"
                ]
            }
        ]
    
    @staticmethod
    def get_underground_languages() -> List[Dict[str, Any]]:
        """Get languages specific to the SF Bay underground culture"""
        return [
            {
                "name": "Moth Tongue",
                "language_family": "Underground Argot",
                "description": (
                    "A coded language developed in the SF underground, blending English with "
                    "BDSM terminology, Russian criminal argot, and Cantonese street slang. "
                    "Emphasizes feminine-dominant grammatical structures where authority "
                    "words default to feminine forms."
                ),
                "writing_system": "Modified English with emoji-based ideograms",
                "primary_regions": ["Mission Underground", "SoMa Shadow District"],
                "minority_regions": ["Tenderloin Shadows", "Financial District After Hours"],
                "formality_levels": [
                    "Street (casual threats and negotiations)",
                    "Sanctum (formal BDSM protocols)", 
                    "Court (addressing the Velvet Court)",
                    "Intimate (between trusted souls)"
                ],
                "common_phrases": {
                    "greeting": "Wings open (I come in peace)",
                    "farewell": "Until the next flame (never 'goodbye')",
                    "submission": "I fold my wings",
                    "dominance": "By my thorns",
                    "danger": "The collector walks",
                    "safety": "Fog's blessing"
                },
                "difficulty": 7,
                "relation_to_power": (
                    "Grammatically enforces female dominance. Male subjects require "
                    "diminutive markers. Authority verbs conjugate differently for women."
                ),
                "dialects": {
                    "Mission": "More Spanish influence, poetic",
                    "SoMa": "Tech jargon mixed in, clinical",
                    "Tenderloin": "Survival focused, terse"
                }
            },
            {
                "name": "Safehouse Sign",
                "language_family": "Gestural Language",
                "description": (
                    "Silent communication system used in the underground railroad network. "
                    "Based on ASL but modified for quick, discrete communication. Every "
                    "gesture can be disguised as casual movement."
                ),
                "writing_system": "None - purely gestural",
                "primary_regions": ["Marina Safehouse", "Butterfly House"],
                "minority_regions": ["All underground venues"],
                "formality_levels": [
                    "Emergency (rapid, urgent)",
                    "Operational (planning, coordinating)",
                    "Social (casual underground chat)"
                ],
                "common_phrases": {
                    "help_needed": "Touch ear, tap twice",
                    "danger_near": "Adjust collar, look down",
                    "safe_to_speak": "Hands flat on table",
                    "follow_me": "Touch moth jewelry",
                    "cops_coming": "Check phone, step back"
                },
                "difficulty": 5,
                "relation_to_power": "Egalitarian - safety transcends hierarchy",
                "dialects": {
                    "Protector": "More aggressive, combat-ready",
                    "Survivor": "Focuses on escape and hiding"
                }
            },
            {
                "name": "Velvet Protocols",
                "language_family": "Ritual Language",
                "description": (
                    "The formal language of power exchange, used in high protocol BDSM "
                    "scenes. Every word is chosen for impact. Silence is part of the language. "
                    "The Moth Queen is considered the highest authority on proper usage."
                ),
                "writing_system": "Calligraphy for contracts, otherwise spoken",
                "primary_regions": ["Velvet Sanctum", "Folsom Sanctuary"],
                "minority_regions": ["Private dungeons throughout the city"],
                "formality_levels": [
                    "Training (teaching beginners)",
                    "Scene (active power exchange)",
                    "High Protocol (formal ceremonies)",
                    "Sacred (the deepest submissions)"
                ],
                "common_phrases": {
                    "consent": "I offer myself willingly",
                    "safeword": "Mercy" / "Red" / "Moth",
                    "praise": "You please me greatly",
                    "correction": "You will learn better",
                    "devotion": "I am yours to command"
                },
                "difficulty": 8,
                "relation_to_power": (
                    "Explicitly hierarchical. Dominants speak in imperatives, "
                    "submissives in requests. The Queen's word is absolute law."
                ),
                "dialects": {
                    "Old Guard": "Traditional, more formal",
                    "New School": "More egalitarian, negotiated"
                }
            }
        ]
    
    @staticmethod
    def get_language_evolution() -> List[Dict[str, Any]]:
        """How underground languages evolve"""
        return [
            {
                "language": "Moth Tongue",
                "evolution_stage": "digital_integration",
                "new_elements": {
                    "emoji_grammar": {
                        "🦋": "safety/transformation",
                        "🕯️": "sanctuary available",
                        "🌫️": "danger approaching",
                        "👑": "Queen's protection active",
                        "🔴": "emergency extraction needed"
                    },
                    "code_switching": {
                        "public_face": "Discussing 'butterfly gardens' or 'moth collecting'",
                        "true_meaning": "Safehouse locations and victim status",
                        "example": "'Found rare moth species in Richmond' = 'Victim rescued in Richmond'"
                    },
                    "generational_differences": {
                        "old_guard": "Physical signals and spoken codes",
                        "new_generation": "Encrypted apps and digital markers",
                        "bridge_speakers": "Those fluent in both"
                    }
                }
            }
        ]
