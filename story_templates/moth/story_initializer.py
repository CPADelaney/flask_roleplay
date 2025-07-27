# story_templates/moth/story_initializer.py
"""
Complete story initialization system for The Moth and Flame
Handles NPC creation, location setup, and special mechanics
"""

import logging
import json
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

from npcs.new_npc_creation import NPCCreationHandler
from db.connection import get_db_connection_context
from story_templates.moth.poem_integrated_loader import PoemIntegratedStoryLoader
from story_templates.character_profiles.lilith_ravencroft import LILITH_RAVENCROFT
from lore.core import canon
from memory.wrapper import MemorySystem
from story_templates.preset_stories import StoryBeat, PresetStory
from nyx.integrate import remember_with_governance

logger = logging.getLogger(__name__)

class MothFlameStoryInitializer:
    """Complete initialization system for The Moth and Flame story"""
    
    @staticmethod
    async def initialize_story(ctx, user_id: int, conversation_id: int) -> Dict[str, Any]:
        """
        Initialize the complete story with all components.
        
        Args:
            ctx: Context object with user_id and conversation_id
            user_id: User ID
            conversation_id: Conversation ID
            
        Returns:
            Dict with initialization results
        """
        try:
            logger.info(f"Initializing The Moth and Flame story for user {user_id}")
            
            # Step 1: Load story structure and poems
            from story_templates.moth.the_moth_and_flame import THE_MOTH_AND_FLAME
            await PoemIntegratedStoryLoader.load_story_with_poems(
                THE_MOTH_AND_FLAME, user_id, conversation_id
            )
            logger.info("Story structure and poems loaded")
            
            # Step 2: Create all locations
            location_ids = await MothFlameStoryInitializer._create_all_locations(
                ctx, user_id, conversation_id
            )
            logger.info(f"Created {len(location_ids)} locations")
            
            # Step 3: Create Lilith Ravencroft
            lilith_id = await MothFlameStoryInitializer._create_lilith_ravencroft(
                ctx, user_id, conversation_id
            )
            logger.info(f"Created Lilith Ravencroft with ID: {lilith_id}")
            
            # Step 4: Create supporting NPCs
            support_npc_ids = await MothFlameStoryInitializer._create_supporting_npcs(
                ctx, user_id, conversation_id
            )
            logger.info(f"Created {len(support_npc_ids)} supporting NPCs")
            
            # Step 5: Establish relationships
            await MothFlameStoryInitializer._setup_all_relationships(
                ctx, user_id, conversation_id, lilith_id, support_npc_ids
            )
            logger.info("Relationships established")
            
            # Step 6: Initialize story state and tracking
            await MothFlameStoryInitializer._initialize_story_state(
                ctx, user_id, conversation_id, lilith_id
            )
            logger.info("Story state initialized")
            
            # Step 7: Set up special mechanics
            await MothFlameStoryInitializer._setup_special_mechanics(
                ctx, user_id, conversation_id, lilith_id
            )
            logger.info("Special mechanics configured")
            
            # Step 8: Create initial atmosphere
            await MothFlameStoryInitializer._set_initial_atmosphere(
                ctx, user_id, conversation_id
            )
            logger.info("Initial atmosphere set")
            
            return {
                "status": "success",
                "story_id": THE_MOTH_AND_FLAME.id,
                "main_npc_id": lilith_id,
                "support_npc_ids": support_npc_ids,
                "location_ids": location_ids,
                "message": "The Moth and Flame story initialized successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize story: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "message": "Failed to initialize The Moth and Flame story"
            }
    
    @staticmethod
    async def _create_lilith_ravencroft(ctx, user_id: int, conversation_id: int) -> int:
        """Create Lilith with all her complexity"""
        
        try:
            npc_handler = NPCCreationHandler()
            
            # Prepare complete NPC data from character profile
            lilith_data = LILITH_RAVENCROFT.copy()
            
            # Build comprehensive physical description
            physical_desc = MothFlameStoryInitializer._build_comprehensive_physical_description(
                lilith_data["physical_description"]
            )
            
            # Create initial memories that establish her character
            initial_memories = MothFlameStoryInitializer._create_lilith_memories()
            
            # Create the NPC through the handler
            result = await npc_handler.create_npc_in_database(ctx, {
                "npc_name": lilith_data["name"],
                "sex": "female",
                "age": 32,
                "physical_description": physical_desc,
                "personality": {
                    "personality_traits": lilith_data["traits"],
                    "likes": lilith_data["personality"]["likes"],
                    "dislikes": lilith_data["personality"]["dislikes"],
                    "hobbies": lilith_data["personality"]["hobbies"]
                },
                "stats": lilith_data["stats"],
                "archetypes": {
                    "archetype_names": [lilith_data["archetype"]],
                    "archetype_summary": f"{lilith_data['role']}. {lilith_data['backstory']['history']}",
                    "archetype_extras_summary": lilith_data["backstory"]["the_transformation"]
                },
                "schedule": lilith_data["schedule"],
                "memories": initial_memories,
                "current_location": "Velvet Sanctum - Preparation Chamber",
                "affiliations": ["Velvet Sanctum", "The Underground Network", "The Moth Queen Identity"],
                "introduced": False  # Player hasn't met her yet
            })
            
            if "error" in result:
                raise Exception(f"Failed to create Lilith: {result['error']}")
            
            npc_id = result["npc_id"]
            
            # Add special properties through canonical updates
            await MothFlameStoryInitializer._add_lilith_special_properties(
                ctx, npc_id, lilith_data, user_id, conversation_id
            )
            
            # Set up her memory system with special focus
            await MothFlameStoryInitializer._initialize_lilith_memory_system(
                user_id, conversation_id, npc_id, lilith_data
            )
            
            return npc_id
            
        except Exception as e:
            logger.error(f"Failed to create Lilith Ravencroft: {e}", exc_info=True)
            raise
    
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
        """Create Lilith's foundational memories"""
        
        return [
            # Trauma and survival
            "I was fifteen when they tried to make me disappear. The van, the men, the promises of modeling work. "
            "But I had already learned that pretty faces hide sharp teeth. I bit, I clawed, I burned their operation "
            "down from the inside. The scars on my wrists aren't from giving up - they're from breaking free.",
            
            # Building the Sanctum
            "The day I signed the lease for what would become the Velvet Sanctum, the realtor said 'It's very dark "
            "down here.' I smiled and replied, 'Perfect for what I have in mind.' That night I stood in the empty "
            "space and whispered to the shadows: 'This will be my temple, where pain becomes prayer.'",
            
            # The dual identity birth
            "Three years ago, I found Maya bleeding in an alley, same age I was when they tried to take me. As I "
            "held her, I realized my power could be more than performance. That night, the Moth Queen was born - "
            "not just a dominatrix, but a protector. Now I rule one world and guard another.",
            
            # First abandonment
            "Alexandra swore she'd never leave. 'You're my gravity,' she said, kneeling so beautifully. Six months "
            "later, I found her engagement announcement in the society pages. I added her porcelain mask to my "
            "collection and her name to the blue list. Another ghost, another lie.",
            
            # The unspoken words
            "Last month, someone almost made me say them - those three words that taste of burning stars. I bit "
            "my tongue until it bled rather than let them escape. Love is a luxury I can't afford. Everyone who "
            "claims to love me disappears. Better to rule through fear than lose through love.",
            
            # A moment of unexpected tenderness
            "Sometimes after the sanctum empties, I sit on my throne and practice. 'I love you,' I whisper to the "
            "darkness. The words feel foreign, like speaking in tongues. How can a moth love the flame that will "
            "consume it? How can a flame love what it must destroy?",
            
            # The lists
            "Red ink for those I failed to save. Blue ink for those who failed to stay. Tonight I added two names: "
            "one red (a girl who didn't make it out), one blue (a submissive who promised forever). The blue list "
            "is longer. It always is. Sometimes I think heartbreak is worse than death.",
            
            # Her greatest fear
            "My deepest terror isn't pain or death - it's the moment someone sees all of me and chooses to leave "
            "anyway. When they know about the lists, the masks, the broken girl beneath the goddess, and they "
            "still walk away. That's why I never remove all the masks. Always keep one layer of protection."
        ]
    
    @staticmethod
    async def _add_lilith_special_properties(
        ctx, npc_id: int, lilith_data: Dict, user_id: int, conversation_id: int
    ):
        """Add Lilith's unique properties through canonical updates"""
        
        canon_ctx = type('CanonicalContext', (), {
            'user_id': user_id,
            'conversation_id': conversation_id
        })()
        
        async with get_db_connection_context() as conn:
            # Add dialogue patterns
            await canon.update_entity_canonically(
                canon_ctx, conn, "NPCStats", npc_id,
                {
                    "dialogue_patterns": json.dumps(lilith_data["dialogue_patterns"]),
                    "dialogue_style": "poetic_gothic"
                },
                "Adding Lilith's complex dialogue patterns"
            )
            
            # Add trauma triggers and responses
            await canon.update_entity_canonically(
                canon_ctx, conn, "NPCStats", npc_id,
                {
                    "trauma_triggers": json.dumps(lilith_data["trauma_triggers"]),
                    "trauma_responses": json.dumps({
                        "abandonment": "becomes coldly cruel",
                        "unexpected_touch": "violence or freeze",
                        "broken_promises": "immediate emotional shutdown",
                        "bright_lights": "anxiety and mask adjustment"
                    })
                },
                "Setting up Lilith's trauma responses"
            )
            
            # Add relationship mechanics
            await canon.update_entity_canonically(
                canon_ctx, conn, "NPCStats", npc_id,
                {
                    "relationship_mechanics": json.dumps(lilith_data["relationship_mechanics"]),
                    "trust_thresholds": json.dumps({
                        "first_mask_removal": 50,
                        "see_private_chambers": 60,
                        "learn_dual_identity": 70,
                        "witness_vulnerability": 80,
                        "hear_three_words": 95
                    })
                },
                "Establishing Lilith's relationship progression system"
            )
            
            # Add memory priorities
            await canon.update_entity_canonically(
                canon_ctx, conn, "NPCStats", npc_id,
                {
                    "memory_priorities": json.dumps(lilith_data["memory_priorities"]),
                    "memory_focus": "abandonment_and_devotion"
                },
                "Setting Lilith's memory focus areas"
            )
            
            # Add secrets and hidden information
            await canon.update_entity_canonically(
                canon_ctx, conn, "NPCStats", npc_id,
                {
                    "secrets": json.dumps({
                        "deepest_secret": lilith_data["backstory"]["deepest_secret"],
                        "the_list": lilith_data["backstory"]["the_list"],
                        "the_three_words": "Words that live beneath her tongue, tasting of burning stars",
                        "mask_room": "A hidden room with masks of everyone who left",
                        "real_name": "She wasn't always Lilith - that name was chosen, not given"
                    }),
                    "hidden_stats": json.dumps({
                        "vulnerability": lilith_data["stats"]["vulnerability"],
                        "abandonment_fear": 95,
                        "true_affection": 85,
                        "self_loathing": 60
                    })
                },
                "Adding Lilith's deepest secrets and hidden nature"
            )
            
            # Add special mechanics flags
            await canon.update_entity_canonically(
                canon_ctx, conn, "NPCStats", npc_id,
                {
                    "special_mechanics": json.dumps(lilith_data["special_mechanics"]),
                    "current_mask": "Porcelain Goddess",
                    "moth_flame_dynamic": "unestablished",
                    "three_words_spoken": False,
                    "dual_identity_revealed": False
                },
                "Setting up Lilith's special story mechanics"
            )
            
            # Add evolution paths
            await canon.update_entity_canonically(
                canon_ctx, conn, "NPCStats", npc_id,
                {
                    "evolution_paths": json.dumps(lilith_data["narrative_evolution"]),
                    "current_evolution_stage": "The Masked Goddess",
                    "evolution_triggers_met": json.dumps([])
                },
                "Establishing Lilith's narrative evolution paths"
            )
    
    @staticmethod
    async def _initialize_lilith_memory_system(
        user_id: int, conversation_id: int, npc_id: int, lilith_data: Dict
    ):
        """Set up Lilith's specialized memory system"""
        
        memory_system = await MemorySystem.get_instance(user_id, conversation_id)
        
        # Create memory schemas specific to her character
        schemas = [
            {
                "name": "Abandonment Patterns",
                "description": "Tracking promises made and broken",
                "category": "trauma",
                "attributes": {
                    "promise_type": "unknown",
                    "promise_keeper": "unknown", 
                    "time_until_broken": "unknown",
                    "her_response": "unknown"
                }
            },
            {
                "name": "Devotion Displays",
                "description": "How others show their submission",
                "category": "relationship",
                "attributes": {
                    "devotion_type": "unknown",
                    "sincerity_assessment": "unknown",
                    "her_satisfaction": "unknown",
                    "trust_impact": "unknown"
                }
            },
            {
                "name": "Mask Moments",
                "description": "When and why masks slip or are removed",
                "category": "vulnerability",
                "attributes": {
                    "mask_type": "unknown",
                    "trigger": "unknown",
                    "witness": "unknown",
                    "aftermath": "unknown"
                }
            },
            {
                "name": "Underground Work",
                "description": "Her secret life as protector",
                "category": "hidden_identity",
                "attributes": {
                    "operation_type": "unknown",
                    "lives_saved": 0,
                    "lives_lost": 0,
                    "emotional_cost": "unknown"
                }
            }
        ]
        
        for schema in schemas:
            await memory_system.schema_manager.create_schema(
                entity_type="npc",
                entity_id=npc_id,
                schema_name=schema["name"],
                description=schema["description"],
                category=schema["category"],
                attributes=schema["attributes"]
            )
        
        # Set up trauma keywords for flashback system
        trauma_keywords = [
            "disappear", "goodbye", "leave", "forever", "always",
            "promise", "trafficking", "van", "fifteen", "scars"
        ]
        
        # Store these as flashback triggers
        async with get_db_connection_context() as conn:
            await conn.execute(
                """
                UPDATE NPCStats 
                SET flashback_triggers = $1
                WHERE user_id = $2 AND conversation_id = $3 AND npc_id = $4
                """,
                json.dumps(trauma_keywords),
                user_id, conversation_id, npc_id
            )
    
    @staticmethod
    async def _create_supporting_npcs(ctx, user_id: int, conversation_id: int) -> List[int]:
        """Create all supporting NPCs for the story"""
        
        npc_handler = NPCCreationHandler()
        npc_ids = []
        
        # Marcus Sterling - The Devoted Pilgrim
        marcus_result = await npc_handler.create_npc_in_database(ctx, {
            "npc_name": "Marcus Sterling",
            "sex": "male",
            "age": 45,
            "physical_description": (
                "A once-powerful businessman now wholly devoted to his Queen. Expensive suits "
                "can't hide the collar marks on his neck or the desperate hunger in his eyes. "
                "Silver hair, always perfectly styled, as if maintaining his appearance might "
                "earn him an extra moment of her attention. Kneels with practiced grace."
            ),
            "personality": {
                "personality_traits": [
                    "obsessively_devoted", "jealous", "broken", "wealthy",
                    "desperately_needy", "completely_submissive", "worship_focused"
                ],
                "likes": [
                    "serving the Queen", "public humiliation", "being used as example",
                    "buying gifts for her", "kneeling", "being ignored (it's still attention)"
                ],
                "dislikes": [
                    "new submissives", "being forgotten", "others getting attention",
                    "leaving the sanctum", "his old life", "independent thought"
                ],
                "hobbies": [
                    "collecting Queen's used items", "writing devotional poetry",
                    "practicing perfect service", "studying her preferences"
                ]
            },
            "stats": {
                "dominance": 5,
                "cruelty": 10,
                "affection": 95,
                "trust": 100,
                "respect": 20,
                "intensity": 80
            },
            "archetypes": {
                "archetype_names": ["Broken Executive", "Devoted Submissive"],
                "archetype_summary": "A powerful man reduced to a worshipful pet",
                "archetype_extras_summary": "Represents the complete dissolution of self in service"
            },
            "schedule": {
                "Monday": {"Evening": "Waiting outside Velvet Sanctum", "Night": "Kneeling in main chamber"},
                "Tuesday": {"All Day": "Preparing offerings for the Queen"},
                "Wednesday": {"Evening": "Early arrival at Sanctum", "Night": "Public display"},
                "Thursday": {"Evening": "Private session if permitted", "Night": "Cleaning duties"},
                "Friday": {"Evening": "First to arrive", "Night": "Demonstration subject"},
                "Saturday": {"All Day": "Living at the Sanctum's edges"},
                "Sunday": {"All Day": "Writing poetry and waiting"}
            },
            "memories": [
                "The first time She noticed me, I was just another suit with a fetish. But when She "
                "looked into my eyes, She saw the hollow man I truly was. 'You're already empty,' "
                "She said. 'Let me fill you with purpose.' I've been Hers ever since.",
                
                "I gave up my CEO position, my marriage, my identity - all for the privilege of "
                "kneeling at Her feet. My ex-wife thinks I'm insane. She doesn't understand that "
                "I've never been saner. I exist to serve. That is sanity."
            ],
            "current_location": "Outside Velvet Sanctum",
            "affiliations": ["Velvet Sanctum", "The Queen's Inner Circle"],
            "introduced": False
        })
        
        if "error" not in marcus_result:
            npc_ids.append(marcus_result["npc_id"])
        
        # Sarah Chen - Rescued Trafficking Victim
        sarah_result = await npc_handler.create_npc_in_database(ctx, {
            "npc_name": "Sarah Chen",
            "sex": "female",
            "age": 22,
            "physical_description": (
                "A young woman still healing from trauma. Asian features marked by a wariness that "
                "never quite leaves her eyes. Thin from years of deprivation, slowly learning to "
                "take up space again. Dresses in layers, always ready to run. A moth tattoo on "
                "her wrist - the mark of those saved by the Moth Queen."
            ),
            "personality": {
                "personality_traits": [
                    "traumatized", "grateful", "suspicious", "healing",
                    "protective_of_others", "alert", "slowly_trusting"
                ],
                "likes": [
                    "feeling safe", "helping other survivors", "quiet spaces",
                    "the Moth Queen's protection", "learning self-defense", "tea"
                ],
                "dislikes": [
                    "sudden movements", "locked doors", "vans", "older men",
                    "being touched without warning", "loud voices", "feeling trapped"
                ],
                "hobbies": [
                    "counseling other survivors", "self-defense training",
                    "gardening in the safehouse", "journaling her recovery"
                ]
            },
            "stats": {
                "dominance": 20,
                "cruelty": 5,
                "affection": 70,
                "trust": 30,
                "respect": 95,
                "intensity": 60
            },
            "archetypes": {
                "archetype_names": ["Trafficking Survivor", "Hidden Strength"],
                "archetype_summary": "A survivor learning to reclaim her power",
                "archetype_extras_summary": "Represents those Lilith saves and why she fights"
            },
            "schedule": {
                "Monday": {"Morning": "Safehouse", "Afternoon": "Therapy", "Evening": "Helping newcomers"},
                "Tuesday": {"Morning": "Self-defense class", "Afternoon": "Safehouse garden"},
                "Wednesday": {"All Day": "Counseling other survivors"},
                "Thursday": {"Morning": "Job training", "Afternoon": "Safehouse"},
                "Friday": {"Evening": "Support group meeting"},
                "Saturday": {"Varies": "Helping with rescue operations"},
                "Sunday": {"All Day": "Rest and recovery"}
            },
            "memories": [
                "I was seventeen when they took me. Promised a waitressing job in the city. The "
                "Moth Queen found me three years later, half-dead in a basement. She burned their "
                "whole operation down. 'No one else,' she whispered as she carried me out. 'No one else.'",
                
                "Sometimes I see her at the safehouse, checking on us. She's different there - no "
                "masks, no performance. Just a woman who understands our pain because she lived it. "
                "She taught me that surviving isn't enough. We deserve to live."
            ],
            "current_location": "Safehouse - Common Area",
            "affiliations": ["The Underground Network", "Moth Queen's Saved"],
            "introduced": False
        })
        
        if "error" not in sarah_result:
            npc_ids.append(sarah_result["npc_id"])
        
        # Viktor Kozlov - Trafficking Ring Enforcer
        viktor_result = await npc_handler.create_npc_in_database(ctx, {
            "npc_name": "Viktor Kozlov",
            "sex": "male", 
            "age": 48,
            "physical_description": (
                "A mountain of barely restrained violence. Russian accent thick as his scarred "
                "knuckles. Prison tattoos peek from under expensive shirts that can't hide what "
                "he is. Dead eyes that see women as commodities. Smells of cologne and cruelty."
            ),
            "personality": {
                "personality_traits": [
                    "violent", "calculating", "misogynistic", "predatory",
                    "intelligent", "ruthless", "well_connected"
                ],
                "likes": [
                    "power over others", "breaking the strong", "money",
                    "fear in others' eyes", "the trafficking trade", "violence"
                ],
                "dislikes": [
                    "the Moth Queen", "losing merchandise", "police attention",
                    "women with power", "being challenged", "witnesses"
                ],
                "hobbies": [
                    "expanding his network", "intimidation", "counting profits",
                    "planning the Moth Queen's downfall"
                ]
            },
            "stats": {
                "dominance": 90,
                "cruelty": 95,
                "affection": 0,
                "trust": 0,
                "respect": -50,
                "intensity": 85
            },
            "archetypes": {
                "archetype_names": ["Human Trafficker", "Dangerous Predator"],
                "archetype_summary": "The evil that Lilith fights against",
                "archetype_extras_summary": "Represents the darkness she escaped and battles"
            },
            "schedule": {
                "Monday": {"Night": "Underground clubs hunting"},
                "Tuesday": {"Night": "Moving 'merchandise'"},
                "Wednesday": {"Evening": "Meeting with corrupt officials"},
                "Thursday": {"Night": "Checking on operations"},
                "Friday": {"Night": "High-end clubs", "Late Night": "Violence"},
                "Saturday": {"Night": "Expanding territory"},
                "Sunday": {"Unknown": "Planning and counting money"}
            },
            "memories": [
                "The Moth Queen cost me three million dollars when she burned down my warehouse. "
                "But worse, she gave the merchandise hope. Hope is bad for business. Soon I will "
                "clip those moth wings and remind her what happens to little girls who forget their place.",
                
                "I remember her from before - just another scared teenager in my van. Should have "
                "killed her when she bit Dmitri. Now she plays queen and steals from me. But "
                "every queen falls. And when she does, I'll be there to collect what's mine."
            ],
            "current_location": "Unknown - Hunting",
            "affiliations": ["Eastern European Crime Syndicate", "The Shadow Trade"],
            "introduced": False
        })
        
        if "error" not in viktor_result:
            npc_ids.append(viktor_result["npc_id"])
        
        # Create a few minor NPCs for atmosphere
        minor_npcs = [
            {
                "name": "Jessica Vale",
                "role": "Sanctum Regular",
                "description": "A lawyer by day who seeks absolution in submission"
            },
            {
                "name": "Amanda Ross", 
                "role": "New Submissive",
                "description": "Curious and eager, hasn't learned the rules yet"
            },
            {
                "name": "Diana Moon",
                "role": "Former Favorite",
                "description": "Once held the Queen's attention, now watches from the shadows"
            }
        ]
        
        for minor in minor_npcs:
            minor_result = await npc_handler.create_npc_in_database(ctx, {
                "npc_name": minor["name"],
                "sex": "female",
                "physical_description": minor["description"],
                "personality": {
                    "personality_traits": ["submissive", "devoted", "seeking"],
                    "likes": ["the Queen", "belonging", "structure"],
                    "dislikes": ["being ignored", "vanilla life"],
                    "hobbies": ["attending the Sanctum"]
                },
                "stats": {
                    "dominance": 20,
                    "cruelty": 10,
                    "affection": 60,
                    "trust": 40,
                    "respect": 70,
                    "intensity": 50
                },
                "archetypes": {
                    "archetype_names": ["Sanctum Regular"],
                    "archetype_summary": minor["role"],
                    "archetype_extras_summary": "Part of the Queen's court"
                },
                "current_location": "Velvet Sanctum",
                "introduced": False
            })
            
            if "error" not in minor_result:
                npc_ids.append(minor_result["npc_id"])
        
        return npc_ids
    
    @staticmethod
    async def _create_all_locations(ctx, user_id: int, conversation_id: int) -> List[str]:
        """Create all story locations"""
        
        canon_ctx = type('CanonicalContext', (), {
            'user_id': user_id,
            'conversation_id': conversation_id
        })()
        
        location_ids = []
        
        locations = [
            {
                "name": "Velvet Sanctum",
                "description": (
                    "An underground temple where pain becomes prayer. Descending the stairs from the "
                    "innocent boutique above, the air grows thick with incense and anticipation. "
                    "Candles gutter in silver stands, casting dancing shadows on velvet-draped walls. "
                    "The main chamber centers on an obsidian throne where the Queen holds court. "
                    "Private booths line the edges for intimate worship, while deeper still lies "
                    "the dungeon where true devotion is tested. Every surface whispers of power and "
                    "submission, every shadow holds a secret."
                ),
                "location_type": "nightclub_dungeon",
                "areas": [
                    "Main Chamber", "Throne Room", "Private Booths", 
                    "The Dungeon", "Preparation Chamber", "Queen's Private Office"
                ],
                "open_hours": {"Monday": "8PM-3AM", "Tuesday": "Private Only", 
                              "Wednesday": "8PM-4AM", "Thursday": "8PM-3AM",
                              "Friday": "8PM-5AM", "Saturday": "8PM-5AM", "Sunday": "Closed"}
            },
            {
                "name": "Empty Sanctum",
                "description": (
                    "The same space when the music dies and shadows lengthen. Without the crowds "
                    "and performance, it becomes a melancholy cathedral. Candles burn low, their "
                    "wax pooling like frozen tears. The throne sits empty, a monument to loneliness. "
                    "Here, masks grow heavy and the goddess becomes mortal again."
                ),
                "location_type": "afterhours_venue",
                "areas": ["Abandoned Stage", "Silent Throne", "Echo Chamber"],
                "open_hours": {"Always": "3AM-8AM when empty"}
            },
            {
                "name": "Private Chambers",
                "description": (
                    "Behind a hidden door in the Sanctum lies her true sanctuary. A apartment that "
                    "tells two stories: the public areas draped in luxury and control, the private "
                    "spaces revealing vulnerability. The mask room holds her collection - porcelain "
                    "faces of everyone who promised to stay. Her desk overflows with letters never "
                    "sent. Moths dance against the windows, drawn to their beautiful destruction."
                ),
                "location_type": "personal_space",
                "areas": [
                    "The Mask Room", "Writing Desk", "Bedroom",
                    "Hidden Room", "Private Bath", "Balcony Overlooking the City"
                ],
                "open_hours": {"Private": "By invitation only"}
            },
            {
                "name": "Safehouse Network - Entry Point",
                "description": (
                    "A flower shop by day, sanctuary by night. Behind the cooler of roses lies a "
                    "hidden door. Those who know the phrase 'moths seek light' find safety beyond. "
                    "The Moth Queen built this network for those like she once was - hunted, "
                    "trafficked, disposable. Here, broken wings learn to fly again."
                ),
                "location_type": "secret_location",
                "areas": [
                    "Front Shop", "Hidden Entrance", "Safe Room",
                    "Medical Station", "Planning Room", "Escape Tunnel"
                ],
                "open_hours": {"Public": "9AM-6PM", "Safehouse": "Always"}
            },
            {
                "name": "Underground District",
                "description": (
                    "The part of the city that exists in perpetual night. Neon signs reflect in "
                    "puddles of questionable origin. Here, desires too dark for daylight find "
                    "their market. The Velvet Sanctum is its crown jewel, but darker things lurk "
                    "in the alleys where the Moth Queen wages her secret war."
                ),
                "location_type": "district",
                "areas": [
                    "Neon Alley", "The Black Market", "Underground Clubs",
                    "Shadow Corners", "Trafficking Routes", "Safe Passages"
                ],
                "open_hours": {"Always": "More active at night"}
            },
            {
                "name": "Abandoned Warehouse",
                "description": (
                    "Once Viktor's operation center, now a burned ruin. The Moth Queen's first "
                    "major strike against the trafficking ring. Scorch marks tell the story of "
                    "her fury. Sometimes she returns here to remember why she fights."
                ),
                "location_type": "abandoned",
                "areas": ["Burned Remains", "Memorial Corner", "Evidence of Victory"],
                "open_hours": {"Always": "Abandoned"}
            }
        ]
        
        async with get_db_connection_context() as conn:
            for loc in locations:
                # Create location canonically
                location_id = await canon.find_or_create_location(
                    canon_ctx, conn, 
                    loc["name"],
                    description=loc.get("description"),
                    metadata={
                        "location_type": loc.get("location_type"),
                        "areas": loc.get("areas", []),
                        "open_hours": loc.get("open_hours", {})
                    }
                )
                location_ids.append(location_id)
                
                # Add location-specific memories
                await remember_with_governance(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    entity_type="location",
                    entity_id=0,
                    memory_text=f"Location established: {loc['name']} - {loc['description'][:100]}...",
                    importance="medium",
                    emotional=False,
                    tags=["location", "story_setup", loc["location_type"]]
                )
        
        return location_ids
    
    @staticmethod
    async def _setup_all_relationships(
        ctx, user_id: int, conversation_id: int, 
        lilith_id: int, support_npc_ids: List[int]
    ):
        """Establish all story relationships"""
        
        canon_ctx = type('CanonicalContext', (), {
            'user_id': user_id,
            'conversation_id': conversation_id
        })()
        
        # Get NPC names for relationship context
        npc_names = {}
        async with get_db_connection_context() as conn:
            for npc_id in [lilith_id] + support_npc_ids:
                row = await conn.fetchrow(
                    "SELECT npc_name FROM NPCStats WHERE npc_id = $1",
                    npc_id
                )
                if row:
                    npc_names[npc_id] = row['npc_name']
        
        # Establish Lilith's relationships
        relationships = [
            {
                "source": lilith_id,
                "target": support_npc_ids[0],  # Marcus
                "type": "owns",
                "reverse": "owned_by",
                "strength": 90
            },
            {
                "source": lilith_id,
                "target": support_npc_ids[1],  # Sarah
                "type": "protector",
                "reverse": "protected_by", 
                "strength": 80
            },
            {
                "source": lilith_id,
                "target": support_npc_ids[2],  # Viktor
                "type": "enemy",
                "reverse": "enemy",
                "strength": 100
            }
        ]
        
        async with get_db_connection_context() as conn:
            for rel in relationships:
                # Create forward relationship
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
                
                # Create reverse relationship
                await canon.find_or_create_social_link(
                    canon_ctx, conn,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    entity1_type="npc",
                    entity1_id=rel["target"],
                    entity2_type="npc",
                    entity2_id=rel["source"],
                    link_type=rel["reverse"],
                    link_level=rel["strength"]
                )
        
        # Create shared memories between connected NPCs
        shared_memories = [
            {
                "npcs": [lilith_id, support_npc_ids[0]],
                "memory": (
                    "The night Marcus Sterling first knelt before the Queen, he wept. 'I've been "
                    "empty so long,' he confessed. She placed a collar around his neck with the "
                    "tenderness of a mother and the finality of a judge. 'Now you're mine.'"
                )
            },
            {
                "npcs": [lilith_id, support_npc_ids[1]],
                "memory": (
                    "Sarah was barely breathing when the Moth Queen found her. As she carried the "
                    "girl from that basement, she whispered, 'You're safe now. I promise you're safe.' "
                    "It was the first promise she'd made in years that she knew she'd keep."
                )
            },
            {
                "npcs": [lilith_id, support_npc_ids[2]],
                "memory": (
                    "Viktor's men had her surrounded that night five years ago. 'Just another moth,' "
                    "he laughed. But moths can burn too. By dawn, his warehouse was ash and three "
                    "girls were free. He's been hunting her ever since."
                )
            }
        ]
        
        for shared in shared_memories:
            for npc_id in shared["npcs"]:
                await remember_with_governance(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    entity_type="npc",
                    entity_id=npc_id,
                    memory_text=shared["memory"],
                    importance="high",
                    emotional=True,
                    tags=["shared_memory", "relationship", "story_foundation"]
                )
    
    @staticmethod
    async def _initialize_story_state(
        ctx, user_id: int, conversation_id: int, lilith_id: int
    ):
        """Initialize story tracking and state"""
        
        async with get_db_connection_context() as conn:
            # Create story state record
            await conn.execute(
                """
                INSERT INTO story_states (
                    user_id, conversation_id, story_id, current_act, 
                    current_beat, story_flags, progress, started_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                ON CONFLICT (user_id, conversation_id, story_id)
                DO UPDATE SET 
                    current_act = EXCLUDED.current_act,
                    current_beat = EXCLUDED.current_beat,
                    started_at = NOW()
                """,
                user_id, conversation_id, "the_moth_and_flame",
                1, "not_started",
                json.dumps({
                    "lilith_npc_id": lilith_id,
                    "trust_level": 0,
                    "masks_witnessed": [],
                    "secrets_discovered": [],
                    "moth_flame_dynamic": "unestablished",
                    "three_words_spoken": False,
                    "dual_identity_revealed": False,
                    "player_role": "unknown",
                    "emotional_intensity": 0,
                    "sessions_completed": 0,
                    "vulnerability_witnessed": 0,
                    "promises_made": []
                }),
                0
            )
            
            # Set initial atmosphere
            await conn.execute(
                """
                INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (user_id, conversation_id, key) 
                DO UPDATE SET value = EXCLUDED.value
                """,
                user_id, conversation_id, "story_atmosphere",
                json.dumps({
                    "tone": "gothic_romantic",
                    "lighting": "candlelit_shadows", 
                    "sound": "distant_gothic_music",
                    "scent": "incense_and_leather",
                    "feeling": "anticipation_and_unease"
                })
            )
            
            # Initialize player stats for the story
            await conn.execute(
                """
                INSERT INTO player_story_stats (
                    user_id, conversation_id, story_id, stats
                )
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (user_id, conversation_id, story_id)
                DO UPDATE SET stats = EXCLUDED.stats
                """,
                user_id, conversation_id, "the_moth_and_flame",
                json.dumps({
                    "submission": 0,
                    "devotion": 0,
                    "curiosity": 50,
                    "fear": 20,
                    "arousal": 0,
                    "pain_tolerance": 0,
                    "trust_given": 0,
                    "promises_kept": 0,
                    "vulnerability_shown": 0,
                    "moth_nature": 50,  # 0 = flame, 100 = moth
                    "corruption": 0,
                    "enlightenment": 0
                })
            )
    
    @staticmethod
    async def _setup_special_mechanics(
        ctx, user_id: int, conversation_id: int, lilith_id: int
    ):
        """Configure special story mechanics"""
        
        async with get_db_connection_context() as conn:
            # Create mask progression tracking
            await conn.execute(
                """
                INSERT INTO npc_special_mechanics (
                    user_id, conversation_id, npc_id, mechanic_type, mechanic_data
                )
                VALUES ($1, $2, $3, $4, $5)
                """,
                user_id, conversation_id, lilith_id, "mask_system",
                json.dumps({
                    "available_masks": [
                        {
                            "name": "Porcelain Goddess",
                            "description": "Perfect, cold, untouchable divinity",
                            "trust_required": 0,
                            "effects": {
                                "dominance_modifier": "+20",
                                "vulnerability": "hidden",
                                "dialogue_style": "commanding"
                            }
                        },
                        {
                            "name": "Leather Predator",
                            "description": "Dangerous, hunting, protective fury",
                            "trust_required": 30,
                            "effects": {
                                "cruelty_modifier": "+30",
                                "intensity": "maximum",
                                "dialogue_style": "threatening"
                            }
                        },
                        {
                            "name": "Lace Vulnerability", 
                            "description": "Soft edges barely containing sharp pain",
                            "trust_required": 60,
                            "effects": {
                                "affection_visible": True,
                                "vulnerability": "glimpses",
                                "dialogue_style": "poetic"
                            }
                        },
                        {
                            "name": "No Mask",
                            "description": "The broken woman, the frightened girl, the truth",
                            "trust_required": 85,
                            "effects": {
                                "all_stats_true": True,
                                "vulnerability": "complete",
                                "dialogue_style": "raw"
                            }
                        }
                    ],
                    "current_mask": "Porcelain Goddess",
                    "mask_integrity": 100,
                    "slippage_triggers": []
                })
            )
            
            # Create poetry moment triggers
            await conn.execute(
                """
                INSERT INTO npc_special_mechanics (
                    user_id, conversation_id, npc_id, mechanic_type, mechanic_data
                )
                VALUES ($1, $2, $3, $4, $5)
                """,
                user_id, conversation_id, lilith_id, "poetry_triggers",
                json.dumps({
                    "trigger_conditions": [
                        {"emotion": "vulnerability", "chance": 0.7},
                        {"emotion": "passion", "chance": 0.6},
                        {"emotion": "fear", "chance": 0.8},
                        {"emotion": "affection", "chance": 0.5}
                    ],
                    "poetry_used": [],
                    "understanding_tracker": {
                        "attempts": 0,
                        "successes": 0,
                        "trust_impact": 5
                    }
                })
            )
            
            # Create three words mechanic
            await conn.execute(
                """
                INSERT INTO npc_special_mechanics (
                    user_id, conversation_id, npc_id, mechanic_type, mechanic_data
                )
                VALUES ($1, $2, $3, $4, $5)
                """,
                user_id, conversation_id, lilith_id, "three_words",
                json.dumps({
                    "near_speaking_moments": [],
                    "player_attempts_to_hear": 0,
                    "trust_threshold": 95,
                    "emotional_threshold": 90,
                    "spoken": False,
                    "player_spoke_first": None,
                    "her_response": None
                })
            )
    
    @staticmethod
    async def _set_initial_atmosphere(ctx, user_id: int, conversation_id: int):
        """Set the initial story atmosphere"""
        
        # Create atmospheric introduction
        intro_message = {
            "type": "story_introduction",
            "content": (
                "The city breathes differently after midnight. In the underground district, "
                "where neon bleeds into shadow and desire takes corporeal form, you've heard "
                "whispers of a place called the Velvet Sanctum. They say a Queen holds court "
                "there - beautiful, terrible, offering transcendence through submission.\n\n"
                
                "You stand before an unmarked door, bass thrumming through your bones like a "
                "second heartbeat. The bouncer, scarred and silent, evaluates you with eyes "
                "that have seen too much. Finally, he steps aside.\n\n"
                
                "'The Queen is holding court tonight,' he says. 'Try not to stare. She notices "
                "everything.'\n\n"
                
                "As you descend the stairs, each step takes you further from the world you know. "
                "The air grows thick with incense and possibility. Somewhere below, a woman in a "
                "porcelain mask rules over hearts willing to break for her attention.\n\n"
                
                "Welcome to the beginning of your beautiful destruction."
            ),
            "atmosphere": {
                "visual": "Candlelight painting shadows on velvet walls",
                "auditory": "Gothic electronica pulsing like a dark heartbeat",
                "olfactory": "Incense, leather, and the faint scent of roses",
                "emotional": "Anticipation mixed with delicious fear"
            }
        }
        
        # Store as initial memory
        await remember_with_governance(
            user_id=user_id,
            conversation_id=conversation_id,
            entity_type="player",
            entity_id=user_id,
            memory_text=intro_message["content"],
            importance="high",
            emotional=True,
            tags=["story_start", "first_impression", "atmosphere"]
        )
        
        # Set initial time and location
        async with get_db_connection_context() as conn:
            await conn.execute(
                """
                UPDATE CurrentRoleplay 
                SET value = $3
                WHERE user_id = $1 AND conversation_id = $2 AND key = 'TimeOfDay'
                """,
                user_id, conversation_id, "Night"
            )
            
            await conn.execute(
                """
                UPDATE CurrentRoleplay
                SET value = $3  
                WHERE user_id = $1 AND conversation_id = $2 AND key = 'CurrentLocation'
                """,
                user_id, conversation_id, "Velvet Sanctum - Entrance"
            )

class MothFlameStoryProgression:
    """Handles story progression and beat triggers"""
    
    @staticmethod
    async def check_beat_triggers(user_id: int, conversation_id: int) -> Optional[str]:
        """Check if any story beats should trigger"""
        
        async with get_db_connection_context() as conn:
            # Get current story state
            state_row = await conn.fetchrow(
                """
                SELECT current_beat, story_flags, progress
                FROM story_states
                WHERE user_id = $1 AND conversation_id = $2 AND story_id = $3
                """,
                user_id, conversation_id, "the_moth_and_flame"
            )
            
            if not state_row:
                return None
            
            current_beat = state_row['current_beat']
            story_flags = json.loads(state_row['story_flags'])
            
            # Get story definition
            from story_templates.moth.the_moth_and_flame import THE_MOTH_AND_FLAME
            
            # Check each beat's trigger conditions
            for beat in THE_MOTH_AND_FLAME.story_beats:
                if beat.id == current_beat:
                    continue  # Skip current beat
                
                if await MothFlameStoryProgression._check_single_beat_conditions(
                    beat, story_flags, user_id, conversation_id
                ):
                    return beat.id
        
        return None
    
    @staticmethod
    async def _check_single_beat_conditions(
        beat: StoryBeat, story_flags: Dict, user_id: int, conversation_id: int
    ) -> bool:
        """Check if a single beat's conditions are met"""
        
        # Implementation would check each trigger condition
        # This is a simplified version
        for condition, value in beat.trigger_conditions.items():
            if condition == "game_start" and value:
                return story_flags.get("progress", 0) == 0
            elif condition == "completed_beats":
                completed = story_flags.get("completed_beats", [])
                if not all(b in completed for b in value):
                    return False
            # Add more condition checks...
        
        return True
    
    @staticmethod
    async def trigger_story_beat(
        user_id: int, conversation_id: int, beat_id: str
    ) -> Dict[str, Any]:
        """Trigger a specific story beat"""
        
        from story_templates.moth.the_moth_and_flame import THE_MOTH_AND_FLAME
        
        # Find the beat
        beat = next((b for b in THE_MOTH_AND_FLAME.story_beats if b.id == beat_id), None)
        if not beat:
            return {"error": "Beat not found"}
        
        # Apply beat outcomes
        # Update story state
        # Create events/memories
        # etc.
        
        return {
            "status": "success",
            "beat_triggered": beat_id,
            "outcomes": beat.outcomes
        }
