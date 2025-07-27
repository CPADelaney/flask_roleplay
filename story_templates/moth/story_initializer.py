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
from npcs.preset_npc_handler import PresetNPCHandler

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
        """Create Lilith with all her complexity using canonical functions"""
        
        try:
            # First, use canonical function to find or create
            async with get_db_connection_context() as conn:
                lilith_id = await canon.find_or_create_npc(
                    ctx, conn,
                    npc_name=LILITH_RAVENCROFT["name"],
                    role=LILITH_RAVENCROFT["role"],
                    affiliations=["Velvet Sanctum", "The Underground Network", "The Moth Queen Identity"]
                )
                
                # Check if this is a new NPC or existing
                npc_details = await conn.fetchrow("""
                    SELECT personality_traits, created_at, age
                    FROM NPCStats 
                    WHERE npc_id = $1
                """, lilith_id)
                
                # Determine if we need to update
                is_new_npc = False
                if npc_details and npc_details['created_at']:
                    time_since_creation = datetime.now() - npc_details['created_at']
                    is_new_npc = time_since_creation.total_seconds() < 60
                else:
                    is_new_npc = True
                
                # Only update if new or has minimal data
                if is_new_npc or not npc_details['personality_traits']:
                    logger.info(f"Updating Lilith (ID: {lilith_id}) with full preset data")
                    
                    # Use PresetNPCHandler to add all the detailed data
                    await PresetNPCHandler.create_detailed_npc(ctx, LILITH_RAVENCROFT, {
                        "story_context": "moth_flame",
                        "is_main_character": True
                    })
                else:
                    logger.info(f"Lilith already exists (ID: {lilith_id}), adding special properties only")
                    
                    # Just add special properties that won't conflict
                    await MothFlameStoryInitializer._add_lilith_special_properties(
                        ctx, lilith_id, LILITH_RAVENCROFT, user_id, conversation_id
                    )
                    
                    # Ensure memory system is initialized
                    memory_count = await conn.fetchval("""
                        SELECT COUNT(*) FROM npc_memories
                        WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
                    """, user_id, conversation_id, lilith_id)
                    
                    if memory_count < 3:
                        await MothFlameStoryInitializer._initialize_lilith_memory_system(
                            user_id, conversation_id, lilith_id, LILITH_RAVENCROFT
                        )
            
            return lilith_id
            
        except Exception as e:
            logger.error(f"Failed to create Lilith Ravencroft: {e}", exc_info=True)
            raise
    
    @staticmethod
    async def _create_supporting_npcs(ctx, user_id: int, conversation_id: int) -> List[int]:
        """Create all supporting NPCs for the story using canonical functions"""
        
        npc_ids = []
        
        # Define all NPCs with their data
        npcs_to_create = [
            {
                "name": "Marcus Sterling",
                "role": "Devoted Submissive / Broken Executive",
                "affiliations": ["Velvet Sanctum", "The Queen's Inner Circle"],
                "data": {
                    "id": "marcus_sterling",
                    "name": "Marcus Sterling",
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
                }
            },
            {
                "name": "Sarah Chen",
                "role": "Trafficking Survivor / Safehouse Resident",
                "affiliations": ["The Underground Network", "Moth Queen's Saved"],
                "data": {
                    "id": "sarah_chen",
                    "name": "Sarah Chen",
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
                }
            },
            {
                "name": "Viktor Kozlov",
                "role": "Human Trafficker / Crime Boss",
                "affiliations": ["Eastern European Crime Syndicate", "The Shadow Trade"],
                "data": {
                    "id": "viktor_kozlov",
                    "name": "Viktor Kozlov",
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
                }
            }
        ]
        
        # Minor NPCs
        minor_npcs = [
            {
                "name": "Jessica Vale",
                "role": "Sanctum Regular / Lawyer",
                "affiliations": ["Velvet Sanctum"],
                "data": {
                    "id": "jessica_vale",
                    "name": "Jessica Vale",
                    "sex": "female",
                    "age": 32,
                    "archetype": "Sanctum Regular",
                    "physical_description": "A lawyer by day who seeks absolution in submission",
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
                    "current_location": "Velvet Sanctum",
                    "introduced": False
                }
            },
            {
                "name": "Amanda Ross",
                "role": "New Submissive",
                "affiliations": ["Velvet Sanctum"],
                "data": {
                    "id": "amanda_ross",
                    "name": "Amanda Ross",
                    "sex": "female",
                    "age": 26,
                    "archetype": "Curious Newcomer",
                    "physical_description": "Curious and eager, hasn't learned the rules yet",
                    "personality": {
                        "personality_traits": ["curious", "eager", "naive"],
                        "likes": ["new experiences", "the Queen's attention"],
                        "dislikes": ["being corrected", "feeling out of place"],
                        "hobbies": ["exploring the scene"]
                    },
                    "stats": {
                        "dominance": 15,
                        "cruelty": 5,
                        "affection": 70,
                        "trust": 60,
                        "respect": 80,
                        "intensity": 40
                    },
                    "current_location": "Velvet Sanctum",
                    "introduced": False
                }
            },
            {
                "name": "Diana Moon",
                "role": "Former Favorite",
                "affiliations": ["Velvet Sanctum"],
                "data": {
                    "id": "diana_moon",
                    "name": "Diana Moon",
                    "sex": "female",
                    "age": 35,
                    "archetype": "Fallen from Grace",
                    "physical_description": "Once held the Queen's attention, now watches from the shadows",
                    "personality": {
                        "personality_traits": ["bitter", "watchful", "experienced"],
                        "likes": ["remembering better times", "the Queen's rare acknowledgment"],
                        "dislikes": ["new favorites", "being replaced"],
                        "hobbies": ["haunting the Sanctum"]
                    },
                    "stats": {
                        "dominance": 30,
                        "cruelty": 40,
                        "affection": 50,
                        "trust": 20,
                        "respect": 60,
                        "intensity": 70
                    },
                    "current_location": "Velvet Sanctum",
                    "introduced": False
                }
            }
        ]
        
        # Combine all NPCs
        all_npcs = npcs_to_create + minor_npcs
        
        # Create each NPC using canonical functions
        async with get_db_connection_context() as conn:
            for npc_def in all_npcs:
                try:
                    # First, find or create the NPC canonically
                    npc_id = await canon.find_or_create_npc(
                        ctx, conn,
                        npc_name=npc_def["name"],
                        role=npc_def["role"],
                        affiliations=npc_def.get("affiliations", [])
                    )
                    
                    if npc_id:
                        # Check if this NPC needs full data update
                        existing = await conn.fetchrow("""
                            SELECT personality_traits, created_at
                            FROM NPCStats WHERE npc_id = $1
                        """, npc_id)
                        
                        # Determine if we need to update
                        is_new_npc = False
                        if existing and existing['created_at']:
                            time_since_creation = datetime.now() - existing['created_at']
                            is_new_npc = time_since_creation.total_seconds() < 60
                        else:
                            is_new_npc = True
                        
                        # Only update if newly created or has minimal data
                        if is_new_npc or not existing['personality_traits']:
                            logger.info(f"Updating {npc_def['name']} (ID: {npc_id}) with full preset data")
                            # Use the preset handler to add full details
                            await PresetNPCHandler.create_detailed_npc(
                                ctx, npc_def["data"], {"story_context": "moth_flame"}
                            )
                        else:
                            logger.info(f"{npc_def['name']} already exists (ID: {npc_id}), skipping full update")
                    
                    npc_ids.append(npc_id)
                    
                except Exception as e:
                    logger.error(f"Failed to create NPC {npc_def['name']}: {e}", exc_info=True)
                    # Continue with other NPCs even if one fails
        
        return npc_ids
    
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
        
        # Store initial memories
        initial_memories = MothFlameStoryInitializer._create_lilith_memories()
        for memory_text in initial_memories:
            await memory_system.remember(
                entity_type="npc",
                entity_id=npc_id,
                memory_text=memory_text,
                importance="high",
                emotional=True,
                tags=["backstory", "core_memory", "trauma", "motivation"]
            )
        
        # Create memory schemas specific to her character
        await memory_system.generate_schemas(
            entity_type="npc",
            entity_id=npc_id
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
        
        # Map names to IDs
        npc_id_map = {name: npc_id for npc_id, name in npc_names.items()}
        
        # Establish Lilith's relationships
        relationships = []
        
        # Marcus Sterling (devoted submissive)
        if "Marcus Sterling" in npc_id_map:
            relationships.append({
                "source": lilith_id,
                "target": npc_id_map["Marcus Sterling"],
                "type": "owns",
                "reverse": "owned_by",
                "strength": 90
            })
        
        # Sarah Chen (saved victim)
        if "Sarah Chen" in npc_id_map:
            relationships.append({
                "source": lilith_id,
                "target": npc_id_map["Sarah Chen"],
                "type": "protector",
                "reverse": "protected_by", 
                "strength": 80
            })
        
        # Viktor Kozlov (enemy)
        if "Viktor Kozlov" in npc_id_map:
            relationships.append({
                "source": lilith_id,
                "target": npc_id_map["Viktor Kozlov"],
                "type": "enemy",
                "reverse": "enemy",
                "strength": 100
            })
        
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
        shared_memories = []
        
        if "Marcus Sterling" in npc_id_map:
            shared_memories.append({
                "npcs": [lilith_id, npc_id_map["Marcus Sterling"]],
                "memory": (
                    "The night Marcus Sterling first knelt before the Queen, he wept. 'I've been "
                    "empty so long,' he confessed. She placed a collar around his neck with the "
                    "tenderness of a mother and the finality of a judge. 'Now you're mine.'"
                )
            })
        
        if "Sarah Chen" in npc_id_map:
            shared_memories.append({
                "npcs": [lilith_id, npc_id_map["Sarah Chen"]],
                "memory": (
                    "Sarah was barely breathing when the Moth Queen found her. As she carried the "
                    "girl from that basement, she whispered, 'You're safe now. I promise you're safe.' "
                    "It was the first promise she'd made in years that she knew she'd keep."
                )
            })
        
        if "Viktor Kozlov" in npc_id_map:
            shared_memories.append({
                "npcs": [lilith_id, npc_id_map["Viktor Kozlov"]],
                "memory": (
                    "Viktor's men had her surrounded that night five years ago. 'Just another moth,' "
                    "he laughed. But moths can burn too. By dawn, his warehouse was ash and three "
                    "girls were free. He's been hunting her ever since."
                )
            })
        
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
                ON CONFLICT (user_id, conversation_id, npc_id, mechanic_type)
                DO UPDATE SET mechanic_data = EXCLUDED.mechanic_data
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
                ON CONFLICT (user_id, conversation_id, npc_id, mechanic_type)
                DO UPDATE SET mechanic_data = EXCLUDED.mechanic_data
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
                ON CONFLICT (user_id, conversation_id, npc_id, mechanic_type)
                DO UPDATE SET mechanic_data = EXCLUDED.mechanic_data
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
                SELECT current_beat, story_flags, progress, current_act
                FROM story_states
                WHERE user_id = $1 AND conversation_id = $2 AND story_id = $3
                """,
                user_id, conversation_id, "the_moth_and_flame"
            )
            
            if not state_row:
                return None
            
            current_beat = state_row['current_beat']
            current_act = state_row['current_act']
            story_flags = json.loads(state_row['story_flags'] or '{}')
            completed_beats = story_flags.get('completed_beats', [])
            
            # Get story definition
            from story_templates.moth.the_moth_and_flame import THE_MOTH_AND_FLAME
            
            # Sort beats by priority (act number, then order in story)
            sorted_beats = sorted(THE_MOTH_AND_FLAME.story_beats, 
                                key=lambda b: (self._get_beat_act(b), 
                                             THE_MOTH_AND_FLAME.story_beats.index(b)))
            
            # Check each beat's trigger conditions
            for beat in sorted_beats:
                # Skip if already completed
                if beat.id in completed_beats:
                    continue
                
                # Skip if it's the current beat
                if beat.id == current_beat:
                    continue
                
                # Check if this beat's conditions are met
                if await MothFlameStoryProgression._check_single_beat_conditions(
                    beat, story_flags, current_act, user_id, conversation_id, conn
                ):
                    logger.info(f"Story beat '{beat.id}' conditions met for user {user_id}")
                    return beat.id
        
        return None
    
    @staticmethod
    def _get_beat_act(beat: StoryBeat) -> int:
        """Determine which act a beat belongs to"""
        # Based on narrative stage progression
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
        """Check if a single beat's conditions are met"""
        
        try:
            # Check each trigger condition
            for condition, value in beat.trigger_conditions.items():
                
                # Game start condition
                if condition == "game_start" and value:
                    if story_flags.get("progress", 0) > 0 or story_flags.get("completed_beats", []):
                        return False
                
                # Act requirement
                elif condition == "act":
                    if current_act != value:
                        return False
                
                # Time of day requirement
                elif condition == "time":
                    current_time = await conn.fetchval(
                        """
                        SELECT value FROM CurrentRoleplay
                        WHERE user_id = $1 AND conversation_id = $2 AND key = 'TimeOfDay'
                        """,
                        user_id, conversation_id
                    )
                    if current_time != value.title():
                        return False
                
                # Location requirement
                elif condition == "location":
                    current_location = await conn.fetchval(
                        """
                        SELECT value FROM CurrentRoleplay
                        WHERE user_id = $1 AND conversation_id = $2 AND key = 'CurrentLocation'
                        """,
                        user_id, conversation_id
                    )
                    if not current_location or value.lower() not in current_location.lower():
                        return False
                
                # Completed beats requirement
                elif condition == "completed_beats":
                    completed = story_flags.get("completed_beats", [])
                    if not all(b in completed for b in value):
                        return False
                
                # Times visited location
                elif condition == "times_visited_sanctum":
                    visit_count = story_flags.get("sanctum_visits", 0)
                    if visit_count < value:
                        return False
                
                # NPC awareness level
                elif condition == "npc_awareness":
                    for npc_name, requirements in value.items():
                        npc_id = story_flags.get(f"{npc_name.lower().replace(' ', '_')}_id")
                        if not npc_id:
                            # Try to find NPC by name
                            npc_id = await conn.fetchval(
                                """
                                SELECT npc_id FROM NPCStats
                                WHERE user_id = $1 AND conversation_id = $2 
                                AND LOWER(npc_name) = LOWER($3)
                                """,
                                user_id, conversation_id, npc_name
                            )
                        
                        if npc_id:
                            # Check awareness (could be trust, closeness, or custom awareness stat)
                            awareness = await conn.fetchval(
                                """
                                SELECT GREATEST(trust, closeness, 
                                    COALESCE((metadata->>'awareness')::int, 0)) as awareness
                                FROM NPCStats
                                WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
                                """,
                                user_id, conversation_id, npc_id
                            )
                            
                            if not awareness or awareness < requirements.get("min", 0):
                                return False
                
                # Player watched performance
                elif condition == "player_watched_performance" and value:
                    if not story_flags.get("watched_performance", False):
                        return False
                
                # Quest completed
                elif condition == "quest_completed":
                    quest_status = await conn.fetchval(
                        """
                        SELECT status FROM Quests
                        WHERE user_id = $1 AND conversation_id = $2 
                        AND quest_name = $3
                        """,
                        user_id, conversation_id, value
                    )
                    if quest_status != "completed":
                        return False
                
                # Has item
                elif condition == "has_item":
                    item_count = await conn.fetchval(
                        """
                        SELECT quantity FROM PlayerInventory
                        WHERE user_id = $1 AND conversation_id = $2 
                        AND item_name = $3
                        """,
                        user_id, conversation_id, value
                    )
                    if not item_count or item_count <= 0:
                        return False
                
                # Relationship requirements
                elif condition == "relationship":
                    for npc_name, requirements in value.items():
                        npc_id = story_flags.get(f"{npc_name.lower().replace(' ', '_')}_id")
                        if not npc_id:
                            npc_id = await conn.fetchval(
                                """
                                SELECT npc_id FROM NPCStats
                                WHERE user_id = $1 AND conversation_id = $2 
                                AND LOWER(npc_name) = LOWER($3)
                                """,
                                user_id, conversation_id, npc_name
                            )
                        
                        if npc_id:
                            trust = await conn.fetchval(
                                """
                                SELECT trust FROM NPCStats
                                WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
                                """,
                                user_id, conversation_id, npc_id
                            )
                            
                            if not trust or trust < requirements.get("min", 0):
                                return False
                
                # Sanctum closed
                elif condition == "sanctum_closed" and value:
                    current_time = await conn.fetchval(
                        """
                        SELECT value FROM CurrentRoleplay
                        WHERE user_id = $1 AND conversation_id = $2 AND key = 'TimeOfDay'
                        """,
                        user_id, conversation_id
                    )
                    # Sanctum is closed during Late Night or Morning
                    if current_time not in ["Late Night", "Morning"]:
                        return False
                
                # Private moment
                elif condition == "private_moment" and value:
                    # Check if alone with Lilith
                    current_location = await conn.fetchval(
                        """
                        SELECT value FROM CurrentRoleplay
                        WHERE user_id = $1 AND conversation_id = $2 AND key = 'CurrentLocation'
                        """,
                        user_id, conversation_id
                    )
                    if not current_location or "private" not in current_location.lower():
                        return False
                
                # Trust established
                elif condition == "trust_established" and value:
                    lilith_id = story_flags.get("lilith_npc_id")
                    if lilith_id:
                        trust = await conn.fetchval(
                            """
                            SELECT trust FROM NPCStats
                            WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
                            """,
                            user_id, conversation_id, lilith_id
                        )
                        if not trust or trust < 40:  # Minimum trust threshold
                            return False
                
                # Intimacy level
                elif condition == "intimacy_level":
                    lilith_id = story_flags.get("lilith_npc_id")
                    if lilith_id:
                        # Use combination of trust and closeness
                        intimacy = await conn.fetchval(
                            """
                            SELECT (trust + closeness) / 2 as intimacy FROM NPCStats
                            WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
                            """,
                            user_id, conversation_id, lilith_id
                        )
                        if not intimacy or intimacy < value.get("min", 0):
                            return False
                
                # Mask removed
                elif condition == "mask_removed":
                    if story_flags.get("mask_removed_count", 0) < 1:
                        return False
                
                # Helped vulnerable NPC
                elif condition == "helped_vulnerable_npc" and value:
                    if not story_flags.get("helped_trafficking_victim", False):
                        return False
                
                # Devotion level
                elif condition == "devotion":
                    player_devotion = await conn.fetchval(
                        """
                        SELECT devotion FROM player_story_stats
                        WHERE user_id = $1 AND conversation_id = $2 
                        AND story_id = 'the_moth_and_flame'
                        """,
                        user_id, conversation_id
                    )
                    
                    # Parse devotion from JSON if needed
                    if player_devotion and isinstance(player_devotion, str):
                        try:
                            stats = json.loads(player_devotion)
                            player_devotion = stats.get('devotion', 0)
                        except:
                            player_devotion = 0
                    
                    if not player_devotion or player_devotion < value.get("min", 0):
                        return False
                
                # Sessions completed
                elif condition == "sessions_completed":
                    session_count = story_flags.get("private_sessions_completed", 0)
                    if session_count < value.get("min", 0):
                        return False
                
                # Discovered secret
                elif condition == "discovered_secret":
                    secrets = story_flags.get("secrets_discovered", [])
                    if value not in secrets:
                        return False
                
                # Random event (for dynamic story beats)
                elif condition == "random_event":
                    # Random events have a chance to trigger
                    import random
                    if random.random() > 0.3:  # 30% chance
                        return False
                
                # Emotional intensity
                elif condition == "emotional_intensity":
                    intensity = story_flags.get("current_emotional_intensity", 0)
                    if intensity < value.get("min", 0):
                        return False
                
                # Crisis resolved
                elif condition == "crisis_resolved" and value:
                    if not story_flags.get("safehouse_crisis_resolved", False):
                        return False
                
                # Major choice made
                elif condition == "major_choice":
                    if value not in story_flags.get("major_choices_made", []):
                        return False
                
                # Mutual confession
                elif condition == "mutual_confession" and value:
                    if not story_flags.get("mutual_love_confession", False):
                        return False
                
                # Major choice made (any)
                elif condition == "major_choice_made" and value:
                    if not story_flags.get("major_choices_made", []):
                        return False
                
                # Story completion percentage
                elif condition == "story_complete":
                    if story_flags.get("progress", 0) < value:
                        return False
            
            # All conditions passed
            return True
            
        except Exception as e:
            logger.error(f"Error checking beat conditions for {beat.id}: {e}", exc_info=True)
            return False
    
    @staticmethod
    async def trigger_story_beat(
        user_id: int, conversation_id: int, beat_id: str
    ) -> Dict[str, Any]:
        """Trigger a specific story beat"""
        
        from story_templates.moth.the_moth_and_flame import THE_MOTH_AND_FLAME
        
        # Find the beat
        beat = next((b for b in THE_MOTH_AND_FLAME.story_beats if b.id == beat_id), None)
        if not beat:
            logger.error(f"Beat {beat_id} not found in story definition")
            return {"error": "Beat not found"}
        
        try:
            async with get_db_connection_context() as conn:
                # Get current story state
                state_row = await conn.fetchrow(
                    """
                    SELECT story_flags, progress FROM story_states
                    WHERE user_id = $1 AND conversation_id = $2 AND story_id = $3
                    """,
                    user_id, conversation_id, "the_moth_and_flame"
                )
                
                if not state_row:
                    return {"error": "Story state not found"}
                
                story_flags = json.loads(state_row['story_flags'] or '{}')
                
                # Apply beat outcomes
                outcomes_applied = await MothFlameStoryProgression._apply_beat_outcomes(
                    beat, story_flags, user_id, conversation_id, conn
                )
                
                # Update story state
                completed_beats = story_flags.get('completed_beats', [])
                if beat_id not in completed_beats:
                    completed_beats.append(beat_id)
                
                story_flags['completed_beats'] = completed_beats
                story_flags['last_beat_triggered'] = beat_id
                story_flags['last_beat_timestamp'] = datetime.now().isoformat()
                
                # Calculate new progress
                total_beats = len(THE_MOTH_AND_FLAME.story_beats)
                progress = (len(completed_beats) / total_beats) * 100
                
                # Update in database
                await conn.execute(
                    """
                    UPDATE story_states
                    SET current_beat = $4, story_flags = $5, progress = $6, updated_at = NOW()
                    WHERE user_id = $1 AND conversation_id = $2 AND story_id = $3
                    """,
                    user_id, conversation_id, "the_moth_and_flame",
                    beat_id, json.dumps(story_flags), progress
                )
                
                # Create memory of this story moment
                if story_flags.get('lilith_npc_id'):
                    await remember_with_governance(
                        user_id=user_id,
                        conversation_id=conversation_id,
                        entity_type="npc",
                        entity_id=story_flags['lilith_npc_id'],
                        memory_text=f"Story moment: {beat.name} - {beat.description}",
                        importance="high",
                        emotional=True,
                        tags=["story_beat", beat_id, beat.narrative_stage.lower().replace(" ", "_")]
                    )
                
                # Log canonical event
                ctx = type('Context', (), {
                    'user_id': user_id,
                    'conversation_id': conversation_id
                })()
                
                await canon.log_canonical_event(
                    ctx, conn,
                    f"Story beat triggered: {beat.name}",
                    tags=["story_progression", beat_id],
                    significance=8
                )
                
                logger.info(f"Successfully triggered story beat {beat_id} for user {user_id}")
                
                return {
                    "status": "success",
                    "beat_triggered": beat_id,
                    "beat_name": beat.name,
                    "narrative_stage": beat.narrative_stage,
                    "outcomes": outcomes_applied,
                    "dialogue_hints": beat.dialogue_hints,
                    "can_skip": beat.can_skip,
                    "progress": progress
                }
                
        except Exception as e:
            logger.error(f"Error triggering story beat {beat_id}: {e}", exc_info=True)
            return {
                "error": str(e),
                "beat_id": beat_id
            }
    
    @staticmethod
    async def _apply_beat_outcomes(
        beat: StoryBeat, 
        story_flags: Dict,
        user_id: int, 
        conversation_id: int,
        conn
    ) -> Dict[str, Any]:
        """Apply all outcomes from a story beat"""
        
        applied_outcomes = {}
        
        for outcome_type, outcome_data in beat.outcomes.items():
            try:
                # Relationship added
                if outcome_type == "relationship_added":
                    npc_name = outcome_data.get("npc")
                    rel_type = outcome_data.get("type", "neutral")
                    
                    # Find the NPC
                    npc_id = await conn.fetchval(
                        """
                        SELECT npc_id FROM NPCStats
                        WHERE user_id = $1 AND conversation_id = $2 
                        AND LOWER(npc_name) = LOWER($3)
                        """,
                        user_id, conversation_id, npc_name
                    )
                    
                    if npc_id:
                        # Get player ID
                        player_id = await conn.fetchval(
                            """
                            SELECT id FROM PlayerStats
                            WHERE user_id = $1 AND conversation_id = $2 
                            AND player_name = 'Chase'
                            """,
                            user_id, conversation_id
                        )
                        
                        if player_id:
                            # Create social link
                            from lore.core import canon
                            ctx = type('Context', (), {
                                'user_id': user_id,
                                'conversation_id': conversation_id
                            })()
                            
                            await canon.find_or_create_social_link(
                                ctx, conn,
                                user_id=user_id,
                                conversation_id=conversation_id,
                                entity1_type="player",
                                entity1_id=player_id,
                                entity2_type="npc",
                                entity2_id=npc_id,
                                link_type=rel_type,
                                link_level=20  # Starting level
                            )
                            
                            applied_outcomes["relationship_added"] = {
                                "npc": npc_name,
                                "type": rel_type
                            }
                
                # Player stats changes
                elif outcome_type == "player_stats":
                    stats_row = await conn.fetchrow(
                        """
                        SELECT stats FROM player_story_stats
                        WHERE user_id = $1 AND conversation_id = $2 
                        AND story_id = 'the_moth_and_flame'
                        """,
                        user_id, conversation_id
                    )
                    
                    if stats_row:
                        current_stats = json.loads(stats_row['stats'])
                        
                        # Apply stat changes
                        for stat, change in outcome_data.items():
                            if isinstance(change, str) and change.startswith(('+', '-')):
                                # Parse the change value
                                change_val = int(change)
                                current_val = current_stats.get(stat, 0)
                                new_val = max(0, min(100, current_val + change_val))
                                current_stats[stat] = new_val
                            else:
                                current_stats[stat] = change
                        
                        # Update stats
                        await conn.execute(
                            """
                            UPDATE player_story_stats
                            SET stats = $3
                            WHERE user_id = $1 AND conversation_id = $2 
                            AND story_id = 'the_moth_and_flame'
                            """,
                            user_id, conversation_id, json.dumps(current_stats)
                        )
                        
                        applied_outcomes["player_stats"] = outcome_data
                
                # Location unlocked
                elif outcome_type == "location_unlocked":
                    # Add to known locations
                    known_locations = story_flags.get("known_locations", [])
                    if outcome_data not in known_locations:
                        known_locations.append(outcome_data)
                        story_flags["known_locations"] = known_locations
                    
                    applied_outcomes["location_unlocked"] = outcome_data
                
                # Knowledge/secrets/facts gained
                elif outcome_type in ["knowledge_gained", "learned_fact", "learned_truth", 
                                     "learned_secret", "discovered_secret"]:
                    knowledge_key = "knowledge_gained" if outcome_type == "knowledge_gained" else "secrets_discovered"
                    knowledge_list = story_flags.get(knowledge_key, [])
                    if outcome_data not in knowledge_list:
                        knowledge_list.append(outcome_data)
                        story_flags[knowledge_key] = knowledge_list
                    
                    applied_outcomes[outcome_type] = outcome_data
                
                # NPC awareness
                elif outcome_type == "npc_awareness":
                    for npc_name, change in outcome_data.items():
                        awareness_key = f"{npc_name.lower().replace(' ', '_')}_awareness"
                        current = story_flags.get(awareness_key, 0)
                        story_flags[awareness_key] = current + change
                    
                    applied_outcomes["npc_awareness"] = outcome_data
                
                # Quest added
                elif outcome_type == "quest_added":
                    # Create quest in database
                    await conn.execute(
                        """
                        INSERT INTO Quests (user_id, conversation_id, quest_name, 
                                          status, quest_giver, created_at)
                        VALUES ($1, $2, $3, 'active', 'Lilith Ravencroft', NOW())
                        ON CONFLICT (user_id, conversation_id, quest_name) DO NOTHING
                        """,
                        user_id, conversation_id, outcome_data
                    )
                    
                    applied_outcomes["quest_added"] = outcome_data
                
                # Item received
                elif outcome_type == "item_received":
                    # Add to inventory
                    await conn.execute(
                        """
                        INSERT INTO PlayerInventory (user_id, conversation_id, player_name,
                                                   item_name, quantity, category)
                        VALUES ($1, $2, 'Chase', $3, 1, 'story_item')
                        ON CONFLICT (user_id, conversation_id, player_name, item_name)
                        DO UPDATE SET quantity = PlayerInventory.quantity + 1
                        """,
                        user_id, conversation_id, outcome_data
                    )
                    
                    applied_outcomes["item_received"] = outcome_data
                
                # Relationship progress
                elif outcome_type == "relationship_progress":
                    for npc_name, change in outcome_data.items():
                        # Find NPC and update trust
                        npc_id = await conn.fetchval(
                            """
                            SELECT npc_id FROM NPCStats
                            WHERE user_id = $1 AND conversation_id = $2 
                            AND LOWER(npc_name) = LOWER($3)
                            """,
                            user_id, conversation_id, npc_name
                        )
                        
                        if npc_id:
                            await conn.execute(
                                """
                                UPDATE NPCStats
                                SET trust = LEAST(100, trust + $4)
                                WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
                                """,
                                user_id, conversation_id, npc_id, change
                            )
                    
                    applied_outcomes["relationship_progress"] = outcome_data
                
                # Skills learned
                elif outcome_type == "learned_skill":
                    skills = story_flags.get("learned_skills", [])
                    if outcome_data not in skills:
                        skills.append(outcome_data)
                        story_flags["learned_skills"] = skills
                    
                    applied_outcomes["learned_skill"] = outcome_data
                
                # Special story flags
                elif outcome_type in ["vulnerability_witnessed", "mask_removed", "three_words_moment",
                                     "permanent_bond", "ending_achieved"]:
                    story_flags[outcome_type] = True
                    if outcome_type == "mask_removed":
                        story_flags["mask_removed_count"] = story_flags.get("mask_removed_count", 0) + 1
                    
                    applied_outcomes[outcome_type] = True
                
                # Choice presented
                elif outcome_type == "choice_presented":
                    story_flags["pending_choice"] = outcome_data
                    applied_outcomes["choice_presented"] = outcome_data
                
                # New quest/role
                elif outcome_type in ["new_quest", "new_role", "gained_title"]:
                    story_flags[outcome_type] = outcome_data
                    applied_outcomes[outcome_type] = outcome_data
                
                # Relationship type/evolution/dynamic
                elif outcome_type in ["relationship_type", "relationship_evolution", "relationship_dynamic"]:
                    story_flags[f"player_{outcome_type}"] = outcome_data
                    applied_outcomes[outcome_type] = outcome_data
                
                # Lilith vulnerability level
                elif outcome_type == "lilith_vulnerability":
                    story_flags["lilith_vulnerability_level"] = outcome_data
                    applied_outcomes["lilith_vulnerability"] = outcome_data
                
                # Potential loss flag
                elif outcome_type == "potential_loss":
                    story_flags["potential_loss_risk"] = outcome_data
                    applied_outcomes["potential_loss"] = outcome_data
                    
            except Exception as e:
                logger.error(f"Error applying outcome {outcome_type}: {e}", exc_info=True)
                applied_outcomes[f"{outcome_type}_error"] = str(e)
        
        return applied_outcomes
