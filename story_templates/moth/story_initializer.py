# story_templates/moth/story_initializer.py
"""
Complete story initialization system for The Queen of Thorns
Handles NPC creation, location setup, and special mechanics
Integrated with SF Bay Area shadow network lore
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
from story_templates.moth.lore.world_lore_manager import SFBayQueenOfThornsPreset

logger = logging.getLogger(__name__)

class QueenOfThornsStoryInitializer:
    """Complete initialization system for The Queen of Thorns story"""
    
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
            logger.info(f"Initializing The Queen of Thorns story for user {user_id}")
            
            # Step 1: Initialize SF Bay Area preset lore
            lore_result = await SFBayQueenOfThornsPreset.initialize_complete_sf_preset(
                ctx, user_id, conversation_id
            )
            logger.info("SF Bay Area lore initialized")
            
            # Step 2: Load story structure and poems
            from story_templates.moth.the_moth_and_flame import THE_MOTH_AND_FLAME
            await PoemIntegratedStoryLoader.load_story_with_poems(
                THE_MOTH_AND_FLAME, user_id, conversation_id
            )
            logger.info("Story structure and poems loaded")
            
            # Step 3: Create all locations (both story-specific and lore-based)
            location_ids = await QueenOfThornsStoryInitializer._create_all_locations(
                ctx, user_id, conversation_id
            )
            logger.info(f"Created {len(location_ids)} locations")
            
            # Step 4: Create Lilith Ravencroft as Queen of Thorns
            lilith_id = await QueenOfThornsStoryInitializer._create_lilith_ravencroft(
                ctx, user_id, conversation_id
            )
            logger.info(f"Created Lilith Ravencroft (Queen of Thorns) with ID: {lilith_id}")
            
            # Step 5: Create supporting NPCs (both story and network members)
            support_npc_ids = await QueenOfThornsStoryInitializer._create_supporting_npcs(
                ctx, user_id, conversation_id
            )
            logger.info(f"Created {len(support_npc_ids)} supporting NPCs")
            
            # Step 6: Establish relationships and network connections
            await QueenOfThornsStoryInitializer._setup_all_relationships(
                ctx, user_id, conversation_id, lilith_id, support_npc_ids
            )
            logger.info("Relationships and network connections established")
            
            # Step 7: Initialize story state and tracking
            await QueenOfThornsStoryInitializer._initialize_story_state(
                ctx, user_id, conversation_id, lilith_id
            )
            logger.info("Story state initialized")
            
            # Step 8: Set up special mechanics (masks, three words, network systems)
            await QueenOfThornsStoryInitializer._setup_special_mechanics(
                ctx, user_id, conversation_id, lilith_id
            )
            logger.info("Special mechanics configured")
            
            # Step 9: Initialize shadow network systems
            await QueenOfThornsStoryInitializer._initialize_network_systems(
                ctx, user_id, conversation_id, lilith_id
            )
            logger.info("Shadow network systems initialized")
            
            # Step 10: Create initial atmosphere
            await QueenOfThornsStoryInitializer._set_initial_atmosphere(
                ctx, user_id, conversation_id
            )
            logger.info("Initial atmosphere set")
            
            return {
                "status": "success",
                "story_id": THE_MOTH_AND_FLAME.id,
                "main_npc_id": lilith_id,
                "support_npc_ids": support_npc_ids,
                "location_ids": location_ids,
                "network_initialized": True,
                "message": "The Queen of Thorns story initialized successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize story: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "message": "Failed to initialize The Queen of Thorns story"
            }
    
    @staticmethod
    async def _create_lilith_ravencroft(ctx, user_id: int, conversation_id: int) -> int:
        """Create Lilith as the Queen of Thorns with all her complexity"""
        
        try:
            # First, use canonical function to find or create
            async with get_db_connection_context() as conn:
                lilith_id = await canon.find_or_create_npc(
                    ctx, conn,
                    npc_name=LILITH_RAVENCROFT["name"],
                    role="The Queen of Thorns",
                    affiliations=[
                        "Velvet Sanctum", 
                        "The Shadow Network", 
                        "The Rose Council (Leader)",
                        "Underground Protection Network"
                    ]
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
                    logger.info(f"Updating Lilith (ID: {lilith_id}) with full Queen of Thorns data")
                    
                    # Update her role and description for the new context
                    enhanced_lilith_data = LILITH_RAVENCROFT.copy()
                    enhanced_lilith_data["role"] = "The Queen of Thorns / Shadow Network Leader"
                    enhanced_lilith_data["backstory"]["current_status"] = (
                        "Rules from multiple thrones - Velvet Sanctum's obsidian seat, "
                        "Rose Council meetings in Pacific Heights mansions, charity galas "
                        "where thorns hide beneath silk. Known to outsiders as head of "
                        "'The Rose & Thorn Society' though the network has no official name. "
                        "Transforms predators into protectors, saves those who need saving."
                    )
                    
                    # Use PresetNPCHandler to add all the detailed data
                    await PresetNPCHandler.create_detailed_npc(ctx, enhanced_lilith_data, {
                        "story_context": "queen_of_thorns",
                        "is_main_character": True,
                        "network_role": "supreme_authority"
                    })
                else:
                    logger.info(f"Lilith already exists (ID: {lilith_id}), adding network properties only")
                    
                    # Just add special properties that won't conflict
                    await QueenOfThornsStoryInitializer._add_lilith_network_properties(
                        ctx, lilith_id, LILITH_RAVENCROFT, user_id, conversation_id
                    )
                    
                    # Ensure memory system is initialized
                    memory_count = await conn.fetchval("""
                        SELECT COUNT(*) FROM NPCMemories
                        WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
                    """, user_id, conversation_id, lilith_id)
                    
                    if memory_count < 3:
                        await QueenOfThornsStoryInitializer._initialize_lilith_memory_system(
                            user_id, conversation_id, lilith_id, LILITH_RAVENCROFT
                        )
            
            return lilith_id
            
        except Exception as e:
            logger.error(f"Failed to create Lilith Ravencroft: {e}", exc_info=True)
            raise
    
    @staticmethod
    async def _create_supporting_npcs(ctx, user_id: int, conversation_id: int) -> List[int]:
        """Create all supporting NPCs including network members"""
        
        npc_ids = []
        
        # Updated NPCs to fit the Queen of Thorns / SF Bay Area context
        npcs_to_create = [
            {
                "name": "Marcus Sterling",
                "role": "Devoted Submissive / Former Tech CEO",
                "affiliations": ["Velvet Sanctum", "The Queen's Inner Circle", "Transformed Executive"],
                "data": {
                    "id": "marcus_sterling",
                    "name": "Marcus Sterling",
                    "sex": "male",
                    "age": 45,
                    "physical_description": (
                        "A once-powerful Silicon Valley CEO now wholly devoted to his Queen. His "
                        "Patagonia vest can't hide the collar marks on his neck or the desperate "
                        "hunger in his eyes. Silver hair, always perfectly styled, as if maintaining "
                        "his appearance might earn him an extra moment of her attention. Kneels with "
                        "practiced grace. The rose tattoo on his wrist marks him as transformed."
                    ),
                    "personality": {
                        "personality_traits": [
                            "obsessively_devoted", "jealous", "broken", "wealthy",
                            "desperately_needy", "completely_submissive", "worship_focused",
                            "funding_the_network", "transformed_predator"
                        ],
                        "likes": [
                            "serving the Queen", "public humiliation", "being used as example",
                            "funding safehouses", "kneeling", "being ignored (it's still attention)",
                            "making amends through submission"
                        ],
                        "dislikes": [
                            "new submissives", "being forgotten", "others getting attention",
                            "leaving the sanctum", "his old predatory self", "independent thought"
                        ],
                        "hobbies": [
                            "collecting Queen's used items", "writing devotional poetry",
                            "practicing perfect service", "funding network operations"
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
                        "archetype_names": ["Transformed Executive", "Devoted Submissive"],
                        "archetype_summary": "A predator transformed into a protector through submission",
                        "archetype_extras_summary": "Represents the network's transformation power"
                    },
                    "schedule": {
                        "Monday": {"Evening": "Waiting outside Velvet Sanctum", "Night": "Kneeling in main chamber"},
                        "Tuesday": {"All Day": "Preparing offerings and donations"},
                        "Wednesday": {"Evening": "Early arrival at Sanctum", "Night": "Public transformation display"},
                        "Thursday": {"Evening": "Private session if permitted", "Night": "Cleaning duties"},
                        "Friday": {"Evening": "First to arrive", "Night": "Demonstration subject", "Late Night": "Funding transfers"},
                        "Saturday": {"All Day": "Living at the Sanctum's edges"},
                        "Sunday": {"All Day": "Writing checks to safehouses"}
                    },
                    "memories": [
                        "The first time She noticed me, I was just another tech bro with too much "
                        "money and wandering hands. The Rose email came with evidence of my sins. "
                        "'You have a choice,' She said when I arrived trembling. 'Prison or transformation.' "
                        "I chose Her collar. Now my billions build safehouses instead of buying silence.",
                        
                        "I gave up my CEO position at the fintech startup, my marriage, my identity - "
                        "all for the privilege of kneeling at Her feet. My ex-wife thinks I joined a cult. "
                        "She's not wrong. But this cult transforms monsters into men. I fund what I once "
                        "would have exploited. That is my penance and my joy."
                    ],
                    "current_location": "Outside Velvet Sanctum",
                    "affiliations": ["Velvet Sanctum", "The Queen's Inner Circle", "Network Funder"],
                    "introduced": False
                }
            },
            {
                "name": "Sarah Chen",
                "role": "Trafficking Survivor / Safehouse Coordinator",
                "affiliations": ["The Underground Network", "Queen's Saved", "Safehouse Network"],
                "data": {
                    "id": "sarah_chen",
                    "name": "Sarah Chen",
                    "sex": "female",
                    "age": 22,
                    "physical_description": (
                        "A young woman still healing from trauma but growing stronger. Asian features "
                        "marked by a wariness that never quite leaves her eyes. Thin from years of "
                        "deprivation but learning to take up space again. Dresses in layers, always "
                        "ready to run. A rose tattoo on her wrist - the mark of those saved by the "
                        "Queen of Thorns. Now helps run the Marina safehouse."
                    ),
                    "personality": {
                        "personality_traits": [
                            "traumatized_but_healing", "grateful", "suspicious", "protective",
                            "protective_of_others", "alert", "slowly_trusting", "network_coordinator"
                        ],
                        "likes": [
                            "feeling safe", "helping other survivors", "quiet spaces",
                            "the Queen's protection", "learning self-defense", "tea",
                            "seeing predators transformed", "the network's reach"
                        ],
                        "dislikes": [
                            "sudden movements", "locked doors", "vans", "older men",
                            "being touched without warning", "loud voices", "feeling trapped",
                            "Viktor Kozlov and his people"
                        ],
                        "hobbies": [
                            "counseling other survivors", "self-defense training",
                            "coordinating safehouse operations", "studying psychology"
                        ]
                    },
                    "stats": {
                        "dominance": 30,
                        "cruelty": 5,
                        "affection": 70,
                        "trust": 40,
                        "respect": 95,
                        "intensity": 60
                    },
                    "archetypes": {
                        "archetype_names": ["Trafficking Survivor", "Rising Phoenix"],
                        "archetype_summary": "A survivor becoming a protector",
                        "archetype_extras_summary": "Represents those the Queen saves and empowers"
                    },
                    "schedule": {
                        "Monday": {"Morning": "Marina Safehouse", "Afternoon": "Therapy", "Evening": "New arrival orientation"},
                        "Tuesday": {"Morning": "Self-defense at Eastern Rose dojo", "Afternoon": "Safehouse admin"},
                        "Wednesday": {"All Day": "Counseling other survivors", "Evening": "Network coordination meeting"},
                        "Thursday": {"Morning": "Job training program", "Afternoon": "Safehouse", "Evening": "Underground Railroad operations"},
                        "Friday": {"Evening": "Support group facilitation"},
                        "Saturday": {"Varies": "Helping with extraction operations"},
                        "Sunday": {"All Day": "Rest and recovery"}
                    },
                    "memories": [
                        "I was seventeen when Kozlov's people took me. Promised a restaurant job in "
                        "the city. The Queen of Thorns found me three years later, half-dead in a "
                        "Tenderloin basement. She burned their whole operation down. 'No one else,' "
                        "she whispered as she carried me out. 'No one else.' The thorns on her arms "
                        "were covered in blood - theirs and hers.",
                        
                        "Sometimes I see her at the safehouses, checking on us. She's different there - "
                        "no masks, no performance. Just a woman who understands our pain because she "
                        "lived it. She taught me that surviving isn't enough. We deserve to bloom. Now "
                        "I help others find their way from darkness to the garden."
                    ],
                    "current_location": "Marina Safehouse - Common Area",
                    "affiliations": ["The Underground Network", "Queen's Saved", "Safehouse Coordinator"],
                    "introduced": False
                }
            },
            {
                "name": "Viktor Kozlov",
                "role": "Human Trafficker / Eastern European Crime Boss",
                "affiliations": ["International Shadows", "The Opposition"],
                "data": {
                    "id": "viktor_kozlov",
                    "name": "Viktor Kozlov",
                    "sex": "male", 
                    "age": 48,
                    "physical_description": (
                        "A mountain of barely restrained violence. Russian accent thick as his scarred "
                        "knuckles. Prison tattoos peek from under expensive shirts that can't hide what "
                        "he is. Dead eyes that see women as commodities. Smells of cologne and cruelty. "
                        "Has burn scars from when the Queen torched his warehouse."
                    ),
                    "personality": {
                        "personality_traits": [
                            "violent", "calculating", "misogynistic", "predatory",
                            "intelligent", "ruthless", "well_connected", "vengeful"
                        ],
                        "likes": [
                            "power over others", "breaking the strong", "money",
                            "fear in others' eyes", "the trafficking trade", "violence",
                            "hunting the Queen of Thorns"
                        ],
                        "dislikes": [
                            "the Queen of Thorns", "losing merchandise", "police attention",
                            "women with power", "being challenged", "witnesses",
                            "the Rose & Thorn Society", "transformed executives who stop paying"
                        ],
                        "hobbies": [
                            "expanding his network", "intimidation", "counting losses",
                            "planning the Queen's downfall", "corrupting officials"
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
                        "archetype_summary": "The evil that the Queen fights against",
                        "archetype_extras_summary": "Represents the darkness she escaped and battles"
                    },
                    "schedule": {
                        "Monday": {"Night": "Port operations"},
                        "Tuesday": {"Night": "Moving 'merchandise'"},
                        "Wednesday": {"Evening": "Meeting with corrupt port officials"},
                        "Thursday": {"Night": "Checking on operations"},
                        "Friday": {"Night": "Marina district hunting", "Late Night": "Violence"},
                        "Saturday": {"Night": "Expanding territory"},
                        "Sunday": {"Unknown": "Planning strikes against the network"}
                    },
                    "memories": [
                        "The Queen of Thorns cost me five million when she burned my Bayview warehouse. "
                        "But worse, she gave the merchandise hope. Started this 'Underground Railroad' "
                        "nonsense. Every month more shipments disappear. Soon I will clip those thorn "
                        "wings and remind her what happens to little girls who forget their place.",
                        
                        "I should have killed her five years ago when my men had her surrounded. Thought "
                        "she was just another runaway playing vigilante. Now she has judges, cops, even "
                        "some of my buyers wearing her collar. But every rose can be cut at the stem. "
                        "And when I find where she hides, I'll make an example that echoes to Moscow."
                    ],
                    "current_location": "Unknown - Port District",
                    "affiliations": ["International Shadows", "Eastern European Syndicate"],
                    "introduced": False
                }
            },
            {
                "name": "Victoria Chen",
                "role": "VC Partner / Rose Council Member",
                "affiliations": ["Rose Council", "Tech Elite", "The Network"],
                "data": {
                    "id": "victoria_chen",
                    "name": "Victoria Chen", 
                    "sex": "female",
                    "age": 35,
                    "physical_description": (
                        "Power wrapped in a perfectly tailored suit. MIT graduate who learned that "
                        "real disruption happens in dungeons, not boardrooms. Drives a sensible Tesla "
                        "to her Noe Valley home where the basement serves other purposes. Always wears "
                        "a vintage rose gold watch - a gift from the Queen."
                    ),
                    "personality": {
                        "personality_traits": [
                            "brilliant", "calculating", "secretly_dominant", "protective",
                            "network_loyal", "predator_identifier", "transformer_of_men"
                        ],
                        "likes": [
                            "identifying problematic founders", "behavioral modification",
                            "the Queen's vision", "power through transformation",
                            "her rose garden", "Monday meetings"
                        ],
                        "dislikes": [
                            "unchecked tech bros", "traditional VC culture",
                            "those who won't transform", "threats to the network"
                        ],
                        "hobbies": [
                            "Cultivating her rose garden", "Executive coaching",
                            "Collecting kompromat", "Training new dominants"
                        ]
                    },
                    "stats": {
                        "dominance": 85,
                        "cruelty": 60,
                        "affection": 40,
                        "trust": 70,
                        "respect": 90,
                        "intensity": 75
                    },
                    "current_location": "555 California Street - Office",
                    "affiliations": ["Rose Council", "Sequoia Capital", "The Network"],
                    "introduced": False
                }
            },
            {
                "name": "Judge Elizabeth Thornfield",
                "role": "Federal Judge / Rose Council Member",
                "affiliations": ["Rose Council", "Legal System", "The Network"],
                "data": {
                    "id": "judge_thornfield",
                    "name": "Judge Elizabeth Thornfield",
                    "sex": "female",
                    "age": 52,
                    "physical_description": (
                        "Authority incarnate in judicial robes. The Thornfield name carries weight "
                        "in old San Francisco. Harvard Law couldn't teach what she learned in "
                        "private chambers. Her gavel has decided more than legal cases."
                    ),
                    "personality": {
                        "personality_traits": [
                            "just", "secretly_ruthless", "network_architect",
                            "protective_of_vulnerable", "alternative_justice"
                        ],
                        "likes": [
                            "Creative sentencing", "The network's growth",
                            "Protecting trafficking victims", "Thursday book clubs"
                        ],
                        "dislikes": [
                            "Mandatory minimums", "Violent predators",
                            "Federal interference", "Those who break sanctuary"
                        ]
                    },
                    "stats": {
                        "dominance": 80,
                        "cruelty": 50,
                        "affection": 60,
                        "trust": 85,
                        "respect": 95,
                        "intensity": 70
                    },
                    "current_location": "Federal Building - Chambers",
                    "affiliations": ["Rose Council", "Federal Judiciary", "The Network"],
                    "introduced": False
                }
            }
        ]
        
        # Minor NPCs at the Sanctum
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
                    "archetype": "Seeking Submissive",
                    "physical_description": "A lawyer by day seeking absolution in submission by night",
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
                "role": "New Petitioner",
                "affiliations": ["Velvet Sanctum"],
                "data": {
                    "id": "amanda_ross",
                    "name": "Amanda Ross",
                    "sex": "female",
                    "age": 26,
                    "archetype": "Curious Newcomer",
                    "physical_description": "Tech worker exploring power dynamics for the first time",
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
                    "physical_description": "Once held the Queen's complete attention, now watches from shadows",
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
                                ctx, npc_def["data"], {"story_context": "queen_of_thorns"}
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
        """Create Lilith's foundational memories as Queen of Thorns"""
        
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
    
    @staticmethod
    async def _add_lilith_network_properties(
        ctx, npc_id: int, lilith_data: Dict, user_id: int, conversation_id: int
    ):
        """Add Lilith's network-specific properties through canonical updates"""
        
        canon_ctx = type('CanonicalContext', (), {
            'user_id': user_id,
            'conversation_id': conversation_id
        })()
        
        async with get_db_connection_context() as conn:
            # Add network authority markers
            await canon.update_entity_canonically(
                canon_ctx, conn, "NPCStats", npc_id,
                {
                    "network_role": "supreme_authority",
                    "network_knowledge": json.dumps({
                        "organization_names": [
                            "the network", "the garden", "what outsiders call Rose & Thorn"
                        ],
                        "leadership_mystery": "Identity deliberately obscured",
                        "rose_council": "Seven senior dominants she commands",
                        "geographic_reach": "Bay Area absolute, influence spreading"
                    })
                },
                "Adding Queen of Thorns network authority"
            )
            
            # Add operational knowledge
            await canon.update_entity_canonically(
                canon_ctx, conn, "NPCStats", npc_id,
                {
                    "operational_knowledge": json.dumps({
                        "transformation_pipeline": "Predators to protectors",
                        "safehouse_network": "Marina, Mission, Tenderloin nodes",
                        "funding_sources": "Transformed executives, guilt payments",
                        "communication_methods": "Rose signals, encrypted channels",
                        "enforcement_arm": "Thorns who handle problems"
                    })
                },
                "Setting network operational knowledge"
            )
            
            # Add key relationships to network figures
            await canon.update_entity_canonically(
                canon_ctx, conn, "NPCStats", npc_id,
                {
                    "network_relationships": json.dumps({
                        "rose_council": ["Victoria Chen", "Judge Thornfield", "5 others"],
                        "key_thorns": "Classified by need-to-know",
                        "protected": "Trafficking survivors across Bay Area",
                        "enemies": ["Viktor Kozlov", "International trafficking rings"],
                        "compromised": "Half of Silicon Valley C-suites"
                    })
                },
                "Establishing network relationship web"
            )
            
            # Add transformation statistics
            await canon.update_entity_canonically(
                canon_ctx, conn, "NPCStats", npc_id,
                {
                    "transformation_record": json.dumps({
                        "executives_transformed": 47,
                        "trafficking_victims_saved": 312,
                        "safehouses_established": 17,
                        "annual_funding_secured": "$50-100M",
                        "success_rate": "87% permanent behavioral change"
                    })
                },
                "Recording transformation achievements"
            )
            
            # Update dialogue patterns for network context
            dialogue_patterns = lilith_data["dialogue_patterns"].copy()
            dialogue_patterns["network_references"] = [
                "The garden tends itself",
                "Thorns protect roses",
                "What outsiders call the Rose & Thorn Society",
                "The network has no name but infinite reach",
                "Monday at 3 PM, decisions are made"
            ]
            
            await canon.update_entity_canonically(
                canon_ctx, conn, "NPCStats", npc_id,
                {
                    "dialogue_patterns": json.dumps(dialogue_patterns),
                    "code_phrases": json.dumps([
                        "interesting energy", "needs pruning", "ready to bloom",
                        "the garden grows", "thorns have purpose"
                    ])
                },
                "Adding network-specific dialogue patterns"
            )
            
            # Add location associations
            await canon.update_entity_canonically(
                canon_ctx, conn, "NPCStats", npc_id,
                {
                    "location_associations": json.dumps({
                        "velvet_sanctum": "Public throne",
                        "rose_garden_cafe": "Recruitment observations",
                        "montenegro_gallery": "Identify targets through art",
                        "private_chambers": "True self revealed",
                        "inner_garden": "Most secret sanctuary",
                        "multiple_safehouses": "Checking on the saved"
                    })
                },
                "Setting location associations"
            )
    
    @staticmethod
    async def _initialize_lilith_memory_system(
        user_id: int, conversation_id: int, npc_id: int, lilith_data: Dict
    ):
        """Set up Lilith's specialized memory system for Queen of Thorns role"""
        
        memory_system = await MemorySystem.get_instance(user_id, conversation_id)
        
        # Store initial memories
        initial_memories = QueenOfThornsStoryInitializer._create_lilith_memories()
        for memory_text in initial_memories:
            await memory_system.remember(
                entity_type="npc",
                entity_id=npc_id,
                memory_text=memory_text,
                importance="high",
                emotional=True,
                tags=["backstory", "core_memory", "trauma", "motivation", "network_foundation"]
            )
        
        # Create memory schemas specific to her character
        await memory_system.generate_schemas(
            entity_type="npc",
            entity_id=npc_id
        )
        
        # Set up trauma keywords for flashback system
        trauma_keywords = [
            "disappear", "goodbye", "leave", "forever", "always",
            "promise", "trafficking", "van", "fifteen", "scars",
            "Kozlov", "abandonment", "FBI", "exposure", "loved"
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
        """Create all story locations including network sites"""
        
        canon_ctx = type('CanonicalContext', (), {
            'user_id': user_id,
            'conversation_id': conversation_id
        })()
        
        location_ids = []
        
        # Story-specific locations that complement the lore
        locations = [
            {
                "name": "Velvet Sanctum",
                "description": (
                    "An underground temple of transformation hidden beneath the city. Descending "
                    "from the innocent boutique above, the air grows thick with incense and "
                    "anticipation. Red velvet drapes, candlelit alcoves, and an obsidian throne "
                    "where the Queen holds court. Every surface whispers of power exchanged, "
                    "every shadow holds a secret. Here, predators learn to kneel and roses "
                    "grow thorns."
                ),
                "location_type": "bdsm_club",
                "areas": [
                    "Main Chamber", "Throne Room", "Private Booths", 
                    "Transformation Chambers", "Preparation Room", "Queen's Private Office"
                ],
                "district": "SoMa",
                "network_role": "Public face of the Queen's power"
            },
            {
                "name": "Empty Sanctum",
                "description": (
                    "The same space when dawn breaks and crowds disperse. Without the performance, "
                    "it becomes a melancholy cathedral. Candles burn low, their wax pooling like "
                    "frozen tears. The throne sits empty, a monument to loneliness. Here, masks "
                    "grow heavy and the goddess becomes mortal again."
                ),
                "location_type": "afterhours_venue",
                "areas": ["Abandoned Stage", "Silent Throne", "Echo Chamber"],
                "district": "SoMa",
                "network_role": "Where the Queen's vulnerability shows"
            },
            {
                "name": "The Queen's Private Chambers",
                "description": (
                    "Behind hidden doors lies her true sanctuary. A Pacific Heights apartment "
                    "that tells two stories: public areas draped in luxury and control, private "
                    "spaces revealing vulnerability. The mask room holds her collection - porcelain "
                    "faces of everyone who promised to stay. Her desk overflows with two lists: "
                    "red ink for the lost, blue for the abandoned."
                ),
                "location_type": "personal_space",
                "areas": [
                    "The Mask Room", "Writing Desk", "Bedroom",
                    "Hidden Safe Room", "Private Garden", "Network Command Center"
                ],
                "district": "Pacific Heights",
                "network_role": "Hidden nerve center"
            },
            {
                "name": "The Rose Garden Café",
                "description": (
                    "A perfectly normal Mission café that serves as the network's softest entry "
                    "point. Lily Chen serves lavender lattes and observes power dynamics. The "
                    "book clubs read between different lines. Tuesday poetry nights encode "
                    "network communications. The back room hosts wine tastings where vintages "
                    "aren't the only thing evaluated."
                ),
                "location_type": "café",
                "areas": [
                    "Main Café", "Reading Nook", "Back Room", "Office"
                ],
                "district": "Mission",
                "network_role": "Recruitment and observation"
            },
            {
                "name": "Marina Safehouse",
                "description": (
                    "A Mediterranean villa overlooking the bay, disguised as an executive women's "
                    "retreat. Here, trafficking survivors heal and transform. The therapeutic "
                    "program includes trauma recovery and optional dominance training. Sarah Chen "
                    "coordinates operations. Gardens grow herbs that heal and thorns that protect."
                ),
                "location_type": "safehouse",
                "areas": [
                    "Common Areas", "Therapy Rooms", "Safe Rooms",
                    "Medical Station", "Training Dojo", "Healing Garden"
                ],
                "district": "Marina",
                "network_role": "Primary recovery center"
            },
            {
                "name": "The Inner Garden",
                "description": (
                    "The Queen's most private sanctuary. Location known only to the Rose Council. "
                    "Some say it's metaphorical, others have seen the thorns. Here she tends both "
                    "roses and those who serve most deeply. The entrance moves, the space exists "
                    "between reality and dream. Where the three words might finally escape."
                ),
                "location_type": "secret_location",
                "areas": ["Unknown - reveals based on trust"],
                "district": "Hidden",
                "network_role": "Ultimate sanctuary"
            },
            {
                "name": "Warehouse District - Kozlov Territory",
                "description": (
                    "Industrial wasteland where shipping containers hold human cargo. Viktor "
                    "Kozlov's operations center before the network strikes. Burn marks on "
                    "certain buildings mark the Queen's victories. A battlefield where the "
                    "Underground Railroad wars with international trafficking."
                ),
                "location_type": "hostile_territory",
                "areas": ["Loading Docks", "Container Yards", "Underground Routes"],
                "district": "Bayview",
                "network_role": "Enemy territory / extraction zone"
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
                        "district": loc.get("district", "Unknown"),
                        "network_role": loc.get("network_role", "Unknown")
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
                    tags=["location", "story_setup", loc["location_type"], "network_infrastructure"]
                )
        
        return location_ids
    
    @staticmethod
    async def _setup_all_relationships(
        ctx, user_id: int, conversation_id: int, 
        lilith_id: int, support_npc_ids: List[int]
    ):
        """Establish all story relationships and network connections"""
        
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
        
        # Establish relationships
        relationships = []
        
        # Marcus Sterling (transformed executive)
        if "Marcus Sterling" in npc_id_map:
            relationships.append({
                "source": lilith_id,
                "target": npc_id_map["Marcus Sterling"],
                "type": "owns",
                "reverse": "owned_by",
                "strength": 95
            })
        
        # Sarah Chen (saved victim)
        if "Sarah Chen" in npc_id_map:
            relationships.append({
                "source": lilith_id,
                "target": npc_id_map["Sarah Chen"],
                "type": "protector",
                "reverse": "protected_by", 
                "strength": 85
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
        
        # Victoria Chen (Rose Council)
        if "Victoria Chen" in npc_id_map:
            relationships.append({
                "source": lilith_id,
                "target": npc_id_map["Victoria Chen"],
                "type": "commands",
                "reverse": "serves",
                "strength": 80
            })
        
        # Judge Thornfield (Rose Council)
        if "Judge Elizabeth Thornfield" in npc_id_map:
            relationships.append({
                "source": lilith_id,
                "target": npc_id_map["Judge Elizabeth Thornfield"],
                "type": "commands",
                "reverse": "serves",
                "strength": 80
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
                    "The night Marcus Sterling received his Rose email, he thought his life was over. "
                    "Evidence of his harassment, his NDAs, his sins. When he arrived at the address "
                    "provided, trembling with fear, the Queen gave him a choice: 'Prison or transformation. "
                    "Destruction or service. Choose.' He chose her collar. Now his billions build "
                    "safehouses instead of buying silence."
                )
            })
        
        if "Sarah Chen" in npc_id_map:
            shared_memories.append({
                "npcs": [lilith_id, npc_id_map["Sarah Chen"]],
                "memory": (
                    "Sarah was barely breathing when the Queen found her in Kozlov's Tenderloin "
                    "basement. As thorns grew from the Queen's anger, she burned the entire operation "
                    "down. 'You're safe now,' she whispered, carrying Sarah to freedom. 'I promise "
                    "you're safe. And I keep my promises to roses.' Sarah now tends other saved "
                    "roses in the Marina safehouse."
                )
            })
        
        if "Viktor Kozlov" in npc_id_map:
            shared_memories.append({
                "npcs": [lilith_id, npc_id_map["Viktor Kozlov"]],
                "memory": (
                    "Five years ago, Kozlov's men had cornered what they thought was just another "
                    "vigilante. 'Little moth,' he laughed, 'playing with fire.' But the Queen of "
                    "Thorns doesn't burn - she incinerates. By dawn, his warehouse was ash, five "
                    "million in 'inventory' freed, and a shadow war declared. He's been hunting "
                    "her identity ever since."
                )
            })
        
        if "Victoria Chen" in npc_id_map:
            shared_memories.append({
                "npcs": [lilith_id, npc_id_map["Victoria Chen"]],
                "memory": (
                    "Victoria was the first Rose Council member the Queen personally selected. "
                    "'You see predators in pitch meetings,' the Queen observed. 'You could do "
                    "more than refuse their funding.' Now Victoria transforms tech bros in her "
                    "Noe Valley basement, and Sequoia Capital unknowingly funds a revolution."
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
                    tags=["shared_memory", "relationship", "network_history"]
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
                    "network_awareness": 0,
                    "queen_identity_suspected": False,
                    "network_identity_revealed": False,
                    "three_words_spoken": False,
                    "dual_identity_revealed": False,
                    "player_role": "unknown",
                    "player_alignment": "neutral",
                    "emotional_intensity": 0,
                    "sessions_completed": 0,
                    "vulnerability_witnessed": 0,
                    "promises_made": [],
                    "transformations_witnessed": 0,
                    "rose_council_awareness": 0,
                    "safehouse_visits": 0,
                    "kozlov_threat_level": 0
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
                    "tone": "noir_gothic",
                    "lighting": "candlelit_shadows", 
                    "sound": "distant_gothic_electronica",
                    "scent": "roses_leather_incense",
                    "feeling": "anticipation_and_hidden_power",
                    "network_presence": "invisible_but_everywhere"
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
                    "dominance": 0,
                    "devotion": 0,
                    "curiosity": 50,
                    "fear": 20,
                    "arousal": 0,
                    "pain_tolerance": 0,
                    "trust_given": 0,
                    "promises_kept": 0,
                    "vulnerability_shown": 0,
                    "moth_nature": 50,  # 0 = predator, 100 = protector
                    "network_loyalty": 0,
                    "transformation_progress": 0,
                    "garden_knowledge": 0,
                    "thorn_bearer_potential": 0
                })
            )
    
    @staticmethod
    async def _setup_special_mechanics(
        ctx, user_id: int, conversation_id: int, lilith_id: int
    ):
        """Configure special story mechanics including network systems"""
        
        async with get_db_connection_context() as conn:
            # Mask system remains the same
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
            
            # Poetry triggers with network themes
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
                        {"emotion": "affection", "chance": 0.5},
                        {"emotion": "network_protection", "chance": 0.9}
                    ],
                    "poetry_used": [],
                    "understanding_tracker": {
                        "attempts": 0,
                        "successes": 0,
                        "trust_impact": 5
                    }
                })
            )
            
            # Three words mechanic
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
            
            # Network revelation mechanic
            await conn.execute(
                """
                INSERT INTO npc_special_mechanics (
                    user_id, conversation_id, npc_id, mechanic_type, mechanic_data
                )
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (user_id, conversation_id, npc_id, mechanic_type)
                DO UPDATE SET mechanic_data = EXCLUDED.mechanic_data
                """,
                user_id, conversation_id, lilith_id, "network_revelation",
                json.dumps({
                    "identity_hints_given": [],
                    "revelation_triggers": {
                        "trust": 70,
                        "witnessed_transformation": True,
                        "helped_victim": True,
                        "location_based": ["safehouse", "inner_garden"]
                    },
                    "revelation_style": None,  # "discovered", "confessed", "demonstrated"
                    "post_revelation_dynamic": None
                })
            )
            
            # Transformation witness mechanic
            await conn.execute(
                """
                INSERT INTO npc_special_mechanics (
                    user_id, conversation_id, npc_id, mechanic_type, mechanic_data
                )
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (user_id, conversation_id, npc_id, mechanic_type)
                DO UPDATE SET mechanic_data = EXCLUDED.mechanic_data
                """,
                user_id, conversation_id, lilith_id, "transformation_system",
                json.dumps({
                    "witnessed_transformations": [],
                    "player_reaction_history": [],
                    "transformation_methods": [
                        "behavioral_modification",
                        "power_exchange_therapy",
                        "submission_training",
                        "identity_reconstruction"
                    ],
                    "trust_required_to_witness": 40,
                    "trust_required_to_assist": 70
                })
            )
    
    @staticmethod
    async def _initialize_network_systems(
        ctx, user_id: int, conversation_id: int, lilith_id: int
    ):
        """Initialize the shadow network infrastructure"""
        
        async with get_db_connection_context() as conn:
            # Create network state tracking
            await conn.execute(
                """
                INSERT INTO network_state (
                    user_id, conversation_id, network_data
                )
                VALUES ($1, $2, $3)
                ON CONFLICT (user_id, conversation_id)
                DO UPDATE SET network_data = EXCLUDED.network_data
                """,
                user_id, conversation_id,
                json.dumps({
                    "organization_names": [
                        "the network",
                        "the garden",
                        "Rose & Thorn Society (outsider name)",
                        "The Thorn Garden (media name)"
                    ],
                    "structure": {
                        "queen_of_thorns": lilith_id,
                        "rose_council": ["Victoria Chen", "Judge Thornfield", "5 others"],
                        "regional_thorns": {},
                        "gardeners": [],
                        "thorns": [],
                        "roses": [],
                        "seedlings": []
                    },
                    "operations": {
                        "safehouses": ["Marina", "Mission", "Tenderloin"],
                        "transformation_centers": ["Velvet Sanctum", "Private locations"],
                        "funding_sources": ["Transformed executives", "Legitimate businesses"],
                        "communication_hubs": ["Rose Garden Café", "Gallery networks"]
                    },
                    "threats": {
                        "viktor_kozlov": "active",
                        "federal_investigation": "dormant",
                        "media_exposure": "managed",
                        "internal_schisms": "none"
                    },
                    "statistics": {
                        "members_approximate": "unknown by design",
                        "saved_this_year": 47,
                        "transformed_this_year": 12,
                        "annual_budget": "$50-100M",
                        "geographic_reach": "Bay Area primary, influence spreading"
                    }
                })
            )
            
            # Create Rose communication system
            await conn.execute(
                """
                INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (user_id, conversation_id, key) 
                DO UPDATE SET value = EXCLUDED.value
                """,
                user_id, conversation_id, "rose_signals",
                json.dumps({
                    "active_signals": [],
                    "signal_meanings": {
                        "single_red_rose": "danger",
                        "white_rose": "all_clear",
                        "black_rose": "transformation_needed",
                        "rose_petals": "meeting_called",
                        "thorns_displayed": "protection_activated"
                    }
                })
            )
    
    @staticmethod
    async def _set_initial_atmosphere(ctx, user_id: int, conversation_id: int):
        """Set the initial story atmosphere"""
        
        # Create atmospheric introduction
        intro_message = {
            "type": "story_introduction",
            "content": (
                "San Francisco after midnight breathes differently. In the SoMa underground, "
                "where Silicon Valley's shadows dance with older powers, you've heard whispers "
                "of the Velvet Sanctum. They say a Queen holds court there - beautiful, terrible, "
                "offering transformation through submission.\n\n"
                
                "But the whispers speak of more than just a dominatrix. They mention roses that "
                "grow in darkness, thorns that protect the vulnerable, a network without a name "
                "that reaches into boardrooms and basements alike. Some call it the Rose & Thorn "
                "Society. Others say it has no name at all.\n\n"
                
                "You stand before an unmarked door beneath a boutique that sells overpriced "
                "leather goods. The bouncer - scarred, silent, seeing too much - evaluates you "
                "with eyes that catalog more than appearance. Finally, he steps aside.\n\n"
                
                "'The Queen is holding court tonight,' he says. 'Try not to stare. She notices "
                "everything. And if you're lucky... she might notice you.'\n\n"
                
                "As you descend the stairs, each step takes you deeper into a world where power "
                "flows in directions Stanford Business School never imagined. The air grows thick "
                "with incense, leather, and the copper scent of transformation.\n\n"
                
                "Welcome to the beginning of your education in thorns."
            ),
            "atmosphere": {
                "visual": "Candlelight painting stories on velvet walls",
                "auditory": "Gothic electronica mixing with whispered negotiations",
                "olfactory": "Roses, leather, incense, and the metal tang of change",
                "emotional": "Anticipation laced with the delicious unknown",
                "hidden": "The sense that you're entering something much larger than a club"
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
            tags=["story_start", "first_impression", "atmosphere", "network_introduction"]
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

# Keep the old name for compatibility but update the implementation
MothFlameStoryInitializer = QueenOfThornsStoryInitializer

# Also keep story progression but update references
class QueenOfThornsStoryProgression:
    """Handles story progression and beat triggers for Queen of Thorns"""
    
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
                                key=lambda b: (QueenOfThornsStoryProgression._get_beat_act(b), 
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
                if await QueenOfThornsStoryProgression._check_single_beat_conditions(
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
                
                # Times visited location (updated for network locations)
                elif condition == "times_visited_sanctum":
                    visit_count = story_flags.get("sanctum_visits", 0)
                    if visit_count < value:
                        return False
                elif condition == "times_visited_safehouse":
                    visit_count = story_flags.get("safehouse_visits", 0)
                    if visit_count < value:
                        return False
                
                # Network awareness level (NEW)
                elif condition == "network_awareness":
                    awareness = story_flags.get("network_awareness", 0)
                    if awareness < value:
                        return False
                
                # Rose Council awareness (NEW)
                elif condition == "rose_council_awareness":
                    awareness = story_flags.get("rose_council_awareness", 0)
                    if awareness < value:
                        return False
                
                # Transformations witnessed (NEW)
                elif condition == "transformations_witnessed":
                    count = story_flags.get("transformations_witnessed", 0)
                    if count < value:
                        return False
                
                # Network identity revealed (NEW)
                elif condition == "network_identity_revealed" and value:
                    if not story_flags.get("network_identity_revealed", False):
                        return False
                
                # Queen identity suspected (NEW)
                elif condition == "queen_identity_suspected" and value:
                    if not story_flags.get("queen_identity_suspected", False):
                        return False
                
                # Met Rose Council member (NEW)
                elif condition == "met_rose_council_member" and value:
                    if not story_flags.get("met_rose_council_member", False):
                        return False
                
                # Helped save trafficking victim (NEW)
                elif condition == "helped_save_victim" and value:
                    if not story_flags.get("helped_save_victim", False):
                        return False
                
                # Kozlov threat level (NEW)
                elif condition == "kozlov_threat_level":
                    threat = story_flags.get("kozlov_threat_level", 0)
                    if threat < value:
                        return False
                
                # NPC awareness level (enhanced for Queen of Thorns)
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
                
                # Player watched Queen's transformation session
                elif condition == "watched_transformation" and value:
                    if not story_flags.get("watched_transformation", False):
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
                
                # Has item (updated for network items)
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
                
                # Relationship requirements (enhanced for Queen of Thorns)
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
                
                # Network test passed (NEW)
                elif condition == "network_test_passed" and value:
                    if not story_flags.get("passed_network_test", False):
                        return False
                
                # Player alignment (NEW)
                elif condition == "player_alignment":
                    alignment = story_flags.get("player_alignment", "neutral")
                    if alignment != value:
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
                
                # Garden knowledge level (NEW)
                elif condition == "garden_knowledge":
                    knowledge = story_flags.get("garden_knowledge", 0)
                    if knowledge < value:
                        return False
                
                # Devotion level
                elif condition == "devotion":
                    player_devotion = await conn.fetchval(
                        """
                        SELECT stats->>'devotion' as devotion 
                        FROM player_story_stats
                        WHERE user_id = $1 AND conversation_id = $2 
                        AND story_id = 'the_moth_and_flame'
                        """,
                        user_id, conversation_id
                    )
                    
                    if not player_devotion or int(player_devotion) < value.get("min", 0):
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
                
                # Rose signal understood (NEW)
                elif condition == "rose_signal_understood" and value:
                    if not story_flags.get("understands_rose_signals", False):
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
                outcomes_applied = await QueenOfThornsStoryProgression._apply_beat_outcomes(
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
                
                # Network awareness changes (NEW)
                elif outcome_type == "network_awareness":
                    current = story_flags.get("network_awareness", 0)
                    if isinstance(outcome_data, str) and outcome_data.startswith(('+', '-')):
                        change = int(outcome_data)
                        story_flags["network_awareness"] = max(0, min(100, current + change))
                    else:
                        story_flags["network_awareness"] = outcome_data
                    
                    applied_outcomes["network_awareness"] = outcome_data
                
                # Rose Council awareness (NEW)
                elif outcome_type == "rose_council_awareness":
                    current = story_flags.get("rose_council_awareness", 0)
                    if isinstance(outcome_data, str) and outcome_data.startswith(('+', '-')):
                        change = int(outcome_data)
                        story_flags["rose_council_awareness"] = max(0, min(100, current + change))
                    else:
                        story_flags["rose_council_awareness"] = outcome_data
                    
                    applied_outcomes["rose_council_awareness"] = outcome_data
                
                # Transformation witnessed (NEW)
                elif outcome_type == "transformation_witnessed":
                    story_flags["transformations_witnessed"] = story_flags.get("transformations_witnessed", 0) + 1
                    story_flags["watched_transformation"] = True
                    
                    # Store details about the transformation
                    witnessed_list = story_flags.get("witnessed_transformations", [])
                    witnessed_list.append({
                        "subject": outcome_data.get("subject", "unnamed executive"),
                        "method": outcome_data.get("method", "behavioral modification"),
                        "timestamp": datetime.now().isoformat()
                    })
                    story_flags["witnessed_transformations"] = witnessed_list[-5:]  # Keep last 5
                    
                    applied_outcomes["transformation_witnessed"] = outcome_data
                
                # Met Rose Council member (NEW)
                elif outcome_type == "met_rose_council_member":
                    story_flags["met_rose_council_member"] = True
                    council_members_met = story_flags.get("rose_council_members_met", [])
                    if outcome_data not in council_members_met:
                        council_members_met.append(outcome_data)
                        story_flags["rose_council_members_met"] = council_members_met
                    
                    applied_outcomes["met_rose_council_member"] = outcome_data
                
                # Network role offered (NEW)
                elif outcome_type == "network_role_offered":
                    story_flags["network_role_offered"] = outcome_data
                    story_flags["player_network_status"] = "recruit"
                    
                    applied_outcomes["network_role_offered"] = outcome_data
                
                # Safehouse access granted (NEW)
                elif outcome_type == "safehouse_access":
                    safehouse_access = story_flags.get("safehouse_access", [])
                    if outcome_data not in safehouse_access:
                        safehouse_access.append(outcome_data)
                        story_flags["safehouse_access"] = safehouse_access
                    
                    applied_outcomes["safehouse_access"] = outcome_data
                
                # Kozlov threat increase (NEW)
                elif outcome_type == "kozlov_threat":
                    current = story_flags.get("kozlov_threat_level", 0)
                    if isinstance(outcome_data, str) and outcome_data.startswith(('+', '-')):
                        change = int(outcome_data)
                        story_flags["kozlov_threat_level"] = max(0, min(100, current + change))
                    else:
                        story_flags["kozlov_threat_level"] = outcome_data
                    
                    applied_outcomes["kozlov_threat"] = outcome_data
                
                # Rose signals learned (NEW)
                elif outcome_type == "rose_signal_learned":
                    signals_known = story_flags.get("rose_signals_known", [])
                    if outcome_data not in signals_known:
                        signals_known.append(outcome_data)
                        story_flags["rose_signals_known"] = signals_known
                    if len(signals_known) >= 3:
                        story_flags["understands_rose_signals"] = True
                    
                    applied_outcomes["rose_signal_learned"] = outcome_data
                
                # Knowledge/secrets/facts gained
                elif outcome_type in ["knowledge_gained", "learned_fact", "learned_truth", 
                                     "learned_secret", "discovered_secret", "network_secret"]:
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
                
                # Skills learned (updated for network skills)
                elif outcome_type == "learned_skill":
                    skills = story_flags.get("learned_skills", [])
                    if outcome_data not in skills:
                        skills.append(outcome_data)
                        story_flags["learned_skills"] = skills
                    
                    # Check for network-specific skills
                    if any(word in outcome_data.lower() for word in ["rose", "thorn", "transformation", "network"]):
                        story_flags["garden_knowledge"] = story_flags.get("garden_knowledge", 0) + 10
                    
                    applied_outcomes["learned_skill"] = outcome_data
                
                # Special story flags
                elif outcome_type in ["vulnerability_witnessed", "mask_removed", "three_words_moment",
                                     "permanent_bond", "ending_achieved", "network_identity_revealed",
                                     "queen_identity_confirmed", "inducted_into_network"]:
                    story_flags[outcome_type] = True
                    if outcome_type == "mask_removed":
                        story_flags["mask_removed_count"] = story_flags.get("mask_removed_count", 0) + 1
                    
                    applied_outcomes[outcome_type] = True
                
                # Choice presented
                elif outcome_type == "choice_presented":
                    story_flags["pending_choice"] = outcome_data
                    applied_outcomes["choice_presented"] = outcome_data
                
                # New quest/role
                elif outcome_type in ["new_quest", "new_role", "gained_title", "network_position"]:
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
                
                # Player alignment shifts (NEW)
                elif outcome_type == "player_alignment":
                    story_flags["player_alignment"] = outcome_data
                    if outcome_data == "protector":
                        story_flags["moth_nature"] = min(100, story_flags.get("moth_nature", 50) + 20)
                    elif outcome_data == "predator":
                        story_flags["moth_nature"] = max(0, story_flags.get("moth_nature", 50) - 20)
                    
                    applied_outcomes["player_alignment"] = outcome_data
                    
            except Exception as e:
                logger.error(f"Error applying outcome {outcome_type}: {e}", exc_info=True)
                applied_outcomes[f"{outcome_type}_error"] = str(e)
        
        return applied_outcomes
    
    @staticmethod
    async def check_network_events(user_id: int, conversation_id: int) -> Optional[Dict[str, Any]]:
        """Check for network-specific events that might trigger"""
        
        async with get_db_connection_context() as conn:
            # Get story flags
            state_row = await conn.fetchrow(
                """
                SELECT story_flags FROM story_states
                WHERE user_id = $1 AND conversation_id = $2 AND story_id = $3
                """,
                user_id, conversation_id, "the_moth_and_flame"
            )
            
            if not state_row:
                return None
            
            story_flags = json.loads(state_row['story_flags'] or '{}')
            
            # Check various network event triggers
            
            # Rose signal event
            if story_flags.get("network_awareness", 0) >= 30 and not story_flags.get("first_rose_signal_sent"):
                return {
                    "event_type": "rose_signal",
                    "description": "You notice a single red rose left at your usual table",
                    "choices": ["investigate", "ignore", "ask_about_it"]
                }
            
            # Transformation opportunity
            if (story_flags.get("transformations_witnessed", 0) >= 2 and 
                story_flags.get("trust_level", 0) >= 60 and
                not story_flags.get("offered_transformation_role")):
                return {
                    "event_type": "transformation_assistant",
                    "description": "The Queen asks if you'd like to help with tonight's session",
                    "choices": ["accept", "observe_only", "decline"]
                }
            
            # Safehouse emergency
            if (story_flags.get("safehouse_visits", 0) >= 3 and
                story_flags.get("kozlov_threat_level", 0) >= 50 and
                random.random() < 0.3):
                return {
                    "event_type": "safehouse_threat",
                    "description": "Sarah Chen contacts you - the Marina safehouse may be compromised",
                    "choices": ["rush_to_help", "alert_the_queen", "call_authorities"]
                }
            
            # Rose Council encounter
            if (story_flags.get("network_awareness", 0) >= 70 and
                story_flags.get("rose_council_awareness", 0) >= 40 and
                not story_flags.get("met_full_council")):
                return {
                    "event_type": "council_meeting",
                    "description": "You're invited to witness a Monday meeting",
                    "choices": ["attend", "politely_decline", "ask_questions_first"]
                }
            
            return None
    
    @staticmethod
    async def advance_network_knowledge(
        user_id: int, conversation_id: int, 
        knowledge_type: str, amount: int = 10
    ) -> Dict[str, Any]:
        """Advance player's understanding of the network"""
        
        async with get_db_connection_context() as conn:
            # Get current state
            state_row = await conn.fetchrow(
                """
                SELECT story_flags FROM story_states
                WHERE user_id = $1 AND conversation_id = $2 AND story_id = $3
                """,
                user_id, conversation_id, "the_moth_and_flame"
            )
            
            if not state_row:
                return {"error": "Story state not found"}
            
            story_flags = json.loads(state_row['story_flags'] or '{}')
            
            # Update appropriate knowledge
            if knowledge_type == "network":
                current = story_flags.get("network_awareness", 0)
                new_value = min(100, current + amount)
                story_flags["network_awareness"] = new_value
                
                # Check for thresholds
                revelations = []
                if current < 30 <= new_value:
                    revelations.append("You begin to understand this is more than a BDSM club")
                if current < 50 <= new_value:
                    revelations.append("The network's true purpose becomes clearer - protection and transformation")
                if current < 70 <= new_value:
                    revelations.append("You realize the Queen of Thorns leads something vast and hidden")
                if current < 90 <= new_value:
                    revelations.append("The full scope of the network's power is staggering")
                
            elif knowledge_type == "rose_council":
                current = story_flags.get("rose_council_awareness", 0)
                new_value = min(100, current + amount)
                story_flags["rose_council_awareness"] = new_value
                
                revelations = []
                if current < 40 <= new_value:
                    revelations.append("Seven women meet on Mondays to shape the Bay Area's hidden currents")
                if current < 80 <= new_value:
                    revelations.append("The Rose Council's influence extends into every major institution")
            
            elif knowledge_type == "garden":
                current = story_flags.get("garden_knowledge", 0)
                new_value = min(100, current + amount)
                story_flags["garden_knowledge"] = new_value
                
                revelations = []
                if current < 25 <= new_value:
                    revelations.append("The garden metaphors aren't just poetry - they're operational language")
                if current < 50 <= new_value:
                    revelations.append("Roses are members, thorns are protectors, gardeners cultivate both")
                if current < 75 <= new_value:
                    revelations.append("You understand the network's communication systems")
            
            # Update database
            await conn.execute(
                """
                UPDATE story_states
                SET story_flags = $3
                WHERE user_id = $1 AND conversation_id = $2 AND story_id = 'the_moth_and_flame'
                """,
                user_id, conversation_id, json.dumps(story_flags)
            )
            
            return {
                "knowledge_type": knowledge_type,
                "new_value": new_value,
                "revelations": revelations
            }

# Maintain compatibility
MothFlameStoryProgression = QueenOfThornsStoryProgression

# Keep compatibility
MothFlameStoryProgression = QueenOfThornsStoryProgression
