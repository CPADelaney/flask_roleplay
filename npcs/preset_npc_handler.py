# npcs/preset_npc_handler.py
from typing import Dict, Any, List, Optional
import json
import logging
import random
from datetime import datetime
from db.connection import get_db_connection_context

# Import new dynamic relationships system
from logic.dynamic_relationships import (
    OptimizedRelationshipManager,
    process_relationship_interaction_tool,
    get_relationship_summary_tool,
    update_relationship_context_tool
)

logger = logging.getLogger(__name__)

class PresetNPCHandler:
    """Handles creation of rich, preset NPCs with full feature parity"""
    
    @staticmethod
    async def create_detailed_npc(ctx, npc_data: Dict[str, Any], story_context: Dict[str, Any]) -> int:
        """Create a detailed NPC from preset data with ALL features"""
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        from db.connection import get_db_connection_context
        from lore.core import canon
        from lore.core.lore_system import LoreSystem
        from npcs.new_npc_creation import NPCCreationHandler
        from memory.wrapper import MemorySystem
        
        # Initialize systems
        lore_system = await LoreSystem.get_instance(user_id, conversation_id)
        npc_handler = NPCCreationHandler()
        relationship_manager = OptimizedRelationshipManager(user_id, conversation_id)
        
        # Step 1: Check if NPC already exists using canonical function
        logger.info(f"Checking if preset NPC {npc_data['name']} already exists")
        
        async with get_db_connection_context() as conn:
            # Use the canonical find_or_create_npc function which has semantic matching
            npc_id = await canon.find_or_create_npc(
                ctx, conn,
                npc_name=npc_data['name'],
                role=npc_data.get('role', ''),
                affiliations=npc_data.get('affiliations', [])
            )
            
            # Check if this is a newly created NPC or existing one
            npc_details = await conn.fetchrow("""
                SELECT age, birthdate, personality_traits, created_at
                FROM NPCStats 
                WHERE npc_id = $1
            """, npc_id)
            
            # If NPC was just created (within last minute), we can fully update it
            # If it's older, we should be more careful about updates
            import datetime
            is_new_npc = False
            if npc_details and npc_details['created_at']:
                time_since_creation = datetime.datetime.now() - npc_details['created_at']
                is_new_npc = time_since_creation.total_seconds() < 60  # Created within last minute
            else:
                is_new_npc = True  # No created_at means it's brand new
        
        # Step 2: Update the NPC with preset data
        logger.info(f"Updating preset NPC {npc_data['name']} (ID: {npc_id}, New: {is_new_npc})")
        
        # Build complete NPC data matching regular creation
        complete_npc_data = PresetNPCHandler._build_complete_npc_data(npc_data, story_context)
        
        if is_new_npc:
            # For new NPCs, do a full update
            result = await PresetNPCHandler._update_npc_fully(
                ctx, npc_id, complete_npc_data, lore_system
            )
        else:
            # For existing NPCs, only update non-conflicting fields
            result = await PresetNPCHandler._update_npc_selectively(
                ctx, npc_id, complete_npc_data, npc_data, lore_system
            )
        
        if "error" in result:
            logger.error(f"Failed to update NPC: {result['error']}")
            # Continue with the existing NPC even if update failed
        
        # Step 3: Add preset-specific enhancements (these are additive, not conflicting)
        await PresetNPCHandler._add_preset_specific_features(
            ctx, npc_id, npc_data, user_id, conversation_id
        )
        
        # Step 4: Initialize memory system (only if new or missing memories)
        async with get_db_connection_context() as conn:
            memory_count = await conn.fetchval("""
                SELECT COUNT(*) FROM NPCMemories
                WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
            """, user_id, conversation_id, npc_id)
            
            if memory_count < 3:  # If NPC has few memories, initialize them
                await PresetNPCHandler._initialize_complete_memory_system(
                    ctx, user_id, conversation_id, npc_id, npc_data
                )
        
        # Step 5: Setup relationships using new system
        await PresetNPCHandler._setup_complete_relationships_new(
            ctx, user_id, conversation_id, npc_id, npc_data, relationship_manager
        )
        
        # Step 6: Initialize special mechanics (additive)
        await PresetNPCHandler._initialize_special_mechanics(
            ctx, user_id, conversation_id, npc_id, npc_data
        )
        
        logger.info(f"Successfully initialized preset NPC {npc_data['name']} with ID {npc_id}")
        return npc_id
    
    @staticmethod
    async def _update_npc_fully(ctx, npc_id: int, complete_data: Dict[str, Any], 
                               lore_system) -> Dict[str, Any]:
        """Fully update an NPC (for new NPCs)"""
        # Extract the fields that can be updated
        updates = {
            "age": complete_data.get("age"),
            "sex": complete_data.get("sex"),
            "physical_description": complete_data.get("physical_description"),
            "personality_traits": json.dumps(complete_data["personality"].get("personality_traits", [])),
            "likes": json.dumps(complete_data["personality"].get("likes", [])),
            "dislikes": json.dumps(complete_data["personality"].get("dislikes", [])),
            "hobbies": json.dumps(complete_data["personality"].get("hobbies", [])),
            "dominance": complete_data["stats"].get("dominance", 50),
            "cruelty": complete_data["stats"].get("cruelty", 30),
            "affection": complete_data["stats"].get("affection", 50),
            "trust": complete_data["stats"].get("trust", 0),
            "respect": complete_data["stats"].get("respect", 0),
            "intensity": complete_data["stats"].get("intensity", 40),
            "archetype_summary": complete_data["archetypes"].get("archetype_summary", ""),
            "archetype_extras_summary": complete_data["archetypes"].get("archetype_extras_summary", ""),
            "introduced": complete_data.get("introduced", False),
            "current_location": complete_data.get("current_location", "Unknown"),
            "affiliations": json.dumps(complete_data.get("affiliations", [])),
            "schedule": json.dumps(complete_data.get("schedule", {}))
        }
        
        # Remove None values
        updates = {k: v for k, v in updates.items() if v is not None}
        
        # Update through LoreSystem for consistency
        result = await lore_system.propose_and_enact_change(
            ctx=ctx,
            entity_type="NPCStats",
            entity_identifier={"npc_id": npc_id},
            updates=updates,
            reason="Initializing preset NPC with complete data"
        )
        
        return result
    
    @staticmethod
    async def _update_npc_selectively(ctx, npc_id: int, complete_data: Dict[str, Any], 
                                     npc_data: Dict[str, Any], lore_system) -> Dict[str, Any]:
        """Selectively update an existing NPC (avoid conflicts)"""
        # Only update fields that are explicitly marked as "should_override" or are new additions
        updates = {}
        
        # Always update these fields as they're story-critical
        critical_updates = {
            "archetype_summary": complete_data["archetypes"].get("archetype_summary", ""),
            "archetype_extras_summary": complete_data["archetypes"].get("archetype_extras_summary", ""),
            "current_location": complete_data.get("current_location", "Unknown"),
            "schedule": json.dumps(complete_data.get("schedule", {}))
        }
        
        updates.update(critical_updates)
        
        # For personality traits, likes, dislikes - merge instead of replace
        async with get_db_connection_context() as conn:
            current_npc = await conn.fetchrow("""
                SELECT personality_traits, likes, dislikes, hobbies
                FROM NPCStats WHERE npc_id = $1
            """, npc_id)
            
            if current_npc:
                # Merge personality data
                current_traits = json.loads(current_npc['personality_traits'] or '[]')
                new_traits = complete_data["personality"].get("personality_traits", [])
                merged_traits = list(set(current_traits + new_traits))
                updates["personality_traits"] = json.dumps(merged_traits)
                
                # Similar merging for likes, dislikes, hobbies
                for field in ['likes', 'dislikes', 'hobbies']:
                    current = json.loads(current_npc[field] or '[]')
                    new = complete_data["personality"].get(field, [])
                    merged = list(set(current + new))
                    updates[field] = json.dumps(merged)
        
        if updates:
            result = await lore_system.propose_and_enact_change(
                ctx=ctx,
                entity_type="NPCStats",
                entity_identifier={"npc_id": npc_id},
                updates=updates,
                reason="Updating existing NPC with preset enhancements"
            )
            return result
        
        return {"status": "no_updates_needed"}
    
    @staticmethod
    def _build_complete_npc_data(npc_data: Dict[str, Any], story_context: Dict[str, Any]) -> Dict[str, Any]:
        """Build complete NPC data structure matching regular creation"""
        
        # Start with base data
        complete_data = {
            "npc_name": npc_data["name"],
            "sex": npc_data.get("sex", "female"),
            "age": npc_data.get("age", random.randint(25, 45)),
            "physical_description": PresetNPCHandler._build_physical_description(npc_data),
            "introduced": npc_data.get("introduced", False),
            "current_location": PresetNPCHandler._get_initial_location(npc_data),
            "affiliations": npc_data.get("affiliations", [])
        }
        
        # Add personality with all components
        complete_data["personality"] = {
            "personality_traits": npc_data.get("traits", []),
            "likes": npc_data.get("personality", {}).get("likes", []),
            "dislikes": npc_data.get("personality", {}).get("dislikes", []),
            "hobbies": npc_data.get("personality", {}).get("hobbies", [])
        }
        
        # Add stats with defaults
        default_stats = {
            "dominance": 50, "cruelty": 30, "closeness": 50,
            "trust": 0, "respect": 0, "intensity": 40
        }
        complete_data["stats"] = {**default_stats, **npc_data.get("stats", {})}
        
        # Add archetypes
        complete_data["archetypes"] = {
            "archetype_names": [npc_data.get("archetype", "Default")],
            "archetype_summary": npc_data.get("role", ""),
            "archetype_extras_summary": npc_data.get("archetype_extras", "")
        }
        
        # Convert schedule to proper format
        complete_data["schedule"] = PresetNPCHandler._convert_schedule(npc_data)
        
        # Add initial memories
        complete_data["memories"] = PresetNPCHandler._create_initial_memories(npc_data)
        
        return complete_data
    
    @staticmethod
    def _build_physical_description(npc_data: Dict[str, Any]) -> str:
        """Build complete physical description from various sources"""
        desc_parts = []
        
        # Direct physical description
        if "physical_description" in npc_data:
            if isinstance(npc_data["physical_description"], str):
                desc_parts.append(npc_data["physical_description"])
            elif isinstance(npc_data["physical_description"], dict):
                pd = npc_data["physical_description"]
                if "base" in pd:
                    desc_parts.append(pd["base"])
                if "style" in pd:
                    desc_parts.append(f"Style: {pd['style']}")
                if "tells" in pd:
                    desc_parts.append(f"Notable behavior: {pd['tells']}")
                if "presence" in pd:
                    desc_parts.append(f"Presence: {pd['presence']}")
                if "public_persona" in pd:
                    desc_parts.append(f"Public appearance: {pd['public_persona']}")
                if "private_self" in pd:
                    desc_parts.append(f"In private: {pd['private_self']}")
        
        # Fallback description based on archetype
        if not desc_parts and "archetype" in npc_data:
            archetype = npc_data["archetype"].lower()
            if "mentor" in archetype:
                desc_parts.append("A figure of quiet authority with knowing eyes")
            elif "victim" in archetype or "survivor" in archetype:
                desc_parts.append("Someone who has seen too much, but refuses to be broken")
            elif "predator" in archetype or "villain" in archetype:
                desc_parts.append("Dangerous presence barely concealed beneath a civilized veneer")
            else:
                desc_parts.append(f"A person who embodies their role as {npc_data.get('role', 'someone important')}")
        
        return "\n\n".join(desc_parts) if desc_parts else "An intriguing individual with hidden depths."
    
    @staticmethod
    def _convert_schedule(npc_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert schedule to game format"""
        if "schedule" not in npc_data:
            return {}
        
        schedule = {}
        for day, periods in npc_data["schedule"].items():
            if isinstance(periods, dict):
                # Already in correct format
                schedule[day] = periods
            elif isinstance(periods, str):
                # Simple string format - convert to all-day
                schedule[day] = {
                    "Morning": periods,
                    "Afternoon": periods,
                    "Evening": periods,
                    "Night": periods
                }
            elif isinstance(periods, list):
                # List format - distribute across periods
                time_periods = ["Morning", "Afternoon", "Evening", "Night"]
                schedule[day] = {}
                for i, activity in enumerate(periods[:4]):  # Max 4 periods
                    schedule[day][time_periods[i]] = activity
                # Fill remaining periods
                for i in range(len(periods), 4):
                    schedule[day][time_periods[i]] = "Continues previous activities"
        
        return schedule
    
    @staticmethod
    def _create_personality_patterns(npc_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create personality pattern entries"""
        patterns = []
        
        # Add dialogue patterns
        if "dialogue_patterns" in npc_data:
            dp = npc_data["dialogue_patterns"]
            for trust_level, dialogue in dp.items():
                if isinstance(dialogue, str):
                    patterns.append({
                        "trigger": trust_level,
                        "response_style": dialogue,
                        "pattern_type": "dialogue"
                    })
                elif isinstance(dialogue, list):
                    for d in dialogue:
                        patterns.append({
                            "trigger": trust_level,
                            "response_style": d,
                            "pattern_type": "dialogue"
                        })
        
        # Add behavioral patterns from traits
        for trait in npc_data.get("traits", []):
            patterns.append({
                "trait": trait,
                "influences": "behavior and dialogue",
                "pattern_type": "personality"
            })
        
        # Add patterns from special mechanics
        if "special_mechanics" in npc_data:
            for mechanic, data in npc_data["special_mechanics"].items():
                if isinstance(data, dict) and "triggers" in data:
                    for trigger in data["triggers"]:
                        patterns.append({
                            "mechanic": mechanic,
                            "trigger": trigger,
                            "pattern_type": "special"
                        })
        
        return patterns
    
    @staticmethod
    def _get_initial_location(npc_data: Dict[str, Any]) -> str:
        """Get initial location for NPC"""
        # Check various fields where location might be stored
        location = (npc_data.get('current_location') or 
                   npc_data.get('initial_location') or
                   npc_data.get('default_location'))
        
        if location:
            return location
        
        # Try to infer from schedule
        if "schedule" in npc_data:
            # Get current day/time and find location
            # For now, just return a default
            for day, periods in npc_data["schedule"].items():
                if isinstance(periods, dict) and "Morning" in periods:
                    activity = periods["Morning"]
                    if "at" in activity:
                        parts = activity.split("at", 1)
                        if len(parts) > 1:
                            return parts[1].strip().split()[0]
        
        return "Town Square"  # Default location
    
    @staticmethod
    def _create_initial_memories(npc_data: Dict[str, Any]) -> List[str]:
        """Create initial memory entries for NPC"""
        memories = []
        
        # Add any preset memories
        if 'memories' in npc_data:
            for memory in npc_data['memories']:
                if isinstance(memory, str):
                    memories.append(memory)
                elif isinstance(memory, dict):
                    memories.append(memory.get('text', ''))
        
        # Add backstory as memories if available
        if 'backstory' in npc_data:
            if isinstance(npc_data['backstory'], str):
                # Split long backstory into memory chunks
                backstory = npc_data['backstory']
                sentences = backstory.split('. ')
                for i in range(0, len(sentences), 3):
                    memory_chunk = '. '.join(sentences[i:i+3])
                    if memory_chunk.strip():
                        memories.append(memory_chunk.strip() + '.')
            elif isinstance(npc_data['backstory'], dict):
                for key, value in npc_data['backstory'].items():
                    if isinstance(value, str) and value.strip():
                        memories.append(value)
        
        # Add role-based memories
        if 'role' in npc_data:
            memories.append(f"I am {npc_data['role']} in this place. This defines much of who I am and what I do.")
        
        # Add archetype-based memories
        if 'archetype' in npc_data:
            archetype = npc_data['archetype'].lower()
            if "mentor" in archetype:
                memories.append("I've guided many before, each one teaching me as much as I taught them.")
            elif "survivor" in archetype:
                memories.append("I survived what should have destroyed me. That strength defines me now.")
            elif "ruler" in archetype or "queen" in archetype:
                memories.append("Power is not taken, it is cultivated. I learned this truth long ago.")
        
        # Ensure we have at least 3 memories
        while len(memories) < 3:
            if npc_data.get('stats', {}).get('dominance', 50) > 70:
                memories.append("Control is not about force, but about understanding what others need.")
            elif npc_data.get('stats', {}).get('cruelty', 30) > 60:
                memories.append("I learned early that mercy is a luxury few can afford.")
            else:
                memories.append("Every interaction shapes us. I am who I am because of those I've known.")
        
        return memories[:8]  # Limit to 8 initial memories
    
    @staticmethod
    async def _add_preset_specific_features(
        ctx, npc_id: int, npc_data: Dict[str, Any], 
        user_id: int, conversation_id: int
    ):
        """Add features specific to preset NPCs"""
        
        from lore.core import canon
        
        canon_ctx = type('CanonicalContext', (), {
            'user_id': user_id,
            'conversation_id': conversation_id
        })()
        
        async with get_db_connection_context() as conn:
            # Add dialogue patterns if specified
            if "dialogue_patterns" in npc_data:
                await canon.update_entity_canonically(
                    canon_ctx, conn, "NPCStats", npc_id,
                    {
                        "dialogue_patterns": json.dumps(npc_data["dialogue_patterns"]),
                        "dialogue_style": npc_data.get("dialogue_style", "contextual")
                    },
                    f"Adding preset dialogue patterns for {npc_data['name']}"
                )
            
            # Add trauma triggers if specified
            if "trauma_triggers" in npc_data:
                flashback_words = PresetNPCHandler._extract_flashback_words(npc_data["trauma_triggers"])
                await canon.update_entity_canonically(
                    canon_ctx, conn, "NPCStats", npc_id,
                    {
                        "trauma_triggers": json.dumps(npc_data["trauma_triggers"]),
                        "flashback_triggers": json.dumps(flashback_words),
                        "trauma_responses": json.dumps(npc_data.get("trauma_responses", {}))
                    },
                    f"Setting trauma triggers for {npc_data['name']}"
                )
            
            # Add relationship mechanics
            if "relationship_mechanics" in npc_data:
                await canon.update_entity_canonically(
                    canon_ctx, conn, "NPCStats", npc_id,
                    {
                        "relationship_mechanics": json.dumps(npc_data["relationship_mechanics"]),
                        "trust_thresholds": json.dumps(npc_data.get("trust_thresholds", {}))
                    },
                    f"Setting relationship mechanics for {npc_data['name']}"
                )
            
            # Add secrets
            secrets = {}
            if "secrets" in npc_data:
                secrets.update(npc_data["secrets"])
            if "backstory" in npc_data and isinstance(npc_data["backstory"], dict):
                for key, value in npc_data["backstory"].items():
                    if key not in ["history", "public_knowledge"]:
                        secrets[key] = value
            
            if secrets:
                await canon.update_entity_canonically(
                    canon_ctx, conn, "NPCStats", npc_id,
                    {
                        "secrets": json.dumps(secrets),
                        "hidden_stats": json.dumps(npc_data.get("hidden_stats", {}))
                    },
                    f"Adding secrets for {npc_data['name']}"
                )
            
            # Add evolution paths
            if "narrative_evolution" in npc_data:
                initial_stage = "Initial"
                if "trust_path" in npc_data["narrative_evolution"]:
                    stages = npc_data["narrative_evolution"]["trust_path"].get("stages", [])
                    if stages:
                        initial_stage = stages[0] if isinstance(stages[0], str) else stages[0].get("name", "Initial")
                
                await canon.update_entity_canonically(
                    canon_ctx, conn, "NPCStats", npc_id,
                    {
                        "evolution_paths": json.dumps(npc_data["narrative_evolution"]),
                        "current_evolution_stage": initial_stage,
                        "evolution_triggers_met": json.dumps([])
                    },
                    f"Setting evolution paths for {npc_data['name']}"
                )
            
            # Add memory priorities
            if "memory_priorities" in npc_data:
                await canon.update_entity_canonically(
                    canon_ctx, conn, "NPCStats", npc_id,
                    {
                        "memory_priorities": json.dumps(npc_data["memory_priorities"]),
                        "memory_focus": npc_data.get("memory_focus", "general")
                    },
                    f"Setting memory priorities for {npc_data['name']}"
                )
            
            # Add personality patterns
            patterns = PresetNPCHandler._create_personality_patterns(npc_data)
            if patterns:
                await canon.update_entity_canonically(
                    canon_ctx, conn, "NPCStats", npc_id,
                    {"personality_patterns": json.dumps(patterns)},
                    f"Setting personality patterns for {npc_data['name']}"
                )
            
            # Add story integration flags
            if "story_flags" in npc_data:
                await canon.update_entity_canonically(
                    canon_ctx, conn, "NPCStats", npc_id,
                    {"story_flags": json.dumps(npc_data["story_flags"])},
                    f"Setting story flags for {npc_data['name']}"
                )
    
    @staticmethod
    async def _initialize_complete_memory_system(
        ctx, user_id: int, conversation_id: int, 
        npc_id: int, npc_data: Dict[str, Any]
    ):
        """Initialize ALL memory subsystems matching regular NPC creation"""
        
        from npcs.new_npc_creation import NPCCreationHandler
        from db.connection import get_db_connection_context
        from memory.wrapper import MemorySystem
        
        handler = NPCCreationHandler()
        
        # Prepare NPC data for memory initialization
        npc_info = {
            "npc_name": npc_data["name"],
            "dominance": npc_data.get("stats", {}).get("dominance", 50),
            "cruelty": npc_data.get("stats", {}).get("cruelty", 30),
            "archetype_summary": npc_data.get("archetype", ""),
            "personality_traits": npc_data.get("traits", []),
            "environment_desc": npc_data.get("environment_desc", "")
        }
        
        # Get initial memories
        memories = npc_data.get("memories", [])
        if isinstance(memories, list) and all(isinstance(m, str) for m in memories):
            # Already in correct format
            pass
        else:
            # Need to generate memories
            memories = await handler.generate_memories(ctx, npc_data["name"])
        
        # 1. Store memories with governance
        await handler.store_npc_memories(user_id, conversation_id, npc_id, memories)
        
        # 2. Initialize emotional state
        await handler.initialize_npc_emotional_state(
            user_id, conversation_id, npc_id, npc_info, memories
        )
        
        # 3. Generate beliefs
        await handler.generate_npc_beliefs(
            user_id, conversation_id, npc_id, npc_info
        )
        
        # 4. Initialize memory schemas (including preset-specific ones)
        await PresetNPCHandler._initialize_enhanced_memory_schemas(
            user_id, conversation_id, npc_id, npc_data
        )
        
        # 5. Setup trauma model if applicable
        await handler.setup_npc_trauma_model(
            user_id, conversation_id, npc_id, npc_info, memories
        )
        
        # 6. Setup flashback triggers
        await handler.setup_npc_flashback_triggers(
            user_id, conversation_id, npc_id, npc_info
        )
        
        # 7. Generate counterfactual memories
        await handler.generate_counterfactual_memories(
            user_id, conversation_id, npc_id, npc_info
        )
        
        # 8. Plan mask revelations if applicable
        if npc_data.get("has_masks", True):  # Most NPCs have psychological masks
            await handler.plan_mask_revelations(
                user_id, conversation_id, npc_id, npc_info
            )
        
        # 9. Setup relationship evolution tracking
        relationships = []
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow(
                "SELECT relationships FROM NPCStats WHERE npc_id = $1",
                npc_id
            )
            if row and row['relationships']:
                try:
                    relationships = json.loads(row['relationships'])
                except:
                    relationships = []
        
        await handler.setup_relationship_evolution_tracking(
            user_id, conversation_id, npc_id, relationships
        )
        
        # 10. Build semantic networks
        await handler.build_initial_semantic_network(
            user_id, conversation_id, npc_id, npc_info
        )
        
        # 11. Detect initial patterns
        await handler.detect_memory_patterns(
            user_id, conversation_id, npc_id
        )
        
        # 12. Schedule maintenance
        await handler.schedule_npc_memory_maintenance(
            user_id, conversation_id, npc_id
        )
        
        # 13. Check for mask slippage conditions (if applicable)
        if npc_data.get("stats", {}).get("dominance", 50) > 60:
            await handler.check_for_mask_slippage(
                user_id, conversation_id, npc_id
            )
    
    @staticmethod
    async def _initialize_enhanced_memory_schemas(
        user_id: int, conversation_id: int, 
        npc_id: int, npc_data: Dict[str, Any]
    ):
        """Initialize memory schemas including preset-specific ones"""
        
        from memory.wrapper import MemorySystem
        from memory.schemas import MemorySchemaManager  # Add this import
        from npcs.new_npc_creation import NPCCreationHandler
        
        handler = NPCCreationHandler()
        memory_system = await MemorySystem.get_instance(user_id, conversation_id)
        
        # Create a MemorySchemaManager instance directly
        schema_manager = MemorySchemaManager(user_id, conversation_id)
        
        # First, create standard schemas
        await handler.initialize_npc_memory_schemas(
            user_id, conversation_id, npc_id, {
                "archetype_summary": npc_data.get("archetype", ""),
                "dominance": npc_data.get("stats", {}).get("dominance", 50)
            }
        )
        
        # Add preset-specific schemas based on character type
        preset_schemas = []
        
        # Analyze character data to determine appropriate schemas
        character_text = json.dumps(npc_data).lower()
        
        # Trauma/Survivor schemas
        if any(word in character_text for word in ["trafficking", "victim", "survivor", "trauma", "rescued"]):
            preset_schemas.append({
                "name": "Survival Patterns",
                "description": "Tracking survival instincts and recovery",
                "category": "trauma",
                "attributes": {
                    "trigger_type": "unknown",
                    "coping_mechanism": "unknown",
                    "recovery_progress": 0,
                    "trust_rebuilding": "slow"
                }
            })
        
        # Mentor schemas
        if any(word in character_text for word in ["mentor", "teacher", "guide", "instructor"]):
            preset_schemas.append({
                "name": "Teaching Moments",
                "description": "Tracking mentorship and guidance given",
                "category": "relationship",
                "attributes": {
                    "lesson_type": "unknown",
                    "student_progress": "unknown",
                    "teaching_style": "unknown",
                    "hidden_agenda": "unknown"
                }
            })
        
        # High dominance schemas
        if npc_data.get("stats", {}).get("dominance", 0) > 70:
            preset_schemas.append({
                "name": "Control Patterns",
                "description": "How control is established and maintained",
                "category": "power",
                "attributes": {
                    "control_method": "unknown",
                    "effectiveness": "unknown",
                    "target_response": "unknown",
                    "satisfaction": "unknown"
                }
            })
        
        # Devotee/Submissive schemas
        if any(word in character_text for word in ["devoted", "submissive", "worshipful", "broken"]):
            preset_schemas.append({
                "name": "Devotion Patterns",
                "description": "Tracking devotion and submission dynamics",
                "category": "relationship",
                "attributes": {
                    "devotion_trigger": "unknown",
                    "service_type": "unknown",
                    "satisfaction_received": "unknown",
                    "identity_dissolution": 0
                }
            })
        
        # Predator/Antagonist schemas
        if any(word in character_text for word in ["predator", "villain", "antagonist", "enemy", "trafficker"]):
            preset_schemas.append({
                "name": "Predatory Patterns",
                "description": "Tracking predatory behavior and victim selection",
                "category": "threat",
                "attributes": {
                    "hunting_method": "unknown",
                    "victim_type": "unknown",
                    "success_rate": "unknown",
                    "weaknesses": "unknown"
                }
            })
        
        # Dual identity schemas
        if "dual_identity" in character_text or "double_life" in character_text:
            preset_schemas.append({
                "name": "Identity Management",
                "description": "Tracking dual identity maintenance",
                "category": "identity",
                "attributes": {
                    "current_identity": "public",
                    "identity_stress": 0,
                    "close_calls": 0,
                    "trusted_with_secret": []
                }
            })
        
        # Create the preset-specific schemas using schema_manager directly
        for schema in preset_schemas:
            await schema_manager.create_schema(  # Use schema_manager instead of memory_system.schema_manager
                entity_type="npc",
                entity_id=npc_id,
                schema_name=schema["name"],
                description=schema["description"],
                category=schema["category"],
                attributes=schema["attributes"]
            )
    
    @staticmethod
    async def _setup_complete_relationships_new(
        ctx, user_id: int, conversation_id: int,
        npc_id: int, npc_data: Dict[str, Any],
        relationship_manager: OptimizedRelationshipManager
    ):
        """Setup relationships using the new dynamic relationships system"""
        
        from npcs.new_npc_creation import NPCCreationHandler
        from lore.core import canon
        from db.connection import get_db_connection_context
        from agents import RunContextWrapper
        
        handler = NPCCreationHandler()
        
        # Check for predefined relationships in the preset data
        preset_relationships = npc_data.get("relationships", [])
        
        if preset_relationships:
            # Create specific preset relationships
            canon_ctx = type('CanonicalContext', (), {
                'user_id': user_id,
                'conversation_id': conversation_id
            })()
            
            async with get_db_connection_context() as conn:
                for rel in preset_relationships:
                    # Determine target
                    target_type = rel.get("target_type", "player")
                    target_id = rel.get("target_id", user_id if target_type == "player" else 0)
                    
                    # Get or create relationship state using new system
                    state = await relationship_manager.get_relationship_state(
                        entity1_type="npc",
                        entity1_id=npc_id,
                        entity2_type=target_type,
                        entity2_id=target_id
                    )
                    
                    # Set initial dimensions based on relationship type
                    rel_type = rel.get("type", "neutral")
                    initial_strength = rel.get("strength", 50)
                    
                    # Map old relationship types to new dimensions
                    if rel_type == "ally":
                        state.dimensions.trust = initial_strength
                        state.dimensions.respect = initial_strength
                        state.dimensions.affection = initial_strength * 0.8
                    elif rel_type == "enemy":
                        state.dimensions.trust = -initial_strength
                        state.dimensions.respect = initial_strength * 0.5  # Respect can exist even with enemies
                        state.dimensions.affection = -initial_strength
                    elif rel_type == "lover":
                        state.dimensions.trust = initial_strength * 0.9
                        state.dimensions.affection = initial_strength
                        state.dimensions.intimacy = initial_strength * 0.8
                        state.dimensions.fascination = initial_strength * 0.7
                    elif rel_type == "mentor":
                        state.dimensions.trust = initial_strength * 0.8
                        state.dimensions.respect = initial_strength
                        state.dimensions.influence = -30  # Mentor has influence over student
                    elif rel_type == "rival":
                        state.dimensions.respect = initial_strength * 0.7
                        state.dimensions.affection = 0
                        state.dimensions.volatility = initial_strength * 0.6
                    elif rel_type == "victim":
                        state.dimensions.trust = -initial_strength * 0.5
                        state.dimensions.respect = -initial_strength * 0.3
                        state.dimensions.influence = initial_strength * 0.7  # NPC has influence over victim
                        state.dimensions.unresolved_conflict = initial_strength * 0.8
                    
                    # Apply additional relationship-specific modifiers from preset data
                    if "dimensions" in rel:
                        for dim, value in rel["dimensions"].items():
                            if hasattr(state.dimensions, dim):
                                setattr(state.dimensions, dim, value)
                    
                    # Clamp all values
                    state.dimensions.clamp()
                    
                    # Queue update to save the relationship
                    await relationship_manager._queue_update(state)
                    
                    # If there are specific contexts defined
                    if "contexts" in rel:
                        ctx_wrapper = RunContextWrapper(context={
                            'user_id': user_id,
                            'conversation_id': conversation_id
                        })
                        
                        for context_name, deltas in rel["contexts"].items():
                            await update_relationship_context_tool(
                                ctx=ctx_wrapper,
                                entity1_type="npc",
                                entity1_id=npc_id,
                                entity2_type=target_type,
                                entity2_id=target_id,
                                situation=context_name,
                                dimension_deltas=deltas
                            )
                    
                    # Add to NPC's relationship list for compatibility
                    rel_query = """
                        SELECT relationships FROM NPCStats
                        WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3
                    """
                    
                    row = await conn.fetchrow(rel_query, user_id, conversation_id, npc_id)
                    current_relationships = []
                    if row and row['relationships']:
                        try:
                            current_relationships = json.loads(row['relationships'])
                        except:
                            current_relationships = []
                    
                    current_relationships.append({
                        "relationship_label": rel_type,
                        "entity_type": target_type,
                        "entity_id": target_id
                    })
                    
                    await canon.update_entity_canonically(
                        canon_ctx, conn, "NPCStats", npc_id,
                        {"relationships": json.dumps(current_relationships)},
                        f"Adding preset relationship: {rel_type}"
                    )
        else:
            # Use the standard random relationship assignment
            await handler.assign_random_relationships_canonical(
                user_id, conversation_id, npc_id, 
                npc_data["name"], 
                [{"name": npc_data.get("archetype", "Default")}]
            )
        
        # Add relationship-specific memories if provided
        if "relationship_memories" in npc_data:
            from memory.wrapper import MemorySystem
            memory_system = await MemorySystem.get_instance(user_id, conversation_id)
            for memory in npc_data["relationship_memories"]:
                await memory_system.remember(
                    entity_type="npc",
                    entity_id=npc_id,
                    memory_text=memory,
                    importance="high",
                    emotional=True,
                    tags=["relationship", "preset_memory"]
                )
        
        # Flush any pending relationship updates
        await relationship_manager._flush_updates()
    
    @staticmethod
    async def _initialize_special_mechanics(
        ctx, user_id: int, conversation_id: int,
        npc_id: int, npc_data: Dict[str, Any]
    ):
        """Initialize any special mechanics for the preset NPC"""
        
        if "special_mechanics" not in npc_data:
            return
        
        from db.connection import get_db_connection_context
        
        # Create the table if it doesn't exist
        async with get_db_connection_context() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS npc_special_mechanics (
                    user_id INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL,
                    npc_id INTEGER NOT NULL,
                    mechanic_type VARCHAR(100) NOT NULL,
                    mechanic_data JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, conversation_id, npc_id, mechanic_type)
                )
            """)
        
        async with get_db_connection_context() as conn:
            for mechanic_type, mechanic_data in npc_data["special_mechanics"].items():
                # Store each special mechanic
                await conn.execute(
                    """
                    INSERT INTO npc_special_mechanics 
                    (user_id, conversation_id, npc_id, mechanic_type, mechanic_data)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (user_id, conversation_id, npc_id, mechanic_type)
                    DO UPDATE SET mechanic_data = EXCLUDED.mechanic_data
                    """,
                    user_id, conversation_id, npc_id, 
                    mechanic_type, json.dumps(mechanic_data)
                )
                
                # Initialize specific mechanics
                if mechanic_type == "mask_system":
                    await PresetNPCHandler._initialize_mask_system(
                        ctx, user_id, conversation_id, npc_id, mechanic_data
                    )
                elif mechanic_type == "dual_identity":
                    await PresetNPCHandler._initialize_dual_identity(
                        ctx, user_id, conversation_id, npc_id, mechanic_data
                    )
                elif mechanic_type == "poetry_triggers":
                    await PresetNPCHandler._initialize_poetry_system(
                        ctx, user_id, conversation_id, npc_id, mechanic_data
                    )
                elif mechanic_type == "three_words":
                    await PresetNPCHandler._initialize_three_words(
                        ctx, user_id, conversation_id, npc_id, mechanic_data
                    )
                elif mechanic_type == "safehouse_network":
                    await PresetNPCHandler._initialize_safehouse_system(
                        ctx, user_id, conversation_id, npc_id, mechanic_data
                    )
                elif mechanic_type == "moth_and_flame":
                    await PresetNPCHandler._initialize_moth_flame_dynamic(
                        ctx, user_id, conversation_id, npc_id, mechanic_data
                    )
    
    @staticmethod
    async def _initialize_mask_system(
        ctx, user_id: int, conversation_id: int,
        npc_id: int, mask_data: Dict[str, Any]
    ):
        """Initialize the mask system for NPCs that use it"""
        
        from lore.core import canon
        
        canon_ctx = type('CanonicalContext', (), {
            'user_id': user_id,
            'conversation_id': conversation_id
        })()
        
        # Set initial mask
        initial_mask = mask_data.get("initial_mask", "default")
        
        async with get_db_connection_context() as conn:
            await canon.update_entity_canonically(
                canon_ctx, conn, "NPCStats", npc_id,
                {
                    "current_mask": initial_mask,
                    "mask_integrity": 100,
                    "masks_available": json.dumps(mask_data.get("types", {}))
                },
                f"Initializing mask system with {initial_mask}"
            )
    
    @staticmethod
    async def _initialize_dual_identity(
        ctx, user_id: int, conversation_id: int,
        npc_id: int, identity_data: Dict[str, Any]
    ):
        """Initialize dual identity system"""
        
        from lore.core import canon
        
        canon_ctx = type('CanonicalContext', (), {
            'user_id': user_id,
            'conversation_id': conversation_id
        })()
        
        async with get_db_connection_context() as conn:
            await canon.update_entity_canonically(
                canon_ctx, conn, "NPCStats", npc_id,
                {
                    "dual_identity": json.dumps(identity_data),
                    "current_identity": identity_data.get("default_identity", "public"),
                    "identity_revealed": False
                },
                f"Initializing dual identity system"
            )
    
    @staticmethod
    async def _initialize_poetry_system(
        ctx, user_id: int, conversation_id: int,
        npc_id: int, poetry_data: Dict[str, Any]
    ):
        """Initialize poetry/lyrical speech system"""
        
        from db.connection import get_db_connection_context
        
        # Store poetry configuration
        async with get_db_connection_context() as conn:
            await conn.execute(
                """
                UPDATE npc_special_mechanics
                SET mechanic_data = mechanic_data || $4
                WHERE user_id = $1 AND conversation_id = $2 
                AND npc_id = $3 AND mechanic_type = 'poetry_triggers'
                """,
                user_id, conversation_id, npc_id,
                json.dumps({"poetry_used": [], "understanding_tracker": {"attempts": 0, "successes": 0}})
            )
    
    @staticmethod
    async def _initialize_three_words(
        ctx, user_id: int, conversation_id: int,
        npc_id: int, words_data: Dict[str, Any]
    ):
        """Initialize the three words mechanic"""
        
        from db.connection import get_db_connection_context
        
        # Set up tracking for the three words
        async with get_db_connection_context() as conn:
            await conn.execute(
                """
                UPDATE npc_special_mechanics
                SET mechanic_data = mechanic_data || $4
                WHERE user_id = $1 AND conversation_id = $2 
                AND npc_id = $3 AND mechanic_type = 'three_words'
                """,
                user_id, conversation_id, npc_id,
                json.dumps({
                    "near_speaking_moments": [],
                    "player_attempts_to_hear": 0,
                    "spoken": False,
                    "trust_threshold": words_data.get("trust_threshold", 95),
                    "emotional_threshold": words_data.get("emotional_threshold", 90)
                })
            )
    
    @staticmethod
    async def _initialize_safehouse_system(
        ctx, user_id: int, conversation_id: int,
        npc_id: int, safehouse_data: Dict[str, Any]
    ):
        """Initialize underground safehouse network"""
        
        from lore.core import canon
        
        canon_ctx = type('CanonicalContext', (), {
            'user_id': user_id,
            'conversation_id': conversation_id
        })()
        
        async with get_db_connection_context() as conn:
            await canon.update_entity_canonically(
                canon_ctx, conn, "NPCStats", npc_id,
                {
                    "safehouse_network": json.dumps(safehouse_data),
                    "people_saved": 0,
                    "network_compromised": False
                },
                f"Initializing safehouse network"
            )
    
    @staticmethod
    async def _initialize_moth_flame_dynamic(
        ctx, user_id: int, conversation_id: int,
        npc_id: int, dynamic_data: Dict[str, Any]
    ):
        """Initialize moth and flame relationship dynamic"""
        
        from lore.core import canon
        
        canon_ctx = type('CanonicalContext', (), {
            'user_id': user_id,
            'conversation_id': conversation_id
        })()
        
        async with get_db_connection_context() as conn:
            await canon.update_entity_canonically(
                canon_ctx, conn, "NPCStats", npc_id,
                {
                    "moth_flame_dynamic": "unestablished",
                    "player_role": "unknown",
                    "dynamic_intensity": 0
                },
                f"Initializing moth and flame dynamic"
            )
    
    @staticmethod
    def _extract_flashback_words(trauma_triggers: List[str]) -> List[str]:
        """Extract key words from trauma triggers for flashback system"""
        flashback_words = []
        
        # Common words to exclude
        exclude_words = {
            "being", "having", "there", "where", "which", "their",
            "about", "would", "could", "should", "these", "those"
        }
        
        for trigger in trauma_triggers:
            # Extract significant words (longer than 4 chars, not common words)
            words = trigger.lower().split()
            for word in words:
                # Clean the word of punctuation
                cleaned_word = ''.join(c for c in word if c.isalpha())
                if len(cleaned_word) > 4 and cleaned_word not in exclude_words:
                    flashback_words.append(cleaned_word)
        
        # Also add some direct trigger words if present
        direct_triggers = ["abandon", "leave", "disappear", "promise", "forever", "goodbye"]
        for trigger in direct_triggers:
            if any(trigger in t.lower() for t in trauma_triggers):
                flashback_words.append(trigger)
        
        return list(set(flashback_words))[:10]  # Limit to 10 unique words
