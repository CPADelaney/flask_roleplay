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
        
        # Initialize systems
        lore_system = await LoreSystem.get_instance(user_id, conversation_id)
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
    def _normalize_preset_memories(npc_data: Dict[str, Any]) -> List[str]:
        """Extract explicit memories from preset data."""

        memories: List[str] = []
        raw_memories = npc_data.get("memories") or npc_data.get("initial_memories")

        if isinstance(raw_memories, list):
            for entry in raw_memories:
                if isinstance(entry, str) and entry.strip():
                    memories.append(entry.strip())
                elif isinstance(entry, dict):
                    text = entry.get("text") or entry.get("memory") or entry.get("content")
                    if isinstance(text, str) and text.strip():
                        memories.append(text.strip())

        return memories

    @staticmethod
    async def _store_preset_memories(memory_system, npc_id: int, memories: List[str]) -> None:
        """Persist preset memories through the memory system."""

        if not memories:
            return

        for idx, memory_text in enumerate(memories):
            if not isinstance(memory_text, str):
                continue

            text = memory_text.strip()
            if not text:
                continue

            importance = "high" if idx < 2 else "medium"
            tags = ["preset_seed", "npc_creation"]

            try:
                await memory_system.remember(
                    entity_type="npc",
                    entity_id=npc_id,
                    memory_text=text,
                    importance=importance,
                    emotional=True,
                    tags=tags,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to store preset memory for NPC %s: %s", npc_id, exc
                )

    @staticmethod
    async def _seed_emotional_state(memory_system, npc_id: int,
                                    npc_data: Dict[str, Any],
                                    memories: List[str]) -> None:
        """Derive a baseline emotional state from preset data."""

        stats = npc_data.get("stats", {})
        provided_state = npc_data.get("emotional_state") or {}

        if isinstance(provided_state, dict):
            emotion = provided_state.get("primary_emotion", "neutral")
            intensity = provided_state.get("intensity", 0.5)
        else:
            dominance = stats.get("dominance", 50)
            cruelty = stats.get("cruelty", 30)

            if dominance > 70 and cruelty < 60:
                emotion = "confidence"
            elif cruelty > 70:
                emotion = "contempt"
            elif dominance > 60:
                emotion = "pride"
            else:
                emotion = "neutral"

            intensity = min(max((dominance + cruelty) / 200.0, 0.2), 0.95)

        try:
            await memory_system.update_npc_emotion(
                npc_id=npc_id,
                emotion=emotion,
                intensity=float(intensity),
            )
        except Exception as exc:
            logger.warning("Failed to initialize emotional state for NPC %s: %s", npc_id, exc)

    @staticmethod
    async def _seed_beliefs(memory_system, npc_id: int, raw_beliefs: Any) -> None:
        """Store preset beliefs when provided."""

        if not raw_beliefs:
            return

        beliefs: List[Dict[str, Any]] = []

        if isinstance(raw_beliefs, list):
            for entry in raw_beliefs:
                if isinstance(entry, str) and entry.strip():
                    beliefs.append({"text": entry.strip(), "confidence": 0.75})
                elif isinstance(entry, dict):
                    text = entry.get("text") or entry.get("belief")
                    if isinstance(text, str) and text.strip():
                        beliefs.append({
                            "text": text.strip(),
                            "confidence": float(entry.get("confidence", 0.75)),
                        })

        for belief in beliefs:
            try:
                await memory_system.create_belief(
                    entity_type="npc",
                    entity_id=npc_id,
                    belief_text=belief["text"],
                    confidence=belief["confidence"],
                )
            except Exception as exc:
                logger.warning(
                    "Failed to seed belief '%s' for NPC %s: %s",
                    belief.get("text", ""),
                    npc_id,
                    exc,
                )

    @staticmethod
    async def _apply_memory_metadata(
        user_id: int,
        conversation_id: int,
        npc_id: int,
        npc_data: Dict[str, Any],
    ) -> None:
        """Persist optional preset metadata directly onto the NPC row."""

        from db.connection import get_db_connection_context
        from lore.core import canon

        updates: Dict[str, Any] = {}

        if "trauma_triggers" in npc_data:
            updates["trauma_triggers"] = json.dumps(npc_data["trauma_triggers"])

        if "flashback_triggers" in npc_data:
            updates["flashback_triggers"] = json.dumps(
                npc_data["flashback_triggers"]
            )

        if "revelation_plan" in npc_data:
            updates["revelation_plan"] = json.dumps(npc_data["revelation_plan"])

        if "personality_patterns" in npc_data:
            updates["personality_patterns"] = json.dumps(
                npc_data["personality_patterns"]
            )

        if not updates:
            return

        canon_ctx = type("CanonicalContext", (), {
            "user_id": user_id,
            "conversation_id": conversation_id,
        })()

        async with get_db_connection_context() as conn:
            await canon.update_entity_canonically(
                canon_ctx,
                conn,
                "NPCStats",
                npc_id,
                updates,
                "Applying preset memory metadata",
            )
    
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
            # Collect all story-specific data
            special_mechanics = {}
            
            # Add story identifier
            special_mechanics["story"] = npc_data.get("story", "queen_of_thorns")
            
            # Dialogue patterns (this could stay as separate column if universal)
            if "dialogue_patterns" in npc_data:
                special_mechanics["dialogue_patterns"] = npc_data["dialogue_patterns"]
                special_mechanics["dialogue_style"] = npc_data.get("dialogue_style", "contextual")
            
            # Trauma system (story-specific)
            if "trauma_triggers" in npc_data:
                special_mechanics["trauma"] = {
                    "triggers": npc_data["trauma_triggers"],
                    "flashback_words": PresetNPCHandler._extract_flashback_words(npc_data["trauma_triggers"]),
                    "responses": npc_data.get("trauma_responses", {})
                }
            
            # Relationship mechanics (could be universal)
            if "relationship_mechanics" in npc_data:
                special_mechanics["relationship_mechanics"] = npc_data["relationship_mechanics"]
                special_mechanics["trust_thresholds"] = npc_data.get("trust_thresholds", {})
            
            # Secrets (story-specific)
            if "secrets" in npc_data or "backstory" in npc_data:
                special_mechanics["secrets"] = {}
                if "secrets" in npc_data:
                    special_mechanics["secrets"].update(npc_data["secrets"])
                if "backstory" in npc_data and isinstance(npc_data["backstory"], dict):
                    for key, value in npc_data["backstory"].items():
                        if key not in ["history", "public_knowledge"]:
                            special_mechanics["secrets"][key] = value
            
            # Evolution paths (story-specific)
            if "narrative_evolution" in npc_data:
                special_mechanics["evolution"] = npc_data["narrative_evolution"]
                special_mechanics["evolution_stage"] = "Initial"
                special_mechanics["evolution_triggers_met"] = []
            
            # Story flags
            if "story_flags" in npc_data:
                special_mechanics["story_flags"] = npc_data["story_flags"]
            
            # Update once with all mechanics
            await canon.update_entity_canonically(
                canon_ctx, conn, "NPCStats", npc_id,
                {"special_mechanics": json.dumps(special_mechanics)},
                f"Setting story-specific features for {npc_data['name']}"
            )
        
    @staticmethod
    async def _initialize_complete_memory_system(
        ctx, user_id: int, conversation_id: int,
        npc_id: int, npc_data: Dict[str, Any]
    ):
        """Initialize memory subsystems using preset data without heavy generation."""

        from memory.wrapper import MemorySystem

        memory_system = await MemorySystem.get_instance(user_id, conversation_id)

        # Seed memories directly from preset definitions
        memories = PresetNPCHandler._normalize_preset_memories(npc_data)
        if not memories:
            memories = PresetNPCHandler._create_initial_memories(npc_data)

        await PresetNPCHandler._store_preset_memories(memory_system, npc_id, memories)

        # Establish an initial emotional state derived from stats or preset overrides
        await PresetNPCHandler._seed_emotional_state(
            memory_system, npc_id, npc_data, memories
        )

        # Persist explicit beliefs if the preset provides them
        await PresetNPCHandler._seed_beliefs(
            memory_system, npc_id, npc_data.get("beliefs")
        )

        # Initialize schema data with lightweight heuristics
        await PresetNPCHandler._initialize_enhanced_memory_schemas(
            user_id, conversation_id, npc_id, npc_data
        )

        # Apply optional metadata (trauma triggers, flashbacks, etc.) only when present
        await PresetNPCHandler._apply_memory_metadata(
            user_id, conversation_id, npc_id, npc_data
        )
    
    @staticmethod
    async def _initialize_enhanced_memory_schemas(
        user_id: int, conversation_id: int,
        npc_id: int, npc_data: Dict[str, Any]
    ):
        """Initialize memory schemas including preset-specific ones"""

        from memory.schemas import MemorySchemaManager

        schema_manager = MemorySchemaManager(user_id, conversation_id)

        created_custom_schema = False

        # Honor any explicit schemas provided by the preset
        preset_defined = npc_data.get("memory_schemas")
        if isinstance(preset_defined, list):
            for schema in preset_defined:
                name = schema.get("name") if isinstance(schema, dict) else None
                if not name:
                    continue

                description = schema.get("description", "")
                category = schema.get("category", "general")
                attributes = schema.get("attributes", {})

                await schema_manager.create_schema(
                    entity_type="npc",
                    entity_id=npc_id,
                    schema_name=name,
                    description=description,
                    category=category,
                    attributes=attributes,
                )
                created_custom_schema = True

        # Always ensure at least a baseline interaction schema exists
        if not created_custom_schema:
            await schema_manager.create_schema(
                entity_type="npc",
                entity_id=npc_id,
                schema_name="Player Interactions",
                description="Patterns in how the player behaves toward the NPC",
                category="social",
                attributes={
                    "compliance_level": "unknown",
                    "respect_shown": "moderate",
                    "vulnerability_signs": "to be observed",
                },
            )

        # Add heuristic schemas based on preset content
        preset_schemas: List[Dict[str, Any]] = []

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

        from db.connection import get_db_connection_context
        from agents import RunContextWrapper

        # Check for predefined relationships in the preset data
        preset_relationships = npc_data.get("relationships", [])

        canon_ctx = type('CanonicalContext', (), {
            'user_id': user_id,
            'conversation_id': conversation_id
        })()

        async with get_db_connection_context() as conn:
            if preset_relationships:
                for rel in preset_relationships:
                    target_type = rel.get("target_type", "player")
                    target_id = rel.get(
                        "target_id", user_id if target_type == "player" else 0
                    )

                    state = await relationship_manager.get_relationship_state(
                        entity1_type="npc",
                        entity1_id=npc_id,
                        entity2_type=target_type,
                        entity2_id=target_id
                    )

                    rel_type = rel.get("type", "neutral")
                    initial_strength = rel.get("strength", 50)

                    if rel_type == "ally":
                        state.dimensions.trust = initial_strength
                        state.dimensions.respect = initial_strength
                        state.dimensions.affection = initial_strength * 0.8
                    elif rel_type == "enemy":
                        state.dimensions.trust = -initial_strength
                        state.dimensions.respect = initial_strength * 0.5
                        state.dimensions.affection = -initial_strength
                    elif rel_type == "lover":
                        state.dimensions.trust = initial_strength * 0.9
                        state.dimensions.affection = initial_strength
                        state.dimensions.intimacy = initial_strength * 0.8
                        state.dimensions.fascination = initial_strength * 0.7
                    elif rel_type == "mentor":
                        state.dimensions.trust = initial_strength * 0.8
                        state.dimensions.respect = initial_strength
                        state.dimensions.influence = -30
                    elif rel_type == "rival":
                        state.dimensions.respect = initial_strength * 0.7
                        state.dimensions.affection = 0
                        state.dimensions.volatility = initial_strength * 0.6
                    elif rel_type == "victim":
                        state.dimensions.trust = -initial_strength * 0.5
                        state.dimensions.respect = -initial_strength * 0.3
                        state.dimensions.influence = initial_strength * 0.7
                        state.dimensions.unresolved_conflict = initial_strength * 0.8

                    if "dimensions" in rel:
                        for dim, value in rel["dimensions"].items():
                            if hasattr(state.dimensions, dim):
                                setattr(state.dimensions, dim, value)

                    state.dimensions.clamp()
                    await relationship_manager._queue_update(state)

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

                    await PresetNPCHandler._append_relationship_entry(
                        conn,
                        canon_ctx,
                        user_id,
                        conversation_id,
                        npc_id,
                        {
                            "relationship_label": rel_type,
                            "entity_type": target_type,
                            "entity_id": target_id,
                        },
                        f"Adding preset relationship: {rel_type}",
                    )
            else:
                await PresetNPCHandler._create_default_player_relationship(
                    conn,
                    canon_ctx,
                    relationship_manager,
                    user_id,
                    conversation_id,
                    npc_id,
                    npc_data,
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
    async def _append_relationship_entry(
        conn,
        canon_ctx,
        user_id: int,
        conversation_id: int,
        npc_id: int,
        entry: Dict[str, Any],
        reason: str,
    ) -> None:
        """Append a relationship record to the NPCStats row."""

        from lore.core import canon

        rel_query = """
            SELECT relationships FROM NPCStats
            WHERE user_id=$1 AND conversation_id=$2 AND npc_id=$3
        """

        row = await conn.fetchrow(rel_query, user_id, conversation_id, npc_id)
        current_relationships: List[Dict[str, Any]] = []

        if row and row['relationships']:
            try:
                current_relationships = json.loads(row['relationships'])
            except Exception:
                current_relationships = []

        current_relationships.append(entry)

        await canon.update_entity_canonically(
            canon_ctx,
            conn,
            "NPCStats",
            npc_id,
            {"relationships": json.dumps(current_relationships)},
            reason,
        )

    @staticmethod
    async def _create_default_player_relationship(
        conn,
        canon_ctx,
        relationship_manager: OptimizedRelationshipManager,
        user_id: int,
        conversation_id: int,
        npc_id: int,
        npc_data: Dict[str, Any],
    ) -> None:
        """Create a neutral starting relationship with the player when none is provided."""

        state = await relationship_manager.get_relationship_state(
            entity1_type="npc",
            entity1_id=npc_id,
            entity2_type="player",
            entity2_id=user_id,
        )

        stats = npc_data.get("stats", {})
        state.dimensions.trust = stats.get("trust", 0)
        state.dimensions.respect = stats.get("respect", 0)
        state.dimensions.affection = stats.get("affection", 0)
        state.dimensions.intimacy = stats.get("intimacy", 0)
        state.dimensions.clamp()

        await relationship_manager._queue_update(state)

        await PresetNPCHandler._append_relationship_entry(
            conn,
            canon_ctx,
            user_id,
            conversation_id,
            npc_id,
            {
                "relationship_label": "associate",
                "entity_type": "player",
                "entity_id": user_id,
            },
            "Initializing default player relationship",
        )

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
        
        # Store mask data in special_mechanics instead of separate columns
        async with get_db_connection_context() as conn:
            # Get existing special_mechanics
            current = await conn.fetchval(
                "SELECT special_mechanics FROM NPCStats WHERE npc_id = $1",
                npc_id
            )
            mechanics = json.loads(current) if current else {}
            
            # Add mask system
            mechanics["mask_system"] = {
                "current_mask": mask_data.get("initial_mask", "default"),
                "masks_available": mask_data.get("types", {}),
                "integrity": 100
            }
            
            # Update with the combined mechanics
            await canon.update_entity_canonically(
                canon_ctx, conn, "NPCStats", npc_id,
                {
                    "special_mechanics": json.dumps(mechanics),
                    "mask_integrity": 100  # Keep this one as it's a core stat
                },
                f"Initializing mask system"
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
            # Get existing special_mechanics
            current = await conn.fetchval(
                "SELECT special_mechanics FROM NPCStats WHERE npc_id = $1",
                npc_id
            )
            mechanics = json.loads(current) if current else {}
            
            # Add dual identity
            mechanics["dual_identity"] = {
                "identities": identity_data,
                "current": identity_data.get("default_identity", "public"),
                "revealed": False
            }
            
            await canon.update_entity_canonically(
                canon_ctx, conn, "NPCStats", npc_id,
                {"special_mechanics": json.dumps(mechanics)},
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
