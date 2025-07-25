# npcs/preset_npc_handler.py
from typing import Dict, Any
import json
import logging

logger = logging.getLogger(__name__)

class PresetNPCHandler:
    """Handles creation of rich, preset NPCs"""
    
    @staticmethod
    async def create_detailed_npc(ctx, npc_data: Dict[str, Any], story_context: Dict[str, Any]) -> int:
        """Create a detailed NPC from preset data"""
        user_id = ctx.context["user_id"]
        conversation_id = ctx.context["conversation_id"]
        
        from db.connection import get_db_connection_context
        from lore.core.lore_system import LoreSystem
        
        # Initialize lore system for canonical creation
        lore_system = await LoreSystem.get_instance(user_id, conversation_id)
        
        # Prepare NPC data
        npc_creation_data = {
            "npc_name": npc_data["name"],
            "introduced": False,  # Not yet met
            "archetypes": [npc_data["archetype"]],
            "physical_description": PresetNPCHandler._build_physical_description(npc_data),
            "personality_traits": npc_data.get("traits", []),
            "dominance": npc_data["stats"]["dominance"],
            "cruelty": npc_data["stats"]["cruelty"],
            "closeness": npc_data["stats"]["closeness"],
            "trust": npc_data["stats"]["trust"],
            "respect": npc_data["stats"]["respect"],
            "intensity": npc_data["stats"]["intensity"],
            "hobbies": npc_data["personality"]["hobbies"],
            "likes": npc_data["personality"]["likes"],
            "dislikes": npc_data["personality"]["dislikes"],
            "role": npc_data["role"],
            "sex": npc_data.get("sex", "female"),
            "age": npc_data.get("age", 30),
            "current_location": PresetNPCHandler._get_initial_location(npc_data),
            "schedule": PresetNPCHandler._convert_schedule(npc_data),
            "memory": PresetNPCHandler._create_initial_memories(npc_data),
            "personality_patterns": PresetNPCHandler._create_personality_patterns(npc_data)
        }
        
        # Create through LoreSystem
        result = await lore_system.propose_and_enact_change(
            ctx=ctx,
            entity_type="NPCStats",
            entity_identifier=None,  # New creation
            updates=npc_creation_data,
            reason=f"Creating preset NPC for story: {story_context.get('story_name', 'Unknown')}"
        )
        
        if result["status"] == "committed":
            npc_id = result["entity_id"]
            
            # Add special mechanics as metadata
            if "special_mechanics" in npc_data:
                await PresetNPCHandler._add_special_mechanics(
                    ctx, npc_id, npc_data["special_mechanics"], lore_system
                )
            
            # Add backstory as memories
            if "backstory" in npc_data:
                await PresetNPCHandler._add_backstory_memories(
                    ctx, npc_id, npc_data["backstory"], lore_system
                )
            
            # Set up relationship mechanics
            if "relationship_mechanics" in npc_data:
                await PresetNPCHandler._setup_relationship_mechanics(
                    ctx, npc_id, npc_data["relationship_mechanics"], lore_system
                )
            
            logger.info(f"Created detailed preset NPC: {npc_data['name']} (ID: {npc_id})")
            return npc_id
        else:
            raise Exception(f"Failed to create NPC: {result.get('message', 'Unknown error')}")
    
    @staticmethod
    def _build_physical_description(npc_data: Dict[str, Any]) -> str:
        """Build complete physical description"""
        desc_parts = []
        
        if "physical_description" in npc_data:
            pd = npc_data["physical_description"]
            desc_parts.append(pd.get("base", ""))
            if "style" in pd:
                desc_parts.append(f"Style: {pd['style']}")
            if "tells" in pd:
                desc_parts.append(f"Notable behavior: {pd['tells']}")
            if "presence" in pd:
                desc_parts.append(f"Presence: {pd['presence']}")
        
        return " ".join(desc_parts)
    
    @staticmethod
    def _convert_schedule(npc_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert schedule to game format"""
        if "schedule" not in npc_data:
            return {}
        
        # Convert the detailed schedule format
        schedule = {}
        for day, periods in npc_data["schedule"].items():
            if isinstance(periods, dict):
                schedule[day] = periods
            else:
                # Simple format
                schedule[day] = {"all_day": periods}
        
        return schedule
    
    @staticmethod
    def _create_personality_patterns(npc_data: Dict[str, Any]) -> list:
        """Create personality pattern entries"""
        patterns = []
        
        # Add dialogue patterns
        if "dialogue_patterns" in npc_data:
            dp = npc_data["dialogue_patterns"]
            for trust_level, dialogue in dp.items():
                if trust_level.startswith("trust_"):
                    patterns.append({
                        "trigger": trust_level,
                        "response_style": dialogue
                    })
        
        # Add behavioral patterns from traits
        for trait in npc_data.get("traits", []):
            patterns.append({
                "trait": trait,
                "influences": "behavior and dialogue"
            })
        
        return patterns
    
    @staticmethod
    def _get_initial_location(npc_data: Dict[str, Any]) -> str:
        """Get initial location for NPC"""
        # Check various fields where location might be stored
        return (npc_data.get('current_location') or 
                npc_data.get('initial_location') or
                npc_data.get('default_location') or
                'Town Square')
    
    @staticmethod
    def _create_initial_memories(npc_data: Dict[str, Any]) -> list:
        """Create initial memory entries for NPC"""
        memories = []
        
        # Add any preset memories
        if 'memories' in npc_data:
            for memory in npc_data['memories']:
                if isinstance(memory, str):
                    memories.append({
                        'text': memory,
                        'importance': 0.8,
                        'timestamp': 'founding'
                    })
                elif isinstance(memory, dict):
                    memories.append(memory)
        
        # Add role-based memories
        if 'role' in npc_data:
            memories.append({
                'text': f"I am {npc_data['role']} in this place.",
                'importance': 0.6,
                'timestamp': 'founding'
            })
        
        return memories
    
    @staticmethod
    async def _add_special_mechanics(ctx, npc_id: int, mechanics: Dict[str, Any], lore_system):
        """Add special mechanics as NPC metadata"""
        from db.connection import get_db_connection_context
        
        async with get_db_connection_context() as conn:
            # Store special mechanics in a dedicated table or as JSON metadata
            for mechanic_type, mechanic_data in mechanics.items():
                await conn.execute("""
                    INSERT INTO npc_special_mechanics 
                    (user_id, conversation_id, npc_id, mechanic_type, mechanic_data)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (user_id, conversation_id, npc_id, mechanic_type)
                    DO UPDATE SET mechanic_data = EXCLUDED.mechanic_data
                """, ctx.context['user_id'], ctx.context['conversation_id'], 
                    npc_id, mechanic_type, json.dumps(mechanic_data))
    
    @staticmethod
    async def _add_backstory_memories(ctx, npc_id: int, backstory: Dict[str, Any], lore_system):
        """Add backstory elements as NPC memories"""
        from memory.wrapper import MemorySystem
        
        memory_system = await MemorySystem.get_instance(
            ctx.context['user_id'], ctx.context['conversation_id']
        )
        
        # Add different backstory elements as memories
        for key, value in backstory.items():
            if isinstance(value, str):
                await memory_system.add_memory(
                    entity_type="npc",
                    entity_id=npc_id,
                    memory_text=value,
                    importance=0.9,
                    tags=[f"backstory_{key}", "founding_memory"]
                )
    
    @staticmethod
    async def _setup_relationship_mechanics(ctx, npc_id: int, mechanics: Dict[str, Any], lore_system):
        """Set up relationship tracking mechanics"""
        # Store trust builders/breakers and other relationship mechanics
        updates = {
            'relationship_mechanics': json.dumps(mechanics)
        }
        
        await lore_system.propose_and_enact_change(
            ctx=ctx,
            entity_type="NPCStats",
            entity_identifier=npc_id,
            updates=updates,
            reason="Setting up relationship mechanics"
        )
