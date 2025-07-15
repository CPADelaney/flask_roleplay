# logic/npc_narrative_progression.py

import logging
from typing import Dict, List, Any, Optional, NamedTuple
from datetime import datetime
import asyncpg

from db.connection import get_db_connection_context
from memory.wrapper import MemorySystem
from lore.core.lore_system import LoreSystem
from agents import RunContextWrapper

logger = logging.getLogger(__name__)

class NPCNarrativeStage(NamedTuple):
    """Represents a narrative stage for an NPC relationship."""
    name: str
    description: str
    required_corruption: int
    required_dependency: int
    required_realization: int

# Define the narrative stages for NPC relationships
NPC_NARRATIVE_STAGES = [
    NPCNarrativeStage(
        name="Innocent Beginning",
        description="The player is unaware of this NPC's true nature or intentions. The NPC maintains a friendly facade.",
        required_corruption=0,
        required_dependency=0,
        required_realization=0
    ),
    NPCNarrativeStage(
        name="First Doubts",
        description="The player begins to notice inconsistencies in this NPC's behavior. Subtle control attempts become visible.",
        required_corruption=20,
        required_dependency=15,
        required_realization=20
    ),
    NPCNarrativeStage(
        name="Creeping Realization",
        description="The player has moments of clarity about this NPC's manipulation. The NPC is less careful about hiding their control.",
        required_corruption=40,
        required_dependency=35,
        required_realization=40
    ),
    NPCNarrativeStage(
        name="Veil Thinning",
        description="This NPC openly manipulates and controls the player. The facade is largely abandoned.",
        required_corruption=60,
        required_dependency=55,
        required_realization=60
    ),
    NPCNarrativeStage(
        name="Full Revelation",
        description="The player fully understands this NPC's nature and their relationship dynamic. No pretense remains.",
        required_corruption=80,
        required_dependency=75,
        required_realization=80
    )
]

async def get_npc_narrative_stage(user_id: int, conversation_id: int, npc_id: int) -> NPCNarrativeStage:
    """
    Get the current narrative stage for a specific NPC relationship.
    """
    try:
        async with get_db_connection_context() as conn:
            # First try dedicated table
            row = await conn.fetchrow("""
                SELECT narrative_stage, corruption, dependency, realization_level
                FROM NPCNarrativeProgression
                WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
            """, user_id, conversation_id, npc_id)
            
            if row:
                stage_name = row['narrative_stage']
                # Find matching stage
                for stage in NPC_NARRATIVE_STAGES:
                    if stage.name == stage_name:
                        return stage
            
            # If no record exists, check the relationship stats
            rel_row = await conn.fetchrow("""
                SELECT link_level, link_type
                FROM SocialLinks
                WHERE user_id = $1 AND conversation_id = $2
                AND entity1_type = 'npc' AND entity1_id = $3
                AND entity2_type = 'player'
            """, user_id, conversation_id, npc_id)
            
            if rel_row:
                # Calculate stage based on relationship level
                link_level = rel_row['link_level'] or 0
                
                # Simple mapping based on link level
                if link_level < 20:
                    return NPC_NARRATIVE_STAGES[0]  # Innocent Beginning
                elif link_level < 40:
                    return NPC_NARRATIVE_STAGES[1]  # First Doubts
                elif link_level < 60:
                    return NPC_NARRATIVE_STAGES[2]  # Creeping Realization
                elif link_level < 80:
                    return NPC_NARRATIVE_STAGES[3]  # Veil Thinning
                else:
                    return NPC_NARRATIVE_STAGES[4]  # Full Revelation
            
            # Default to first stage
            return NPC_NARRATIVE_STAGES[0]
            
    except Exception as e:
        logger.error(f"Error getting NPC narrative stage: {e}")
        return NPC_NARRATIVE_STAGES[0]

async def progress_npc_narrative_stage(
    user_id: int, 
    conversation_id: int, 
    npc_id: int,
    corruption_change: int = 0,
    dependency_change: int = 0,
    realization_change: int = 0,
    force_stage: Optional[str] = None
) -> Dict[str, Any]:
    """
    Progress or update the narrative stage for a specific NPC.
    """
    try:
        async with get_db_connection_context() as conn:
            # Get or create progression record
            current = await conn.fetchrow("""
                SELECT * FROM NPCNarrativeProgression
                WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
            """, user_id, conversation_id, npc_id)
            
            if not current:
                # Create new record
                await conn.execute("""
                    INSERT INTO NPCNarrativeProgression 
                    (user_id, conversation_id, npc_id, narrative_stage, corruption, dependency, realization_level)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                """, user_id, conversation_id, npc_id, "Innocent Beginning", 0, 0, 0)
                
                current = {
                    'narrative_stage': 'Innocent Beginning',
                    'corruption': 0,
                    'dependency': 0,
                    'realization_level': 0,
                    'stage_history': []
                }
            
            # Calculate new values
            new_corruption = max(0, min(100, (current['corruption'] or 0) + corruption_change))
            new_dependency = max(0, min(100, (current['dependency'] or 0) + dependency_change))
            new_realization = max(0, min(100, (current['realization_level'] or 0) + realization_change))
            
            # Determine new stage
            old_stage = current['narrative_stage']
            new_stage = old_stage
            
            if force_stage:
                new_stage = force_stage
            else:
                # Find appropriate stage based on stats
                for stage in reversed(NPC_NARRATIVE_STAGES):
                    if (new_corruption >= stage.required_corruption and 
                        new_dependency >= stage.required_dependency and
                        new_realization >= stage.required_realization):
                        new_stage = stage.name
                        break
            
            # Update record
            stage_changed = new_stage != old_stage
            
            # Update stage history if changed
            stage_history = current.get('stage_history', [])
            if stage_changed:
                stage_history.append({
                    'from': old_stage,
                    'to': new_stage,
                    'timestamp': datetime.now().isoformat(),
                    'stats': {
                        'corruption': new_corruption,
                        'dependency': new_dependency,
                        'realization': new_realization
                    }
                })
            
            await conn.execute("""
                UPDATE NPCNarrativeProgression
                SET narrative_stage = $4,
                    corruption = $5,
                    dependency = $6,
                    realization_level = $7,
                    stage_history = $8,
                    stage_updated_at = CASE WHEN $9 THEN CURRENT_TIMESTAMP ELSE stage_updated_at END
                WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
            """, user_id, conversation_id, npc_id, new_stage, new_corruption, 
                new_dependency, new_realization, stage_history, stage_changed)
            
            # Create memory if stage changed
            if stage_changed:
                memory_system = await MemorySystem.get_instance(user_id, conversation_id)
                
                # Get NPC name
                npc_name = await conn.fetchval("""
                    SELECT npc_name FROM NPCStats WHERE npc_id = $1
                """, npc_id)
                
                await memory_system.remember(
                    entity_type="player",
                    entity_id=user_id,
                    memory_text=f"My relationship with {npc_name} has evolved to a new stage: {new_stage}",
                    importance="high",
                    tags=["narrative_progression", f"npc_{npc_id}", new_stage.lower().replace(" ", "_")]
                )
            
            return {
                'success': True,
                'old_stage': old_stage,
                'new_stage': new_stage,
                'stage_changed': stage_changed,
                'stats': {
                    'corruption': new_corruption,
                    'dependency': new_dependency,
                    'realization': new_realization
                }
            }
            
    except Exception as e:
        logger.error(f"Error progressing NPC narrative stage: {e}")
        return {'success': False, 'error': str(e)}

async def check_for_npc_revelation(user_id: int, conversation_id: int, npc_id: int) -> Optional[Dict[str, Any]]:
    """
    Check if conditions are right for a revelation specific to this NPC.
    """
    stage = await get_npc_narrative_stage(user_id, conversation_id, npc_id)
    
    # Only generate revelations after first stage
    if stage.name == "Innocent Beginning":
        return None
    
    try:
        async with get_db_connection_context() as conn:
            # Check recent revelations for this NPC
            recent_count = await conn.fetchval("""
                SELECT COUNT(*) FROM PlayerJournal
                WHERE user_id = $1 AND conversation_id = $2 
                AND entry_type = 'npc_revelation'
                AND entry_metadata->>'npc_id' = $3::text
                AND timestamp > NOW() - INTERVAL '3 days'
            """, user_id, conversation_id, str(npc_id))
            
            if recent_count > 1:
                return None
            
            # Get NPC data
            npc_data = await conn.fetchrow("""
                SELECT npc_name, dominance, personality_traits
                FROM NPCStats
                WHERE npc_id = $1 AND user_id = $2 AND conversation_id = $3
            """, npc_id, user_id, conversation_id)
            
            if not npc_data:
                return None
            
            # Generate revelation based on stage
            revelation_templates = {
                "First Doubts": [
                    f"Why does {npc_data['npc_name']} always seem to know what I'm thinking before I say it?",
                    f"I noticed {npc_data['npc_name']} steering our conversation again. She's subtle, but it's there.",
                    f"There's something about the way {npc_data['npc_name']} phrases her suggestions... I always end up agreeing."
                ],
                "Creeping Realization": [
                    f"{npc_data['npc_name']} isn't as innocent as she appears. Those 'suggestions' are really commands.",
                    f"I'm starting to see the pattern with {npc_data['npc_name']}. She's been guiding me all along.",
                    f"The way {npc_data['npc_name']} looks at me when I obey... she's enjoying this control."
                ],
                "Veil Thinning": [
                    f"{npc_data['npc_name']} doesn't even pretend anymore. She tells me what to do and I... I do it.",
                    f"I understand now what {npc_data['npc_name']} has been doing. The worst part? I don't want it to stop.",
                    f"{npc_data['npc_name']}'s control over me is nearly complete. When did I become so dependent on her?"
                ],
                "Full Revelation": [
                    f"{npc_data['npc_name']} owns me now. We both know it. There's no going back.",
                    f"I belong to {npc_data['npc_name']}. The transformation is complete.",
                    f"{npc_data['npc_name']} has made me exactly what she wanted. And I helped her do it."
                ]
            }
            
            templates = revelation_templates.get(stage.name, [])
            if not templates:
                return None
            
            revelation_text = random.choice(templates)
            
            # Create the revelation
            journal_id = await conn.fetchval("""
                INSERT INTO PlayerJournal 
                (user_id, conversation_id, entry_type, entry_text, entry_metadata, timestamp)
                VALUES ($1, $2, 'npc_revelation', $3, $4, CURRENT_TIMESTAMP)
                RETURNING id
            """, user_id, conversation_id, revelation_text, {'npc_id': npc_id, 'stage': stage.name})
            
            return {
                'type': 'npc_revelation',
                'npc_id': npc_id,
                'npc_name': npc_data['npc_name'],
                'stage': stage.name,
                'revelation_text': revelation_text,
                'journal_id': journal_id
            }
            
    except Exception as e:
        logger.error(f"Error checking for NPC revelation: {e}")
        return None
