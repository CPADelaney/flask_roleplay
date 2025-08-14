# logic/conflict_system/conflict_canon_integration.py
"""
Integration layer between the Conflict System and the Canon System.
Ensures all conflict operations respect canonical consistency.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass

from db.connection import get_db_connection_context
from lore.core.canon import (
    ensure_canonical_context,
    log_canonical_event,
    find_or_create_npc,
    find_or_create_location,
    find_or_create_faction,
    find_or_create_event,
    update_entity_canonically,
    CanonicalContext
)
from lore.core.validation import CanonValidationAgent
from embedding.vector_store import generate_embedding
from agents import Runner

logger = logging.getLogger(__name__)

# ===============================================================================
# UPDATED CONFLICT CANON MANAGER
# ===============================================================================

class ImprovedConflictCanonManager:
    """
    Improved version that properly integrates with canon.py
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        # Create canonical context for all operations
        self.ctx = CanonicalContext(user_id, conversation_id)
        
        # Lazy-loaded agents (keep existing)
        self._lore_integrator = None
        self._precedent_analyzer = None
        self._cultural_interpreter = None
        self._legacy_writer = None
        self._reference_generator = None
    
    async def evaluate_for_canon(
        self,
        conflict_id: int,
        resolution_data: Dict[str, Any]
    ) -> Optional['CanonicalEvent']:
        """Evaluate if a conflict resolution should become canonical"""
        
        async with get_db_connection_context() as conn:
            conflict = await conn.fetchrow("""
                SELECT * FROM Conflicts WHERE conflict_id = $1
            """, conflict_id)
            
            stakeholders = await conn.fetch("""
                SELECT * FROM conflict_stakeholders WHERE conflict_id = $1
            """, conflict_id)
        
        # ... existing evaluation logic ...
        
        if data['should_be_canonical']:
            return await self._create_canonical_event(
                conflict_id,
                conflict,
                resolution_data,
                CanonEventType(data['event_type']),
                data['significance']
            )
        
        return None
    
    async def _create_canonical_event(
        self,
        conflict_id: int,
        conflict: Dict[str, Any],
        resolution_data: Dict[str, Any],
        event_type: 'CanonEventType',
        significance: float
    ) -> 'CanonicalEvent':
        """Create a new canonical event using canon.py functions"""
        
        async with get_db_connection_context() as conn:
            # Generate canonical description (existing logic)
            data = await self._generate_canonical_description(
                conflict, resolution_data, event_type, significance
            )
            
            # Generate legacy
            legacy = await self._generate_legacy(conflict, resolution_data, data)
            
            # Create embedding for semantic search
            embedding_text = f"{data['canonical_name']} {data['canonical_description']} {legacy}"
            embedding = await generate_embedding(embedding_text)
            
            # Store in database with proper canon logging
            event_id = await conn.fetchval("""
                INSERT INTO canonical_events
                (conflict_id, event_type, name, description,
                 significance, cultural_impact, creates_precedent, legacy, embedding)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                RETURNING event_id
            """, conflict_id, event_type.value, data['canonical_name'],
            data['canonical_description'], significance,
            json.dumps(data['cultural_impact']),
            data['creates_precedent'], legacy, embedding)
            
            # Use canon.py's log function for consistency
            await log_canonical_event(
                self.ctx, conn,
                f"Conflict '{conflict['conflict_name']}' became canonical: {data['canonical_name']}",
                tags=['conflict', 'canonical', event_type.value],
                significance=int(significance * 10)  # Convert 0-1 to 1-10 scale
            )
        
        return CanonicalEvent(
            event_id=event_id,
            conflict_id=conflict_id,
            event_type=event_type,
            name=data['canonical_name'],
            description=data['canonical_description'],
            significance=significance,
            cultural_impact=data['cultural_impact'],
            referenced_by=[],
            creates_precedent=data['creates_precedent'],
            legacy=legacy
        )
    
    async def check_for_duplicate_conflicts(
        self,
        conflict_name: str,
        conflict_type: str,
        participants: List[int]
    ) -> Optional[int]:
        """Check for semantically similar conflicts before creating new ones"""
        
        async with get_db_connection_context() as conn:
            # Create embedding for the proposed conflict
            embedding_text = f"{conflict_name} {conflict_type} involving NPCs {participants}"
            search_vector = await generate_embedding(embedding_text)
            
            # Search for similar conflicts
            similar_conflicts = await conn.fetch("""
                SELECT conflict_id, conflict_name, conflict_type, description,
                       1 - (embedding <=> $1) AS similarity
                FROM Conflicts
                WHERE user_id = $2 AND conversation_id = $3
                AND is_active = true
                AND embedding IS NOT NULL
                AND 1 - (embedding <=> $1) > 0.85
                ORDER BY embedding <=> $1
                LIMIT 3
            """, search_vector, self.user_id, self.conversation_id)
            
            if similar_conflicts:
                # Use validation agent to confirm duplicates
                validation_agent = CanonValidationAgent()
                for conflict in similar_conflicts:
                    prompt = f"""
                    Are these the same conflict?
                    
                    Proposed: {conflict_name} ({conflict_type})
                    Participants: {participants}
                    
                    Existing: {conflict['conflict_name']} ({conflict['conflict_type']})
                    Description: {conflict['description'][:200]}
                    Similarity: {conflict['similarity']:.2f}
                    
                    Answer only 'true' or 'false'.
                    """
                    
                    result = await Runner.run(validation_agent.agent, prompt)
                    if result.final_output.strip().lower() == 'true':
                        logger.info(f"Conflict '{conflict_name}' matched to existing conflict ID {conflict['conflict_id']}")
                        return conflict['conflict_id']
            
            return None


# ===============================================================================
# ENHANCED SYNTHESIZER WITH CANON INTEGRATION
# ===============================================================================

class CanonIntegratedConflictSynthesizer:
    """
    Enhanced synthesizer that properly uses canon.py functions
    """
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.ctx = CanonicalContext(user_id, conversation_id)
        
        # Keep existing subsystem structure
        self._subsystems = {}
        self._subsystem_loaders = {
            'resolution': self._load_resolution_system,
            'generation': self._load_generation_system,
            'stakeholders': self._load_stakeholder_system,
            'integration': self._load_integration_system,
            'analysis': self._load_analysis_system,
            'rewards': self._load_rewards_system,
            'canon': self._load_canon_system  # NEW: Add canon subsystem
        }
        
        # Keep existing state management
        self.active_conflicts = {}
        self.orchestration_history = []
        
    async def _load_canon_system(self):
        """Load the canon integration system"""
        return ImprovedConflictCanonManager(self.user_id, self.conversation_id)
    
    async def _create_conflict_with_canon(self, conflict_data: Dict[str, Any]) -> int:
        """Create a conflict with full canon integration"""
        
        async with get_db_connection_context() as conn:
            # Check for duplicates first
            canon_manager = await self.get_subsystem('canon')
            duplicate_id = await canon_manager.check_for_duplicate_conflicts(
                conflict_data.get('conflict_name', 'Unknown Conflict'),
                conflict_data.get('conflict_type', 'generic'),
                conflict_data.get('participants', [])
            )
            
            if duplicate_id:
                return duplicate_id
            
            # Ensure all NPCs exist canonically
            canonical_participants = []
            for participant in conflict_data.get('participants', []):
                if isinstance(participant, dict):
                    npc_id = await find_or_create_npc(
                        self.ctx, conn,
                        npc_name=participant.get('name', 'Unknown'),
                        role=participant.get('role'),
                        affiliations=participant.get('affiliations', [])
                    )
                    canonical_participants.append(npc_id)
                elif isinstance(participant, int):
                    canonical_participants.append(participant)
                else:  # String name
                    npc_id = await find_or_create_npc(self.ctx, conn, str(participant))
                    canonical_participants.append(npc_id)
            
            # Ensure location exists if specified
            location_id = None
            if conflict_data.get('location'):
                location_name = await find_or_create_location(
                    self.ctx, conn,
                    conflict_data['location'],
                    location_type='conflict_zone'
                )
                conflict_data['location'] = location_name
            
            # Create embedding for the conflict
            embedding_text = (
                f"{conflict_data.get('conflict_name', '')} "
                f"{conflict_data.get('conflict_type', '')} "
                f"{conflict_data.get('description', '')}"
            )
            embedding = await generate_embedding(embedding_text)
            
            # Create the conflict
            conflict_id = await conn.fetchval("""
                INSERT INTO Conflicts (
                    user_id, conversation_id, conflict_name, conflict_type,
                    description, phase, intensity, progress, is_active,
                    location, complexity, embedding, created_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, CURRENT_TIMESTAMP)
                RETURNING conflict_id
            """, 
            self.user_id, self.conversation_id,
            conflict_data.get('conflict_name', 'Unknown Conflict'),
            conflict_data.get('conflict_type', 'generic'),
            conflict_data.get('description', ''),
            conflict_data.get('phase', 'brewing'),
            conflict_data.get('intensity', 0.5),
            0.0, True,
            conflict_data.get('location'),
            conflict_data.get('complexity', 0.5),
            embedding)
            
            # Add stakeholders
            for npc_id in canonical_participants:
                await conn.execute("""
                    INSERT INTO conflict_stakeholders 
                    (conflict_id, npc_id, stake, influence, position)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (conflict_id, npc_id) DO NOTHING
                """, conflict_id, npc_id, 
                'involved', 0.5, 'neutral')
            
            # Log canonical event
            await log_canonical_event(
                self.ctx, conn,
                f"Conflict '{conflict_data.get('conflict_name')}' initiated with {len(canonical_participants)} stakeholders",
                tags=['conflict', 'creation', conflict_data.get('conflict_type', 'generic')],
                significance=7
            )
            
            return conflict_id
    
    async def create_conflict(self, conflict_data: Dict[str, Any]) -> 'OrchestrationResult':
        """Override to use canon-aware creation"""
        
        try:
            # Use the canon-aware creation method
            conflict_id = await self._create_conflict_with_canon(conflict_data)
            
            # Continue with existing orchestration logic
            strategy = await self._determine_orchestration_strategy('create', conflict_data)
            module_results = {'conflict_id': conflict_id}
            
            # ... rest of existing orchestration ...
            
            return OrchestrationResult(
                success=True,
                mode=OrchestrationMode(strategy['mode']),
                modules_engaged=strategy['modules'],
                primary_result={'conflict_id': conflict_id},
                module_results=module_results,
                cascade_effects=[],
                state_changes={'created': conflict_id},
                narrative_impact=0.8
            )
            
        except Exception as e:
            logger.error(f"Error in canon-integrated conflict creation: {e}")
            return OrchestrationResult(
                success=False,
                mode=OrchestrationMode.SEQUENTIAL,
                modules_engaged=[],
                primary_result={'error': str(e)},
                module_results={},
                cascade_effects=[],
                state_changes={},
                narrative_impact=0.0
            )


# ===============================================================================
# HELPER FUNCTIONS FOR MIGRATION
# ===============================================================================

async def ensure_conflict_tables_have_embeddings(conn):
    """Ensure conflict-related tables have embedding columns for semantic search"""
    
    tables_to_update = [
        ('Conflicts', 'conflict_name, conflict_type, description'),
        ('canonical_events', 'name, description, legacy'),
        ('conflict_resolution_paths', 'path_name, description'),
        ('cultural_traditions', 'tradition_name, description')
    ]
    
    for table, text_fields in tables_to_update:
        # Check if embedding column exists
        exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = $1 AND column_name = 'embedding'
            )
        """, table.lower())
        
        if not exists:
            logger.info(f"Adding embedding column to {table}")
            await conn.execute(f"""
                ALTER TABLE {table} 
                ADD COLUMN embedding VECTOR(1536)
            """)
            
            # Create index for fast similarity search
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table.lower()}_embedding_hnsw
                ON {table}
                USING hnsw (embedding vector_cosine_ops)
            """)


async def migrate_existing_conflicts_to_canon(user_id: int, conversation_id: int):
    """One-time migration to add embeddings to existing conflicts"""
    
    async with get_db_connection_context() as conn:
        # Ensure tables have embedding columns
        await ensure_conflict_tables_have_embeddings(conn)
        
        # Get conflicts without embeddings
        conflicts = await conn.fetch("""
            SELECT conflict_id, conflict_name, conflict_type, description
            FROM Conflicts
            WHERE user_id = $1 AND conversation_id = $2
            AND embedding IS NULL
        """, user_id, conversation_id)
        
        for conflict in conflicts:
            # Generate embedding
            embedding_text = (
                f"{conflict['conflict_name']} "
                f"{conflict['conflict_type']} "
                f"{conflict['description'] or ''}"
            )
            embedding = await generate_embedding(embedding_text)
            
            # Update conflict
            await conn.execute("""
                UPDATE Conflicts
                SET embedding = $1
                WHERE conflict_id = $2
            """, embedding, conflict['conflict_id'])
            
            logger.info(f"Added embedding to conflict {conflict['conflict_id']}")
        
        # Do the same for canonical_events
        events = await conn.fetch("""
            SELECT event_id, name, description, legacy
            FROM canonical_events
            WHERE embedding IS NULL
        """)
        
        for event in events:
            embedding_text = (
                f"{event['name']} "
                f"{event['description']} "
                f"{event['legacy'] or ''}"
            )
            embedding = await generate_embedding(embedding_text)
            
            await conn.execute("""
                UPDATE canonical_events
                SET embedding = $1
                WHERE event_id = $2
            """, embedding, event['event_id'])
            
            logger.info(f"Added embedding to canonical event {event['event_id']}")


# ===============================================================================
# WRAPPER FUNCTIONS FOR BACKWARDS COMPATIBILITY
# ===============================================================================

def wrap_context(ctx_or_wrapper) -> CanonicalContext:
    """Convert RunContextWrapper to CanonicalContext"""
    if isinstance(ctx_or_wrapper, CanonicalContext):
        return ctx_or_wrapper
    elif hasattr(ctx_or_wrapper, 'data'):
        # It's a RunContextWrapper
        return CanonicalContext(
            user_id=ctx_or_wrapper.data.get('user_id'),
            conversation_id=ctx_or_wrapper.data.get('conversation_id')
        )
    else:
        # Try to extract from dict or object
        return ensure_canonical_context(ctx_or_wrapper)


# Update the function_tool decorators to use proper context
from agents import function_tool, RunContextWrapper

@function_tool
async def canonize_conflict_resolution(
    ctx: RunContextWrapper,
    conflict_id: int,
    resolution_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Evaluate and potentially canonize a conflict resolution"""
    
    # Convert context
    canonical_ctx = wrap_context(ctx)
    
    manager = ImprovedConflictCanonManager(
        canonical_ctx.user_id, 
        canonical_ctx.conversation_id
    )
    
    # Rest of the function remains the same...
    canonical_event = await manager.evaluate_for_canon(conflict_id, resolution_data)
    
    if canonical_event:
        references = await manager.generate_canon_references(
            canonical_event.event_id,
            "formal"
        )
        
        return {
            'became_canonical': True,
            'event_name': canonical_event.name,
            'significance': canonical_event.significance,
            'legacy': canonical_event.legacy,
            'sample_references': references[:3]
        }
    else:
        return {
            'became_canonical': False,
            'reason': 'Not significant enough for canon'
        }


@function_tool
async def check_conflict_lore_alignment(
    ctx: RunContextWrapper,
    conflict_type: str,
    participants: List[int]
) -> Dict[str, Any]:
    """Check if a potential conflict aligns with established lore"""
    
    # Convert context
    canonical_ctx = wrap_context(ctx)
    
    manager = ImprovedConflictCanonManager(
        canonical_ctx.user_id,
        canonical_ctx.conversation_id
    )
    
    context = {
        'conflict_type': conflict_type,
        'participants': participants,
        'location': ctx.data.get('location', 'unknown')
    }
    
    compliance = await manager.check_lore_compliance(conflict_type, context)
    
    return compliance
