# logic/conflict_system/stakeholders.py  
"""
Dynamic stakeholder management and autonomous actions
"""

import logging
import json
import random
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from agents import Agent, Runner, ModelSettings
from db.connection import get_db_connection_context
from npcs.npc_relationship import NPCRelationshipManager

from .core import ConflictCore

logger = logging.getLogger(__name__)

# Stakeholder Personality Agent
personality_agent = Agent(
    name="Stakeholder Personality Agent",
    model_settings=ModelSettings(model="gpt-4o", temperature=0.8),
    instructions="""
    You embody a stakeholder's personality in conflicts.
    
    Given character profile and situation, decide:
    - What action to take
    - Who to target
    - How to pursue goals
    - When to reveal secrets
    
    Consider:
    - Public vs private motivations
    - Personality traits
    - Relationships
    - Resources
    
    Make in-character decisions that advance the narrative.
    Output decision as JSON.
    """
)

class StakeholderManager:
    """Manages stakeholder behaviors and interactions"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.core = ConflictCore(user_id, conversation_id)
    
    async def process_stakeholder_turns(self, conflict_id: int) -> List[Dict[str, Any]]:
        """Process autonomous actions for all stakeholders"""
        actions = []
        
        # Get conflict and stakeholders
        conflict = await self.core.get_conflict_details(conflict_id)
        if not conflict:
            return actions
        
        # Process each stakeholder
        for stakeholder in conflict['stakeholders']:
            if await self._should_act(stakeholder, conflict):
                action = await self._determine_action(
                    stakeholder, conflict
                )
                if action:
                    result = await self._execute_action(
                        conflict_id, stakeholder, action
                    )
                    actions.append(result)
        
        return actions
    
    async def process_stakeholder_reactions(self, conflict_id: int,
                                          trigger_event: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process stakeholder reactions to an event"""
        reactions = []
        
        conflict = await self.core.get_conflict_details(conflict_id)
        if not conflict:
            return reactions
        
        for stakeholder in conflict['stakeholders']:
            if await self._should_react(stakeholder, trigger_event):
                reaction = await self._determine_reaction(
                    stakeholder, conflict, trigger_event
                )
                if reaction:
                    reactions.append(reaction)
        
        return reactions
    
    async def create_manipulation_attempt(self, conflict_id: int,
                                        manipulator_id: int,
                                        target: str,
                                        manipulation_type: str,
                                        content: str) -> Dict[str, Any]:
        """Create a manipulation attempt"""
        async with get_db_connection_context() as conn:
            # Get manipulator details
            manipulator = await conn.fetchrow("""
                SELECT n.*, s.involvement_level
                FROM NPCStats n
                JOIN ConflictStakeholders s ON n.npc_id = s.npc_id
                WHERE n.npc_id = $1 AND s.conflict_id = $2
            """, manipulator_id, conflict_id)
            
            if not manipulator:
                return {"error": "Manipulator not found"}
            
            # Create attempt
            attempt_id = await conn.fetchval("""
                INSERT INTO PlayerManipulationAttempts
                (conflict_id, user_id, conversation_id, npc_id,
                 manipulation_type, content, goal, leverage_used,
                 intimacy_level)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                RETURNING attempt_id
            """,
            conflict_id, self.user_id, self.conversation_id,
            manipulator_id, manipulation_type, content,
            json.dumps({"target": target}),
            json.dumps(self._calculate_leverage(manipulator, manipulation_type)),
            self._calculate_intimacy(manipulator, manipulation_type)
            )
            
            return {
                "attempt_id": attempt_id,
                "manipulator": manipulator['npc_name'],
                "type": manipulation_type,
                "content": content
            }
    
    async def reveal_secret(self, conflict_id: int,
                          revealer_id: int,
                          secret_id: str,
                          audience: str = "public") -> Dict[str, Any]:
        """Reveal a stakeholder secret"""
        async with get_db_connection_context() as conn:
            async with conn.transaction():
                # Get secret
                secret = await conn.fetchrow("""
                    SELECT * FROM StakeholderSecrets
                    WHERE secret_id = $1 AND conflict_id = $2
                """, secret_id, conflict_id)
                
                if not secret or secret['is_revealed']:
                    return {"error": "Secret not found or already revealed"}
                
                # Mark as revealed
                await conn.execute("""
                    UPDATE StakeholderSecrets
                    SET is_revealed = TRUE,
                        revealed_by = $1,
                        revealed_to = $2,
                        is_public = $3,
                        revealed_at = NOW()
                    WHERE secret_id = $4
                """, revealer_id, audience, audience == "public", secret_id)
                
                # Create memory event
                await conn.execute("""
                    INSERT INTO ConflictMemoryEvents
                    (conflict_id, memory_text, significance,
                     entity_type, entity_id)
                    VALUES ($1, $2, $3, 'secret', $1)
                """,
                conflict_id,
                f"Secret revealed: {secret['content']}",
                8
                )
                
                # Increase conflict tension
                await self.core.update_conflict_progress(conflict_id, 15)
                
                return {
                    "secret_type": secret['secret_type'],
                    "content": secret['content'],
                    "impact": "high",
                    "audience": audience
                }
    
    async def _should_act(self, stakeholder: Dict[str, Any],
                        conflict: Dict[str, Any]) -> bool:
        """Determine if stakeholder should act"""
        # Base chance from involvement
        base_chance = stakeholder['involvement_level'] / 10.0
        
        # Modify by personality
        if stakeholder['dominance'] > 70:
            base_chance *= 1.3
        
        # Modify by conflict phase
        phase_multipliers = {
            'brewing': 0.5,
            'active': 1.0,
            'climax': 2.0,
            'resolution': 1.5
        }
        base_chance *= phase_multipliers.get(conflict['phase'], 1.0)
        
        return random.random() < min(base_chance, 0.8)
    
    async def _should_react(self, stakeholder: Dict[str, Any],
                          trigger: Dict[str, Any]) -> bool:
        """Determine if stakeholder should react to event"""
        # Always react if directly involved
        if stakeholder['npc_id'] in trigger.get('affected_npcs', []):
            return True
        
        # React based on involvement
        return random.random() < stakeholder['involvement_level'] / 20.0
    
    async def _determine_action(self, stakeholder: Dict[str, Any],
                              conflict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Determine stakeholder's action"""
        # Get available actions
        actions = await self._get_available_actions(stakeholder, conflict)
        
        # Use personality agent
        prompt = f"""
        You are {stakeholder['npc_name']} in conflict "{conflict['conflict_name']}".
        
        Profile:
        - Public goal: {stakeholder['public_motivation']}
        - Private goal: {stakeholder['private_motivation']}
        - Dominance: {stakeholder['dominance']}
        - Involvement: {stakeholder['involvement_level']}/10
        
        Conflict phase: {conflict['phase']} ({conflict['progress']}%)
        
        Available actions:
        {json.dumps(actions, indent=2)}
        
        Choose the best action to advance your goals.
        """
        
        result = await Runner.run(
            personality_agent,
            prompt,
            self.core.ctx
        )
        
        try:
            action = json.loads(result.final_output)
            action['stakeholder_id'] = stakeholder['npc_id']
            return action
        except:
            return None
    
    async def _get_available_actions(self, stakeholder: Dict[str, Any],
                                   conflict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get available actions for stakeholder"""
        actions = []
        
        # Basic actions always available
        actions.extend([
            {"type": "negotiate", "description": "Negotiate with another stakeholder"},
            {"type": "pressure", "description": "Apply pressure to advance goals"},
            {"type": "investigate", "description": "Gather information"}
        ])
        
        # Manipulation (for dominant NPCs)
        if stakeholder['dominance'] > 60:
            actions.append({
                "type": "manipulate",
                "description": "Manipulate player or other NPCs"
            })
        
        # Secret revelation
        async with get_db_connection_context() as conn:
            secrets = await conn.fetchval("""
                SELECT COUNT(*) FROM StakeholderSecrets
                WHERE conflict_id = $1 AND npc_id = $2 
                    AND is_revealed = FALSE
            """, conflict['conflict_id'], stakeholder['npc_id'])
            
            if secrets > 0:
                actions.append({
                    "type": "reveal_secret",
                    "description": "Reveal a secret",
                    "count": secrets
                })
        
        # Betrayal (if allied)
        if stakeholder.get('alliances'):
            actions.append({
                "type": "betray",
                "description": "Betray an ally"
            })
        
        return actions
    
    async def _execute_action(self, conflict_id: int,
                            stakeholder: Dict[str, Any],
                            action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute stakeholder action"""
        result = {
            "stakeholder_id": stakeholder['npc_id'],
            "stakeholder_name": stakeholder['npc_name'],
            "action_type": action['type'],
            "success": False
        }
        
        if action['type'] == 'manipulate':
            manipulation = await self.create_manipulation_attempt(
                conflict_id,
                stakeholder['npc_id'],
                action.get('target', 'player'),
                action.get('method', 'domination'),
                action.get('content', 'You will help me in this conflict.')
            )
            result.update(manipulation)
            result['success'] = 'error' not in manipulation
            
        elif action['type'] == 'reveal_secret':
            # Get a random unrevealed secret
            async with get_db_connection_context() as conn:
                secret = await conn.fetchrow("""
                    SELECT secret_id FROM StakeholderSecrets
                    WHERE conflict_id = $1 AND npc_id = $2 
                        AND is_revealed = FALSE
                    ORDER BY RANDOM() LIMIT 1
                """, conflict_id, stakeholder['npc_id'])
                
                if secret:
                    revelation = await self.reveal_secret(
                        conflict_id,
                        stakeholder['npc_id'],
                        secret['secret_id'],
                        action.get('audience', 'public')
                    )
                    result.update(revelation)
                    result['success'] = 'error' not in revelation
        
        else:
            # Generic action
            result['success'] = True
            result['description'] = f"{stakeholder['npc_name']} takes {action['type']} action"
        
        return result
    
    async def _determine_reaction(self, stakeholder: Dict[str, Any],
                                conflict: Dict[str, Any],
                                trigger: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Determine stakeholder's reaction to event"""
        # Simple reaction logic - could be expanded
        if trigger.get('type') == 'betrayal' and stakeholder['npc_id'] == trigger.get('victim'):
            return {
                "reactor_id": stakeholder['npc_id'],
                "reactor_name": stakeholder['npc_name'],
                "reaction_type": "retaliation",
                "target": trigger.get('betrayer'),
                "intensity": "high"
            }
        
        return None
    
    def _calculate_leverage(self, manipulator: Dict[str, Any],
                          manipulation_type: str) -> Dict[str, Any]:
        """Calculate manipulation leverage"""
        if manipulation_type == "domination":
            return {
                "type": "authority",
                "strength": manipulator['dominance']
            }
        elif manipulation_type == "seduction":
            return {
                "type": "attraction",
                "strength": manipulator.get('closeness', 50)
            }
        else:
            return {
                "type": "general",
                "strength": 50
            }
    
    def _calculate_intimacy(self, manipulator: Dict[str, Any],
                          manipulation_type: str) -> int:
        """Calculate intimacy level of manipulation"""
        base = manipulator.get('closeness', 0) // 10
        
        if manipulation_type == "seduction":
            return min(10, base + 3)
        elif manipulation_type == "domination":
            return min(10, base + manipulator['dominance'] // 20)
        else:
            return base
