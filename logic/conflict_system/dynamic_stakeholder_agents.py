# logic/conflict_system/dynamic_stakeholder_agents.py
"""
Dynamic Stakeholder Agent System for autonomous conflict participation
Refactored to use the new dynamic_relationships system
"""

import logging
import json
import random
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from agents import Agent, function_tool, ModelSettings, RunContextWrapper, Runner
from db.connection import get_db_connection_context
from lore.core import canon
from logic.dynamic_relationships import (
    OptimizedRelationshipManager,
    RelationshipDimensions,
    process_relationship_interaction_tool,
    get_relationship_summary_tool
)
from logic.conflict_system.conflict_agents import (
    ConflictContext,
    initialize_conflict_assistants,
    ask_assistant
)

logger = logging.getLogger(__name__)

class StakeholderAutonomySystem:
    """System for autonomous stakeholder actions in conflicts"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.relationship_manager = OptimizedRelationshipManager(user_id, conversation_id)
        self.context = ConflictContext(user_id, conversation_id)
        self._assistants = None  # Lazy initialization
    
    async def _get_assistants(self):
        """Get or initialize assistants"""
        if self._assistants is None:
            self._assistants = await initialize_conflict_assistants()
        return self._assistants
        
    async def process_stakeholder_turn(self, conflict_id: int) -> List[Dict[str, Any]]:
        """Process autonomous actions for all stakeholders in a conflict"""
        actions_taken = []
        
        async with get_db_connection_context() as conn:
            # Get all stakeholders
            stakeholders = await conn.fetch("""
                SELECT s.*, n.*, 
                       array_agg(DISTINCT ss.secret_id) FILTER (WHERE ss.is_revealed = FALSE) as unrevealed_secrets
                FROM ConflictStakeholders s
                JOIN NPCStats n ON s.npc_id = n.npc_id
                LEFT JOIN StakeholderSecrets ss ON ss.npc_id = s.npc_id AND ss.conflict_id = s.conflict_id
                WHERE s.conflict_id = $1
                GROUP BY s.id, n.npc_id
                ORDER BY s.involvement_level DESC, n.dominance DESC
            """, conflict_id)
            
            # Get conflict details
            conflict = await conn.fetchrow("""
                SELECT * FROM Conflicts WHERE conflict_id = $1
            """, conflict_id)
            
            # Process each stakeholder
            for stakeholder in stakeholders:
                # Determine if this stakeholder acts this turn
                if await self._should_stakeholder_act(stakeholder, conflict):
                    action = await self._determine_stakeholder_action(
                        stakeholder, conflict, stakeholders
                    )
                    
                    if action:
                        # Execute the action
                        result = await self._execute_stakeholder_action(
                            action, stakeholder, conflict_id
                        )
                        actions_taken.append(result)
                        
                        # Check for reactive actions
                        reactions = await self._check_for_reactions(
                            result, stakeholders, conflict_id
                        )
                        actions_taken.extend(reactions)
            
        return actions_taken
    
    async def _should_stakeholder_act(self, stakeholder: Dict[str, Any], 
                                    conflict: Dict[str, Any]) -> bool:
        """Determine if a stakeholder should act this turn"""
        # Base chance from involvement level
        base_chance = stakeholder['involvement_level'] / 10.0
        
        # Modify by personality
        if stakeholder['dominance'] > 70:
            base_chance *= 1.5
        if stakeholder['intensity'] > 80:
            base_chance *= 1.3
            
        # Modify by conflict phase
        if conflict['phase'] == 'climax':
            base_chance *= 2.0
        elif conflict['phase'] == 'resolution':
            base_chance *= 1.5
            
        # Modify by unrevealed secrets
        if stakeholder['unrevealed_secrets']:
            base_chance *= 1.2
            
        return random.random() < min(base_chance, 0.9)
    
    async def _determine_stakeholder_action(self, stakeholder: Dict[str, Any],
                                          conflict: Dict[str, Any],
                                          all_stakeholders: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Determine what action a stakeholder will take"""
        
        # Get stakeholder's relationships using new system
        relationships = await self._get_stakeholder_relationships(
            stakeholder['npc_id'], all_stakeholders
        )
        
        # Get available actions
        available_actions = await self._get_available_actions(
            stakeholder, conflict, relationships
        )
        
        # Use personality agent to decide
        decision_prompt = f"""
        You are {stakeholder['npc_name']}, a stakeholder in the conflict "{conflict['conflict_name']}".
        
        Your profile:
        - Public Motivation: {stakeholder['public_motivation']}
        - Private Motivation: {stakeholder['private_motivation']}
        - Desired Outcome: {stakeholder['desired_outcome']}
        - Personality: Dominance {stakeholder['dominance']}, Cruelty {stakeholder['cruelty']}, Intensity {stakeholder['intensity']}
        - Unrevealed Secrets: {len(stakeholder['unrevealed_secrets'] or [])}
        
        Current Situation:
        - Conflict Phase: {conflict['phase']}
        - Progress: {conflict['progress']}%
        - Your Involvement: {stakeholder['involvement_level']}/10
        
        Your Relationships:
        {json.dumps(relationships, indent=2)}
        
        Available Actions:
        {json.dumps(available_actions, indent=2)}
        
        Choose the best action to advance your goals. Consider both your public face and private ambitions.
        
        Output JSON:
        {{
            "action_type": "negotiate/manipulate/reveal_secret/form_alliance/betray/escalate/de-escalate",
            "target": "target_npc_id or null",
            "details": {{...}},
            "reasoning": "why this action now"
        }}
        """
        
        # Get assistants and use stakeholder personality agent
        assistants = await self._get_assistants()
        result = await ask_assistant(
            assistants["stakeholder_personality"],
            decision_prompt,
            self.context
        )
        
        try:
            # Handle both dict and string responses
            action = result if isinstance(result, dict) else json.loads(result)
            action['stakeholder_id'] = stakeholder['npc_id']
            action['stakeholder_name'] = stakeholder['npc_name']
            return action
        except:
            logger.error(f"Failed to parse stakeholder action: {result}")
            return None
    
    async def _get_stakeholder_relationships(self, npc_id: int,
                                           all_stakeholders: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get a stakeholder's relationships with other stakeholders using new system"""
        relationships = {}
        
        for other in all_stakeholders:
            if other['npc_id'] != npc_id:
                # Get relationship state using new system
                state = await self.relationship_manager.get_relationship_state(
                    'npc', npc_id, 'npc', other['npc_id']
                )
                
                # Get summary for decision making
                summary = state.to_summary()
                
                relationships[other['npc_id']] = {
                    'name': other['npc_name'],
                    'trust': summary['dimensions'].get('trust', 0),
                    'respect': summary['dimensions'].get('respect', 0),
                    'affection': summary['dimensions'].get('affection', 0),
                    'influence': summary['dimensions'].get('influence', 0),
                    'patterns': summary.get('patterns', []),
                    'archetypes': summary.get('archetypes', []),
                    'public_stance': other['public_motivation'],
                    'involvement': other['involvement_level']
                }
        
        return relationships
    
    async def _get_available_actions(self, stakeholder: Dict[str, Any],
                                   conflict: Dict[str, Any],
                                   relationships: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get available actions for a stakeholder"""
        actions = []
        
        # Negotiation options
        for npc_id, rel in relationships.items():
            if rel['trust'] > -50:  # Not completely distrustful
                actions.append({
                    'type': 'negotiate',
                    'target': npc_id,
                    'target_name': rel['name'],
                    'success_chance': self._calculate_negotiation_chance(stakeholder, rel)
                })
        
        # Alliance options
        for npc_id, rel in relationships.items():
            # Use trust and respect from new system
            if rel['trust'] > 25 and rel['respect'] > 40 and rel['involvement'] > 3:
                actions.append({
                    'type': 'form_alliance',
                    'target': npc_id,
                    'target_name': rel['name'],
                    'mutual_benefit': self._calculate_alliance_benefit(stakeholder, rel)
                })
        
        # Betrayal options (if already allied)
        if stakeholder.get('alliances'):
            for ally_id in json.loads(stakeholder['alliances']):
                if ally_id in relationships:
                    # Check for toxic patterns that might encourage betrayal
                    rel = relationships[ally_id]
                    if 'toxic_bond' in rel.get('archetypes', []) or 'push_pull' in rel.get('patterns', []):
                        actions.append({
                            'type': 'betray',
                            'target': ally_id,
                            'target_name': rel['name'],
                            'impact': 'high',
                            'pattern_driven': True
                        })
                    else:
                        actions.append({
                            'type': 'betray',
                            'target': ally_id,
                            'target_name': rel['name'],
                            'impact': 'high',
                            'pattern_driven': False
                        })
        
        # Secret revelation
        if stakeholder['unrevealed_secrets']:
            actions.append({
                'type': 'reveal_secret',
                'available_secrets': len(stakeholder['unrevealed_secrets']),
                'phase_appropriate': conflict['phase'] in ['active', 'climax']
            })
        
        # Escalation/De-escalation
        if stakeholder['involvement_level'] > 5:
            actions.append({
                'type': 'escalate',
                'current_phase': conflict['phase'],
                'impact': 'significant'
            })
            
        if conflict['progress'] > 50:
            actions.append({
                'type': 'de-escalate',
                'current_phase': conflict['phase'],
                'wisdom': stakeholder.get('wisdom', 50) > 60
            })
        
        return actions
    
    def _calculate_negotiation_chance(self, stakeholder: Dict[str, Any],
                                    relationship: Dict[str, Any]) -> float:
        """Calculate chance of successful negotiation using new relationship data"""
        base_chance = 0.5
        
        # Trust is now -100 to 100 in new system
        base_chance += relationship['trust'] / 200.0
        
        # Respect modifier
        base_chance += relationship['respect'] / 300.0
        
        # Dominance differential
        if stakeholder['dominance'] > 70:
            base_chance += 0.1
            
        # Involvement differential
        if stakeholder['involvement_level'] > relationship['involvement']:
            base_chance += 0.1
            
        # Pattern modifiers
        if 'slow_burn' in relationship.get('patterns', []):
            base_chance += 0.1  # Building trust helps
        if 'frenemies' in relationship.get('patterns', []):
            base_chance -= 0.1  # Complicated dynamic
            
        return min(max(base_chance, 0.1), 0.9)
    
    def _calculate_alliance_benefit(self, stakeholder: Dict[str, Any],
                                  relationship: Dict[str, Any]) -> str:
        """Calculate mutual benefit of alliance"""
        combined_involvement = stakeholder['involvement_level'] + relationship['involvement']
        
        # Check for synergistic archetypes
        if 'battle_partners' in relationship.get('archetypes', []):
            return "very high"
        
        if combined_involvement > 15:
            return "high"
        elif combined_involvement > 10:
            return "moderate"
        else:
            return "low"
    
    async def _execute_stakeholder_action(self, action: Dict[str, Any],
                                        stakeholder: Dict[str, Any],
                                        conflict_id: int) -> Dict[str, Any]:
        """Execute a stakeholder's chosen action"""
        
        action_type = action['action_type']
        result = {
            'stakeholder_id': stakeholder['npc_id'],
            'stakeholder_name': stakeholder['npc_name'],
            'action_type': action_type,
            'timestamp': datetime.utcnow().isoformat(),
            'success': False,
            'consequences': []
        }
        
        if action_type == 'negotiate':
            result.update(await self._execute_negotiation(
                action, stakeholder, conflict_id
            ))
        elif action_type == 'form_alliance':
            result.update(await self._execute_alliance_formation(
                action, stakeholder, conflict_id
            ))
        elif action_type == 'reveal_secret':
            result.update(await self._execute_secret_revelation(
                action, stakeholder, conflict_id
            ))
        elif action_type == 'betray':
            result.update(await self._execute_betrayal(
                action, stakeholder, conflict_id
            ))
        elif action_type == 'escalate':
            result.update(await self._execute_escalation(
                action, stakeholder, conflict_id
            ))
        elif action_type == 'de-escalate':
            result.update(await self._execute_deescalation(
                action, stakeholder, conflict_id
            ))
        
        # Log the action
        await self._log_stakeholder_action(result, conflict_id)
        
        return result
    
    async def _execute_negotiation(self, action: Dict[str, Any],
                                 stakeholder: Dict[str, Any],
                                 conflict_id: int) -> Dict[str, Any]:
        """Execute a negotiation between stakeholders"""
        
        target_id = action['target']
        
        async with get_db_connection_context() as conn:
            # Get target stakeholder
            target = await conn.fetchrow("""
                SELECT s.*, n.* FROM ConflictStakeholders s
                JOIN NPCStats n ON s.npc_id = n.npc_id
                WHERE s.conflict_id = $1 AND s.npc_id = $2
            """, conflict_id, target_id)
            
            if not target:
                return {'success': False, 'reason': 'Target not found'}
            
            # Use negotiation agent
            negotiation_prompt = f"""
            Facilitate negotiation between:
            
            {stakeholder['npc_name']}:
            - Wants: {stakeholder['desired_outcome']}
            - Offers: {action['details'].get('offer', 'Support')}
            - Dominance: {stakeholder['dominance']}
            
            {target['npc_name']}:
            - Wants: {target['desired_outcome']}
            - Current stance: {target['public_motivation']}
            - Dominance: {target['dominance']}
            
            Conflict: {conflict_id}
            
            Determine:
            1. Does {target['npc_name']} accept?
            2. What counter-demands might they make?
            3. What is the final agreement (if any)?
            
            Consider personality traits and power dynamics.
            
            Output JSON:
            {{
                "accepted": true/false,
                "reason": "explanation",
                "agreement": "terms if accepted",
                "counter_demands": ["list of demands"]
            }}
            """
            
            # Get assistants and use alliance negotiation agent
            assistants = await self._get_assistants()
            negotiation_result = await ask_assistant(
                assistants["alliance_negotiation"],
                negotiation_prompt,
                self.context
            )
            
            # Parse result
            result_data = negotiation_result if isinstance(negotiation_result, dict) else json.loads(negotiation_result)
            
            if result_data['accepted']:
                # Update stakeholder stances
                await conn.execute("""
                    UPDATE ConflictStakeholders
                    SET alliances = alliances || $1::jsonb
                    WHERE conflict_id = $2 AND npc_id = $3
                """, json.dumps([target_id]), conflict_id, stakeholder['npc_id'])
                
                await conn.execute("""
                    UPDATE ConflictStakeholders
                    SET alliances = alliances || $1::jsonb
                    WHERE conflict_id = $2 AND npc_id = $3
                """, json.dumps([stakeholder['npc_id']]), conflict_id, target_id)
                
                # Process positive relationship interaction
                ctx = RunContextWrapper({
                    "user_id": self.user_id,
                    "conversation_id": self.conversation_id
                })
                
                interaction_result = await process_relationship_interaction_tool(
                    ctx,
                    entity1_type='npc',
                    entity1_id=stakeholder['npc_id'],
                    entity2_type='npc',
                    entity2_id=target_id,
                    interaction_type='shared_success',
                    context='negotiation'
                )
                
                return {
                    'success': True,
                    'target_id': target_id,
                    'target_name': target['npc_name'],
                    'agreement': result_data.get('agreement', 'Mutual support'),
                    'relationship_impact': interaction_result.get('impacts', {}),
                    'consequences': [{
                        'type': 'alliance_formed',
                        'parties': [stakeholder['npc_id'], target_id]
                    }]
                }
            else:
                # Process negative interaction
                ctx = RunContextWrapper({
                    "user_id": self.user_id,
                    "conversation_id": self.conversation_id
                })
                
                interaction_result = await process_relationship_interaction_tool(
                    ctx,
                    entity1_type='npc',
                    entity1_id=stakeholder['npc_id'],
                    entity2_type='npc',
                    entity2_id=target_id,
                    interaction_type='conflict_resolved',  # Failed negotiation creates tension
                    context='failed_negotiation'
                )
                
                return {
                    'success': False,
                    'target_id': target_id,
                    'target_name': target['npc_name'],
                    'reason': result_data.get('reason', 'Terms unacceptable'),
                    'relationship_impact': interaction_result.get('impacts', {}),
                    'consequences': [{
                        'type': 'negotiation_failed',
                        'parties': [stakeholder['npc_id'], target_id]
                    }]
                }
    
    async def _execute_alliance_formation(self, action: Dict[str, Any],
                                        stakeholder: Dict[str, Any],
                                        conflict_id: int) -> Dict[str, Any]:
        """Execute alliance formation (similar to negotiation but focused on alliance)"""
        # For now, redirect to negotiation with alliance-specific details
        action['details'] = action.get('details', {})
        action['details']['offer'] = 'Alliance and mutual support'
        return await self._execute_negotiation(action, stakeholder, conflict_id)
    
    async def _execute_secret_revelation(self, action: Dict[str, Any],
                                       stakeholder: Dict[str, Any],
                                       conflict_id: int) -> Dict[str, Any]:
        """Execute revelation of a secret"""
        
        async with get_db_connection_context() as conn:
            # Get unrevealed secrets
            secrets = await conn.fetch("""
                SELECT * FROM StakeholderSecrets
                WHERE conflict_id = $1 AND npc_id = $2 AND is_revealed = FALSE
            """, conflict_id, stakeholder['npc_id'])
            
            if not secrets:
                return {'success': False, 'reason': 'No secrets to reveal'}
            
            # Get conflict context
            conflict = await conn.fetchrow("""
                SELECT * FROM Conflicts WHERE conflict_id = $1
            """, conflict_id)
            
            # Use secret revelation agent
            revelation_prompt = f"""
            {stakeholder['npc_name']} is considering revealing a secret.
            
            Character:
            - Motivation: {stakeholder['private_motivation']}
            - Personality: Dominance {stakeholder['dominance']}, Cruelty {stakeholder['cruelty']}
            
            Conflict Status:
            - Phase: {conflict['phase']}
            - Progress: {conflict['progress']}%
            
            Available Secrets:
            {json.dumps([{
                'id': s['secret_id'],
                'type': s['secret_type'],
                'content': s['content'],
                'about': s['target_npc_id']
            } for s in secrets], indent=2)}
            
            Choose:
            1. Which secret to reveal (if any)
            2. How to reveal it (public announcement, private threat, etc.)
            3. Who to reveal it to
            4. Expected impact
            
            Output JSON:
            {{
                "reveal": true/false,
                "secret_id": "id if revealing",
                "method": "public/private/threat",
                "revealed_to": ["list of recipients or 'all'"],
                "impact": "expected impact description"
            }}
            """
            
            # Get assistants and use secret revelation agent
            assistants = await self._get_assistants()
            revelation_result = await ask_assistant(
                assistants["secret_revelation"],
                revelation_prompt,
                self.context
            )
            
            # Parse result
            result_data = revelation_result if isinstance(revelation_result, dict) else json.loads(revelation_result)
            
            if result_data.get('reveal'):
                secret_id = result_data['secret_id']
                secret = next(s for s in secrets if s['secret_id'] == secret_id)
                
                # Update secret as revealed
                await conn.execute("""
                    UPDATE StakeholderSecrets
                    SET is_revealed = TRUE, 
                        revealed_to = $1,
                        is_public = $2,
                        revealed_at = NOW()
                    WHERE secret_id = $3
                """, 
                result_data.get('revealed_to'),
                result_data.get('method') == 'public',
                secret_id
                )
                
                # Update conflict tension
                await conn.execute("""
                    UPDATE Conflicts
                    SET progress = LEAST(progress + 10, 100)
                    WHERE conflict_id = $1
                """, conflict_id)
                
                # If secret is about someone, impact that relationship
                if secret.get('target_npc_id'):
                    ctx = RunContextWrapper({
                        "user_id": self.user_id,
                        "conversation_id": self.conversation_id
                    })
                    
                    interaction_result = await process_relationship_interaction_tool(
                        ctx,
                        entity1_type='npc',
                        entity1_id=stakeholder['npc_id'],
                        entity2_type='npc',
                        entity2_id=secret['target_npc_id'],
                        interaction_type='deception_discovered',
                        context='secret_revealed'
                    )
                
                return {
                    'success': True,
                    'secret_type': secret['secret_type'],
                    'method': result_data['method'],
                    'impact': result_data['impact'],
                    'consequences': [{
                        'type': 'secret_revealed',
                        'secret_type': secret['secret_type'],
                        'revealer': stakeholder['npc_id'],
                        'target': secret.get('target_npc_id')
                    }]
                }
            else:
                return {
                    'success': False,
                    'reason': 'Chose not to reveal any secrets yet'
                }
    
    async def _execute_betrayal(self, action: Dict[str, Any],
                              stakeholder: Dict[str, Any],
                              conflict_id: int) -> Dict[str, Any]:
        """Execute a betrayal using new relationship system"""
        
        target_id = action['target']
        
        async with get_db_connection_context() as conn:
            ctx = RunContextWrapper({
                "user_id": self.user_id,
                "conversation_id": self.conversation_id
            })
            
            # Remove alliance
            await conn.execute("""
                UPDATE ConflictStakeholders
                SET alliances = alliances - $1::text,
                    rivalries = rivalries || $2::jsonb
                WHERE conflict_id = $3 AND npc_id = $4
            """, 
            str(target_id), json.dumps([target_id]), 
            conflict_id, stakeholder['npc_id']
            )
            
            await conn.execute("""
                UPDATE ConflictStakeholders
                SET alliances = alliances - $1::text,
                    rivalries = rivalries || $2::jsonb
                WHERE conflict_id = $3 AND npc_id = $4
            """, 
            str(stakeholder['npc_id']), json.dumps([stakeholder['npc_id']]),
            conflict_id, target_id
            )
            
            # Create grudge in conflict history
            await conn.execute("""
                INSERT INTO ConflictHistory
                (user_id, conversation_id, conflict_id, affected_npc_id,
                 impact_type, grudge_level, narrative_impact)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            self.user_id, self.conversation_id, conflict_id, target_id,
            'betrayal', 80, f"Betrayed by {stakeholder['npc_name']}"
            )
            
            # Process betrayal interaction with new system
            interaction_result = await process_relationship_interaction_tool(
                ctx,
                entity1_type='npc',
                entity1_id=stakeholder['npc_id'],
                entity2_type='npc',
                entity2_id=target_id,
                interaction_type='betrayal',
                context='conflict_betrayal'
            )
            
            # Increase conflict tension
            await conn.execute("""
                UPDATE Conflicts
                SET progress = LEAST(progress + 15, 100)
                WHERE conflict_id = $1
            """, conflict_id)
            
            # Log canonical event
            await canon.log_canonical_event(
                ctx, conn,
                f"{stakeholder['npc_name']} betrayed {action.get('target_name', f'NPC {target_id}')} in conflict",
                tags=["conflict", "betrayal", "relationship"],
                significance=8
            )
            
            return {
                'success': True,
                'target_id': target_id,
                'betrayal_type': action['details'].get('method', 'sudden'),
                'relationship_impact': interaction_result.get('impacts', {}),
                'new_patterns': interaction_result.get('new_patterns', []),
                'consequences': [{
                    'type': 'betrayal',
                    'betrayer': stakeholder['npc_id'],
                    'betrayed': target_id,
                    'new_grudge_level': 80
                }]
            }
    
    async def _execute_escalation(self, action: Dict[str, Any],
                                stakeholder: Dict[str, Any],
                                conflict_id: int) -> Dict[str, Any]:
        """Execute conflict escalation"""
        
        async with get_db_connection_context() as conn:
            # Get current conflict state
            conflict = await conn.fetchrow("""
                SELECT * FROM Conflicts WHERE conflict_id = $1
            """, conflict_id)
            
            new_phase = conflict['phase']
            if conflict['phase'] == 'brewing' and conflict['progress'] >= 25:
                new_phase = 'active'
            elif conflict['phase'] == 'active' and conflict['progress'] >= 50:
                new_phase = 'climax'
            
            # Update conflict
            await conn.execute("""
                UPDATE Conflicts
                SET progress = LEAST(progress + 20, 100),
                    phase = $1
                WHERE conflict_id = $2
            """, new_phase, conflict_id)
            
            # Increase stakeholder involvement
            await conn.execute("""
                UPDATE ConflictStakeholders
                SET involvement_level = LEAST(involvement_level + 2, 10)
                WHERE conflict_id = $1 AND npc_id = $2
            """, conflict_id, stakeholder['npc_id'])
            
            return {
                'success': True,
                'escalation_method': action['details'].get('method', 'aggressive action'),
                'new_phase': new_phase,
                'consequences': [{
                    'type': 'conflict_escalated',
                    'escalator': stakeholder['npc_id'],
                    'new_phase': new_phase
                }]
            }
    
    async def _execute_deescalation(self, action: Dict[str, Any],
                                  stakeholder: Dict[str, Any],
                                  conflict_id: int) -> Dict[str, Any]:
        """Execute conflict de-escalation"""
        
        async with get_db_connection_context() as conn:
            # Reduce conflict progress slightly
            await conn.execute("""
                UPDATE Conflicts
                SET progress = GREATEST(progress - 10, 0)
                WHERE conflict_id = $1
            """, conflict_id)
            
            # Reduce tensions between some stakeholders
            await conn.execute("""
                UPDATE ConflictStakeholders
                SET involvement_level = GREATEST(involvement_level - 1, 0)
                WHERE conflict_id = $1
            """, conflict_id)
            
            # Process supportive interactions with key stakeholders
            ctx = RunContextWrapper({
                "user_id": self.user_id,
                "conversation_id": self.conversation_id
            })
            
            # Get other high-involvement stakeholders
            other_stakeholders = await conn.fetch("""
                SELECT npc_id FROM ConflictStakeholders
                WHERE conflict_id = $1 AND npc_id != $2 
                AND involvement_level > 5
                LIMIT 2
            """, conflict_id, stakeholder['npc_id'])
            
            for other in other_stakeholders:
                await process_relationship_interaction_tool(
                    ctx,
                    entity1_type='npc',
                    entity1_id=stakeholder['npc_id'],
                    entity2_type='npc',
                    entity2_id=other['npc_id'],
                    interaction_type='support_provided',
                    context='peacemaking'
                )
            
            return {
                'success': True,
                'method': action['details'].get('method', 'peace offering'),
                'consequences': [{
                    'type': 'tensions_reduced',
                    'peacemaker': stakeholder['npc_id']
                }]
            }
    
    async def _check_for_reactions(self, action_result: Dict[str, Any],
                                 all_stakeholders: List[Dict[str, Any]],
                                 conflict_id: int) -> List[Dict[str, Any]]:
        """Check if other stakeholders react to an action"""
        reactions = []
        
        # Betrayals always cause reactions
        if action_result['action_type'] == 'betray' and action_result['success']:
            for consequence in action_result.get('consequences', []):
                if consequence['type'] == 'betrayal':
                    betrayed_id = consequence['betrayed']
                    betrayed = next(s for s in all_stakeholders if s['npc_id'] == betrayed_id)
                    
                    # Get current relationship state to inform reaction
                    state = await self.relationship_manager.get_relationship_state(
                        'npc', betrayed_id, 'npc', action_result['stakeholder_id']
                    )
                    
                    # High volatility or toxic patterns lead to explosive reactions
                    if state.dimensions.volatility > 70 or 'toxic_bond' in state.active_archetypes:
                        reaction_type = 'explosive_retaliation'
                    else:
                        reaction_type = 'calculated_revenge'
                    
                    reaction = {
                        'stakeholder_id': betrayed_id,
                        'stakeholder_name': betrayed['npc_name'],
                        'action_type': 'retaliation',
                        'reaction_type': reaction_type,
                        'trigger': 'betrayal',
                        'target': action_result['stakeholder_id'],
                        'success': True,
                        'consequences': [{
                            'type': 'retaliation',
                            'method': 'expose_weakness'
                        }]
                    }
                    reactions.append(reaction)
        
        # Secret revelations may cause reactions
        if action_result['action_type'] == 'reveal_secret' and action_result['success']:
            for consequence in action_result.get('consequences', []):
                if consequence['type'] == 'secret_revealed' and consequence.get('target'):
                    target_id = consequence['target']
                    target = next((s for s in all_stakeholders if s['npc_id'] == target_id), None)
                    
                    if target and target['dominance'] > 60:
                        # Dominant NPCs likely to react to being exposed
                        reaction = {
                            'stakeholder_id': target_id,
                            'stakeholder_name': target['npc_name'],
                            'action_type': 'counter_reveal',
                            'trigger': 'secret_exposed',
                            'success': random.random() < 0.7,
                            'consequences': []
                        }
                        reactions.append(reaction)
        
        return reactions
    
    async def _log_stakeholder_action(self, action_result: Dict[str, Any],
                                    conflict_id: int):
        """Log stakeholder action to conflict memory"""
        try:
            async with get_db_connection_context() as conn:
                # Create memory event
                action_desc = f"{action_result['stakeholder_name']} performed {action_result['action_type']}"
                if action_result.get('target_name'):
                    action_desc += f" targeting {action_result['target_name']}"
                if not action_result['success']:
                    action_desc += " (failed)"
                
                await conn.execute("""
                    INSERT INTO ConflictMemoryEvents
                    (conflict_id, memory_text, significance, entity_type, entity_id)
                    VALUES ($1, $2, $3, $4, $5)
                """,
                conflict_id, action_desc, 
                7 if action_result['success'] else 5,
                'npc', action_result['stakeholder_id']
                )
                
                # Log canonical event for significant actions
                if action_result['success'] and action_result['action_type'] in ['betray', 'reveal_secret']:
                    ctx = RunContextWrapper({
                        "user_id": self.user_id,
                        "conversation_id": self.conversation_id
                    })
                    
                    await canon.log_canonical_event(
                        ctx, conn, action_desc,
                        tags=["conflict", "stakeholder_action", action_result['action_type']],
                        significance=8
                    )
                    
        except Exception as e:
            logger.error(f"Error logging stakeholder action: {e}")

# Tool functions
@function_tool
async def process_conflict_stakeholder_turns(ctx: RunContextWrapper, 
                                           conflict_id: int) -> List[Dict[str, Any]]:
    """
    Process autonomous turns for all stakeholders in a conflict.
    
    Returns:
        List of actions taken by stakeholders
    """
    context = ctx.context
    system = StakeholderAutonomySystem(context.user_id, context.conversation_id)
    
    try:
        actions = await system.process_stakeholder_turn(conflict_id)
        return actions
    except Exception as e:
        logger.error(f"Error processing stakeholder turns: {e}", exc_info=True)
        return []

@function_tool(strict_mode=False)
async def force_stakeholder_action(ctx: RunContextWrapper,
                                 conflict_id: int,
                                 npc_id: int,
                                 action_type: str,
                                 action_details: Dict[str, Any]) -> Dict[str, Any]:
    """
    Force a specific stakeholder to take an action.
    
    Args:
        conflict_id: The conflict ID
        npc_id: The stakeholder NPC ID
        action_type: Type of action to force
        action_details: Details of the action
        
    Returns:
        Result of the action
    """
    context = ctx.context
    system = StakeholderAutonomySystem(context.user_id, context.conversation_id)
    
    try:
        async with get_db_connection_context() as conn:
            # Get stakeholder
            stakeholder = await conn.fetchrow("""
                SELECT s.*, n.* FROM ConflictStakeholders s
                JOIN NPCStats n ON s.npc_id = n.npc_id
                WHERE s.conflict_id = $1 AND s.npc_id = $2
            """, conflict_id, npc_id)
            
            if not stakeholder:
                return {"error": "Stakeholder not found"}
            
            action = {
                "action_type": action_type,
                "stakeholder_id": npc_id,
                "stakeholder_name": stakeholder['npc_name'],
                "details": action_details,
                "target": action_details.get("target")
            }
            
            result = await system._execute_stakeholder_action(
                action, dict(stakeholder), conflict_id
            )
            
            return result
            
    except Exception as e:
        logger.error(f"Error forcing stakeholder action: {e}", exc_info=True)
        return {"error": str(e)}
