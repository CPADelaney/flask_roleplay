# nyx/governance/player.py
"""
Player action validation and disagreement handling.
"""
import logging
import json
import time
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
from .constants import DirectiveType
from db.connection import get_db_connection_context
from logic.chatgpt_integration import generate_text_completion

logger = logging.getLogger(__name__)


class PlayerGovernanceMixin:
    """Handles player action validation and disagreement."""
    
    async def handle_player_disagreement(
        self,
        user_id: int,
        conversation_id: int,
        action_type: str,
        action_details: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Handle direct disagreement with a player's action.
        
        Args:
            user_id: User ID
            conversation_id: Conversation ID
            action_type: Type of action being performed
            action_details: Details of the action
            context: Additional context information
            
        Returns:
            Dictionary containing disagreement response and reasoning
        """
        # Get current state and context
        current_state = await self.get_current_state(user_id, conversation_id)
        context = context or {}
        
        # Analyze the action's impact
        impact_analysis = await self._analyze_action_impact(
            action_type,
            action_details,
            current_state,
            context
        )
        
        # Check if disagreement is warranted
        if not self._should_disagree(impact_analysis):
            return {
                "disagrees": False,
                "reasoning": "Action is acceptable within current context",
                "impact_analysis": impact_analysis
            }
        
        # Generate disagreement response
        disagreement = await self._generate_disagreement_response(
            impact_analysis,
            current_state,
            context
        )
        
        # Track disagreement
        await self._track_disagreement(
            user_id,
            conversation_id,
            action_type,
            disagreement
        )
        
        return {
            "disagrees": True,
            "reasoning": disagreement["reasoning"],
            "suggested_alternative": disagreement.get("alternative"),
            "impact_analysis": impact_analysis,
            "narrative_context": disagreement.get("narrative_context")
        }

    async def check_action_permission(
        self,
        agent_type: str,
        agent_id: Union[int, str],
        action_type: str,
        action_details: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Check if an action is permitted by governance.
        
        Args:
            agent_type: Type of agent performing the action
            agent_id: ID of agent performing the action
            action_type: Type of action being performed
            action_details: Details of the action
            context: Additional context (optional)
            
        Returns:
            Dictionary with permission result
        """
        # Initialize the result
        result = {
            "approved": True,
            "reasoning": "Action is permitted by default"
        }
        
        # Check for active prohibitions on this action
        prohibitions = self._get_active_prohibitions(agent_type, action_type)
        if prohibitions:
            # Action is prohibited
            prohibition = prohibitions[0]  # Get the highest priority prohibition
            result = {
                "approved": False,
                "reasoning": prohibition.get("reason", "Action is prohibited"),
                "prohibition_id": prohibition.get("id")
            }
            return result
        
        # Return the result
        return result

    async def _analyze_action_impact(
        self,
        action_type: str,
        action_details: Dict[str, Any],
        current_state: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze the impact of a player's action."""
        impact_scores = {
            "narrative_impact": 0.0,
            "character_consistency": 0.0,
            "world_integrity": 0.0,
            "player_experience": 0.0
        }
        
        # Analyze narrative impact
        narrative_context = current_state.get("narrative_context", {})
        impact_scores["narrative_impact"] = await self._calculate_narrative_impact(
            action_type,
            action_details,
            narrative_context
        )
        
        # Analyze character consistency
        character_state = current_state.get("character_state", {})
        impact_scores["character_consistency"] = await self._calculate_character_consistency(
            action_type,
            action_details,
            character_state
        )
        
        # Analyze world integrity
        world_state = current_state.get("world_state", {})
        impact_scores["world_integrity"] = await self._calculate_world_integrity(
            action_type,
            action_details,
            world_state
        )
        
        # Analyze player experience impact
        player_context = context.get("player_context", {})
        impact_scores["player_experience"] = await self._calculate_player_experience_impact(
            action_type,
            action_details,
            player_context
        )
        
        return impact_scores

    def _should_disagree(self, impact_analysis: Dict[str, float]) -> bool:
        """Determine if Nyx should disagree with the action."""
        for metric, threshold in self.disagreement_thresholds.items():
            if impact_analysis.get(metric, 0) > threshold:
                return True
        return False

    async def _generate_disagreement_response(
        self,
        impact_analysis: Dict[str, Any],
        current_state: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a detailed disagreement response."""
        # Identify primary concerns
        concerns = []
        for metric, score in impact_analysis.items():
            if score > self.disagreement_thresholds.get(metric, 1.0):
                concerns.append(metric)
        
        # Generate reasoning based on concerns
        reasoning = await self._generate_reasoning(concerns, current_state, context)
        
        # Generate alternative suggestion if possible
        alternative = await self._generate_alternative_suggestion(
            concerns,
            current_state,
            context
        )
        
        return {
            "reasoning": reasoning,
            "alternative": alternative,
            "narrative_context": current_state.get("narrative_context", {})
        }

    async def _track_disagreement(
        self,
        user_id: int,
        conversation_id: int,
        action_type: str,
        disagreement: Dict[str, Any]
    ):
        """Track disagreement history for pattern analysis."""
        key = f"{user_id}:{conversation_id}"
        if key not in self.disagreement_history:
            self.disagreement_history[key] = []
        
        self.disagreement_history[key].append({
            "timestamp": time.time(),
            "action_type": action_type,
            "disagreement": disagreement
        })
        
        # Keep only recent history
        self.disagreement_history[key] = self.disagreement_history[key][-100:]

    def _get_active_prohibitions(self, agent_type: str, action_type: str) -> List[Dict[str, Any]]:
        """
        Get active prohibitions for an agent and action type.
        
        Args:
            agent_type: Type of agent
            action_type: Type of action
            
        Returns:
            List of active prohibitions, sorted by priority
        """
        if not hasattr(self, "directives"):
            return []
        
        # Get active prohibitions
        now = datetime.now()
        prohibitions = []
        
        for directive_id, directive in getattr(self, "directives", {}).items():
            # Check if directive is a prohibition
            if directive["type"] != DirectiveType.PROHIBITION:
                continue
            
            # Check if prohibition applies to this agent and action
            prohibited_agent = directive["data"].get("agent_type")
            prohibited_action = directive["data"].get("action_type")
            
            if (prohibited_agent == agent_type or prohibited_agent == "*") and \
               (prohibited_action == action_type or prohibited_action == "*"):
                # Check if prohibition is still active
                expires_at = datetime.fromisoformat(directive["expires_at"])
                if expires_at > now:
                    prohibitions.append(directive)
        
        # Sort by priority
        prohibitions.sort(key=lambda p: p.get("priority", 0), reverse=True)
        
        return prohibitions

    async def _calculate_character_consistency(
        self,
        action_type: str,
        action_details: Dict[str, Any],
        character_state: Dict[str, Any]
    ) -> float:
        """Calculate how consistent an action is with character development."""
        impact_score = 0.0
        
        # Check character motivation alignment
        if not await self._aligns_with_motivation(action_type, action_details, character_state):
            impact_score += 0.4
        
        # Check character development trajectory
        if self._disrupts_development(action_type, action_details, character_state):
            impact_score += 0.3
        
        # Check relationship consistency
        if not self._maintains_relationships(action_type, action_details, character_state):
            impact_score += 0.3
        
        return min(1.0, impact_score)

    async def _calculate_player_experience_impact(
        self,
        action_type: str,
        action_details: Dict[str, Any],
        player_context: Dict[str, Any]
    ) -> float:
        """Calculate the impact of an action on player experience."""
        impact_score = 0.0
        
        # Check for engagement disruption
        if self._would_disrupt_engagement(action_type, action_details, player_context):
            impact_score += 0.3
        
        # Check for immersion breaking
        if self._would_break_immersion(action_type, action_details, player_context):
            impact_score += 0.3
        
        # Check for agency preservation
        if not self._preserves_player_agency(action_type, action_details, player_context):
            impact_score += 0.2
        
        return min(1.0, impact_score)

    async def _generate_reasoning(
        self,
        concerns: List[str],
        current_state: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Generate detailed reasoning for disagreement using LLM."""
        # Build a detailed prompt for the LLM
        concern_descriptions = {
            "narrative_impact": "The action would disrupt narrative flow and pacing",
            "character_consistency": "The action doesn't align with character motivations",
            "world_integrity": "The action violates world rules and consistency",
            "player_experience": "The action would negatively impact player experience"
        }
        
        # Create context for the LLM
        prompt = f"""
        As Nyx, the governance system, explain why you disagree with a player's action.
        
        Primary concerns:
        {', '.join([concern_descriptions.get(c, c) for c in concerns])}
        
        Current narrative context:
        - Arc: {current_state.get('narrative_context', {}).get('current_arc', 'Unknown')}
        - Active quests: {', '.join([q['name'] for q in current_state.get('narrative_context', {}).get('plot_points', [])])}
        - Player location: {current_state.get('game_state', {}).get('current_location', 'Unknown')}
        
        Character state:
        - Stats: {current_state.get('character_state', {}).get('corruption', 'Unknown')} corruption
        - Key relationships: {len(current_state.get('character_state', {}).get('relationships', {}))} active
        
        Provide a concise but authoritative explanation (2-3 sentences) that:
        1. Clearly states the main issue
        2. References specific game context
        3. Maintains Nyx's dominant personality
        """
        
        try:
            reasoning = await generate_text_completion(
                system_prompt="You are Nyx, an authoritative governance system maintaining narrative coherence.",
                user_prompt=prompt,
                temperature=0.7,
                max_tokens=200,
                task_type="decision"
            )
            return reasoning.strip()
        except Exception as e:
            logger.error(f"Error generating reasoning with LLM: {e}")
            # Fallback to improved static reasoning
            reasoning_parts = []
            for concern in concerns[:2]:  # Focus on top 2 concerns
                reasoning_parts.append(concern_descriptions.get(concern, f"Issue with {concern}"))
            
            # Add context
            arc = current_state.get('narrative_context', {}).get('current_arc')
            if arc:
                reasoning_parts.append(f"This conflicts with the current {arc} narrative arc.")
            
            return " ".join(reasoning_parts)

    async def _generate_alternative_suggestion(
        self,
        concerns: List[str],
        current_state: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate an alternative suggestion using LLM for better context awareness."""
        primary_concern = concerns[0] if concerns else None
        
        # Build context for alternative generation
        prompt = f"""
        As Nyx, suggest an alternative action for the player that addresses these concerns:
        {', '.join(concerns)}
        
        Current game state:
        - Location: {current_state.get('game_state', {}).get('current_location', 'Unknown')}
        - Active quests: {[q['quest_name'] for q in current_state.get('game_state', {}).get('active_quests', [])]}
        - Nearby NPCs: {[n['npc_name'] for n in current_state.get('game_state', {}).get('current_npcs', [])][:3]}
        
        Generate 3 specific, actionable alternatives that:
        1. Respect the current narrative
        2. Offer meaningful player agency
        3. Are contextually appropriate
        
        Format as a JSON object with 'specific_options' array.
        """
        
        try:
            response = await generate_text_completion(
                system_prompt="You are Nyx, suggesting alternatives that enhance the game experience.",
                user_prompt=prompt,
                temperature=0.8,
                max_tokens=300,
                task_type="decision"
            )
            
            # Try to parse as JSON
            try:
                suggestions = json.loads(response)
                if "specific_options" in suggestions:
                    return {
                        "type": f"{primary_concern}_alternative",
                        "suggestion": f"Consider actions that respect {primary_concern.replace('_', ' ')}",
                        "specific_options": suggestions["specific_options"][:3],
                        "reasoning": "These alternatives maintain game coherence while preserving player agency"
                    }
            except json.JSONDecodeError:
                pass
        except Exception as e:
            logger.error(f"Error generating alternatives with LLM: {e}")
        
        # Fallback to improved context-aware suggestions
        if primary_concern == "narrative_impact":
            return await self._suggest_narrative_alternative(current_state, context)
        elif primary_concern == "character_consistency":
            return await self._suggest_character_alternative(
                current_state.get("character_state", {}), context)
        elif primary_concern == "world_integrity":
            return await self._suggest_world_alternative(
                current_state.get("world_state", {}), context)
        elif primary_concern == "player_experience":
            return await self._suggest_experience_alternative(current_state, context)
        
        return None

    async def _suggest_character_alternative(self, character_state: Dict[str, Any], 
                                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an alternative that respects character development."""
        relationships = character_state.get("relationships", {})
        stats = {k: v for k, v in character_state.items() 
                 if k in ["corruption", "confidence", "willpower", "obedience"]}
        
        # Find opportunities based on current character state
        suggestions = []
        
        # Suggest based on stats
        if stats.get("confidence", 0) < 30:
            suggestions.append("Build confidence through small victories")
        if stats.get("willpower", 0) < 40:
            suggestions.append("Exercise restraint to strengthen willpower")
            
        # Suggest based on relationships
        for npc, rel in relationships.items():
            if rel["level"] < 30:
                suggestions.append(f"Improve your relationship with {npc}")
        
        return {
            "type": "character_alternative",
            "suggestion": "Focus on character development that aligns with your journey",
            "specific_options": suggestions[:3] if suggestions else [
                "Reflect on recent events in your journal",
                "Seek guidance from a trusted NPC",
                "Train to improve your abilities"
            ],
            "reasoning": "Character consistency creates more meaningful progression"
        }

    async def _suggest_experience_alternative(self, current_state: Dict[str, Any], 
                                            context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an alternative that enhances player experience."""
        game_state = current_state.get("game_state", {})
        active_quests = game_state.get("active_quests", [])
        current_npcs = game_state.get("current_npcs", [])
        
        options = []
        
        if active_quests:
            options.append(f"Progress the '{active_quests[0]['quest_name']}' quest")
        if current_npcs:
            options.append(f"Interact with {current_npcs[0]['npc_name']} who is nearby")
        
        options.extend([
            "Explore a new area of the game world",
            "Engage with the unique mechanics of this setting",
            "Pursue personal goals that interest you"
        ])
        
        return {
            "type": "experience_alternative",
            "suggestion": "Choose actions that enhance your enjoyment",
            "specific_options": options[:3],
            "reasoning": "Player agency and engagement are paramount"
        }

    def _would_disrupt_engagement(
        self,
        action_type: str,
        action_details: Dict[str, Any],
        player_context: Dict[str, Any]
    ) -> bool:
        """Check if an action would disrupt player engagement."""
        # Get player engagement metrics
        engagement = player_context.get("engagement", {})
        flow_state = engagement.get("flow_state", {})
        
        # Check if action would break flow
        if action_type == "break_flow":
            return True
            
        # Check if action would reduce engagement
        if action_type == "reduce_engagement":
            return True
            
        return False

    def _would_break_immersion(
        self,
        action_type: str,
        action_details: Dict[str, Any],
        player_context: Dict[str, Any]
    ) -> bool:
        """Check if an action would break player immersion."""
        # Get immersion metrics
        immersion = player_context.get("immersion", {})
        suspension = immersion.get("suspension_of_disbelief", 1.0)
        
        # Check if action would break immersion
        if action_type == "break_immersion":
            return True
            
        # Check if action would reduce suspension of disbelief
        if action_type == "reduce_suspension":
            return True
            
        return False

    def _preserves_player_agency(
        self,
        action_type: str,
        action_details: Dict[str, Any],
        player_context: Dict[str, Any]
    ) -> bool:
        """Check if an action preserves player agency."""
        # Get agency metrics
        agency = player_context.get("agency", {})
        choices = agency.get("meaningful_choices", [])
        
        # Check if action would remove agency
        if action_type == "remove_agency":
            return False
            
        # Check if action preserves choices
        if action_type == "preserve_choices":
            return True
            
        return True
