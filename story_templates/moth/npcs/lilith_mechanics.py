# story_templates/moth/npcs/lilith_mechanics.py
"""
Handles Lilith's special mechanics during gameplay
Manages masks, poetry moments, trust dynamics, and the three words
Adapted for the Queen of Thorns / Shadow Network setting
"""

import json
import logging
import random
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from db.connection import get_db_connection_context
from memory.wrapper import MemorySystem
from lore.core import canon

logger = logging.getLogger(__name__)

class LilithMechanicsHandler:
    """Handles all of Lilith's special mechanics during gameplay"""
    
    def __init__(self, user_id: int, conversation_id: int, npc_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.npc_id = npc_id
        self._current_state = None
        self._last_update = None
    
    async def initialize(self):
        """Load current state from database"""
        await self._load_current_state()
    
    async def _load_current_state(self):
        """Load Lilith's current state from database"""
        async with get_db_connection_context() as conn:
            # Get base stats
            stats_row = await conn.fetchrow(
                """
                SELECT trust, dominance, cruelty, affection, intensity,
                       current_mask, dialogue_patterns, trauma_triggers,
                       secrets, special_mechanics
                FROM NPCStats
                WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
                """,
                self.user_id, self.conversation_id, self.npc_id
            )
            
            if not stats_row:
                raise ValueError(f"NPC {self.npc_id} not found")
            
            # Get special mechanics data
            mechanics_rows = await conn.fetch(
                """
                SELECT mechanic_type, mechanic_data
                FROM npc_special_mechanics
                WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
                """,
                self.user_id, self.conversation_id, self.npc_id
            )
            
            mechanics = {}
            for row in mechanics_rows:
                mechanics[row['mechanic_type']] = json.loads(row['mechanic_data'])
            
            self._current_state = {
                "stats": {
                    "trust": stats_row['trust'],
                    "dominance": stats_row['dominance'],
                    "cruelty": stats_row['cruelty'],
                    "affection": stats_row['affection'],
                    "intensity": stats_row['intensity']
                },
                "current_mask": stats_row['current_mask'],
                "dialogue_patterns": json.loads(stats_row['dialogue_patterns']) if stats_row['dialogue_patterns'] else {},
                "trauma_triggers": json.loads(stats_row['trauma_triggers']) if stats_row['trauma_triggers'] else [],
                "secrets": json.loads(stats_row['secrets']) if stats_row['secrets'] else {},
                "mechanics": mechanics
            }
            
            self._last_update = datetime.now()
    
    async def check_mask_state(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check current mask state and whether it should change
        
        Args:
            context: Current scene context
            
        Returns:
            Dict with mask status and any changes
        """
        if not self._current_state:
            await self._load_current_state()
        
        current_mask = self._current_state['current_mask']
        mask_data = self._current_state['mechanics'].get('mask_system', {})
        available_masks = mask_data.get('available_masks', [])
        
        # Find current mask info
        current_mask_info = next(
            (m for m in available_masks if m['name'] == current_mask),
            None
        )
        
        if not current_mask_info:
            return {"error": "Current mask not found in system"}
        
        # Check if mask should slip or change
        trust = self._current_state['stats']['trust']
        location = context.get('location', '').lower()
        emotional_intensity = context.get('emotional_intensity', 0)
        is_private = context.get('is_private', False)
        player_action = context.get('player_action', '').lower()
        
        # Mask slippage conditions
        slippage_chance = 0
        slippage_reasons = []
        
        # Trust-based slippage
        if trust > 50 and current_mask == "Porcelain Goddess":
            slippage_chance += 0.2
            slippage_reasons.append("growing trust")
        
        # Location-based (updated for SF setting)
        if any(loc in location for loc in ["private", "chamber", "rose garden", "inner sanctum"]):
            slippage_chance += 0.3
            slippage_reasons.append("private setting")
        
        # Emotional intensity
        if emotional_intensity > 0.7:
            slippage_chance += 0.4
            slippage_reasons.append("high emotion")
        
        # Trigger words in player action
        trigger_words = ["love", "stay", "promise", "forever", "disappear", "leave", "network", "queen"]
        if any(word in player_action for word in trigger_words):
            slippage_chance += 0.5
            slippage_reasons.append("triggered response")
        
        # Check for slippage
        mask_slipped = random.random() < slippage_chance
        
        result = {
            "current_mask": current_mask,
            "mask_integrity": mask_data.get('mask_integrity', 100),
            "slipped": mask_slipped,
            "slippage_reasons": slippage_reasons
        }
        
        if mask_slipped:
            # Determine what shows through
            if trust > 70:
                result["revealed"] = "vulnerability"
                result["description"] = "For a moment, the mask slips. You see exhaustion and the weight of countless saved and lost flicker across her features before she catches herself."
            elif emotional_intensity > 0.8:
                result["revealed"] = "raw_emotion"
                result["description"] = "Her carefully constructed facade cracks. Raw emotion bleeds through - pain from those she couldn't save, longing for someone who won't vanish."
            else:
                result["revealed"] = "humanity"
                result["description"] = "The Queen persona wavers. Beneath, you glimpse a woman carrying the weight of an entire shadow network on her shoulders."
            
            # Update mask integrity
            await self._update_mask_integrity(mask_data['mask_integrity'] - 10)
            
            # Check if mask should change entirely
            if is_private and trust > current_mask_info['trust_required'] + 20:
                next_mask = await self._get_next_appropriate_mask(trust, context)
                if next_mask and next_mask != current_mask:
                    result["mask_change"] = True
                    result["new_mask"] = next_mask
                    await self._change_mask(next_mask)
        
        return result
    
    async def check_network_revelation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if player should discover her role in the network
        
        Args:
            context: Current scene context
            
        Returns:
            Dict with network revelation details
        """
        trust = self._current_state['stats']['trust']
        location = context.get('location', '').lower()
        
        # Already revealed?
        story_flags = await self._get_story_flags()
        if story_flags.get('network_identity_revealed', False):
            return {"reveal_possible": False, "already_known": True}
        
        # Check conditions
        reveal_chance = 0
        reveal_reasons = []
        
        if trust > 60:
            reveal_chance += 0.3
            reveal_reasons.append("trust established")
        if any(loc in location for loc in ["rose garden", "thornfield", "safehouse"]):
            reveal_chance += 0.5
            reveal_reasons.append("network location")
        if context.get('witnessed_transformation', False):
            reveal_chance += 0.7
            reveal_reasons.append("saw behavioral modification")
        if context.get('helped_vulnerable_person', False):
            reveal_chance += 0.4
            reveal_reasons.append("demonstrated protection instinct")
        
        if random.random() < reveal_chance:
            return {
                "reveal_possible": True,
                "reveal_type": "discovered" if "safehouse" in location else "confessed",
                "reveal_reasons": reveal_reasons,
                "description": (
                    "The woman before you isn't just a dominant playing with power. She's the Queen of Thorns - "
                    "or one of them. Leader of a network without a name, transforming predators into protectors, "
                    "saving those who need saving. What outsiders call 'The Rose & Thorn Society' is so much more."
                ),
                "network_access_offered": trust > 70
            }
        
        return {"reveal_possible": False, "chance": reveal_chance}
    
    async def check_transformation_witness(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if player witnesses a predator transformation session
        
        Args:
            context: Current scene context
            
        Returns:
            Dict with transformation scene details
        """
        if self._current_state['stats']['trust'] < 40:
            return {"witness_possible": False, "trust_too_low": True}
        
        location = context.get('location', '').lower()
        suitable_locations = ["maison noir", "private chambers", "executive office", "therapy room"]
        
        if not any(loc in location for loc in suitable_locations):
            return {"witness_possible": False, "wrong_location": True}
        
        if context.get('executive_present', False) or context.get('predator_identified', False):
            return {
                "witness_possible": True,
                "transformation_type": "behavioral_modification",
                "subject": context.get('subject_name', 'A tech executive'),
                "method": "power_exchange_therapy",
                "description": (
                    "You watch as she systematically breaks down his predatory patterns, replacing them with "
                    "submission and service. It's brutal, beautiful, and utterly transformative. This is how "
                    "the garden grows - turning those who would harm into those who heal."
                ),
                "player_reaction_options": [
                    "fascination",
                    "horror",
                    "understanding",
                    "arousal"
                ]
            }
        
        return {"witness_possible": False}
    
    async def check_poetry_moment(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if Lilith should speak in poetry based on context
        
        Args:
            context: Current scene context
            
        Returns:
            Dict with poetry moment details
        """
        poetry_data = self._current_state['mechanics'].get('poetry_triggers', {})
        trigger_conditions = poetry_data.get('trigger_conditions', [])
        
        # Check each trigger condition
        current_emotion = context.get('lilith_emotion', 'neutral')
        trigger = next(
            (t for t in trigger_conditions if t['emotion'] == current_emotion),
            None
        )
        
        if not trigger:
            return {"poetry_moment": False}
        
        # Roll for poetry
        if random.random() < trigger['chance']:
            # Select appropriate poem lines
            poem_lines = await self._get_contextual_poetry(context)
            
            # Track usage
            poetry_used = poetry_data.get('poetry_used', [])
            poetry_used.append({
                "timestamp": datetime.now().isoformat(),
                "emotion": current_emotion,
                "context": context.get('scene_type', 'unknown')
            })
            
            # Update tracking
            await self._update_poetry_tracking(poetry_used)
            
            return {
                "poetry_moment": True,
                "suggested_lines": poem_lines,
                "emotion": current_emotion,
                "interpretation_required": True
            }
        
        return {"poetry_moment": False}
    
    async def _get_contextual_poetry(self, context: Dict[str, Any]) -> List[str]:
        """Get poetry lines appropriate to context"""
        emotion = context.get('lilith_emotion', 'neutral')
        trust = self._current_state['stats']['trust']
        
        # Character-specific variations for Queen of Thorns
        poetry_lines = []
        
        if emotion == "vulnerability" and trust > 60:
            poetry_lines.extend([
                "I am a garden of broken glass roses, and you... you tend me too gently for safety.",
                "Between heartbeats, I practice saying three words. They taste of thorns and starlight.",
                "The crown grows heavy when no one sees you wear it."
            ])
        elif emotion == "fear":
            poetry_lines.extend([
                "Don't disappear. The words taste of copper and every promise broken.",
                "Everyone swears they'll stay to see the roses bloom. I keep their masks as mulch.",
                "You are morning dew, and I the thorn that fears the sun."
            ])
        elif emotion == "passion":
            poetry_lines.extend([
                "Your skin tastes of prayers I've taught predators to speak.",
                "I plant gardens in the ruins of what you were, cultivate submission where dominance grew wild.",
                "We are binary stars, locked in orbits that remake the universe between us."
            ])
        elif emotion == "power":
            poetry_lines.extend([
                "Kneel, and I will show you how thorns can be tender.",
                "I transform wolves into roses, one petal-soft submission at a time.",
                "The network has no name, but my touch leaves invisible tattoos."
            ])
        
        return poetry_lines[:3]  # Return top 3 most relevant
    
    async def check_three_words_moment(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if this is a moment for the three words
        
        Args:
            context: Current scene context
            
        Returns:
            Dict with three words moment details
        """
        three_words_data = self._current_state['mechanics'].get('three_words', {})
        
        if three_words_data.get('spoken', False):
            return {"three_words_possible": False, "already_spoken": True}
        
        trust = self._current_state['stats']['trust']
        trust_threshold = three_words_data.get('trust_threshold', 95)
        emotional_threshold = three_words_data.get('emotional_threshold', 90)
        emotional_intensity = context.get('emotional_intensity', 0) * 100
        
        # Check basic thresholds
        if trust < trust_threshold or emotional_intensity < emotional_threshold:
            # But track near-misses
            if trust > 80 and emotional_intensity > 70:
                await self._track_near_miss(context)
                
                return {
                    "three_words_possible": False,
                    "near_miss": True,
                    "reaction": "She opens her mouth, three syllables rising like thorns through her throat, but she swallows them back down with practiced pain.",
                    "trust_needed": trust_threshold - trust,
                    "emotion_needed": emotional_threshold - emotional_intensity
                }
            
            return {"three_words_possible": False}
        
        # Check context triggers
        player_action = context.get('player_action', '').lower()
        triggers_present = []
        
        if "love" in player_action or "i love you" in player_action:
            triggers_present.append("player_confession")
        if context.get('location', '') in ["Private Chambers", "The Inner Garden", "The Mask Room"]:
            triggers_present.append("private_setting")
        if context.get('mask_removed', False):
            triggers_present.append("no_mask")
        if context.get('network_crisis_resolved', False):
            triggers_present.append("after_crisis")
        if context.get('transformation_completed_together', False):
            triggers_present.append("shared_purpose")
        
        if len(triggers_present) >= 2:  # Need multiple triggers
            return {
                "three_words_possible": True,
                "triggers": triggers_present,
                "buildup_description": (
                    "Something shifts in her eyes - all masks falling at once. The words that have lived "
                    "beneath her tongue for so long rise like prayers through thorns. Her lips part, trembling "
                    "with the weight of a crown she never meant to share."
                ),
                "requires_player_response": True
            }
        
        return {
            "three_words_possible": False,
            "triggers_present": triggers_present,
            "triggers_needed": 2 - len(triggers_present)
        }
    
    async def check_trust_test(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if Lilith needs to test player's loyalty to the network
        
        Args:
            context: Current scene context
            
        Returns:
            Dict with trust test details
        """
        trust = self._current_state['stats']['trust']
        
        if trust < 40 or trust > 80:
            return {"test_needed": False}
        
        test_triggers = []
        if context.get('asked_about_network', False):
            test_triggers.append("curiosity_about_network")
        if context.get('witnessed_transformation', False):
            test_triggers.append("saw_too_much")
        if context.get('met_other_network_members', False):
            test_triggers.append("expanding_connections")
        
        if len(test_triggers) >= 1:
            return {
                "test_needed": True,
                "test_type": random.choice([
                    "loyalty_choice",
                    "secret_keeping",
                    "participation_request"
                ]),
                "triggers": test_triggers,
                "description": (
                    "She watches you with calculating eyes. 'The garden has many secrets,' she says softly. "
                    "'Some tend roses, others become them. Which are you, I wonder?'"
                ),
                "consequences": {
                    "pass": "Deeper network access",
                    "fail": "Memory modification or exile"
                }
            }
        
        return {"test_needed": False}
    
    async def speak_three_words(self, player_response: str) -> Dict[str, Any]:
        """
        Handle the actual speaking of the three words
        
        Args:
            player_response: How player responds to the moment
            
        Returns:
            Dict with the outcome
        """
        three_words_data = self._current_state['mechanics'].get('three_words', {})
        
        # Determine outcome based on player response
        response_lower = player_response.lower()
        
        if any(phrase in response_lower for phrase in ["i love you", "love you too", "always loved"]):
            outcome = "mutual_confession"
            her_words = "I love you"
            aftermath = (
                "The words escape like butterflies through thorns. 'I love you,' she breathes, each syllable "
                "a thorn pulled from her throat. 'I've loved you since you first knelt in my garden and chose "
                "to bloom rather than wither.' Tears trace silver paths down her cheeks, watering seeds "
                "planted in secret."
            )
        elif any(word in response_lower for word in ["stay", "never leave", "tend your garden", "serve"]):
            outcome = "promise_response"
            her_words = "I love you"
            aftermath = (
                "Your promise breaks the last lock. 'I love you,' she whispers, the words barely audible, "
                "like roses blooming in winter. 'God help me, I love you. Every mask I wear, every thorn "
                "I grow, every predator I transform - it's all been searching for someone who could see "
                "the garden and choose to stay.'"
            )
        elif any(word in response_lower for word in ["wait", "don't", "scared", "not ready"]):
            outcome = "fear_reflected"
            her_words = "[unspoken]"
            aftermath = (
                "She sees her own fear reflected and the words wilt unborn. Her mouth closes, the three "
                "syllables swallowed like thorns. 'I... I understand. The garden asks much of those who "
                "would tend it. Perhaps... perhaps some roses are meant to grow in solitude.'"
            )
        else:
            outcome = "spoken_into_void"
            her_words = "I love you"
            aftermath = (
                "'I love you.' The words fall like rose petals onto marble - beautiful, futile, final. "
                "She doesn't wait for a response, turning away to face the masks on her walls. 'There. "
                "Now you know what grows beneath the thorns. Whether you stay to tend this garden or "
                "flee from its shadows... that choice was always yours.'"
            )
        
        # Update database
        await self._mark_words_spoken(outcome, her_words, player_response)
        
        # Update relationship
        trust_change = 0
        if outcome == "mutual_confession":
            trust_change = 15
        elif outcome == "promise_response":
            trust_change = 10
        elif outcome == "fear_reflected":
            trust_change = -5
        else:
            trust_change = 5
        
        await self._update_trust(self._current_state['stats']['trust'] + trust_change)
        
        return {
            "words_spoken": her_words != "[unspoken]",
            "outcome": outcome,
            "her_words": her_words,
            "aftermath": aftermath,
            "relationship_change": "deepened" if trust_change > 0 else "complicated",
            "new_dynamic": "words_between_us",
            "network_impact": "Your position in the garden is forever changed"
        }
    
    async def get_dialogue_style(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get appropriate dialogue style based on current state
        
        Args:
            context: Current scene context
            
        Returns:
            Dict with dialogue guidance
        """
        trust = self._current_state['stats']['trust']
        mask = self._current_state['current_mask']
        emotion = context.get('lilith_emotion', 'neutral')
        
        # Base patterns from character data
        patterns = self._current_state.get('dialogue_patterns', {})
        
        # Determine pattern based on trust
        if trust < 30:
            pattern_key = "trust_low"
        elif trust < 60:
            pattern_key = "trust_medium"  
        elif trust < 85:
            pattern_key = "trust_high"
        else:
            pattern_key = "vulnerability_showing"
        
        base_pattern = patterns.get(pattern_key, "")
        
        # Modify based on mask
        style_modifiers = {
            "Porcelain Goddess": {
                "tone": "commanding",
                "formality": "high",
                "vulnerability": "none",
                "sample_phrases": ["Kneel before the thorns", "You may approach", "The garden has rules"]
            },
            "Leather Predator": {
                "tone": "dangerous", 
                "formality": "medium",
                "vulnerability": "none",
                "sample_phrases": ["Predators become prey here", "Test me and be transformed", "Come. Now."]
            },
            "Lace Vulnerability": {
                "tone": "soft_edges",
                "formality": "low",
                "vulnerability": "glimpses",
                "sample_phrases": ["Please...", "The garden grows lonely", "Don't vanish like morning dew"]
            },
            "No Mask": {
                "tone": "raw",
                "formality": "none",
                "vulnerability": "complete",
                "sample_phrases": ["I'm drowning in roses", "Hold me", "Why did you stay when others fled?"]
            }
        }
        
        style = style_modifiers.get(mask, style_modifiers["Porcelain Goddess"])
        
        # Add poetry if applicable
        poetry_check = await self.check_poetry_moment(context)
        
        # Add network-specific language
        network_phrases = []
        if context.get('network_topic', False):
            network_phrases = [
                "The garden tends itself",
                "Thorns protect the roses", 
                "We transform through cultivation",
                "The network has no name but infinite reach"
            ]
        
        return {
            "base_pattern": base_pattern,
            "style": style,
            "emotion": emotion,
            "poetry_suggested": poetry_check.get('poetry_moment', False),
            "poetry_lines": poetry_check.get('suggested_lines', []),
            "special_phrases": self._get_special_phrases(trust, emotion),
            "network_phrases": network_phrases,
            "forbidden_words": ["goodbye", "leave", "forever"] if trust < 50 else []
        }
    
    def _get_special_phrases(self, trust: int, emotion: str) -> List[str]:
        """Get Lilith's special phrases based on state"""
        phrases = []
        
        if emotion == "fear":
            phrases.extend(["Don't disappear", "Stay in the garden", "Promise me"])
        elif emotion == "passion":
            phrases.extend(["Mine to cultivate", "Bloom for me", "Marked by thorns"])
        elif emotion == "vulnerability" and trust > 60:
            phrases.extend(["I can't lose another", "Help me tend them", "Why do you remain?"])
        elif emotion == "dominant":
            phrases.extend(["Kneel", "Submit to transformation", "Let me remake you"])
        elif emotion == "protective":
            phrases.extend(["The garden protects its own", "No one hurts my roses", "Safe here"])
        
        return phrases
    
    async def _mark_words_spoken(self, outcome: str, words: str, player_response: str):
        """Mark the three words as spoken in database"""
        async with get_db_connection_context() as conn:
            three_words_data = self._current_state['mechanics'].get('three_words', {})
            three_words_data['spoken'] = words != "[unspoken]"
            three_words_data['outcome'] = outcome
            three_words_data['her_words'] = words
            three_words_data['player_response'] = player_response[:500]
            three_words_data['spoken_at'] = datetime.now().isoformat()
            
            await conn.execute(
                """
                UPDATE npc_special_mechanics
                SET mechanic_data = $4
                WHERE user_id = $1 AND conversation_id = $2 
                AND npc_id = $3 AND mechanic_type = 'three_words'
                """,
                self.user_id, self.conversation_id, self.npc_id,
                json.dumps(three_words_data)
            )
            
            # Also update main NPC record
            await conn.execute(
                """
                UPDATE NPCStats
                SET three_words_spoken = $4
                WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
                """,
                self.user_id, self.conversation_id, self.npc_id,
                words != "[unspoken]"
            )
    
    async def _update_trust(self, new_trust: int):
        """Update trust level"""
        canon_ctx = type('CanonicalContext', (), {
            'user_id': self.user_id,
            'conversation_id': self.conversation_id
        })()
        
        new_trust = max(-100, min(100, new_trust))
        
        async with get_db_connection_context() as conn:
            await canon.update_entity_canonically(
                canon_ctx, conn, "NPCStats", self.npc_id,
                {"trust": new_trust},
                f"Trust updated to {new_trust}"
            )
            
        self._current_state['stats']['trust'] = new_trust
    
    async def _get_story_flags(self) -> Dict[str, Any]:
        """Get current story flags"""
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow(
                """
                SELECT story_flags
                FROM story_states
                WHERE user_id = $1 AND conversation_id = $2 AND story_id = $3
                """,
                self.user_id, self.conversation_id, "the_queen_of_thorns"
            )
            
            return json.loads(row['story_flags']) if row else {}
    
    async def _get_next_appropriate_mask(self, trust: int, context: Dict[str, Any]) -> Optional[str]:
        """Determine appropriate mask based on trust and context"""
        mask_data = self._current_state['mechanics'].get('mask_system', {})
        available_masks = mask_data.get('available_masks', [])
        
        # Sort by trust requirement
        eligible_masks = [
            m for m in available_masks 
            if m['trust_required'] <= trust
        ]
        
        if not eligible_masks:
            return None
        
        # Context-based selection
        if context.get('threat_detected') or context.get('predator_present'):
            return "Leather Predator"
        elif context.get('is_private') and trust > 60:
            return "Lace Vulnerability"
        elif trust > 85 and context.get('emotional_intensity', 0) > 0.8:
            return "No Mask"
        
        # Default to highest eligible
        return eligible_masks[-1]['name']
    
    async def _update_mask_integrity(self, new_integrity: int):
        """Update mask integrity in database"""
        async with get_db_connection_context() as conn:
            mask_data = self._current_state['mechanics'].get('mask_system', {})
            mask_data['mask_integrity'] = max(0, min(100, new_integrity))
            
            await conn.execute(
                """
                UPDATE npc_special_mechanics
                SET mechanic_data = $4
                WHERE user_id = $1 AND conversation_id = $2 
                AND npc_id = $3 AND mechanic_type = 'mask_system'
                """,
                self.user_id, self.conversation_id, self.npc_id,
                json.dumps(mask_data)
            )
            
            self._current_state['mechanics']['mask_system'] = mask_data
    
    async def _change_mask(self, new_mask: str):
        """Change current mask"""
        canon_ctx = type('CanonicalContext', (), {
            'user_id': self.user_id,
            'conversation_id': self.conversation_id
        })()
        
        async with get_db_connection_context() as conn:
            await canon.update_entity_canonically(
                canon_ctx, conn, "NPCStats", self.npc_id,
                {"current_mask": new_mask},
                f"Lilith changes mask to {new_mask}"
            )
            
        self._current_state['current_mask'] = new_mask
    
    async def _update_poetry_tracking(self, poetry_used: List[Dict]):
        """Update poetry usage tracking"""
        async with get_db_connection_context() as conn:
            poetry_data = self._current_state['mechanics'].get('poetry_triggers', {})
            poetry_data['poetry_used'] = poetry_used[-20:]  # Keep last 20
            
            await conn.execute(
                """
                UPDATE npc_special_mechanics
                SET mechanic_data = $4
                WHERE user_id = $1 AND conversation_id = $2 
                AND npc_id = $3 AND mechanic_type = 'poetry_triggers'
                """,
                self.user_id, self.conversation_id, self.npc_id,
                json.dumps(poetry_data)
            )
    
    async def _track_near_miss(self, context: Dict[str, Any]):
        """Track when the three words almost emerge"""
        async with get_db_connection_context() as conn:
            three_words_data = self._current_state['mechanics'].get('three_words', {})
            near_misses = three_words_data.get('near_speaking_moments', [])
            
            near_misses.append({
                "timestamp": datetime.now().isoformat(),
                "trust": self._current_state['stats']['trust'],
                "context": context.get('scene_type', 'unknown'),
                "trigger": context.get('player_action', '')[:100]
            })
            
            three_words_data['near_speaking_moments'] = near_misses[-10:]  # Keep last 10
            
            await conn.execute(
                """
                UPDATE npc_special_mechanics
                SET mechanic_data = $4
                WHERE user_id = $1 AND conversation_id = $2 
                AND npc_id = $3 AND mechanic_type = 'three_words'
                """,
                self.user_id, self.conversation_id, self.npc_id,
                json.dumps(three_words_data)
            )
    
    async def check_trauma_trigger(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if current context triggers trauma response
        
        Args:
            context: Current scene context
            
        Returns:
            Dict with trauma trigger details
        """
        trauma_triggers = self._current_state.get('trauma_triggers', [])
        player_action = context.get('player_action', '').lower()
        
        triggered = []
        for trigger in trauma_triggers:
            if trigger.lower() in player_action:
                triggered.append(trigger)
        
        if not triggered:
            return {"trauma_triggered": False}
        
        # Determine response based on trigger
        responses = {
            "sudden departures": {
                "reaction": "panic",
                "description": "Her whole body goes rigid. 'No. No, you don't get to leave like that. Not like the others. Not after I've shown you the garden.'"
            },
            "i'll always be here": {
                "reaction": "bitter_laugh", 
                "description": "She laughs, but it's all thorns and no roses. 'They all say that. Right until the garden asks too much.'"
            },
            "being seen without consent": {
                "reaction": "rage",
                "description": "Her eyes flash dangerous crimson. 'You do NOT look beneath the mask without permission. The Queen chooses her revelations.'"
            },
            "betrayal": {
                "reaction": "shutdown",
                "description": "Something dies in her eyes. The mask becomes her face. 'I see. Another moth burned by getting too close.'"
            },
            "bright_lights": {
                "reaction": "anxiety",
                "description": "She flinches from the brightness, hand moving to adjust a mask that isn't there. 'Shadows are kinder to gardens like mine.'"
            }
        }
        
        primary_trigger = triggered[0]
        response = responses.get(primary_trigger, {
            "reaction": "defensive",
            "description": "Old wounds flare. Her walls slam back into place like thorns erupting from skin."
        })
        
        return {
            "trauma_triggered": True,
            "triggers": triggered,
            "reaction": response["reaction"],
            "description": response["description"],
            "trust_impact": -10,
            "suggestion": "Proceed very carefully. She's retreating into protective thorns."
        }
