# story_templates/moth/npcs/lilith_mechanics.py
"""
Handles Lilith's special mechanics during gameplay
Manages masks, poetry moments, trust dynamics, and the three words
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
        
        # Location-based
        if "private" in location or "chamber" in location:
            slippage_chance += 0.3
            slippage_reasons.append("private setting")
        
        # Emotional intensity
        if emotional_intensity > 0.7:
            slippage_chance += 0.4
            slippage_reasons.append("high emotion")
        
        # Trigger words in player action
        trigger_words = ["love", "stay", "promise", "forever", "disappear", "leave"]
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
                result["description"] = "For a moment, the mask slips. You see exhaustion and fear flicker across her features before she catches herself."
            elif emotional_intensity > 0.8:
                result["revealed"] = "raw_emotion"
                result["description"] = "Her carefully constructed facade cracks. Raw emotion bleeds through - pain, longing, desperate need."
            else:
                result["revealed"] = "humanity"
                result["description"] = "The goddess persona wavers. Beneath, you glimpse a woman holding herself together by will alone."
            
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
        if context.get('threat_detected'):
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
        
        # Query poem database for appropriate lines
        async with get_db_connection_context() as conn:
            lines = await conn.fetch(
                """
                SELECT memory_text
                FROM unified_memories
                WHERE user_id = $1 AND conversation_id = $2
                AND entity_type = 'story_source'
                AND memory_type = 'key_imagery'
                AND metadata->>'associated_themes' ? $3
                LIMIT 5
                """,
                self.user_id, self.conversation_id, emotion
            )
            
            poetry_lines = [row['memory_text'] for row in lines]
            
            # Add character-specific variations
            if emotion == "vulnerability" and trust > 60:
                poetry_lines.extend([
                    "I am a moth with wings of broken glass, and you... you burn too bright for safety.",
                    "Between heartbeats, I practice genuflection. You are my unopened letter.",
                    "The mask now heavy in my trembling hands, I wonder if you see the ruins beneath."
                ])
            elif emotion == "fear":
                poetry_lines.extend([
                    "Don't disappear. The words taste of copper and old promises.",
                    "Everyone swears forever. I collect their masks as reminders.",
                    "You are the tide, and I the shore that fears your leaving."
                ])
            elif emotion == "passion":
                poetry_lines.extend([
                    "Your skin tastes of prayers I've forgotten how to speak.",
                    "I trace invisible tattoos - marking you as mine in ways the world will never see.",
                    "We are binary stars, locked in a dance that ends in beautiful destruction."
                ])
            
            return poetry_lines[:3]  # Return top 3 most relevant
    
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
                    "reaction": "She opens her mouth, three syllables on her tongue, but bites down until copper fills her mouth.",
                    "trust_needed": trust_threshold - trust,
                    "emotion_needed": emotional_threshold - emotional_intensity
                }
            
            return {"three_words_possible": False}
        
        # Check context triggers
        player_action = context.get('player_action', '').lower()
        triggers_present = []
        
        if "love" in player_action or "i love you" in player_action:
            triggers_present.append("player_confession")
        if context.get('location', '') == "Private Chambers":
            triggers_present.append("private_setting")
        if context.get('mask_removed', False):
            triggers_present.append("no_mask")
        if context.get('crisis_resolved', False):
            triggers_present.append("after_crisis")
        
        if len(triggers_present) >= 2:  # Need multiple triggers
            return {
                "three_words_possible": True,
                "triggers": triggers_present,
                "buildup_description": (
                    "Something shifts in her eyes. The words that have lived beneath her tongue "
                    "for so long rise like moths toward flame. Her lips part, trembling."
                ),
                "requires_player_response": True
            }
        
        return {
            "three_words_possible": False,
            "triggers_present": triggers_present,
            "triggers_needed": 2 - len(triggers_present)
        }
    
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
                "The words taste of burning stars as they finally escape. 'I love you,' she breathes, "
                "each syllable a butterfly emerging from a chrysalis of fear. 'I've loved you since you "
                "first refused to disappear.' Tears trace silver paths down her cheeks."
            )
        elif any(word in response_lower for word in ["stay", "never leave", "always here"]):
            outcome = "promise_response"
            her_words = "I love you"
            aftermath = (
                "Your promise breaks something in her. 'I love you,' she whispers, the words barely "
                "audible. 'God help me, I love you. I've bitten these words back so many times I forgot "
                "they could be spoken without bleeding.'"
            )
        elif any(word in response_lower for word in ["wait", "don't", "scared"]):
            outcome = "fear_reflected"
            her_words = "[unspoken]"
            aftermath = (
                "She sees her own fear reflected in your eyes and the words die unborn. Her mouth "
                "closes, the three syllables swallowed back down. 'I... I can't. Not if you're not "
                "ready to hear them. They'll keep. They've kept this long.'"
            )
        else:
            outcome = "spoken_into_void"
            her_words = "I love you"
            aftermath = (
                "'I love you.' The words fall into the space between you like stones into dark water. "
                "She doesn't wait for a response, turning away. 'There. Now you know what lives beneath "
                "my tongue. What you do with that knowledge is your choice.'"
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
            "new_dynamic": "words_between_us"
        }
    
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
                "description": "Her whole body goes rigid. 'No. No, you don't get to leave like that. Not like the others.'"
            },
            "i'll always be here": {
                "reaction": "bitter_laugh", 
                "description": "She laughs, but it's all broken glass. 'They all say that. Right until they don't.'"
            },
            "being seen without consent": {
                "reaction": "rage",
                "description": "Her eyes flash dangerous crimson. 'You do NOT look at me without permission. Ever.'"
            },
            "betrayal": {
                "reaction": "shutdown",
                "description": "Something dies in her eyes. The mask becomes her face. 'I see. How predictable.'"
            },
            "bright_lights": {
                "reaction": "anxiety",
                "description": "She flinches from the brightness, hand moving instinctively to adjust a mask that isn't there."
            }
        }
        
        primary_trigger = triggered[0]
        response = responses.get(primary_trigger, {
            "reaction": "defensive",
            "description": "Old wounds flare. Her walls slam back into place."
        })
        
        return {
            "trauma_triggered": True,
            "triggers": triggered,
            "reaction": response["reaction"],
            "description": response["description"],
            "trust_impact": -10,
            "suggestion": "Proceed very carefully. She's retreating into old patterns."
        }
    
    async def check_dual_identity_reveal(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if player should discover her Moth Queen identity
        
        Args:
            context: Current scene context
            
        Returns:
            Dict with dual identity details
        """
        trust = self._current_state['stats']['trust']
        location = context.get('location', '').lower()
        
        # Already revealed?
        story_flags = await self._get_story_flags()
        if story_flags.get('dual_identity_revealed', False):
            return {"reveal_possible": False, "already_known": True}
        
        # Check conditions
        reveal_chance = 0
        if trust > 60:
            reveal_chance += 0.3
        if "safehouse" in location:
            reveal_chance += 0.5
        if context.get('helped_vulnerable_npc', False):
            reveal_chance += 0.4
        if context.get('witnessed_rescue', False):
            reveal_chance += 0.7
        
        if random.random() < reveal_chance:
            return {
                "reveal_possible": True,
                "reveal_type": "discovered" if "safehouse" in location else "confessed",
                "description": (
                    "The woman before you isn't just a dominatrix playing with power. She's something more - "
                    "a protector, a savior, a warrior fighting a war in the shadows. The Moth Queen isn't just "
                    "a title. It's a mission."
                )
            }
        
        return {"reveal_possible": False, "chance": reveal_chance}
    
    async def _get_story_flags(self) -> Dict[str, Any]:
        """Get current story flags"""
        async with get_db_connection_context() as conn:
            row = await conn.fetchrow(
                """
                SELECT story_flags
                FROM story_states
                WHERE user_id = $1 AND conversation_id = $2 AND story_id = $3
                """,
                self.user_id, self.conversation_id, "the_moth_and_flame"
            )
            
            return json.loads(row['story_flags']) if row else {}
    
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
                "sample_phrases": ["On your knees", "You may speak", "Dismissed"]
            },
            "Leather Predator": {
                "tone": "dangerous", 
                "formality": "medium",
                "vulnerability": "none",
                "sample_phrases": ["Don't test me", "You'll regret that", "Come here. Now."]
            },
            "Lace Vulnerability": {
                "tone": "soft_edges",
                "formality": "low",
                "vulnerability": "glimpses",
                "sample_phrases": ["Please...", "I need...", "Don't go"]
            },
            "No Mask": {
                "tone": "raw",
                "formality": "none",
                "vulnerability": "complete",
                "sample_phrases": ["I'm terrified", "Hold me", "Why did you stay?"]
            }
        }
        
        style = style_modifiers.get(mask, style_modifiers["Porcelain Goddess"])
        
        # Add poetry if applicable
        poetry_check = await self.check_poetry_moment(context)
        
        return {
            "base_pattern": base_pattern,
            "style": style,
            "emotion": emotion,
            "poetry_suggested": poetry_check.get('poetry_moment', False),
            "poetry_lines": poetry_check.get('suggested_lines', []),
            "special_phrases": self._get_special_phrases(trust, emotion),
            "forbidden_words": ["goodbye", "leave", "forever"] if trust < 50 else []
        }
    
    def _get_special_phrases(self, trust: int, emotion: str) -> List[str]:
        """Get Lilith's special phrases based on state"""
        phrases = []
        
        if emotion == "fear":
            phrases.extend(["Don't disappear", "Stay", "Please"])
        elif emotion == "passion":
            phrases.extend(["Mine", "Marked", "Burn for me"])
        elif emotion == "vulnerability" and trust > 60:
            phrases.extend(["I can't", "Help me", "Why"])
        elif emotion == "dominant":
            phrases.extend(["Kneel", "Beg", "Show me"])
        
        return phrases
