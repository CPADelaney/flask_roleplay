# memory/masks.py

import json
import logging
import random
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import asyncpg

from .connection import TransactionContext, with_transaction
from .core import UnifiedMemoryManager, Memory, MemoryType, MemorySignificance

logger = logging.getLogger("memory_masks")

class RevealType:
    """Types of revelations for progressive character development"""
    VERBAL_SLIP = "verbal_slip"  # Verbal slip revealing true nature
    BEHAVIOR = "behavior"        # Behavioral inconsistency
    EMOTIONAL = "emotional"      # Emotional reaction revealing depth
    PHYSICAL = "physical"        # Physical tell or appearance change
    KNOWLEDGE = "knowledge"      # Knowledge they shouldn't have
    OVERHEARD = "overheard"      # Overheard saying something revealing
    OBJECT = "object"            # Possession of revealing object
    OPINION = "opinion"          # Expressed opinion revealing true nature
    SKILL = "skill"              # Demonstration of unexpected skill
    HISTORY = "history"          # Revealed history inconsistent with persona

class RevealSeverity:
    """How significant/obvious the revelation is"""
    SUBTLE = 1     # Barely noticeable, could be dismissed
    MINOR = 2      # Noticeable but easily explained away
    MODERATE = 3   # Clearly inconsistent with presented persona
    MAJOR = 4      # Significant revelation of true nature
    COMPLETE = 5   # Mask completely drops, true nature fully revealed

class NPCMask:
    """Represents the facade an NPC presents versus their true nature"""
    def __init__(self, presented_traits=None, hidden_traits=None, reveal_history=None):
        self.presented_traits = presented_traits or {}
        self.hidden_traits = hidden_traits or {}
        self.reveal_history = reveal_history or []
        self.integrity = 100  # How intact the mask is (100 = perfect mask, 0 = completely revealed)
        
    def to_dict(self):
        return {
            "presented_traits": self.presented_traits,
            "hidden_traits": self.hidden_traits,
            "reveal_history": self.reveal_history,
            "integrity": self.integrity
        }
    
    @classmethod
    def from_dict(cls, data):
        mask = cls(data.get("presented_traits"), data.get("hidden_traits"), data.get("reveal_history"))
        mask.integrity = data.get("integrity", 100)
        return mask

class ProgressiveRevealManager:
    """
    Manages the progressive revelation of NPC true natures,
    tracking facade integrity and creating revelation events.
    """
    
    # Opposing trait pairs for mask/true nature contrast
    OPPOSING_TRAITS = {
        "kind": "cruel",
        "gentle": "harsh",
        "caring": "callous",
        "patient": "impatient",
        "humble": "arrogant",
        "honest": "deceptive",
        "selfless": "selfish",
        "supportive": "manipulative",
        "trusting": "suspicious",
        "relaxed": "controlling",
        "open": "secretive",
        "egalitarian": "domineering",
        "casual": "formal",
        "empathetic": "cold",
        "nurturing": "exploitative"
    }
    
    # Physical tells for different hidden traits
    PHYSICAL_TELLS = {
        "cruel": ["momentary smile at others' discomfort", "subtle gleam in eyes when causing pain", "fingers flexing as if eager to inflict harm"],
        "harsh": ["brief scowl before composing face", "jaw tightening when frustrated", "eyes hardening momentarily"],
        "callous": ["dismissive flick of the wrist", "eyes briefly glazing over during others' emotional moments", "impatient foot tapping"],
        "arrogant": ["subtle sneer quickly hidden", "looking down nose briefly", "momentary eye-roll"],
        "deceptive": ["micro-expression of calculation", "eyes darting briefly", "subtle change in vocal pitch"],
        "manipulative": ["predatory gaze quickly masked", "fingers steepling then separating", "momentary satisfied smirk"],
        "controlling": ["unconscious straightening of surroundings", "stiffening posture when not obeyed", "fingers drumming impatiently"],
        "domineering": ["stance widening to take up space", "chin raising imperiously before catching themselves", "hand gesture that suggests expectation of obedience"]
    }
    
    # Verbal slips for different hidden traits
    VERBAL_SLIPS = {
        "cruel": ["That will teach th-- I mean, I hope they learn from this experience", "The pain should be excru-- educational for them", "I enjoy seeing-- I mean, I hope they recover quickly"],
        "manipulative": ["Once they're under my-- I mean, once they understand my point", "They're so easy to contr-- convince", "Just as I've planned-- I mean, just as I'd hoped"],
        "domineering": ["They will obey-- I mean, they will understand", "I expect complete submis-- cooperation", "My orders are-- I mean, my suggestions are"],
        "deceptive": ["The lie is perfect-- I mean, the explanation is clear", "They never suspect that-- they seem to understand completely", "I've fabricated-- formulated a perfect response"]
    }
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
    
    def _get_safe_stat_value(self, value, default=50):
        """Safely get a stat value, returning default if None"""
        return value if value is not None else default
    
    @with_transaction
    async def initialize_npc_mask(self, npc_id: int, overwrite: bool = False, conn = None) -> Dict[str, Any]:
        """
        Create an initial mask for an NPC based on their attributes,
        generating contrasting presented vs. hidden traits
        """
        # First check if mask already exists
        if not overwrite:
            row = await conn.fetchrow("""
                SELECT mask_data FROM NPCMasks
                WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
            """, self.user_id, self.conversation_id, npc_id)
            
            if row:
                return {"message": "Mask already exists for this NPC", "already_exists": True}
        
        # Get NPC data
        row = await conn.fetchrow("""
            SELECT npc_name, dominance, cruelty, personality_traits, archetype_summary
            FROM NPCStats
            WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
        """, self.user_id, self.conversation_id, npc_id)
        
        if not row:
            return {"error": f"NPC with id {npc_id} not found"}
            
        npc_name, dominance, cruelty, personality_traits_json, archetype_summary = row
        
        # Handle null values with sensible defaults
        dominance = self._get_safe_stat_value(dominance)
        cruelty = self._get_safe_stat_value(cruelty)
        
        # Parse personality traits
        personality_traits = []
        if personality_traits_json:
            if isinstance(personality_traits_json, str):
                try:
                    personality_traits = json.loads(personality_traits_json)
                except json.JSONDecodeError:
                    personality_traits = []
            else:
                personality_traits = personality_traits_json
        
        # Generate presented and hidden traits
        presented_traits = {}
        hidden_traits = {}
        
        # Use dominance and cruelty to determine mask severity
        mask_depth = (dominance + cruelty) / 2
        
        # More dominant/cruel NPCs have more to hide
        num_masked_traits = int(mask_depth / 20) + 1  # 1-5 masked traits
        
        # Generate contrasting traits based on existing personality
        trait_candidates = {}
        for trait in personality_traits:
            trait_lower = trait.lower()
            
            # Find traits that have opposites in our OPPOSING_TRAITS dictionary
            reversed_dict = {v: k for k, v in self.OPPOSING_TRAITS.items()}
            
            if trait_lower in self.OPPOSING_TRAITS:
                # This is a "good" trait that could mask a "bad" one
                opposite = self.OPPOSING_TRAITS[trait_lower]
                trait_candidates[trait] = opposite
            elif trait_lower in reversed_dict:
                # This is already a "bad" trait, so it's part of the hidden nature
                hidden_traits[trait] = {"intensity": random.randint(60, 90)}
                
                # Generate a presented trait to mask it
                presented_traits[reversed_dict[trait_lower]] = {"confidence": random.randint(60, 90)}
        
        # If we don't have enough contrasting traits, add some
        if len(trait_candidates) < num_masked_traits:
            additional_needed = num_masked_traits - len(trait_candidates)
            
            # Choose random traits from OPPOSING_TRAITS
            available_traits = list(self.OPPOSING_TRAITS.keys())
            random.shuffle(available_traits)
            
            for i in range(min(additional_needed, len(available_traits))):
                trait = available_traits[i]
                opposite = self.OPPOSING_TRAITS[trait]
                
                if trait not in trait_candidates and trait not in presented_traits:
                    trait_candidates[trait] = opposite
        
        # Select traits to mask
        masked_traits = dict(list(trait_candidates.items())[:num_masked_traits])
        
        # Add to presented and hidden traits
        for presented, hidden in masked_traits.items():
            if presented not in presented_traits:
                presented_traits[presented] = {"confidence": random.randint(60, 90)}
            
            if hidden not in hidden_traits:
                hidden_traits[hidden] = {"intensity": random.randint(60, 90)}
        
        # Create mask object
        mask = NPCMask(presented_traits, hidden_traits, [])
        
        # Store in database
        await conn.execute("""
            INSERT INTO NPCMasks (user_id, conversation_id, npc_id, mask_data)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (user_id, conversation_id, npc_id)
            DO UPDATE SET mask_data = EXCLUDED.mask_data
        """, self.user_id, self.conversation_id, npc_id, json.dumps(mask.to_dict()))
        
        return {
            "npc_id": npc_id,
            "npc_name": npc_name,
            "mask_created": True,
            "presented_traits": presented_traits,
            "hidden_traits": hidden_traits,
            "message": "NPC mask created successfully"
        }
    
    @with_transaction
    async def get_npc_mask(self, npc_id: int, conn = None) -> Dict[str, Any]:
        """
        Retrieve an NPC's mask data
        """
        row = await conn.fetchrow("""
            SELECT mask_data FROM NPCMasks
            WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
        """, self.user_id, self.conversation_id, npc_id)
        
        if not row:
            # Try to initialize a mask
            result = await self.initialize_npc_mask(npc_id, conn=conn)
            
            if "error" in result:
                return result
            
            # Get the new mask
            row = await conn.fetchrow("""
                SELECT mask_data FROM NPCMasks
                WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
            """, self.user_id, self.conversation_id, npc_id)
            
            if not row:
                return {"error": "Failed to create mask"}
        
        mask_data_json = row[0]
        
        mask_data = {}
        if mask_data_json:
            if isinstance(mask_data_json, str):
                try:
                    mask_data = json.loads(mask_data_json)
                except json.JSONDecodeError:
                    mask_data = {}
            else:
                mask_data = mask_data_json
        
        # Get NPC basic info
        npc_row = await conn.fetchrow("""
            SELECT npc_name, dominance, cruelty
            FROM NPCStats
            WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
        """, self.user_id, self.conversation_id, npc_id)
        
        if not npc_row:
            return {"error": f"NPC with id {npc_id} not found"}
            
        npc_name, dominance, cruelty = npc_row
        
        # Handle null values
        dominance = self._get_safe_stat_value(dominance)
        cruelty = self._get_safe_stat_value(cruelty)
        
        # Create mask object
        mask = NPCMask.from_dict(mask_data)
        
        return {
            "npc_id": npc_id,
            "npc_name": npc_name,
            "dominance": dominance,
            "cruelty": cruelty,
            "presented_traits": mask.presented_traits,
            "hidden_traits": mask.hidden_traits,
            "integrity": mask.integrity,
            "reveal_history": mask.reveal_history
        }
    
    @with_transaction
    async def generate_mask_slippage(self, 
                                   npc_id: int, 
                                   trigger: str = None, 
                                   severity: int = None, 
                                   reveal_type: str = None,
                                   conn = None) -> Dict[str, Any]:
        """
        Generate a mask slippage event for an NPC based on their true nature
        """
        # Get mask data
        mask_result = await self.get_npc_mask(npc_id, conn=conn)
        
        if "error" in mask_result:
            return mask_result
            
        npc_name = mask_result["npc_name"]
        presented_traits = mask_result["presented_traits"]
        hidden_traits = mask_result["hidden_traits"]
        integrity = mask_result["integrity"]
        reveal_history = mask_result["reveal_history"]
        
        # Choose a severity level if not provided
        if severity is None:
            # Higher chance of subtle reveals early, more major reveals as integrity decreases
            if integrity > 80:
                severity_weights = [0.7, 0.2, 0.1, 0, 0]  # Mostly subtle
            elif integrity > 60:
                severity_weights = [0.4, 0.4, 0.2, 0, 0]  # Subtle to minor
            elif integrity > 40:
                severity_weights = [0.2, 0.3, 0.4, 0.1, 0]  # Minor to moderate
            elif integrity > 20:
                severity_weights = [0.1, 0.2, 0.3, 0.4, 0]  # Moderate to major
            else:
                severity_weights = [0, 0.1, 0.2, 0.4, 0.3]  # Major to complete
            
            severity_levels = [
                RevealSeverity.SUBTLE,
                RevealSeverity.MINOR,
                RevealSeverity.MODERATE,
                RevealSeverity.MAJOR,
                RevealSeverity.COMPLETE
            ]
            
            severity = random.choices(severity_levels, weights=severity_weights, k=1)[0]
        
        # Choose a reveal type if not provided
        if reveal_type is None:
            reveal_types = [
                RevealType.VERBAL_SLIP,
                RevealType.BEHAVIOR,
                RevealType.EMOTIONAL,
                RevealType.PHYSICAL,
                RevealType.KNOWLEDGE,
                RevealType.OPINION
            ]
            
            # Check if we've used any types recently to avoid repetition
            recent_types = [event["type"] for event in reveal_history[-3:]]
            available_types = [t for t in reveal_types if t not in recent_types]
            
            if not available_types:
                available_types = reveal_types
            
            reveal_type = random.choice(available_types)
        
        # Choose a hidden trait to reveal
        if hidden_traits:
            trait, trait_info = random.choice(list(hidden_traits.items()))
        else:
            # Fallback if no hidden traits defined
            trait = random.choice(["manipulative", "controlling", "domineering"])
            trait_info = {"intensity": random.randint(60, 90)}
        
        # Generate reveal description based on type and trait
        reveal_description = ""
        
        if reveal_type == RevealType.VERBAL_SLIP:
            if trait in self.VERBAL_SLIPS:
                slip = random.choice(self.VERBAL_SLIPS[trait])
                reveal_description = f"{npc_name} lets slip: \"{slip}\""
            else:
                reveal_description = f"{npc_name} momentarily speaks in a {trait} tone before catching themselves."
        
        elif reveal_type == RevealType.PHYSICAL:
            if trait in self.PHYSICAL_TELLS:
                tell = random.choice(self.PHYSICAL_TELLS[trait])
                reveal_description = f"{npc_name} displays a {tell} before resuming their usual demeanor."
            else:
                reveal_description = f"{npc_name}'s expression briefly shifts to something more {trait} before they compose themselves."
        
        elif reveal_type == RevealType.EMOTIONAL:
            reveal_description = f"{npc_name} has an uncharacteristic emotional reaction, revealing a {trait} side that's usually hidden."
        
        elif reveal_type == RevealType.BEHAVIOR:
            reveal_description = f"{npc_name}'s behavior momentarily shifts, showing a {trait} tendency that contradicts their usual persona."
        
        elif reveal_type == RevealType.KNOWLEDGE:
            reveal_description = f"{npc_name} reveals knowledge they shouldn't have, suggesting a {trait} side to their character."
        
        elif reveal_type == RevealType.OPINION:
            reveal_description = f"{npc_name} expresses an opinion that reveals {trait} tendencies, contrasting with their usual presented self."
        
        # If trigger provided, incorporate it
        if trigger:
            reveal_description += f" This was triggered by {trigger}."
        
        # Calculate integrity damage based on severity
        integrity_damage = {
            RevealSeverity.SUBTLE: random.randint(1, 3),
            RevealSeverity.MINOR: random.randint(3, 7),
            RevealSeverity.MODERATE: random.randint(7, 12),
            RevealSeverity.MAJOR: random.randint(12, 20),
            RevealSeverity.COMPLETE: random.randint(20, 40)
        }[severity]
        
        # Apply damage
        new_integrity = max(0, integrity - integrity_damage)
        
        # Create event record
        event = {
            "date": datetime.now().isoformat(),
            "type": reveal_type,
            "severity": severity,
            "trait_revealed": trait,
            "description": reveal_description,
            "integrity_before": integrity,
            "integrity_after": new_integrity,
            "trigger": trigger
        }
        
        # Update mask
        reveal_history.append(event)
        
        mask = NPCMask(presented_traits, hidden_traits, reveal_history)
        mask.integrity = new_integrity
        
        # Save to database
        await conn.execute("""
            UPDATE NPCMasks
            SET mask_data = $1
            WHERE user_id = $2 AND conversation_id = $3 AND npc_id = $4
        """, json.dumps(mask.to_dict()), self.user_id, self.conversation_id, npc_id)
        
        # Add to player journal
        await conn.execute("""
            INSERT INTO PlayerJournal (
                user_id, conversation_id, entry_type, entry_text, timestamp
            )
            VALUES ($1, $2, 'npc_revelation', $3, CURRENT_TIMESTAMP)
        """, self.user_id, self.conversation_id,
            f"Observed {npc_name} reveal: {reveal_description}")
        
        # If integrity falls below thresholds, trigger special events
        special_event = None
        
        if new_integrity <= 50 and integrity > 50:
            special_event = {
                "type": "mask_threshold",
                "threshold": 50,
                "message": f"{npc_name}'s mask is beginning to crack significantly. Their true nature is becoming more difficult to hide."
            }
        elif new_integrity <= 20 and integrity > 20:
            special_event = {
                "type": "mask_threshold",
                "threshold": 20,
                "message": f"{npc_name}'s facade is crumbling. Their true nature is now plainly visible to those paying attention."
            }
        elif new_integrity <= 0 and integrity > 0:
            special_event = {
                "type": "mask_threshold",
                "threshold": 0,
                "message": f"{npc_name}'s mask has completely fallen away. They no longer attempt to hide their true nature from you."
            }
        
        # Add this event to NPCMemoryManager
        memory_manager = UnifiedMemoryManager(
            entity_type="npc",
            entity_id=npc_id,
            user_id=self.user_id,
            conversation_id=self.conversation_id
        )
        
        await memory_manager.add_memory(
            Memory(
                text=reveal_description,
                memory_type=MemoryType.OBSERVATION,
                significance=RevealSeverity.MODERATE + 1,  # Match severity to significance
                emotional_intensity=50,
                tags=["mask_slippage", "character_reveal", trait],
                metadata={
                    "reveal_type": reveal_type,
                    "trait_revealed": trait,
                    "severity": severity,
                    "integrity_change": integrity - new_integrity
                },
                timestamp=datetime.now()
            ),
            conn=conn
        )
        
        return {
            "npc_id": npc_id,
            "npc_name": npc_name,
            "reveal_type": reveal_type,
            "severity": severity,
            "trait_revealed": trait,
            "description": reveal_description,
            "integrity_before": integrity,
            "integrity_after": new_integrity,
            "special_event": special_event
        }
    
    @with_transaction
    async def check_for_automated_reveals(self, conn = None) -> List[Dict[str, Any]]:
        """
        Check for automatic reveals based on various triggers
        """
        # Get all NPCs with masks
        rows = await conn.fetch("""
            SELECT m.npc_id, m.mask_data, n.npc_name, n.dominance, n.cruelty
            FROM NPCMasks m
            JOIN NPCStats n ON m.npc_id = n.npc_id 
                AND m.user_id = n.user_id 
                AND m.conversation_id = n.conversation_id
            WHERE m.user_id = $1 AND m.conversation_id = $2
        """, self.user_id, self.conversation_id)
        
        # Get current time
        time_row = await conn.fetchrow("""
            SELECT value FROM CurrentRoleplay
            WHERE user_id = $1 AND conversation_id = $2 AND key = 'TimeOfDay'
        """, self.user_id, self.conversation_id)
        
        time_of_day = time_row[0] if time_row else "Morning"
        
        # Each time period has a chance of reveal for each NPC
        reveal_chance = {
            "Morning": 0.1,
            "Afternoon": 0.15,
            "Evening": 0.2,
            "Night": 0.25  # Higher chance at night when guards are down
        }
        
        base_chance = reveal_chance.get(time_of_day, 0.15)
        reveals = []
        
        for row in rows:
            npc_id = row["npc_id"]
            mask_data_json = row["mask_data"]
            npc_name = row["npc_name"]
            dominance = self._get_safe_stat_value(row["dominance"])
            cruelty = self._get_safe_stat_value(row["cruelty"])
            
            mask_data = {}
            if mask_data_json:
                if isinstance(mask_data_json, str):
                    try:
                        mask_data = json.loads(mask_data_json)
                    except json.JSONDecodeError:
                        mask_data = {}
                else:
                    mask_data = mask_data_json
            
            mask = NPCMask.from_dict(mask_data)
            
            # Higher dominance/cruelty increases chance of slip
            modifier = (dominance + cruelty) / 200  # 0.0 to 1.0
            
            # Lower integrity increases chance of slip
            integrity_factor = (100 - mask.integrity) / 100  # 0.0 to 1.0
            
            # Calculate final chance
            final_chance = base_chance + (modifier * 0.2) + (integrity_factor * 0.3)
            
            # Roll for reveal
            if random.random() < final_chance:
                # Generate a reveal
                reveal_result = await self.generate_mask_slippage(npc_id, conn=conn)
                
                if "error" not in reveal_result:
                    reveals.append(reveal_result)
        
        return reveals
    
    @with_transaction
    async def get_perception_difficulty(self, npc_id: int, conn = None) -> Dict[str, Any]:
        """
        Calculate how difficult it is to see through an NPC's mask
        """
        # Get mask data
        row = await conn.fetchrow("""
            SELECT mask_data FROM NPCMasks
            WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
        """, self.user_id, self.conversation_id, npc_id)
        
        if not row:
            return {"error": f"No mask found for NPC with id {npc_id}"}
            
        mask_data_json = row[0]
        
        mask_data = {}
        if mask_data_json:
            if isinstance(mask_data_json, str):
                try:
                    mask_data = json.loads(mask_data_json)
                except json.JSONDecodeError:
                    mask_data = {}
            else:
                mask_data = mask_data_json
        
        # Get NPC stats
        row = await conn.fetchrow("""
            SELECT dominance, cruelty
            FROM NPCStats
            WHERE user_id = $1 AND conversation_id = $2 AND npc_id = $3
        """, self.user_id, self.conversation_id, npc_id)
        
        if not row:
            return {"error": f"NPC with id {npc_id} not found"}
            
        dominance, cruelty = row
        
        # Handle null values
        dominance = self._get_safe_stat_value(dominance)
        cruelty = self._get_safe_stat_value(cruelty)
        
        # Get player stats
        row = await conn.fetchrow("""
            SELECT mental_resilience, confidence
            FROM PlayerStats
            WHERE user_id = $1 AND conversation_id = $2 AND player_name = 'Chase'
        """, self.user_id, self.conversation_id)
        
        if row:
            mental_resilience, confidence = row
        else:
            mental_resilience, confidence = 50, 50  # Default values
        
        mask = NPCMask.from_dict(mask_data)
        
        # Calculate base difficulty based on integrity
        base_difficulty = mask.integrity / 2  # 0-50
        
        # Add difficulty based on dominance/cruelty (higher = better at deception)
        stat_factor = (dominance + cruelty) / 4  # 0-50
        
        # Calculate total difficulty
        total_difficulty = base_difficulty + stat_factor
        
        # Calculate player's perception ability
        perception_ability = (mental_resilience + confidence) / 2
        
        # Calculate final difficulty rating
        if perception_ability > 0:
            relative_difficulty = total_difficulty / perception_ability
        else:
            relative_difficulty = total_difficulty
        
        difficulty_rating = ""
        if relative_difficulty < 0.5:
            difficulty_rating = "Very Easy"
        elif relative_difficulty < 0.8:
            difficulty_rating = "Easy"
        elif relative_difficulty < 1.2:
            difficulty_rating = "Moderate"
        elif relative_difficulty < 1.5:
            difficulty_rating = "Difficult"
        else:
            difficulty_rating = "Very Difficult"
        
        return {
            "npc_id": npc_id,
            "mask_integrity": mask.integrity,
            "difficulty_score": total_difficulty,
            "player_perception": perception_ability,
            "relative_difficulty": relative_difficulty,
            "difficulty_rating": difficulty_rating
        }

# Create the necessary table if it doesn't exist
async def create_mask_tables():
    """Create the necessary tables for the mask system if they don't exist."""
    from db.connection import get_db_connection_context
    
    async with get_db_connection_context() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS NPCMasks (
                user_id INTEGER NOT NULL,
                conversation_id INTEGER NOT NULL,
                npc_id INTEGER NOT NULL,
                mask_data JSONB NOT NULL,
                last_updated TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, conversation_id, npc_id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_npc_masks_lookup 
            ON NPCMasks(user_id, conversation_id, npc_id);
        """)
