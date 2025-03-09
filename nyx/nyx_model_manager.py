# nyx/nyx_model_manager.py

import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
import asyncpg

from db.connection import get_db_connection
from nyx.memory_system import NyxMemorySystem

logger = logging.getLogger(__name__)

class UserModelManager:
    """
    Manages persistent user models that track preferences, patterns, and behaviors
    across multiple game sessions.
    """
    
    def __init__(self, user_id: int, conversation_id: Optional[int] = None):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.memory_system = NyxMemorySystem(user_id, conversation_id)
        
        # Cache for current user model
        self._model_cache = None
        self._cache_timestamp = None
        self._cache_ttl = 300  # 5 minutes
    
    async def get_user_model(self) -> Dict[str, Any]:
        """
        Get current user model, updating cache if necessary.
        
        Returns:
            Complete user model with preferences, behavior patterns, etc.
        """
        # Check cache first
        if self._is_cache_valid():
            return self._model_cache
        
        async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
            async with pool.acquire() as conn:
                # Get persistent user model
                row = await conn.fetchrow(
                    "SELECT model_data FROM UserModels WHERE user_id = $1",
                    self.user_id
                )
                
                if row:
                    user_model = json.loads(row["model_data"])
                else:
                    # Initialize a new model if none exists
                    user_model = self._initialize_empty_model()
                    await conn.execute(
                        "INSERT INTO UserModels (user_id, model_data) VALUES ($1, $2)",
                        self.user_id, json.dumps(user_model)
                    )
                
                # Get current session data if in a game
                if self.conversation_id:
                    session_row = await conn.fetchrow(
                        "SELECT value FROM CurrentRoleplay WHERE user_id = $1 AND conversation_id = $2 AND key = 'UserSession'",
                        self.user_id, self.conversation_id
                    )
                    
                    if session_row:
                        session_data = json.loads(session_row["value"])
                        # Merge session data into model
                        user_model["current_session"] = session_data
                
                # Update cache
                self._model_cache = user_model
                self._cache_timestamp = datetime.now()
                
                return user_model
    
    async def update_user_model(self, updates: Dict[str, Any], persist: bool = True) -> Dict[str, Any]:
        """
        Update user model with new information.
        
        Args:
            updates: Dictionary of updates to apply to the model
            persist: Whether to persist changes to database
            
        Returns:
            Updated user model
        """
        # Get current model
        user_model = await self.get_user_model()
        
        # Apply updates
        self._apply_updates(user_model, updates)
        
        # Persist if requested
        if persist:
            async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
                async with pool.acquire() as conn:
                    # Update persistent model (excluding current_session)
                    persistent_model = user_model.copy()
                    persistent_model.pop("current_session", None)
                    
                    await conn.execute(
                        "UPDATE UserModels SET model_data = $1, updated_at = CURRENT_TIMESTAMP WHERE user_id = $2",
                        json.dumps(persistent_model), self.user_id
                    )
                    
                    # Update session data if in a game
                    if self.conversation_id and "current_session" in user_model:
                        await conn.execute(
                            """
                            INSERT INTO CurrentRoleplay (user_id, conversation_id, key, value)
                            VALUES ($1, $2, 'UserSession', $3)
                            ON CONFLICT (user_id, conversation_id, key) 
                            DO UPDATE SET value = $3
                            """,
                            self.user_id, self.conversation_id, json.dumps(user_model["current_session"])
                        )
        
        # Update cache
        self._model_cache = user_model
        self._cache_timestamp = datetime.now()
        
        return user_model
    
    async def track_kink_preference(
        self, 
        kink_name: str, 
        intensity: float = 0.5, 
        detected_from: str = None
    ) -> Dict[str, Any]:
        """
        Track detected kink preference, updating user model.
        
        Args:
            kink_name: Name of the kink (e.g., "ass", "goth", "tattoos")
            intensity: Detected intensity of preference (0.0-1.0)
            detected_from: Source of detection (e.g., "explicit_mention", "reaction")
            
        Returns:
            Updated user model
        """
        user_model = await self.get_user_model()
        
        # Get current kink profile
        kink_profile = user_model.get("kink_profile", {})
        
        # Get or create kink entry
        kink_entry = kink_profile.get(kink_name, {
            "level": 0,
            "first_detected": datetime.now().isoformat(),
            "detection_count": 0,
            "last_detected": None,
            "sources": []
        })
        
        # Update kink entry
        kink_entry["detection_count"] += 1
        kink_entry["last_detected"] = datetime.now().isoformat()
        
        # Add detection source if provided
        if detected_from:
            if "sources" not in kink_entry:
                kink_entry["sources"] = []
            
            kink_entry["sources"].append({
                "source": detected_from,
                "timestamp": datetime.now().isoformat(),
                "intensity": intensity
            })
            
            # Keep only last 10 sources
            if len(kink_entry["sources"]) > 10:
                kink_entry["sources"] = kink_entry["sources"][-10:]
        
        # Calculate new level based on detections and intensity
        if kink_entry["level"] < 1 and kink_entry["detection_count"] >= 1:
            kink_entry["level"] = 1
        elif kink_entry["level"] < 2 and kink_entry["detection_count"] >= 3:
            kink_entry["level"] = 2
        elif kink_entry["level"] < 3 and kink_entry["detection_count"] >= 5:
            kink_entry["level"] = 3
        elif kink_entry["level"] < 4 and kink_entry["detection_count"] >= 10:
            kink_entry["level"] = 4
        
        # Update user model
        kink_profile[kink_name] = kink_entry
        user_model["kink_profile"] = kink_profile
        
        # Store memory about this kink detection
        await self.memory_system.add_memory(
            memory_text=f"Player showed interest in '{kink_name}' with intensity {intensity:.2f}",
            memory_type="observation",
            memory_scope="user",  # User-level persistence
            significance=3 + kink_entry["level"],  # Higher level = more significant
            tags=["kink_preference", kink_name],
            metadata={
                "kink_name": kink_name,
                "intensity": intensity,
                "source": detected_from,
                "kink_level": kink_entry["level"]
            }
        )
        
        # If this appears to be a strong preference (high level), also create a reflection
        if kink_entry["level"] >= 3:
            await self.memory_system.generate_reflection(topic=f"{kink_name}_preference")
        
        # Update user model
        return await self.update_user_model({"kink_profile": kink_profile})
    
    async def track_behavior_pattern(
        self,
        pattern_type: str,
        pattern_value: str,
        intensity: float = 0.5,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Track a behavior pattern in the user model.
        
        Args:
            pattern_type: Type of pattern (e.g., "response_style", "aggression", "submission")
            pattern_value: Specific value detected
            intensity: Strength of the pattern (0.0-1.0)
            context: Optional context information
            
        Returns:
            Updated user model
        """
        user_model = await self.get_user_model()
        
        # Get behavior patterns
        behavior_patterns = user_model.get("behavior_patterns", {})
        
        # Get or create pattern category
        pattern_category = behavior_patterns.get(pattern_type, {
            "occurrences": 0,
            "values": {},
            "first_detected": datetime.now().isoformat(),
        })
        
        # Update category
        pattern_category["occurrences"] += 1
        pattern_category["last_detected"] = datetime.now().isoformat()
        
        # Add or update value
        if pattern_value not in pattern_category["values"]:
            pattern_category["values"][pattern_value] = {
                "count": 0,
                "intensity_sum": 0,
                "first_seen": datetime.now().isoformat()
            }
        
        value_data = pattern_category["values"][pattern_value]
        value_data["count"] += 1
        value_data["intensity_sum"] += intensity
        value_data["last_seen"] = datetime.now().isoformat()
        value_data["average_intensity"] = value_data["intensity_sum"] / value_data["count"]
        
        # Update behavior patterns
        behavior_patterns[pattern_type] = pattern_category
        user_model["behavior_patterns"] = behavior_patterns
        
        # Store memory about this behavior pattern
        memory_text = f"Player exhibited {pattern_type} behavior: {pattern_value} (intensity: {intensity:.2f})"
        
        await self.memory_system.add_memory(
            memory_text=memory_text,
            memory_type="observation",
            memory_scope="user",  # User-level persistence
            significance=4 if value_data["count"] > 3 else 3,  # More significant if repeated
            tags=["behavior_pattern", pattern_type, pattern_value],
            metadata={
                "pattern_type": pattern_type,
                "pattern_value": pattern_value,
                "intensity": intensity,
                "occurrence_count": value_data["count"],
                "context": context
            }
        )
        
        # If this is a repeated pattern, create a reflection
        if value_data["count"] >= 5:
            await self.memory_system.generate_reflection(topic=f"{pattern_type}_behavior")
        
        # Update user model
        return await self.update_user_model({"behavior_patterns": behavior_patterns})
    
    async def track_conversation_response(
        self,
        user_message: str,
        nyx_response: str,
        user_reaction: str = None,
        conversation_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Track a conversation interaction to learn from user responses.
        
        Args:
            user_message: What the user said
            nyx_response: What Nyx responded with
            user_reaction: Optional follow-up from user
            conversation_context: Additional context
            
        Returns:
            Updated user model
        """
        user_model = await self.get_user_model()
        
        # Get conversation history
        conversation_patterns = user_model.get("conversation_patterns", {
            "response_types": {},
            "reaction_patterns": {},
            "tracked_conversations": []
        })
        
        # TODO: Implement conversation pattern analysis
        # This would analyze what kinds of responses the user engages with most,
        # what tone they respond best to, etc.
        
        # Basic tracking of conversation
        tracked_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message[:100],  # Truncate for storage
            "nyx_response_sample": nyx_response[:100],  # Truncate
            "context": conversation_context,
            "user_reaction": user_reaction
        }
        
        conversation_patterns["tracked_conversations"].append(tracked_entry)
        
        # Keep only most recent 20 conversations
        if len(conversation_patterns["tracked_conversations"]) > 20:
            conversation_patterns["tracked_conversations"] = conversation_patterns["tracked_conversations"][-20:]
        
        # Store memory
        await self.memory_system.add_memory(
            memory_text=f"Conversation interaction - User: '{user_message[:50]}...' Nyx: '{nyx_response[:50]}...'",
            memory_type="observation",
            memory_scope="user",
            significance=3,
            tags=["conversation_pattern"],
            metadata={
                "user_message": user_message[:200],
                "nyx_response": nyx_response[:200],
                "user_reaction": user_reaction,
                "context": conversation_context
            }
        )
        
        # Update user model
        return await self.update_user_model({"conversation_patterns": conversation_patterns})
    
    async def get_response_guidance(self) -> Dict[str, Any]:
        """
        Generate guidance for how Nyx should respond to this user based on model.
        
        Returns:
            Guidance object with personality traits, tone suggestions, etc.
        """
        user_model = await self.get_user_model()
        
        # Get recent reflections on player
        reflections = await self.memory_system.retrieve_memories(
            query="player personality preferences behavior",
            scopes=["user", "game"],
            memory_types=["reflection"],
            limit=3,
            min_significance=4
        )
        
        reflection_texts = [r["memory_text"] for r in reflections]
        
        # Extract key information from user model
        kink_profile = user_model.get("kink_profile", {})
        top_kinks = sorted(
            [(k, v["level"]) for k, v in kink_profile.items()],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        behavior_patterns = user_model.get("behavior_patterns", {})
        
        # Build guidance
        guidance = {
            "reflections": reflection_texts,
            "top_kinks": top_kinks,
            "suggested_intensity": self._calculate_suggested_intensity(user_model),
            "content_themes": self._extract_content_themes(user_model),
            "personality_traits": self._extract_personality_guidance(user_model),
            "behavior_patterns": self._format_behavior_patterns(behavior_patterns)
        }
        
        return guidance
    
    def _initialize_empty_model(self) -> Dict[str, Any]:
        """Initialize a new empty user model."""
        return {
            "user_id": self.user_id,
            "created_at": datetime.now().isoformat(),
            "kink_profile": {},
            "behavior_patterns": {},
            "conversation_patterns": {
                "response_types": {},
                "reaction_patterns": {},
                "tracked_conversations": []
            },
            "personality_assessment": {
                "dominance_preference": 0,  # -100 to 100 scale
                "intensity_preference": 0,   # 0 to 100 scale
                "humiliation_tolerance": 0,  # 0 to 100 scale
                "creative_tolerance": 0      # 0 to 100 scale (tolerance for surreal/creative content)
            },
            "version": 1
        }
    
    def _is_cache_valid(self) -> bool:
        """Check if cache is valid."""
        if not self._model_cache or not self._cache_timestamp:
            return False
            
        seconds_since_update = (datetime.now() - self._cache_timestamp).total_seconds()
        return seconds_since_update < self._cache_ttl
    
    def _apply_updates(self, model: Dict[str, Any], updates: Dict[str, Any]):
        """Apply updates to user model, with proper merging for nested structures."""
        for key, value in updates.items():
            if key in model and isinstance(model[key], dict) and isinstance(value, dict):
                # Recursive merge for dicts
                self._apply_updates(model[key], value)
            else:
                # Direct update for other types
                model[key] = value
    
    def _calculate_suggested_intensity(self, user_model: Dict[str, Any]) -> float:
        """Calculate suggested intensity level based on user model."""
        # Start with base preference from personality assessment
        base_intensity = user_model.get("personality_assessment", {}).get("intensity_preference", 50) / 100.0
        
        # Adjust based on behavior patterns
        behavior = user_model.get("behavior_patterns", {})
        
        # If user has shown positive response to intensity, increase
        aggression = behavior.get("aggression", {}).get("occurrences", 0)
        submission = behavior.get("submission", {}).get("occurrences", 0)
        
        # More aggressive users might want slightly higher intensity
        if aggression > submission:
            base_intensity += 0.1
        
        # Cap between 0.2 and 0.9
        return max(0.2, min(0.9, base_intensity))
    
    def _extract_content_themes(self, user_model: Dict[str, Any]) -> List[str]:
        """Extract content themes based on user preferences."""
        themes = []
        
        # Extract from kink profile
        kink_profile = user_model.get("kink_profile", {})
        for kink, data in kink_profile.items():
            if data.get("level", 0) >= 2:
                themes.append(kink)
        
        # Add from behavior patterns
        behavior = user_model.get("behavior_patterns", {})
        
        # Add specific theme preferences
        return themes
    
    def _extract_personality_guidance(self, user_model: Dict[str, Any]) -> Dict[str, float]:
        """Extract personality guidance for Nyx based on user model."""
        personality = {}
        
        # Extract from personality assessment
        assessment = user_model.get("personality_assessment", {})
        
        # Normalize to 0-1 range
        if "dominance_preference" in assessment:
            personality["dominance"] = (assessment["dominance_preference"] + 100) / 200.0
            
        if "intensity_preference" in assessment:
            personality["intensity"] = assessment["intensity_preference"] / 100.0
            
        if "humiliation_tolerance" in assessment:
            personality["cruelty"] = assessment["humiliation_tolerance"] / 100.0
        
        return personality
    
    def _format_behavior_patterns(self, behavior_patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Format behavior patterns for guidance."""
        result = {}
        
        for category, data in behavior_patterns.items():
            if "values" in data:
                # Find most common value
                most_common = sorted(
                    [(v, d["count"]) for v, d in data["values"].items()],
                    key=lambda x: x[1],
                    reverse=True
                )
                
                if most_common:
                    result[category] = most_common[0][0]
        
        return result

# Initialize database tables for user model
async def initialize_user_model_tables():
    """Create database tables for user model storage."""
    async with asyncpg.create_pool(dsn=get_db_connection()) as pool:
        async with pool.acquire() as conn:
            # Create user models table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS UserModels (
                    user_id INTEGER PRIMARY KEY,
                    model_data JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """)
