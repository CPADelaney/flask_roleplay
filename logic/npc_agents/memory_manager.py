# logic/memory_manager.py

import asyncio
import json
import logging
import os
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union

# Import necessary dependencies
try:
    import asyncpg
    import openai
except ImportError:
    logging.warning("Some dependencies (asyncpg, openai) may not be installed.")

# Import memory subsystem components
from memory.core import Memory, MemoryType, MemorySignificance, UnifiedMemoryManager
from memory.emotional import EmotionalMemoryManager
from memory.schemas import MemorySchemaManager
from memory.flashbacks import FlashbackManager
from memory.masks import ProgressiveRevealManager
from memory.semantic import SemanticMemoryManager
from memory.reconsolidation import ReconsolidationManager
from memory.integrated import IntegratedMemorySystem

logger = logging.getLogger(__name__)

# Default DB connection string
DB_DSN = os.getenv("DB_DSN", "postgresql://user:pass@localhost:5432/yourdb")

class EnhancedMemoryManager:
    """
    Enhanced memory manager for NPCs that combines direct database access and
    memory subsystems for sophisticated memory management including:
    
    - Memory lifecycle (active, summarized, archived)
    - Emotional memory and state tracking
    - Personality-based biases and recall
    - Embedding-based retrieval
    - Schema application and interpretation
    - Flashbacks and progressive mask revealing
    - Memory propagation (rumors)
    - Reputation tracking
    """
    
    def __init__(
        self, 
        npc_id: int, 
        user_id: int, 
        conversation_id: int,
        db_pool=None,
        npc_personality: str = "neutral",
        npc_intelligence: float = 1.0,
        use_subsystems: bool = True
    ):
        """
        Initialize the enhanced memory manager for a specific NPC.
        
        Args:
            npc_id: ID of the NPC
            user_id: ID of the user/player
            conversation_id: ID of the current conversation
            db_pool: Optional asyncpg connection pool (will create connections if None)
            npc_personality: Personality type affecting memory biases ('gullible', 'skeptical', 'paranoid', 'neutral')
            npc_intelligence: Factor affecting memory decay rate (0.5-2.0)
            use_subsystems: Whether to use the memory subsystem managers (vs direct DB access only)
        """
        self.npc_id = npc_id
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.db_pool = db_pool
        self.npc_personality = npc_personality
        self.npc_intelligence = npc_intelligence
        self.use_subsystems = use_subsystems
        
        # For optimized memory access
        self.memory_cache = {}
        
        # Initialize subsystem managers if enabled
        if use_subsystems:
            # Initialize the core memory components
            self.core_memory = UnifiedMemoryManager(
                entity_type="npc",
                entity_id=npc_id,
                user_id=user_id,
                conversation_id=conversation_id
            )
            
            # Initialize specialized memory components
            self.emotional_memory = EmotionalMemoryManager(user_id, conversation_id)
            self.schema_manager = MemorySchemaManager(user_id, conversation_id)
            self.flashback_manager = FlashbackManager(user_id, conversation_id)
            self.mask_manager = ProgressiveRevealManager(user_id, conversation_id)
            self.semantic_manager = SemanticMemoryManager(user_id, conversation_id)
            
            # Initialize mask if not already done
            asyncio.create_task(self._ensure_mask_initialized())
    
    async def _ensure_mask_initialized(self):
        """Ensure the NPC has a mask initialized."""
        if not self.use_subsystems:
            return
            
        try:
            await self.mask_manager.initialize_npc_mask(self.npc_id)
        except Exception as e:
            logger.warning(f"Could not initialize mask for NPC {self.npc_id}: {e}")
    
    async def generate_embedding(self, text: str) -> list:
        """
        Generate an embedding vector for text using OpenAI's API.
        
        Args:
            text: The text to generate an embedding for
            
        Returns:
            List of floats representing the embedding vector
        """
        try:
            response = await openai.Embedding.acreate(
                model="text-embedding-ada-002",
                input=text
            )
            return response["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    async def add_memory(
        self, 
        memory_text: str, 
        memory_type: str = "observation", 
        significance: int = 3,
        emotional_valence: int = 0,
        emotional_intensity: Optional[int] = None,
        tags: List[str] = None,
        status: str = "active",
        confidence: float = 1.0
    ) -> Optional[int]:
        """
        Add a new memory for the NPC.
        
        Args:
            memory_text: The memory text to store
            memory_type: Type of memory (observation, reflection, etc.)
            significance: Importance of the memory (1-10)
            emotional_valence: Emotion direction (-10 to 10)
            emotional_intensity: Optional direct intensity value (0-100)
            tags: Optional tags for the memory
            status: Memory status ('active', 'summarized', 'archived')
            confidence: Confidence in this memory (0.0-1.0)
            
        Returns:
            ID of the created memory or None if failed
        """
        tags = tags or []
        
        # Auto-detect content tags
        content_tags = await self.analyze_memory_content(memory_text)
        tags.extend(content_tags)
        
        # If emotional_intensity not provided, calculate from valence or analyze text
        if emotional_intensity is None:
            if self.use_subsystems:
                try:
                    emotion_analysis = await self.emotional_memory.analyze_emotional_content(memory_text)
                    primary_emotion = emotion_analysis.get("primary_emotion", "neutral")
                    emotional_intensity = int(emotion_analysis.get("intensity", 0.5) * 100)
                except Exception as e:
                    logger.error(f"Error analyzing emotional content: {e}")
                    emotional_intensity = await self.calculate_emotional_intensity(memory_text, emotional_valence)
            else:
                emotional_intensity = await self.calculate_emotional_intensity(memory_text, emotional_valence)
        
        if self.use_subsystems:
            try:
                return await self._add_memory_with_subsystems(
                    memory_text, memory_type, significance, 
                    emotional_intensity, tags
                )
            except Exception as e:
                logger.error(f"Error adding memory with subsystems: {e}")
                # Fall back to direct DB access
                return await self._add_memory_with_db(
                    memory_text, memory_type, significance,
                    emotional_intensity, tags, status, confidence
                )
        else:
            return await self._add_memory_with_db(
                memory_text, memory_type, significance,
                emotional_intensity, tags, status, confidence
            )

    async def _track_operation_performance(self, operation_name: str, start_time: float) -> None:
        """Track performance of memory operations and log slow operations."""
        from utils.performance import STATS
        
        # Calculate elapsed time
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Record in stats
        STATS.record_memory_access_time(elapsed_ms)
        
        # Log slow operations
        if elapsed_ms > 100:  # Threshold for slow operations (100ms)
            logging.warning(f"Slow memory operation: {operation_name} took {elapsed_ms:.2f}ms for NPC {self.npc_id}")
    
    async def _add_memory_with_subsystems(
        self,
        memory_text: str,
        memory_type: str,
        significance: int,
        emotional_intensity: int,
        tags: List[str]
    ) -> int:
        """Add memory using memory subsystem managers."""
        try:
            # Try to add with emotional context if possible
            memory_result = await self.emotional_memory.add_emotional_memory(
                entity_type="npc",
                entity_id=self.npc_id,
                memory_text=memory_text,
                primary_emotion="unknown",  # We'd need to determine this
                emotion_intensity=emotional_intensity / 100.0,  # Convert to 0-1 scale
                secondary_emotions={},
                significance=significance,
                tags=tags
            )
            
            memory_id = memory_result["memory_id"]
        except Exception as e:
            logger.error(f"Error adding emotional memory: {e}")
            # Fall back to regular memory
            memory = Memory(
                text=memory_text,
                memory_type=memory_type,
                significance=significance,
                emotional_intensity=emotional_intensity,
                tags=tags,
                timestamp=datetime.now()
            )
            
            memory_id = await self.core_memory.add_memory(memory)
        
        # Auto-detect and apply schemas
        try:
            await self.schema_manager.apply_schema_to_memory(
                memory_id=memory_id,
                entity_type="npc",
                entity_id=self.npc_id,
                auto_detect=True
            )
        except Exception as e:
            logger.error(f"Error applying schemas to memory {memory_id}: {e}")
        
        # If significance is high, propagate the memory to connected NPCs
        if significance >= 4:
            try:
                await self._propagate_memory_subsystems(memory_text, tags, significance, emotional_intensity)
            except Exception as e:
                logger.error(f"Error propagating memory: {e}")
        
        return memory_id
    
    async def _add_memory_with_db(
        self,
        memory_text: str,
        memory_type: str,
        significance: int,
        emotional_intensity: int,
        tags: List[str],
        status: str,
        confidence: float
    ) -> Optional[int]:
        """Add memory using direct database access."""
        # Generate embedding for the memory text
        embedding_data = await self.generate_embedding(memory_text)
        
        conn = None
        try:
            # Get a connection from the pool or create a new one
            if self.db_pool:
                conn = await self.db_pool.acquire()
            else:
                conn = await asyncpg.connect(dsn=DB_DSN)
            
            # Insert row with the memory data
            memory_id = await conn.fetchval("""
                INSERT INTO NPCMemories (
                    npc_id, memory_text, memory_type, tags,
                    emotional_intensity, significance, embedding,
                    associated_entities, is_consolidated, status, confidence
                )
                VALUES (
                    $1, $2, $3, $4,
                    $5, $6, $7,
                    $8, FALSE, $9, $10
                )
                RETURNING id
            """,
                self.npc_id,
                memory_text,
                memory_type,
                tags,
                emotional_intensity,
                significance,
                embedding_data,
                json.dumps({}),
                status,
                confidence
            )
            
            # Apply personality-based bias to memory confidence
            await self.apply_npc_memory_bias(conn, memory_id)
            
            # If significance is high, propagate the memory to connected NPCs
            if significance >= 4:
                await self.propagate_memory(conn, memory_text, tags, significance, emotional_intensity)
            
            # Update reputation if memory involves player actions
            await self.update_reputation(conn, memory_text)
            
            # Update future outcome probabilities
            await self.update_future_outcomes(conn, memory_text)
            
            return memory_id
        except Exception as e:
            logger.error(f"Error adding memory with DB: {e}")
            return None
        finally:
            # Release the connection
            if conn:
                if self.db_pool:
                    await self.db_pool.release(conn)
                else:
                    await conn.close()
    
    async def analyze_memory_content(self, memory_text: str) -> List[str]:
        """
        Analyze memory text content to extract relevant tags.
        Enhanced with femdom-specific categories and themes.
        """
        tags = []
        lower_text = memory_text.lower()
        
        # Emotional content
        if any(word in lower_text for word in ["angry", "upset", "mad", "furious", "betrayed"]):
            tags.append("negative_emotion")
        if any(word in lower_text for word in ["happy", "pleased", "joy", "delighted", "thrilled"]):
            tags.append("positive_emotion")
        if any(word in lower_text for word in ["afraid", "scared", "fearful", "terrified"]):
            tags.append("fear")
        if any(word in lower_text for word in ["aroused", "excited", "turned on", "desire"]):
            tags.append("arousal")
        
        # Player-related
        if "player" in lower_text or "user" in lower_text or "chase" in lower_text:
            tags.append("player_related")
        
        # Rumor or secondhand
        if "heard" in lower_text or "told me" in lower_text or "said that" in lower_text:
            tags.append("rumor")
        
        # Social interactions
        if any(word in lower_text for word in ["helped", "assisted", "supported", "saved"]):
            tags.append("positive_interaction")
        if any(word in lower_text for word in ["betrayed", "attacked", "deceived", "tricked"]):
            tags.append("negative_interaction")
        
        # FEMDOM SPECIFIC TAGS
        
        # Dominance dynamics
        if any(word in lower_text for word in ["command", "ordered", "instructed", "demanded"]):
            tags.append("dominance_assertion")
        if any(word in lower_text for word in ["obey", "comply", "submit", "kneel", "bow"]):
            tags.append("submission_behavior")
        
        # Punishment/discipline
        if any(word in lower_text for word in ["punish", "discipline", "correct", "consequences"]):
            tags.append("discipline")
        if any(word in lower_text for word in ["spank", "whip", "paddle", "impact"]):
            tags.append("physical_discipline")
        
        # Humiliation
        if any(word in lower_text for word in ["humiliate", "embarrass", "shame", "mock"]):
            tags.append("humiliation")
        
        # Training/conditioning
        if any(word in lower_text for word in ["train", "condition", "learn", "lesson", "teach"]):
            tags.append("training")
        
        # Control
        if any(word in lower_text for word in ["control", "restrict", "limit", "permission"]):
            tags.append("control")
        
        # Worship/service
        if any(word in lower_text for word in ["worship", "serve", "service", "please", "satisfy"]):
            tags.append("service")
        
        # Devotion/loyalty
        if any(word in lower_text for word in ["devoted", "loyal", "faithful", "belonging"]):
            tags.append("devotion")
        
        # Resistance
        if any(word in lower_text for word in ["resist", "disobey", "refuse", "defy"]):
            tags.append("resistance")
        
        # Power exchange
        if any(word in lower_text for word in ["power", "exchange", "protocol", "dynamic", "role"]):
            tags.append("power_exchange")
        
        # Praise
        if any(word in lower_text for word in ["praise", "good", "well done", "proud"]):
            tags.append("praise")
        
        # Rituals
        if any(word in lower_text for word in ["ritual", "ceremony", "protocol", "procedure"]):
            tags.append("ritual")
        
        # Collaring and ownership
        if any(word in lower_text for word in ["collar", "own", "belong", "possess", "property"]):
            tags.append("ownership")
        
        # Psychological
        if any(word in lower_text for word in ["mind", "psyche", "thoughts", "mental", "mindset"]):
            tags.append("psychological")
        
        return tags
    
    async def calculate_emotional_intensity(self, memory_text: str, base_valence: float) -> float:
        """
        Calculate emotional intensity from text content and base valence.
        
        Args:
            memory_text: The memory text 
            base_valence: Base emotional valence (-10 to 10)
            
        Returns:
            Emotional intensity (0-100)
        """
        # Convert valence to base intensity
        intensity = (base_valence + 10) * 5  # Map [-10,10] to [0,100]
        
        # Emotional words and their intensity boost values
        emotion_words = {
            "furious": 20, "ecstatic": 20, "devastated": 20, "thrilled": 20,
            "angry": 15, "delighted": 15, "sad": 15, "happy": 15,
            "annoyed": 10, "pleased": 10, "upset": 10, "glad": 10,
            "concerned": 5, "fine": 5, "worried": 5, "okay": 5
        }
        
        # Scan text for emotional keywords
        lower_text = memory_text.lower()
        for word, boost in emotion_words.items():
            if word in lower_text:
                intensity += boost
                break
        
        return float(min(100, max(0, intensity)))
    
    async def apply_npc_memory_bias(self, conn, memory_id: int):
        """
        Adjust memory confidence based on NPC personality type.
        
        Args:
            conn: Database connection
            memory_id: ID of the memory to adjust
        """
        personality_factors = {
            "gullible": 1.2,   # More likely to believe (higher confidence)
            "skeptical": 0.8,  # Less likely to believe (lower confidence)
            "paranoid": 1.5,   # Much more likely to believe negative things
            "neutral": 1.0     # No bias
        }
        
        factor = personality_factors.get(self.npc_personality, 1.0)
        
        try:
            # For paranoid NPCs, check if the memory has negative connotations
            if self.npc_personality == "paranoid":
                memory_text = await conn.fetchval(
                    "SELECT memory_text FROM NPCMemories WHERE id = $1", 
                    memory_id
                )
                if memory_text:
                    lower_text = memory_text.lower()
                    if any(word in lower_text for word in ["betray", "trick", "lie", "deceive", "attack"]):
                        factor = 1.5  # Higher confidence in negative memories
                    else:
                        factor = 0.9  # Lower confidence in positive/neutral memories
            
            # Apply the confidence adjustment
            await conn.execute("""
                UPDATE NPCMemories
                SET confidence = LEAST(confidence * $1, 1.0)
                WHERE id = $2
            """, factor, memory_id)
        except Exception as e:
            logger.error(f"Error applying personality bias: {e}")
    
    async def propagate_memory(self, conn, memory_text: str, tags: List[str], significance: int, emotional_intensity: float):
        """
        Propagate important memories to related NPCs as secondhand information.
        
        Args:
            conn: Database connection
            memory_text: The memory text
            tags: Tags for the memory
            significance: Importance of the memory
            emotional_intensity: Emotional intensity of the memory
        """
        try:
            # Find NPCs connected to this NPC
            rows = await conn.fetch("""
                SELECT entity2_id
                FROM SocialLinks
                WHERE user_id = $1
                  AND conversation_id = $2
                  AND entity1_type = 'npc'
                  AND entity1_id = $3
                  AND entity2_type = 'npc'
            """, self.user_id, self.conversation_id, self.npc_id)
            
            related_npcs = [r["entity2_id"] for r in rows]
            
            # Get this NPC's name
            row = await conn.fetchrow("""
                SELECT npc_name
                FROM NPCStats
                WHERE user_id = $1 
                  AND conversation_id = $2
                  AND npc_id = $3
            """, self.user_id, self.conversation_id, self.npc_id)
            
            npc_name = row["npc_name"] if row else f"NPC_{self.npc_id}"
            
            # Create secondhand memories for each related NPC
            for related_id in related_npcs:
                # Distort the message slightly based on relationship
                distorted_text = self.distort_text(memory_text, severity=0.3)
                secondhand_text = f"I heard that {npc_name} {distorted_text}"
                
                # Lower significance and intensity for secondhand information
                secondhand_significance = max(1, significance - 2)
                secondhand_intensity = max(0, emotional_intensity - 20)
                
                # Add the secondhand memory
                await conn.execute("""
                    INSERT INTO NPCMemories (
                        npc_id, memory_text, memory_type, tags,
                        emotional_intensity, significance, status,
                        confidence, is_consolidated
                    )
                    VALUES (
                        $1, $2, 'secondhand', $3,
                        $4, $5, 'active',
                        0.7, FALSE
                    )
                """, 
                    related_id, 
                    secondhand_text, 
                    tags + ["secondhand", "rumor"],
                    secondhand_intensity,
                    secondhand_significance
                )
            
            logger.debug(f"Propagated memory to {len(related_npcs)} related NPCs")
        except Exception as e:
            logger.error(f"Error propagating memory: {e}")
    
    async def categorize_memories_by_significance(self, retention_threshold=5):
        """
        Separate core memories from passive ones based on significance, 
        emotional impact, and frequency of recall (implements TO_DO.TXT strategies).
        """
        active_count = 0
        archived_count = 0
        
        with get_db_connection() as conn, conn.cursor() as cursor:
            # Get all active memories for this NPC
            cursor.execute("""
                SELECT id, significance, emotional_intensity, times_recalled, 
                       memory_text, tags
                FROM NPCMemories
                WHERE npc_id = %s AND status = 'active'
            """, (self.npc_id,))
            
            memories = cursor.fetchall()
            for memory in memories:
                memory_id, significance, emotional_intensity, recall_count, memory_text, tags = memory
                
                # Calculate retention score - memories recalled more often, with higher
                # emotional/significance values, or tagged as emotional should stay active
                retention_score = significance + (emotional_intensity / 20) + (recall_count * 0.5)
                
                # Intelligence modifier - smarter NPCs retain more
                if hasattr(self, 'npc_intelligence') and self.npc_intelligence > 1.0:
                    retention_score *= self.npc_intelligence
                
                # Emotional tagging bonus - emotional memories are "stickier"
                if any(tag in tags for tag in ["emotional", "traumatic", "significant"]):
                    retention_score += 2
                    
                # Context relevance - keep memories relevant to current environment
                current_location = self._get_npc_current_location()
                if current_location and current_location.lower() in memory_text.lower():
                    retention_score += 1.5
                
                # Check if memory should be kept active or archived
                if retention_score < retention_threshold:
                    # Move to passive storage but remember it can be triggered by context
                    cursor.execute("""
                        UPDATE NPCMemories
                        SET status = 'passive', context_triggers = %s
                        WHERE id = %s
                    """, (json.dumps(self._extract_context_triggers(memory_text)), memory_id))
                    archived_count += 1
                else:
                    active_count += 1
        
        return {"active": active_count, "archived": archived_count}
    
    def _extract_context_triggers(self, memory_text):
        """Extract keywords that could trigger this passive memory when encountered."""
        # Simple extraction of potentially important context triggers
        triggers = []
        
        # Extract proper nouns and locations
        for word in memory_text.split():
            if word[0].isupper() and len(word) > 3 and word.lower() not in COMMON_WORDS:
                triggers.append(word.lower())
        
        # Extract phrases related to emotions
        for phrase in ["angry with", "afraid of", "happy about", "saddened by"]:
            if phrase in memory_text.lower():
                index = memory_text.lower().find(phrase)
                # Try to get what comes after the phrase
                remaining = memory_text[index + len(phrase):].split()
                if remaining and len(remaining) > 0:
                    triggers.append(phrase + " " + remaining[0])
        
        return triggers
    
    def _get_npc_current_location(self):
        """Get the NPC's current location for context relevance checks."""
        with get_db_connection() as conn, conn.cursor() as cursor:
            cursor.execute("""
                SELECT current_location FROM NPCStats
                WHERE npc_id = %s AND user_id = %s AND conversation_id = %s
            """, (self.npc_id, self.user_id, self.conversation_id))
            row = cursor.fetchone()
            return row[0] if row else None
    
    async def _propagate_memory_subsystems(self, memory_text: str, tags: List[str], significance: int, emotional_intensity: float):
        """
        Propagate memory using memory subsystems instead of direct DB access.
        
        Args:
            memory_text: The memory text
            tags: Tags for the memory
            significance: Importance of the memory
            emotional_intensity: Emotional intensity of the memory
        """
        # This implementation would use the subsystem APIs instead of direct DB queries
        # For demonstration, we'll leave this as a placeholder
        logger.info(f"Would propagate memory with significance {significance} using subsystems")
        pass
    
    def distort_text(self, original_text: str, severity=0.3) -> str:
        """
        Word-level partial rewrite for rumor distortion.
        
        Args:
            original_text: The original text to distort
            severity: How much to distort the text (0.0-1.0)
            
        Returns:
            Distorted version of the text
        """
        synonyms_map = {
            "attacked": ["assaulted", "ambushed", "jumped"],
            "betrayed": ["backstabbed", "double-crossed", "deceived"],
            "stole": ["looted", "swiped", "snatched", "took"],
            "helped": ["assisted", "saved", "aided", "supported"],
            "rescued": ["freed", "saved", "liberated", "pulled out"],
            "said": ["mentioned", "claimed", "stated", "told me"],
            "saw": ["noticed", "spotted", "observed", "glimpsed"],
            "went": ["traveled", "journeyed", "ventured", "headed"],
            "found": ["discovered", "located", "uncovered", "came across"]
        }
        
        words = original_text.split()
        for i in range(len(words)):
            # Chance to modify a word based on severity
            if random.random() < severity:
                word_lower = words[i].lower()
                # Replace with synonym if available
                if word_lower in synonyms_map:
                    words[i] = random.choice(synonyms_map[word_lower])
                # Small chance to delete a word
                elif random.random() < 0.2:
                    words[i] = ""
        
        # Reconstruct text, removing empty strings
        return " ".join([w for w in words if w])
    
    async def update_reputation(self, conn, memory_text: str):
        """
        Update player reputation based on memory content.
        
        Args:
            conn: Database connection
            memory_text: Memory text to analyze
        """
        try:
            lower_text = memory_text.lower()
            
            # Check for negative/positive interaction terms
            if any(word in lower_text for word in ["betray", "attack", "deceive", "trick"]):
                await conn.execute("""
                    INSERT INTO PlayerReputation (user_id, npc_id, reputation_score)
                    VALUES ($1, $2, -5)
                    ON CONFLICT (user_id, npc_id) 
                    DO UPDATE SET reputation_score = 
                        GREATEST(PlayerReputation.reputation_score - 5, -100)
                """, self.user_id, self.npc_id)
            elif any(word in lower_text for word in ["help", "save", "assist", "support"]):
                await conn.execute("""
                    INSERT INTO PlayerReputation (user_id, npc_id, reputation_score)
                    VALUES ($1, $2, 5)
                    ON CONFLICT (user_id, npc_id) 
                    DO UPDATE SET reputation_score = 
                        LEAST(PlayerReputation.reputation_score + 5, 100)
                """, self.user_id, self.npc_id)
        except Exception as e:
            logger.error(f"Error updating reputation: {e}")
    
    async def update_future_outcomes(self, conn, memory_text: str):
        """
        Update future outcome probabilities based on memory.
        
        Args:
            conn: Database connection
            memory_text: Memory text to analyze
        """
        try:
            lower_text = memory_text.lower()
            
            # Update betrayal-related outcomes
            if any(word in lower_text for word in ["betray", "deceive", "trick"]):
                await conn.execute("""
                    INSERT INTO FutureOutcomes (npc_id, event, probability)
                    VALUES ($1, 'NPC seeks revenge', 0.5)
                    ON CONFLICT (npc_id, event) DO NOTHING
                """, self.npc_id)
                
                await conn.execute("""
                    UPDATE FutureOutcomes
                    SET probability = LEAST(probability + 0.1, 1.0)
                    WHERE npc_id = $1 AND event = 'NPC seeks revenge'
                """, self.npc_id)
            
            # Update help-related outcomes
            elif any(word in lower_text for word in ["help", "save", "assist"]):
                await conn.execute("""
                    INSERT INTO FutureOutcomes (npc_id, event, probability)
                    VALUES ($1, 'NPC allies with player', 0.5)
                    ON CONFLICT (npc_id, event) DO NOTHING
                """, self.npc_id)
                
                await conn.execute("""
                    UPDATE FutureOutcomes
                    SET probability = LEAST(probability + 0.1, 1.0)
                    WHERE npc_id = $1 AND event = 'NPC allies with player'
                """, self.npc_id)
        except Exception as e:
            logger.error(f"Error updating future outcomes: {e}")
    
    async def retrieve_relevant_memories(
        self, 
        context: Dict[str, Any], 
        limit: int = 5,
        memory_types: List[str] = None,
        include_archived: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories relevant to the current context.
        
        Args:
            context: Context information (can include text, environment, etc.)
            limit: Maximum number of memories to retrieve
            memory_types: Optional filter for memory types
            include_archived: Whether to include archived memories
            
        Returns:
            List of relevant memories
        """
        memory_types = memory_types or ["observation", "reflection", "semantic", "secondhand"]
        
        if self.use_subsystems:
            try:
                return await self._retrieve_memories_subsystems(context, limit, memory_types)
            except Exception as e:
                logger.error(f"Error retrieving memories with subsystems: {e}")
                # Fall back to direct DB access
                return await self._retrieve_memories_db(context, limit, memory_types, include_archived)
        else:
            return await self._retrieve_memories_db(context, limit, memory_types, include_archived)
    
    async def _retrieve_memories_subsystems(
        self, 
        context: Dict[str, Any], 
        limit: int,
        memory_types: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories using memory subsystem managers.
        
        Args:
            context: Context information
            limit: Maximum memories to retrieve
            memory_types: Types of memories to include
            
        Returns:
            List of relevant memories
        """
        # Get current emotional state to influence recall
        emotional_state = await self.emotional_memory.get_entity_emotional_state(
            entity_type="npc",
            entity_id=self.npc_id
        )
        
        # Check if there are any traumatic triggers in the context
        if isinstance(context, dict) and "text" in context:
            trigger_result = await self.emotional_memory.process_traumatic_triggers(
                entity_type="npc",
                entity_id=self.npc_id,
                text=context["text"]
            )
            
            if trigger_result.get("triggered", False):
                # If trauma is triggered, prioritize these memories
                triggered_memories = trigger_result.get("triggered_memories", [])
                if triggered_memories:
                    return triggered_memories
        
        # Use emotional state for recall if applicable
        current_emotion = emotional_state.get("current_emotion", {})
        if current_emotion.get("intensity", 0) > 0.6:
            # Strong emotions influence memory recall
            mood_memories = await self.emotional_memory.retrieve_mood_congruent_memories(
                entity_type="npc",
                entity_id=self.npc_id,
                current_mood=current_emotion,
                limit=limit
            )
            
            if mood_memories:
                return mood_memories
        
        # Extract query text from context
        query = ""
        if isinstance(context, str):
            query = context
        elif isinstance(context, dict):
            query = context.get("text", "")
            if not query:
                # Try to extract from other context elements
                context_elements = []
                for key, value in context.items():
                    if isinstance(value, str):
                        context_elements.append(value)
                query = " ".join(context_elements)
        
        # Standard memory retrieval
        memories = await self.core_memory.retrieve_memories(
            query=query,
            memory_types=memory_types,
            limit=limit
        )
        
        # Format and return memories
        formatted_memories = []
        current_emotion_data = current_emotion or {}
        
        for memory in memories:
            # Check if this memory should be reconsolidated
            if memory.times_recalled > 0 and random.random() < 0.2:  # 20% chance
                try:
                    recon_manager = ReconsolidationManager(self.user_id, self.conversation_id)
                    await recon_manager.reconsolidate_memory(
                        memory_id=memory.id,
                        entity_type="npc",
                        entity_id=self.npc_id,
                        emotional_context=current_emotion_data,
                        recall_context=str(context),
                        alteration_strength=0.05  # Subtle alterations
                    )
                except Exception as e:
                    logger.error(f"Error reconsolidating memory: {e}")
            
            # Add schema interpretation if available
            interpretation = None
            try:
                interpretation_result = await self.schema_manager.interpret_memory_with_schemas(
                    memory_id=memory.id,
                    entity_type="npc",
                    entity_id=self.npc_id
                )
                
                if "interpretation" in interpretation_result:
                    interpretation = interpretation_result["interpretation"]
            except Exception as e:
                logger.error(f"Error interpreting memory with schemas: {e}")
            
            formatted_memories.append({
                "id": memory.id,
                "text": memory.text,
                "type": memory.memory_type,
                "significance": memory.significance,
                "emotional_intensity": memory.emotional_intensity,
                "schema_interpretation": interpretation,
                "timestamp": memory.timestamp.isoformat() if memory.timestamp else None
            })
        
        # Optionally generate a flashback (10% chance)
        if random.random() < 0.1:
            try:
                flashback = await self.flashback_manager.generate_flashback(
                    entity_type="npc",
                    entity_id=self.npc_id,
                    current_context=query
                )
                
                if flashback:
                    # Insert the flashback at a random position
                    insert_pos = min(len(formatted_memories), random.randint(0, 2))
                    formatted_memories.insert(insert_pos, {
                        "id": f"flashback_{datetime.now().timestamp()}",
                        "text": flashback["text"],
                        "type": "flashback",
                        "significance": 4,  # High significance
                        "flashback": True
                    })
            except Exception as e:
                logger.error(f"Error generating flashback: {e}")
        
        return formatted_memories
    
    async def _retrieve_memories_db(
        self, 
        context: Any, 
        limit: int,
        memory_types: List[str],
        include_archived: bool
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories using direct database access.
        
        Args:
            context: Query text or context object
            limit: Maximum memories to retrieve
            memory_types: Types of memories to include
            include_archived: Whether to include archived memories
            
        Returns:
            List of relevant memories
        """
        # Extract query text
        query_text = ""
        if isinstance(context, str):
            query_text = context
        elif isinstance(context, dict):
            query_text = context.get("text", context.get("description", ""))
        
        conn = None
        memories = []
        
        try:
            # Get a connection from the pool or create a new one
            if self.db_pool:
                conn = await self.db_pool.acquire()
            else:
                conn = await asyncpg.connect(dsn=DB_DSN)
            
            # Determine status filter
            status_filter = "'active','summarized'"
            if include_archived:
                status_filter += ",'archived'"
            
            try:
                # Generate embedding for the query
                query_vector = await self.generate_embedding(query_text)
                
                if query_vector:
                    # Use vector similarity search if embedding was generated
                    rows = await conn.fetch(f"""
                        SELECT id, memory_text, memory_type, tags,
                               emotional_intensity, significance,
                               times_recalled, timestamp, status, confidence
                        FROM NPCMemories
                        WHERE npc_id = $1
                          AND status IN ({status_filter})
                          AND memory_type = ANY($3)
                        ORDER BY embedding <-> $2
                        LIMIT $4
                    """, self.npc_id, query_vector, memory_types, limit)
                else:
                    # Fall back to keyword search
                    raise Exception("No embedding generated, falling back to keywords")
            
            except Exception as e:
                logger.error(f"Vector search failed: {e}")
                # Fallback: keyword search
                words = query_text.lower().split()
                if words:
                    conditions = " OR ".join(
                        [f"LOWER(memory_text) LIKE '%'||${i+2}||'%'" for i in range(len(words))]
                    )
                    # Parameters: npc_id, words..., memory_types, limit
                    params = [self.npc_id] + words + [memory_types, limit]
                    q = f"""
                        SELECT id, memory_text, memory_type, tags,
                               emotional_intensity, significance,
                               times_recalled, timestamp, status, confidence
                        FROM NPCMemories
                        WHERE npc_id = $1
                          AND status IN ({status_filter})
                          AND memory_type = ANY(${len(params) - 1})
                          AND ({conditions})
                        ORDER BY significance DESC, timestamp DESC
                        LIMIT ${len(params)}
                    """
                    rows = await conn.fetch(q, *params)
                else:
                    # No query words, just get recent memories
                    rows = await conn.fetch(f"""
                        SELECT id, memory_text, memory_type, tags,
                               emotional_intensity, significance,
                               times_recalled, timestamp, status, confidence
                        FROM NPCMemories
                        WHERE npc_id = $1
                          AND status IN ({status_filter})
                          AND memory_type = ANY($2)
                        ORDER BY timestamp DESC
                        LIMIT $3
                    """, self.npc_id, memory_types, limit)
            
            # Process the results
            for row in rows:
                memories.append({
                    "id": row["id"],
                    "text": row["memory_text"],
                    "type": row["memory_type"],
                    "tags": row["tags"] or [],
                    "emotional_intensity": row["emotional_intensity"],
                    "significance": row["significance"],
                    "times_recalled": row["times_recalled"],
                    "timestamp": row["timestamp"].isoformat() if isinstance(row["timestamp"], datetime) else row["timestamp"],
                    "status": row["status"],
                    "confidence": row["confidence"],
                    "relevance_score": 0.0
                })
            
            # Apply biases to the memories
            memories = await self.apply_recency_bias(memories)
            memories = await self.apply_emotional_bias(memories)
            memories = await self.apply_personality_bias(memories)
            
            # Sort by relevance score
            memories.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            # Update retrieval stats
            await self.update_memory_retrieval_stats(conn, [m["id"] for m in memories])
            
            return memories[:limit]  # Ensure we don't return more than requested
        
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return []
        
        finally:
            # Release the connection
            if conn:
                if self.db_pool:
                    await self.db_pool.release(conn)
                else:
                    await conn.close()
    
    async def apply_recency_bias(self, memories: List[dict]) -> List[dict]:
        """
        Boost recent memories' relevance_score.
        
        Args:
            memories: List of memories to adjust
            
        Returns:
            Adjusted memories
        """
        now = datetime.now()
        for mem in memories:
            ts = mem.get("timestamp")
            
            # Parse timestamp if needed
            if isinstance(ts, str):
                try:
                    dt = datetime.fromisoformat(ts)
                    days_ago = (now - dt).days
                except ValueError:
                    days_ago = 30  # Default if parsing fails
            elif isinstance(ts, datetime):
                days_ago = (now - ts).days
            else:
                days_ago = 30  # Default if no timestamp
            
            # Calculate recency factor (1.0 for very recent, decreasing over time)
            recency_factor = max(0, 30 - days_ago) / 30.0
            
            # Add to relevance score
            mem["relevance_score"] += recency_factor * 5.0
        
        return memories
    
    async def apply_emotional_bias(self, memories: List[dict]) -> List[dict]:
        """
        Adjust memory relevance based on emotional intensity and significance.
        
        Args:
            memories: List of memories to adjust
            
        Returns:
            Adjusted memories
        """
        for mem in memories:
            # Normalize values to 0-1 range
            emotional_intensity = mem.get("emotional_intensity", 0) / 100.0
            significance = mem.get("significance", 0) / 10.0
            
            # Add to relevance score
            mem["relevance_score"] += (emotional_intensity * 3.0 + significance * 2.0)
        
        return memories
    
    async def apply_personality_bias(self, memories: List[dict]) -> List[dict]:
        """
        Apply personality-specific biases to memory relevance.
        
        Args:
            memories: List of memories to adjust
            
        Returns:
            Adjusted memories
        """
        for mem in memories:
            text = mem.get("text", "").lower()
            tags = mem.get("tags", [])
            
            # Personality-specific biases
            if self.npc_personality == "paranoid":
                # Paranoid NPCs prioritize threatening or negative memories
                if any(word in text for word in ["threat", "danger", "betray", "attack"]):
                    mem["relevance_score"] += 3.0
                if "negative_emotion" in tags or "negative_interaction" in tags:
                    mem["relevance_score"] += 2.0
            
            elif self.npc_personality == "gullible":
                # Gullible NPCs prioritize secondhand information
                if "rumor" in tags or "secondhand" in tags:
                    mem["relevance_score"] += 2.0
            
            elif self.npc_personality == "skeptical":
                # Skeptical NPCs deprioritize rumors and secondhand info
                if "rumor" in tags or "secondhand" in tags:
                    mem["relevance_score"] -= 1.5
            
            # Adjust by confidence
            confidence = mem.get("confidence", 1.0)
            mem["relevance_score"] *= confidence
        
        return memories
    
    async def update_memory_retrieval_stats(self, conn, memory_ids: List[int]):
        """
        Update statistics for retrieved memories.
        
        Args:
            conn: Database connection
            memory_ids: IDs of memories that were retrieved
        """
        if not memory_ids:
            return
        
        try:
            await conn.execute("""
                UPDATE NPCMemories
                SET times_recalled = times_recalled + 1,
                    last_recalled = CURRENT_TIMESTAMP
                WHERE id = ANY($1)
            """, memory_ids)
        except Exception as e:
            logger.error(f"Error updating memory retrieval stats: {e}")
    
    async def update_emotional_state(self, 
                                   primary_emotion: str, 
                                   intensity: float, 
                                   trigger: str = None) -> Dict[str, Any]:
        """
        Update the NPC's emotional state.
        
        Args:
            primary_emotion: The primary emotion
            intensity: Intensity of the emotion (0.0-1.0)
            trigger: What triggered this emotion
            
        Returns:
            Updated emotional state
        """
        if self.use_subsystems:
            # Create emotional state update
            current_emotion = {
                "primary_emotion": primary_emotion,
                "intensity": intensity,
                "secondary_emotions": {},
                "trigger": trigger
            }
            
            # Record the update
            return await self.emotional_memory.update_entity_emotional_state(
                entity_type="npc",
                entity_id=self.npc_id,
                current_emotion=current_emotion
            )
        else:
            # Direct DB implementation
            conn = None
            try:
                if self.db_pool:
                    conn = await self.db_pool.acquire()
                else:
                    conn = await asyncpg.connect(dsn=DB_DSN)
                
                # Update or insert emotional state
                await conn.execute("""
                    INSERT INTO NPCEmotionalStates (
                        npc_id, primary_emotion, intensity, trigger, timestamp
                    )
                    VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP)
                    ON CONFLICT (npc_id) DO UPDATE SET
                        primary_emotion = $2,
                        intensity = $3,
                        trigger = $4,
                        timestamp = CURRENT_TIMESTAMP
                """, self.npc_id, primary_emotion, intensity, trigger)
                
                return {
                    "npc_id": self.npc_id,
                    "current_emotion": {
                        "primary_emotion": primary_emotion,
                        "intensity": intensity,
                        "trigger": trigger
                    },
                    "timestamp": datetime.now().isoformat()
                }
            
            except Exception as e:
                logger.error(f"Error updating emotional state: {e}")
                return {"error": str(e)}
            
            finally:
                if conn:
                    if self.db_pool:
                        await self.db_pool.release(conn)
                    else:
                        await conn.close()
    
    async def get_emotional_state(self) -> Dict[str, Any]:
        """
        Get the NPC's current emotional state.
        
        Returns:
            Current emotional state
        """
        if self.use_subsystems:
            return await self.emotional_memory.get_entity_emotional_state(
                entity_type="npc",
                entity_id=self.npc_id
            )
        else:
            # Direct DB implementation
            conn = None
            try:
                if self.db_pool:
                    conn = await self.db_pool.acquire()
                else:
                    conn = await asyncpg.connect(dsn=DB_DSN)
                
                row = await conn.fetchrow("""
                    SELECT primary_emotion, intensity, trigger, timestamp
                    FROM NPCEmotionalStates
                    WHERE npc_id = $1
                """, self.npc_id)
                
                if row:
                    return {
                        "npc_id": self.npc_id,
                        "current_emotion": {
                            "primary_emotion": row["primary_emotion"],
                            "intensity": row["intensity"],
                            "trigger": row["trigger"]
                        },
                        "timestamp": row["timestamp"].isoformat()
                    }
                else:
                    return {
                        "npc_id": self.npc_id,
                        "current_emotion": {
                            "primary_emotion": "neutral",
                            "intensity": 0.0,
                            "trigger": None
                        }
                    }
            
            except Exception as e:
                logger.error(f"Error getting emotional state: {e}")
                return {"error": str(e)}
            
            finally:
                if conn:
                    if self.db_pool:
                        await self.db_pool.release(conn)
                    else:
                        await conn.close()
    
    async def generate_mask_slippage(self, 
                                   trigger: str = None, 
                                   severity: int = None) -> Dict[str, Any]:
        """
        Generate a mask slippage event where the NPC's true nature shows through.
        
        Args:
            trigger: What triggered the mask slippage
            severity: How severe the slippage is (1-5)
            
        Returns:
            Mask slippage details
        """
        if self.use_subsystems:
            return await self.mask_manager.generate_mask_slippage(
                npc_id=self.npc_id,
                trigger=trigger,
                severity=severity
            )
        else:
            # Simple direct implementation
            conn = None
            try:
                if self.db_pool:
                    conn = await self.db_pool.acquire()
                else:
                    conn = await asyncpg.connect(dsn=DB_DSN)
                
                # Get NPC mask info
                row = await conn.fetchrow("""
                    SELECT true_nature, presented_traits
                    FROM NPCMasks
                    WHERE npc_id = $1
                """, self.npc_id)
                
                if not row:
                    return {"error": "NPC has no mask defined"}
                
                true_nature = row["true_nature"]
                severity = severity or random.randint(1, 5)
                
                slippage = {
                    "npc_id": self.npc_id,
                    "severity": severity,
                    "trigger": trigger,
                    "true_nature_glimpse": f"Brief glimpse of {true_nature} (severity {severity}/5)",
                    "timestamp": datetime.now().isoformat()
                }
                
                # Log the slippage
                await conn.execute("""
                    INSERT INTO NPCMaskSlippages (
                        npc_id, severity, trigger, timestamp
                    ) VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
                """, self.npc_id, severity, trigger)
                
                return slippage
            
            except Exception as e:
                logger.error(f"Error generating mask slippage: {e}")
                return {"error": str(e)}
            
            finally:
                if conn:
                    if self.db_pool:
                        await self.db_pool.release(conn)
                    else:
                        await conn.close()
    
    async def get_npc_mask(self) -> Dict[str, Any]:
        """
        Get information about the NPC's mask.
        
        Returns:
            Mask information (integrity, presented traits, etc.)
        """
        if self.use_subsystems:
            return await self.mask_manager.get_npc_mask(self.npc_id)
        else:
            # Direct DB implementation
            conn = None
            try:
                if self.db_pool:
                    conn = await self.db_pool.acquire()
                else:
                    conn = await asyncpg.connect(dsn=DB_DSN)
                
                row = await conn.fetchrow("""
                    SELECT true_nature, presented_traits, integrity
                    FROM NPCMasks
                    WHERE npc_id = $1
                """, self.npc_id)
                
                if row:
                    # Get recent slippages
                    slippages = await conn.fetch("""
                        SELECT severity, trigger, timestamp
                        FROM NPCMaskSlippages
                        WHERE npc_id = $1
                        ORDER BY timestamp DESC
                        LIMIT 3
                    """, self.npc_id)
                    
                    recent_slippages = [{
                        "severity": s["severity"],
                        "trigger": s["trigger"],
                        "timestamp": s["timestamp"].isoformat()
                    } for s in slippages]
                    
                    return {
                        "npc_id": self.npc_id,
                        "true_nature": row["true_nature"],
                        "presented_traits": row["presented_traits"],
                        "integrity": row["integrity"],
                        "recent_slippages": recent_slippages
                    }
                else:
                    return {"error": "NPC has no mask defined"}
            
            except Exception as e:
                logger.error(f"Error getting NPC mask: {e}")
                return {"error": str(e)}
            
            finally:
                if conn:
                    if self.db_pool:
                        await self.db_pool.release(conn)
                    else:
                        await conn.close()
    
    async def prune_old_memories(self,
                              age_days: int = 14,
                              significance_threshold: int = 3,
                              intensity_threshold: int = 15):
        """
        Prune or archive old memories based on their significance and emotional intensity.
        
        Args:
            age_days: Age threshold in days
            significance_threshold: Minimum significance to retain
            intensity_threshold: Minimum emotional intensity to retain
        """
        conn = None
        try:
            if self.db_pool:
                conn = await self.db_pool.acquire()
            else:
                conn = await asyncpg.connect(dsn=DB_DSN)
            
            cutoff = datetime.now() - timedelta(days=age_days)
            
            # Delete truly trivial memories
            del_result = await conn.execute("""
                DELETE FROM NPCMemories
                WHERE npc_id = $1
                  AND timestamp < $2
                  AND significance < $3
                  AND emotional_intensity < $4
            """, self.npc_id, cutoff, significance_threshold, intensity_threshold)
            
            logger.info(f"Pruned old memories: {del_result}")
            
            # Mark older memories as summarized
            upd_result = await conn.execute("""
                UPDATE NPCMemories
                SET status = 'summarized'
                WHERE npc_id = $1
                  AND timestamp < $2
                  AND status = 'active'
            """, self.npc_id, cutoff)
            
            logger.info(f"Updated memory status to summarized: {upd_result}")
        
        except Exception as e:
            logger.error(f"Error pruning old memories: {e}")
        
        finally:
            if conn:
                if self.db_pool:
                    await self.db_pool.release(conn)
                else:
                    await conn.close()
    
    async def apply_memory_decay(self, age_days: int = 30, decay_rate: float = 0.2):
        """
        Apply memory decay to older memories to simulate forgetting.
        
        Args:
            age_days: Age threshold in days
            decay_rate: Rate of decay (0.0-1.0)
        """
        conn = None
        try:
            if self.db_pool:
                conn = await self.db_pool.acquire()
            else:
                conn = await asyncpg.connect(dsn=DB_DSN)
            
            cutoff = datetime.now() - timedelta(days=age_days)
            
            # Get older memories
            rows = await conn.fetch("""
                SELECT id, emotional_intensity, significance, times_recalled
                FROM NPCMemories
                WHERE npc_id = $1
                  AND timestamp < $2
                  AND status IN ('active', 'summarized')
            """, self.npc_id, cutoff)
            
            # Apply intelligence-based decay modifier
            intelligence_modifier = 1.0
            if self.npc_intelligence > 1.0:
                # Higher intelligence = slower decay
                intelligence_modifier = 1.0 / self.npc_intelligence
            elif self.npc_intelligence < 1.0:
                # Lower intelligence = faster decay
                intelligence_modifier = 2.0 - self.npc_intelligence
            
            adjusted_decay_rate = decay_rate * intelligence_modifier
            
            # Process each memory
            for row in rows:
                mem_id = row["id"]
                intensity = row["emotional_intensity"]
                significance = row["significance"]
                recalled = row["times_recalled"]
                
                # Memories recalled more often decay less
                recall_factor = min(1.0, recalled / 10.0)
                decay_factor = adjusted_decay_rate * (1.0 - recall_factor)
                
                # Calculate new values
                new_intensity = max(0, int(intensity * (1.0 - decay_factor)))
                new_significance = max(1, int(significance * (1.0 - decay_factor)))
                
                # Update the memory
                await conn.execute("""
                    UPDATE NPCMemories
                    SET emotional_intensity = $1,
                        significance = $2
                    WHERE id = $3
                """, new_intensity, new_significance, mem_id)
            
            logger.info(f"Applied memory decay to {len(rows)} memories")
        
        except Exception as e:
            logger.error(f"Error applying memory decay: {e}")
        
        finally:
            if conn:
                if self.db_pool:
                    await self.db_pool.release(conn)
                else:
                    await conn.close()
    
    async def summarize_repetitive_memories(self, lookback_days: int = 7, min_count: int = 3):
        """
        Summarize repetitive memories into a consolidated memory.
        
        Args:
            lookback_days: Days to look back for repetitive memories
            min_count: Minimum number of repetitions to trigger summarization
        """
        conn = None
        try:
            if self.db_pool:
                conn = await self.db_pool.acquire()
            else:
                conn = await asyncpg.connect(dsn=DB_DSN)
            
            # Find repetitive memories
            rows = await conn.fetch("""
                SELECT memory_text, COUNT(*) AS cnt, array_agg(id) AS mem_ids
                FROM NPCMemories
                WHERE npc_id = $1
                  AND timestamp > (NOW() - $2::INTERVAL)
                  AND status = 'active'
                  AND is_consolidated = FALSE
                GROUP BY memory_text
                HAVING COUNT(*) >= $3
            """, self.npc_id, f"{lookback_days} days", min_count)
            
            for row in rows:
                text = row["memory_text"]
                count = row["cnt"]
                mem_ids = row["mem_ids"]
                
                if not mem_ids or count < min_count:
                    continue
                
                # Create a summary memory
                summary_text = f"I recall {count} similar moments: '{text}' repeated multiple times."
                
                # Insert the summary
                new_id = await conn.fetchval("""
                    INSERT INTO NPCMemories (
                        npc_id, memory_text, memory_type, 
                        emotional_intensity, significance, status,
                        is_consolidated
                    )
                    VALUES ($1, $2, 'consolidated', 10, 5, 'summarized', FALSE)
                    RETURNING id
                """, self.npc_id, summary_text)
                
                # Mark original memories as consolidated
                await conn.execute("""
                    UPDATE NPCMemories
                    SET is_consolidated = TRUE, status = 'summarized'
                    WHERE id = ANY($1)
                """, mem_ids)
                
                logger.info(f"Summarized {count} memories into memory {new_id}")
            
            logger.info(f"Summarization complete for NPC {self.npc_id}")
        
        except Exception as e:
            logger.error(f"Error summarizing repetitive memories: {e}")
        
        finally:
            if conn:
                if self.db_pool:
                    await self.db_pool.release(conn)
                else:
                    await conn.close()
    
    async def archive_stale_memories(self, older_than_days: int = 60, max_significance: int = 4):
        """
        Archive old memories with low significance.
        
        Args:
            older_than_days: Age threshold in days
            max_significance: Maximum significance to archive
        """
        conn = None
        try:
            if self.db_pool:
                conn = await self.db_pool.acquire()
            else:
                conn = await asyncpg.connect(dsn=DB_DSN)
            
            cutoff = datetime.now() - timedelta(days=older_than_days)
            
            # Update status to archived
            result = await conn.execute("""
                UPDATE NPCMemories
                SET status = 'archived'
                WHERE npc_id = $1
                  AND timestamp < $2
                  AND significance <= $3
                  AND status IN ('active', 'summarized')
            """, self.npc_id, cutoff, max_significance)
            
            logger.info(f"Archived stale memories: {result}")
        
        except Exception as e:
            logger.error(f"Error archiving stale memories: {e}")
        
        finally:
            if conn:
                if self.db_pool:
                    await self.db_pool.release(conn)
                else:
                    await conn.close()
    
    async def consolidate_memories(self, threshold: int = 10, max_days: int = 30):
        """
        Consolidate memories with similar tags into a summary memory.
        
        Args:
            threshold: Minimum number of memories to trigger consolidation
            max_days: Maximum age in days to consider
        """
        conn = None
        try:
            if self.db_pool:
                conn = await self.db_pool.acquire()
            else:
                conn = await asyncpg.connect(dsn=DB_DSN)
            
            # Find groups of memories with the same tags
            rows = await conn.fetch("""
                SELECT tags, COUNT(*) as count
                FROM NPCMemories
                WHERE npc_id = $1
                  AND timestamp > NOW() - INTERVAL '$2 days'
                  AND is_consolidated = FALSE
                  AND status IN ('active', 'summarized')
                GROUP BY tags
                HAVING COUNT(*) >= $3
            """, self.npc_id, max_days, threshold)
            
            for row in rows:
                group_tags = row["tags"]
                cnt = row["count"]
                
                if not group_tags:
                    continue
                
                # Get all memories in this group
                mem_rows = await conn.fetch("""
                    SELECT id, memory_text, memory_type, 
                           emotional_intensity, significance
                    FROM NPCMemories
                    WHERE npc_id = $1
                      AND tags = $2
                      AND is_consolidated = FALSE
                      AND status IN ('active', 'summarized')
                """, self.npc_id, group_tags)
                
                if len(mem_rows) < threshold:
                    continue
                
                # Create a summary
                joined_text = "\n".join([r["memory_text"] for r in mem_rows])
                summary_text = (
                    f"I recall {len(mem_rows)} similar events with tags {group_tags}: \n{joined_text[:200]}..."
                )
                
                # Insert consolidated memory
                new_id = await conn.fetchval("""
                    INSERT INTO NPCMemories (
                        npc_id, memory_text, memory_type,
                        emotional_intensity, significance,
                        status, is_consolidated
                    )
                    VALUES ($1, $2, 'consolidated', 10, 5, 'summarized', FALSE)
                    RETURNING id
                """, self.npc_id, summary_text)
                
                # Mark original memories as consolidated
                old_ids = [r["id"] for r in mem_rows]
                await conn.execute("""
                    UPDATE NPCMemories
                    SET is_consolidated = TRUE, status = 'summarized'
                    WHERE id = ANY($1)
                """, old_ids)
                
                logger.info(f"Consolidated {len(old_ids)} memories with tags {group_tags}")
            
            logger.info(f"Memory consolidation complete for NPC {self.npc_id}")
        
        except Exception as e:
            logger.error(f"Error consolidating memories: {e}")
        
        finally:
            if conn:
                if self.db_pool:
                    await self.db_pool.release(conn)
                else:
                    await conn.close()
    
    async def run_maintenance(self) -> Dict[str, Any]:
        """
        Run maintenance tasks on the NPC's memory system.
        
        Returns:
            Results of maintenance operations
        """
        if self.use_subsystems:
            try:
                integrated = IntegratedMemorySystem(self.user_id, self.conversation_id)
                return await integrated.run_memory_maintenance(
                    entity_type="npc",
                    entity_id=self.npc_id
                )
            except Exception as e:
                logger.error(f"Error running subsystem maintenance: {e}")
                # Fall back to direct maintenance
        
        # Direct maintenance operations
        results = {
            "pruned_memories": 0,
            "decayed_memories": 0,
            "summarized_memories": 0,
            "archived_memories": 0,
            "consolidated_memories": 0
        }
        
        try:
            # Run all maintenance operations
            await self.prune_old_memories()
            results["pruned_memories"] = 1  # Just indicating success
            
            await self.apply_memory_decay()
            results["decayed_memories"] = 1
            
            await self.summarize_repetitive_memories()
            results["summarized_memories"] = 1
            
            await self.archive_stale_memories()
            results["archived_memories"] = 1
            
            await self.consolidate_memories()
            results["consolidated_memories"] = 1
            
            logger.info(f"Maintenance complete for NPC {self.npc_id}")
            return results
        
        except Exception as e:
            logger.error(f"Error running maintenance: {e}")
            return {"error": str(e)}
    
    async def create_belief(self, belief_text: str, confidence: float = 0.7) -> Dict[str, Any]:
        """
        Create a belief for the NPC.
        
        Args:
            belief_text: The belief statement
            confidence: Confidence level (0.0-1.0)
            
        Returns:
            Created belief information
        """
        if self.use_subsystems:
            return await self.semantic_manager.create_belief(
                entity_type="npc",
                entity_id=self.npc_id,
                belief_text=belief_text,
                confidence=confidence
            )
        else:
            # Direct DB implementation
            conn = None
            try:
                if self.db_pool:
                    conn = await self.db_pool.acquire()
                else:
                    conn = await asyncpg.connect(dsn=DB_DSN)
                
                # Extract topic from belief
                words = belief_text.lower().split()
                topic = words[0] if words else "general"
                
                belief_id = await conn.fetchval("""
                    INSERT INTO NPCBeliefs (
                        npc_id, belief_text, confidence, topic, created_at
                    )
                    VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP)
                    RETURNING id
                """, self.npc_id, belief_text, confidence, topic)
                
                return {
                    "belief_id": belief_id,
                    "belief_text": belief_text,
                    "confidence": confidence,
                    "topic": topic,
                    "created_at": datetime.now().isoformat()
                }
            
            except Exception as e:
                logger.error(f"Error creating belief: {e}")
                return {"error": str(e)}
            
            finally:
                if conn:
                    if self.db_pool:
                        await self.db_pool.release(conn)
                    else:
                        await conn.close()
    
    async def get_beliefs(self, topic: str = None) -> List[Dict[str, Any]]:
        """
        Get the NPC's beliefs, optionally filtered by topic.
        
        Args:
            topic: Optional topic filter
            
        Returns:
            List of beliefs
        """
        if self.use_subsystems:
            return await self.semantic_manager.get_beliefs(
                entity_type="npc",
                entity_id=self.npc_id,
                topic=topic
            )
        else:
            # Direct DB implementation
            conn = None
            try:
                if self.db_pool:
                    conn = await self.db_pool.acquire()
                else:
                    conn = await asyncpg.connect(dsn=DB_DSN)
                
                if topic:
                    rows = await conn.fetch("""
                        SELECT id, belief_text, confidence, topic, created_at
                        FROM NPCBeliefs
                        WHERE npc_id = $1 AND topic = $2
                        ORDER BY confidence DESC
                    """, self.npc_id, topic)
                else:
                    rows = await conn.fetch("""
                        SELECT id, belief_text, confidence, topic, created_at
                        FROM NPCBeliefs
                        WHERE npc_id = $1
                        ORDER BY confidence DESC
                    """, self.npc_id)
                
                return [{
                    "belief_id": r["id"],
                    "belief_text": r["belief_text"],
                    "confidence": r["confidence"],
                    "topic": r["topic"],
                    "created_at": r["created_at"].isoformat()
                } for r in rows]
            
            except Exception as e:
                logger.error(f"Error retrieving beliefs: {e}")
                return []
            
            finally:
                if conn:
                    if self.db_pool:
                        await self.db_pool.release(conn)
                    else:
                        await conn.close()
    
    async def detect_patterns(self) -> Dict[str, Any]:
        """
        Detect patterns in the NPC's memories and generate insights.
        
        Returns:
            Pattern detection results
        """
        if self.use_subsystems:
            return await self.semantic_manager.find_patterns_across_memories(
                entity_type="npc",
                entity_id=self.npc_id
            )
        else:
            # Simplified implementation - in a real system, this would use
            # more sophisticated analysis
            conn = None
            try:
                if self.db_pool:
                    conn = await self.db_pool.acquire()
                else:
                    conn = await asyncpg.connect(dsn=DB_DSN)
                
                # Get memory tags and their frequencies
                tag_rows = await conn.fetch("""
                    SELECT unnest(tags) as tag, COUNT(*) as count
                    FROM NPCMemories
                    WHERE npc_id = $1 AND status IN ('active', 'summarized')
                    GROUP BY tag
                    ORDER BY count DESC
                    LIMIT 5
                """, self.npc_id)
                
                tag_patterns = [{
                    "tag": r["tag"],
                    "frequency": r["count"]
                } for r in tag_rows]
                
                # Get frequent memory types
                type_rows = await conn.fetch("""
                    SELECT memory_type, COUNT(*) as count
                    FROM NPCMemories
                    WHERE npc_id = $1 AND status IN ('active', 'summarized')
                    GROUP BY memory_type
                    ORDER BY count DESC
                """, self.npc_id)
                
                type_patterns = [{
                    "memory_type": r["memory_type"],
                    "frequency": r["count"]
                } for r in type_rows]
                
                return {
                    "tag_patterns": tag_patterns,
                    "type_patterns": type_patterns,
                    "analysis": f"NPC {self.npc_id} shows patterns in memory tags and types"
                }
            
            except Exception as e:
                logger.error(f"Error detecting patterns: {e}")
                return {"error": str(e)}
            
            finally:
                if conn:
                    if self.db_pool:
                        await self.db_pool.release(conn)
                    else:
                        await conn.close()
    
    async def generate_counterfactual(self, memory_id: int, variation_type: str = "alternative") -> Dict[str, Any]:
        """
        Generate a counterfactual memory (what could have happened differently).
        
        Args:
            memory_id: ID of the base memory
            variation_type: Type of counterfactual variation
            
        Returns:
            Counterfactual memory
        """
        if self.use_subsystems:
            return await self.semantic_manager.generate_counterfactual(
                memory_id=memory_id,
                entity_type="npc",
                entity_id=self.npc_id,
                variation_type=variation_type
            )
        else:
            # Direct DB implementation
            conn = None
            try:
                if self.db_pool:
                    conn = await self.db_pool.acquire()
                else:
                    conn = await asyncpg.connect(dsn=DB_DSN)
                
                # Get the original memory
                row = await conn.fetchrow("""
                    SELECT memory_text, memory_type, tags, significance
                    FROM NPCMemories
                    WHERE id = $1 AND npc_id = $2
                """, memory_id, self.npc_id)
                
                if not row:
                    return {"error": "Memory not found"}
                
                original_text = row["memory_text"]
                
                # Create a simple counterfactual by negating or altering the original
                if variation_type == "alternative":
                    # Replace positive with negative or vice versa
                    if any(word in original_text.lower() for word in ["good", "happy", "success"]):
                        cf_text = f"What if instead of {original_text}, the opposite had happened?"
                    else:
                        cf_text = f"What if instead of {original_text}, things had gone better?"
                elif variation_type == "prevention":
                    cf_text = f"How could I have prevented: {original_text}?"
                else:
                    cf_text = f"Alternate scenario for: {original_text}"
                
                # Store the counterfactual
                cf_id = await conn.fetchval("""
                    INSERT INTO NPCCounterfactuals (
                        npc_id, original_memory_id, counterfactual_text, 
                        variation_type, created_at
                    )
                    VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP)
                    RETURNING id
                """, self.npc_id, memory_id, cf_text, variation_type)
                
                return {
                    "counterfactual_id": cf_id,
                    "original_memory_id": memory_id,
                    "original_text": original_text,
                    "counterfactual_text": cf_text,
                    "variation_type": variation_type
                }
            
            except Exception as e:
                logger.error(f"Error generating counterfactual: {e}")
                return {"error": str(e)}
            
            finally:
                if conn:
                    if self.db_pool:
                        await self.db_pool.release(conn)
                    else:
                        await conn.close()
    
    # Memory cache methods for optimization
    def preload_memories_into_cache(self, memory_rows):
        """
        Preload frequently accessed memories into local cache.
        
        Args:
            memory_rows: Memory data to cache
        """
        self.memory_cache["cached_relevant_memories"] = memory_rows
        self.memory_cache["last_cached"] = datetime.utcnow()
    
    def get_cached_memories(self):
        """
        Get preloaded memories from cache if available.
        
        Returns:
            Cached memories or empty list
        """
        return self.memory_cache.get("cached_relevant_memories", [])
    
    async def dynamic_npc_memory_preloading(
        self, 
        query_threshold: int = 10,
        time_window_minutes: int = 5,
        cache_timeout_minutes: int = 30
    ):
        """
        Preload memories for frequently queried NPCs.
        
        Args:
            query_threshold: Query count threshold
            time_window_minutes: Time window to count queries
            cache_timeout_minutes: Cache timeout in minutes
        """
        conn = None
        try:
            if self.db_pool:
                conn = await self.db_pool.acquire()
            else:
                conn = await asyncpg.connect(dsn=DB_DSN)
            
            # Check query frequency
            query_count = await conn.fetchval("""
                SELECT COUNT(*)
                FROM InteractionLogs
                WHERE npc_id = $1
                  AND timestamp > (NOW() - INTERVAL '$2 minutes')
            """, self.npc_id, time_window_minutes)
            
            # Check if cache is stale
            current_time = datetime.utcnow()
            last_cached = self.memory_cache.get("last_cached")
            is_cache_stale = not last_cached or (current_time - last_cached).total_seconds() > (cache_timeout_minutes * 60)
            
            # Refresh cache if needed
            if query_count and query_count >= query_threshold and is_cache_stale:
                logger.info(f"Refreshing memory cache for high-frequency NPC {self.npc_id}")
                
                # Get frequently accessed memories
                query_embedding = await self.generate_embedding("general recall")
                
                if query_embedding:
                    cached_memories = await conn.fetch("""
                        SELECT id, memory_text, memory_type, tags,
                               emotional_intensity, significance,
                               times_recalled, timestamp, status
                        FROM NPCMemories
                        WHERE npc_id = $1 AND status = 'active'
                        ORDER BY embedding <-> $2
                        LIMIT 5
                    """, self.npc_id, query_embedding)
                else:
                    # Fallback to recency
                    cached_memories = await conn.fetch("""
                        SELECT id, memory_text, memory_type, tags,
                               emotional_intensity, significance,
                               times_recalled, timestamp, status
                        FROM NPCMemories
                        WHERE npc_id = $1 AND status = 'active'
                        ORDER BY significance DESC, timestamp DESC
                        LIMIT 5
                    """, self.npc_id)
                
                self.preload_memories_into_cache(cached_memories)
        
        except Exception as e:
            logger.error(f"Error preloading memories: {e}")
        
        finally:
            if conn:
                if self.db_pool:
                    await self.db_pool.release(conn)
                else:
                    await conn.close()

# Utility functions for memory system training

async def train_reflection_model(db_pool):
    """
    Train reflection settings based on accuracy of past reflections.
    
    Args:
        db_pool: Database connection pool
    """
    conn = await db_pool.acquire()
    try:
        # Get statistics on reflection accuracy
        row = await conn.fetchrow("""
            SELECT
                SUM(CASE WHEN was_accurate=TRUE THEN confidence_weight ELSE 0 END) AS correct_conf_sum,
                SUM(confidence_weight) AS total_conf_sum,
                COUNT(*) FILTER (WHERE was_accurate=TRUE) AS correct_count,
                COUNT(*) AS total_count
            FROM ReflectionLogs
        """)
        
        if not row or row["total_count"] == 0:
            return
        
        correct_count = row["correct_count"]
        total_count = row["total_count"]
        accuracy_rate = correct_count / total_count
        
        # Calculate weighted accuracy if possible
        if row["total_conf_sum"] and row["total_conf_sum"] > 0:
            weighted_accuracy = row["correct_conf_sum"] / row["total_conf_sum"]
        else:
            weighted_accuracy = 0
        
        # Combined metric
        final_metric = (accuracy_rate + weighted_accuracy) / 2
        
        # Update model settings
        await conn.execute("""
            UPDATE AIReflectionSettings
            SET temperature = GREATEST(0.2, 1.0 - ($1 * 0.8)),
                max_tokens = LEAST(4000, 2000 + ($1 * 2000))
            WHERE id = 1
        """, final_metric)
        
        logger.info(f"Trained reflection model: accuracy={accuracy_rate:.2f}, weighted={weighted_accuracy:.2f}")
    
    except Exception as e:
        logger.error(f"Error training reflection model: {e}")
    
    finally:
        await db_pool.release(conn)

async def periodic_reflection_training(db_pool, interval_hours=12):
    """
    Periodically train the reflection model.
    
    Args:
        db_pool: Database connection pool
        interval_hours: Training interval in hours
    """
    while True:
        logger.info("Running periodic reflection training")
        await train_reflection_model(db_pool)
        await asyncio.sleep(interval_hours * 3600)

    async def batch_create_memories(
        self, 
        npc_id: int,
        memories: List[Dict[str, Any]]
    ) -> List[int]:
        """
        Create multiple memories in a single operation for better performance.
        
        Args:
            npc_id: ID of the NPC
            memories: List of memory objects with text, type, significance, etc.
            
        Returns:
            List of created memory IDs
        """
        if not memories:
            return []
            
        memory_system = await self._get_memory_system()
        
        # Prepare all memories for batch insertion
        batch_values = []
        for memory in memories:
            # Get emotional analysis for each memory text
            emotion_analysis = await memory_system.emotional_manager.analyze_emotional_content(
                memory.get("text", "")
            )
            
            batch_values.append({
                "entity_type": "npc",
                "entity_id": npc_id,
                "memory_text": memory.get("text", ""),
                "importance": memory.get("importance", "medium"),
                "emotional": memory.get("emotional", False),
                "primary_emotion": emotion_analysis.get("primary_emotion", "neutral"),
                "emotion_intensity": emotion_analysis.get("intensity", 0.5),
                "tags": memory.get("tags", [])
            })
        
        # Execute batch insert
        results = await memory_system.batch_remember(batch_values)
        
        # Process schemas in background task for efficiency
        self._process_batch_schemas(results, npc_id)
        
        return results.get("memory_ids", [])
    
    def _process_batch_schemas(self, results, npc_id):
        """Process schemas for batch memories in background to avoid blocking."""
        async def process_task():
            try:
                memory_system = await self._get_memory_system()
                memory_ids = results.get("memory_ids", [])
                
                # Process schemas in smaller batches for better control
                batch_size = 5
                for i in range(0, len(memory_ids), batch_size):
                    batch = memory_ids[i:i+batch_size]
                    try:
                        await memory_system.integrated.batch_apply_schemas(
                            memory_ids=batch,
                            entity_type="npc",
                            entity_id=npc_id,
                            auto_detect=True
                        )
                    except Exception as e:
                        logger.error(f"Error applying schemas to memory batch: {e}")
                    
                    # Small delay to avoid overwhelming the system
                    await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in schema processing task: {e}")
                    
        # Create and start the task properly
        task = asyncio.create_task(process_task())
        return task  # Return task so caller can await if needed
    
    async def batch_retrieve_memories(
        self,
        query: str,
        npc_ids: List[int],
        limit_per_npc: int = 3
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Retrieve memories for multiple NPCs in a single operation.
        
        Args:
            query: Search query
            npc_ids: List of NPC IDs
            limit_per_npc: Maximum memories per NPC
            
        Returns:
            Dictionary mapping NPC IDs to their memory lists
        """
        if not npc_ids:
            return {}
            
        memory_system = await self._get_memory_system()
        
        # Prepare batch query parameters
        batch_params = [
            {
                "entity_type": "npc",
                "entity_id": npc_id,
                "query": query,
                "limit": limit_per_npc
            }
            for npc_id in npc_ids
        ]
        
        # Execute batch memory retrieval
        batch_results = await memory_system.batch_recall(batch_params)
        
        # Format results by NPC ID
        memory_map = {}
        for npc_id, result in zip(npc_ids, batch_results):
            memory_map[npc_id] = result.get("memories", [])
            
        return memory_map
