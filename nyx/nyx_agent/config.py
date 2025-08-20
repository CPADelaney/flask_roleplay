# nyx/nyx_agent/config.py
"""Configuration constants for Nyx Agent SDK"""

class Config:
    """Configuration constants to avoid magic numbers"""
    # Entity types
    ENTITY_TYPE_INTEGRATED = "integrated"
    ENTITY_TYPE_USER = "user"
    ENTITY_TYPE_NPC = "npc"
    ENTITY_TYPE_ENTITY = "entity"
    
    # Memory thresholds
    HIGH_MEMORY_THRESHOLD_MB = 500
    MAX_RESPONSE_TIMES = 100
    MAX_ERROR_LOG_ENTRIES = 100
    MAX_ADAPTATION_HISTORY = 100
    MAX_LEARNED_PATTERNS = 50
    
    # Performance thresholds
    HIGH_RESPONSE_TIME_THRESHOLD = 2.0
    MIN_SUCCESS_RATE = 0.8
    HIGH_ERROR_COUNT = 100
    
    # Relationship thresholds
    INTIMATE_TRUST_THRESHOLD = 0.8
    INTIMATE_BOND_THRESHOLD = 0.7
    FRIENDLY_TRUST_THRESHOLD = 0.6
    HOSTILE_TRUST_THRESHOLD = 0.3
    DOMINANT_POWER_THRESHOLD = 0.7
    SUBMISSIVE_POWER_THRESHOLD = 0.3
    
    # Task thresholds
    MIN_NPC_RELATIONSHIP_FOR_TASK = 30
    
    # Emotional thresholds
    HIGH_AROUSAL_THRESHOLD = 0.8
    NEGATIVE_VALENCE_THRESHOLD = -0.5
    POSITIVE_VALENCE_THRESHOLD = 0.5
    HIGH_DOMINANCE_THRESHOLD = 0.8
    EMOTIONAL_VARIANCE_THRESHOLD = 0.5
    
    # Memory relevance thresholds
    VIVID_RECALL_THRESHOLD = 0.8
    REMEMBER_THRESHOLD = 0.6
    THINK_RECALL_THRESHOLD = 0.4
    
    # Decision thresholds
    MIN_DECISION_SCORE = 0.3
    FALLBACK_DECISION_SCORE = 0.4
    
    # Conflict detection
    POWER_CONFLICT_THRESHOLD = 0.7
    MAX_STABILITY_ISSUES = 10
