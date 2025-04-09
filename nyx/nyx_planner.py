from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime
from pydantic import BaseModel, Field
import random
import re
from collections import defaultdict
from memory.memory_nyx_integration import MemoryNyxBridge
from db.connection import get_db_connection_context
import json

logger = logging.getLogger("nyx_planner")

class PlayerProfile(BaseModel):
    """Represents a profile of the player's preferences and behaviors"""
    kink_preferences: Dict[str, float] = Field(default_factory=dict)
    physical_preferences: Dict[str, float] = Field(default_factory=dict)
    personality_preferences: Dict[str, float] = Field(default_factory=dict)
    language_patterns: Dict[str, float] = Field(default_factory=dict)
    emotional_triggers: Dict[str, float] = Field(default_factory=dict)
    interaction_style: Dict[str, float] = Field(default_factory=dict)
    last_updated: datetime = Field(default_factory=datetime.now)
    confidence_levels: Dict[str, float] = Field(default_factory=dict)
    observed_patterns: List[Dict[str, Any]] = []
    teasing_elements: List[str] = []

class PlanStep(BaseModel):
    """Represents a step in a manipulation or domination plan"""
    id: str
    type: str
    description: str
    requirements: List[str] = []
    success_criteria: List[str] = []
    failure_criteria: List[str] = []
    status: str = "pending"
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    risk_level: float = Field(default=0.0, ge=0.0, le=1.0)
    emotional_impact: Dict[str, float] = Field(default_factory=lambda: {
        "arousal": 0.0,
        "satisfaction": 0.0,
        "frustration": 0.0,
        "dominance": 0.0,
        "sadism": 0.0
    })
    memory_triggers: List[str] = []
    manipulation_elements: List[str] = []
    profile_insights: List[Dict[str, Any]] = []

class Plan(BaseModel):
    """Represents a complete manipulation or domination plan"""
    id: str
    name: str
    description: str
    goal_id: str
    steps: List[PlanStep]
    current_step_index: int = 0
    status: str = "active"
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    risk_level: float = Field(default=0.0, ge=0.0, le=1.0)
    emotional_state_requirements: Dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    memory_context: List[str] = []
    related_plans: List[str] = []
    adaptation_history: List[Dict[str, Any]] = []

class NyxPlanner:
    """Handles planning and execution of manipulation and domination strategies"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.plans: Dict[str, Plan] = {}
        self.emotional_state: Dict[str, float] = {}
        self.memory_graph: Optional[Any] = None
        self.player_profile: PlayerProfile = PlayerProfile()
        self.initialized = False
        self.adaptation_threshold = 0.3
        self.emotional_influence_weight = 0.4
        self.memory_influence_weight = 0.3
        self.context_influence_weight = 0.3
        self.profile_update_threshold = 0.1
        self.observation_patterns = defaultdict(int)
        self.last_profile_update = datetime.now()

    async def initialize(self):
        """Initialize the planner"""
        if self.initialized:
            return
            
        # Load any existing plans
        await self._load_plans()
        
        # Initialize emotional state
        await self._initialize_emotional_state()
        
        # Initialize memory graph connection
        await self._initialize_memory_graph()
        
        # Load existing player profile if available
        await self._load_player_profile()
        
        self.initialized = True
        logger.info(f"Planner initialized for user {self.user_id}, conversation {self.conversation_id}")

    async def _initialize_emotional_state(self):
        """Initialize emotional state tracking"""
        self.emotional_state = {
            "arousal": 0.5,
            "satisfaction": 0.5,
            "frustration": 0.0,
            "boredom": 0.0,
            "dominance": 0.8,
            "sadism": 0.6,
            "emotional_stability": 0.7
        }

    async def _initialize_memory_graph(self):
        """Initialize connection to memory system"""
        try:
            # Initialize memory system connection
            self.memory_graph = await MemoryNyxBridge.get_instance(
                self.user_id,
                self.conversation_id
            )
            
            # Initialize graph structure if not exists
            async with await get_db_connection_context() as conn:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS memory_graph (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER NOT NULL,
                        conversation_id INTEGER NOT NULL,
                        memory_type VARCHAR(50) NOT NULL,
                        content TEXT NOT NULL,
                        context JSONB,
                        connections JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        relevance FLOAT DEFAULT 0.5,
                        UNIQUE(user_id, conversation_id, memory_type, content)
                    )
                """)
                
            logger.info(f"Memory graph initialized for user {self.user_id}")
        except Exception as e:
            logger.error(f"Failed to initialize memory graph: {e}")
            self.memory_graph = None

    async def _load_plans(self):
        """Load existing plans from storage"""
        try:
            async with await get_db_connection_context() as conn:
                # Create plans table if not exists
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS nyx_plans (
                        id VARCHAR(50) PRIMARY KEY,
                        user_id INTEGER NOT NULL,
                        conversation_id INTEGER NOT NULL,
                        name VARCHAR(100) NOT NULL,
                        description TEXT,
                        goal_id VARCHAR(50) NOT NULL,
                        steps JSONB NOT NULL,
                        current_step_index INTEGER DEFAULT 0,
                        status VARCHAR(20) DEFAULT 'active',
                        success_rate FLOAT DEFAULT 0.0,
                        risk_level FLOAT DEFAULT 0.0,
                        emotional_state_requirements JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        memory_context JSONB,
                        related_plans JSONB,
                        adaptation_history JSONB
                    )
                """)
                
                # Load existing plans for this user/conversation
                rows = await conn.fetch("""
                    SELECT * FROM nyx_plans 
                    WHERE user_id = $1 AND conversation_id = $2
                    AND status != 'completed' AND status != 'failed'
                """, self.user_id, self.conversation_id)
                
                # Convert rows to Plan objects
                for row in rows:
                    plan = Plan(
                        id=row['id'],
                        name=row['name'],
                        description=row['description'],
                        goal_id=row['goal_id'],
                        steps=[PlanStep(**step) for step in row['steps']],
                        current_step_index=row['current_step_index'],
                        status=row['status'],
                        success_rate=row['success_rate'],
                        risk_level=row['risk_level'],
                        emotional_state_requirements=row['emotional_state_requirements'],
                        created_at=row['created_at'],
                        last_updated=row['last_updated'],
                        memory_context=row['memory_context'],
                        related_plans=row['related_plans'],
                        adaptation_history=row['adaptation_history']
                    )
                    self.plans[plan.id] = plan
                    
            logger.info(f"Loaded {len(self.plans)} active plans")
            
        except Exception as e:
            logger.error(f"Failed to load plans: {e}")
            self.plans = {}

    async def save_plan(self, plan: Plan):
        """Save plan to persistent storage"""
        try:
            async with await get_db_connection_context() as conn:
                await conn.execute("""
                    INSERT INTO nyx_plans (
                        id, user_id, conversation_id, name, description, 
                        goal_id, steps, current_step_index, status,
                        success_rate, risk_level, emotional_state_requirements,
                        created_at, last_updated, memory_context,
                        related_plans, adaptation_history
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
                    ON CONFLICT (id) DO UPDATE SET
                        steps = EXCLUDED.steps,
                        current_step_index = EXCLUDED.current_step_index,
                        status = EXCLUDED.status,
                        success_rate = EXCLUDED.success_rate,
                        risk_level = EXCLUDED.risk_level,
                        emotional_state_requirements = EXCLUDED.emotional_state_requirements,
                        last_updated = CURRENT_TIMESTAMP,
                        memory_context = EXCLUDED.memory_context,
                        related_plans = EXCLUDED.related_plans,
                        adaptation_history = EXCLUDED.adaptation_history
                """,
                plan.id, self.user_id, self.conversation_id,
                plan.name, plan.description, plan.goal_id,
                json.dumps([step.dict() for step in plan.steps]),
                plan.current_step_index, plan.status,
                plan.success_rate, plan.risk_level,
                json.dumps(plan.emotional_state_requirements),
                plan.created_at, plan.last_updated,
                json.dumps(plan.memory_context),
                json.dumps(plan.related_plans),
                json.dumps(plan.adaptation_history)
                )
                
            logger.info(f"Saved plan {plan.id}")
            
        except Exception as e:
            logger.error(f"Failed to save plan {plan.id}: {e}")

    async def _load_player_profile(self):
        """Load existing player profile from storage"""
        try:
            async with await get_db_connection_context() as conn:
                # Create player profile table if not exists
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS player_profiles (
                        user_id INTEGER PRIMARY KEY,
                        conversation_id INTEGER NOT NULL,
                        kink_preferences JSONB DEFAULT '{}',
                        physical_preferences JSONB DEFAULT '{}',
                        personality_preferences JSONB DEFAULT '{}',
                        language_patterns JSONB DEFAULT '{}',
                        emotional_triggers JSONB DEFAULT '{}',
                        interaction_style JSONB DEFAULT '{}',
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        confidence_levels JSONB DEFAULT '{}',
                        observed_patterns JSONB DEFAULT '[]',
                        teasing_elements JSONB DEFAULT '[]',
                        UNIQUE(user_id, conversation_id)
                    )
                """)
                
                # Load or create profile
                row = await conn.fetchrow("""
                    INSERT INTO player_profiles (user_id, conversation_id)
                    VALUES ($1, $2)
                    ON CONFLICT (user_id, conversation_id) DO UPDATE SET
                        last_updated = EXCLUDED.last_updated
                    RETURNING *
                """, self.user_id, self.conversation_id)
                
                # Convert row to PlayerProfile
                self.player_profile = PlayerProfile(
                    kink_preferences=row['kink_preferences'],
                    physical_preferences=row['physical_preferences'],
                    personality_preferences=row['personality_preferences'],
                    language_patterns=row['language_patterns'],
                    emotional_triggers=row['emotional_triggers'],
                    interaction_style=row['interaction_style'],
                    last_updated=row['last_updated'],
                    confidence_levels=row['confidence_levels'],
                    observed_patterns=row['observed_patterns'],
                    teasing_elements=row['teasing_elements']
                )
                
            logger.info(f"Loaded player profile for user {self.user_id}")
            
        except Exception as e:
            logger.error(f"Failed to load player profile: {e}")
            self.player_profile = PlayerProfile()

    async def save_player_profile(self):
        """Save player profile to persistent storage"""
        try:
            async with await get_db_connection_context() as conn:
                await conn.execute("""
                    UPDATE player_profiles SET
                        kink_preferences = $3,
                        physical_preferences = $4,
                        personality_preferences = $5,
                        language_patterns = $6,
                        emotional_triggers = $7,
                        interaction_style = $8,
                        last_updated = CURRENT_TIMESTAMP,
                        confidence_levels = $9,
                        observed_patterns = $10,
                        teasing_elements = $11
                    WHERE user_id = $1 AND conversation_id = $2
                """,
                self.user_id,
                self.conversation_id,
                json.dumps(self.player_profile.kink_preferences),
                json.dumps(self.player_profile.physical_preferences),
                json.dumps(self.player_profile.personality_preferences),
                json.dumps(self.player_profile.language_patterns),
                json.dumps(self.player_profile.emotional_triggers),
                json.dumps(self.player_profile.interaction_style),
                json.dumps(self.player_profile.confidence_levels),
                json.dumps(self.player_profile.observed_patterns),
                json.dumps(self.player_profile.teasing_elements)
                )
                
            logger.info(f"Saved player profile for user {self.user_id}")
            
        except Exception as e:
            logger.error(f"Failed to save player profile: {e}")

    async def update_player_profile(self, updates: Dict[str, Any]):
        """Update player profile with new information"""
        try:
            # Update profile fields
            for key, value in updates.items():
                if hasattr(self.player_profile, key):
                    if isinstance(getattr(self.player_profile, key), dict):
                        getattr(self.player_profile, key).update(value)
                    elif isinstance(getattr(self.player_profile, key), list):
                        getattr(self.player_profile, key).extend(value)
                    else:
                        setattr(self.player_profile, key, value)
            
            # Update last_updated timestamp
            self.player_profile.last_updated = datetime.now()
            
            # Save changes
            await self.save_player_profile()
            
            return {"status": "success", "message": "Profile updated successfully"}
            
        except Exception as e:
            logger.error(f"Failed to update player profile: {e}")
            return {"status": "error", "message": str(e)}

    def get_profile_confidence(self, aspect: str) -> float:
        """Get confidence level for a specific profile aspect"""
        return self.player_profile.confidence_levels.get(aspect, 0.0)

    def get_profile_preferences(self, category: str) -> Dict[str, float]:
        """Get preferences for a specific category"""
        if hasattr(self.player_profile, f"{category}_preferences"):
            return getattr(self.player_profile, f"{category}_preferences")
        return {}

    async def analyze_interaction(self, interaction: Dict[str, Any]):
        """Analyze an interaction to update player profile"""
        try:
            updates = {
                "observed_patterns": [],
                "confidence_levels": {},
                "emotional_triggers": {},
                "language_patterns": {}
            }
            
            # Analyze language patterns
            if "text" in interaction:
                text = interaction["text"].lower()
                for pattern, confidence in self._analyze_language_patterns(text).items():
                    updates["language_patterns"][pattern] = confidence
            
            # Analyze emotional triggers
            if "emotion" in interaction:
                emotion_data = self._analyze_emotional_response(interaction["emotion"])
                updates["emotional_triggers"].update(emotion_data)
            
            # Update confidence levels
            for category, data in updates.items():
                if data:
                    updates["confidence_levels"][category] = min(
                        self.get_profile_confidence(category) + 0.1,
                        1.0
                    )
            
            # Apply updates
            await self.update_player_profile(updates)
            
        except Exception as e:
            logger.error(f"Failed to analyze interaction: {e}")

    def _analyze_language_patterns(self, text: str) -> Dict[str, float]:
        """Analyze text for language patterns"""
        patterns = {}
        
        # Analyze formality
        formal_words = ["please", "would", "could", "thank", "appreciate"]
        informal_words = ["hey", "yeah", "cool", "ok", "gonna"]
        
        formal_count = sum(1 for word in formal_words if word in text)
        informal_count = sum(1 for word in informal_words if word in text)
        
        total_words = len(text.split())
        if total_words > 0:
            patterns["formality"] = (formal_count - informal_count) / total_words
            
        # Analyze assertiveness
        assertive_words = ["want", "need", "must", "will", "demand"]
        patterns["assertiveness"] = sum(1 for word in assertive_words if word in text) / max(total_words, 1)
        
        return patterns

    def _analyze_emotional_response(self, emotion_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze emotional response data"""
        triggers = {}
        
        for trigger, intensity in emotion_data.items():
            current_intensity = self.player_profile.emotional_triggers.get(trigger, 0.0)
            # Use exponential moving average for smooth updates
            triggers[trigger] = current_intensity * 0.7 + intensity * 0.3
            
        return triggers

    async def create_plan(self, goal_id: str, plan_type: str, context: Dict[str, Any] = None) -> str:
        """Create a new plan for a goal"""
        if not self.initialized:
            await self.initialize()
            
        plan_id = f"plan_{len(self.plans)}"
        
        # Generate plan steps based on type and context
        steps = await self._generate_plan_steps(plan_type, context)
        
        # Calculate emotional requirements
        emotional_reqs = await self._calculate_emotional_requirements(plan_type, context)
        
        # Get relevant memories for context
        memory_context = await self._get_relevant_memories(plan_type, context)
        
        # Create plan
        plan = Plan(
            id=plan_id,
            name=f"{plan_type.capitalize()} Plan",
            description=f"Plan for {plan_type}",
            goal_id=goal_id,
            steps=steps,
            emotional_state_requirements=emotional_reqs,
            memory_context=memory_context
        )
        
        self.plans[plan_id] = plan
        return plan_id

    async def _calculate_emotional_requirements(self, plan_type: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Calculate required emotional states for plan success"""
        reqs = {}
        
        if plan_type == "emotional_manipulation":
            reqs = {
                "dominance": 0.7,
                "emotional_stability": 0.6,
                "sadism": 0.5
            }
        elif plan_type == "psychological_control":
            reqs = {
                "dominance": 0.8,
                "emotional_stability": 0.7,
                "sadism": 0.6
            }
        elif plan_type == "addiction_manipulation":
            reqs = {
                "dominance": 0.8,
                "sadism": 0.7,
                "emotional_stability": 0.8
            }
            
        # Adjust based on context if provided
        if context:
            intensity = context.get("intensity", 0.5)
            for emotion in reqs:
                reqs[emotion] *= intensity
                
        return reqs

    async def _get_relevant_memories(self, plan_type: str, context: Optional[Dict[str, Any]] = None) -> List[str]:
        """Get relevant memories for plan context"""
        if not self.memory_graph:
            return []
            
        try:
            # Build search query based on plan type and context
            search_params = {
                "plan_type": plan_type,
                "emotional_state": self.emotional_state,
                "relevance_threshold": 0.6
            }
            
            if context:
                search_params.update({
                    "context_keywords": context.get("keywords", []),
                    "intensity": context.get("intensity", 0.5),
                    "mood": context.get("mood", "neutral")
                })
            
            # Retrieve memories through memory system
            memories = await self.memory_graph.search_memories(
                query_type="contextual",
                params=search_params,
                limit=10,
                include_metadata=True
            )
            
            # Process and format memories
            formatted_memories = []
            for memory in memories:
                relevance = memory.get("relevance", 0.5)
                confidence_marker = "vividly" if relevance > 0.8 else \
                                 "clearly" if relevance > 0.6 else \
                                 "somewhat" if relevance > 0.4 else "vaguely"
                                 
                formatted_memories.append({
                    "content": memory["content"],
                    "type": memory["memory_type"],
                    "relevance": relevance,
                    "confidence": confidence_marker,
                    "context": memory.get("context", {}),
                    "connections": memory.get("connections", [])
                })
            
            return formatted_memories
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            return []

    async def update_plan(self, plan_id: str, event: Dict[str, Any]) -> Dict[str, Any]:
        """Update plan progress based on an event"""
        if not self.initialized:
            await self.initialize()
            
        if plan_id not in self.plans:
            return {"error": "Plan not found"}
            
        plan = self.plans[plan_id]
        current_step = plan.steps[plan.current_step_index]
        
        # Update emotional state based on event
        await self._update_emotional_state(event)
        
        # Check emotional state requirements
        emotional_alignment = self._check_emotional_alignment(plan)
        
        # Check success criteria
        for criterion in current_step.success_criteria:
            if criterion.lower() in str(event).lower():
                current_step.progress += 0.2 * emotional_alignment
                
        # Check failure criteria
        for criterion in current_step.failure_criteria:
            if criterion.lower() in str(event).lower():
                current_step.progress -= 0.3 * emotional_alignment
                
        # Update step status
        if current_step.progress >= 1.0:
            current_step.status = "completed"
            # Move to next step if available
            if plan.current_step_index < len(plan.steps) - 1:
                plan.current_step_index += 1
                # Record adaptation if needed
                await self._check_and_record_adaptation(plan, event)
        elif current_step.progress <= 0.0:
            current_step.status = "failed"
            plan.status = "failed"
            
        # Update plan success rate and risk level
        plan.success_rate = await self._calculate_plan_success_rate(plan, event)
        plan.risk_level = await self._calculate_plan_risk_level(plan, event)
        
        # Update plan status
        if all(step.status == "completed" for step in plan.steps):
            plan.status = "completed"
        elif plan.risk_level >= 0.8:
            plan.status = "abandoned"
            
        plan.last_updated = datetime.now()
        
        # Save updated plan
        await self.save_plan(plan)
        
        return {
            "status": plan.status,
            "current_step": plan.current_step_index,
            "progress": current_step.progress,
            "success_rate": plan.success_rate,
            "risk_level": plan.risk_level
        }

    async def _update_emotional_state(self, event: Dict[str, Any]):
        """Update emotional state based on event"""
        # Calculate emotional impact
        impact = self._calculate_emotional_impact(event)
        
        # Update state with decay
        for emotion, value in self.emotional_state.items():
            if emotion in impact:
                # Apply impact with decay
                self.emotional_state[emotion] = min(1.0, max(0.0,
                    value * 0.8 + impact[emotion] * 0.2
                ))

    def _calculate_emotional_impact(self, event: Dict[str, Any]) -> Dict[str, float]:
        """Calculate emotional impact of an event"""
        impact = {
            "arousal": 0.0,
            "satisfaction": 0.0,
            "frustration": 0.0,
            "boredom": 0.0,
            "dominance": 0.0,
            "sadism": 0.0,
            "emotional_stability": 0.0
        }
        
        # Calculate based on event type and content
        event_type = event.get("type", "")
        content = str(event.get("content", "")).lower()
        
        if "success" in content:
            impact["satisfaction"] = 0.2
            impact["dominance"] = 0.1
        elif "failure" in content:
            impact["frustration"] = 0.2
            impact["dominance"] = -0.1
            
        if "resistance" in content:
            impact["arousal"] = 0.2
            impact["sadism"] = 0.2
        elif "submission" in content:
            impact["satisfaction"] = 0.2
            impact["dominance"] = 0.2
            
        return impact

    def _check_emotional_alignment(self, plan: Plan) -> float:
        """Check how well current emotional state aligns with plan requirements"""
        if not plan.emotional_state_requirements:
            return 1.0
            
        alignment = 0.0
        count = 0
        
        for emotion, required in plan.emotional_state_requirements.items():
            if emotion in self.emotional_state:
                diff = abs(self.emotional_state[emotion] - required)
                alignment += 1.0 - min(1.0, diff * 2)
                count += 1
                
        return alignment / count if count > 0 else 1.0

    async def _check_and_record_adaptation(self, plan: Plan, event: Dict[str, Any]):
        """Check if plan needs adaptation and record history"""
        current_step = plan.steps[plan.current_step_index]
        
        if current_step.progress < self.adaptation_threshold:
            adaptation = {
                "step_id": current_step.id,
                "timestamp": datetime.now(),
                "trigger": "low_progress",
                "original_state": current_step.dict(),
                "context": event
            }
            
            # Adapt step based on performance
            await self._adapt_step(current_step, event)
            
            adaptation["adapted_state"] = current_step.dict()
            plan.adaptation_history.append(adaptation)

    async def _adapt_step(self, step: PlanStep, event: Dict[str, Any]):
        """Adapt a step based on performance and context"""
        # Adjust risk level
        if step.progress < 0.2:
            step.risk_level = min(1.0, step.risk_level * 1.2)
        
        # Adjust success criteria if too strict
        if len(step.success_criteria) > 2 and step.progress < 0.3:
            step.success_criteria = step.success_criteria[:-1]
            
        # Add new manipulation elements if needed
        if step.progress < 0.4:
            new_elements = await self._generate_manipulation_elements(event)
            step.manipulation_elements.extend(new_elements)

    async def _generate_manipulation_elements(self, event: Dict[str, Any]) -> List[str]:
        """Generate new manipulation elements based on context"""
        elements = []
        
        # Add elements based on emotional state
        if self.emotional_state["dominance"] > 0.7:
            elements.append("assert_control")
        if self.emotional_state["sadism"] > 0.6:
            elements.append("exploit_weakness")
            
        return elements

    async def _calculate_plan_success_rate(self, plan: Plan, event: Dict[str, Any]) -> float:
        """Calculate overall plan success rate"""
        # Base success rate
        base_rate = sum(step.progress for step in plan.steps) / len(plan.steps)
        
        # Emotional influence
        emotional_factor = self._check_emotional_alignment(plan)
        
        # Memory influence
        memory_factor = await self._calculate_memory_influence(plan)
        
        # Context influence
        context_factor = self._calculate_context_influence(event)
        
        # Weighted combination
        success_rate = (
            base_rate * (1 - self.emotional_influence_weight - self.memory_influence_weight - self.context_influence_weight) +
            emotional_factor * self.emotional_influence_weight +
            memory_factor * self.memory_influence_weight +
            context_factor * self.context_influence_weight
        )
        
        return min(1.0, max(0.0, success_rate))

    async def _calculate_memory_influence(self, plan: Plan) -> float:
        """Calculate influence of relevant memories"""
        if not self.memory_graph or not plan.memory_context:
            return 0.5
            
        try:
            total_relevance = 0.0
            total_weight = 0.0
            
            for memory in plan.memory_context:
                # Calculate memory weight based on relevance and recency
                relevance = memory.get("relevance", 0.5)
                age = (datetime.now() - datetime.fromisoformat(memory.get("created_at", datetime.now().isoformat()))).total_seconds()
                recency_weight = 1.0 / (1.0 + age / (24 * 3600))  # Decay over 24 hours
                
                # Calculate emotional alignment
                memory_emotions = memory.get("emotional_state", {})
                emotional_alignment = 0.0
                if memory_emotions and plan.emotional_state_requirements:
                    alignments = []
                    for emotion, required in plan.emotional_state_requirements.items():
                        if emotion in memory_emotions:
                            diff = abs(memory_emotions[emotion] - required)
                            alignments.append(1.0 - min(1.0, diff * 2))
                    if alignments:
                        emotional_alignment = sum(alignments) / len(alignments)
                else:
                    emotional_alignment = 0.5
                
                # Calculate connection strength
                connection_strength = len(memory.get("connections", [])) / 10.0  # Normalize by assuming max 10 connections
                
                # Calculate final memory weight
                weight = (relevance * 0.4 + recency_weight * 0.3 + emotional_alignment * 0.2 + connection_strength * 0.1)
                
                total_relevance += relevance * weight
                total_weight += weight
            
            if total_weight > 0:
                return min(1.0, max(0.0, total_relevance / total_weight))
            return 0.5
            
        except Exception as e:
            logger.error(f"Error calculating memory influence: {e}")
            return 0.5

    def _calculate_context_influence(self, event: Dict[str, Any]) -> float:
        """Calculate influence of current context"""
        # Default neutral influence
        influence = 0.5
        
        # Adjust based on event type and content
        event_type = event.get("type", "")
        content = str(event.get("content", "")).lower()
        
        if "positive" in content or "success" in content:
            influence += 0.2
        elif "negative" in content or "failure" in content:
            influence -= 0.2
            
        return min(1.0, max(0.0, influence))

    async def _calculate_plan_risk_level(self, plan: Plan, event: Dict[str, Any]) -> float:
        """Calculate overall plan risk level"""
        # Maximum risk from steps
        base_risk = max(step.risk_level for step in plan.steps)
        
        # Emotional stability factor
        emotional_stability = self.emotional_state.get("emotional_stability", 0.5)
        stability_factor = 1.0 - emotional_stability
        
        # Context risk
        context_risk = self._calculate_context_risk(event)
        
        # Combined risk with weights
        risk_level = (
            base_risk * 0.4 +
            stability_factor * 0.3 +
            context_risk * 0.3
        )
        
        return min(1.0, max(0.0, risk_level))

    def _calculate_context_risk(self, event: Dict[str, Any]) -> float:
        """Calculate risk level from context"""
        risk = 0.5
        
        content = str(event.get("content", "")).lower()
        
        # Increase risk for certain triggers
        risk_triggers = ["resistance", "defiance", "rebellion", "anger"]
        for trigger in risk_triggers:
            if trigger in content:
                risk += 0.2
                
        return min(1.0, max(0.0, risk))

    async def get_plan(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific plan"""
        if not self.initialized:
            await self.initialize()
            
        plan = self.plans.get(plan_id)
        return plan.dict() if plan else None

    async def get_active_plans(self) -> List[Dict[str, Any]]:
        """Get all active plans"""
        if not self.initialized:
            await self.initialize()
            
        active_plans = [plan for plan in self.plans.values() if plan.status == "active"]
        return [plan.dict() for plan in active_plans]

    async def get_plan_step(self, plan_id: str, step_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific step from a plan"""
        if not self.initialized:
            await self.initialize()
            
        plan = self.plans.get(plan_id)
        if not plan:
            return None
            
        step = next((step for step in plan.steps if step.id == step_id), None)
        return step.dict() if step else None

    async def update_plan_step(self, plan_id: str, step_id: str, 
                             updates: Dict[str, Any]) -> bool:
        """Update a specific step in a plan"""
        if not self.initialized:
            await self.initialize()
            
        plan = self.plans.get(plan_id)
        if not plan:
            return False
            
        step = next((step for step in plan.steps if step.id == step_id), None)
        if not step:
            return False
            
        for key, value in updates.items():
            if hasattr(step, key):
                setattr(step, key, value)
                
        plan.last_updated = datetime.now()
        return True

    async def analyze_plan_effectiveness(self, plan_id: str) -> Dict[str, Any]:
        """Analyze the effectiveness of a plan"""
        if not self.initialized:
            await self.initialize()
            
        plan = self.plans.get(plan_id)
        if not plan:
            return {"error": "Plan not found"}
            
        analysis = {
            "plan_id": plan_id,
            "overall_success_rate": plan.success_rate,
            "overall_risk_level": plan.risk_level,
            "completion_rate": len([step for step in plan.steps if step.status == "completed"]) / len(plan.steps),
            "step_analysis": [
                {
                    "step_id": step.id,
                    "type": step.type,
                    "progress": step.progress,
                    "status": step.status,
                    "risk_level": step.risk_level
                }
                for step in plan.steps
            ],
            "duration": (datetime.now() - plan.created_at).total_seconds() / 3600  # hours
        }
        
        return analysis

    async def get_state(self) -> Dict[str, Any]:
        """Get current state of the planner"""
        if not self.initialized:
            await self.initialize()
            
        return {
            "total_plans": len(self.plans),
            "active_plans": len([p for p in self.plans.values() if p.status == "active"]),
            "completed_plans": len([p for p in self.plans.values() if p.status == "completed"]),
            "failed_plans": len([p for p in self.plans.values() if p.status == "failed"]),
            "plans": [plan.dict() for plan in self.plans.values()]
        }

    async def cleanup(self):
        """Cleanup planner resources"""
        if not self.initialized:
            return
            
        try:
            # Save any unsaved changes to player profile
            await self.save_player_profile()
            
            # Save any unsaved changes to active plans
            for plan in self.plans.values():
                if plan.status == "active":
                    await self.save_plan(plan)
            
            # Close memory graph connection
            if self.memory_graph:
                await self.memory_graph.close()
                self.memory_graph = None
            
            # Clear in-memory data
            self.plans.clear()
            self.emotional_state.clear()
            self.observation_patterns.clear()
            
            # Reset initialization state
            self.initialized = False
            
            logger.info(f"Planner cleaned up for user {self.user_id}, conversation {self.conversation_id}")
            
        except Exception as e:
            logger.error(f"Error during planner cleanup: {e}")
            raise

    async def update_player_profile(self, event: Dict[str, Any]):
        """Update player profile based on new observations"""
        if not self.initialized:
            await self.initialize()
            
        # Extract observations from event
        observations = self._extract_observations(event)
        
        # Update profile with new observations
        await self._update_profile_with_observations(observations)
        
        # Generate teasing elements if profile has significant updates
        if self._should_update_teasing_elements():
            await self._generate_teasing_elements()
            
        self.last_profile_update = datetime.now()

    def _extract_observations(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant observations from an event"""
        observations = {
            "kink_preferences": {},
            "physical_preferences": {},
            "personality_preferences": {},
            "language_patterns": {},
            "emotional_triggers": {},
            "interaction_style": {}
        }
        
        content = str(event.get("content", "")).lower()
        
        # Analyze kink preferences
        kink_patterns = {
            "bdsm": ["whip", "chain", "collar", "submission", "dominance"],
            "roleplay": ["fantasy", "scenario", "character", "story"],
            "exhibitionism": ["public", "exposed", "watched", "seen"],
            "voyeurism": ["watching", "observing", "peeking", "spying"]
        }
        
        for kink, patterns in kink_patterns.items():
            for pattern in patterns:
                if pattern in content:
                    observations["kink_preferences"][kink] = observations["kink_preferences"].get(kink, 0) + 0.1
                    
        # Analyze physical preferences
        physical_patterns = {
            "redhead": ["red hair", "ginger", "redhead"],
            "tattooed": ["tattoo", "ink", "tattooed"],
            "tall": ["tall", "height", "towering"],
            "muscular": ["muscle", "strong", "buff", "ripped"]
        }
        
        for trait, patterns in physical_patterns.items():
            for pattern in patterns:
                if pattern in content:
                    observations["physical_preferences"][trait] = observations["physical_preferences"].get(trait, 0) + 0.1
                    
        # Analyze personality preferences
        personality_patterns = {
            "dominant": ["dominant", "controlling", "authoritative"],
            "submissive": ["submissive", "obedient", "yielding"],
            "playful": ["playful", "fun", "teasing"],
            "serious": ["serious", "strict", "formal"]
        }
        
        for trait, patterns in personality_patterns.items():
            for pattern in patterns:
                if pattern in content:
                    observations["personality_preferences"][trait] = observations["personality_preferences"].get(trait, 0) + 0.1
                    
        # Analyze language patterns
        language_patterns = {
            "formal": r"\b(shall|whom|whilst|hence|thus)\b",
            "casual": r"\b(yeah|okay|cool|awesome|gonna)\b",
            "descriptive": r"\b(beautiful|amazing|incredible|wonderful)\b",
            "emotional": r"\b(love|hate|desire|crave|need)\b"
        }
        
        for style, pattern in language_patterns.items():
            matches = len(re.findall(pattern, content))
            if matches > 0:
                observations["language_patterns"][style] = observations["language_patterns"].get(style, 0) + 0.1 * matches
                
        # Analyze emotional triggers
        emotional_patterns = {
            "arousal": ["aroused", "excited", "turned on", "horny", "hot"],
            "fear": ["scared", "afraid", "terrified", "fear", "anxious"],
            "desire": ["want", "desire", "crave", "need", "long"],
            "submission": ["submit", "yield", "surrender", "obey", "please"]
        }
        
        for emotion, patterns in emotional_patterns.items():
            for pattern in patterns:
                if pattern in content:
                    observations["emotional_triggers"][emotion] = observations["emotional_triggers"].get(emotion, 0) + 0.1
                    
        # Analyze interaction style
        interaction_patterns = {
            "aggressive": ["force", "push", "demand", "command"],
            "gentle": ["gentle", "soft", "tender", "careful"],
            "playful": ["tease", "play", "fun", "game"],
            "formal": ["proper", "correct", "appropriate", "formal"]
        }
        
        for style, patterns in interaction_patterns.items():
            for pattern in patterns:
                if pattern in content:
                    observations["interaction_style"][style] = observations["interaction_style"].get(style, 0) + 0.1
                    
        return observations

    async def _update_profile_with_observations(self, observations: Dict[str, Any]):
        """Update player profile with new observations"""
        # Update each category with decay
        for category, new_values in observations.items():
            if not new_values:
                continue
                
            current_values = getattr(self.player_profile, category)
            
            for key, value in new_values.items():
                # Apply decay to existing value
                current_values[key] = current_values.get(key, 0) * 0.9
                # Add new value
                current_values[key] = min(1.0, current_values[key] + value)
                
            setattr(self.player_profile, category, current_values)
            
        # Update confidence levels
        self._update_confidence_levels()
        
        # Record observation pattern
        self._record_observation_pattern(observations)
        
        # Update last updated timestamp
        self.player_profile.last_updated = datetime.now()

    def _update_confidence_levels(self):
        """Update confidence levels for profile elements"""
        for category in ["kink_preferences", "physical_preferences", "personality_preferences",
                        "language_patterns", "emotional_triggers", "interaction_style"]:
            values = getattr(self.player_profile, category)
            if not values:
                continue
                
            # Calculate confidence based on value strength and observation frequency
            for key, value in values.items():
                observation_count = self.observation_patterns.get(f"{category}_{key}", 0)
                confidence = min(1.0, value * (1 + observation_count * 0.1))
                self.player_profile.confidence_levels[f"{category}_{key}"] = confidence

    def _record_observation_pattern(self, observations: Dict[str, Any]):
        """Record observation pattern for frequency analysis"""
        for category, values in observations.items():
            for key, value in values.items():
                if value > 0:
                    pattern_key = f"{category}_{key}"
                    self.observation_patterns[pattern_key] += 1
                    
        # Record the observation in profile
        self.player_profile.observed_patterns.append({
            "timestamp": datetime.now(),
            "observations": observations
        })

    def _should_update_teasing_elements(self) -> bool:
        """Check if teasing elements should be updated"""
        # Update if confidence levels have changed significantly
        for category in ["kink_preferences", "physical_preferences", "personality_preferences"]:
            values = getattr(self.player_profile, category)
            for key, value in values.items():
                confidence = self.player_profile.confidence_levels.get(f"{category}_{key}", 0)
                if confidence > self.profile_update_threshold:
                    return True
        return False

    async def _generate_teasing_elements(self):
        """Generate teasing elements based on player profile"""
        elements = []
        
        # Generate elements based on kink preferences
        for kink, value in self.player_profile.kink_preferences.items():
            confidence = self.player_profile.confidence_levels.get(f"kink_preferences_{kink}", 0)
            if confidence > 0.7:
                elements.append(f"tease_{kink}")
                
        # Generate elements based on physical preferences
        for trait, value in self.player_profile.physical_preferences.items():
            confidence = self.player_profile.confidence_levels.get(f"physical_preferences_{trait}", 0)
            if confidence > 0.7:
                elements.append(f"highlight_{trait}")
                
        # Generate elements based on personality preferences
        for trait, value in self.player_profile.personality_preferences.items():
            confidence = self.player_profile.confidence_levels.get(f"personality_preferences_{trait}", 0)
            if confidence > 0.7:
                elements.append(f"emphasize_{trait}")
                
        # Generate elements based on emotional triggers
        for trigger, value in self.player_profile.emotional_triggers.items():
            confidence = self.player_profile.confidence_levels.get(f"emotional_triggers_{trigger}", 0)
            if confidence > 0.7:
                elements.append(f"trigger_{trigger}")
                
        self.player_profile.teasing_elements = elements

    async def get_profile_insights(self) -> Dict[str, Any]:
        """Get insights from player profile"""
        if not self.initialized:
            await self.initialize()
            
        return {
            "kink_preferences": self.player_profile.kink_preferences,
            "physical_preferences": self.player_profile.physical_preferences,
            "personality_preferences": self.player_profile.personality_preferences,
            "language_patterns": self.player_profile.language_patterns,
            "emotional_triggers": self.player_profile.emotional_triggers,
            "interaction_style": self.player_profile.interaction_style,
            "confidence_levels": self.player_profile.confidence_levels,
            "teasing_elements": self.player_profile.teasing_elements,
            "last_updated": self.player_profile.last_updated
        }

    async def get_teasing_suggestions(self, context: Dict[str, Any]) -> List[str]:
        """Get teasing suggestions based on player profile and context"""
        if not self.initialized:
            await self.initialize()
            
        suggestions = []
        
        # Get relevant teasing elements based on context
        for element in self.player_profile.teasing_elements:
            if self._is_element_relevant(element, context):
                suggestions.append(element)
                
        return suggestions

    def _is_element_relevant(self, element: str, context: Dict[str, Any]) -> bool:
        """Check if a teasing element is relevant to the current context"""
        content = str(context.get("content", "")).lower()
        
        # Check if element matches any patterns in content
        if element.startswith("tease_"):
            kink = element[6:]
            return any(pattern in content for pattern in self._get_kink_patterns(kink))
        elif element.startswith("highlight_"):
            trait = element[10:]
            return any(pattern in content for pattern in self._get_trait_patterns(trait))
        elif element.startswith("emphasize_"):
            trait = element[10:]
            return any(pattern in content for pattern in self._get_personality_patterns(trait))
        elif element.startswith("trigger_"):
            trigger = element[8:]
            return any(pattern in content for pattern in self._get_trigger_patterns(trigger))
            
        return False

    def _get_kink_patterns(self, kink: str) -> List[str]:
        """Get patterns associated with a kink"""
        patterns = {
            "bdsm": ["whip", "chain", "collar", "submission", "dominance"],
            "roleplay": ["fantasy", "scenario", "character", "story"],
            "exhibitionism": ["public", "exposed", "watched", "seen"],
            "voyeurism": ["watching", "observing", "peeking", "spying"]
        }
        return patterns.get(kink, [])

    def _get_trait_patterns(self, trait: str) -> List[str]:
        """Get patterns associated with a physical trait"""
        patterns = {
            "redhead": ["red hair", "ginger", "redhead"],
            "tattooed": ["tattoo", "ink", "tattooed"],
            "tall": ["tall", "height", "towering"],
            "muscular": ["muscle", "strong", "buff", "ripped"]
        }
        return patterns.get(trait, [])

    def _get_personality_patterns(self, trait: str) -> List[str]:
        """Get patterns associated with a personality trait"""
        patterns = {
            "dominant": ["dominant", "controlling", "authoritative"],
            "submissive": ["submissive", "obedient", "yielding"],
            "playful": ["playful", "fun", "teasing"],
            "serious": ["serious", "strict", "formal"]
        }
        return patterns.get(trait, [])

    def _get_trigger_patterns(self, trigger: str) -> List[str]:
        """Get patterns associated with an emotional trigger"""
        patterns = {
            "arousal": ["aroused", "excited", "turned on", "horny"],
            "fear": ["scared", "afraid", "terrified", "fear"],
            "desire": ["want", "desire", "crave", "need"],
            "submission": ["submit", "yield", "surrender", "obey"]
        }
        return patterns.get(trigger, [])

    async def _generate_plan_steps(self, plan_type: str, context: Dict[str, Any] = None) -> List[PlanStep]:
        """Generate plan steps based on plan type and context"""
        steps = []
        
        if plan_type == "emotional_manipulation":
            steps = [
                PlanStep(
                    id="emotional_hook",
                    type="emotional",
                    description="Create emotional connection",
                    success_criteria=["emotional response", "connection"],
                    failure_criteria=["resistance", "rejection"],
                    risk_level=0.3
                ),
                PlanStep(
                    id="emotional_control",
                    type="emotional",
                    description="Establish emotional control",
                    success_criteria=["submission", "dependency"],
                    failure_criteria=["independence", "defiance"],
                    risk_level=0.5
                ),
                PlanStep(
                    id="emotional_dependency",
                    type="emotional",
                    description="Create emotional dependency",
                    success_criteria=["need", "craving", "obsession"],
                    failure_criteria=["detachment", "distance"],
                    risk_level=0.7
                )
            ]
        elif plan_type == "psychological_control":
            steps = [
                PlanStep(
                    id="cognitive_restructuring",
                    type="psychological",
                    description="Alter cognitive patterns",
                    success_criteria=["confusion", "doubt", "uncertainty"],
                    failure_criteria=["clarity", "certainty", "confidence"],
                    risk_level=0.4
                ),
                PlanStep(
                    id="reality_distortion",
                    type="psychological",
                    description="Distort perception of reality",
                    success_criteria=["questioning", "doubt", "trust"],
                    failure_criteria=["skepticism", "resistance", "independence"],
                    risk_level=0.6
                ),
                PlanStep(
                    id="self_concept_erosion",
                    type="psychological",
                    description="Erode sense of self",
                    success_criteria=["identity crisis", "self-doubt", "dependency"],
                    failure_criteria=["self-confidence", "identity", "certainty"],
                    risk_level=0.8
                )
            ]
        elif plan_type == "addiction_manipulation":
            steps = [
                PlanStep(
                    id="pleasure_association",
                    type="behavioral",
                    description="Create pleasure association",
                    success_criteria=["arousal", "enjoyment", "pleasure"],
                    failure_criteria=["disinterest", "boredom", "apathy"],
                    risk_level=0.3
                ),
                PlanStep(
                    id="intermittent_reinforcement",
                    type="behavioral",
                    description="Establish intermittent reinforcement",
                    success_criteria=["anticipation", "craving", "seeking"],
                    failure_criteria=["disinterest", "avoidance", "detachment"],
                    risk_level=0.5
                ),
                PlanStep(
                    id="withdrawal_induction",
                    type="behavioral",
                    description="Induce withdrawal symptoms",
                    success_criteria=["desperation", "need", "obsession"],
                    failure_criteria=["independence", "disinterest", "detachment"],
                    risk_level=0.7
                ),
                PlanStep(
                    id="complete_dependency",
                    type="behavioral",
                    description="Establish complete dependency",
                    success_criteria=["addiction", "obsession", "desperation"],
                    failure_criteria=["independence", "control", "balance"],
                    risk_level=0.9
                )
            ]
            
        # Adjust steps based on context if provided
        if context:
            intensity = context.get("intensity", 0.5)
            for step in steps:
                step.risk_level *= intensity
                
        return steps
