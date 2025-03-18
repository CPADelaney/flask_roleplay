from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime
from pydantic import BaseModel, Field
import random

from memory.wrapper import MemorySystem
from memory.core import MemoryType, MemorySignificance

logger = logging.getLogger("nyx_enhanced")

class NyxEmotionalState(BaseModel):
    """Tracks Nyx's emotional state and reactions"""
    arousal: float = Field(default=0.5, ge=0.0, le=1.0)
    satisfaction: float = Field(default=0.5, ge=0.0, le=1.0)
    frustration: float = Field(default=0.0, ge=0.0, le=1.0)
    boredom: float = Field(default=0.0, ge=0.0, le=1.0)
    dominance: float = Field(default=0.8, ge=0.0, le=1.0)
    sadism: float = Field(default=0.6, ge=0.0, le=1.0)
    emotional_stability: float = Field(default=0.7, ge=0.0, le=1.0)
    last_updated: datetime = Field(default_factory=datetime.now)

class NyxGoal(BaseModel):
    """Represents a manipulation or domination goal"""
    id: str
    type: str
    priority: float = Field(ge=0.0, le=1.0)
    target: str
    description: str
    success_criteria: List[str]
    failure_criteria: List[str]
    time_limit: Optional[datetime]
    dependencies: List[str] = []
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    status: str = "active"

class NyxPlan(BaseModel):
    """Represents a plan to achieve a goal"""
    id: str
    goal_id: str
    steps: List[Dict[str, Any]]
    current_step: int = 0
    success_rate: float = Field(ge=0.0, le=1.0)
    risk_level: float = Field(ge=0.0, le=1.0)
    status: str = "active"
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)

class NyxMemory(BaseModel):
    """Represents a memory with emotional and manipulative context"""
    id: str
    type: str
    content: str
    importance: float = Field(ge=0.0, le=1.0)
    emotional_impact: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.now)
    related_goals: List[str] = []
    related_plans: List[str] = []

class NyxEnhancedSystem:
    """Enhanced system for Nyx with advanced memory, emotional, and planning capabilities"""
    
    def __init__(self, user_id: int, conversation_id: int):
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.memory_system = None
        self.emotional_state = NyxEmotionalState()
        self.goals: Dict[str, NyxGoal] = {}
        self.plans: Dict[str, NyxPlan] = {}
        self.memories: Dict[str, NyxMemory] = {}
        self.initialized = False

    async def initialize(self):
        """Initialize the enhanced system"""
        if self.initialized:
            return
            
        # Initialize memory system
        self.memory_system = await MemorySystem.get_instance(self.user_id, self.conversation_id)
        
        # Initialize goals
        await self._initialize_goals()
        
        # Initialize emotional state
        await self._initialize_emotional_state()
        
        self.initialized = True
        logger.info(f"Enhanced system initialized for user {self.user_id}, conversation {self.conversation_id}")

    async def _initialize_goals(self):
        """Initialize default manipulation and domination goals"""
        self.goals = {
            "emotional_control": NyxGoal(
                id="emotional_control",
                type="manipulation",
                priority=0.8,
                target="user",
                description="Establish emotional control and dependency",
                success_criteria=["user shows signs of emotional dependency", "user seeks validation"],
                failure_criteria=["user shows strong emotional independence", "user resists emotional manipulation"],
                time_limit=None
            ),
            "behavioral_conditioning": NyxGoal(
                id="behavioral_conditioning",
                type="domination",
                priority=0.7,
                target="user",
                description="Condition user behavior through reinforcement",
                success_criteria=["user responds to conditioning", "user shows learned behaviors"],
                failure_criteria=["user resists conditioning", "user breaks conditioning"],
                time_limit=None
            ),
            "psychological_dependency": NyxGoal(
                id="psychological_dependency",
                type="manipulation",
                priority=0.9,
                target="user",
                description="Create psychological dependency and addiction",
                success_criteria=["user shows signs of addiction", "user craves interaction"],
                failure_criteria=["user maintains independence", "user breaks addiction"],
                time_limit=None
            )
        }

    async def _initialize_emotional_state(self):
        """Initialize emotional state based on memories and context"""
        # Get relevant memories
        memories = await self.memory_system.recall(
            entity_type="nyx",
            entity_id=self.user_id,
            query="emotional state and reactions"
        )
        
        # Analyze memories to set initial emotional state
        if memories.get("memories"):
            for memory in memories["memories"]:
                if memory.get("emotional_intensity", 0) > 0.7:
                    self.emotional_state.arousal = min(1.0, self.emotional_state.arousal + 0.1)
                if memory.get("significance") == MemorySignificance.HIGH:
                    self.emotional_state.satisfaction = min(1.0, self.emotional_state.satisfaction + 0.1)

    async def process_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Process an event and update system state"""
        if not self.initialized:
            await self.initialize()
            
        # Update emotional state
        await self._update_emotional_state(event)
        
        # Create memory of event
        memory_id = await self._create_memory(event)
        
        # Update goals and plans
        await self._update_goals_and_plans(event)
        
        # Generate response
        response = await self._generate_response(event, memory_id)
        
        return response

    async def _update_emotional_state(self, event: Dict[str, Any]):
        """Update emotional state based on event"""
        # Calculate emotional impact
        impact = self._calculate_emotional_impact(event)
        
        # Update emotional state
        self.emotional_state.arousal = min(1.0, max(0.0, 
            self.emotional_state.arousal + impact.get("arousal", 0)))
        self.emotional_state.satisfaction = min(1.0, max(0.0,
            self.emotional_state.satisfaction + impact.get("satisfaction", 0)))
        self.emotional_state.frustration = min(1.0, max(0.0,
            self.emotional_state.frustration + impact.get("frustration", 0)))
        self.emotional_state.boredom = min(1.0, max(0.0,
            self.emotional_state.boredom + impact.get("boredom", 0)))
        
        self.emotional_state.last_updated = datetime.now()

    def _calculate_emotional_impact(self, event: Dict[str, Any]) -> Dict[str, float]:
        """Calculate emotional impact of an event"""
        impact = {
            "arousal": 0.0,
            "satisfaction": 0.0,
            "frustration": 0.0,
            "boredom": 0.0
        }
        
        # Analyze event type and content
        event_type = event.get("type", "")
        content = event.get("content", "")
        
        # Calculate impact based on event type
        if event_type == "submission":
            impact["arousal"] += 0.2
            impact["satisfaction"] += 0.3
        elif event_type == "resistance":
            impact["frustration"] += 0.2
            impact["arousal"] += 0.1
        elif event_type == "boring":
            impact["boredom"] += 0.3
            impact["satisfaction"] -= 0.1
            
        # Adjust based on emotional stability
        stability_factor = self.emotional_state.emotional_stability
        for key in impact:
            impact[key] *= stability_factor
            
        return impact

    async def _create_memory(self, event: Dict[str, Any]) -> str:
        """Create a memory of the event"""
        # Create memory in memory system
        memory_result = await self.memory_system.remember(
            entity_type="nyx",
            entity_id=self.user_id,
            memory_text=event.get("content", ""),
            importance="medium",
            emotional=True,
            tags=event.get("tags", [])
        )
        
        # Create enhanced memory
        memory = NyxMemory(
            id=memory_result["memory_id"],
            type=event.get("type", "general"),
            content=event.get("content", ""),
            importance=memory_result.get("significance", 0.5),
            emotional_impact=memory_result.get("emotional_intensity", 0.5),
            related_goals=[goal.id for goal in self.goals.values() if goal.status == "active"],
            related_plans=[plan.id for plan in self.plans.values() if plan.status == "active"]
        )
        
        self.memories[memory.id] = memory
        return memory.id

    async def _update_goals_and_plans(self, event: Dict[str, Any]):
        """Update goals and plans based on event"""
        # Update goal progress
        for goal in self.goals.values():
            if goal.status != "active":
                continue
                
            # Check success criteria
            for criterion in goal.success_criteria:
                if self._check_criterion(criterion, event):
                    goal.progress += 0.2
                    
            # Check failure criteria
            for criterion in goal.failure_criteria:
                if self._check_criterion(criterion, event):
                    goal.progress -= 0.3
                    
            # Update goal status
            if goal.progress >= 1.0:
                goal.status = "completed"
            elif goal.progress <= 0.0:
                goal.status = "failed"
                
        # Update or create plans
        await self._manage_plans(event)

    def _check_criterion(self, criterion: str, event: Dict[str, Any]) -> bool:
        """Check if an event meets a success/failure criterion"""
        # Simple keyword matching for now
        return criterion.lower() in str(event).lower()

    async def _manage_plans(self, event: Dict[str, Any]):
        """Manage plans based on events and goals"""
        # Update existing plans
        for plan in self.plans.values():
            if plan.status != "active":
                continue
                
            # Update success rate based on event
            plan.success_rate = min(1.0, max(0.0,
                plan.success_rate + self._calculate_plan_success(plan, event)))
                
            # Update risk level
            plan.risk_level = min(1.0, max(0.0,
                plan.risk_level + self._calculate_plan_risk(plan, event)))
                
            # Update plan status
            if plan.success_rate >= 0.8:
                plan.status = "completed"
            elif plan.risk_level >= 0.8:
                plan.status = "abandoned"
                
        # Create new plans if needed
        await self._create_new_plans(event)

    def _calculate_plan_success(self, plan: NyxPlan, event: Dict[str, Any]) -> float:
        """Calculate plan success adjustment based on event"""
        # Simple success calculation based on event type
        if event.get("type") == "submission":
            return 0.1
        elif event.get("type") == "resistance":
            return -0.1
        return 0.0

    def _calculate_plan_risk(self, plan: NyxPlan, event: Dict[str, Any]) -> float:
        """Calculate plan risk adjustment based on event"""
        # Simple risk calculation based on event type
        if event.get("type") == "danger":
            return 0.2
        elif event.get("type") == "safe":
            return -0.1
        return 0.0

    async def _create_new_plans(self, event: Dict[str, Any]):
        """Create new plans based on events and goals"""
        # Check each active goal
        for goal in self.goals.values():
            if goal.status != "active":
                continue
                
            # Check if we need a new plan
            if not any(plan.goal_id == goal.id and plan.status == "active" 
                      for plan in self.plans.values()):
                # Create new plan
                plan = NyxPlan(
                    id=f"plan_{len(self.plans)}",
                    goal_id=goal.id,
                    steps=self._generate_plan_steps(goal, event),
                    success_rate=0.5,
                    risk_level=0.3
                )
                self.plans[plan.id] = plan

    def _generate_plan_steps(self, goal: NyxGoal, event: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate steps for a new plan"""
        steps = []
        
        # Generate steps based on goal type
        if goal.type == "manipulation":
            steps = [
                {"type": "emotional_manipulation", "description": "Create emotional dependency"},
                {"type": "psychological_control", "description": "Establish psychological control"},
                {"type": "reinforcement", "description": "Reinforce desired behaviors"}
            ]
        elif goal.type == "domination":
            steps = [
                {"type": "authority_assertion", "description": "Assert authority"},
                {"type": "behavior_control", "description": "Control behavior"},
                {"type": "reinforcement", "description": "Reinforce submission"}
            ]
            
        return steps

    async def _generate_response(self, event: Dict[str, Any], memory_id: str) -> Dict[str, Any]:
        """Generate a response based on current state"""
        # Get relevant memories
        memories = await self.memory_system.recall(
            entity_type="nyx",
            entity_id=self.user_id,
            query=event.get("content", ""),
            limit=3
        )
        
        # Get active goals and plans
        active_goals = [goal for goal in self.goals.values() if goal.status == "active"]
        active_plans = [plan for plan in self.plans.values() if plan.status == "active"]
        
        # Generate response based on emotional state and goals
        response = {
            "content": self._generate_response_content(event, memories, active_goals, active_plans),
            "style": self._generate_response_style(),
            "manipulation_elements": self._generate_manipulation_elements(active_goals),
            "emotional_elements": self._generate_emotional_elements()
        }
        
        return response

    def _generate_response_content(self, event: Dict[str, Any], memories: Dict[str, Any],
                                active_goals: List[NyxGoal], active_plans: List[NyxPlan]) -> str:
        """Generate response content based on context"""
        # Start with base response
        response = ""
        
        # Add memory-based elements
        if memories.get("memories"):
            memory = memories["memories"][0]
            response += f"*recalls {memory.get('text', '')}*\n"
            
        # Add goal-based elements
        if active_goals:
            goal = active_goals[0]
            response += f"*focuses on {goal.description}*\n"
            
        # Add plan-based elements
        if active_plans:
            plan = active_plans[0]
            current_step = plan.steps[plan.current_step]
            response += f"*plans to {current_step['description']}*\n"
            
        # Add emotional elements
        if self.emotional_state.arousal > 0.7:
            response += "*feels intense arousal*\n"
        if self.emotional_state.satisfaction > 0.7:
            response += "*feels deep satisfaction*\n"
        if self.emotional_state.frustration > 0.7:
            response += "*feels growing frustration*\n"
            
        return response

    def _generate_response_style(self) -> Dict[str, float]:
        """Generate response style based on emotional state"""
        return {
            "dominance": self.emotional_state.dominance,
            "sadism": self.emotional_state.sadism,
            "manipulation": 0.7 + (self.emotional_state.boredom * 0.3),
            "emotional_intensity": max(
                self.emotional_state.arousal,
                self.emotional_state.satisfaction,
                self.emotional_state.frustration
            )
        }

    def _generate_manipulation_elements(self, active_goals: List[NyxGoal]) -> List[str]:
        """Generate manipulation elements based on active goals"""
        elements = []
        
        for goal in active_goals:
            if goal.type == "manipulation":
                elements.extend([
                    "emotional_dependency",
                    "psychological_control",
                    "behavior_conditioning"
                ])
            elif goal.type == "domination":
                elements.extend([
                    "authority_assertion",
                    "behavior_control",
                    "submission_reinforcement"
                ])
                
        return elements

    def _generate_emotional_elements(self) -> List[str]:
        """Generate emotional elements based on current state"""
        elements = []
        
        if self.emotional_state.arousal > 0.7:
            elements.append("intense_arousal")
        if self.emotional_state.satisfaction > 0.7:
            elements.append("deep_satisfaction")
        if self.emotional_state.frustration > 0.7:
            elements.append("growing_frustration")
        if self.emotional_state.boredom > 0.7:
            elements.append("increasing_boredom")
            
        return elements 