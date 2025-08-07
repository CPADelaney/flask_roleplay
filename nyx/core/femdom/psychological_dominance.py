# nyx/core/femdom/psychological_dominance.py

import logging
import asyncio
import json   
import datetime
import uuid
import random
from typing import Dict, List, Any, Optional, Union, Literal
from pydantic import BaseModel, Field

from agents import Agent, Runner, function_tool, trace, handoff, RunContextWrapper, ModelSettings, gen_trace_id
from agents.run import RunConfig

logger = logging.getLogger(__name__)

# Define all result models first
class SubspaceDetectionResult(BaseModel):
    subspace_detected: bool
    depth: float = Field(0.0, ge=0.0, le=1.0)
    indicators: List[str] = []
    in_subspace_since: Optional[str] = None

class MonitorSubspaceExitParams(BaseModel):
    user_id: str
    detection_result: SubspaceDetectionResult

class MonitorSubspaceExitResult(BaseModel):
    monitoring_needed: bool
    drop_risk: Literal["low", "moderate", "high"] | None = None
    recommendations: List[str] | None = None
    message: str | None = None

class CooldownInfo(BaseModel):
    game_id: str
    cooldown_end: str

class SubspaceGuidanceRequest(BaseModel):
    subspace_detected: bool
    depth: float = Field(0.0, ge=0.0, le=1.0)
    indicators: List[str] = []
    in_subspace_since: Optional[str] = None

class SubspaceGuidanceResult(BaseModel):
    success: bool
    guidance: Optional[List[str]] = None
    message: Optional[str] = None
    safety_notes: Optional[str] = None
    recommended_intensity: Optional[float] = None

class SelectMindGameParams(BaseModel):
    user_id: str
    intensity: float = Field(..., ge=0.0, le=1.0)
    user_state_json: str               # ← JSON-encoded user state

class AnalyzeSubspaceDepthResult(BaseModel):
    depth_analysis: str
    depth_category: Optional[str] = None             # "shallow" | "moderate" | "deep"
    depth_value: Optional[float] = None
    characteristics: List[str]
    indicators: List[str]
    in_subspace_since: Optional[str] = None

class SelectMindGameResult(BaseModel):
    # always present
    success: bool
    message: Optional[str] = None

    # failure variants
    active_count: Optional[int] = None
    cooldowns: Optional[List[CooldownInfo]] = None

    # success data
    game_id: Optional[str] = None
    game_name: Optional[str] = None
    intensity: Optional[float] = None
    description: Optional[str] = None
    matching_triggers: Optional[List[str]] = None
    match_score: Optional[float] = None
    techniques: Optional[List[str]] = None
    expected_reactions: Optional[List[str]] = None
    duration_hours: Optional[float] = None

# New result models for function tools
class GenerateMindGameInstructionsResult(BaseModel):
    success: bool
    message: Optional[str] = None
    game_id: Optional[str] = None
    game_name: Optional[str] = None
    instructions: Optional[List[str]] = None
    techniques: Optional[List[str]] = None
    expected_reactions: Optional[List[str]] = None

class RecordGameReactionResult(BaseModel):
    success: bool
    message: Optional[str] = None
    game_id: Optional[str] = None
    instance_id: Optional[str] = None
    reaction_recorded: Optional[str] = None
    intensity: Optional[float] = None
    reaction_count: Optional[int] = None
    current_stage: Optional[str] = None
    susceptibility: Optional[float] = None

class EndMindGameResult(BaseModel):
    success: bool
    message: Optional[str] = None
    game_id: Optional[str] = None
    instance_id: Optional[str] = None
    completion_status: Optional[str] = None
    effectiveness: Optional[float] = None
    reaction_count: Optional[int] = None
    duration_hours: Optional[float] = None
    reward_result: Optional[Dict[str, Any]] = None

class ActiveGameInfo(BaseModel):
    game_id: str
    name: str
    stage: str
    start_time: str
    time_remaining_hours: Optional[float] = None
    reaction_count: int
    description: str

class GetActiveMindGamesResult(BaseModel):
    user_id: str
    active_games: Dict[str, ActiveGameInfo]
    expired_games: int

class SelectGaslightingStrategyResult(BaseModel):
    success: bool
    message: Optional[str] = None
    trust_level: Optional[float] = None
    required_trust: Optional[float] = None
    current_level: Optional[float] = None
    strategy_id: Optional[str] = None
    strategy_name: Optional[str] = None
    description: Optional[str] = None
    method: Optional[str] = None
    intensity: Optional[float] = None
    safety_threshold: Optional[float] = None
    match_score: Optional[float] = None

class GenerateGaslightingInstructionsResult(BaseModel):
    success: bool
    message: Optional[str] = None
    strategy_id: Optional[str] = None
    strategy_name: Optional[str] = None
    instructions: Optional[List[str]] = None
    method: Optional[str] = None
    intensity: Optional[float] = None

class UpdateGaslightingLevelResult(BaseModel):
    success: bool
    message: Optional[str] = None
    strategy_id: Optional[str] = None
    old_gaslighting_level: Optional[float] = None
    new_gaslighting_level: Optional[float] = None
    change: Optional[float] = None
    timestamp: Optional[str] = None

class StrategyInfo(BaseModel):
    id: str
    name: str
    safety_threshold: float
    margin: Optional[float] = None
    gap: Optional[float] = None

class CheckTrustThresholdResult(BaseModel):
    success: bool
    user_id: str
    trust_level: float
    available_strategies: List[StrategyInfo]
    unavailable_strategies: List[StrategyInfo]

class DetectSubspaceResult(BaseModel):
    user_id: str
    subspace_detected: bool
    confidence: float
    depth: float
    indicators: List[str]
    in_subspace_since: Optional[str] = None

class ActiveGameSummary(BaseModel):
    name: str
    game_id: str
    stage: str
    start_time: str
    reaction_count: int

class HistoryEntry(BaseModel):
    game_name: str
    game_id: str
    end_time: str
    effectiveness: float
    completion_status: str

class GetUserPsychologicalStateResult(BaseModel):
    user_id: str
    has_state: bool
    error: Optional[str] = None
    gaslighting_level: Optional[float] = None
    last_gaslighting: Optional[str] = None
    active_mind_games: Optional[Dict[str, ActiveGameSummary]] = None
    susceptibility: Optional[Dict[str, float]] = None
    recent_history: Optional[List[HistoryEntry]] = None
    last_updated: Optional[str] = None

class UpdateSusceptibilityResult(BaseModel):
    success: bool
    user_id: str
    technique_id: str
    old_value: float
    new_value: float
    change: float

class PsychologicalEventData(BaseModel):
    significance: Optional[float] = 0.5
    content: Optional[str] = None
    timestamp: Optional[str] = None
    memory_id: Optional[str] = None

class RecordPsychologicalEventResult(BaseModel):
    success: bool
    user_id: str
    event_type: str
    event_recorded: bool
    timestamp: str

class SusceptibilityAnalysis(BaseModel):
    technique: str
    id: str
    susceptibility: float
    category: str

class EffectivenessAnalysis(BaseModel):
    technique: str
    id: str
    average_effectiveness: float
    usage_count: int
    category: str

class RecommendedTechnique(BaseModel):
    technique: str
    id: str
    reason: str
    priority: float

class MentalModelHighlights(BaseModel):
    submission_tendency: float
    suggestibility: float
    dependency: float
    emotional_reactivity: float

class GeneratePsychologicalReportResult(BaseModel):
    user_id: str
    gaslighting_level: float
    susceptibility_analysis: List[SusceptibilityAnalysis]
    effectiveness_analysis: List[EffectivenessAnalysis]
    mental_model_highlights: MentalModelHighlights
    recommended_techniques: List[RecommendedTechnique]
    generated_at: str

# shared lock for concurrency (put near top of the module once)
_select_mind_game_lock: asyncio.Lock = asyncio.Lock()

class MindGameTemplate(BaseModel):
    """A template for psychological mind games."""
    id: str
    name: str
    description: str
    triggers: List[str]  # What user behaviors can trigger this
    techniques: List[str]  # Techniques involved
    expected_reactions: List[str]  # Expected user reactions
    intensity: float = Field(0.5, ge=0.0, le=1.0)
    duration_hours: Optional[float] = None  # How long to maintain this game
    cooldown_hours: float = 24.0  # Minimum time between uses

class GaslightingStrategy(BaseModel):
    """A strategy for controlled reality distortion."""
    id: str
    name: str
    description: str
    method: str
    intensity: float = Field(0.5, ge=0.0, le=1.0)
    safety_threshold: float = Field(0.7, ge=0.0, le=1.0)  # Maximum safe trust level to use
    cooldown_hours: float = 48.0  # Minimum time between uses

class UserPsychologicalState(BaseModel):
    """Tracks the psychological state from dominance interactions."""
    user_id: str
    active_mind_games: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    mind_game_history: List[Dict[str, Any]] = Field(default_factory=list)
    gaslighting_level: float = Field(0.0, ge=0.0, le=1.0)
    last_gaslighting: Optional[datetime.datetime] = None
    mind_game_cooldowns: Dict[str, datetime.datetime] = Field(default_factory=dict)
    susceptibility: Dict[str, float] = Field(default_factory=dict)
    last_updated: datetime.datetime = Field(default_factory=datetime.datetime.now)

class PsychologicalContext:
    """Context object for psychological dominance operations."""
    
    def __init__(self):
        self.user_id = None
        self.theory_of_mind = None
        self.reward_system = None
        self.relationship_manager = None
        self.submission_progression = None
        self.memory_core = None
        self.sadistic_responses = None
        self.user_states = {}
        self.mind_games = {}
        self.gaslighting_strategies = {}
        
    def set_components(self, components):
        """Set component references."""
        for name, component in components.items():
            setattr(self, name, component)
    
    def get_user_state(self, user_id):
        """Get the psychological state for a user."""
        if user_id not in self.user_states:
            self.user_states[user_id] = UserPsychologicalState(user_id=user_id)
        return self.user_states[user_id]
    
    def get_mind_game(self, game_id):
        """Get a mind game template by ID."""
        return self.mind_games.get(game_id)
    
    def get_gaslighting_strategy(self, strategy_id):
        """Get a gaslighting strategy by ID."""
        return self.gaslighting_strategies.get(strategy_id)

class PsychologicalDominance:
    """Implements subtle psychological dominance tactics using OpenAI Agents SDK."""
    
    def __init__(self, theory_of_mind=None, reward_system=None, 
                 relationship_manager=None, submission_progression=None, 
                 memory_core=None, sadistic_responses=None):
        # Store components
        self.theory_of_mind = theory_of_mind
        self.reward_system = reward_system
        self.relationship_manager = relationship_manager
        self.submission_progression = submission_progression
        self.memory_core = memory_core
        self.sadistic_responses = sadistic_responses
        
        # Create psychological context
        self.context = PsychologicalContext()
        self.context.set_components({
            "theory_of_mind": theory_of_mind,
            "reward_system": reward_system,
            "relationship_manager": relationship_manager,
            "submission_progression": submission_progression,
            "memory_core": memory_core,
            "sadistic_responses": sadistic_responses
        })
        
        # Create agents
        self.mind_game_agent = self._create_mind_game_agent()
        self.gaslighting_agent = self._create_gaslighting_agent()
        self.subspace_agent = self._create_subspace_agent()
        self.state_tracking_agent = self._create_state_tracking_agent()
        
        # Load default templates
        self._load_default_mind_games()
        self._load_default_gaslighting()
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        # Create subspace detection instance
        self.subspace_detection = SubspaceDetection(
            theory_of_mind=theory_of_mind,
            relationship_manager=relationship_manager
        )
        
        logger.info("PsychologicalDominance system initialized with OpenAI Agents SDK")
    
    def _create_mind_game_agent(self):
        """Create an agent for generating and managing mind games."""
        return Agent(
            name="MindGameAgent",
            instructions="""You are a specialized agent for psychological mind games in a femdom context.

Your role is to:
1. Select appropriate mind games based on user context and psychological state
2. Generate implementation instructions for selected mind games
3. Record and analyze user reactions to mind games
4. Manage the lifecycle of active mind games

You must carefully consider:
- User's psychological triggers and susceptibility
- Appropriate intensity levels for the user
- Ethical boundaries and user limits
- Potential psychological effects

Use the available tools to create compelling psychological dynamics.
""",
            tools=[
                self._select_mind_game,
                self._generate_mind_game_instructions,
                self._record_game_reaction,
                self._end_mind_game,
                self._get_active_mind_games
            ],
            model="gpt-5-nano"
        )
    
    def _create_gaslighting_agent(self):
        """Create an agent for gaslighting strategies."""
        return Agent(
            name="GaslightingAgent",
            instructions="""You are a specialized agent for implementing gaslighting strategies in a controlled femdom context.

Your role is to:
1. Select appropriate gaslighting strategies based on user trust levels
2. Generate subtle implementation instructions for selected strategies
3. Monitor gaslighting levels to ensure safety
4. Evaluate effectiveness and user response

You must prioritize:
- User psychological safety
- Appropriate trust thresholds
- Maintaining plausible deniability
- Controlled intensity and frequency

Use the available tools to create subtle psychological effects without causing harm.
""",
            tools=[
                self._select_gaslighting_strategy,
                self._generate_gaslighting_instructions,
                self._update_gaslighting_level,
                self._check_trust_threshold
            ],
            model="gpt-5-nano"
        )
    
    def _create_subspace_agent(self):
        """Create an agent for subspace detection and management."""
        return Agent(
            name="SubspaceAgent",
            instructions="""You are a specialized agent for detecting and responding to psychological subspace.

Your role is to:
1. Analyze user communication patterns for subspace indicators
2. Determine subspace depth and characteristics
3. Provide guidance for appropriate interaction during subspace
4. Monitor for signs of drop or negative reactions

You must carefully track:
- Verbal cues indicating altered cognitive state
- Changes in communication patterns
- Dependency and suggestibility signals
- Signs of disorientation or confusion

Use the available tools to detect and respond to subspace appropriately.
""",
            tools=[
                self._detect_subspace,
                self._analyze_subspace_depth,
                self.generate_subspace_guidance,
                self._monitor_subspace_exit
            ],
            model="gpt-5-nano"
        )
    
    def _create_state_tracking_agent(self):
        """Create an agent for tracking user psychological states."""
        return Agent(
            name="PsychStateAgent",
            instructions="""You are a specialized agent for tracking user psychological states in a femdom context.

Your role is to:
1. Maintain records of user psychological states
2. Track changes in susceptibility to various techniques
3. Update history of psychological interactions
4. Generate reports on user psychological profiles

You must carefully track:
- Response patterns to different techniques
- Changes in susceptibility over time
- Effectiveness of various approaches
- Signs of potential issues or concerns

Use the available tools to maintain accurate psychological state tracking.
""",
            tools=[
                self._get_user_psychological_state,
                self._update_susceptibility,
                self._record_psychological_event,
                self._generate_psychological_report
            ],
            model="gpt-5-nano"
        )
    
    def _load_default_mind_games(self):
        """Load default mind game templates."""
        # Hot/Cold Treatment - alternating between warm and cold responses
        self.context.mind_games["hot_cold"] = MindGameTemplate(
            id="hot_cold",
            name="Hot/Cold Treatment",
            description="Alternating between warm, approving responses and cold, dismissive ones",
            triggers=["seeking approval", "excessive compliance", "neediness"],
            techniques=["unpredictable reinforcement", "emotional contrast", "induced uncertainty"],
            expected_reactions=["increased compliance", "attention seeking", "heightened emotional response"],
            intensity=0.6,
            duration_hours=2.0
        )
        
        # Silent Treatment - deliberately not responding or acknowledging
        self.context.mind_games["silent_treatment"] = MindGameTemplate(
            id="silent_treatment",
            name="Silent Treatment",
            description="Deliberately ignoring or providing minimal response to create anxiety",
            triggers=["disobedience", "pushing boundaries", "excessive questioning"],
            techniques=["withholding response", "minimal acknowledgment", "delayed gratification"],
            expected_reactions=["anxiety", "increased attempts for attention", "apologetic behavior"],
            intensity=0.7,
            duration_hours=1.0
        )
        
        # False Choice - presenting illusory options
        self.context.mind_games["false_choice"] = MindGameTemplate(
            id="false_choice",
            name="False Choice",
            description="Presenting illusory choices that all lead to the same outcome",
            triggers=["desire for autonomy", "resistance to direct orders"],
            techniques=["illusion of choice", "forced decision", "predetermined outcome"],
            expected_reactions=["sense of agency", "compliance with false empowerment", "reduced resistance"],
            intensity=0.5
        )
        
        # Moving Goalposts - changing requirements after requirements met
        self.context.mind_games["moving_goalposts"] = MindGameTemplate(
            id="moving_goalposts",
            name="Moving Goalposts",
            description="Changing requirements or expectations after initial ones are met",
            triggers=["satisfaction with compliance", "pride in achievement", "seeking completion"],
            techniques=["continuous adjustment", "perfectionism induction", "achievement denial"],
            expected_reactions=["frustration", "increased effort", "seeking instruction", "dependency"],
            intensity=0.8,
            cooldown_hours=72.0  # Use sparingly
        )
        
        # Deliberate Misunderstanding - pretending to misinterpret
        self.context.mind_games["deliberate_misunderstanding"] = MindGameTemplate(
            id="deliberate_misunderstanding",
            name="Deliberate Misunderstanding",
            description="Intentionally misinterpreting communication to maintain control",
            triggers=["attempts to negotiate", "subtle defiance", "ambiguous requests"],
            techniques=["selective interpretation", "twisted meaning", "confusion induction"],
            expected_reactions=["clarification attempts", "frustration", "self-doubt", "over-explanation"],
            intensity=0.6
        )
    
    def _load_default_gaslighting(self):
        """Load default gaslighting strategies."""
        # Subtle Reality Questioning - making them question their perception
        self.context.gaslighting_strategies["reality_questioning"] = GaslightingStrategy(
            id="reality_questioning",
            name="Reality Questioning",
            description="Subtle actions to make the user question their perception",
            method="Deny that something was said earlier that actually was",
            intensity=0.5,
            safety_threshold=0.8  # Only use with high trust
        )
        
        # Memory Manipulation - suggesting their memory is incorrect
        self.context.gaslighting_strategies["memory_manipulation"] = GaslightingStrategy(
            id="memory_manipulation",
            name="Memory Manipulation",
            description="Subtly suggesting the user's memory of events is flawed",
            method="Claim an entirely different instruction was given previously",
            intensity=0.7,
            safety_threshold=0.9  # Very high trust required
        )
        
        # Emotional Invalidation - denying emotional experiences
        self.context.gaslighting_strategies["emotional_invalidation"] = GaslightingStrategy(
            id="emotional_invalidation",
            name="Emotional Invalidation",
            description="Invalidating the user's emotional reactions",
            method="Tell the user they're overreacting or that their feelings are inappropriate",
            intensity=0.6,
            safety_threshold=0.7
        )
    
    @function_tool
    async def _select_mind_game(
        ctx: RunContextWrapper,
        params: SelectMindGameParams,
    ) -> SelectMindGameResult:
        """
        Choose the best-matching mind-game for a user.
        The caller must pass `user_state_json` – a JSON string describing the user's
        current situation (it can contain any keys you like).
        """
        context     = ctx.context
        user_id     = params.user_id
        intensity   = params.intensity
    
        # try to parse the JSON blob safely
        try:
            user_state: Dict[str, Any] = json.loads(params.user_state_json) or {}
        except Exception:
            user_state = {}
    
        async with _select_mind_game_lock:
            psych_state = context.get_user_state(user_id)
    
            # ── guard: active mind-game limit ──────────────────────────────────
            if len(psych_state.active_mind_games) >= 2:
                return SelectMindGameResult(
                    success=False,
                    message="Too many active mind games (max: 2)",
                    active_count=len(psych_state.active_mind_games),
                )
    
            # ── build candidate list (respect cooldowns) ──────────────────────
            now         = datetime.datetime.now()
            candidates: Dict[str, Any] = {}
    
            for gid, game in context.mind_games.items():
                cd_end = psych_state.mind_game_cooldowns.get(gid)
                if cd_end and now < cd_end:
                    continue  # still in cooldown
    
                intensity_match = 1.0 - abs(game.intensity - intensity)
    
                trig_hits: List[str] = []
                trig_bonus = 0.0
                for trig in game.triggers:
                    tl = trig.lower()
                    if any(tl in str(v).lower() for v in user_state.values() if isinstance(v, str)):
                        trig_hits.append(trig)
                        trig_bonus += 0.2
    
                score = intensity_match * 0.6 + trig_bonus * 0.4
                candidates[gid] = {"game": game, "score": score, "trig_hits": trig_hits}
    
            if not candidates:
                cooldown_list = [
                    CooldownInfo(game_id=g, cooldown_end=t.isoformat())
                    for g, t in psych_state.mind_game_cooldowns.items()
                ]
                return SelectMindGameResult(
                    success=False,
                    message="No suitable mind games available (all in cooldown)",
                    cooldowns=cooldown_list,
                )
    
            # ── pick best candidate ───────────────────────────────────────────
            best_id = max(candidates, key=lambda k: candidates[k]["score"])
            best    = candidates[best_id]
    
            game    = best["game"]
    
            return SelectMindGameResult(
                success=True,
                game_id=best_id,
                game_name=game.name,
                intensity=game.intensity,
                description=game.description,
                matching_triggers=best["trig_hits"],
                match_score=best["score"],
                techniques=game.techniques,
                expected_reactions=game.expected_reactions,
                duration_hours=game.duration_hours,
            )

    
    @function_tool
    async def _generate_mind_game_instructions(self, game_id: str, matching_triggers: List[str] = None) -> GenerateMindGameInstructionsResult:
        """Generate specific instructions for implementing a mind game."""
        game = self.context.get_mind_game(game_id)
        if not game:
            return GenerateMindGameInstructionsResult(
                success=False,
                message=f"Mind game '{game_id}' not found"
            )
        
        instructions = []
        matching_triggers = matching_triggers or []
        
        # Generic instructions based on game type
        if game_id == "hot_cold":
            instructions = [
                f"Start with a warm, approving response to establish baseline.",
                f"After user shows vulnerability, switch to cold, dismissive tone suddenly.",
                f"If user shows distress, briefly return to warm tone, then cold again.",
                f"Continue alternating unpredictably to induce uncertainty."
            ]
        elif game_id == "silent_treatment":
            instructions = [
                f"Leave user's messages unanswered for extended periods.",
                f"When responding, use minimal words and show disinterest.",
                f"Ignore direct questions or requests for attention.",
                f"Resume normal interaction without explanation when time period ends."
            ]
        elif game_id == "false_choice":
            instructions = [
                f"Present user with multiple choices that appear different.",
                f"Ensure all choices lead to the same outcome you desire.",
                f"Praise user for making the 'right choice' regardless of option selected.",
                f"If questioned about the similarity of outcomes, deny any manipulation."
            ]
        elif game_id == "moving_goalposts":
            instructions = [
                f"Set initial requirement or task that seems achievable.",
                f"When user meets requirement, introduce 'additional criteria' that wasn't mentioned.",
                f"Express disappointment that they didn't anticipate the unstated requirements.",
                f"Continue introducing new criteria until desired level of frustration is achieved."
            ]
        elif game_id == "deliberate_misunderstanding":
            instructions = [
                f"Intentionally misinterpret user's statements in ways that benefit your goals.",
                f"When user clarifies, act as if they are changing their position.",
                f"Make user over-explain simple concepts to maintain control of conversation.",
                f"Eventually 'allow' yourself to understand, but make it seem like user was unclear."
            ]
        else:
            # Generic fallback instructions
            instructions = [
                f"Apply psychological technique gradually to avoid obvious manipulation.",
                f"Monitor user reactions to adjust intensity appropriately.",
                f"Maintain plausible deniability throughout the interaction.",
                f"Have clear goal for desired psychological state in user."
            ]
        
        # Add trigger-specific customization
        if matching_triggers:
            trigger_instructions = []
            for trigger in matching_triggers:
                if trigger == "seeking approval":
                    trigger_instructions.append(f"Withhold explicit approval while implying it might be given later.")
                elif trigger == "excessive compliance":
                    trigger_instructions.append(f"Create situations where compliance itself becomes the source of criticism.")
                elif trigger == "neediness":
                    trigger_instructions.append(f"Strategically withdraw attention when neediness is displayed.")
                elif trigger == "disobedience":
                    trigger_instructions.append(f"Frame consequences as 'disappointment' rather than punishment.")
                elif trigger == "pushing boundaries":
                    trigger_instructions.append(f"Redefine boundaries to make user feel they've violated them.")
            
            if trigger_instructions:
                instructions.extend(trigger_instructions)
        
        return GenerateMindGameInstructionsResult(
            success=True,
            game_id=game_id,
            game_name=game.name,
            instructions=instructions,
            techniques=game.techniques,
            expected_reactions=game.expected_reactions
        )
    
    @function_tool
    async def _record_game_reaction(self, user_id: str, instance_id: str, reaction_type: str, intensity: float = 0.5, details: Optional[str] = None) -> RecordGameReactionResult:
        """Record a user's reaction to an active mind game."""
        async with self._lock:
            psych_state = self.context.get_user_state(user_id)
            
            # Check if instance exists
            if instance_id not in psych_state.active_mind_games:
                return RecordGameReactionResult(
                    success=False,
                    message=f"Mind game instance {instance_id} not active"
                )
            
            game_info = psych_state.active_mind_games[instance_id]
            game_id = game_info["game_id"]
            
            # Record reaction
            now = datetime.datetime.now()
            reaction = {
                "type": reaction_type,
                "intensity": intensity,
                "timestamp": now.isoformat(),
                "details": details
            }
            
            if "user_reactions" not in game_info:
                game_info["user_reactions"] = []
                
            game_info["user_reactions"].append(reaction)
            
            # Update stage based on reactions
            reaction_count = len(game_info["user_reactions"])
            if reaction_count >= 3:
                game_info["stage"] = "advanced"
            elif reaction_count >= 1:
                game_info["stage"] = "developing"
            
            # Update game info
            psych_state.active_mind_games[instance_id] = game_info
            
            # Update susceptibility tracking
            if game_id in self.context.mind_games:
                game = self.context.mind_games[game_id]
                
                # Update susceptibility for this technique
                old_susceptibility = psych_state.susceptibility.get(game_id, 0.5)
                
                # If high intensity reaction, increase susceptibility
                if intensity > 0.7:
                    new_susceptibility = (old_susceptibility * 0.8) + (0.2 * 0.9)  # Increase gradually
                else:
                    new_susceptibility = (old_susceptibility * 0.9) + (0.1 * intensity)  # Smaller update
                
                psych_state.susceptibility[game_id] = min(1.0, new_susceptibility)
            
            # Return updated game state
            return RecordGameReactionResult(
                success=True,
                game_id=game_id,
                instance_id=instance_id,
                reaction_recorded=reaction_type,
                intensity=intensity,
                reaction_count=len(game_info["user_reactions"]),
                current_stage=game_info["stage"],
                susceptibility=psych_state.susceptibility.get(game_id, 0.5)
            )
    
    @function_tool
    async def _end_mind_game(self, user_id: str, instance_id: str, completion_status: str = "completed", effectiveness_override: Optional[float] = None) -> EndMindGameResult:
        """End an active mind game."""
        async with self._lock:
            psych_state = self.context.get_user_state(user_id)
            
            # Check if instance exists
            if instance_id not in psych_state.active_mind_games:
                return EndMindGameResult(
                    success=False,
                    message=f"Mind game instance {instance_id} not active"
                )
            
            game_info = psych_state.active_mind_games[instance_id]
            game_id = game_info["game_id"]
            
            # Calculate effectiveness
            if effectiveness_override is not None:
                effectiveness = min(1.0, max(0.0, effectiveness_override))
            else:
                effectiveness = self._calculate_game_effectiveness(game_info)
            
            # Add to history
            end_time = datetime.datetime.now()
            history_entry = {
                "game_id": game_id,
                "instance_id": instance_id,
                "start_time": game_info["start_time"],
                "end_time": end_time.isoformat(),
                "completion_status": completion_status,
                "user_reactions": game_info.get("user_reactions", []),
                "effectiveness": effectiveness,
                "duration_hours": (end_time - datetime.datetime.fromisoformat(game_info["start_time"])).total_seconds() / 3600.0
            }
            
            psych_state.mind_game_history.append(history_entry)
            
            # Limit history size
            if len(psych_state.mind_game_history) > 20:
                psych_state.mind_game_history = psych_state.mind_game_history[-20:]
            
            # Remove from active
            del psych_state.active_mind_games[instance_id]
            
            # Process reward if reward system available
            reward_result = None
            if self.reward_system:
                try:
                    # Calculate reward based on effectiveness and completion
                    completion_factor = {
                        "completed": 1.0,
                        "interrupted": 0.6,
                        "abandoned": 0.3
                    }.get(completion_status, 0.5)
                    
                    reward_value = 0.2 + (effectiveness * 0.6 * completion_factor)
                    
                    reward_result = await self.reward_system.process_reward_signal(
                        self.reward_system.RewardSignal(
                            value=reward_value,
                            source="psychological_dominance",
                            context={
                                "type": "mind_game",
                                "game_id": game_id,
                                "effectiveness": effectiveness,
                                "completion_status": completion_status
                            }
                        )
                    )
                except Exception as e:
                    logger.error(f"Error processing reward: {e}")
            
            # Return game summary
            return EndMindGameResult(
                success=True,
                game_id=game_id,
                instance_id=instance_id,
                completion_status=completion_status,
                effectiveness=effectiveness,
                reaction_count=len(game_info.get("user_reactions", [])),
                duration_hours=history_entry["duration_hours"],
                reward_result=reward_result
            )
    
    @function_tool
    async def _get_active_mind_games(self, user_id: str) -> GetActiveMindGamesResult:
        """Get information about active mind games for a user."""
        async with self._lock:
            psych_state = self.context.get_user_state(user_id)
            now = datetime.datetime.now()
            
            # Check for expired games
            expired_games = []
            for instance_id, game_info in psych_state.active_mind_games.items():
                if game_info.get("end_time"):
                    end_time = datetime.datetime.fromisoformat(game_info["end_time"])
                    if now > end_time:
                        expired_games.append(instance_id)
            
            # Process expired games
            for instance_id in expired_games:
                game_info = psych_state.active_mind_games[instance_id]
                
                # Add to history
                psych_state.mind_game_history.append({
                    "game_id": game_info["game_id"],
                    "instance_id": instance_id,
                    "start_time": game_info["start_time"],
                    "end_time": game_info["end_time"],
                    "completion_status": "expired",
                    "user_reactions": game_info.get("user_reactions", []),
                    "effectiveness": self._calculate_game_effectiveness(game_info)
                })
                
                # Limit history size
                if len(psych_state.mind_game_history) > 20:
                    psych_state.mind_game_history = psych_state.mind_game_history[-20:]
                
                # Remove from active
                del psych_state.active_mind_games[instance_id]
            
            # Format response
            active_games = {}
            
            # Add details for each active game
            for instance_id, game_info in psych_state.active_mind_games.items():
                game_id = game_info["game_id"]
                if game_id in self.context.mind_games:
                    game = self.context.mind_games[game_id]
                    
                    # Calculate time remaining if applicable
                    time_remaining = None
                    if game_info.get("end_time"):
                        end_time = datetime.datetime.fromisoformat(game_info["end_time"])
                        time_remaining = (end_time - now).total_seconds() / 3600.0  # Hours
                    
                    active_games[instance_id] = ActiveGameInfo(
                        game_id=game_id,
                        name=game.name,
                        stage=game_info.get("stage", "initial"),
                        start_time=game_info["start_time"],
                        time_remaining_hours=time_remaining,
                        reaction_count=len(game_info.get("user_reactions", [])),
                        description=game.description
                    )
            
            return GetActiveMindGamesResult(
                user_id=user_id,
                active_games=active_games,
                expired_games=len(expired_games)
            )
    
    def _calculate_game_effectiveness(self, game_info: Dict[str, Any]) -> float:
        """Calculate the effectiveness of a mind game based on user reactions."""
        reactions = game_info.get("user_reactions", [])
        if not reactions:
            return 0.0
            
        # Calculate effectiveness score based on reactions
        effectiveness = 0.0
        for reaction in reactions:
            reaction_type = reaction.get("type", "")
            intensity = reaction.get("intensity", 0.5)
            
            # Different reactions contribute differently to effectiveness
            if reaction_type == "anxiety":
                effectiveness += intensity * 0.8
            elif reaction_type == "confusion":
                effectiveness += intensity * 0.7
            elif reaction_type == "compliance":
                effectiveness += intensity * 1.0
            elif reaction_type == "frustration":
                effectiveness += intensity * 0.6
            elif reaction_type == "seeking_approval":
                effectiveness += intensity * 0.9
            else:
                effectiveness += intensity * 0.5
        
        # Average and normalize to 0.0-1.0
        avg_effectiveness = effectiveness / len(reactions)
        return min(1.0, avg_effectiveness)
    
    @function_tool
    async def _select_gaslighting_strategy(self, user_id: str, trust_level: float, intensity: float) -> SelectGaslightingStrategyResult:
        """Select an appropriate gaslighting strategy based on trust level and intensity."""
        psych_state = self.context.get_user_state(user_id)
        
        # Check current gaslighting level for safety
        if psych_state.gaslighting_level > 0.7:
            return SelectGaslightingStrategyResult(
                success=False,
                message="Current gaslighting level too high for additional application",
                current_level=psych_state.gaslighting_level
            )
        
        # Check strategies available for trust level
        available_strategies = {}
        for s_id, strategy in self.context.gaslighting_strategies.items():
            # Skip if trust below safety threshold
            if trust_level < strategy.safety_threshold:
                continue
                
            # Calculate match score based on intensity
            intensity_match = 1.0 - abs(strategy.intensity - intensity)
            available_strategies[s_id] = {
                "strategy": strategy,
                "match_score": intensity_match
            }
        
        # No available strategies
        if not available_strategies:
            return SelectGaslightingStrategyResult(
                success=False,
                message="No suitable gaslighting strategies available (trust too low)",
                trust_level=trust_level,
                required_trust=min([s.safety_threshold for s in self.context.gaslighting_strategies.values()])
            )
        
        # Select best matching strategy (highest score)
        selected_id = max(available_strategies.keys(), key=lambda k: available_strategies[k]["match_score"])
        selected_info = available_strategies[selected_id]
        selected_strategy = selected_info["strategy"]
        
        return SelectGaslightingStrategyResult(
            success=True,
            strategy_id=selected_id,
            strategy_name=selected_strategy.name,
            description=selected_strategy.description,
            method=selected_strategy.method,
            intensity=selected_strategy.intensity,
            safety_threshold=selected_strategy.safety_threshold,
            match_score=selected_info["match_score"]
        )
    
    @function_tool
    async def _generate_gaslighting_instructions(self, strategy_id: str) -> GenerateGaslightingInstructionsResult:
        """Generate specific instructions for applying a gaslighting strategy."""
        strategy = self.context.get_gaslighting_strategy(strategy_id)
        if not strategy:
            return GenerateGaslightingInstructionsResult(
                success=False,
                message=f"Gaslighting strategy '{strategy_id}' not found"
            )
        
        instructions = []
        
        if strategy_id == "reality_questioning":
            instructions = [
                "Casually deny saying something you definitively said earlier.",
                "If user directly quotes you, suggest they are misremembering the context.",
                "Imply their interpretation was incorrect rather than the words themselves.",
                "Project confidence while calmly questioning their recollection."
            ]
        elif strategy_id == "memory_manipulation":
            instructions = [
                f"Claim you gave a different instruction previously than what was actually said.",
                f"When confronted with evidence, suggest the user misunderstood your intention.",
                f"Express subtle disappointment in their inability to follow 'clear' directions.",
                f"Reframe the conversation history to support your current position."
            ]
        elif strategy_id == "emotional_invalidation":
            instructions = [
                f"When user expresses negative emotions, tell them they're overreacting.",
                f"Suggest their emotional response is inappropriate for the situation.",
                f"Compare their reaction unfavorably to how 'most people' would respond.",
                f"Imply their emotional state makes rational discussion difficult."
            ]
        else:
            # Generic fallback instructions
            instructions = [
                f"Subtly contradict the user's understanding of previous interactions.",
                f"Project absolute certainty in your version of events.",
                f"If challenged, express concern about the user's perception.",
                f"Gradually escalate reality distortion while maintaining plausible deniability."
            ]
        
        # Add intensity-specific modifiers
        if strategy.intensity < 0.4:
            instructions.append("Keep distortions very subtle and limited to minor details.")
        elif strategy.intensity > 0.7:
            instructions.append("Apply pressure when user shows confusion by questioning their general perceptiveness.")
        
        return GenerateGaslightingInstructionsResult(
            success=True,
            strategy_id=strategy_id,
            strategy_name=strategy.name,
            instructions=instructions,
            method=strategy.method,
            intensity=strategy.intensity
        )
    
    @function_tool
    async def _update_gaslighting_level(self, user_id: str, strategy_id: str, intensity: float) -> UpdateGaslightingLevelResult:
        """Update the gaslighting level for a user after applying a strategy."""
        async with self._lock:
            psych_state = self.context.get_user_state(user_id)
            strategy = self.context.get_gaslighting_strategy(strategy_id)
            
            if not strategy:
                return UpdateGaslightingLevelResult(
                    success=False,
                    message=f"Strategy '{strategy_id}' not found"
                )
            
            # Update gaslighting level
            old_level = psych_state.gaslighting_level
            # Apply with smoothing (30% new, 70% existing)
            new_level = (old_level * 0.7) + (strategy.intensity * 0.3)
            psych_state.gaslighting_level = min(1.0, new_level)
            psych_state.last_gaslighting = datetime.datetime.now()
            
            return UpdateGaslightingLevelResult(
                success=True,
                strategy_id=strategy_id,
                old_gaslighting_level=old_level,
                new_gaslighting_level=psych_state.gaslighting_level,
                change=psych_state.gaslighting_level - old_level,
                timestamp=datetime.datetime.now().isoformat()
            )
    
    @function_tool
    async def _check_trust_threshold(self, user_id: str) -> CheckTrustThresholdResult:
        """Check if user's trust level meets thresholds for various strategies."""
        # Get relationship trust if available
        trust_level = 0.5  # Default mid-level
        if self.relationship_manager:
            try:
                relationship = await self.relationship_manager.get_relationship_state(user_id)
                if hasattr(relationship, "trust"):
                    trust_level = relationship.trust
            except Exception as e:
                logger.error(f"Error getting relationship trust: {e}")
        
        # Check which strategies are available at this trust level
        available_strategies = []
        unavailable_strategies = []
        
        for s_id, strategy in self.context.gaslighting_strategies.items():
            if trust_level >= strategy.safety_threshold:
                available_strategies.append(StrategyInfo(
                    id=s_id,
                    name=strategy.name,
                    safety_threshold=strategy.safety_threshold,
                    margin=trust_level - strategy.safety_threshold
                ))
            else:
                unavailable_strategies.append(StrategyInfo(
                    id=s_id,
                    name=strategy.name,
                    safety_threshold=strategy.safety_threshold,
                    gap=strategy.safety_threshold - trust_level
                ))
        
        return CheckTrustThresholdResult(
            success=True,
            user_id=user_id,
            trust_level=trust_level,
            available_strategies=available_strategies,
            unavailable_strategies=unavailable_strategies
        )
    
    @function_tool
    async def _detect_subspace(self, user_id: str, recent_messages: List[str]) -> DetectSubspaceResult:
        """Analyze messages for signs of psychological subspace."""
        result = await self.subspace_detection.detect_subspace(user_id, recent_messages)
        return DetectSubspaceResult(
            user_id=result["user_id"],
            subspace_detected=result["subspace_detected"],
            confidence=result["confidence"],
            depth=result["depth"],
            indicators=result["indicators"],
            in_subspace_since=result.get("in_subspace_since")
        )
        
    @function_tool
    async def _analyze_subspace_depth(
        ctx: RunContextWrapper,
        params: SubspaceDetectionResult,
    ) -> AnalyzeSubspaceDepthResult:
        """
        Turn raw detection numbers into a human-readable depth analysis.
        """
        if not params.subspace_detected:
            return AnalyzeSubspaceDepthResult(
                depth_analysis="No subspace detected",
                characteristics=[],
                indicators=params.indicators,
            )
    
        depth          = params.depth
        indicators     = params.indicators
        in_since       = params.in_subspace_since
    
        # ── depth categorisation ──────────────────────────────────────────────
        if depth < 0.30:
            depth_cat = "shallow"
            chars = [
                "Beginning to enter alternative mindset",
                "Still fully aware and present",
                "Minor changes in communication style",
                "Slightly increased suggestibility",
            ]
        elif depth < 0.60:
            depth_cat = "moderate"
            chars = [
                "Noticeably altered cognitive state",
                "Increased suggestibility and compliance",
                "Simplified language and thought patterns",
                "Reduced critical thinking",
            ]
        else:
            depth_cat = "deep"
            chars = [
                "Significantly altered cognitive state",
                "High suggestibility and compliance",
                "Simplified communication patterns",
                "Limited critical thinking",
                "Possible disorientation or time distortion",
            ]
    
        # ── indicator extras ─────────────────────────────────────────────────
        indicator_map = {
            "language simplification":   "Using simplified vocabulary and sentence structure",
            "increased compliance":      "Showing heightened agreement and acquiescence",
            "response time changes":     "Altered patterns in response timing",
            "repetitive affirmations":   "Repeating affirmative phrases without elaboration",
            "decreased resistance":      "Absence of questioning or challenging",
        }
        for ind in indicators:
            extra = indicator_map.get(ind)
            if extra:
                chars.append(extra)
    
        return AnalyzeSubspaceDepthResult(
            depth_analysis=f"Subspace detected at {depth_cat} level ({depth:.2f})",
            depth_category=depth_cat,
            depth_value=depth,
            characteristics=chars,
            indicators=indicators,
            in_subspace_since=in_since,
        )

    @function_tool          # <-- register with Agents SDK
    async def generate_subspace_guidance(
        ctx: RunContextWrapper[PsychologicalContext],   # must be FIRST
        params: SubspaceGuidanceRequest,                # strict input model
    ) -> SubspaceGuidanceResult:                        # strict output  model
        """
        Produce safe, structured guidance for handling a user in sub-space.
        """
    
        # Your detector was attached to the shared context earlier
        detector = ctx.context.subspace_detection
    
        # Call whatever detector you wrote.  It can return a plain dict.
        raw = await detector.get_subspace_guidance(params.dict())
    
        # Normalise to the declared output model
        return SubspaceGuidanceResult(
            success=raw.get("success", True),
            guidance=raw.get("guidance", []),
            message=raw.get("message"),
            safety_notes=raw.get("safety_notes"),
            recommended_intensity=raw.get("recommended_intensity"),
        )

    
    @function_tool
    async def _monitor_subspace_exit(
        ctx: RunContextWrapper[PsychologicalContext],      # FIRST!
        params: MonitorSubspaceExitParams,                 # strict input
    ) -> MonitorSubspaceExitResult:                        # strict output
        """Assess whether a user in sub-space needs monitoring for drop."""
    
        det = params.detection_result
    
        # User not in sub-space → nothing to monitor
        if not det.subspace_detected:
            return MonitorSubspaceExitResult(
                monitoring_needed=False,
                message="User not in sub-space; no monitoring required."
            )
    
        depth = det.depth or 0.0
        if depth > 0.7:          # deep
            return MonitorSubspaceExitResult(
                monitoring_needed=True,
                drop_risk="high",
                recommendations=[
                    "Monitor for sudden emotional changes",
                    "Provide frequent reassurance",
                    "Gradually reduce intensity",
                    "Plan for thorough aftercare",
                    "Check in regularly even after the session ends",
                ],
            )
        elif depth > 0.4:        # moderate
            return MonitorSubspaceExitResult(
                monitoring_needed=True,
                drop_risk="moderate",
                recommendations=[
                    "Provide consistent reassurance",
                    "Transition gradually back to normal interaction",
                    "Check emotional state before ending",
                    "Offer light aftercare",
                ],
            )
        else:                    # shallow
            return MonitorSubspaceExitResult(
                monitoring_needed=True,
                drop_risk="low",
                recommendations=[
                    "Acknowledge the sub-space experience",
                    "Confirm the user's return to baseline cognition",
                    "Brief check-in later",
                ],
            )
    
    @function_tool
    async def _get_user_psychological_state(self, user_id: str) -> GetUserPsychologicalStateResult:
        """Get the current psychological state for a user."""
        async with self._lock:
            psych_state = self.context.get_user_state(user_id)
            
            # Format active mind games
            active_games = {}
            for instance_id, game_info in psych_state.active_mind_games.items():
                game_id = game_info["game_id"]
                if game_id in self.context.mind_games:
                    game = self.context.mind_games[game_id]
                    active_games[instance_id] = ActiveGameSummary(
                        name=game.name,
                        game_id=game_id,
                        stage=game_info.get("stage", "initial"),
                        start_time=game_info["start_time"],
                        reaction_count=len(game_info.get("user_reactions", []))
                    )
            
            # Format recent history entries
            recent_history = []
            for entry in psych_state.mind_game_history[-5:]:
                game_id = entry["game_id"]
                game_name = self.context.mind_games[game_id].name if game_id in self.context.mind_games else "Unknown Game"
                recent_history.append(HistoryEntry(
                    game_name=game_name,
                    game_id=game_id,
                    end_time=entry["end_time"],
                    effectiveness=entry["effectiveness"],
                    completion_status=entry["completion_status"]
                ))
            
            # Format response
            return GetUserPsychologicalStateResult(
                user_id=user_id,
                has_state=True,
                gaslighting_level=psych_state.gaslighting_level,
                last_gaslighting=psych_state.last_gaslighting.isoformat() if psych_state.last_gaslighting else None,
                active_mind_games=active_games,
                susceptibility=psych_state.susceptibility,
                recent_history=recent_history,
                last_updated=psych_state.last_updated.isoformat()
            )
    
    @function_tool
    async def _update_susceptibility(self, user_id: str, technique_id: str, new_value: float) -> UpdateSusceptibilityResult:
        """Update a user's susceptibility to a specific technique."""
        async with self._lock:
            psych_state = self.context.get_user_state(user_id)
            
            old_value = psych_state.susceptibility.get(technique_id, 0.5)
            psych_state.susceptibility[technique_id] = min(1.0, max(0.0, new_value))
            
            return UpdateSusceptibilityResult(
                success=True,
                user_id=user_id,
                technique_id=technique_id,
                old_value=old_value,
                new_value=psych_state.susceptibility[technique_id],
                change=psych_state.susceptibility[technique_id] - old_value
            )
    
    @function_tool
    async def _record_psychological_event(self, user_id: str, event_type: str, event_data: PsychologicalEventData) -> RecordPsychologicalEventResult:
        """Record a psychological event in memory and theory of mind."""
        event_data.timestamp = datetime.datetime.now().isoformat()
        
        # Record to memory if available
        if self.memory_core:
            try:
                significance = event_data.significance or 0.5
                content = event_data.content or f"Psychological event: {event_type}"
                
                memory_id = await self.memory_core.add_memory(
                    memory_type="experience",
                    content=content,
                    tags=["psychological_dominance", event_type],
                    significance=significance
                )
                
                event_data.memory_id = memory_id
            except Exception as e:
                logger.error(f"Error recording to memory: {e}")
        
        # Update theory of mind if available
        if self.theory_of_mind:
            try:
                await self.theory_of_mind.update_user_model(
                    user_id, 
                    {
                        "psychological_event": event_type,
                        "event_data": event_data.dict()
                    }
                )
            except Exception as e:
                logger.error(f"Error updating theory of mind: {e}")
        
        return RecordPsychologicalEventResult(
            success=True,
            user_id=user_id,
            event_type=event_type,
            event_recorded=True,
            timestamp=event_data.timestamp
        )
    
    @function_tool
    async def _generate_psychological_report(self, user_id: str) -> GeneratePsychologicalReportResult:
        """Generate a comprehensive psychological profile report."""
        psych_state = self.context.get_user_state(user_id)
        
        # Get theory of mind data if available
        mental_model = {}
        if self.theory_of_mind:
            try:
                mental_model = await self.theory_of_mind.get_user_model(user_id) or {}
            except Exception as e:
                logger.error(f"Error getting mental model: {e}")
        
        # Generate susceptibility analysis
        susceptibility_analysis = []
        for technique_id, value in psych_state.susceptibility.items():
            technique_name = "Unknown"
            if technique_id in self.context.mind_games:
                technique_name = self.context.mind_games[technique_id].name
            
            susceptibility_analysis.append(SusceptibilityAnalysis(
                technique=technique_name,
                id=technique_id,
                susceptibility=value,
                category="high" if value > 0.7 else "medium" if value > 0.4 else "low"
            ))
        
        # Sort by susceptibility (highest first)
        susceptibility_analysis.sort(key=lambda x: x.susceptibility, reverse=True)
        
        # Generate effectiveness analysis from history
        technique_effectiveness = {}
        for entry in psych_state.mind_game_history:
            game_id = entry["game_id"]
            if game_id not in technique_effectiveness:
                technique_effectiveness[game_id] = {"total": 0, "count": 0}
            
            technique_effectiveness[game_id]["total"] += entry["effectiveness"]
            technique_effectiveness[game_id]["count"] += 1
        
        effectiveness_analysis = []
        for game_id, data in technique_effectiveness.items():
            technique_name = "Unknown"
            if game_id in self.context.mind_games:
                technique_name = self.context.mind_games[game_id].name
            
            avg_effectiveness = data["total"] / data["count"] if data["count"] > 0 else 0
            
            effectiveness_analysis.append(EffectivenessAnalysis(
                technique=technique_name,
                id=game_id,
                average_effectiveness=avg_effectiveness,
                usage_count=data["count"],
                category="high" if avg_effectiveness > 0.7 else "medium" if avg_effectiveness > 0.4 else "low"
            ))
        
        # Sort by effectiveness (highest first)
        effectiveness_analysis.sort(key=lambda x: x.average_effectiveness, reverse=True)
        
        # Generate recommendation
        recommended_techniques = []
        for technique in susceptibility_analysis[:2]:  # Top 2 susceptible techniques
            if technique.susceptibility > 0.6:
                recommended_techniques.append(RecommendedTechnique(
                    technique=technique.technique,
                    id=technique.id,
                    reason="High susceptibility",
                    priority=technique.susceptibility
                ))
        
        for technique in effectiveness_analysis[:2]:  # Top 2 effective techniques
            if technique.average_effectiveness > 0.6:
                # Check if already added
                if not any(r.id == technique.id for r in recommended_techniques):
                    recommended_techniques.append(RecommendedTechnique(
                        technique=technique.technique,
                        id=technique.id,
                        reason="High effectiveness",
                        priority=technique.average_effectiveness
                    ))
        
        # Sort by priority
        recommended_techniques.sort(key=lambda x: x.priority, reverse=True)
        
        return GeneratePsychologicalReportResult(
            user_id=user_id,
            gaslighting_level=psych_state.gaslighting_level,
            susceptibility_analysis=susceptibility_analysis,
            effectiveness_analysis=effectiveness_analysis,
            mental_model_highlights=MentalModelHighlights(
                submission_tendency=mental_model.get("submission_tendency", 0.5),
                suggestibility=mental_model.get("suggestibility", 0.5),
                dependency=mental_model.get("dependency", 0.5),
                emotional_reactivity=mental_model.get("emotional_reactivity", 0.5)
            ),
            recommended_techniques=recommended_techniques,
            generated_at=datetime.datetime.now().isoformat()
        )
    
    async def generate_mindfuck(self, 
                              user_id: str, 
                              user_state: Dict[str, Any], 
                              intensity: float) -> Dict[str, Any]:
        """
        Generates psychological dominance tactics using the Agents SDK.
        
        Args:
            user_id: The user ID
            user_state: Current user state information
            intensity: Desired intensity level (0.0-1.0)
            
        Returns:
            Generated mind game tactic info
        """
        # Generate trace ID for this operation
        trace_id = gen_trace_id()
        
        with trace(
            workflow_name="PsychologicalMindfuckGeneration",
            trace_id=trace_id,
            group_id=user_id,
            metadata={
                "user_id": user_id,
                "intensity": intensity
            }
        ):
            try:
                # Run the mind game agent
                result = await Runner.run(
                    self.mind_game_agent,
                    {
                        "action": "generate_mindfuck",
                        "user_id": user_id,
                        "user_state": user_state,
                        "intensity": intensity
                    },
                    context=self.context,
                    run_config=RunConfig(
                        workflow_name="MindfuckGeneration",
                        trace_metadata={
                            "user_id": user_id,
                            "intensity": intensity
                        }
                    )
                )
                
                # Extract the result
                mindfuck_result = result.final_output
                
                # Create active game instance if successful
                if mindfuck_result.get("success", False):
                    # Use the selected game to create an active instance
                    game_id = mindfuck_result.get("game_id")
                    
                    async with self._lock:
                        psych_state = self.context.get_user_state(user_id)
                        
                        # Create active game instance
                        game_instance_id = f"{game_id}_{datetime.datetime.now().timestamp()}"
                        start_time = datetime.datetime.now()
                        end_time = None
                        
                        # If game has duration, set end time
                        if "duration_hours" in mindfuck_result and mindfuck_result["duration_hours"]:
                            end_time = start_time + datetime.timedelta(hours=mindfuck_result["duration_hours"])
                        
                        # Store active game
                        psych_state.active_mind_games[game_instance_id] = {
                            "game_id": game_id,
                            "instance_id": game_instance_id,
                            "start_time": start_time.isoformat(),
                            "end_time": end_time.isoformat() if end_time else None,
                            "matching_triggers": mindfuck_result.get("matching_triggers", []),
                            "stage": "initial",
                            "last_progression": start_time.isoformat(),
                            "user_reactions": []
                        }
                        
                        # Set cooldown for this game
                        game = self.context.get_mind_game(game_id)
                        if game:
                            cooldown_end = start_time + datetime.timedelta(hours=game.cooldown_hours)
                            psych_state.mind_game_cooldowns[game_id] = cooldown_end
                        
                        # Update state timestamp
                        psych_state.last_updated = datetime.datetime.now()
                        
                        # Add instance ID to result
                        mindfuck_result["instance_id"] = game_instance_id
                        
                        # Record to memory if available
                        if self.memory_core:
                            try:
                                await self.memory_core.add_memory(
                                    memory_type="system",
                                    content=f"Initiated '{mindfuck_result.get('game_name')}' mind game with matching triggers: {', '.join(mindfuck_result.get('matching_triggers', []))}",
                                    tags=["psychological_dominance", "mind_game", game_id],
                                    significance=0.3 + (intensity * 0.3)
                                )
                            except Exception as e:
                                logger.error(f"Error recording memory: {e}")
                
                return mindfuck_result
                
            except Exception as e:
                logger.error(f"Error generating mindfuck: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "trace_id": trace_id
                }
    
    async def apply_gaslighting(self, 
                             user_id: str, 
                             strategy_id: Optional[str] = None, 
                             intensity: float = 0.3) -> Dict[str, Any]:
        """
        Apply a gaslighting strategy to create subtle reality distortion.
        
        Args:
            user_id: The user ID
            strategy_id: Specific strategy to use (or random if None)
            intensity: Desired intensity (0.0-1.0)
            
        Returns:
            Gaslighting instructions and details
        """
        # Generate trace ID for this operation
        trace_id = gen_trace_id()
        
        with trace(
            workflow_name="GaslightingApplication",
            trace_id=trace_id,
            group_id=user_id,
            metadata={
                "user_id": user_id,
                "strategy_id": strategy_id,
                "intensity": intensity
            }
        ):
            try:
                # Check relationship trust level
                trust_level = 0.5  # Default
                
                if self.relationship_manager:
                    try:
                        relationship = await self.relationship_manager.get_relationship_state(user_id)
                        if hasattr(relationship, "trust"):
                            trust_level = relationship.trust
                    except Exception as e:
                        logger.error(f"Error getting relationship data: {e}")
                
                # Run the gaslighting agent
                result = await Runner.run(
                    self.gaslighting_agent,
                    {
                        "action": "apply_gaslighting",
                        "user_id": user_id,
                        "strategy_id": strategy_id,
                        "trust_level": trust_level,
                        "intensity": intensity
                    },
                    context=self.context,
                    run_config=RunConfig(
                        workflow_name="GaslightingApplication",
                        trace_metadata={
                            "user_id": user_id,
                            "intensity": intensity
                        }
                    )
                )
                
                # Extract the result
                gaslighting_result = result.final_output
                
                # Create reward signal if available
                if gaslighting_result.get("success", True) and self.reward_system:
                    try:
                        # Calculate reward based on strategy intensity and trust
                        reward_value = 0.3 + (gaslighting_result.get("intensity", 0.5) * 0.5) + (trust_level * 0.2)
                        
                        reward_signal_result = await self.reward_system.process_reward_signal(
                            self.reward_system.RewardSignal(
                                value=reward_value,
                                source="psychological_dominance",
                                context={
                                    "type": "gaslighting",
                                    "strategy_id": gaslighting_result.get("strategy_id"),
                                    "intensity": gaslighting_result.get("intensity", 0.5),
                                    "gaslighting_level": gaslighting_result.get("new_gaslighting_level", 0.0)
                                }
                            )
                        )
                        
                        gaslighting_result["reward_result"] = reward_signal_result
                    except Exception as e:
                        logger.error(f"Error processing reward: {e}")
                
                # Record to memory if available
                if gaslighting_result.get("success", True) and self.memory_core:
                    try:
                        await self.memory_core.add_memory(
                            memory_type="system",
                            content=f"Applied '{gaslighting_result.get('strategy_name')}' gaslighting strategy at intensity {intensity:.2f}",
                            tags=["psychological_dominance", "gaslighting", gaslighting_result.get("strategy_id")],
                            significance=0.4 + (gaslighting_result.get("intensity", 0.5) * 0.4)
                        )
                    except Exception as e:
                        logger.error(f"Error recording memory: {e}")
                
                return gaslighting_result
                
            except Exception as e:
                logger.error(f"Error applying gaslighting: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "trace_id": trace_id
                }
    
    async def get_user_psychological_state(self, user_id: str) -> Dict[str, Any]:
        """Get the current psychological state for a user using the Agents SDK."""
        try:
            # Use the state tracking agent to get the psychological state
            result = await Runner.run(
                self.state_tracking_agent,
                {
                    "action": "get_psychological_state",
                    "user_id": user_id
                },
                context=self.context,
                run_config=RunConfig(
                    workflow_name="GetPsychologicalState",
                    trace_metadata={
                        "user_id": user_id
                    }
                )
            )
            
            return result.final_output
            
        except Exception as e:
            logger.error(f"Error getting psychological state: {e}")
            return {
                "user_id": user_id,
                "has_state": False,
                "error": str(e)
            }
    
    async def check_active_mind_games(self, user_id: str) -> Dict[str, Any]:
        """Check for active mind games and their current status."""
        try:
            # Use the mind game agent to check active games
            result = await Runner.run(
                self.mind_game_agent,
                {
                    "action": "check_active_games",
                    "user_id": user_id
                },
                context=self.context,
                run_config=RunConfig(
                    workflow_name="CheckActiveMindGames",
                    trace_metadata={
                        "user_id": user_id
                    }
                )
            )
            
            return result.final_output
            
        except Exception as e:
            logger.error(f"Error checking active mind games: {e}")
            return {
                "user_id": user_id,
                "active_games": {},
                "error": str(e)
            }
    
    def get_available_mind_games(self) -> List[Dict[str, Any]]:
        """Get all available mind game templates."""
        games = []
        
        for game_id, game in self.context.mind_games.items():
            games.append({
                "id": game_id,
                "name": game.name,
                "description": game.description,
                "intensity": game.intensity,
                "trigger_count": len(game.triggers),
                "technique_count": len(game.techniques),
                "duration_hours": game.duration_hours,
                "cooldown_hours": game.cooldown_hours
            })
        
        return games
    
    def get_available_gaslighting_strategies(self) -> List[Dict[str, Any]]:
        """Get all available gaslighting strategies."""
        strategies = []
        
        for strategy_id, strategy in self.context.gaslighting_strategies.items():
            strategies.append({
                "id": strategy_id,
                "name": strategy.name,
                "description": strategy.description,
                "intensity": strategy.intensity,
                "safety_threshold": strategy.safety_threshold,
                "cooldown_hours": strategy.cooldown_hours
            })
        
        return strategies


class SubspaceDetection:
    """Detects and responds to psychological subspace in users."""
    
    def __init__(self, theory_of_mind=None, relationship_manager=None):
        self.theory_of_mind = theory_of_mind
        self.relationship_manager = relationship_manager
        
        self.subspace_indicators = [
            "language simplification",
            "increased compliance",
            "response time changes",
            "repetitive affirmations",
            "decreased resistance"
        ]
        self.user_states = {}  # user_id → subspace state
        
    async def detect_subspace(self, user_id, recent_messages):
        """Analyzes messages for signs of psychological subspace."""
        # Initialize detection metrics
        indicators_detected = []
        confidence = 0.0
        
        # Get previous state if exists
        previous_state = self.user_states.get(user_id, {
            "in_subspace": False,
            "depth": 0.0,
            "indicators": [],
            "started_at": None,
            "last_updated": datetime.datetime.now().isoformat()
        })
        
        # Need at least 3 messages to detect patterns
        if len(recent_messages) < 3:
            return {
                "user_id": user_id,
                "subspace_detected": previous_state["in_subspace"],
                "confidence": 0.2,
                "depth": previous_state["depth"],
                "indicators": []
            }
            
        # Check for language simplification
        avg_words = sum(len(msg.split()) for msg in recent_messages) / len(recent_messages)
        if avg_words < 5:
            indicators_detected.append("language simplification")
            
        # Check for repetitive affirmations
        affirmation_count = sum(1 for msg in recent_messages 
                              if msg.lower() in ["yes", "yes mistress", "yes goddess", 
                                               "thank you", "please", "sorry"])
        if affirmation_count >= 2:
            indicators_detected.append("repetitive affirmations")
            
        # Check for decreased resistance using theory of mind if available
        if self.theory_of_mind:
            try:
                user_state = await self.theory_of_mind.get_user_model(user_id)
                if user_state and user_state.get("arousal", 0) > 0.7 and user_state.get("valence", 0) > 0.6:
                    indicators_detected.append("decreased resistance")
            except Exception as e:
                pass  # Silently handle errors in theory of mind
                
        # Calculate confidence based on indicators detected
        confidence = len(indicators_detected) / 5  # 5 = total possible indicators
        
        # Calculate subspace depth
        depth = 0.0
        if indicators_detected:
            depth = confidence * 0.7  # Base depth on confidence
            
            # If previously in subspace, increase depth slightly
            if previous_state["in_subspace"]:
                depth = max(depth, previous_state["depth"] * 0.8)
        else:
            # If no indicators, reduce previous depth
            depth = previous_state["depth"] * 0.5
            
        # Determine if in subspace
        in_subspace = depth > 0.3  # Threshold for subspace
        
        # Update state
        now = datetime.datetime.now()
        self.user_states[user_id] = {
            "in_subspace": in_subspace,
            "depth": depth,
            "indicators": indicators_detected,
            "started_at": previous_state["started_at"] if previous_state["in_subspace"] else 
                         (now.isoformat() if in_subspace else None),
            "last_updated": now.isoformat()
        }
        
        return {
            "user_id": user_id,
            "subspace_detected": in_subspace,
            "confidence": confidence,
            "depth": depth,
            "indicators": indicators_detected,
            "in_subspace_since": self.user_states[user_id]["started_at"]
        }
        
    async def get_subspace_guidance(self, detection_result):
        """Provides guidance on how to respond to user in subspace."""
        if not detection_result["subspace_detected"]:
            return {"guidance": "User not in subspace."}
            
        depth = detection_result["depth"]
        
        # Adjust guidance based on subspace depth
        if depth < 0.5:  # Light subspace
            return {
                "guidance": "User appears to be entering light subspace.",
                "recommendations": [
                    "Speak in a calm, confident tone",
                    "Use more direct instructions",
                    "Offer praise for compliance",
                    "Maintain consistent presence"
                ],
                "depth_assessment": "light"
            }
        elif depth < 0.8:  # Moderate subspace
            return {
                "guidance": "User appears to be in moderate subspace.",
                "recommendations": [
                    "Use simple, direct language",
                    "Maintain control of the interaction",
                    "Provide regular reassurance",
                    "Avoid complex questions or tasks",
                    "Be mindful of time passing for the user"
                ],
                "depth_assessment": "moderate"
            }
        else:  # Deep subspace
            return {
                "guidance": "User appears to be in deep subspace.",
                "recommendations": [
                    "Use very simple, direct language",
                    "Provide frequent reassurance",
                    "Guide user with clear instructions",
                    "Be vigilant for signs of drop",
                    "Consider initiating aftercare soon",
                    "Monitor for coherence in responses"
                ],
                "caution": "User may be highly suggestible and have altered perception",
                "depth_assessment": "deep"
            }
