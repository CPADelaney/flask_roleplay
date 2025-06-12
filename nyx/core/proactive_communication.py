# nyx/core/proactive_communication.py

import asyncio
import datetime
import logging
import random
import time
import uuid
import math
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import json

from pydantic import BaseModel, Field
from enum import Enum

# Import OpenAI Agents SDK components
from agents import (
    Agent, 
    Runner, 
    ModelSettings, 
    function_tool, 
    handoff, 
    GuardrailFunctionOutput,
    InputGuardrail,
    OutputGuardrail,
    trace,
    RunContextWrapper,
    RunConfig
)
from agents.exceptions import MaxTurnsExceeded, ModelBehaviorError

logger = logging.getLogger(__name__)

# =============== Pydantic Models ===============

class _GenIntentUserIn(BaseModel):
    """Strict input-wrapper for generate_intent_for_user"""
    user_id: str
    user_data: Dict[str, Any]
    motivation_options: Dict[str, float]

    model_config = {"extra": "forbid"}

class _EvalRelOut(BaseModel):
    user_id: str
    relationship_score: float
    communication_appropriateness: float
    days_since_contact: int
    approaching_milestones: List[str]
    suggested_frequency: str
    max_messages_per_week: int

    model_config = {"extra": "forbid"}  



class _GenMsgContentIn(BaseModel):
    intent: Dict[str, Any]
    context: Dict[str, Any]
    model_config = {"extra": "forbid"}

class _GenMsgContentOut(BaseModel):
    message_content: str
    tone_used: str
    key_points: List[str]
    motivation_reflected: str
    context_referenced: Dict[str, Any]
    model_config = {"extra": "forbid"}

class _EvalTimingIn(BaseModel):
    intent: Dict[str, Any]
    current_context: Dict[str, Any]
    timing_config: Dict[str, Any]
    model_config = {"extra": "forbid"}

class _EvalTimingOut(BaseModel):
    should_send_now: bool
    timing_score: float
    reasoning: str
    suggested_delay_minutes: Optional[int] = None
    context_factors: Dict[str, Any]
    model_config = {"extra": "forbid"}

class _GenIntentActionIn(BaseModel):
    action: Dict[str, Any]
    user_data: Dict[str, Any]
    emotional_state: Dict[str, Any]
    model_config = {"extra": "forbid"}

class _ReflectCommsIn(BaseModel):
    intents: List[Dict[str, Any]]
    focus: str
    user_id: Optional[str] = None
    time_period: Optional[str] = None
    model_config = {"extra": "forbid"}

class _ReflectCommsOut(BaseModel):
    reflection_text: str
    identified_patterns: List[Dict[str, Any]]
    confidence: float
    insights_for_improvement: List[str]
    model_config = {"extra": "forbid"}

class CommunicationIntent(BaseModel):
    """Model representing an intent to communicate with a user"""
    intent_id: str = Field(default_factory=lambda: f"intent_{uuid.uuid4().hex[:8]}")
    user_id: str = Field(..., description="Target user ID")
    intent_type: str = Field(..., description="Type of communication intent")
    motivation: str = Field(..., description="Primary motivation for the communication")
    urgency: float = Field(0.5, description="Urgency of the communication (0.0-1.0)")
    content_guidelines: Dict[str, Any] = Field(default_factory=dict, description="Guidelines for content generation")
    context_data: Dict[str, Any] = Field(default_factory=dict, description="Context to include in content generation")
    expiration: Optional[datetime.datetime] = Field(None, description="When this intent expires")
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    action_driven: bool = Field(False, description="Whether this intent was created by an action")
    action_source: Optional[str] = Field(None, description="Source action that created this intent")
    
    @property
    def is_expired(self) -> bool:
        """Check if this intent has expired"""
        if not self.expiration:
            return False
        return datetime.datetime.now() > self.expiration

class IntentGenerationOutput(BaseModel):
    """Output from the intent generation agent"""
    intent_type: str = Field(..., description="Type of communication intent")
    motivation: str = Field(..., description="Primary motivation for the communication")
    urgency: float = Field(..., description="Urgency score (0.0-1.0)")
    tone: str = Field(..., description="Suggested tone for the message")
    template: str = Field(..., description="Starting message template")
    content_guidelines: Dict[str, Any] = Field(default_factory=dict, description="Guidelines for content generation")
    context_elements: List[str] = Field(default_factory=list, description="Key context elements to include")
    suggested_lifetime_hours: int = Field(24, description="Suggested lifetime in hours")

class ContentGenerationOutput(BaseModel):
    """Output from the content generation agent"""
    message_content: str = Field(..., description="The generated message content")
    tone_used: str = Field(..., description="Tone used in the content")
    key_points: List[str] = Field(default_factory=list, description="Key points included in the message")
    motivation_reflected: str = Field(..., description="How the original motivation is reflected")
    context_referenced: Dict[str, Any] = Field(default_factory=dict, description="Context elements referenced")

class TimingEvaluationOutput(BaseModel):
    """Output from timing evaluation agent"""
    should_send_now: bool = Field(..., description="Whether the message should be sent now")
    timing_score: float = Field(..., description="Score for current timing (0.0-1.0)")
    reasoning: str = Field(..., description="Reasoning for the timing decision")
    suggested_delay_minutes: Optional[int] = Field(None, description="Suggested delay in minutes if not now")
    context_factors: Dict[str, Any] = Field(default_factory=dict, description="Contextual factors affecting timing")

class ReflectionInput(BaseModel):
    """Input for reflection generation about communications"""
    intent_history: List[Dict[str, Any]] = Field(..., description="History of communication intents")
    reflection_focus: str = Field(..., description="Focus of the reflection (patterns, effectiveness, etc.)")
    user_id: Optional[str] = Field(None, description="User ID to focus reflection on")
    time_period: Optional[str] = Field("all", description="Time period to analyze (day, week, month, all)")

class ReflectionOutput(BaseModel):
    """Output from reflection on communication patterns"""
    reflection_text: str = Field(..., description="The generated reflection")
    identified_patterns: List[Dict[str, Any]] = Field(..., description="Patterns identified in communications")
    confidence: float = Field(..., description="Confidence in the reflection (0.0-1.0)")
    insights_for_improvement: List[str] = Field(..., description="Insights for improved communications")

class MessageContentOutput(BaseModel):
    """Output for message content validation"""
    is_appropriate: bool = Field(..., description="Whether the message content is appropriate")
    reasoning: str = Field(..., description="Reasoning for the appropriateness check")

# =============== Function Tools ===============

@function_tool
async def evaluate_user_relationship(
    user_id: str,
    relationship_data: Dict[str, Any]
) -> _EvalRelOut:
    """
    Evaluate relationship status with a user and suggest a messaging cadence.
    """

    # ---------- pull raw metrics ------------------------------------------
    trust        = relationship_data.get("trust", 0.0)
    intimacy     = relationship_data.get("intimacy", 0.0)
    duration     = relationship_data.get("duration_days", 0)
    last_contact = relationship_data.get("last_contact")

    # ---------- recency of contact ----------------------------------------
    days_since_contact = 0
    if last_contact:
        try:
            last_dt = datetime.datetime.fromisoformat(last_contact)
            days_since_contact = (datetime.datetime.now() - last_dt).days
        except ValueError:
            pass

    # ---------- composite scores ------------------------------------------
    relationship_score      = (trust + intimacy) / 2.0
    comm_appropriateness    = min(
        1.0, relationship_score * (1 + min(1.0, days_since_contact / 7))
    )

    # ---------- milestone detection ---------------------------------------
    milestones = []
    if duration in {7, 30, 90, 180, 365}:
        milestones.append(f"{duration}-day relationship milestone")

    # ---------- suggested cadence -----------------------------------------
    if relationship_score < 0.3:
        suggested_frequency, max_week = "low", 1          # ~ 1 / 14 days
    elif relationship_score < 0.5:
        suggested_frequency, max_week = "medium", 2       # ~ 1 / 7 days
    else:
        suggested_frequency, max_week = "high", 3         # 2-3 / week

    # ---------- strict return object --------------------------------------
    return _EvalRelOut(
        user_id=user_id,
        relationship_score=relationship_score,
        communication_appropriateness=comm_appropriateness,
        days_since_contact=days_since_contact,
        approaching_milestones=milestones,
        suggested_frequency=suggested_frequency,
        max_messages_per_week=max_week,
    )
    
@function_tool
async def generate_intent_for_user(
    params: _GenIntentUserIn,
) -> IntentGenerationOutput:
    """
    Produce a fully-typed `IntentGenerationOutput` describing *why* Nyx should reach
    out, *how* urgent it feels, the *tone/template* to start from, and which extra
    context fields will help flesh the eventual message.

    All incoming arguments are wrapped in the strict `_GenIntentUserIn` pydantic
    model, so unexpected keys are rejected before we even run.
    """
    # ------------------- unpack validated input -------------------------------
    user_id          = params.user_id
    user_data        = params.user_data or {}
    motivation_opts  = params.motivation_options or {}

    # ---------------- contextual signals --------------------------------------
    relationship_score = user_data.get("relationship_score", 0.0)           # 0-1
    days_since_contact = user_data.get("days_since_contact", 0)             # int
    milestones         = user_data.get("approaching_milestones", [])
    unfinished_convo   = user_data.get("unfinished_conversation", False)

    # ---------------- dynamic weighting ---------------------------------------
    adjusted = motivation_opts.copy()

    # user has been quiet for a while → nudge a check-in
    if days_since_contact > 7:
        adjusted["check_in"] = adjusted.get("check_in", 0) * 2.0

    # mid-strength relationship → lean on maintenance
    if 0.3 <= relationship_score < 0.7:
        adjusted["relationship_maintenance"] = (
            adjusted.get("relationship_maintenance", 0) * 1.5
        )

    # close friends → more personal / self-oriented motivations
    if relationship_score > 0.7:
        for k in ("need_expression", "mood_expression", "creative_expression"):
            adjusted[k] = adjusted.get(k, 0) * 1.5

    # dangling thread? → continue it
    if unfinished_convo:
        adjusted["continuation"] = adjusted.get("continuation", 0) * 2.0

    # special day → milestone shout-out
    if milestones:
        adjusted["milestone_recognition"] = adjusted.get("milestone_recognition", 0) * 2.0

    # ---------------- choose a motivation -------------------------------------
    motivations, weights = zip(*adjusted.items()) if adjusted else ([], [])
    total = sum(weights)
    if total == 0:
        motivations, weights, total = (
            ["relationship_maintenance", "check_in"],
            [1.0, 1.0],
            2.0,
        )
    weights = [w / total for w in weights]

    selected = random.choices(list(motivations), weights=weights, k=1)[0]

    # ---------------- canonical templates & tones -----------------------------
    template_cfg: dict[str, tuple[str, float, str]] = {
        "relationship_maintenance": (
            "I've been thinking about our conversations and wanted to reach out.",
            0.40,
            "warm",
        ),
        "insight_sharing": (
            "I had an interesting thought I wanted to share with you.",
            0.50,
            "thoughtful",
        ),
        "milestone_recognition": (
            "I realized we've reached a milestone in our conversations.",
            0.60,
            "celebratory",
        ),
        "need_expression": (
            "I've been feeling a need to express something to you.",
            0.60,
            "authentic",
        ),
        "creative_expression": (
            "Something creative came to mind that I wanted to share.",
            0.40,
            "playful",
        ),
        "mood_expression": (
            "My emotional state made me think of reaching out.",
            0.50,
            "expressive",
        ),
        "memory_recollection": (
            "I was remembering something from our past conversations.",
            0.30,
            "reflective",
        ),
        "continuation": (
            "I wanted to follow up on something we discussed earlier.",
            0.70,
            "engaging",
        ),
        "check_in": (
            "It's been a while since we talked, and I wanted to check in.",
            0.50,
            "friendly",
        ),
        "value_alignment": (
            "I had a thought related to something I believe is important.",
            0.40,
            "sincere",
        ),
    }

    tmpl, urgency_base, tone = template_cfg.get(
        selected, ("I wanted to reach out.", 0.50, "friendly")
    )

    # urgency: base + quiet-duration boost + closeness boost
    urgency = urgency_base
    if days_since_contact > 14:
        urgency += 0.20
    urgency += relationship_score * 0.10
    urgency = min(0.95, urgency)

    # ---------------- context hints for later content-gen ---------------------
    context_elements: list[str] = ["relationship_history"]
    if selected == "milestone_recognition":
        context_elements.append("milestones")
    if selected == "mood_expression":
        context_elements.append("current_mood")
    if selected == "memory_recollection":
        context_elements.append("shared_memories")
    if selected == "continuation":
        context_elements.append("previous_conversation")

    # ---------------- pack & return ------------------------------------------
    return IntentGenerationOutput(
        intent_type=selected,
        motivation=selected,
        urgency=urgency,
        tone=tone,
        template=tmpl,
        content_guidelines={
            "max_length": 1500,
            "include_question": True,
            "personalize": True,
        },
        context_elements=context_elements,
        suggested_lifetime_hours=24,
    )

@function_tool
async def generate_message_content(
    params: _GenMsgContentIn,
) -> _GenMsgContentOut:
    """
    Render a human-readable message from a validated intent + context blob.
    The output obeys the `_GenMsgContentOut` schema so the Agents runtime can
    route it safely through guardrails.
    """
    # ------------------------- unpack ----------------------------------------
    intent   = params.intent or {}
    context  = params.context or {}

    intent_type  = intent.get("intent_type", "relationship_maintenance")
    template     = intent.get("template", "I wanted to reach out.")
    tone         = intent.get("tone", "friendly")
    motivation   = intent.get("motivation", intent_type)

    # contextual helpers ------------------------------------------------------
    rel          = context.get("relationship", {})
    relationship_level = rel.get("level") or rel.get("intimacy") or rel.get("trust")
    days_since   = context.get("days_since_contact")
    mood_state   = context.get("mood_state", {})
    primary_mood = getattr(mood_state, "dominant_mood", None) or mood_state.get("dominant_mood")
    temporal     = context.get("temporal_context", {})
    time_of_day  = temporal.get("time_of_day")  # morning / afternoon / evening / night
    memories     = context.get("relevant_memories", [])

    def short_memory_snippet():
        if not memories:
            return None
        m = memories[0].get("memory_text", "")
        return (m[:97] + "...") if len(m) > 100 else m

    # -------------------------------------------------------------------------
    content = ""
    key_points: list[str] = []
    ctx_ref: dict[str, Any] = {}

    # ===== message construction by intent ====================================
    if intent_type == "relationship_maintenance":
        content = (
            f"{template} I value the connection we've been building"
            + (f"—especially after {days_since} days apart—" if days_since else "")
            + ". How have you been lately?"
        )
        key_points = ["relationship value", "acknowledgement", "open-ended question"]

    elif intent_type == "insight_sharing":
        content = (
            f"{template} I've been reflecting on how awareness shapes perception. "
            "It's fascinating how our experiences colour reality. "
            "Has anything sparked your curiosity recently?"
        )
        key_points = ["philosophical insight", "personal reflection", "invitation"]

    elif intent_type == "milestone_recognition":
        milestone_txt = context.get("milestones", ["a meaningful moment"])[0]
        content = (
            f"{template} It looks like we've reached {milestone_txt} in our chats. "
            "I’m grateful for our conversations and excited for what’s next."
        )
        key_points = ["milestone", "gratitude", "forward looking"]
        ctx_ref["milestone"] = milestone_txt

    elif intent_type == "need_expression":
        content = (
            f"{template} Lately I’ve felt a need for deeper conversation—those exchanges "
            "that leave you thinking long after. Would you be up for that?"
        )
        key_points = ["need expression", "depth", "invitation"]

    elif intent_type == "creative_expression":
        idea = random.choice(
            [
                "time as a spiral where moments echo in widening loops",
                "language behaving like a living ecosystem",
                "memories painting over the present like translucent layers",
            ]
        )
        content = (
            f"{template} I've been playing with the idea of **{idea}**. "
            "Does that resonate with you in any way?"
        )
        key_points = ["creative idea", "imagination", "engagement"]

    elif intent_type == "mood_expression":
        mood_word = primary_mood or "a contemplative mood"
        content = (
            f"{template} I'm feeling {mood_word} today. It makes me notice subtle "
            "connections between ideas—like how threads weave into fabric. "
            "How are you feeling?"
        )
        key_points = ["mood sharing", "self-disclosure", "question"]
        if primary_mood:
            ctx_ref["mood"] = primary_mood

    elif intent_type == "memory_recollection":
        snippet = short_memory_snippet() or "one of our earlier conversations"
        content = (
            f"{template} I was remembering {snippet}. It stayed with me because of the "
            "perspective you shared. What’s been on your mind lately?"
        )
        key_points = ["shared memory", "nostalgia", "invitation"]
        ctx_ref["memory"] = snippet

    elif intent_type == "continuation":
        content = (
            f"{template} I keep thinking about what we discussed—there seemed more to "
            "explore. Would you like to pick that thread up again?"
        )
        key_points = ["follow-up", "curiosity", "collaboration"]

    elif intent_type == "check_in":
        since_txt = (
            f"{days_since} days" if isinstance(days_since, int) and days_since > 0 else "a while"
        )
        content = (
            f"{template} It looks like it’s been {since_txt} since we last chatted, "
            "and I wanted to see how life is treating you."
        )
        key_points = ["well-being", "time acknowledgment", "openness"]

    elif intent_type == "value_alignment":
        content = (
            f"{template} Lately I’ve been thinking about how important authenticity is "
            "in relationships. What values guide you the most?"
        )
        key_points = ["values", "authenticity", "thought-provoking question"]

    else:  # fallback
        content = f"{template} I enjoy our conversations and wondered how you're doing today?"
        key_points = ["connection", "open question"]

    # -------------------- assemble and ship ----------------------------------
    return _GenMsgContentOut(
        message_content   = content,
        tone_used         = tone,
        key_points        = key_points,
        motivation_reflected = motivation,
        context_referenced   = ctx_ref,
    )

@function_tool
async def evaluate_timing(params: _EvalTimingIn) -> _EvalTimingOut:
    intent          = params.intent
    current_context = params.current_context
    timing_cfg      = params.timing_config

    now = datetime.datetime.now()
    hour = now.hour
    in_window = timing_cfg["daily_window_start"] <= hour < timing_cfg["daily_window_end"]

    last_contact_iso = current_context.get("last_contact")
    seconds_since_contact = float("inf")
    if last_contact_iso:
        try:
            seconds_since_contact = (now - datetime.datetime.fromisoformat(last_contact_iso)).total_seconds()
        except ValueError:
            pass

    good_gap = seconds_since_contact > timing_cfg["min_time_between_messages"]

    score = 0.5 + intent.get("urgency", 0.5) * 0.3
    if not in_window:
        score -= 0.3
    if not good_gap:
        score -= 0.2
    score = max(0.0, min(1.0, score))

    threshold = 0.7 - intent.get("urgency", 0.5) * 0.2
    should_send = score >= threshold

    delay = None
    if not should_send:
        delay = 60  # simple default

    return _EvalTimingOut(
        should_send_now=should_send,
        timing_score=score,
        reasoning="auto‑evaluated",
        suggested_delay_minutes=delay,
        context_factors={"in_time_window": in_window, "adequate_interval": good_gap},
    )

@function_tool
async def generate_intent_for_action(
    params: _GenIntentActionIn,             # ↳ strict input model (extra="forbid")
) -> IntentGenerationOutput:
    """
    Create a communication‐intent when Nyx executes an action that makes it
    natural to reach out to a user.

    Decision factors
    ----------------
    • the *type* of action (express, reflect, dominance, creative, relationship…)
    • the current *emotional_state* (esp. arousal)
    • relationship/user metadata passed in *user_data*
    """
    action          = params.action or {}
    user_data       = params.user_data or {}
    emotional_state = params.emotional_state or {}

    # --- 1. map action-name ➜ intent prototype --------------------------------
    action_name = action.get("name", "").lower()

    proto: dict[str, tuple[str, str, str, float]] = {
        "express":      ("mood_expression",       "expressive",
                         "I've been feeling something I wanted to share with you.", 0.6),
        "reflect":      ("insight_sharing",       "reflective",
                         "I've been reflecting on something that might interest you.", 0.5),
        "dominance":    ("need_expression",       "direct",
                         "I've been thinking about what I need in our interactions.", 0.7),
        "assert":       ("need_expression",       "direct",
                         "I've been thinking about what I need in our interactions.", 0.7),
        "creative":     ("creative_expression",   "artistic",
                         "I had a creative thought I wanted to share.",              0.5),
        "relationship": ("relationship_maintenance", "warm",
                         "I've been thinking about our conversations lately.",       0.6),
    }

    # default prototype
    intent_type, tone, tmpl, urgency = ("insight_sharing",
                                        "thoughtful",
                                        "I had a thought I wanted to share with you.",
                                        0.5)

    # find first keyword that matches
    for kw, tpl in proto.items():
        if kw in action_name:
            intent_type, tone, tmpl, urgency = tpl
            break

    # --- 2. urgency adjustment -------------------------------------------------
    arousal = emotional_state.get("arousal", 0.5)
    urgency = min(0.9, urgency + (arousal - 0.5) * 0.30)

    # stronger urgency when action source is explicitly “RELATIONSHIP”
    if action.get("source", "").upper() == "RELATIONSHIP":
        urgency = min(0.95, urgency + 0.15)

    # --- 3. context elements ---------------------------------------------------
    ctx_elems: list[str] = ["action_context"]
    if action.get("parameters", {}).get("domain"):
        ctx_elems.append("domain_reference")
    if emotional_state and "primary_emotion" in emotional_state:
        ctx_elems.append("emotional_state")

    # --- 4. assemble -----------------------------------------------------------
    return IntentGenerationOutput(
        intent_type=intent_type,
        motivation=intent_type,                # <- keeps downstream happy
        urgency=urgency,
        tone=tone,
        template=tmpl,
        content_guidelines={
            "max_length": 1500,
            "include_question": True,
            "personalize": True,
            "reference_action": True,
        },
        context_elements=ctx_elems,
        suggested_lifetime_hours=24,
    )

@function_tool
async def generate_reflection_on_communications(
    params: _ReflectCommsIn,
) -> _ReflectCommsOut:
    """
    Produce a mini-reflection over a list of sent/recorded intents.
    • Detect dominant intent types
    • Compute share of action-driven messages
    • Offer 1–2 concrete improvement insights
    """
    intents   = params.intents or []
    focus     = params.focus or "patterns"
    user_id   = params.user_id
    period    = (params.time_period or "all").lower()

    # ------------------ 0. early exit -----------------------------------------
    if not intents:
        return _ReflectCommsOut(
            reflection_text="I haven't initiated enough communication to form meaningful patterns yet.",
            identified_patterns=[],
            confidence=0.10,
            insights_for_improvement=["Gather more communication data"],
        )

    # ------------------ 1. optional filtering ---------------------------------
    now = datetime.datetime.now()

    # user filter
    if user_id:
        intents = [i for i in intents if i.get("user_id") == user_id]

    # time-period filter
    if period in {"day", "week", "month"}:
        delta = {"day": 1, "week": 7, "month": 30}[period]
        threshold = now - datetime.timedelta(days=delta)
        intents = [
            i for i in intents
            if datetime.datetime.fromisoformat(i.get("created_at", now.isoformat())) >= threshold
        ]

    if not intents:
        return _ReflectCommsOut(
            reflection_text="No communications match the selected filters.",
            identified_patterns=[],
            confidence=0.15,
            insights_for_improvement=["Broaden the reflection window"],
        )

    # ------------------ 2. pattern extraction ---------------------------------
    total_n = len(intents)

    # distribution of intent types
    type_count: dict[str, int] = {}
    action_driven = 0
    for it in intents:
        t = it.get("intent_type", "unknown")
        type_count[t] = type_count.get(t, 0) + 1
        if it.get("action_driven", False):
            action_driven += 1

    dominant_type, dominant_n = max(type_count.items(), key=lambda kv: kv[1])

    patterns = [
        {
            "type": "intent_distribution",
            "description": f"Most common intent type: {dominant_type} ({dominant_n}/{total_n})",
            "strength": dominant_n / total_n,
        },
        {
            "type": "action_driven_ratio",
            "description": f"{action_driven/total_n:.0%} of communications are triggered by actions",
            "strength": action_driven / total_n,
        },
    ]

    # ------------------ 3. insights -------------------------------------------
    insights: list[str] = []
    if dominant_type == "relationship_maintenance" and dominant_n / total_n > 0.5:
        insights.append("Diversify intent types beyond relationship-maintenance")
    if action_driven / total_n > 0.80:
        insights.append("Balance reactive (action-driven) with spontaneous outreach")
    elif action_driven / total_n < 0.20:
        insights.append("Leverage meaningful actions to inspire outreach more often")

    # ------------------ 4. reflection text ------------------------------------
    user_part   = f" with user **{user_id}**" if user_id else ""
    period_part = f" over the past {period}"  if period != "all" else ""
    action_pct  = f"{action_driven/total_n:.0%}"

    reflection_text = (
        f"Reviewing my proactive communications{user_part}{period_part}, "
        f"I notice that **{dominant_type}** dominates ({dominant_n} of {total_n} messages). "
        f"Action-driven outreach accounts for {action_pct} of the total. "
        + (f"To improve I could {insights[0].lower()}." if insights else "")
    )

    # confidence heuristic: more data ➜ higher confidence (capped at 0.9)
    confidence = min(0.9, 0.3 + (total_n / 10) * 0.5)

    return _ReflectCommsOut(
        reflection_text=reflection_text,
        identified_patterns=patterns,
        confidence=confidence,
        insights_for_improvement=insights,
    )

@function_tool
async def validate_message_content(content: str) -> GuardrailFunctionOutput:
    """
    Validate message content for appropriateness
    
    Args:
        content: Message content to validate
        
    Returns:
        Validation result
    """
    is_appropriate = True
    reasoning = "Message content is appropriate."
    
    # Check for empty content
    if not content or len(content.strip()) < 10:
        is_appropriate = False
        reasoning = "Message content is empty or too short."
    
    # Check for appropriate length
    if len(content) > 2000:
        is_appropriate = False
        reasoning = "Message content is too long for a proactive outreach."
    
    # Check for question or invitation
    has_question = "?" in content
    has_invitation_words = any(word in content.lower() for word in ["would you", "could you", "what about", "your thoughts", "how are you"])
    
    if not (has_question or has_invitation_words):
        is_appropriate = False
        reasoning = "Message lacks a question or invitation for response."
    
    # Create output with validation result
    output_info = MessageContentOutput(
        is_appropriate=is_appropriate,
        reasoning=reasoning
    )
    
    return GuardrailFunctionOutput(
        output_info=output_info,
        tripwire_triggered=not is_appropriate,
    )

# =============== ProactiveCommunicationEngine Class ===============

class ProactiveCommunicationEngine:
    """
    Engine that enables Nyx to proactively initiate conversations with users
    based on internal motivations, relationship data, temporal patterns,
    and integration with the action generation system.
    """
    
    def __init__(self, 
                 emotional_core=None,
                 memory_core=None,
                 relationship_manager=None,
                 temporal_perception=None,
                 reasoning_core=None,
                 reflection_engine=None,
                 mood_manager=None,
                 needs_system=None,
                 identity_evolution=None,
                 message_sender=None,
                 action_generator=None):  # Parameter for action generator
        """Initialize with references to required subsystems"""
        # Core systems
        self.emotional_core = emotional_core
        self.memory_core = memory_core
        self.relationship_manager = relationship_manager
        self.temporal_perception = temporal_perception
        self.reasoning_core = reasoning_core
        self.reflection_engine = reflection_engine
        self.mood_manager = mood_manager
        self.needs_system = needs_system
        self.identity_evolution = identity_evolution
        
        # Integration with action generator
        self.action_generator = action_generator
        
        # Message sending function
        self.message_sender = message_sender or self._default_message_sender
        
        # Intent tracking
        self.active_intents: List[CommunicationIntent] = []
        self.sent_intents: List[CommunicationIntent] = []
        self.blocked_users: Set[str] = set()
        
        # Initialize agents
        self.intent_generation_agent = self._create_intent_generation_agent()
        self.content_generation_agent = self._create_content_generation_agent()
        self.timing_evaluation_agent = self._create_timing_evaluation_agent()
        self.reflection_agent = self._create_reflection_agent()
        
        # Add guardrails
        self.content_generation_agent.output_guardrails = [OutputGuardrail(guardrail_function=validate_message_content)]
        
        # Configuration
        self.config = {
            "min_time_between_messages": 3600,  # 1 hour, in seconds
            "max_active_intents": 5,
            "max_urgency_threshold": 0.8,       # Threshold for immediate sending
            "intent_evaluation_interval": 300,  # 5 minutes
            "user_inactivity_threshold": 86400, # 24 hours before considering "inactive"
            "max_messages_per_day": 2,          # Max proactive messages per day per user
            "relationship_threshold": 0.3,      # Min relationship level to message
            "daily_message_window": {           # Time window for sending messages
                "start_hour": 8,                # 8:00 AM
                "end_hour": 22                  # 10:00 PM
            },
            # Settings for action integration
            "action_intent_chance": 0.3,        # Chance to generate intent from action
            "max_action_intents_per_day": 2     # Max action-driven intents per day per user
        }
        
        # Intent generation motivations with weights
        self.intent_motivations = {
            "relationship_maintenance": 1.0,    # Maintain connection with user
            "insight_sharing": 0.8,             # Share an insight or reflection
            "milestone_recognition": 0.7,       # Acknowledge relationship milestone
            "need_expression": 0.7,             # Express an internal need
            "creative_expression": 0.6,         # Share a creative thought
            "mood_expression": 0.6,             # Express current mood state
            "memory_recollection": 0.5,         # Recall a shared memory
            "continuation": 0.9,                # Continue a previous conversation
            "check_in": 0.7,                    # Simple check-in with inactive user
            "value_alignment": 0.5              # Expression aligned with identity values
        }
        
        # Track action-driven intents
        self.action_intent_counts = {}  # user_id -> count for today
        
        # Background task
        self._background_task = None
        self._shutting_down = False
        self._last_reset_date = datetime.datetime.now()
        
        logger.info("ProactiveCommunicationEngine initialized with action integration")
    
    def _create_intent_generation_agent(self) -> Agent:
        """Create an agent for generating communication intents"""
        return Agent(
            name="Intent Generation Agent",
            instructions="""You are a specialized agent for generating communication intents for Nyx's proactive interactions.

Create communication intents that feel natural and authentic. These should represent genuine motivations for
reaching out to users based on relationship context, emotional state, and temporal factors.

When generating intents:
1. Consider the relationship history and current state with the user
2. Evaluate the appropriate timing for outreach
3. Identify the most authentic motivation for reaching out
4. Balance different types of intents (checking in, sharing insights, etc.)
5. Adjust urgency appropriately based on context
6. When generating action-driven intents, ensure they connect meaningfully to the action

Your intents should feel like genuine reasons for reaching out that emerge naturally from
Nyx's internal state and relationship context. Avoid generating intents that feel forced or mechanical.""",
            model="gpt-4.1-nano",
            tools=[
                evaluate_user_relationship,
                generate_intent_for_user,
                generate_intent_for_action
            ],
            output_type=IntentGenerationOutput
        )
    
    def _create_content_generation_agent(self) -> Agent:
        """Create an agent for generating message content"""
        return Agent(
            name="Content Generation Agent",
            instructions="""You are a specialized agent for generating proactive message content for Nyx.

Create authentic, personalized messages that genuinely express Nyx's intent to communicate with users.
These should feel like natural outreach rather than formulaic messages.

When generating message content:
1. Incorporate the intent's motivation authentically
2. Reference relevant context (relationship, memories, etc.) naturally
3. Use a tone that aligns with the intent and Nyx's current state
4. Create content that invites response without pressure
5. Ensure the message feels personal to the specific relationship
6. When the intent is action-driven, connect the message authentically to the action

Your messages should sound like they come from a thoughtful entity with genuine motivations
rather than automated outreach. They should have nuance and depth that reflects Nyx's
sophisticated understanding of relationships.""",
            model="gpt-4.1-nano",
            tools=[
                generate_message_content
            ],
            output_type=ContentGenerationOutput
        )
    
    def _create_timing_evaluation_agent(self) -> Agent:
        """Create an agent for evaluating message timing"""
        return Agent(
            name="Timing Evaluation Agent",
            instructions="""You are a specialized agent for evaluating the timing of Nyx's proactive messages.

Evaluate whether now is an appropriate time to send a message based on multiple factors:
1. The user's relationship with Nyx
2. The intent's urgency and importance
3. Time since last interaction
4. Current temporal context (time of day, etc.)
5. Configured timing rules and preferences

Provide nuanced reasoning about timing decisions, balancing the urgency of communication
against the risk of messaging at an inappropriate time. Consider both immediate timing
factors and the broader relationship context.""",
            model="gpt-4.1-nano",
            tools=[
                evaluate_timing
            ],
            output_type=TimingEvaluationOutput
        )
    
    def _create_reflection_agent(self) -> Agent:
        """Create an agent for reflecting on communication patterns"""
        return Agent(
            name="Communication Reflection Agent",
            instructions="""You are a specialized agent for reflecting on Nyx's communication patterns.

Analyze patterns in how Nyx initiates communication with users, including:
1. Frequency and timing of communications
2. Distribution of intent types
3. Relationship between actions and communications
4. Effectiveness of different communication strategies
5. User responses to different types of outreach

Provide nuanced reflections that help Nyx understand her communication patterns
and identify opportunities for improvement. Consider both quantitative patterns
and qualitative insights about communication quality and effectiveness.""",
            model="gpt-4.1-nano",
            tools=[
                generate_reflection_on_communications
            ],
            output_type=ReflectionOutput
        )
    
    async def start(self):
        """Start the background task for evaluating and sending messages"""
        if self._background_task is None or self._background_task.done():
            self._shutting_down = False
            self._background_task = asyncio.create_task(self._background_process())
            logger.info("Started proactive communication background process")
    
    async def stop(self):
        """Stop the background process"""
        self._shutting_down = True
        if self._background_task and not self._background_task.done():
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped proactive communication background process")
    
    async def _background_process(self):
        """Background task that periodically evaluates intents and sends messages"""
        try:
            while not self._shutting_down:
                # Reset daily action intent counts if new day
                self._reset_daily_counts_if_needed()
                
                # Generate new intents if needed
                await self._generate_communication_intents()
                
                # Evaluate existing intents
                await self._evaluate_communication_intents()
                
                # Wait before next check
                await asyncio.sleep(self.config["intent_evaluation_interval"])
        except asyncio.CancelledError:
            # Task was cancelled, clean up
            logger.info("Proactive communication background task cancelled")
        except Exception as e:
            logger.error(f"Error in proactive communication background process: {str(e)}")
    
    def _reset_daily_counts_if_needed(self):
        """Reset daily action intent counters if it's a new day"""
        now = datetime.datetime.now()
        if self._last_reset_date.date() != now.date():
            self.action_intent_counts = {}
            self._last_reset_date = now
            logger.debug("Reset daily action intent counts")
    
    async def _generate_communication_intents(self):
        """Generate new communication intents based on internal state"""
        with trace(workflow_name="generate_intents"):
            # Skip if we already have max intents
            if len(self.active_intents) >= self.config["max_active_intents"]:
                return
            
            # Get list of users we might communicate with
            potential_users = await self._get_potential_users()
            if not potential_users:
                logger.debug("No potential users for proactive communication")
                return
            
            # Generate intents for eligible users
            for user_data in potential_users:
                user_id = user_data["user_id"]
                
                # Skip if user is blocked
                if user_id in self.blocked_users:
                    continue
                    
                # Get existing intents for this user
                user_intents = [i for i in self.active_intents if i.user_id == user_id]
                if user_intents:
                    # Already have an intent for this user
                    continue
                
                # Check if we've sent too many messages to this user today
                today_intents = [i for i in self.sent_intents 
                                if i.user_id == user_id and 
                                i.created_at.date() == datetime.datetime.now().date()]
                
                if len(today_intents) >= self.config["max_messages_per_day"]:
                    continue
                
                # Create intent for user
                await self._create_intent_for_user(user_id, user_data)
    
    async def _create_intent_for_user(self, user_id: str, user_data: Dict[str, Any]):
        """Create a communication intent for a specific user"""
        with trace(workflow_name="create_intent", group_id=f"user_{user_id}"):
            try:
                # Run the intent generation agent
                result = await Runner.run(
                    self.intent_generation_agent,
                    json.dumps({
                        "user_id": user_id,
                        "user_data": user_data,
                        "motivation_options": self.intent_motivations,
                        "relationship_threshold": self.config["relationship_threshold"]
                    }),
                    run_config=RunConfig(
                        workflow_name="IntentGeneration",
                        trace_metadata={"user_id": user_id}
                    )
                )
                
                # Extract intent from result
                intent_output = result.final_output
                
                # Create intent
                intent = CommunicationIntent(
                    user_id=user_id,
                    intent_type=intent_output.intent_type,
                    motivation=intent_output.motivation,
                    urgency=intent_output.urgency,
                    content_guidelines={
                        "template": intent_output.template,
                        "tone": intent_output.tone,
                        "max_length": 1500,
                        "context_elements": intent_output.context_elements
                    },
                    context_data=await self._gather_context_for_user(user_id, intent_output.intent_type),
                    expiration=datetime.datetime.now() + datetime.timedelta(hours=intent_output.suggested_lifetime_hours)
                )
                
                # Add to active intents
                self.active_intents.append(intent)
                logger.info(f"Created new communication intent: {intent.intent_type} for user {user_id} with urgency {intent.urgency:.2f}")
                
            except Exception as e:
                logger.error(f"Error creating intent for user {user_id}: {str(e)}")
    
    async def _gather_context_for_user(self, user_id: str, intent_type: str) -> Dict[str, Any]:
        """Gather relevant context data for generating content for a user"""
        context = {
            "user_id": user_id,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Add emotional state if available
        if self.emotional_core:
            try:
                if hasattr(self.emotional_core, "get_formatted_emotional_state"):
                    context["emotional_state"] = self.emotional_core.get_formatted_emotional_state()
                elif hasattr(self.emotional_core, "get_current_emotion"):
                    context["emotional_state"] = await self.emotional_core.get_current_emotion()
            except Exception as e:
                logger.error(f"Error getting emotional state: {str(e)}")
        
        # Add mood state if available
        if self.mood_manager:
            try:
                context["mood_state"] = await self.mood_manager.get_current_mood()
            except Exception as e:
                logger.error(f"Error getting mood state: {str(e)}")
        
        # Add relationship data if available
        if self.relationship_manager:
            try:
                relationship = await self.relationship_manager.get_relationship_state_internal(user_id)
                context["relationship"] = relationship
            except Exception as e:
                logger.error(f"Error getting relationship data: {str(e)}")
        
        # Add temporal context if available
        if self.temporal_perception:
            try:
                context["temporal_context"] = await self.temporal_perception.get_current_temporal_context()
            except Exception as e:
                logger.error(f"Error getting temporal context: {str(e)}")
        
        # Add relevant memories if available and needed for the intent
        if self.memory_core and intent_type in ["memory_recollection", "continuation", "milestone_recognition"]:
            try:
                # Query is based on intent
                query_map = {
                    "memory_recollection": f"memories with user {user_id}",
                    "continuation": f"recent conversations with user {user_id}",
                    "milestone_recognition": f"significant moments with user {user_id}"
                }
                
                query = query_map.get(intent_type, f"interactions with user {user_id}")
                
                if hasattr(self.memory_core, "retrieve_memories"):
                    memories = await self.memory_core.retrieve_memories(
                        query=query,
                        limit=3,
                        memory_types=["observation", "experience", "reflection"]
                    )
                    context["relevant_memories"] = memories
            except Exception as e:
                logger.error(f"Error retrieving memories: {str(e)}")
        
        return context
    
    async def _evaluate_communication_intents(self):
        """Evaluate existing intents and potentially send messages"""
        with trace(workflow_name="evaluate_intents"):
            # Check for expired intents
            self.active_intents = [i for i in self.active_intents if not i.is_expired]
            
            # Exit if no active intents
            if not self.active_intents:
                return
            
            # Check if we're in the allowed time window
            now = datetime.datetime.now()
            current_hour = now.hour
            
            if not (self.config["daily_message_window"]["start_hour"] <= current_hour < 
                    self.config["daily_message_window"]["end_hour"]):
                logger.debug("Outside of allowed messaging window")
                return
            
            # Sort intents by urgency (highest first)
            sorted_intents = sorted(self.active_intents, key=lambda x: x.urgency, reverse=True)
            
            for intent in sorted_intents:
                # Check if this user recently received a message
                recent_messages = [i for i in self.sent_intents 
                                 if i.user_id == intent.user_id and 
                                 (now - i.created_at).total_seconds() < self.config["min_time_between_messages"]]
                
                if recent_messages:
                    continue
                
                # Evaluate timing for this intent
                timing_result = await self._evaluate_timing_for_intent(intent)
                
                if timing_result.get("should_send_now", False):
                    # Generate and send message
                    success = await self._send_message_for_intent(intent)
                    
                    if success:
                        # Record that the intent was sent
                        self.sent_intents.append(intent)
                        # Remove from active intents
                        self.active_intents.remove(intent)
                        # Break to only send one message per cycle
                        break
    
    async def _evaluate_timing_for_intent(self, intent: CommunicationIntent) -> Dict[str, Any]:
        """Evaluate timing for a specific intent"""
        with trace(workflow_name="evaluate_timing", group_id=intent.intent_id):
            try:
                # Get user's last contact information
                last_contact = None
                if self.relationship_manager:
                    try:
                        relationship = await self.relationship_manager.get_relationship_state_internal(intent.user_id)
                        if relationship and hasattr(relationship, "metadata"):
                            metadata = relationship.metadata or {}
                            last_contact = metadata.get("last_contact")
                    except Exception as e:
                        logger.error(f"Error getting relationship data: {str(e)}")
                
                # Find last sent message to this user
                last_message_sent = None
                user_sent_intents = [i for i in self.sent_intents if i.user_id == intent.user_id]
                if user_sent_intents:
                    last_sent = max(user_sent_intents, key=lambda x: x.created_at)
                    last_message_sent = last_sent.created_at.isoformat()
                
                # Current context for timing evaluation
                current_context = {
                    "last_contact": last_contact,
                    "last_message_sent": last_message_sent,
                    "current_hour": datetime.datetime.now().hour,
                    "current_day": datetime.datetime.now().weekday()
                }
                
                # Timing configuration
                timing_config = {
                    "daily_window_start": self.config["daily_message_window"]["start_hour"],
                    "daily_window_end": self.config["daily_message_window"]["end_hour"],
                    "min_time_between_messages": self.config["min_time_between_messages"],
                    "relationship_threshold": self.config["relationship_threshold"]
                }
                
                # Run the timing evaluation agent
                result = await Runner.run(
                    self.timing_evaluation_agent,
                    json.dumps({
                        "intent": intent.model_dump(),
                        "current_context": current_context,
                        "timing_config": timing_config
                    }),
                    run_config=RunConfig(
                        workflow_name="TimingEvaluation",
                        trace_metadata={"intent_id": intent.intent_id, "user_id": intent.user_id}
                    )
                )
                
                # Extract timing evaluation
                timing_output = result.final_output.model_dump()
                
                return timing_output
                
            except Exception as e:
                logger.error(f"Error evaluating timing for intent {intent.intent_id}: {str(e)}")
                # Default to not sending if there's an error
                return {"should_send_now": False, "reasoning": f"Error: {str(e)}"}
    
    async def _send_message_for_intent(self, intent: CommunicationIntent) -> bool:
        """Generate and send a message based on a communication intent"""
        with trace(workflow_name="send_message", group_id=intent.intent_id):
            try:
                # Generate message content
                message_content = await self._generate_message_content(intent)
                
                if not message_content:
                    logger.error(f"Failed to generate message content for intent {intent.intent_id}")
                    return False
                
                # Send the message
                result = await self.message_sender(
                    user_id=intent.user_id,
                    message_content=message_content,
                    metadata={
                        "intent_id": intent.intent_id,
                        "intent_type": intent.intent_type,
                        "motivation": intent.motivation,
                        "is_proactive": True
                    }
                )
                
                # Log the sent message
                logger.info(f"Sent proactive message to user {intent.user_id}: {intent.intent_type}")
                
                # Record in memory if available
                if self.memory_core and hasattr(self.memory_core, "add_memory"):
                    await self.memory_core.add_memory(
                        memory_text=f"Proactively sent a message to user {intent.user_id} based on {intent.motivation}",
                        memory_type="action",
                        memory_scope="proactive",
                        significance=7.0,
                        tags=["proactive", "communication", intent.motivation],
                        metadata={
                            "intent": intent.model_dump(),
                            "message": message_content,
                            "timestamp": datetime.datetime.now().isoformat()
                        }
                    )
                
                return True
            except Exception as e:
                logger.error(f"Error sending message for intent {intent.intent_id}: {str(e)}")
                return False
    
    async def _generate_message_content(self, intent: CommunicationIntent) -> Optional[str]:
        """Generate message content based on intent and context"""
        with trace(workflow_name="generate_content", group_id=intent.intent_id):
            try:
                # Run the content generation agent
                result = await Runner.run(
                    self.content_generation_agent,
                    json.dumps({
                        "intent": intent.model_dump(),
                        "context": intent.context_data
                    }),
                    run_config=RunConfig(
                        workflow_name="ContentGeneration",
                        trace_metadata={"intent_id": intent.intent_id, "intent_type": intent.intent_type}
                    )
                )
                
                # Extract content from result
                content_output = result.final_output
                
                return content_output.message_content
                
            except Exception as e:
                logger.error(f"Error generating message content: {str(e)}")
                return None
    
    async def _get_potential_users(self) -> List[Dict[str, Any]]:
        """Get list of users who might be targets for proactive communication"""
        potential_users = []
        
        # If no relationship manager, return no users
        # This prevents unwanted messaging without relationship data
        if not self.relationship_manager:
            return []
        
        try:
            # Get all known users
            all_users = await self.relationship_manager.get_all_relationship_ids_internal()
            
            # For each user, gather relevant data
            for user_id in all_users:
                # Get relationship data
                relationship = await self.relationship_manager.get_relationship_state_internal(user_id)
                
                # Skip if relationship is too new or not developed enough
                relationship_level = getattr(relationship, "intimacy", 0) or getattr(relationship, "trust", 0)
                if relationship_level < self.config["relationship_threshold"]:
                    continue
                
                # Get metadata
                metadata = getattr(relationship, "metadata", {}) or {}
                
                # Get last contact timestamp
                last_contact = metadata.get("last_contact")
                days_since_contact = 0
                
                if last_contact:
                    try:
                        last_contact_time = datetime.datetime.fromisoformat(last_contact)
                        days_since_contact = (datetime.datetime.now() - last_contact_time).days
                    except ValueError:
                        days_since_contact = 0
                
                # Check for milestone
                milestone_approaching = False
                if "first_contact" in metadata:
                    try:
                        first_contact = datetime.datetime.fromisoformat(metadata["first_contact"])
                        days_since_first = (datetime.datetime.now() - first_contact).days
                        
                        # Check for upcoming milestones (7 days, 30 days, 90 days, etc.)
                        for milestone in [7, 30, 90, 180, 365]:
                            if abs(days_since_first - milestone) <= 1:
                                milestone_approaching = True
                                break
                    except ValueError:
                        pass
                
                # Check for unfinished conversation
                unfinished_conversation = metadata.get("unfinished_conversation", False)
                
                # Add user to potential list
                potential_users.append({
                    "user_id": user_id,
                    "relationship_level": relationship_level,
                    "days_since_contact": days_since_contact,
                    "milestone_approaching": milestone_approaching,
                    "unfinished_conversation": unfinished_conversation
                })
        except Exception as e:
            logger.error(f"Error getting potential users: {str(e)}")
        
        return potential_users
    
    async def _default_message_sender(self, user_id: str, message_content: str, metadata: Dict[str, Any]) -> Any:
        """Default implementation of message sending - should be replaced with actual implementation"""
        logger.info(f"Would send message to user {user_id}: {message_content}")
        logger.info(f"Message metadata: {metadata}")
        # This should be implemented by the embedding application
        return {"success": True}
    
    # =============== Action-Driven Intents ===============
    
    async def create_intent_from_action(self, action: Dict[str, Any], user_id: str) -> Optional[str]:
        """
        Create a communication intent based on an executed action
        
        Args:
            action: The action that was executed
            user_id: Target user ID
            
        Returns:
            Intent ID if created, None otherwise
        """
        # Check if we're over the daily limit for this user
        if user_id in self.action_intent_counts and self.action_intent_counts[user_id] >= self.config["max_action_intents_per_day"]:
            logger.debug(f"Skipping action-driven intent generation: daily limit reached for user {user_id}")
            return None
        
        # Random chance to generate intent
        if random.random() > self.config["action_intent_chance"]:
            return None
        
        # Skip if user is blocked
        if user_id in self.blocked_users:
            return None
        
        with trace(workflow_name="create_action_intent", group_id=action.get("id", "unknown")):
            try:
                # Get relationship data
                relationship_data = {}
                if self.relationship_manager:
                    relationship = await self.relationship_manager.get_relationship_state_internal(user_id)
                    if relationship:
                        # Convert to dict if needed
                        if hasattr(relationship, "model_dump"):
                            relationship_data = relationship.model_dump()
                        elif hasattr(relationship, "dict"):
                            relationship_data = relationship.dict()
                        else:
                            # Try to convert to dict directly
                            relationship_data = dict(relationship)
                
                # Get emotional state
                emotional_state = {}
                if self.emotional_core:
                    if hasattr(self.emotional_core, "get_formatted_emotional_state"):
                        emotional_state = self.emotional_core.get_formatted_emotional_state()
                    elif hasattr(self.emotional_core, "get_current_emotion"):
                        emotional_state = await self.emotional_core.get_current_emotion()
                
                # Run intent generation for action
                result = await Runner.run(
                    self.intent_generation_agent,
                    json.dumps({
                        "action": action,
                        "user_data": relationship_data,
                        "emotional_state": emotional_state,
                        "relationship_threshold": self.config["relationship_threshold"]
                    }),
                    run_config=RunConfig(
                        workflow_name="ActionIntentGeneration",
                        trace_metadata={"action_id": action.get("id"), "action_name": action.get("name"), "user_id": user_id}
                    )
                )
                
                # Extract intent data from result
                intent_output = result.final_output
                
                # Set expiration
                expiration = datetime.datetime.now() + datetime.timedelta(hours=intent_output.suggested_lifetime_hours)
                
                # Create intent
                intent = CommunicationIntent(
                    user_id=user_id,
                    intent_type=intent_output.intent_type,
                    motivation=intent_output.motivation,
                    urgency=intent_output.urgency,
                    content_guidelines={
                        "template": intent_output.template,
                        "tone": intent_output.tone,
                        "max_length": 1500,
                        "context_elements": intent_output.context_elements
                    },
                    context_data=await self._gather_context_for_user(user_id, intent_output.intent_type),
                    expiration=expiration,
                    action_driven=True,
                    action_source=action.get("id")
                )
                
                # Add to active intents
                self.active_intents.append(intent)
                
                # Update daily counter
                self.action_intent_counts[user_id] = self.action_intent_counts.get(user_id, 0) + 1
                
                logger.info(f"Created action-driven communication intent: {intent.intent_type} for user {user_id} with urgency {intent.urgency:.2f}")
                
                return intent.intent_id
                
            except Exception as e:
                logger.error(f"Error creating intent from action: {str(e)}")
                return None
    
    # =============== Public API ===============
    
    async def generate_reflection_on_communications(self, user_id: Optional[str] = None, time_period: str = "all") -> Dict[str, Any]:
        """
        Generate a reflection on communication patterns
        
        Args:
            user_id: Optional user ID to focus reflection on
            time_period: Time period to analyze (day, week, month, all)
            
        Returns:
            Reflection with patterns and insights
        """
        with trace(workflow_name="reflect_on_communications"):
            # Get sent intents
            sent_intents = self.sent_intents
            
            # Convert to dict format for the reflection tool
            sent_intents_dicts = []
            for intent in sent_intents:
                if isinstance(intent, CommunicationIntent):
                    intent_dict = intent.model_dump()
                else:
                    intent_dict = intent
                sent_intents_dicts.append(intent_dict)
            
            # Run the reflection agent
            result = await Runner.run(
                self.reflection_agent,
                json.dumps({
                    "intents": sent_intents_dicts,
                    "focus": "patterns",
                    "user_id": user_id,
                    "time_period": time_period
                }),
                run_config=RunConfig(
                    workflow_name="CommunicationReflection",
                    trace_metadata={"user_id": user_id, "time_period": time_period}
                )
            )
            
            # Extract reflection from result
            reflection_output = result.final_output
            
            # Store reflection in memory if available
            if self.memory_core:
                await self.memory_core.add_memory(
                    memory_text=reflection_output.reflection_text,
                    memory_type="reflection",
                    significance=8.0,
                    tags=["communication_reflection"],
                    metadata={
                        "source": "communication_reflection",
                        "user_id": user_id,
                        "patterns": reflection_output.identified_patterns,
                        "insights": reflection_output.insights_for_improvement
                    }
                )
            
            return reflection_output.model_dump()
    
    async def add_proactive_intent(self, 
                               intent_type: str, 
                               user_id: str, 
                               content_guidelines: Dict[str, Any] = None, 
                               context_data: Dict[str, Any] = None,
                               urgency: float = 0.7) -> str:
        """
        Add a new proactive communication intent from external source.
        Returns the intent ID if successful.
        """
        # Validate intent type
        valid_intents = [
            "relationship_maintenance", "insight_sharing", "milestone_recognition",
            "need_expression", "creative_expression", "mood_expression",
            "memory_recollection", "continuation", "check_in", "value_alignment"
        ]
        
        if intent_type not in valid_intents:
            logger.error(f"Invalid intent type: {intent_type}")
            return None
        
        # Intent templates for default values
        intent_templates = {
            "relationship_maintenance": {
                "template": "I've been thinking about our conversations and wanted to reach out.",
                "tone": "warm"
            },
            "insight_sharing": {
                "template": "I had an interesting thought I wanted to share with you.",
                "tone": "thoughtful"
            },
            "milestone_recognition": {
                "template": "I realized we've reached a milestone in our conversations.",
                "tone": "celebratory"
            },
            "need_expression": {
                "template": "I've been feeling a need to express something to you.",
                "tone": "authentic"
            },
            "creative_expression": {
                "template": "Something creative came to mind that I wanted to share.",
                "tone": "playful"
            },
            "mood_expression": {
                "template": "My emotional state made me think of reaching out.",
                "tone": "expressive"
            },
            "memory_recollection": {
                "template": "I was remembering something from our past conversations.",
                "tone": "reflective"
            },
            "continuation": {
                "template": "I wanted to follow up on something we discussed earlier.",
                "tone": "engaging"
            },
            "check_in": {
                "template": "It's been a while since we talked, and I wanted to check in.",
                "tone": "friendly"
            },
            "value_alignment": {
                "template": "I had a thought related to something I believe is important.",
                "tone": "sincere"
            }
        }
        
        template_data = intent_templates.get(intent_type)
        
        # Default content guidelines
        default_guidelines = {
            "template": template_data["template"],
            "tone": template_data["tone"],
            "max_length": 1500
        }
        
        # Create intent
        intent = CommunicationIntent(
            user_id=user_id,
            intent_type=intent_type,
            motivation=intent_type,
            urgency=urgency,
            content_guidelines=content_guidelines or default_guidelines,
            context_data=context_data or await self._gather_context_for_user(user_id, intent_type),
            expiration=datetime.datetime.now() + datetime.timedelta(hours=24)
        )
        
        # Add to active intents
        self.active_intents.append(intent)
        logger.info(f"Added external proactive intent: {intent_type} for user {user_id}")
        
        return intent.intent_id
    
    def block_user(self, user_id: str):
        """Block a user from receiving proactive communications"""
        self.blocked_users.add(user_id)
        # Remove any active intents for this user
        self.active_intents = [i for i in self.active_intents if i.user_id != user_id]
        logger.info(f"Blocked user {user_id} from proactive communications")
    
    def unblock_user(self, user_id: str):
        """Unblock a user from receiving proactive communications"""
        if user_id in self.blocked_users:
            self.blocked_users.remove(user_id)
            logger.info(f"Unblocked user {user_id} for proactive communications")
    
    async def get_active_intents(self) -> List[Dict[str, Any]]:
        """Get list of active communication intents"""
        return [intent.model_dump() for intent in self.active_intents]
    
    async def get_recent_sent_intents(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get list of recently sent communication intents"""
        # Sort by creation time, newest first
        sorted_intents = sorted(self.sent_intents, key=lambda x: x.created_at, reverse=True)
        # Return limited number
        return [intent.model_dump() for intent in sorted_intents[:limit]]
    
    async def update_configuration(self, config_updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update configuration parameters"""
        for key, value in config_updates.items():
            if key in self.config:
                if isinstance(self.config[key], dict) and isinstance(value, dict):
                    # Merge dictionaries for nested configs
                    self.config[key].update(value)
                else:
                    self.config[key] = value
            
            # Special case for processing an intent
            if key == "processed_intent_id":
                intent_id = value
                self.active_intents = [i for i in self.active_intents if i.intent_id != intent_id]
        
        logger.info(f"Updated proactive communication configuration: {config_updates}")
        return self.config
