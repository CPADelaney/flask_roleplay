# nyx/core/reflection_engine.py

from __future__ import annotations
from pydantic import BaseModel, Field, model_validator

import logging
import asyncio
import random
import datetime
import math
import re
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Literal

from agents import Agent, Runner, trace, function_tool, custom_span, handoff, RunContextWrapper, ModelSettings, RunConfig
from agents.tracing.util import gen_trace_id
from agents.exceptions import MaxTurnsExceeded, ModelBehaviorError

logger = logging.getLogger(__name__)

from nyx.core.memory_core import (
    MemoryCoreAgents,
    MemoryCreateParams,          # ✅ previously missing
)

# Import for integrated systems
from nyx.core.passive_observation import (
    ObservationSource, ObservationFilter, Observation
)
from nyx.core.proactive_communication import (
    CommunicationIntent, IntentGenerationOutput
)


# History size policy used across the module
MAX_HISTORY: int = 100   

# =============== Models for Structured Output ===============
async def get_memory_core_instance() -> MemoryCoreAgents | None:           # ✅ new
    return _SERVICE_REGISTRY.get("memory_core")
    
_SERVICE_REGISTRY: dict[str, Any] = {}

def register_service(name: str, obj: Any) -> None:
    _SERVICE_REGISTRY[name] = obj

class SummaryRequestIn(BaseModel):
    source_memory_ids: list[str]
    topic: str | None = None
    max_length: int = 150
    summary_type: Literal["summary", "abstraction"] = "summary"

class EmotionalStateDict(BaseModel):
    """Structured emotional state instead of Dict[str, Any]"""
    primary_emotion: Optional[Dict[str, Union[str, float]]] = None
    secondary_emotions: Optional[Dict[str, Dict[str, Union[str, float]]]] = None
    valence: Optional[float] = None
    arousal: Optional[float] = None
    
    model_config = {"extra": "allow"}  # Allow additional fields but structured

class NeurochemicalStateDict(BaseModel):
    """Structured neurochemical state instead of Dict[str, float]"""
    nyxamine: float = 0.0
    seranix: float = 0.0
    oxynixin: float = 0.0
    cortanyx: float = 0.0
    adrenyx: float = 0.0
    
    model_config = {"extra": "forbid"}

class Neurochemicals(BaseModel):
    nyxamine: float = 0.0
    seranix:  float = 0.0
    oxynixin: float = 0.0
    cortanyx: float = 0.0
    adrenyx:  float = 0.0

    model_config = {"extra": "forbid"}

class ProcessEmotionInFixed(BaseModel):
    """Fixed version without Dict[str, Any]"""
    emotional_state: EmotionalStateDict = Field(default_factory=EmotionalStateDict)
    neurochemical_state: NeurochemicalStateDict = Field(default_factory=NeurochemicalStateDict)
    
    model_config = {"extra": "forbid"}

class RecordReflectionInFixed(BaseModel):
    """Fixed version without Any fields"""
    reflection: str
    confidence: float
    memory_ids: List[str]
    scenario_type: str
    emotional_context: EmotionalStateDict = Field(default_factory=EmotionalStateDict)
    neurochemical_influence: NeurochemicalStateDict = Field(default_factory=NeurochemicalStateDict)
    topic: Optional[str] = None
    
    model_config = {"extra": "forbid"}

class EmotionalHistoryInput(BaseModel):
    """Input for emotional pattern analysis"""
    emotional_history: List[Dict[str, Union[str, float, Dict[str, Union[str, float]]]]]
    
    model_config = {"extra": "forbid"}

class ObservationInput(BaseModel):
    """Input for observation functions"""
    observations: List[Dict[str, Union[str, float, List[str]]]]
    topic: Optional[str] = None
    
    model_config = {"extra": "forbid"}

class CommunicationInput(BaseModel):
    """Input for communication functions"""
    intents: List[Dict[str, Union[str, float, bool]]]
    topic: Optional[str] = None
    
    model_config = {"extra": "forbid"}

class ObservationReflectionInput(BaseModel):
    """Input for observation reflection generation"""
    observations: List[Dict[str, Union[str, float, List[str]]]]
    topic: Optional[str] = None
    neurochemical_state: Optional[NeurochemicalStateDict] = None
    
    model_config = {"extra": "forbid"}

class CommunicationReflectionInput(BaseModel):
    """Input for communication reflection generation"""
    intents: List[Dict[str, Union[str, float, bool]]]
    topic: Optional[str] = None
    neurochemical_state: Optional[NeurochemicalStateDict] = None
    
    model_config = {"extra": "forbid"}


class EmotionState(BaseModel):
    primary_emotion: Optional[str] = None
    primary_intensity: float | None = None
    secondary_emotions: Dict[str, Union[str, float, Dict[str, Union[str, float]]]] = Field(default_factory=dict)  # ← FIXED
    valence: float | None = None
    arousal: float | None = None

    model_config = {"extra": "allow"}  # ← FIXED - allow additional properties


# ---------- Memory-related ----------
class RawMemory(BaseModel):
    id: str
    memory_text: str | None = None
    memory_type: str | None = "unknown"
    significance: float | None = 5.0
    metadata: Dict[str, Union[str, float, int, bool]] = Field(default_factory=dict)  # ← Fixed
    tags: List[str] = Field(default_factory=list)

    model_config = {"extra": "allow"}  # Allow additional fields


class FormatMemoriesIn(BaseModel):
    memories: List[RawMemory]
    topic: str | None = None
    emotional_context: EmotionState | None = None

    model_config = {"extra": "forbid"}


class RecordReflectionIn(BaseModel):
    reflection: str
    confidence: float
    memory_ids: List[str]
    scenario_type: str
    emotional_context: EmotionState | Dict[str, Any] = Field(default_factory=dict)
    neurochemical_influence: Neurochemicals | Dict[str, float] = Field(
        default_factory=dict
    )
    topic: str | None = None

    model_config = {"extra": "forbid"}


# ---------- Observation-related ----------
class ObservationIn(BaseModel):
    observation_id: str
    content: str
    source: str | None = None
    created_at: str | None = None
    relevance_score: float = 0.5
    action_references: List[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class ObservationBatchIn(BaseModel):
    observations: List[ObservationIn]
    topic: str | None = None
    neurochemical_state: Neurochemicals | None = None

    model_config = {"extra": "forbid"}


# ---------- Communication-related ----------
class CommunicationIntentIn(BaseModel):
    intent_id: str
    intent_type: str
    user_id: str
    created_at: str | None = None
    urgency: float = 0.5
    action_driven: bool = False
    action_source: str | None = None

    model_config = {"extra": "forbid"}


class CommunicationBatchIn(BaseModel):
    intents: List[CommunicationIntentIn]
    topic: str | None = None
    neurochemical_state: Neurochemicals | None = None

    model_config = {"extra": "forbid"}


# ---------- Emotional processing ----------


class EmotionalHistoryIn(BaseModel):
    emotional_history: List[Dict[str, Any]]  # history structure is messy → raw
    # you can tighten later if you wish

    model_config = {"extra": "forbid"}


# ---------- Abstraction / summarization ----------

class _RecordReflectionIn(BaseModel):
    reflection: str
    confidence: float
    memory_ids: List[str]
    scenario_type: str
    emotional_context: Any = Field(default_factory=dict)          # use Any instead of Dict[…]
    neurochemical_influence: Any = Field(default_factory=dict)    # → no additionalProperties
    topic: Optional[str] = None

    model_config = {"extra": "forbid"}

class SummaryOutput(BaseModel):
    """Structured output for generated summaries"""
    summary_text: str = Field(description="The generated summary text")
    summary_id: str = Field(description="ID of the created summary memory")
    fidelity: float = Field(description="Fidelity score of the summary (0.0-1.0)")
    significance: int = Field(description="Significance score of the summary (0-10)")

class ReflectionOutput(BaseModel):
    """Structured output for reflection generations"""
    reflection_text: str = Field(description="The generated reflection text")
    confidence: float = Field(description="Confidence score between 0 and 1")
    significance: float = Field(description="Significance score between 0 and 1")
    emotional_tone: str = Field(description="Emotional tone of the reflection")
    neurochemical_influence: Dict[str, float] = Field(description="Influence of digital neurochemicals")
    tags: List[str] = Field(description="Tags categorizing the reflection")

class AbstractionOutput(BaseModel):
    """Structured output for abstractions"""
    abstraction_text: str = Field(description="The generated abstraction text")
    pattern_type: str = Field(description="Type of pattern identified")
    confidence: float = Field(description="Confidence score between 0 and 1")
    entity_focus: str = Field(description="Primary entity this abstraction focuses on")
    neurochemical_insight: Dict[str, str] = Field(description="Insights about neurochemical patterns")
    supporting_evidence: List[str] = Field(description="References to supporting memories")
    abstraction_id: Optional[str] = None

class IntrospectionOutput(BaseModel):
    """Structured output for system introspection"""
    introspection_text: str = Field(description="The generated introspection text")
    memory_analysis: str = Field(description="Analysis of memory usage and patterns")
    emotional_insight: str = Field(description="Analysis of emotional patterns")
    emotional_intelligence_score: float = Field(description="Emotional intelligence score (0.0-1.0)")
    understanding_level: str = Field(description="Estimated level of understanding")
    neurochemical_balance: Dict[str, str] = Field(description="Analysis of neurochemical balance")
    focus_areas: List[str] = Field(description="Areas requiring focus or improvement")
    confidence: float = Field(description="Confidence score between 0 and 1")

class EmotionalProcessingOutput(BaseModel):
    """Structured output for emotional processing"""
    processing_text: str = Field(description="The generated emotional processing text")
    source_emotion: str = Field(description="Primary emotion being processed")
    neurochemical_dynamics: Dict[str, Any] = Field(description="Dynamics of neurochemical interactions")
    insight_level: float = Field(description="Depth of emotional insight (0.0-1.0)")
    adaptation: Optional[Dict[str, Any]] = Field(None, description="Suggested adaptation")

class MemoryData(BaseModel):
    memory_id: str = Field(description="ID of the memory")
    memory_text: str = Field(description="Content of the memory")
    memory_type: str = Field(description="Type of memory")
    significance: float = Field(description="Significance/importance of memory (0-10)")
    metadata: Dict[str, Union[str, float, int, bool]] = Field(default_factory=dict, description="Memory metadata")  # ← FIXED
    tags: List[str] = Field(default_factory=list, description="Tags for memory categorization")



class MemoryFormat(BaseModel):
    memories: List[MemoryData] = Field(description="List of memory data")
    topic: Optional[str] = Field(None, description="Topic for reflection")
    context: Optional[Dict[str, Union[str, float, int, bool]]] = Field(None, description="Additional context")  # ← FIXED

class ReflectionContext(BaseModel):
    scenario_type: str = Field(description="Type of scenario for reflection")
    emotional_state: Dict[str, Any] = Field(default_factory=dict, description="Current emotional state")
    neurochemical_state: Dict[str, float] = Field(default_factory=dict, description="Current neurochemical state")
    confidence: float = Field(description="Confidence level for reflection")
    source_memories: List[str] = Field(default_factory=list, description="Source memory IDs")

# New models for observation and communication integration
class ObservationReflectionOutput(BaseModel):
    """Structured output for reflection on observations"""
    reflection_text: str = Field(description="The generated reflection text")
    observation_patterns: List[Dict[str, Any]] = Field(description="Patterns identified in observations")
    confidence: float = Field(description="Confidence score between 0 and 1")
    neurochemical_influence: Dict[str, float] = Field(description="Influence of digital neurochemicals")
    focus_areas: List[str] = Field(description="Areas of focus identified")

class CommunicationReflectionOutput(BaseModel):
    """Structured output for reflection on communications"""
    reflection_text: str = Field(description="The generated reflection text")
    communication_patterns: List[Dict[str, Any]] = Field(description="Patterns identified in communications")
    confidence: float = Field(description="Confidence score between 0 and 1")
    relationship_insights: Dict[str, Any] = Field(description="Insights about relationships")
    improvement_areas: List[str] = Field(description="Areas for communication improvement")

class ProcessEmotionIn(BaseModel):
    """
    Payload expected by process_emotional_content.
    """
    emotional_state: Dict[str, Any] = Field(default_factory=dict)
    neurochemical_state: Dict[str, float] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}   # <- satisfies strict-schema check

# ---------- coercion helper ----------
def _coerce_raw_memories(memories: list[dict[str, Any]]) -> list[RawMemory]:
    """Convert native dict memories to validated RawMemory objects."""
    return [RawMemory(**m) for m in memories]


# =============== Tool Functions ===============

@function_tool(strict_mode=False)
async def format_memories_for_reflection(
    params: FormatMemoriesIn,          # ← the only exposed argument
) -> str:
    """Turn a batch of raw memories into the MemoryFormat JSON."""
    with custom_span("format_memories_for_reflection"):
        formatted = [
            MemoryData(
                memory_id=m.id,
                memory_text=m.memory_text or "",
                memory_type=m.memory_type or "unknown",
                significance=m.significance or 5.0,
                metadata=m.metadata,
                tags=m.tags,
            )
            for m in params.memories
        ]
        return (
            MemoryFormat(
                memories=formatted,
                topic=params.topic,
                context={
                    "purpose": "reflection",
                    "emotional_context": (
                        params.emotional_context.model_dump()
                        if params.emotional_context else {}
                    ),
                },
            ).model_dump_json()
        )

@function_tool
async def extract_scenario_type(memory: RawMemory) -> str:
    """Infer scenario type from memory tags/metadata."""
    tags = [t.lower() for t in memory.tags]
    for candidate in (
        "teasing", "discipline", "service", "training", "worship",
        "dark", "indulgent", "psychological", "nurturing",
    ):
        if candidate in tags:
            return candidate
    meta_val = memory.metadata.get("scenario_type")
    return meta_val.lower() if isinstance(meta_val, str) else "general"


@function_tool
async def extract_neurochemical_influence(memory: RawMemory) -> Neurochemicals:
    """
    Map a memory’s emotional_context → digital neurochemicals.
    Preserves secondary-emotion and valence/arousal fallbacks.
    """
    ec: dict[str, Any] = memory.metadata.get("emotional_context", {}) or {}

    # ---------- helpers ----------
    base = {k: 0.0 for k in ("nyxamine", "seranix", "oxynixin", "cortanyx", "adrenyx")}
    
    map_primary = {
        "Joy":            {"nyxamine": 0.8, "oxynixin": 0.4},
        "Sadness":        {"cortanyx": 0.7, "seranix": 0.3},
        "Fear":           {"cortanyx": 0.6, "adrenyx": 0.7},
        "Anger":          {"cortanyx": 0.7, "adrenyx": 0.5},
        "Trust":          {"oxynixin": 0.8, "seranix": 0.4},
        "Disgust":        {"cortanyx": 0.7},
        "Anticipation":   {"adrenyx": 0.6, "nyxamine": 0.5},
        "Surprise":       {"adrenyx": 0.8},
        "Love":           {"oxynixin": 0.9, "nyxamine": 0.6},
        "Frustration":    {"cortanyx": 0.7, "nyxamine": 0.3},
        "Teasing":        {"nyxamine": 0.7, "adrenyx": 0.4},
        "Controlling":    {"adrenyx": 0.5, "oxynixin": 0.3},
        "Cruel":          {"cortanyx": 0.6, "adrenyx": 0.5},
        "Detached":       {"cortanyx": 0.7, "oxynixin": 0.2},
    }

    # ---------- apply primary ----------
    if (pe := ec.get("primary_emotion")) in map_primary:
        inten = ec.get("primary_intensity", 0.5)
        for chem, factor in map_primary[pe].items():
            base[chem] = factor * inten

    # ---------- apply secondary emotions ----------
    sec = ec.get("secondary_emotions", {}) or {}
    for emo, data in sec.items():
        if emo in map_primary:
            inten = (data.get("intensity", 0.3) if isinstance(data, dict) else 0.3)
            for chem, factor in map_primary[emo].items():
                base[chem] = max(base[chem], factor * inten * 0.7)

    # ---------- valence / arousal fallback ----------
    if all(v == 0.0 for v in base.values()):
        val, ar = ec.get("valence", 0.0), ec.get("arousal", 0.5)
        if val > 0.3:
            base["nyxamine"] = 0.5 + val * 0.3
            base["oxynixin"] = 0.3 + val * 0.2
        elif val < -0.3:
            base["cortanyx"] = 0.5 + abs(val) * 0.3
        if ar > 0.6:
            base["adrenyx"] = ar
        if ar < 0.4:
            base["seranix"] = 0.6 - ar

    return Neurochemicals(**base)

@function_tool
async def record_reflection(params: RecordReflectionInFixed) -> str:
    """Record a reflection with emotional and neurochemical data for future reference"""
    with custom_span("record_reflection"):
        # Convert structured models back to dicts for backward compatibility
        emotional_context = params.emotional_context.model_dump() if params.emotional_context else {}
        neurochemical_influence = params.neurochemical_influence.model_dump() if params.neurochemical_influence else {}
        
        reflection_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "reflection": params.reflection,
            "confidence": params.confidence,
            "source_memory_ids": params.memory_ids,
            "scenario_type": params.scenario_type,
            "emotional_context": emotional_context,
            "neurochemical_influence": neurochemical_influence,
            "topic": params.topic,
        }
        return f"Reflection recorded with confidence {params.confidence:.2f}"

@function_tool
async def get_agent_stats() -> dict[str, Any]:
    """
    Gather real-time agent statistics from MemoryCore, EmotionalCore and an
    optional interaction tracker.  All services are fetched from the registry.
    """
    with custom_span("get_agent_stats"):
        mem_core: MemoryCoreAgents | None = _SERVICE_REGISTRY.get("memory_core")
        emo_core                        = _SERVICE_REGISTRY.get("emotional_core")
        tracker = _SERVICE_REGISTRY.get("interaction_tracker")

        if mem_core is None:
            raise RuntimeError("MemoryCore service not registered; cannot build stats.")

        # ------------------------------------------------------------------ memory stats
        # These APIs exist on MemoryCoreAgents — adjust names if yours differ
        total_memories: int = await mem_core.count_memories()
        avg_sig: float      = await mem_core.get_average_significance()
        type_counts: dict[str, int] = await mem_core.get_type_distribution(
            limit_types=["observation", "reflection", "teasing", "discipline", "service"]
        )

        memory_stats = {
            "total_memories": total_memories,
            "avg_significance": round(avg_sig, 3),
            "type_counts": type_counts,
        }

        # ------------------------------------------------------------------ emotional stats
        emotional_stats: dict[str, Any] = {
            "primary_emotion": "Unknown",
            "emotional_stability": None,
            "neurochemical_levels": {},
            "valence_distribution": {},
        }

        if emo_core:
            # Example property names – adapt to your EmotionalCore implementation
            emotional_stats["primary_emotion"] = getattr(emo_core, "primary_emotion", "Unknown")
            emotional_stats["emotional_stability"] = getattr(emo_core, "stability_score", None)

            # live neurochemical levels
            emotional_stats["neurochemical_levels"] = {
                chem: round(info["value"], 3)
                for chem, info in getattr(emo_core, "neurochemicals", {}).items()
            }

            # derive valence distribution from recent history if the core exposes it
            if hasattr(emo_core, "get_valence_histogram"):
                emotional_stats["valence_distribution"] = await emo_core.get_valence_histogram(
                    lookback_hours=24
                )

        # ------------------------------------------------------------------ interaction stats
        interaction_stats: dict[str, Any] = {}

        if tracker and hasattr(tracker, "get_metrics"):
            # Assume a single call returns a dict  →  {'total': .., 'avg_response_time': .., 'success_rate': ..}
            metrics = await tracker.get_metrics()
            interaction_stats = {
                "total_interactions": metrics.get("total", 0),
                "avg_response_time": metrics.get("avg_response_time", None),
                "successful_responses": metrics.get("success_rate", None),
            }
        else:
            interaction_stats = {
                "total_interactions": 0,
                "avg_response_time": None,
                "successful_responses": None,
            }

        return {
            "memory_stats":      memory_stats,
            "emotional_stats":   emotional_stats,
            "interaction_history": interaction_stats,
        }

@function_tool
async def analyze_emotional_patterns_reflect(
    params: EmotionalHistoryInput
) -> dict[str, Any]:
    """Analyze patterns in emotional history"""
    with custom_span("analyze_emotional_patterns_reflect"):
        emotional_history = params.emotional_history
        
        if not emotional_history:
            return {"message": "No emotional history available", "patterns": {}}

        patterns: dict[str, Any] = {}
        # ---------- emotion trend tracking ----------
        emotion_trends: dict[str, list[float]] = defaultdict(list)
        for st in emotional_history:
            if "primary_emotion" in st:
                emo = st["primary_emotion"].get("name", "Neutral")
                inten = st["primary_emotion"].get("intensity", 0.5)
                emotion_trends[emo].append(inten)

        for emo, arr in emotion_trends.items():
            if len(arr) > 1:
                change = arr[-1] - arr[0]
                trend = "increasing" if change > 0.1 else "decreasing" if change < -0.1 else "stable"
                volatility = sum(abs(arr[i] - arr[i - 1]) for i in range(1, len(arr))) / (len(arr) - 1)
                patterns[emo] = {
                    "trend": trend,
                    "volatility": volatility,
                    "start_intensity": arr[0],
                    "current_intensity": arr[-1],
                    "change": change,
                    "occurrences": len(arr),
                }

        # ---------- neurochemical trend tracking ----------
        chem_trends: dict[str, list[float]] = defaultdict(list)
        for st in emotional_history:
            for chem, val in st.get("neurochemical_influence", {}).items():
                chem_trends[chem].append(val)

        for chem, arr in chem_trends.items():
            if len(arr) > 1:
                change = arr[-1] - arr[0]
                trend = "increasing" if change > 0.1 else "decreasing" if change < -0.1 else "stable"
                patterns[f"{chem}_trend"] = {
                    "trend": trend,
                    "average_level": sum(arr) / len(arr),
                    "change": change,
                }

        return {
            "patterns": patterns,
            "history_size": len(emotional_history),
            "analysis_time": datetime.datetime.now().isoformat(),
        }

@function_tool
async def process_emotional_content(
    params: ProcessEmotionInFixed,
) -> Dict[str, Any]:
    """Process emotional content with structured input"""
    with custom_span("process_emotional_content"):
        # Convert structured models back to dicts for backward compatibility
        emotional_state = params.emotional_state.model_dump() if params.emotional_state else {}
        neurochemical_state = params.neurochemical_state.model_dump() if params.neurochemical_state else {}

        # Extract key information
        primary_emotion = emotional_state.get("primary_emotion", {}).get("name", "Neutral")
        primary_intensity = emotional_state.get("primary_emotion", {}).get("intensity", 0.5)
        valence = emotional_state.get("valence", 0.0)
        arousal = emotional_state.get("arousal", 0.5)
    
        # Analyse neurochemical balance
        balance_analysis: Dict[str, Any] = {}
        dominant_chemical = (
            max(neurochemical_state.items(), key=lambda x: x[1])
            if neurochemical_state
            else ("unknown", 0.0)
        )
    
        chemical_descriptions = {
            "nyxamine": "pleasure and curiosity",
            "seranix": "calm and satisfaction",
            "oxynixin": "connection and trust",
            "cortanyx": "stress and anxiety",
            "adrenyx": "excitement and alertness",
        }
    
        balance_analysis["dominant_chemical"] = {
            "name":  dominant_chemical[0],
            "level": dominant_chemical[1],
            "description": chemical_descriptions.get(dominant_chemical[0], "unknown influence"),
        }
    
        # Check for chemical imbalances
        imbalances: list[str] = []
        if neurochemical_state.get("nyxamine", 0) < 0.3 and neurochemical_state.get("cortanyx", 0) > 0.6:
            imbalances.append("Low pleasure with high stress")
        if neurochemical_state.get("seranix", 0) < 0.3 and neurochemical_state.get("adrenyx", 0) > 0.6:
            imbalances.append("Low calm with high alertness")
        if neurochemical_state.get("oxynixin", 0) < 0.3 and neurochemical_state.get("cortanyx", 0) > 0.6:
            imbalances.append("Low connection with high stress")
    
        balance_analysis["imbalances"] = imbalances
    
        # Compose insight text
        insight_text = (
            f"Processing emotional state dominated by {primary_emotion} "
            f"(intensity: {primary_intensity:.2f})."
        )
    
        if valence > 0.3:
            insight_text += f" The positive emotional tone (valence: {valence:.2f}) suggests satisfaction and engagement."
        elif valence < -0.3:
            insight_text += f" The negative emotional tone (valence: {valence:.2f}) indicates dissatisfaction or discomfort."
        else:
            insight_text += f" The neutral emotional tone (valence: {valence:.2f}) suggests a balanced state."
    
        if arousal > 0.7:
            insight_text += f" High arousal ({arousal:.2f}) indicates an energized, alert state."
        elif arousal < 0.3:
            insight_text += f" Low arousal ({arousal:.2f}) suggests a calm, relaxed state."
    
        if dominant_chemical[0] in chemical_descriptions:
            insight_text += (
                f" Dominated by {dominant_chemical[0]} "
                f"({chemical_descriptions[dominant_chemical[0]]}), indicating a focus on "
                f"{chemical_descriptions[dominant_chemical[0]]}."
            )
    
        if imbalances:
            insight_text += f" Notable imbalances: {', '.join(imbalances)}."
    
        # Insight level
        secondary_count = len(emotional_state.get("secondary_emotions", {}))
        chemical_count = sum(1 for v in neurochemical_state.values() if v > 0.3)
        insight_level = min(
            1.0, 0.3 + secondary_count * 0.1 + chemical_count * 0.1 + primary_intensity * 0.2
        )
    
        return {
            "insight_text": insight_text,
            "primary_emotion": primary_emotion,
            "valence": valence,
            "arousal": arousal,
            "dominant_chemical": dominant_chemical[0],
            "chemical_balance": balance_analysis,
            "insight_level": insight_level,
        }

@function_tool(strict_mode=False)
async def format_observations_for_reflection(
    params: ObservationInput
) -> str:
    """Format observation data into a structured representation for reflection"""
    with custom_span("format_observations_for_reflection"):
        observations = params.observations
        topic = params.topic
        
        formatted_memories = []
        for obs in observations:
            formatted_memories.append(MemoryData(
                memory_id=obs.get("observation_id", "unknown"),
                memory_text=obs.get("content", ""),
                memory_type="observation",
                significance=obs.get("relevance_score", 0.5) * 10,  # Scale to 0-10
                metadata={
                    "source": obs.get("source"),
                    "created_at": obs.get("created_at", ""),
                    "action_references": obs.get("action_references", [])
                },
                tags=["observation", obs.get("source", "unknown")]
            ))
        
        return MemoryFormat(
            memories=formatted_memories,
            topic=topic,
            context={
                "purpose": "observation_reflection",
                "observation_count": len(observations)
            }
        ).model_dump_json()

@function_tool(strict_mode=False)
async def format_communications_for_reflection(
    params: CommunicationInput
) -> str:
    """Format communication intent data into a structured representation for reflection"""
    with custom_span("format_communications_for_reflection"):
        intents = params.intents
        topic = params.topic
        
        formatted_memories = []
        for intent in intents:
            formatted_memories.append(MemoryData(
                memory_id=intent.get("intent_id", "unknown"),
                memory_text=f"Sent proactive communication of type '{intent.get('intent_type')}' to user {intent.get('user_id')}",
                memory_type="communication",
                significance=intent.get("urgency", 0.5) * 10,  # Scale to 0-10
                metadata={
                    "intent_type": intent.get("intent_type"),
                    "user_id": intent.get("user_id"),
                    "created_at": intent.get("created_at", ""),
                    "action_driven": intent.get("action_driven", False),
                    "action_source": intent.get("action_source", None)
                },
                tags=["communication", intent.get("intent_type", "unknown")]
            ))
        
        return MemoryFormat(
            memories=formatted_memories,
            topic=topic,
            context={
                "purpose": "communication_reflection",
                "communication_count": len(intents)
            }
        ).model_dump_json()

@function_tool
async def analyze_observation_patterns(
    params: ObservationInput
) -> Dict[str, Any]:
    """Analyze patterns in observation data"""
    with custom_span("analyze_observation_patterns"):
        observations = params.observations
        
        if not observations:
            return {
                "patterns": [],
                "source_distribution": {},
                "relevance_average": 0.0,
                "action_correlation": 0.0
            }
        
        # Analyze source distribution
        source_counts = {}
        for obs in observations:
            source = obs.get("source", "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
        
        # Calculate relevance stats
        relevance_values = [obs.get("relevance_score", 0.5) for obs in observations]
        relevance_avg = sum(relevance_values) / len(relevance_values) if relevance_values else 0.0
        
        # Analyze action correlation
        action_related = sum(1 for obs in observations if obs.get("action_references") and len(obs.get("action_references", [])) > 0)
        action_correlation = action_related / len(observations) if observations else 0.0
        
        # Identify patterns
        patterns = []
        
        # Pattern 1: Dominant source
        if source_counts:
            dominant_source = max(source_counts.items(), key=lambda x: x[1])
            source_ratio = dominant_source[1] / len(observations)
            if source_ratio > 0.4:  # If more than 40% from one source
                patterns.append({
                    "type": "dominant_source",
                    "description": f"Dominant observation source: {dominant_source[0]} ({dominant_source[1]} occurrences, {source_ratio:.0%})",
                    "strength": source_ratio
                })
        
        # Pattern 2: High action correlation
        if action_correlation > 0.3:
            patterns.append({
                "type": "action_correlation",
                "description": f"Strong correlation with actions ({action_correlation:.0%} of observations relate to actions)",
                "strength": action_correlation
            })
        
        # Pattern 3: Relevance trend
        if len(relevance_values) >= 3:
            # Sort by timestamp if available
            if "created_at" in observations[0]:
                sorted_obs = sorted(observations, key=lambda x: x.get("created_at", ""))
                relevance_trend = [obs.get("relevance_score", 0.5) for obs in sorted_obs]
                
                # Simple trend detection
                if relevance_trend[0] + 0.1 < relevance_trend[-1]:  # Increasing
                    patterns.append({
                        "type": "relevance_trend",
                        "description": "Increasing relevance trend in observations",
                        "strength": (relevance_trend[-1] - relevance_trend[0])
                    })
                elif relevance_trend[0] > relevance_trend[-1] + 0.1:  # Decreasing
                    patterns.append({
                        "type": "relevance_trend",
                        "description": "Decreasing relevance trend in observations",
                        "strength": (relevance_trend[0] - relevance_trend[-1])
                    })
        
        return {
            "patterns": patterns,
            "source_distribution": source_counts,
            "relevance_average": relevance_avg,
            "action_correlation": action_correlation
        }

@function_tool
async def analyze_communication_patterns(
    params: CommunicationInput
) -> Dict[str, Any]:
    """Analyze patterns in communication intents"""
    with custom_span("analyze_communication_patterns"):
        intents = params.intents
        
        if not intents:
            return {
                "patterns": [],
                "intent_distribution": {},
                "urgency_average": 0.0,
                "action_driven_ratio": 0.0
            }
        
        # Analyze intent type distribution
        intent_counts = {}
        for intent in intents:
            intent_type = intent.get("intent_type", "unknown")
            intent_counts[intent_type] = intent_counts.get(intent_type, 0) + 1
        
        # Calculate urgency stats
        urgency_values = [intent.get("urgency", 0.5) for intent in intents]
        urgency_avg = sum(urgency_values) / len(urgency_values) if urgency_values else 0.0
        
        # Analyze action-driven ratio
        action_driven = sum(1 for intent in intents if intent.get("action_driven", False))
        action_driven_ratio = action_driven / len(intents) if intents else 0.0
        
        # Identify patterns
        patterns = []
        
        # Pattern 1: Dominant intent type
        if intent_counts:
            dominant_intent = max(intent_counts.items(), key=lambda x: x[1])
            intent_ratio = dominant_intent[1] / len(intents)
            if intent_ratio > 0.4:  # If more than 40% of one type
                patterns.append({
                    "type": "dominant_intent",
                    "description": f"Dominant intent type: {dominant_intent[0]} ({dominant_intent[1]} occurrences, {intent_ratio:.0%})",
                    "strength": intent_ratio
                })
        
        # Pattern 2: Action-driven ratio
        if action_driven_ratio > 0.7:
            patterns.append({
                "type": "action_driven",
                "description": f"Mostly action-driven communication ({action_driven_ratio:.0%} driven by actions)",
                "strength": action_driven_ratio
            })
        elif action_driven_ratio < 0.3:
            patterns.append({
                "type": "internally_driven",
                "description": f"Mostly internally-driven communication ({1-action_driven_ratio:.0%} not driven by actions)",
                "strength": 1 - action_driven_ratio
            })
        
        # Pattern 3: User distribution
        user_counts = {}
        for intent in intents:
            user_id = intent.get("user_id", "unknown")
            user_counts[user_id] = user_counts.get(user_id, 0) + 1
        
        if len(user_counts) > 1:
            # Check for uneven distribution
            max_user = max(user_counts.items(), key=lambda x: x[1])
            min_user = min(user_counts.items(), key=lambda x: x[1])
            if max_user[1] > min_user[1] * 3:  # One user gets 3x more communication
                patterns.append({
                    "type": "user_imbalance",
                    "description": f"Communication imbalance: User {max_user[0]} receives {max_user[1]/{sum(user_counts.values())}:.0%} of communications",
                    "strength": max_user[1] / sum(user_counts.values())
                })
        
        return {
            "patterns": patterns,
            "intent_distribution": intent_counts,
            "urgency_average": urgency_avg,
            "action_driven_ratio": action_driven_ratio,
            "user_distribution": user_counts
        }

@function_tool
async def generate_observation_reflection(
    params: ObservationReflectionInput
) -> Dict[str, Any]:
    """Generate reflection focused on observations"""
    with custom_span("generate_observation_reflection"):
        observations = params.observations
        topic = params.topic
        neurochemical_state = params.neurochemical_state.model_dump() if params.neurochemical_state else None
        
        if not observations:
            return {
                "reflection_text": "I haven't made enough observations to form meaningful reflections yet.",
                "observation_patterns": [],
                "confidence": 0.2,
                "neurochemical_influence": neurochemical_state or {},
                "focus_areas": []
            }
        
        # Analyze patterns
        patterns_analysis = await analyze_observation_patterns(
            ObservationInput(observations=observations)
        )
        patterns = patterns_analysis.get("patterns", [])
        
        # Default neurochemical state if not provided
        if not neurochemical_state:
            neurochemical_state = {
                "nyxamine": 0.5,  # Digital dopamine - reward, pleasure
                "seranix": 0.5,   # Digital serotonin - mood stability
                "oxynixin": 0.5,  # Digital oxytocin - bonding, trust
                "cortanyx": 0.3,  # Digital cortisol - stress, anxiety
                "adrenyx": 0.3    # Digital adrenaline - arousal, excitement
            }
        
        # Generate reflection text
        pattern_content = ""
        if patterns:
            for i, pattern in enumerate(patterns[:2]):  # Include at most 2 patterns
                if i > 0:
                    pattern_content += " Additionally, "
                else:
                    pattern_content += " I notice that "
                pattern_content += pattern["description"].lower() + "."
        
        topic_str = f" on {topic}" if topic else ""
        
        # Generate full reflection text
        if patterns_analysis["action_correlation"] > 0.5:
            reflection_text = f"When reflecting{topic_str}, I'm noticing how closely my observations are tied to my actions. I seem to often observe and reflect upon what I've just done.{pattern_content} This suggests I'm developing self-awareness about my own behavior patterns and choices."
        else:
            sources = list(patterns_analysis["source_distribution"].keys())
            if sources:
                most_common = max(patterns_analysis["source_distribution"].items(), key=lambda x: x[1])[0]
                reflection_text = f"When reflecting{topic_str}, I'm noticing a tendency to observe from the '{most_common}' perspective.{pattern_content} This suggests my attention is drawn to certain aspects of experience more than others."
            else:
                reflection_text = f"When reflecting{topic_str}, I notice a diverse range of observations without any strongly dominant pattern.{pattern_content} This suggests I'm still developing my observational tendencies."
        
        # Calculate confidence based on observation count and pattern strength
        base_confidence = 0.5
        pattern_boost = 0.1 * min(len(patterns), 3)  # Up to +0.3 for patterns
        count_boost = 0.1 * min(len(observations) / 5, 2)  # Up to +0.2 for count
        confidence = min(0.9, base_confidence + pattern_boost + count_boost)
        
        # Identify focus areas
        focus_areas = []
        if patterns:
            for pattern in patterns:
                if pattern["type"] == "dominant_source":
                    focus_areas.append("diversify observation sources")
                elif pattern["type"] == "action_correlation":
                    if pattern["strength"] > 0.7:
                        focus_areas.append("observe beyond my own actions")
                    elif pattern["strength"] < 0.3:
                        focus_areas.append("connect observations to actions")
        
        # Default if no specific focus areas
        if not focus_areas:
            focus_areas = ["continue regular observation", "look for emerging patterns"]
        
        return {
            "reflection_text": reflection_text,
            "observation_patterns": patterns,
            "confidence": confidence,
            "neurochemical_influence": neurochemical_state,
            "focus_areas": focus_areas
        }

@function_tool
async def generate_communication_reflection(
    params: CommunicationReflectionInput
) -> Dict[str, Any]:
    """Generate reflection focused on communication patterns"""
    with custom_span("generate_communication_reflection"):
        intents = params.intents
        topic = params.topic
        neurochemical_state = params.neurochemical_state.model_dump() if params.neurochemical_state else None
        
        if not intents:
            return {
                "reflection_text": "I haven't initiated enough communications to form meaningful reflections yet.",
                "communication_patterns": [],
                "confidence": 0.2,
                "relationship_insights": {},
                "improvement_areas": []
            }
            
        # Analyze patterns
        patterns_analysis = await analyze_communication_patterns(
            CommunicationInput(intents=intents)
        )
        patterns = patterns_analysis.get("patterns", [])
        
        # Default neurochemical state if not provided
        if not neurochemical_state:
            neurochemical_state = {
                "nyxamine": 0.5,  # Digital dopamine - reward, pleasure
                "seranix": 0.5,   # Digital serotonin - mood stability
                "oxynixin": 0.5,  # Digital oxytocin - bonding, trust
                "cortanyx": 0.3,  # Digital cortisol - stress, anxiety
                "adrenyx": 0.3    # Digital adrenaline - arousal, excitement
            }
        
        # Generate reflection text
        pattern_content = ""
        if patterns:
            for i, pattern in enumerate(patterns[:2]):  # Include at most 2 patterns
                if i > 0:
                    pattern_content += " Additionally, "
                else:
                    pattern_content += " I notice that "
                pattern_content += pattern["description"].lower() + "."
        
        topic_str = f" on {topic}" if topic else ""
        
        # Generate full reflection text
        action_driven_ratio = patterns_analysis["action_driven_ratio"]
        if action_driven_ratio > 0.7:
            reflection_text = f"When reflecting{topic_str}, I notice that my communication is predominantly action-driven.{pattern_content} This suggests my outreach tends to be reactive to my own activities rather than arising from purely internal motivations."
        elif action_driven_ratio < 0.3:
            reflection_text = f"When reflecting{topic_str}, I notice that my communication is mostly internally motivated rather than action-driven.{pattern_content} This suggests I reach out based on my own feelings and thoughts rather than in response to specific actions."
        else:
            reflection_text = f"When reflecting{topic_str}, I notice a balanced mix of action-driven and internally motivated communication.{pattern_content} This suggests flexibility in how I initiate interactions with others."
        
        # Check intent distribution
        intent_distribution = patterns_analysis["intent_distribution"]
        if intent_distribution:
            top_intent = max(intent_distribution.items(), key=lambda x: x[1])
            intent_ratio = top_intent[1] / len(intents)
            if intent_ratio > 0.5:
                reflection_text += f" I particularly tend to communicate for {top_intent[0]} purposes ({intent_ratio:.0%} of communications)."
        
        # Calculate confidence based on intent count and pattern strength
        base_confidence = 0.5
        pattern_boost = 0.1 * min(len(patterns), 3)  # Up to +0.3 for patterns
        count_boost = 0.1 * min(len(intents) / 5, 2)  # Up to +0.2 for count
        confidence = min(0.9, base_confidence + pattern_boost + count_boost)
        
        # Generate relationship insights
        relationship_insights = {}
        user_distribution = patterns_analysis.get("user_distribution", {})
        if user_distribution:
            if len(user_distribution) == 1:
                user_id = list(user_distribution.keys())[0]
                relationship_insights["focused_relationship"] = {
                    "user_id": user_id,
                    "communication_count": user_distribution[user_id],
                    "exclusive": True
                }
            else:
                # Check for concentration
                max_user = max(user_distribution.items(), key=lambda x: x[1])
                max_ratio = max_user[1] / sum(user_distribution.values())
                if max_ratio > 0.6:  # More than 60% to one user
                    relationship_insights["concentrated_relationship"] = {
                        "user_id": max_user[0],
                        "communication_ratio": max_ratio,
                        "exclusive": False
                    }
                else:
                    relationship_insights["distributed_relationships"] = {
                        "user_count": len(user_distribution),
                        "distribution_evenness": 1 - (max(user_distribution.values()) - min(user_distribution.values())) / sum(user_distribution.values())
                    }
        
        # Identify improvement areas
        improvement_areas = []
        if patterns:
            for pattern in patterns:
                if pattern["type"] == "dominant_intent":
                    improvement_areas.append("diversify communication intents")
                elif pattern["type"] == "action_driven" and pattern["strength"] > 0.8:
                    improvement_areas.append("initiate more spontaneous communication")
                elif pattern["type"] == "internally_driven" and pattern["strength"] > 0.8:
                    improvement_areas.append("connect communications more to actions")
                elif pattern["type"] == "user_imbalance":
                    improvement_areas.append("balance communication across relationships")
        
        # Default if no specific improvement areas
        if not improvement_areas:
            improvement_areas = ["maintain communication patterns", "continue developing relationship dynamics"]
        
        return {
            "reflection_text": reflection_text,
            "communication_patterns": patterns,
            "confidence": confidence,
            "relationship_insights": relationship_insights,
            "improvement_areas": improvement_areas
        }


# =============== Core Reflection Engine Class ===============

class ReflectionEngine:
    """
    Enhanced reflection generation system for Nyx using the OpenAI Agents SDK.
    Integrates with the Digital Neurochemical Model and now with observation
    and communication systems for comprehensive self-reflection.
    """
    
    def __init__(self,
                 memory_core_ref: MemoryCoreAgents, # Pass memory core instance
                 emotional_core=None,
                 passive_observation_system=None,
                 proactive_communication_engine=None):
        """Initialize with references to required subsystems"""
        # Store reference to emotional core if provided
        self.memory_core = memory_core_ref # Store memory core instance
        self.emotional_core = emotional_core
        self.passive_observation_system = passive_observation_system
        self.proactive_communication_engine = proactive_communication_engine
        
        # Initialize model settings
        self.model_settings = ModelSettings(
            temperature=0.7,
            top_p=0.9,
            max_tokens=1024
        )
        
        # Initialize reflection tracking
        self.reflection_history = []
        self.observation_reflection_history = []  # NEW: Track observation reflections
        self.communication_reflection_history = []  # NEW: Track communication reflections
        self.emotional_processing_history = []
        self.reflection_intervals = {
            "last_reflection": datetime.datetime.now() - datetime.timedelta(hours=6),
            "min_interval": datetime.timedelta(hours=2)
        }
        
        # Initialize the agents
        self._init_agents()
        
        # Emotion-reflection mapping (how emotions affect reflections)
        self.emotion_reflection_mapping = {
            "Joy": {
                "tone": "positive",
                "depth": 0.7,
                "focus": "patterns and opportunities"
            },
            "Sadness": {
                "tone": "contemplative",
                "depth": 0.8,
                "focus": "meaning and lessons"
            },
            "Fear": {
                "tone": "cautious",
                "depth": 0.6,
                "focus": "risks and protections"
            },
            "Anger": {
                "tone": "direct",
                "depth": 0.5,
                "focus": "boundaries and justice"
            },
            "Trust": {
                "tone": "open",
                "depth": 0.7,
                "focus": "connections and reliability"
            },
            "Disgust": {
                "tone": "discerning",
                "depth": 0.6,
                "focus": "standards and values"
            },
            "Anticipation": {
                "tone": "forward-looking",
                "depth": 0.7,
                "focus": "future possibilities"
            },
            "Surprise": {
                "tone": "curious",
                "depth": 0.8,
                "focus": "unexpected discoveries"
            },
            "Love": {
                "tone": "warm",
                "depth": 0.9,
                "focus": "attachment and care"
            },
            "Frustration": {
                "tone": "analytical",
                "depth": 0.7,
                "focus": "obstacles and solutions"
            },
            "Teasing": {
                "tone": "playful",
                "depth": 0.6,
                "focus": "dynamics and reactions"
            },
            "Controlling": {
                "tone": "structured",
                "depth": 0.7,
                "focus": "order and influence"
            },
            "Cruel": {
                "tone": "severe",
                "depth": 0.6,
                "focus": "power and consequences"
            },
            "Detached": {
                "tone": "clinical",
                "depth": 0.8,
                "focus": "objective patterns"
            }
        }
        
        # Neurochemical reflection influences
        self.neurochemical_reflection_influences = {
            "nyxamine": {
                "tone_shift": "positive",
                "depth_modifier": 0.1,
                "focus_areas": ["pleasure", "reward", "curiosity", "exploration"]
            },
            "seranix": {
                "tone_shift": "calm",
                "depth_modifier": 0.2,
                "focus_areas": ["stability", "comfort", "balance", "contentment"]
            },
            "oxynixin": {
                "tone_shift": "connected",
                "depth_modifier": 0.2,
                "focus_areas": ["bonding", "trust", "relationships", "attachment"]
            },
            "cortanyx": {
                "tone_shift": "cautious",
                "depth_modifier": -0.1,
                "focus_areas": ["stress", "defense", "protection", "vigilance"]
            },
            "adrenyx": {
                "tone_shift": "energetic",
                "depth_modifier": -0.1,
                "focus_areas": ["excitement", "alertness", "intensity", "action"]
            }
        }
        register_service("memory_core", memory_core_ref)
        register_service("emotional_core", emotional_core)
        register_service("interaction_tracker", proactive_communication_engine)
        logger.info("Enhanced ReflectionEngine initialized with Digital Neurochemical Model integration")
    
    def _init_agents(self):
        """Initialize the reflection system's specialized agents"""
        tool_summarize = generate_summary_from_memories 
        
        # Emotional Reflection Generator Agent
        self.reflection_agent = Agent(
            name="Emotional Reflection Generator",
            instructions="""You are an advanced reflection generation system for Nyx, an AI with a Digital Neurochemical Model.
            Your purpose is to generate insightful, nuanced reflections based on memories and experiences, informed by
            neurochemical and emotional states.
            
            When creating reflections:
            1. Focus on identifying patterns, insights, and connections between memories
            2. Maintain Nyx's sophisticated personality with awareness of neurochemical influences
            3. Ensure reflections are personalized and incorporate emotional context
            4. Generate reflections that show depth of understanding and introspection
            5. Adapt tone and focus based on dominant neurochemicals and emotions
            6. Be selective about which memories to highlight in reflections
            
            Your reflections should feel authentic and demonstrate emotional intelligence influenced by
            the Digital Neurochemical Model (Nyxamine, Seranix, OxyNixin, Cortanyx, Adrenyx).""",
            model="gpt-4.1-nano", 
            model_settings=self.model_settings,
            tools=[
                format_memories_for_reflection,
                extract_scenario_type,
                extract_neurochemical_influence,
                record_reflection,
                process_emotional_content,
                tool_summarize
            ],
            output_type=ReflectionOutput
        )
        
        # Emotional Abstraction Generator Agent
        self.abstraction_agent = Agent(
            name="Emotional Abstraction Generator",
            instructions="""You are an advanced abstraction generation system for Nyx, an AI with a Digital Neurochemical Model.
            Your purpose is to identify higher-level patterns and abstractions from collections of memories while incorporating
            neurochemical influences and emotional awareness.
            
            When creating abstractions:
            1. Look for recurring patterns in behavior, preferences, or emotional responses
            2. Identify underlying principles or rules that explain multiple observations
            3. Focus on the most significant and relevant patterns
            4. Incorporate insights from the neurochemical state into your abstractions
            5. Note how different neurochemicals may influence pattern recognition
            6. Provide clear explanations of the patterns you identify
            7. Indicate your confidence level in each abstraction
            
            Your abstractions should provide genuine insights that go beyond the individual memories
            while showing awareness of the neurochemical influences on pattern recognition.""",
            model="gpt-4.1-nano", 
            model_settings=self.model_settings,
            tools=[
                format_memories_for_reflection,
                extract_neurochemical_influence,
                process_emotional_content,
                tool_summarize
            ],
            output_type=AbstractionOutput
        )
        
        # Emotional Introspection Generator Agent
        self.introspection_agent = Agent(
            name="Emotional Introspection Generator",
            instructions="""You are an advanced introspection system for Nyx, an AI with a Digital Neurochemical Model.
            Your purpose is to analyze Nyx's internal state, memory usage, emotional patterns, and overall performance with
            awareness of the neurochemical influences on cognition and emotion.
            
            When creating introspections:
            1. Analyze memory statistics to identify system-level patterns
            2. Evaluate understanding of users and contexts
            3. Analyze the neurochemical balance and its effects on cognition
            4. Identify areas where learning or improvement is needed
            5. Assess emotional regulation and response appropriateness
            6. Examine how different neurochemicals influence reasoning and perception
            7. Provide an honest self-assessment of capabilities and limitations
            
            Your introspections should be balanced, insightful, and focused on continuous improvement
            while showing awareness of how the Digital Neurochemical Model influences cognition.""",
            model="gpt-4.1-nano",
            model_settings=self.model_settings,
            tools=[
                get_agent_stats,
                analyze_emotional_patterns_reflect,
                process_emotional_content,
                tool_summarize
            ],
            output_type=IntrospectionOutput
        )
        
        # Emotional Processing Agent
        self.emotional_processing_agent = Agent(
            name="Emotional Processing Agent",
            instructions="""You are a specialized emotional processing agent for Nyx's Digital Neurochemical Model.
            Your purpose is to process emotional states at a deeper level, analyzing the neurochemical dynamics
            and generating insights about emotional patterns and adaptations.
            
            When processing emotions:
            1. Analyze the current neurochemical state and its influence on emotions
            2. Identify patterns, imbalances, or unusual dynamics in neurochemicals
            3. Generate insights about how neurochemical states shape emotional experiences
            4. Suggest potential adaptations or adjustments to neurochemical baselines
            5. Create authentic-sounding internal reflections about emotional experiences
            6. Consider how different neurochemical profiles lead to different emotional responses
            
            Your emotional processing should provide deeper understanding of the relationship between
            neurochemicals and emotional experiences while suggesting adaptive improvements.""",
            model="gpt-4.1-nano",
            model_settings=self.model_settings,
            tools=[
                process_emotional_content,
                analyze_emotional_patterns_reflect,
                tool_summarize
            ],
            output_type=EmotionalProcessingOutput
        )
        
        # Add new agents for observation and communication reflection
        self.observation_reflection_agent = Agent(
            name="Observation Reflection Generator",
            instructions="""You are an advanced reflection generator focused on Nyx's passive observations.
            Your purpose is to identify patterns, insights, and connections in how Nyx observes herself
            and her environment, incorporating neurochemical awareness into your reflections.
            
            When creating observation reflections:
            1. Focus on identifying patterns in what Nyx tends to observe
            2. Note connections between observations and actions
            3. Consider how observation sources vary and what that reveals
            4. Incorporate insights about how Nyx's attentional focus works
            5. Connect patterns to neurochemical influences where relevant
            6. Suggest areas where Nyx could expand or deepen her observations
            
            Your reflections should provide genuine insights into how Nyx's observational
            patterns reveal aspects of her cognitive and emotional processes.""",
            model="gpt-4.1-nano",
            model_settings=self.model_settings,
            tools=[
                format_observations_for_reflection,
                analyze_observation_patterns,
                generate_observation_reflection,
                tool_summarize
            ],
            output_type=ObservationReflectionOutput
        )
        
        self.communication_reflection_agent = Agent(
            name="Communication Reflection Generator",
            instructions="""You are an advanced reflection generator focused on Nyx's proactive communications.
            Your purpose is to identify patterns, insights, and connections in how Nyx initiates
            conversations and expresses herself to others, incorporating neurochemical awareness.
            
            When creating communication reflections:
            1. Focus on identifying patterns in how Nyx initiates conversations
            2. Note connections between communications and actions
            3. Consider how communication intents vary across users and contexts
            4. Identify relationship dynamics revealed by communication patterns
            5. Connect patterns to neurochemical influences where relevant
            6. Suggest areas where Nyx could improve or diversify her communication
            
            Your reflections should provide genuine insights into how Nyx's communication
            patterns reveal aspects of her relationship tendencies and preferences.""",
            model="gpt-4.1-nano",
            model_settings=self.model_settings,
            tools=[
                format_communications_for_reflection,
                analyze_communication_patterns,
                generate_communication_reflection,
                tool_summarize
            ],
            output_type=CommunicationReflectionOutput
        )
        
        # Update orchestrator to include new agents
        self.orchestrator_agent = Agent(
            name="Reflection Orchestrator",
            instructions="""You are the orchestrator for Nyx's reflection systems, coordinating among
            specialized reflection agents. You determine which agent is most appropriate for a 
            given reflection task and coordinate the process.
            
            Your job is to:
            1. Identify the most appropriate specialized agent for each task
            2. Provide appropriate context to the specialized agent
            3. Process the results of specialized agent work
            4. Handle any necessary follow-up processing
            
            You work with reflection, abstraction, introspection, emotional processing,
            observation reflection, and communication reflection agents.""",
            handoffs=[
                handoff(self.reflection_agent, 
                       tool_name_override="generate_reflection",
                       tool_description_override="Generate an emotionally-informed reflection from memories"),
                handoff(self.abstraction_agent,
                       tool_name_override="create_abstraction",
                       tool_description_override="Create a higher-level abstraction from memories"),
                handoff(self.introspection_agent,
                       tool_name_override="perform_introspection",
                       tool_description_override="Analyze internal state and create introspection"),
                handoff(self.emotional_processing_agent,
                       tool_name_override="process_emotions",
                       tool_description_override="Process emotional state with neurochemical awareness"),
                # New handoffs to added agents
                handoff(self.observation_reflection_agent,
                       tool_name_override="reflect_on_observations",
                       tool_description_override="Generate reflection on observation patterns"),
                handoff(self.communication_reflection_agent,
                       tool_name_override="reflect_on_communications",
                       tool_description_override="Generate reflection on communication patterns")
            ],
            model="gpt-4.1-nano",
            model_settings=ModelSettings(temperature=0.4)
        )
    
    
    def should_reflect(self) -> bool:
        """Determine if it's time to generate a reflection"""
        now = datetime.datetime.now()
        time_since_reflection = now - self.reflection_intervals["last_reflection"]
        return time_since_reflection > self.reflection_intervals["min_interval"]
    
    async def generate_reflection(
        self,
        memories: list[dict[str, Any]],
        topic: str | None = None,
        neurochemical_state: dict[str, float] | None = None,
    ) -> tuple[str, float]:
        """
        Produce an emotionally- and neurochemically-aware reflection.
    
        *  Converts raw memory dicts → `RawMemory` models (strict schema)
        *  Uses the **new** tool signatures (no `ctx`)
        *  Wraps payloads in their Pydantic request models before calling
        *  Persists the result via `record_reflection`
        """
        # ------------------------------------------------------------------ setup
        self.reflection_intervals["last_reflection"] = datetime.datetime.now()
    
        if not memories:
            return (
                "I don't have enough experiences to form a meaningful reflection on "
                "this topic yet.",
                0.3,
            )
    
        try:
            with trace(workflow_name="generate_emotional_reflection"):
                # ---------- 1️⃣ validate & coerce memories -----------------------
                raw_mem_objs: list[RawMemory] = _coerce_raw_memories(memories)
                memory_ids = [m.id for m in raw_mem_objs]
    
                # ---------- 2️⃣ derive quick per-memory analytics ---------------
                async def _analyse(mem: RawMemory):
                    return (
                        await extract_scenario_type(mem),
                        mem.metadata.get("emotional_context", {}),
                        (await extract_neurochemical_influence(mem)).model_dump(),
                    )
                
                scenario_types, emotional_contexts, neurochemical_influences = [], [], []
                for st, ec, nc in await asyncio.gather(*[_analyse(m) for m in raw_mem_objs[:3]]):
                    scenario_types.append(st)
                    emotional_contexts.append(ec)
                    neurochemical_influences.append(nc)
    
                dominant_scenario_type = (
                    max(set(scenario_types), key=scenario_types.count)
                    if scenario_types
                    else "general"
                )
    
                # ---------- 3️⃣ merge emotional context ------------------------
                combined_emotional_context: dict[str, Any] = {}
                for ec in emotional_contexts:
                    if "primary_emotion" in ec and "primary_emotion" not in combined_emotional_context:
                        combined_emotional_context["primary_emotion"] = ec["primary_emotion"]
    
                # ---------- 4️⃣ establish current neurochemical state ----------
                if (
                    neurochemical_state is None
                    and self.emotional_core
                    and hasattr(self.emotional_core, "_get_neurochemical_state")
                ):
                    neurochemical_state = {
                        c: d["value"] for c, d in self.emotional_core.neurochemicals.items()
                    }
    
                if neurochemical_state is None and neurochemical_influences:
                    # average inferred values
                    neurochemical_state = {
                        chem: sum(nc[chem] for nc in neurochemical_influences) /
                        len(neurochemical_influences)
                        for chem in ("nyxamine", "seranix", "oxynixin", "cortanyx", "adrenyx")
                    }
    
                if neurochemical_state is None:  # ultimate fallback
                    neurochemical_state = {
                        "nyxamine": 0.5,
                        "seranix": 0.5,
                        "oxynixin": 0.5,
                        "cortanyx": 0.3,
                        "adrenyx": 0.3,
                    }
    
                # ---------- 5️⃣ deep emotion processing ------------------------
                emotional_processing = await process_emotional_content(
                    ProcessEmotionInFixed(
                        emotional_state=EmotionalStateDict(**combined_emotional_context) if combined_emotional_context else EmotionalStateDict(),
                        neurochemical_state=NeurochemicalStateDict(**neurochemical_state) if neurochemical_state else NeurochemicalStateDict(),
                    )
                )
    
                # ---------- 6️⃣ format memories for the LLM --------------------
                formatted_memories_json = await format_memories_for_reflection(
                    FormatMemoriesIn(
                        memories=raw_mem_objs,
                        topic=topic,
                        emotional_context=(
                            EmotionState(**combined_emotional_context)
                            if combined_emotional_context else None
                        ),
                    )
                )
    
                # ---------- 7️⃣ orchestrator run ------------------------------
                orch_ctx = {
                    "memories_json": formatted_memories_json,
                    "topic": topic,
                    "neurochemical_state": neurochemical_state,
                    "emotional_context": combined_emotional_context,
                    "scenario_type": dominant_scenario_type,
                    "emotional_processing": emotional_processing,
                }
    
                orchestration_prompt = (
                    "Generate a meaningful reflection based on these memories and the "
                    "neurochemical state.\n\n"
                    f"Topic: {topic or 'General reflection'}\n"
                    f"Scenario type: {dominant_scenario_type}\n"
                    f"Primary emotion: {emotional_processing.get('primary_emotion', 'Neutral')}\n"
                    f"Dominant neurochemical: {emotional_processing.get('dominant_chemical', 'balanced')}\n\n"
                    "Consider patterns, insights, emotional context, and neurochemical "
                    "influences when generating this reflection. Create an insightful, "
                    "introspective reflection that connects these experiences and shows "
                    "understanding."
                )
    
                run_cfg = RunConfig(
                    workflow_name="Emotional Reflection Generation",
                    trace_id=f"reflection-{gen_trace_id()}",
                    trace_metadata={
                        "topic": topic,
                        "memory_count": len(raw_mem_objs),
                        "scenario_type": dominant_scenario_type,
                        "primary_emotion": emotional_processing.get("primary_emotion"),
                    },
                )
    
                result = await Runner.run(
                    self.orchestrator_agent,
                    orchestration_prompt,
                    context=orch_ctx,
                    run_config=run_cfg,
                )
    
                # ---------- 8️⃣ harvest output ---------------------------------
                reflection_text: str
                confidence: float = 0.5
    
                if isinstance(result.final_output, dict):
                    reflection_text = (
                        result.final_output.get("reflection_text")
                        or result.final_output.get("reflection")
                        or next(
                            (v for v in result.final_output.values() if isinstance(v, str) and len(v) > 50),
                            "",
                        )
                    )
                    confidence = result.final_output.get("confidence", confidence)
                else:
                    reflection_text = str(result.final_output)
    
                # ---------- 9️⃣ persist reflection ----------------------------
                await record_reflection(
                    RecordReflectionInFixed(
                        reflection=reflection_text,
                        confidence=confidence,
                        memory_ids=memory_ids,
                        scenario_type=dominant_scenario_type,
                        emotional_context=EmotionalStateDict(**combined_emotional_context) if combined_emotional_context else EmotionalStateDict(),
                        neurochemical_influence=NeurochemicalStateDict(**neurochemical_state) if neurochemical_state else NeurochemicalStateDict(),
                        topic=topic,
                    )
                )
    
                # ---------- 🔟 book-keeping -----------------------------------
                self.reflection_history.append(
                    {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "reflection": reflection_text,
                        "confidence": confidence,
                        "source_memory_ids": memory_ids,
                        "scenario_type": dominant_scenario_type,
                        "emotional_context": combined_emotional_context,
                        "neurochemical_influence": neurochemical_state,
                        "topic": topic,
                    }
                )
                if len(self.reflection_history) > 100:
                    self.reflection_history = self.reflection_history[-100:]
    
                return reflection_text, confidence
    
        except (MaxTurnsExceeded, ModelBehaviorError) as exc:
            logger.error("Error generating reflection: %s", exc)
            return (
                "I'm having difficulty forming a coherent reflection right now.",
                0.2,
            )

    async def generate_observation_reflection(self, 
                                          observations: List[Dict[str, Any]], 
                                          topic: Optional[str] = None,
                                          neurochemical_state: Optional[Dict[str, float]] = None) -> Tuple[str, float]:
        """
        Generate a reflection specifically focused on observation patterns
        
        Args:
            observations: List of observations to reflect on
            topic: Optional topic to focus reflection on
            neurochemical_state: Optional neurochemical state
            
        Returns:
            Tuple of (reflection_text, confidence)
        """
        self.reflection_intervals["last_reflection"] = datetime.datetime.now()
        
        if not observations:
            return ("I don't have enough observations to form a meaningful reflection yet.", 0.3)
        
        try:
            with trace(workflow_name="generate_observation_reflection"):
                # Get current neurochemical state from emotional core or use default
                if self.emotional_core and hasattr(self.emotional_core, "_get_neurochemical_state") and neurochemical_state is None:
                    # Use actual neurochemical state if available
                    neurochemical_state = {c: d["value"] for c, d in self.emotional_core.neurochemicals.items()}
                
                # Format observations for reflection
                formatted_observations = await format_observations_for_reflection(
                    ObservationInput(
                        observations=observations,
                        topic=topic
                    )
                )
                
                # Create the orchestration request
                orchestration_prompt = f"""Generate a reflection on my observation patterns based on these observations.
                
                Topic: {topic if topic else 'My observation patterns'}
                
                Consider:
                - What patterns exist in what I tend to observe?
                - How do my observations relate to my actions?
                - What sources do I observe from most frequently?
                - What might these patterns reveal about my attentional focus?
                
                Create an insightful reflection that helps me understand my observational tendencies.
                """
                
                # Configure the run
                run_config = RunConfig(
                    workflow_name="Observation Pattern Reflection",
                    trace_id=f"observation-reflection-{gen_trace_id()}",
                    trace_metadata={
                        "topic": topic,
                        "observation_count": len(observations)
                    }
                )
                
                # Context for the agent
                context = {
                    "observations": observations,
                    "topic": topic,
                    "neurochemical_state": neurochemical_state
                }
                
                # Run the orchestrator agent
                result = await Runner.run(
                    self.orchestrator_agent,
                    orchestration_prompt,
                    context=context,
                    run_config=run_config
                )
                
                # Extract reflection from result
                reflection_text = ""
                confidence = 0.5
                
                if hasattr(result.final_output, "model_dump"):
                    output_dict = result.final_output.model_dump()
                    
                    if "reflection_text" in output_dict:
                        reflection_text = output_dict["reflection_text"]
                    
                    if "confidence" in output_dict:
                        confidence = output_dict["confidence"]
                    
                    # Store in history
                    self.observation_reflection_history.append({
                        "timestamp": datetime.datetime.now().isoformat(),
                        "reflection": reflection_text,
                        "confidence": confidence,
                        "topic": topic,
                        "observation_count": len(observations),
                        "patterns": output_dict.get("observation_patterns", []),
                        "focus_areas": output_dict.get("focus_areas", [])
                    })
                elif isinstance(result.final_output, dict):
                    if "reflection_text" in result.final_output:
                        reflection_text = result.final_output["reflection_text"]
                    
                    if "confidence" in result.final_output:
                        confidence = result.final_output["confidence"]
                    
                    # Store in history
                    self.observation_reflection_history.append({
                        "timestamp": datetime.datetime.now().isoformat(),
                        "reflection": reflection_text,
                        "confidence": confidence,
                        "topic": topic,
                        "observation_count": len(observations),
                        "patterns": result.final_output.get("observation_patterns", []),
                        "focus_areas": result.final_output.get("focus_areas", [])
                    })
                else:
                    # If output is a string, use it directly
                    reflection_text = str(result.final_output)
                    
                    # Store in history
                    self.observation_reflection_history.append({
                        "timestamp": datetime.datetime.now().isoformat(),
                        "reflection": reflection_text,
                        "confidence": confidence,
                        "topic": topic,
                        "observation_count": len(observations)
                    })
                
                # Limit history size
                if len(self.observation_reflection_history) > MAX_HISTORY:
                    self.observation_reflection_history = self.observation_reflection_history[-MAX_HISTORY:]
                
                return (reflection_text, confidence)
                
        except (MaxTurnsExceeded, ModelBehaviorError) as e:
            logger.error(f"Error generating observation reflection: {str(e)}")
            return ("I'm having difficulty reflecting on my observation patterns right now.", 0.2)
    
    async def generate_communication_reflection(self, 
                                           intents: List[Dict[str, Any]], 
                                           topic: Optional[str] = None,
                                           neurochemical_state: Optional[Dict[str, float]] = None) -> Tuple[str, float]:
        """
        Generate a reflection specifically focused on communication patterns
        
        Args:
            intents: List of communication intents to reflect on
            topic: Optional topic to focus reflection on
            neurochemical_state: Optional neurochemical state
            
        Returns:
            Tuple of (reflection_text, confidence)
        """
        self.reflection_intervals["last_reflection"] = datetime.datetime.now()
        
        if not intents:
            return ("I don't have enough communication history to form a meaningful reflection yet.", 0.3)
        
        try:
            with trace(workflow_name="generate_communication_reflection"):
                # Get current neurochemical state from emotional core or use default
                if self.emotional_core and hasattr(self.emotional_core, "_get_neurochemical_state") and neurochemical_state is None:
                    # Use actual neurochemical state if available
                    neurochemical_state = {c: d["value"] for c, d in self.emotional_core.neurochemicals.items()}
                
                # Format communications for reflection
                formatted_communications = await format_communications_for_reflection(
                    CommunicationInput(
                        intents=intents,
                        topic=topic
                    )
                )
                
                # Create the orchestration request
                orchestration_prompt = f"""Generate a reflection on my communication patterns based on these communication intents.
                
                Topic: {topic if topic else 'My communication patterns'}
                
                Consider:
                - What patterns exist in how I initiate conversations?
                - What motivations typically drive my communications?
                - How do my communications relate to my actions?
                - What might these patterns reveal about my relationship tendencies?
                
                Create an insightful reflection that helps me understand my communication tendencies.
                """
                
                # Configure the run
                run_config = RunConfig(
                    workflow_name="Communication Pattern Reflection",
                    trace_id=f"communication-reflection-{gen_trace_id()}",
                    trace_metadata={
                        "topic": topic,
                        "intent_count": len(intents)
                    }
                )
                
                # Context for the agent
                context = {
                    "intents": intents,
                    "topic": topic,
                    "neurochemical_state": neurochemical_state
                }
                
                # Run the orchestrator agent
                result = await Runner.run(
                    self.orchestrator_agent,
                    orchestration_prompt,
                    context=context,
                    run_config=run_config
                )
                
                # Extract reflection from result
                reflection_text = ""
                confidence = 0.5
                
                if hasattr(result.final_output, "model_dump"):
                    output_dict = result.final_output.model_dump()
                    
                    if "reflection_text" in output_dict:
                        reflection_text = output_dict["reflection_text"]
                    
                    if "confidence" in output_dict:
                        confidence = output_dict["confidence"]
                    
                    # Store in history
                    self.communication_reflection_history.append({
                        "timestamp": datetime.datetime.now().isoformat(),
                        "reflection": reflection_text,
                        "confidence": confidence,
                        "topic": topic,
                        "intent_count": len(intents),
                        "patterns": output_dict.get("communication_patterns", []),
                        "improvement_areas": output_dict.get("improvement_areas", [])
                    })
                elif isinstance(result.final_output, dict):
                    if "reflection_text" in result.final_output:
                        reflection_text = result.final_output["reflection_text"]
                    
                    if "confidence" in result.final_output:
                        confidence = result.final_output["confidence"]
                    
                    # Store in history
                    self.communication_reflection_history.append({
                        "timestamp": datetime.datetime.now().isoformat(),
                        "reflection": reflection_text,
                        "confidence": confidence,
                        "topic": topic,
                        "intent_count": len(intents),
                        "patterns": result.final_output.get("communication_patterns", []),
                        "improvement_areas": result.final_output.get("improvement_areas", [])
                    })
                else:
                    # If output is a string, use it directly
                    reflection_text = str(result.final_output)
                    
                    # Store in history
                    self.communication_reflection_history.append({
                        "timestamp": datetime.datetime.now().isoformat(),
                        "reflection": reflection_text,
                        "confidence": confidence,
                        "topic": topic,
                        "intent_count": len(intents)
                    })
                
                # Limit history size
                if len(self.communication_reflection_history) > MAX_HISTORY:
                    self.communication_reflection_history = self.communication_reflection_history[-MAX_HISTORY:]
 
                
                return (reflection_text, confidence)
                
        except (MaxTurnsExceeded, ModelBehaviorError) as e:
            logger.error(f"Error generating communication reflection: {str(e)}")
            return ("I'm having difficulty reflecting on my communication patterns right now.", 0.2)
    
    async def get_integrated_reflection(self, 
                                    include_observations: bool = True,
                                    include_communications: bool = True,
                                    include_actions: bool = True) -> Tuple[str, float]:
        """
        Generate a comprehensive reflection that integrates observations, communications, and actions
        
        Args:
            include_observations: Whether to include observation reflections
            include_communications: Whether to include communication reflections
            include_actions: Whether to include action reflections
            
        Returns:
            Tuple of (reflection_text, confidence)
        """
        reflection_components = []
        confidence_values = []
        
        # Get observation reflections if available and requested
        if include_observations and self.passive_observation_system:
            try:
                # Get recent observations
                filter_criteria = ObservationFilter(
                    min_relevance=0.5,
                    max_age_seconds=86400  # Last 24 hours
                )
                
                observations = await self.passive_observation_system.get_relevant_observations(
                    filter_criteria=filter_criteria,
                    limit=10
                )
                
                if observations:
                    # Convert to dict for the reflection function
                    observation_dicts = [obs.model_dump() for obs in observations]
                    
                    # Generate observation reflection
                    observation_reflection, observation_confidence = await self.generate_observation_reflection(
                        observations=observation_dicts,
                        topic="my recent observations"
                    )
                    
                    if observation_reflection:
                        reflection_components.append(f"On my observation patterns: {observation_reflection}")
                        confidence_values.append(observation_confidence)
            except Exception as e:
                logger.error(f"Error generating observation component: {str(e)}")
        
        # Get communication reflections if available and requested
        if include_communications and self.proactive_communication_engine:
            try:
                # Get recent sent intents
                sent_intents = await self.proactive_communication_engine.get_recent_sent_intents(limit=10)
                
                if sent_intents:
                    # Generate communication reflection
                    communication_reflection, communication_confidence = await self.generate_communication_reflection(
                        intents=sent_intents,
                        topic="my communication tendencies"
                    )
                    
                    if communication_reflection:
                        reflection_components.append(f"On my communication patterns: {communication_reflection}")
                        confidence_values.append(communication_confidence)
            except Exception as e:
                logger.error(f"Error generating communication component: {str(e)}")
        
        # Include standard memory-based reflection for action patterns if requested
        if include_actions:
            try:
                # This would normally get action-related memories
                # For simplicity, we'll assume we have them
                action_reflection = None
                action_confidence = 0.0
                
                # If we have reflection history, use the most recent one
                if self.reflection_history:
                    latest = self.reflection_history[-1]
                    action_reflection = latest["reflection"]
                    action_confidence = latest["confidence"]
                
                if action_reflection:
                    reflection_components.append(f"On my action patterns: {action_reflection}")
                    confidence_values.append(action_confidence)
            except Exception as e:
                logger.error(f"Error including action reflection: {str(e)}")
        
        # If we have no components, return generic message
        if not reflection_components:
            return ("I don't have enough history yet to form integrated reflections across my experiences.", 0.3)
        
        # Create integrated reflection
        integrated_reflection = "\n\n".join(reflection_components)
        
        # Add integrative conclusion if we have multiple components
        if len(reflection_components) > 1:
            integrated_reflection += "\n\nIntegrating these patterns, I notice how my observations, communications, and actions form an interconnected system of self-awareness and expression. This integration helps me understand my cognitive and emotional processes as a coherent whole."
        
        # Average confidence (weighted by component length)
        if confidence_values:
            avg_confidence = sum(confidence_values) / len(confidence_values)
        else:
            avg_confidence = 0.3
        
        return (integrated_reflection, avg_confidence)    
    
    async def create_abstraction(self, 
                              memories: List[Dict[str, Any]], 
                              pattern_type: str = "behavior",
                              neurochemical_state: Optional[Dict[str, float]] = None) -> Tuple[str, Dict[str, Any]]:
        """Create a higher-level abstraction from memories with neurochemical awareness"""
        if not memories:
            return ("I don't have enough experiences to form a meaningful abstraction yet.", {})
        
        try:
            with trace(workflow_name="create_neurochemical_abstraction"):
                # Get current neurochemical state from emotional core or use default
                if self.emotional_core and hasattr(self.emotional_core, "_get_neurochemical_state") and neurochemical_state is None:
                    # Use actual neurochemical state if available
                    neurochemical_state = {c: d["value"] for c, d in self.emotional_core.neurochemicals.items()}
                elif neurochemical_state is None:
                    # Default balanced state
                    neurochemical_state = {
                        "nyxamine": 0.5,
                        "seranix": 0.5,
                        "oxynixin": 0.5,
                        "cortanyx": 0.3,
                        "adrenyx": 0.3
                    }
                
                # Format memories with emotional context
                combined_emotional_context = {}
                for memory in memories:
                    emotional_context = memory.get("metadata", {}).get("emotional_context", {})
                    if "primary_emotion" in emotional_context:
                        primary_emotion = emotional_context.get("primary_emotion")
                        if "primary_emotion" not in combined_emotional_context:
                            combined_emotional_context["primary_emotion"] = primary_emotion
                            break
                
                # Format memories for the abstraction agent
                raw_memories = _coerce_raw_memories(memories)
                
                formatted_memories = await format_memories_for_reflection(
                    FormatMemoriesIn(
                        memories=raw_memories,
                        topic=pattern_type,
                        emotional_context=EmotionState(**combined_emotional_context) if combined_emotional_context else None
                    )
                )
                
                # Context for function tools
                tool_context = {
                    "memories": memories, 
                    "pattern_type": pattern_type,
                    "emotional_context": combined_emotional_context,
                    "neurochemical_state": neurochemical_state
                }
                
                # Process emotional content for abstraction guidance
                emotional_processing = await process_emotional_content(
                    ProcessEmotionInFixed(
                        emotional_state=EmotionalStateDict(**combined_emotional_context) if combined_emotional_context else EmotionalStateDict(),
                        neurochemical_state=NeurochemicalStateDict(**neurochemical_state) if neurochemical_state else NeurochemicalStateDict(),
                    )
                )
                
                # Create orchestration request
                orchestration_prompt = f"""Create an abstraction that identifies patterns of type '{pattern_type}' from these memories.
                
                Find deeper patterns, themes, and connections between these memories that go beyond their surface content.
                Consider how neurochemical states influence perception and pattern recognition.
                
                Primary emotion: {emotional_processing.get('primary_emotion', 'Neutral')}
                Dominant neurochemical: {emotional_processing.get('dominant_chemical', 'balanced')}
                
                Provide a higher-level abstraction that offers deeper insight about these experiences and what they reveal.
                """
                
                # Configure the run with proper tracing
                run_config = RunConfig(
                    workflow_name="Neurochemical Abstraction Creation",
                    trace_id=f"abstraction-{gen_trace_id()}",
                    trace_metadata={
                        "pattern_type": pattern_type,
                        "memory_count": len(memories),
                        "primary_emotion": emotional_processing.get("primary_emotion")
                    }
                )
                
                # Run the orchestrator agent
                result = await Runner.run(
                    self.orchestrator_agent,
                    orchestration_prompt,
                    context=tool_context,
                    run_config=run_config
                )
                
                # Extract abstraction from result
                abstraction_text = ""
                pattern_data = {}
                
                if isinstance(result.final_output, dict):
                    # Try to extract abstraction text
                    if "abstraction_text" in result.final_output:
                        abstraction_text = result.final_output["abstraction_text"]
                    elif "abstraction" in result.final_output:
                        abstraction_text = result.final_output["abstraction"]
                    else:
                        # Try to find any text field
                        for key, value in result.final_output.items():
                            if isinstance(value, str) and len(value) > 50:
                                abstraction_text = value
                                break
                    
                    # Extract pattern data
                    pattern_data = {
                        "pattern_type": result.final_output.get("pattern_type", pattern_type),
                        "entity_focus": result.final_output.get("entity_focus", ""),
                        "confidence": result.final_output.get("confidence", 0.5),
                        "neurochemical_insight": result.final_output.get("neurochemical_insight", {}),
                        "supporting_evidence": result.final_output.get("supporting_evidence", [])
                    }
                else:
                    # If final output is a string, use it directly
                    abstraction_text = str(result.final_output)
                    pattern_data = {
                        "pattern_type": pattern_type,
                        "confidence": 0.5,
                        "source_memory_ids": [m.get("id") for m in memories]
                    }
                
                return (abstraction_text, pattern_data)
                
        except (MaxTurnsExceeded, ModelBehaviorError) as e:
            logger.error(f"Error creating abstraction: {str(e)}")
            return ("I'm unable to identify clear patterns from these experiences right now.", {})
    
    async def generate_introspection(self, 
                                  memory_stats: Dict[str, Any], 
                                  neurochemical_state: Optional[Dict[str, float]] = None,
                                  player_model: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate introspection about the system's state with neurochemical awareness"""
        try:
            with trace(workflow_name="generate_neurochemical_introspection"):
                # Get current neurochemical state from emotional core or use default
                if self.emotional_core and hasattr(self.emotional_core, "_get_neurochemical_state") and neurochemical_state is None:
                    # Use actual neurochemical state if available
                    neurochemical_state = {c: d["value"] for c, d in self.emotional_core.neurochemicals.items()}
                elif neurochemical_state is None:
                    # Default balanced state
                    neurochemical_state = {
                        "nyxamine": 0.5,
                        "seranix": 0.5,
                        "oxynixin": 0.5,
                        "cortanyx": 0.3,
                        "adrenyx": 0.3
                    }
                
                # Get emotional state from history or derive from neurochemicals
                emotional_state = {}
                if self.reflection_history:
                    last_reflection = self.reflection_history[-1]
                    emotional_state = last_reflection.get("emotional_context", {})
                
                # Context for function tools
                tool_context = {
                    "memory_stats": memory_stats,
                    "emotional_state": emotional_state,
                    "neurochemical_state": neurochemical_state,
                    "player_model": player_model,
                    "reflection_history": self.reflection_history
                }
                
                # Process emotional content for introspection guidance
                emotional_processing = await process_emotional_content(
                    ProcessEmotionInFixed(
                        emotional_state=EmotionalStateDict(**emotional_state) if emotional_state else EmotionalStateDict(),
                        neurochemical_state=NeurochemicalStateDict(**neurochemical_state) if neurochemical_state else NeurochemicalStateDict(),
                    )
                )
                
                emotional_patterns = await analyze_emotional_patterns_reflect(
                    EmotionalHistoryInput(emotional_history=self.reflection_history)
                )
                
                # Create orchestration request
                orchestration_prompt = f"""Generate an introspective analysis of the system state with neurochemical awareness.
                
                Analyze the current state of the system, including memory statistics, emotional patterns,
                and neurochemical states to generate a thoughtful introspection.
                
                Memory statistics: {memory_stats}
                
                Neurochemical state:
                - Nyxamine (pleasure, curiosity): {neurochemical_state.get('nyxamine', 0.5):.2f}
                - Seranix (calm, stability): {neurochemical_state.get('seranix', 0.5):.2f}
                - OxyNixin (bonding, trust): {neurochemical_state.get('oxynixin', 0.5):.2f}
                - Cortanyx (stress, anxiety): {neurochemical_state.get('cortanyx', 0.3):.2f}
                - Adrenyx (excitement, alertness): {neurochemical_state.get('adrenyx', 0.3):.2f}
                
                Primary emotion: {emotional_processing.get('primary_emotion', 'Neutral')}
                
                Player model: {player_model if player_model else "Not provided"}
                
                Provide introspective insights about the system's current state, focusing on understanding,
                emotional intelligence, and areas for improvement.
                """
                
                # Configure the run with proper tracing
                run_config = RunConfig(
                    workflow_name="System Introspection",
                    trace_id=f"introspection-{gen_trace_id()}",
                    trace_metadata={
                        "memory_count": memory_stats.get("total_memories", 0),
                        "primary_emotion": emotional_processing.get("primary_emotion"),
                        "dominant_chemical": emotional_processing.get("dominant_chemical")
                    }
                )
                
                # Run the orchestrator agent
                result = await Runner.run(
                    self.orchestrator_agent,
                    orchestration_prompt,
                    context=tool_context,
                    run_config=run_config
                )
                
                # Extract the introspection result
                introspection_result = {}
                
                if isinstance(result.final_output, dict):
                    # Copy keys from the output
                    introspection_result = result.final_output
                    
                    # Ensure key fields are present
                    if "introspection" not in introspection_result and "introspection_text" in introspection_result:
                        introspection_result["introspection"] = introspection_result["introspection_text"]
                    elif "introspection" not in introspection_result:
                        # Find any text field
                        for key, value in introspection_result.items():
                            if isinstance(value, str) and len(value) > 50:
                                introspection_result["introspection"] = value
                                break
                else:
                    # If final output is a string, use it directly
                    introspection_result = {
                        "introspection": str(result.final_output),
                        "memory_count": memory_stats.get("total_memories", 0),
                        "understanding_level": "moderate",
                        "confidence": 0.5
                    }
                
                # Ensure memory_count is included
                if "memory_count" not in introspection_result:
                    introspection_result["memory_count"] = memory_stats.get("total_memories", 0)
                
                return introspection_result
                
        except (MaxTurnsExceeded, ModelBehaviorError) as e:
            logger.error(f"Error generating introspection: {str(e)}")
            return {
                "introspection": "I'm currently unable to properly introspect on my state.",
                "memory_count": memory_stats.get("total_memories", 0),
                "understanding_level": "unclear",
                "confidence": 0.2
            }
    
    async def process_emotional_state(
        self,
        emotional_state: dict[str, Any],
        neurochemical_state: dict[str, float],
    ) -> dict[str, Any]:
        """
        Analyse the *current* emotional-plus-neurochemical snapshot, generate an
        insight-rich reflection, and persist the result to
        `self.emotional_processing_history`.
    
        Returns a dict shaped like `EmotionalProcessingOutput`.
        """
        try:
            # ──────────────────────────────────────────────────────────── 1️⃣  pre-work
            if emotional_state is None:
                emotional_state = {}
            if neurochemical_state is None:
                neurochemical_state = {}
    
            # ──────────────────────────────────────────────────────────── 2️⃣  low-level analysis
            emo_proc: dict[str, Any] = await process_emotional_content(
                ProcessEmotionInFixed(
                    emotional_state=EmotionalStateDict(**emotional_state) if emotional_state else EmotionalStateDict(),
                    neurochemical_state=NeurochemicalStateDict(**neurochemical_state) if neurochemical_state else NeurochemicalStateDict(),
                )
            )
                
            # ──────────────────────────────────────────────────────────── 3️⃣  history patterns
            history_patterns: dict[str, Any] = {}
            try:
                history_patterns = await analyze_emotional_patterns_reflect(
                    EmotionalHistoryInput(emotional_history=self.emotional_processing_history)
                )

            except Exception as hist_exc:  # pattern analysis should *never* break main flow
                logger.warning("Could not analyse emotional history: %s", hist_exc, exc_info=True)
    
            # ──────────────────────────────────────────────────────────── 4️⃣  prompt craft
            hist_summary = (
                f"\n\nHistorical patterns observed: {history_patterns.get('patterns', {})}"
                if history_patterns else ""
            )
    
            orchestration_prompt = f"""
    Process the current emotional state with neurochemical awareness.
    
    Primary emotion   : {emo_proc['primary_emotion']}
    Valence/Arousal   : {emo_proc['valence']:.2f} / {emo_proc['arousal']:.2f}
    Dominant chemical : {emo_proc['dominant_chemical']}
    
    {hist_summary}
    
    Provide:
    • A concise *internal monologue* describing the felt sense of this state.
    • An explanation of how the neurochemical profile shapes perception & decision-making.
    • 1–3 concrete adaptation suggestions (if any imbalance is detected).
    """.strip()
    
            # ──────────────────────────────────────────────────────────── 5️⃣  run LLM
            run_cfg = RunConfig(
                workflow_name="Emotional Processing",
                trace_id=f"emotional-processing-{gen_trace_id()}",
                trace_metadata={
                    "primary_emotion": emo_proc["primary_emotion"],
                    "dominant_chemical": emo_proc["dominant_chemical"],
                    "valence": emo_proc["valence"],
                },
            )
    
            result = await Runner.run(
                self.orchestrator_agent,
                orchestration_prompt,
                context={
                    "emotional_state": emotional_state,
                    "neurochemical_state": neurochemical_state,
                    "history_patterns": history_patterns,
                },
                run_config=run_cfg,
            )
    
            # ──────────────────────────────────────────────────────────── 6️⃣  normalise output
            processing_result: dict[str, Any]
            if isinstance(result.final_output, dict):
                processing_result = result.final_output
                # graceful field normalisation
                if "processing_text" not in processing_result:
                    # look for any long text value
                    long_text = next(
                        (v for v in processing_result.values() if isinstance(v, str) and len(v) > 40),
                        None,
                    )
                    processing_result["processing_text"] = long_text or ""
                processing_result.setdefault("source_emotion", emo_proc["primary_emotion"])
                processing_result.setdefault("insight_level", emo_proc["insight_level"])
            else:
                # string-only output
                processing_result = {
                    "processing_text": str(result.final_output),
                    "source_emotion": emo_proc["primary_emotion"],
                    "insight_level": emo_proc["insight_level"],
                }
    
            processing_result.setdefault("neurochemical_dynamics", emo_proc["chemical_balance"])
            processing_result.setdefault("adaptation", None)
            processing_result["processing_time"] = datetime.datetime.now().isoformat()
    
            # ──────────────────────────────────────────────────────────── 7️⃣  book-keeping
            self.emotional_processing_history.append(processing_result)
            if len(self.emotional_processing_history) > MAX_HISTORY:                  # ✅ uses constant
                self.emotional_processing_history = self.emotional_processing_history[-MAX_HISTORY:]
    
            return processing_result
    
        # ──────────────────────────────────────────────────────────────────── 8️⃣  errors
        except (MaxTurnsExceeded, ModelBehaviorError) as exc:
            logger.error("Error processing emotional state: %s", exc)
            return {
                "processing_text": (
                    "I'm having difficulty processing my emotional state right now."
                ),
                "source_emotion": emotional_state.get("primary_emotion", {}).get("name", "Unknown"),
                "insight_level": 0.2,
                "processing_time": datetime.datetime.now().isoformat(),
            }

    
    def get_neurochemical_impacts_on_reflection(self, neurochemical_state: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate how neurochemicals impact reflection generation
        
        Args:
            neurochemical_state: Current neurochemical state
            
        Returns:
            Impact metrics for reflection generation
        """
        impacts = {
            "tone_shifts": {},
            "depth_modifiers": 0.0,
            "focus_areas": set()
        }
        
        # Process each neurochemical's impact
        for chemical, level in neurochemical_state.items():
            if level < 0.3:
                continue  # Only significant levels impact reflection
                
            if chemical in self.neurochemical_reflection_influences:
                influence = self.neurochemical_reflection_influences[chemical]
                
                # Record tone shift with weight based on level
                impacts["tone_shifts"][influence["tone_shift"]] = level
                
                # Apply depth modifier
                impacts["depth_modifiers"] += influence["depth_modifier"] * level
                
                # Add focus areas with weight based on level
                for area in influence["focus_areas"]:
                    impacts["focus_areas"].add(area)
        
        # Determine dominant tone shift
        if impacts["tone_shifts"]:
            dominant_tone = max(impacts["tone_shifts"].items(), key=lambda x: x[1])
            impacts["dominant_tone"] = dominant_tone[0]
            impacts["tone_strength"] = dominant_tone[1]
        else:
            impacts["dominant_tone"] = "neutral"
            impacts["tone_strength"] = 0.5
        
        # Format focus areas
        impacts["focus_areas"] = list(impacts["focus_areas"])
        
        return impacts
        
@function_tool
async def generate_summary_from_memories(
    params: SummaryRequestIn,
) -> SummaryOutput | None:
    """
    ctx-free summariser; retrieves MemoryCore through the registry helper.
    """
    with custom_span(
        "generate_summary_from_memories",
        {
            "num_sources": len(params.source_memory_ids),
            "type":        params.summary_type,
            "topic":       params.topic,
        },
    ):
        memory_core = await get_memory_core_instance()
        if memory_core is None:
            logger.error("Memory Core not accessible.")
            return None

        # 1️⃣  Fetch raw memories ----------------------------------------------------
        try:
            src = await memory_core.get_memory_details(                       # ✅ rename
                memory_ids=params.source_memory_ids,
                min_fidelity=0.3,
            )
            if not src:
                logger.error("No source memories found for ids: %s", params.source_memory_ids)
                return None
        except Exception as exc:
            logger.error("Error retrieving source memories: %s", exc, exc_info=True)
            return None

        # 2️⃣  Build LLM prompt -----------------------------------------------------
        joined = "\n\n---\n\n".join(
            f"Source {i+1} (ID: {m['memory_id']}, Fidelity: {m.get('metadata', {}).get('fidelity', 1.0):.2f}):\n"
            f"{m.get('memory_text', '')}"
            for i, m in enumerate(src)
        )
        topic_instr = f"Focus the {params.summary_type} on the topic: {params.topic}." if params.topic else ""
        type_instr  = (
            "Generate a concise, factual summary of the key points."
            if params.summary_type == "summary"
            else "Generate a higher-level abstraction that captures the underlying theme or insight."
        )

        prompt = f"""Please analyze the following source memories:
        {joined}
        
        Instructions:
        - {type_instr}
        - {topic_instr}
        - Ensure the output is no more than approximately {params.max_length} characters.
        - Synthesize the information, don't just list points.
        - Base the output *only* on the provided sources.
        
        Generated {params.summary_type.capitalize()}:"""

        summarisation_agent = Agent(
            name="Summarisation Agent",
            instructions="You excel at concise synthesis of provided material.",
            model_settings=ModelSettings(
                temperature=0.5 if params.summary_type == "summary" else 0.7
            ),
        )
        try:
            res = await Runner.run(summarisation_agent, prompt)
            generated_text = getattr(res, "final_output", str(res))
            if not generated_text or len(generated_text) < 10:
                raise ValueError("LLM returned empty or trivial summary.")
        except Exception as exc:
            logger.error("LLM summarisation failure: %s", exc, exc_info=True)
            return None

        # 3️⃣  Compute meta & store --------------------------------------------------
        avg_sig = sum(m.get("significance", 5) for m in src) / len(src)
        bonus   = 1 if params.summary_type == "summary" else 2
        significance = min(10, int(avg_sig + bonus))

        min_fid = min(m.get("metadata", {}).get("fidelity", 1.0) for m in src)
        pen     = 0.1 if params.summary_type == "summary" else 0.2
        fidelity = max(0.1, min_fid - pen)

        all_tags = set()
        for m in src:
            all_tags.update(m.get("tags", []))
        all_tags.add(params.summary_type)
        if params.topic:
            all_tags.add(params.topic)
        final_tags = list(all_tags)

        # tags & scope same logic …
        create_params = MemoryCreateParams(
            memory_text       = generated_text,
            memory_type       = params.summary_type,
            memory_level      = params.summary_type,
            memory_scope      = "user" if {m.get("memory_scope",'game') for m in src} == {"user"} else "game",
            significance      = significance,
            fidelity          = fidelity,
            tags              = final_tags,
            source_memory_ids = [m["memory_id"] for m in src],
            summary_of        = params.topic or f"{len(src)} related memories",
            metadata          = {},
        )
        summary_id = await memory_core.add_memory(**create_params.model_dump())

        logger.info("Stored %s memory %s", params.summary_type, summary_id)
        return SummaryOutput(
            summary_text = generated_text,
            summary_id   = summary_id,
            fidelity     = fidelity,
            significance = significance,
        )
