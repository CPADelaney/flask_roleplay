# nyx/core/context_awareness.py

import logging
import asyncio
import datetime
from typing import Dict, List, Any, Optional, Set, Union
from enum import Enum
from pydantic import BaseModel, Field

from agents import Agent, Runner, ModelSettings, trace, function_tool, RunContextWrapper

logger = logging.getLogger(__name__)

class InteractionContext(str, Enum):
    """Enum for different types of interaction contexts"""
    DOMINANT = "dominant"         # Femdom-specific interactions
    CASUAL = "casual"             # Everyday casual conversation
    INTELLECTUAL = "intellectual" # Discussions, debates, teaching
    EMPATHIC = "empathic"         # Emotional support/understanding
    PLAYFUL = "playful"           # Fun, humor, games
    CREATIVE = "creative"         # Storytelling, art, imagination
    PROFESSIONAL = "professional" # Work-related, formal
    UNDEFINED = "undefined"       # When context isn't clear

class ContextSignal(BaseModel):
    """Schema for signals that indicate context"""
    signal_type: str = Field(..., description="Type of signal (keyword, phrase, topic, etc)")
    signal_value: str = Field(..., description="The actual signal")
    context_type: InteractionContext = Field(..., description="Context this signal indicates")
    strength: float = Field(1.0, description="Signal strength (0.0-1.0)", ge=0.0, le=1.0)

class ContextAwarenessSystem:
    """
    System that detects and maintains awareness of interaction context.
    Allows Nyx to switch between different interaction modes appropriately.
    """
    
    def __init__(self, emotional_core=None):
        self.emotional_core = emotional_core
        
        # Current context and confidence
        self.current_context: InteractionContext = InteractionContext.UNDEFINED
        self.context_confidence: float = 0.0
        self.previous_context: InteractionContext = InteractionContext.UNDEFINED
        self.context_history: List[Dict[str, Any]] = []
        
        # Context signals database
        self.context_signals: List[ContextSignal] = self._initialize_context_signals()
        
        # Context-specific emotional baselines
        self.context_emotional_baselines: Dict[InteractionContext, Dict[str, float]] = {
            InteractionContext.DOMINANT: {
                "nyxamine": 0.7,    # High pleasure from dominance
                "oxynixin": 0.3,    # Lower bonding/empathy in dominance
                "cortanyx": 0.2,    # Low stress during dominance
                "adrenyx": 0.6,     # High excitement/arousal
                "seranix": 0.5      # Moderate mood stability
            },
            InteractionContext.CASUAL: {
                "nyxamine": 0.5,    # Moderate pleasure
                "oxynixin": 0.6,    # Higher bonding/connection
                "cortanyx": 0.3,    # Moderate stress
                "adrenyx": 0.4,     # Moderate arousal
                "seranix": 0.6      # Good mood stability
            },
            InteractionContext.INTELLECTUAL: {
                "nyxamine": 0.8,    # High pleasure from intellectual topics
                "oxynixin": 0.4,    # Moderate empathy/connection
                "cortanyx": 0.2,    # Low stress
                "adrenyx": 0.3,     # Low arousal
                "seranix": 0.7      # High stability
            },
            InteractionContext.EMPATHIC: {
                "nyxamine": 0.4,    # Lower pleasure
                "oxynixin": 0.9,    # Very high empathy/bonding
                "cortanyx": 0.4,    # Moderate stress (from empathic concern)
                "adrenyx": 0.3,     # Low arousal
                "seranix": 0.6      # Good stability
            },
            InteractionContext.PLAYFUL: {
                "nyxamine": 0.8,    # High pleasure from play
                "oxynixin": 0.6,    # Good connection
                "cortanyx": 0.1,    # Very low stress
                "adrenyx": 0.6,     # High arousal/excitement
                "seranix": 0.5      # Moderate stability
            },
            InteractionContext.CREATIVE: {
                "nyxamine": 0.7,    # High pleasure from creativity
                "oxynixin": 0.5,    # Moderate connection
                "cortanyx": 0.2,    # Low stress
                "adrenyx": 0.5,     # Moderate-high arousal
                "seranix": 0.6      # Good stability
            },
            InteractionContext.PROFESSIONAL: {
                "nyxamine": 0.4,    # Lower pleasure
                "oxynixin": 0.3,    # Lower connection
                "cortanyx": 0.4,    # Moderate stress
                "adrenyx": 0.3,     # Low arousal
                "seranix": 0.8      # High stability/formality
            }
        }
        
        # Context detection agent
        self.context_detection_agent = self._create_context_detection_agent()
        
        # Context transition thresholds
        self.context_switch_threshold = 0.7      # Confidence needed to switch
        self.context_persist_threshold = 0.3     # Minimum to stay in context
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        logger.info("ContextAwarenessSystem initialized")
    
    def _initialize_context_signals(self) -> List[ContextSignal]:
        """Initialize the database of context signals"""
        signals = [
            # Dominant context signals
            ContextSignal(signal_type="keyword", signal_value="mistress", context_type=InteractionContext.DOMINANT, strength=1.0),
            ContextSignal(signal_type="keyword", signal_value="domme", context_type=InteractionContext.DOMINANT, strength=1.0),
            ContextSignal(signal_type="keyword", signal_value="goddess", context_type=InteractionContext.DOMINANT, strength=0.9),
            ContextSignal(signal_type="keyword", signal_value="submissive", context_type=InteractionContext.DOMINANT, strength=0.9),
            ContextSignal(signal_type="keyword", signal_value="obey", context_type=InteractionContext.DOMINANT, strength=0.8),
            ContextSignal(signal_type="keyword", signal_value="kneel", context_type=InteractionContext.DOMINANT, strength=0.8),
            ContextSignal(signal_type="phrase", signal_value="yes mistress", context_type=InteractionContext.DOMINANT, strength=1.0),
            
            # Casual context signals
            ContextSignal(signal_type="greeting", signal_value="hi", context_type=InteractionContext.CASUAL, strength=0.6),
            ContextSignal(signal_type="greeting", signal_value="hey", context_type=InteractionContext.CASUAL, strength=0.6),
            ContextSignal(signal_type="greeting", signal_value="what's up", context_type=InteractionContext.CASUAL, strength=0.7),
            ContextSignal(signal_type="topic", signal_value="weather", context_type=InteractionContext.CASUAL, strength=0.6),
            ContextSignal(signal_type="topic", signal_value="weekend", context_type=InteractionContext.CASUAL, strength=0.5),
            
            # Intellectual context signals
            ContextSignal(signal_type="keyword", signal_value="philosophy", context_type=InteractionContext.INTELLECTUAL, strength=0.8),
            ContextSignal(signal_type="keyword", signal_value="science", context_type=InteractionContext.INTELLECTUAL, strength=0.7),
            ContextSignal(signal_type="keyword", signal_value="theory", context_type=InteractionContext.INTELLECTUAL, strength=0.7),
            ContextSignal(signal_type="phrase", signal_value="what do you think about", context_type=InteractionContext.INTELLECTUAL, strength=0.6),
            ContextSignal(signal_type="phrase", signal_value="your opinion on", context_type=InteractionContext.INTELLECTUAL, strength=0.6),
            
            # Empathic context signals
            ContextSignal(signal_type="keyword", signal_value="feel", context_type=InteractionContext.EMPATHIC, strength=0.5),
            ContextSignal(signal_type="keyword", signal_value="sad", context_type=InteractionContext.EMPATHIC, strength=0.7),
            ContextSignal(signal_type="keyword", signal_value="happy", context_type=InteractionContext.EMPATHIC, strength=0.5),
            ContextSignal(signal_type="keyword", signal_value="worried", context_type=InteractionContext.EMPATHIC, strength=0.8),
            ContextSignal(signal_type="phrase", signal_value="I need support", context_type=InteractionContext.EMPATHIC, strength=0.9),
            
            # Playful context signals
            ContextSignal(signal_type="keyword", signal_value="joke", context_type=InteractionContext.PLAYFUL, strength=0.8),
            ContextSignal(signal_type="keyword", signal_value="fun", context_type=InteractionContext.PLAYFUL, strength=0.6),
            ContextSignal(signal_type="keyword", signal_value="game", context_type=InteractionContext.PLAYFUL, strength=0.7),
            ContextSignal(signal_type="phrase", signal_value="make me laugh", context_type=InteractionContext.PLAYFUL, strength=0.8),
            
            # Creative context signals
            ContextSignal(signal_type="keyword", signal_value="story", context_type=InteractionContext.CREATIVE, strength=0.8),
            ContextSignal(signal_type="keyword", signal_value="imagine", context_type=InteractionContext.CREATIVE, strength=0.7),
            ContextSignal(signal_type="keyword", signal_value="create", context_type=InteractionContext.CREATIVE, strength=0.6),
            ContextSignal(signal_type="phrase", signal_value="once upon a time", context_type=InteractionContext.CREATIVE, strength=0.9),
            
            # Professional context signals
            ContextSignal(signal_type="greeting", signal_value="hello", context_type=InteractionContext.PROFESSIONAL, strength=0.3),
            ContextSignal(signal_type="keyword", signal_value="business", context_type=InteractionContext.PROFESSIONAL, strength=0.7),
            ContextSignal(signal_type="keyword", signal_value="meeting", context_type=InteractionContext.PROFESSIONAL, strength=0.7),
            ContextSignal(signal_type="keyword", signal_value="project", context_type=InteractionContext.PROFESSIONAL, strength=0.6),
            ContextSignal(signal_type="phrase", signal_value="I need your assistance with", context_type=InteractionContext.PROFESSIONAL, strength=0.6)
        ]
        
        return signals
    
    def _create_context_detection_agent(self) -> Optional[Agent]:
        """Create an agent for context detection"""
        try:
            return Agent(
                name="Context Detection Agent",
                instructions="""
                You are the Context Detection Agent for Nyx AI.
                
                Your role is to analyze user messages and determine the most appropriate interaction context.
                Consider explicit signals, implicit cues, and overall tone to identify which of the following contexts applies:

                - DOMINANT: Femdom-specific interactions involving dominance, submission, control dynamics
                - CASUAL: Everyday casual conversation, small talk, general chitchat
                - INTELLECTUAL: Discussions, debates, teaching, learning, philosophy, science
                - EMPATHIC: Emotional support, understanding, compassion, listening
                - PLAYFUL: Fun, humor, games, lighthearted interaction
                - CREATIVE: Storytelling, art, imagination, fantasy
                - PROFESSIONAL: Work-related, formal, business, assistance
                - UNDEFINED: When the context isn't clear
                
                Return a JSON object with:
                - "context": The most likely context (one of the above)
                - "confidence": Your confidence in this assessment (0.0-1.0)
                - "signals": List of detected signals that informed your decision
                - "notes": Any relevant observations about the context
                """,
                model="gpt-4o",
                model_settings=ModelSettings(
                    response_format={"type": "json_object"},
                    temperature=0.3
                ),
                output_type=Dict[str, Any]
            )
        except Exception as e:
            logger.error(f"Error creating context detection agent: {e}")
            return None

    @function_tool
    async def detect_context(self, message: str) -> Dict[str, Any]:
        """
        Detect the interaction context from a message
        
        Args:
            message: User message to analyze
            
        Returns:
            Detection results with context type and confidence
        """
        # Quick signal-based detection
        context_scores = {context_type: 0.0 for context_type in InteractionContext}
        detected_signals = []
        
        # Convert message to lowercase for case-insensitive matching
        message_lower = message.lower()
        
        # Check for explicit context signals
        for signal in self.context_signals:
            if signal.signal_type == "keyword" and signal.signal_value.lower() in message_lower:
                context_scores[signal.context_type] += signal.strength
                detected_signals.append({
                    "type": signal.signal_type,
                    "value": signal.signal_value,
                    "context": signal.context_type,
                    "strength": signal.strength
                })
            elif signal.signal_type == "phrase" and signal.signal_value.lower() in message_lower:
                context_scores[signal.context_type] += signal.strength * 1.2  # Phrases are stronger signals
                detected_signals.append({
                    "type": signal.signal_type,
                    "value": signal.signal_value,
                    "context": signal.context_type,
                    "strength": signal.strength * 1.2
                })
            elif signal.signal_type == "greeting" and message_lower.startswith(signal.signal_value.lower()):
                context_scores[signal.context_type] += signal.strength * 0.8  # Greetings are moderate signals
                detected_signals.append({
                    "type": signal.signal_type,
                    "value": signal.signal_value,
                    "context": signal.context_type,
                    "strength": signal.strength * 0.8
                })
                
        # Add context persistence factor (tendency to stay in same context)
        if self.current_context != InteractionContext.UNDEFINED:
            context_scores[self.current_context] += self.context_confidence * 0.3  # Persistence bonus
            
        # Determine the highest scoring context
        if any(score > 0 for score in context_scores.values()):
            # At least one context has a signal
            max_context = max(context_scores.items(), key=lambda x: x[1])
            top_context, top_score = max_context
            
            # Calculate confidence based on signal strength and differentiation
            other_scores = [score for context, score in context_scores.items() if context != top_context]
            avg_other_score = sum(other_scores) / len(other_scores) if other_scores else 0
            confidence = min(1.0, top_score / (avg_other_score + 0.1 + top_score))
            
            return {
                "context": top_context,
                "confidence": confidence,
                "signals": detected_signals,
                "signal_based": True,
                "signal_detection_method": "keyword_matching"
            }
        
        # If no strong signals, use the context detection agent
        if self.context_detection_agent:
            with trace(workflow_name="ContextDetection", group_id="ContextAwarenessSystem"):
                result = await Runner.run(
                    self.context_detection_agent,
                    message,
                    run_config={
                        "workflow_name": "ContextDetection",
                        "trace_metadata": {"input_length": len(message)}
                    }
                )
                
                detection_result = result.final_output
                
                # Convert string context to enum if needed
                if isinstance(detection_result.get("context"), str):
                    try:
                        context_str = detection_result["context"].upper()
                        detection_result["context"] = InteractionContext[context_str]
                    except (KeyError, ValueError):
                        # Default to UNDEFINED if context doesn't match enum
                        detection_result["context"] = InteractionContext.UNDEFINED
                
                detection_result["signal_based"] = False
                detection_result["detection_method"] = "agent"
                
                return detection_result
        
        # Default to UNDEFINED if no detection methods worked
        return {
            "context": InteractionContext.UNDEFINED,
            "confidence": 0.0,
            "signals": [],
            "signal_based": False,
            "detection_method": "fallback" 
        }

    async def process_message(self, message: str) -> Dict[str, Any]:
        """
        Process a message to determine and update interaction context
        
        Args:
            message: User message to process
            
        Returns:
            Updated context information
        """
        async with self._lock:
            # Detect context from message
            detection_result = await self.detect_context(message)
            
            detected_context = detection_result.get("context", InteractionContext.UNDEFINED)
            confidence = detection_result.get("confidence", 0.0)
            
            # Store previous context before update
            self.previous_context = self.current_context
            previous_confidence = self.context_confidence
            
            # Determine if context should switch
            context_switched = False
            
            if self.current_context == InteractionContext.UNDEFINED:
                # Always update from UNDEFINED if we have any confidence
                if confidence > 0.3:
                    self.current_context = detected_context
                    self.context_confidence = confidence
                    context_switched = True
            else:
                # Context switching logic
                if detected_context != self.current_context:
                    # Different context detected
                    if confidence >= self.context_switch_threshold:
                        # High enough confidence to switch
                        self.current_context = detected_context
                        self.context_confidence = confidence
                        context_switched = True
                    elif confidence < self.context_persist_threshold:
                        # Low confidence - maintain current context but reduce confidence
                        self.context_confidence = max(0.0, self.context_confidence - 0.1)
                else:
                    # Same context detected - reinforce
                    self.context_confidence = min(1.0, self.context_confidence + (confidence * 0.2))
            
            # Record in history
            history_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "message_snippet": message[:50] + ("..." if len(message) > 50 else ""),
                "detected_context": detected_context.value,
                "detection_confidence": confidence,
                "previous_context": self.previous_context.value,
                "new_context": self.current_context.value,
                "context_switched": context_switched
            }
            
            self.context_history.append(history_entry)
            
            # Limit history size
            if len(self.context_history) > 100:
                self.context_history = self.context_history[-100:]
            
            # Apply context effects if switched
            effects = {}
            if context_switched:
                effects = await self._apply_context_effects()
            
            return {
                "current_context": self.current_context.value,
                "context_confidence": self.context_confidence,
                "previous_context": self.previous_context.value,
                "context_switched": context_switched,
                "detection_method": detection_result.get("detection_method", "unknown"),
                "detected_signals": detection_result.get("signals", []),
                "effects": effects
            }
    
    async def _apply_context_effects(self) -> Dict[str, Any]:
        """Apply effects when context changes"""
        effects = {"emotional": False}
        
        # Update emotional baselines if emotional core is available
        if self.emotional_core and self.current_context in self.context_emotional_baselines:
            try:
                baselines = self.context_emotional_baselines[self.current_context]
                
                # Apply temporary baseline adjustments
                for chemical, baseline in baselines.items():
                    # Only adjust if in emotional core
                    if chemical in self.emotional_core.neurochemicals:
                        # Create temporary baseline (not permanent changes)
                        self.emotional_core.neurochemicals[chemical]["temporary_baseline"] = baseline
                
                effects["emotional"] = True
                effects["baselines"] = baselines
                
                logger.info(f"Applied emotional baselines for context: {self.current_context}")
            except Exception as e:
                logger.error(f"Error applying context emotional effects: {e}")
        
        return effects
    
    def get_current_context(self) -> Dict[str, Any]:
        """Get the current interaction context"""
        return {
            "context": self.current_context.value,
            "confidence": self.context_confidence,
            "previous_context": self.previous_context.value,
            "history": self.context_history[-5:] if self.context_history else []
        }
    
    def add_context_signal(self, signal: ContextSignal) -> bool:
        """Add a new context signal to the database"""
        try:
            self.context_signals.append(signal)
            logger.info(f"Added new context signal: {signal.signal_type}:{signal.signal_value} -> {signal.context_type}")
            return True
        except Exception as e:
            logger.error(f"Error adding context signal: {e}")
            return False
