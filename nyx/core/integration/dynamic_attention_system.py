# nyx/core/integration/dynamic_attention_system.py

import logging
import asyncio
import datetime
import json
import heapq
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from collections import deque, defaultdict

from nyx.core.integration.event_bus import Event, get_event_bus
from nyx.core.integration.system_context import get_system_context
from nyx.core.integration.integrated_tracer import get_tracer, TraceLevel, trace_method

logger = logging.getLogger(__name__)

class AttentionFocus:
    """A single focus of attention with metadata."""
    
    def __init__(self, 
                target: str, 
                target_type: str, 
                attention_level: float = 0.5,
                source: str = "system",
                expiration: Optional[datetime.datetime] = None):
        """
        Initialize an attention focus.
        
        Args:
            target: Target of attention (e.g., specific content)
            target_type: Type of attention target (e.g., sensory, memory, need)
            attention_level: Level of attention (0.0-1.0)
            source: Source of attention request
            expiration: Optional expiration time
        """
        self.target = target
        self.target_type = target_type
        self.attention_level = attention_level
        self.source = source
        self.created = datetime.datetime.now()
        self.last_updated = datetime.datetime.now()
        self.expiration = expiration if expiration else self.created + datetime.timedelta(minutes=5)
        self.id = f"focus_{hash(target)}_{datetime.datetime.now().timestamp()}"
        self.decay_rate = 0.1  # How quickly attention decays per minute
        self.activation_count = 1  # How many times this focus has been activated
        self.metadata = {}  # Additional context
    
    @property
    def is_expired(self) -> bool:
        """Check if the focus has expired."""
        return datetime.datetime.now() > self.expiration
    
    @property
    def age(self) -> float:
        """Get age of focus in seconds."""
        return (datetime.datetime.now() - self.created).total_seconds()
    
    @property
    def time_since_update(self) -> float:
        """Get time since last update in seconds."""
        return (datetime.datetime.now() - self.last_updated).total_seconds()
    
    def update(self, attention_level: Optional[float] = None) -> None:
        """
        Update the attention focus.
        
        Args:
            attention_level: Optional new attention level
        """
        self.last_updated = datetime.datetime.now()
        self.activation_count += 1
        
        if attention_level is not None:
            self.attention_level = attention_level

class DynamicAttentionSystem:
    """
    Coordinated attention system across all subsystems.
    
    This module manages a dynamic attentional focus across sensory, memory,
    goal, and conceptual systems, ensuring coherent resource allocation and
    focus shifting based on salience, urgency, and context.
    
    Key functions:
    1. Coordinates attention across all subsystems
    2. Manages attention shifts based on priorities
    3. Handles competing attentional demands
    4. Controls attentional resource allocation
    5. Maintains attention state history
    """
    
    def __init__(self, 
                brain_reference=None, 
                attentional_controller=None,
                emotional_core=None,
                needs_system=None,
                multimodal_integrator=None):
        """Initialize the dynamic attention system."""
        self.brain = brain_reference
        self.attentional_controller = attentional_controller
        self.emotional_core = emotional_core
        self.needs_system = needs_system
        self.multimodal_integrator = multimodal_integrator
        
        # Get system-wide components
        self.event_bus = get_event_bus()
        self.system_context = get_system_context()
        self.tracer = get_tracer()
        
        # Attention configuration
        self.max_focus_slots = 3  # Maximum number of simultaneous focus targets
        self.sensory_priority_factor = 1.5  # Priority multiplier for sensory input
        self.need_priority_factor = 1.2  # Priority multiplier for high-drive needs
        self.emotional_priority_factor = 1.3  # Priority multiplier for strong emotions
        self.goal_priority_factor = 1.1  # Priority multiplier for high-priority goals
        
        # Active focus tracking
        self.active_focus_targets = {}  # id -> AttentionFocus
        self.active_focus_by_type = defaultdict(list)  # target_type -> list of focus ids
        
        # System attentional state
        self.default_attention_level = 0.5  # Default attention level
        self.attention_decay_enabled = True  # Whether attention decay is enabled
        self.last_attention_cycle = datetime.datetime.now()
        
        # Attentional history
        self.attention_history = deque(maxlen=100)  # History of attention shifts
        self.attention_durations = {}  # target -> total duration of attention
        
        # Competing targets tracking
        self.competing_targets = []  # List of targets competing for attention
        
        # Integration event subscriptions
        self._subscribed = False
        
        # Startup time
        self.startup_time = datetime.datetime.now()
        
        logger.info("DynamicAttentionSystem initialized")
    
    async def initialize(self) -> bool:
        """Initialize the system and establish connections to subsystems."""
        try:
            # Set up connections to required systems if needed
            if not self.attentional_controller and hasattr(self.brain, "attentional_controller"):
                self.attentional_controller = self.brain.attentional_controller
                
            if not self.emotional_core and hasattr(self.brain, "emotional_core"):
                self.emotional_core = self.brain.emotional_core
                
            if not self.needs_system and hasattr(self.brain, "needs_system"):
                self.needs_system = self.brain.needs_system
                
            if not self.multimodal_integrator and hasattr(self.brain, "multimodal_integrator"):
                self.multimodal_integrator = self.brain.multimodal_integrator
            
            # Subscribe to relevant events
            if not self._subscribed:
                self.event_bus.subscribe("sensory_input", self._handle_sensory_input)
                self.event_bus.subscribe("need_state_change", self._handle_need_state_change)
                self.event_bus.subscribe("emotional_state_change", self._handle_emotional_state_change)
                self.event_bus.subscribe("goal_status_change", self._handle_goal_status_change)
                self._subscribed = True
            
            logger.info("DynamicAttentionSystem successfully initialized")
            return True
        except Exception as e:
            logger.error(f"Error initializing DynamicAttentionSystem: {e}")
            return False
    
    @trace_method(level=TraceLevel.INFO, group_id="DynamicAttention")
    async def focus_attention(self, 
                           target: str, 
                           target_type: str, 
                           attention_level: float = 0.8,
                           source: str = "system",
                           duration_minutes: float = 5.0,
                           metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Focus attention on a specific target.
        
        Args:
            target: Target of attention
            target_type: Type of attention target
            attention_level: Level of attention (0.0-1.0)
            source: Source of attention request
            duration_minutes: Duration in minutes
            metadata: Optional additional context
            
        Returns:
            Focus result
        """
        try:
            # Calculate expiration time
            expiration = datetime.datetime.now() + datetime.timedelta(minutes=duration_minutes)
            
            # Check if this target is already in focus
            existing_focus = None
            for focus in self.active_focus_targets.values():
                if focus.target == target and focus.target_type == target_type:
                    existing_focus = focus
                    break
            
            # Update existing focus if found
            if existing_focus:
                # Record previous level for reporting
                previous_level = existing_focus.attention_level
                
                # Update the focus
                existing_focus.update(attention_level)
                
                # Update expiration if longer than current
                if expiration > existing_focus.expiration:
                    existing_focus.expiration = expiration
                
                # Update metadata if provided
                if metadata:
                    existing_focus.metadata.update(metadata)
                
                logger.info(f"Updated attention on {target_type} target '{target}': {previous_level:.2f} -> {attention_level:.2f}")
                
                return {
                    "status": "updated",
                    "focus_id": existing_focus.id,
                    "target": target,
                    "target_type": target_type,
                    "attention_level": attention_level,
                    "previous_level": previous_level,
                    "expiration": existing_focus.expiration.isoformat()
                }
            
            # Check if we need to free up a focus slot
            if len(self.active_focus_targets) >= self.max_focus_slots:
                await self._free_focus_slot()
            
            # Create new focus
            new_focus = AttentionFocus(
                target=target,
                target_type=target_type,
                attention_level=attention_level,
                source=source,
                expiration=expiration
            )
            
            # Add metadata if provided
            if metadata:
                new_focus.metadata = metadata
            
            # Add to active focus
            self.active_focus_targets[new_focus.id] = new_focus
            self.active_focus_by_type[target_type].append(new_focus.id)
            
            # Record in history
            self.attention_history.append({
                "timestamp": datetime.datetime.now().isoformat(),
                "action": "focus",
                "target": target,
                "target_type": target_type,
                "attention_level": attention_level,
                "source": source
            })
            
            # Start tracking duration
            if target not in self.attention_durations:
                self.attention_durations[target] = 0.0
            
            logger.info(f"Focused attention on {target_type} target '{target}' at level {attention_level:.2f}")
            
            return {
                "status": "created",
                "focus_id": new_focus.id,
                "target": target,
                "target_type": target_type,
                "attention_level": attention_level,
                "expiration": expiration.isoformat()
            }
        except Exception as e:
            logger.error(f"Error focusing attention: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="DynamicAttention")
    async def shift_attention(self, 
                           from_target: str,
                           to_target: str,
                           to_target_type: str,
                           attention_level: float = 0.8,
                           source: str = "system") -> Dict[str, Any]:
        """
        Explicitly shift attention from one target to another.
        
        Args:
            from_target: Current target of attention
            to_target: New target of attention
            to_target_type: Type of new attention target
            attention_level: Level of attention for new target
            source: Source of attention shift request
            
        Returns:
            Shift result
        """
        try:
            # Find focus on current target
            current_focus = None
            for focus in self.active_focus_targets.values():
                if focus.target == from_target:
                    current_focus = focus
                    break
            
            # Reduce attention on current target if found
            if current_focus:
                # Record the shift
                self.attention_history.append({
                    "timestamp": datetime.datetime.now().isoformat(),
                    "action": "shift_from",
                    "target": from_target,
                    "target_type": current_focus.target_type,
                    "attention_level": current_focus.attention_level,
                    "source": source
                })
                
                # Reduce attention level
                current_focus.attention_level = max(0.1, current_focus.attention_level - 0.5)
                current_focus.last_updated = datetime.datetime.now()
            
            # Focus on new target
            focus_result = await self.focus_attention(
                target=to_target,
                target_type=to_target_type,
                attention_level=attention_level,
                source=source,
                metadata={"shifted_from": from_target}
            )
            
            # Add shift to history
            self.attention_history.append({
                "timestamp": datetime.datetime.now().isoformat(),
                "action": "shift_to",
                "target": to_target,
                "target_type": to_target_type,
                "attention_level": attention_level,
                "source": source,
                "shifted_from": from_target
            })
            
            logger.info(f"Shifted attention from '{from_target}' to '{to_target}' at level {attention_level:.2f}")
            
            return {
                "status": "shifted",
                "from_target": from_target,
                "to_target": to_target,
                "to_target_type": to_target_type,
                "new_focus_id": focus_result.get("focus_id"),
                "attention_level": attention_level
            }
        except Exception as e:
            logger.error(f"Error shifting attention: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.DEBUG, group_id="DynamicAttention")
    async def update_attention_cycle(self) -> Dict[str, Any]:
        """
        Update the attention cycle, handling decay and shifting between competing targets.
        
        Returns:
            Cycle update results
        """
        try:
            # Record start time
            cycle_start = datetime.datetime.now()
            
            # 1. Cleanup expired focus targets
            expired = []
            for focus_id, focus in list(self.active_focus_targets.items()):
                if focus.is_expired:
                    expired.append(focus_id)
            
            # Remove expired focus targets
            for focus_id in expired:
                focus = self.active_focus_targets[focus_id]
                
                # Remove from collections
                del self.active_focus_targets[focus_id]
                if focus_id in self.active_focus_by_type[focus.target_type]:
                    self.active_focus_by_type[focus.target_type].remove(focus_id)
                
                # Record in history
                self.attention_history.append({
                    "timestamp": datetime.datetime.now().isoformat(),
                    "action": "expired",
                    "target": focus.target,
                    "target_type": focus.target_type,
                    "source": "system"
                })
            
            # 2. Apply attention decay if enabled
            if self.attention_decay_enabled:
                for focus in self.active_focus_targets.values():
                    # Calculate decay based on time since last update
                    minutes_since_update = focus.time_since_update / 60.0
                    decay_amount = focus.decay_rate * minutes_since_update
                    
                    # Apply decay
                    focus.attention_level = max(0.1, focus.attention_level - decay_amount)
            
            # 3. Update attention durations
            for focus in self.active_focus_targets.values():
                if focus.target in self.attention_durations:
                    # Add time since last cycle
                    seconds_since_cycle = (cycle_start - self.last_attention_cycle).total_seconds()
                    weighted_seconds = seconds_since_cycle * focus.attention_level
                    self.attention_durations[focus.target] += weighted_seconds
            
            # 4. Handle competing targets
            if self.competing_targets:
                # Process the highest priority competing target
                highest_priority = self.competing_targets[0]
                target = highest_priority["target"]
                target_type = highest_priority["target_type"]
                attention_level = highest_priority["attention_level"]
                
                # Focus on this target
                await self.focus_attention(
                    target=target,
                    target_type=target_type,
                    attention_level=attention_level,
                    source="competing_target"
                )
                
                # Remove from competing targets
                self.competing_targets.pop(0)
            
            # Update last cycle timestamp
            self.last_attention_cycle = cycle_start
            
            return {
                "expired_count": len(expired),
                "active_focus_count": len(self.active_focus_targets),
                "competing_targets_count": len(self.competing_targets),
                "cycle_time": (datetime.datetime.now() - cycle_start).total_seconds()
            }
        except Exception as e:
            logger.error(f"Error updating attention cycle: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="DynamicAttention")
    async def get_current_focus(self) -> Dict[str, Any]:
        """
        Get the current focus of attention.
        
        Returns:
            Current attention focus
        """
        try:
            # First update the attention cycle
            await self.update_attention_cycle()
            
            # Get highest attention level focus
            top_focus = None
            top_level = 0.0
            
            for focus in self.active_focus_targets.values():
                if focus.attention_level > top_level:
                    top_level = focus.attention_level
                    top_focus = focus
            
            if not top_focus:
                return {
                    "focus": None,
                    "active_focus_count": 0,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            
            # Prepare focus data
            focus_data = {
                "target": top_focus.target,
                "target_type": top_focus.target_type,
                "attention_level": top_focus.attention_level,
                "age_seconds": top_focus.age,
                "source": top_focus.source,
                "activation_count": top_focus.activation_count
            }
            
            # Add metadata if present
            if top_focus.metadata:
                focus_data["metadata"] = top_focus.metadata
            
            return {
                "focus": focus_data,
                "active_focus_count": len(self.active_focus_targets),
                "timestamp": datetime.datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting current focus: {e}")
            return {"status": "error", "message": str(e)}
    
    @trace_method(level=TraceLevel.INFO, group_id="DynamicAttention")
    async def calculate_attention_score(self, 
                                     content: Any, 
                                     content_type: str) -> float:
        """
        Calculate attention score for content.
        
        Args:
            content: Content to evaluate
            content_type: Type of content
            
        Returns:
            Attention score (0.0-1.0)
        """
        try:
            # Delegate to attentional controller if available
            if self.attentional_controller and hasattr(self.attentional_controller, "calculate_attention"):
                controller_score = await self.attentional_controller.calculate_attention(content, content_type)
                if controller_score is not None:
                    return controller_score
            
            # Default calculation based on content type
            base_score = 0.5  # Default score
            
            if content_type == "text":
                text = content if isinstance(content, str) else str(content)
                # Attention heuristics for text
                if "?" in text:  # Questions get more attention
                    base_score += 0.2
                if "!" in text:  # Exclamations get more attention
                    base_score += 0.1
                # Longer texts get slightly less attention per word
                word_count = len(text.split())
                base_score -= min(0.3, word_count * 0.01)  # Max penalty of 0.3
            elif content_type == "image":
                # Images generally get high attention
                base_score += 0.3
            elif content_type == "audio":
                # Audio gets medium-high attention
                base_score += 0.2
            elif content_type == "need":
                # Needs get attention based on drive strength
                if isinstance(content, dict) and "drive_strength" in content:
                    base_score = content["drive_strength"] * self.need_priority_factor
            elif content_type == "emotion":
                # Emotions get attention based on intensity
                if isinstance(content, dict) and "intensity" in content:
                    base_score = content["intensity"] * self.emotional_priority_factor
            elif content_type == "goal":
                # Goals get attention based on priority
                if isinstance(content, dict) and "priority" in content:
                    base_score = content["priority"] * self.goal_priority_factor
            
            # Clamp to valid range
            return max(0.0, min(1.0, base_score))
        except Exception as e:
            logger.error(f"Error calculating attention score: {e}")
            return self.default_attention_level  # Default on error
    
    @trace_method(level=TraceLevel.INFO, group_id="DynamicAttention")
    async def get_attentional_state(self) -> Dict[str, Any]:
        """
        Get the current state of the attention system.
        
        Returns:
            Attentional state
        """
        try:
            # Update the attention cycle first
            await self.update_attention_cycle()
            
            # Get all active focus targets
            active_foci = []
            for focus in self.active_focus_targets.values():
                active_foci.append({
                    "id": focus.id,
                    "target": focus.target,
                    "target_type": focus.target_type,
                    "attention_level": focus.attention_level,
                    "source": focus.source,
                    "age_seconds": focus.age,
                    "expires_in_seconds": (focus.expiration - datetime.datetime.now()).total_seconds()
                })
            
            # Sort by attention level
            active_foci.sort(key=lambda x: x["attention_level"], reverse=True)
            
            # Get competing targets
            competing = []
            for target in self.competing_targets:
                competing.append({
                    "target": target["target"],
                    "target_type": target["target_type"],
                    "attention_level": target["attention_level"],
                    "priority": target["priority"]
                })
            
            # Get top attention durations
            durations = []
            for target, duration in sorted(self.attention_durations.items(), key=lambda x: x[1], reverse=True)[:10]:
                durations.append({
                    "target": target,
                    "duration_seconds": duration
                })
            
            # Get recent history
            recent_history = list(self.attention_history)[-10:]
            
            return {
                "active_focus": active_foci,
                "competing_targets": competing,
                "attention_durations": durations,
                "recent_history": recent_history,
                "focus_slots": {
                    "used": len(self.active_focus_targets),
                    "maximum": self.max_focus_slots
                },
                "timestamp": datetime.datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting attentional state: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _free_focus_slot(self) -> None:
        """Free up a focus slot by removing lowest priority focus."""
        if not self.active_focus_targets:
            return
            
        # Calculate priority scores for each focus
        priorities = []
        for focus_id, focus in self.active_focus_targets.items():
            priority = focus.attention_level
            
            # Adjust priority based on type
            if focus.target_type == "sensory":
                priority *= self.sensory_priority_factor
            elif focus.target_type == "need":
                priority *= self.need_priority_factor
            elif focus.target_type == "emotion":
                priority *= self.emotional_priority_factor
            elif focus.target_type == "goal":
                priority *= self.goal_priority_factor
            
            # Adjust based on recency
            recency_factor = max(0.5, 1.0 - (focus.time_since_update / 60.0) * 0.1)  # Reduce by 10% per minute
            priority *= recency_factor
            
            priorities.append((focus_id, priority))
        
        # Sort by priority (lowest first)
        priorities.sort(key=lambda x: x[1])
        
        # Remove lowest priority focus
        lowest_focus_id = priorities[0][0]
        lowest_focus = self.active_focus_targets[lowest_focus_id]
        
        # Record in history
        self.attention_history.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "action": "removed",
            "target": lowest_focus.target,
            "target_type": lowest_focus.target_type,
            "attention_level": lowest_focus.attention_level,
            "source": "free_slot"
        })
        
        # Remove from collections
        del self.active_focus_targets[lowest_focus_id]
        if lowest_focus_id in self.active_focus_by_type[lowest_focus.target_type]:
            self.active_focus_by_type[lowest_focus.target_type].remove(lowest_focus_id)
        
        logger.info(f"Freed focus slot by removing {lowest_focus.target_type} target '{lowest_focus.target}'")
    
    async def _add_competing_target(self, 
                                  target: str, 
                                  target_type: str, 
                                  attention_level: float,
                                  priority: float) -> None:
        """
        Add a target competing for attention.
        
        Args:
            target: Target competing for attention
            target_type: Type of attention target
            attention_level: Level of attention if selected
            priority: Priority for selection
        """
        # Check if target is already competing
        for existing in self.competing_targets:
            if existing["target"] == target and existing["target_type"] == target_type:
                # Update priority if higher
                if priority > existing["priority"]:
                    existing["priority"] = priority
                    existing["attention_level"] = attention_level
                return
        
        # Add new competing target
        self.competing_targets.append({
            "target": target,
            "target_type": target_type,
            "attention_level": attention_level,
            "priority": priority,
            "added": datetime.datetime.now().isoformat()
        })
        
        # Sort by priority (highest first)
        self.competing_targets.sort(key=lambda x: x["priority"], reverse=True)
        
        # Limit list size
        if len(self.competing_targets) > 10:
            self.competing_targets = self.competing_targets[:10]
    
    async def _handle_sensory_input(self, event: Event) -> None:
        """
        Handle sensory input events from the event bus.
        
        Args:
            event: Sensory input event
        """
        try:
            # Extract event data
            modality = event.data.get("modality", "unknown")
            content = event.data.get("content")
            
            if not content:
                return
            
            # Calculate attention score
            attention_score = await self.calculate_attention_score(content, modality)
            
            # Check if attention score is high enough to trigger focus
            if attention_score >= 0.6:  # High importance
                # Focus attention directly
                await self.focus_attention(
                    target=str(content)[:50],  # Truncate long content
                    target_type="sensory",
                    attention_level=attention_score,
                    source="sensory_input",
                    metadata={"modality": modality}
                )
            elif attention_score >= 0.4:  # Medium importance
                # Add as competing target
                await self._add_competing_target(
                    target=str(content)[:50],  # Truncate long content
                    target_type="sensory",
                    attention_level=attention_score,
                    priority=attention_score * self.sensory_priority_factor
                )
        except Exception as e:
            logger.error(f"Error handling sensory input event: {e}")
    
    async def _handle_need_state_change(self, event: Event) -> None:
        """
        Handle need state change events from the event bus.
        
        Args:
            event: Need state change event
        """
        try:
            # Extract event data
            need_name = event.data.get("need_name")
            drive_strength = event.data.get("drive_strength")
            
            if not need_name or drive_strength is None:
                return
            
            # Check if drive strength is high enough to trigger attention
            if drive_strength >= 0.7:  # High drive
                # Focus attention directly
                await self.focus_attention(
                    target=need_name,
                    target_type="need",
                    attention_level=drive_strength,
                    source="need_system",
                    metadata={"drive_strength": drive_strength}
                )
            elif drive_strength >= 0.5:  # Medium drive
                # Add as competing target
                await self._add_competing_target(
                    target=need_name,
                    target_type="need",
                    attention_level=drive_strength,
                    priority=drive_strength * self.need_priority_factor
                )
        except Exception as e:
            logger.error(f"Error handling need state change event: {e}")
    
    async def _handle_emotional_state_change(self, event: Event) -> None:
        """
        Handle emotional state change events from the event bus.
        
        Args:
            event: Emotional state change event
        """
        try:
            # Extract event data
            emotion = event.data.get("emotion")
            intensity = event.data.get("intensity", 0.5)
            
            if not emotion:
                return
            
            # Check if intensity is high enough to trigger attention
            if intensity >= 0.7:  # High intensity
                # Focus attention directly
                await self.focus_attention(
                    target=emotion,
                    target_type="emotion",
                    attention_level=intensity,
                    source="emotional_core",
                    metadata={"intensity": intensity}
                )
            elif intensity >= 0.5:  # Medium intensity
                # Add as competing target
                await self._add_competing_target(
                    target=emotion,
                    target_type="emotion",
                    attention_level=intensity,
                    priority=intensity * self.emotional_priority_factor
                )
        except Exception as e:
            logger.error(f"Error handling emotional state change event: {e}")
    
    async def _handle_goal_status_change(self, event: Event) -> None:
        """
        Handle goal status change events from the event bus.
        
        Args:
            event: Goal status change event
        """
        try:
            # Extract event data
            goal_id = event.data.get("goal_id")
            status = event.data.get("status")
            priority = event.data.get("priority", 0.5)
            
            if not goal_id or not status:
                return
            
            # Focus attention on goal if it's activated or completed
            if status == "active":
                # Focus attention directly for high priority goals
                if priority >= 0.7:
                    await self.focus_attention(
                        target=goal_id,
                        target_type="goal",
                        attention_level=priority,
                        source="goal_manager",
                        metadata={"status": status, "priority": priority}
                    )
                elif priority >= 0.5:
                    # Add as competing target for medium priority goals
                    await self._add_competing_target(
                        target=goal_id,
                        target_type="goal",
                        attention_level=priority,
                        priority=priority * self.goal_priority_factor
                    )
            elif status == "completed":
                # Brief focus on completed goals
                await self.focus_attention(
                    target=goal_id,
                    target_type="goal",
                    attention_level=0.7,  # Higher attention for completion
                    source="goal_manager",
                    duration_minutes=1.0,  # Brief duration
                    metadata={"status": status, "completion": True}
                )
        except Exception as e:
            logger.error(f"Error handling goal status change event: {e}")

# Function to create the dynamic attention system
def create_dynamic_attention_system(brain_reference=None):
    """Create a dynamic attention system for the given brain."""
    return DynamicAttentionSystem(brain_reference=brain_reference)
